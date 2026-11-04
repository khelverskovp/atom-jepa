"""
Second-stage Atom-JEPA pretraining (Atom-JEPA-Bio): continue a stage-1 run with its
self-supervised JEPA objectives plus a supervised multi-task objective on the
dataset's labels, here the PCBA bioassays (conf/pretrain2.yaml):

    L = optim.jepa_weight * L_jepa + supervised.weight * L_uw

L_jepa is the stage-1 objective (pretraining/train.py). L_uw comes from the context
encoder on the full molecule: per-atom invariants -> attention pooling -> one logit
per assay -> masked focal loss per assay -> uncertainty weighting
(pretraining/supervised.py).

init.ckpt_path is a full train state of the stage-1 run (train_state_*.pt), whose
online encoder and trained predictor are loaded (init.load_predictor), or a
context-encoder checkpoint / released encoder name, in which case the predictor
starts from random init. The EMA target starts as a copy of the encoder and the
supervised head from random init. For the first init.predictor_warmup_steps the
encoder is frozen and only the predictor and the head train; the encoder then
unfreezes with its own LR warmup.

Validation on the held-out PCBA split (mean AUROC over assays) and the checkpoints
run at the end of every epoch; a completed epoch also writes
train_state_<run>_epochNNNN.pt, the only states a supervised run resumes from.

    python -m pretraining.train2 init.ckpt_path=/path/to/train_state_stage1.pt
    torchrun --nproc_per_node=8 -m pretraining.train2 init.ckpt_path=...

On AMD GPUs (LUMI), conf/pretrain2_lumi.yaml adds the execution settings Atom-JEPA-Bio
was trained with; see scripts/lumi/README.md.
"""

import builtins
import copy
import math
import os
import time

if int(os.environ.get("RANK", "0")) != 0:
    os.environ.setdefault("WANDB_MODE", "disabled")

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import wandb
from atom_jepa.checkpoint import _cfg_from_ckpt, read_checkpoint
from atom_jepa.data.collate import STRUCTURE_KEYS, GraphCollator, JEPACollator
from atom_jepa.execution import configure_pretraining_execution
from atom_jepa.models.jepa_equiformer import (
    EquiformerV3Config, EquiformerV3Encoder, EquiformerV3Predictor, irrep_slices,
)
from data.datasets import build_pretraining_dataset
from pretraining.metrics import rankme, alpha_req, vicreg_variance_loss, vicreg_covariance_loss
from pretraining.supervised import (
    SupervisedHead, UncertaintyWeighting, evaluate_supervised, focal_loss_per_task,
)
# the stage-1 building blocks, shared unchanged
from pretraining.train import (
    OverlappedGradSync, add_loss, average_gradients, broadcast_module, centroid_delta,
    centroid_delta_pbc, cosine_lr, ema_update, gather_representations, linear_schedule,
    move_batch, mse_loss_irreps, online_output_std, setup_distributed,
)


@torch.no_grad()
def gradient_balance(grads_a, grads_b):
    """Norms, norm ratio and cosine of two gradients over the same parameters
    (None entries = parameter unused by that objective)."""
    pairs = [(a if a is not None else torch.zeros_like(b), b if b is not None else torch.zeros_like(a))
             for a, b in zip(grads_a, grads_b) if a is not None or b is not None]
    a = torch.cat([p[0].reshape(-1).float() for p in pairs])
    b = torch.cat([p[1].reshape(-1).float() for p in pairs])
    na, nb = a.norm(), b.norm()
    return {
        "grad_balance/jepa_norm": na.item(),
        "grad_balance/sup_norm": nb.item(),
        "grad_balance/sup_over_jepa": (nb / na.clamp_min(1e-12)).item(),
        "grad_balance/cosine": (a @ b / (na * nb).clamp_min(1e-12)).item(),
    }


class LabeledJEPACollator(JEPACollator):
    """JEPACollator whose full-sample batch also carries the samples' graph-level
    labels (every tensor field outside STRUCTURE_KEYS not starting with "_", e.g.
    y [n_assays] -> [G, n_assays]), for the supervised objective on the full molecule."""

    def __call__(self, samples):
        view_a, view_b, full_target = super().__call__(samples)
        full_target["full"].update({
            key: torch.stack([s[key] for s in samples], dim=0)
            for key, value in samples[0].items()
            if key not in STRUCTURE_KEYS and not key.startswith("_") and torch.is_tensor(value)})
        return view_a, view_b, full_target


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def train(cfg: DictConfig):
    is_dist, rank, world_size, device, local_rank, cpu_group = setup_distributed(cfg)
    is_main = (rank == 0)
    if cfg.misc.get("require_rocm", False) and (device.type != "cuda" or not torch.version.hip):
        raise RuntimeError("This recipe requires an allocated ROCm GPU")

    # Route all print() calls in train() (and its closures) through rank 0 only,
    # so multi-GPU logs aren't duplicated. builtins.print is the real print.
    def print(*args, **kwargs):
        if is_main:
            builtins.print(*args, **kwargs)

    def sync_ranks():
        """Barrier every rank must reach. Uses the gloo group so waiting ranks
        block on the CPU instead of holding a GPU kernel for the duration of
        rank 0's validation."""
        if is_dist:
            dist.barrier(group=cpu_group)

    torch.manual_seed(cfg.misc.seed)   # identical across ranks -> identical init
    if is_dist:
        local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        print(f"distributed: world_size={world_size} "
              f"({world_size // local_ws} nodes x {local_ws} ranks/node)", flush=True)

    # TF32: keep fp32 exponent range (Wigner-D/SO2 path stays stable) while
    # speeding up the attention/FFN matmuls on Ampere+. Near-free vs. true fp32.
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_wandb = bool(cfg.wandb.enabled) and cfg.wandb.mode != "disabled" and is_main
    do_resume = bool(cfg.misc.get("resume", False))
    resume_path_cfg = cfg.misc.get("resume_path", None)

    def wlog(d, step):
        if use_wandb:
            wandb.log(d, step=step)

    def tprint(msg):
        builtins.print(f"[rank {rank}] [t={time.perf_counter() - _t0:.1f}s] {msg}",
                       flush=True)
    _t0 = time.perf_counter()

    # data. Under multi-GPU, stagger dataset loading so ranks don't all hit the
    # shared filesystem simultaneously (LMDB shard scanning etc.).
    if is_dist:
        for r in range(world_size):
            if rank == r:
                tprint("loading dataset")
                dataset = build_pretraining_dataset(cfg.data)
                tprint(f"dataset ready ({len(dataset)} samples)")
            dist.barrier()
    else:
        dataset = build_pretraining_dataset(cfg.data)
    # crystals get periodic graphs and minimum-image target-atom queries
    is_crystal = bool(dataset.periodic)

    # model config. When initializing from a checkpoint, the architecture comes from
    # the checkpoint (so the weights load and the predictor is rebuilt exactly as in
    # the original run); only the training-time knobs are taken from cfg.model.
    tprint("building model")
    init_ckpt = None
    if cfg.init.get("ckpt_path", None):
        init_ckpt = read_checkpoint(cfg.init.ckpt_path)
        eqv3_cfg = _cfg_from_ckpt(init_ckpt)
        if float(cfg.model.cutoff) != float(eqv3_cfg.max_radius):
            raise ValueError("model.cutoff must match the initialization checkpoint")
        for key, cast in [("dropout", float), ("drop_path_rate", float),
                          ("grad_checkpointing", bool), ("pred_grad_checkpointing", bool)]:
            if cfg.model.get(key, None) is not None:
                setattr(eqv3_cfg, key, cast(cfg.model[key]))
        if cfg.model.get("pred_grad_checkpointing", None) is None:
            # null follows grad_checkpointing, as for a model built from cfg.model
            eqv3_cfg.pred_grad_checkpointing = eqv3_cfg.grad_checkpointing
        print(f"architecture from init checkpoint {cfg.init.ckpt_path}", flush=True)
    else:
        eqv3_cfg = EquiformerV3Config(
            num_channels=cfg.model.F,
            pred_num_channels=int(cfg.model.get("pred_F", cfg.model.F)),
            num_layers=cfg.model.num_layers,
            pred_num_layers=int(cfg.model.get("pred_num_layers", 2)),
            lmax=cfg.model.Lmax,
            mmax=int(cfg.model.get("mmax", cfg.model.Lmax)),
            max_radius=cfg.model.cutoff,
            num_radial_basis=cfg.model.num_rbfs,
            max_num_elements=int(cfg.model.get("max_num_elements", 128)),
            num_heads=cfg.model.num_heads,
            attn_hidden_channels=int(cfg.model.get("attn_hidden_channels", 64)),
            attn_alpha_channels=int(cfg.model.get("attn_alpha_channels", 32)),
            attn_value_channels=int(cfg.model.get("attn_value_channels", 16)),
            ffn_hidden_channels=int(cfg.model.get("ffn_hidden_channels", cfg.model.F)),
            edge_channels=int(cfg.model.get("edge_channels", cfg.model.F)),
            norm_type=cfg.model.get("norm_type", "merge_layer_norm"),
            attn_grid_resolution_list=cfg.model.get("attn_grid_resolution_list", None),
            ffn_grid_resolution_list=cfg.model.get("ffn_grid_resolution_list", None),
            readout_pool=cfg.model.readout_pool,
            dropout=cfg.model.dropout,
            drop_path_rate=float(cfg.model.get("drop_path_rate", 0.0)),
            predict_irreps=bool(cfg.model.get("predict_irreps", False)),
            query_radius=cfg.model.get("query_radius", None),
            grad_checkpointing=bool(cfg.model.get("grad_checkpointing", True)),
            pred_grad_checkpointing=(
                None if cfg.model.get("pred_grad_checkpointing", None) is None
                else bool(cfg.model.get("pred_grad_checkpointing"))
            ),
        )
    print(f"EquiformerV3 encoder: F={eqv3_cfg.num_channels} "
          f"predictor F={eqv3_cfg.pred_num_channels} layers={eqv3_cfg.num_layers} "
          f"lmax={eqv3_cfg.lmax} mmax={eqv3_cfg.mmax}", flush=True)

    # Whether the predictor outputs full irreps (vs. l=0 only). Used to switch the
    # target encoding, the loss functions, and the diagnostic slicing below.
    predict_irreps = bool(eqv3_cfg.predict_irreps)

    online = EquiformerV3Encoder(eqv3_cfg).to(device)
    init_predictor = None
    if init_ckpt is not None:
        # a full train state holds the encoder as 'online' (its 'target' is not used)
        online.load_state_dict(init_ckpt.get("context_encoder", init_ckpt.get("online")), strict=True)
        if cfg.init.get("load_predictor", True):
            init_predictor = init_ckpt.get("predictor")
        del init_ckpt
        print(f"context encoder initialized from {cfg.init.ckpt_path}; "
              f"EMA target starts as a copy of it", flush=True)
    configure_pretraining_execution(online, cfg.misc)
    tprint("encoder built")
    if is_dist:
        broadcast_module(online, src=0)   # ensure identical online start on all ranks
    target = copy.deepcopy(online).to(device)
    for p in target.parameters():
        p.requires_grad_(False)
    # The EMA target must be deterministic: keep it in eval mode so stochastic
    # depth / dropout (if enabled) never inject noise into the prediction targets.
    # (With drop_path=dropout=0 this is a no-op, since EquiformerV3 has only
    # per-sample norms and no running statistics.)
    target.eval()

    # Sync buffers ONCE here (deepcopy already did, but be explicit). They are
    # constant for this backbone, so the EMA hot path never re-copies them.
    with torch.no_grad():
        for bt, bo in zip(target.buffers(), online.buffers()):
            bt.copy_(bo)
    # Cache parameter lists for the fused EMA update (avoids rebuilding per step).
    target_params = list(target.parameters())
    online_params = list(online.parameters())

    def encode_target_nodes(batch):
        """Per-node target features: full irreps [N,sphere,C] or l=0 [N,C]."""
        if predict_irreps:
            return target.encode_nodes_full(batch)[0]   # [N, sphere, C]
        return target.encode_nodes(batch)[0]            # [N, C]

    # ----------------------------------------------------------------------- #
    # objective gating: resolve the two weights -> effective on/off flags.
    # Compute these BEFORE building the predictor so we only construct the heads
    # we will actually use.
    # ----------------------------------------------------------------------- #
    graph_loss_weight = float(cfg.optim.get("graph_loss_weight", 1.0))
    tgt_atom_loss_weight = float(cfg.optim.get("tgt_atom_loss_weight", 0.0))

    symmetric = bool(cfg.optim.get("symmetric", True))

    use_graph_loss = graph_loss_weight > 0.0
    use_tgt_atom_loss = tgt_atom_loss_weight > 0.0
    tgt_atom_pbc_ref = str(cfg.optim.get("tgt_atom_pbc_ref", "anchor")).lower()
    if use_tgt_atom_loss and is_crystal:
        print(f"tgt_atom under PBC: minimum-image query vectors, "
              f"reference={tgt_atom_pbc_ref}", flush=True)

    if not (use_graph_loss or use_tgt_atom_loss):
        raise ValueError(
            "both objectives are disabled: graph_loss_weight and tgt_atom_loss_weight "
            "are 0. Enable at least one objective."
        )

    def query_delta(fc, full_batch, full_target, ctx_idx, g_ctx, tgt_idx, g_tgt, ng, anchor_key):
        """Positional query for the tgt_atom head: the molecular centroid delta, or
        the minimum-image periodic version for crystals. `fc` and the index tensors
        are already on device; `cell` rides along on full_batch."""
        if not is_crystal:
            return centroid_delta(fc, ctx_idx, g_ctx, tgt_idx, g_tgt, ng)
        anchor = full_target[anchor_key].to(device, non_blocking=True)
        return centroid_delta_pbc(fc, full_batch["cell"], ctx_idx, g_ctx, tgt_idx,
                                  g_tgt, ng, anchor, tgt_atom_pbc_ref)

    predictor = EquiformerV3Predictor(
        eqv3_cfg,
        graph_pred=use_graph_loss,
        tgt_atom_pred=use_tgt_atom_loss,
        readout_use_mlp=cfg.model.get("readout_use_mlp", True),
        readout_use_ln=cfg.model.get("readout_use_ln", True),
    ).to(device)
    if init_predictor is not None:
        predictor.load_state_dict(init_predictor, strict=True)
        del init_predictor
        print(f"predictor initialized from {cfg.init.ckpt_path}", flush=True)
    else:
        print("predictor starts from random init", flush=True)
    configure_pretraining_execution(predictor, cfg.misc)
    tprint("predictor built")
    if is_dist:
        broadcast_module(predictor, src=0)
        # Init is already identical (same seed + the explicit broadcasts above),
        # so from here on identical RNG streams just mean every rank draws the
        # same drop-path masks. Decorrelate them.
        torch.manual_seed(cfg.misc.seed + rank)
        torch.cuda.manual_seed_all(cfg.misc.seed + rank)
    print(f"online params={sum(p.numel() for p in online.parameters())} "
          f"predictor params={sum(p.numel() for p in predictor.parameters())}", flush=True)

    # supervised multi-task objective on the dataset's labels (context encoder on
    # the full molecule -> attention pooling -> Linear(C, K) -> focal loss + UW)
    sup_cfg = cfg.supervised
    use_sup = bool(sup_cfg.enabled)
    sup_head = uw = None
    if use_sup:
        num_tasks = int(dataset.num_tasks)
        sup_head = SupervisedHead(eqv3_cfg.num_channels, num_tasks, lmax=eqv3_cfg.lmax,
                                  degrees=list(sup_cfg.degrees),
                                  pool_heads=int(sup_cfg.pool_heads),
                                  pool_dropout=float(sup_cfg.pool_dropout)).to(device)
        uw = UncertaintyWeighting(num_tasks, init=float(sup_cfg.uw_init),
                                  mask_absent=bool(sup_cfg.uw_mask_absent),
                                  reduction=str(sup_cfg.uw_reduction)).to(device)
        if is_dist:
            broadcast_module(sup_head, src=0)
            broadcast_module(uw, src=0)
        print(f"supervised: {num_tasks} tasks, degrees={sup_head.degrees} -> "
              f"Linear({sup_head.linear.in_features}, {num_tasks}), weight={sup_cfg.weight}, "
              f"focal(gamma={sup_cfg.focal_gamma}, alpha={sup_cfg.focal_alpha}), "
              f"uw_mask_absent={sup_cfg.uw_mask_absent}, uw_reduction={sup_cfg.uw_reduction}",
              flush=True)
    jepa_weight = float(cfg.optim.get("jepa_weight", 1.0))

    collator = LabeledJEPACollator(
        cutoff=cfg.model.cutoff,
        ego_hops=cfg.mask.ego_hops,
        ego_topology_cutoff=cfg.mask.ego_topology_cutoff,
        max_num_elements=eqv3_cfg.max_num_elements,
    )
    # Choose DataLoader workers. Under multi-GPU, every rank spawns its own pool,
    # so total worker processes = num_workers * (local ranks). That easily
    # oversubscribes the node's CPUs and starves the (CPU-bound) collator, which
    # is the usual reason multi-GPU barely speeds up small-graph datasets. Cap the
    # per-rank workers to the rank's CPU share.
    num_workers = cfg.data.num_workers
    if is_dist and num_workers > 0:
        try:
            avail_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            avail_cpus = os.cpu_count() or 1
        local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
        cap = max(1, avail_cpus // local_ws)
        if num_workers > cap:
            print(f"warning: num_workers={num_workers} x {local_ws} local ranks "
                  f"oversubscribes {avail_cpus} usable CPUs; capping to {cap}/rank. "
                  f"Allocate more CPUs or lower data.num_workers to change this.",
                  flush=True)
            num_workers = cap
    loader_kwargs = dict(
        batch_size=cfg.optim.batch_size, num_workers=num_workers,
        collate_fn=collator, drop_last=True, pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.data.get("prefetch_factor", 4))
    if is_dist:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        loader = DataLoader(dataset, sampler=sampler, **loader_kwargs)
    else:
        sampler = None
        loader = DataLoader(dataset, shuffle=True, **loader_kwargs)

    tprint("dataloader ready")

    # AMP: off by default for EquiformerV3 (Wigner-D casts to fp16 under autocast).
    use_amp = bool(cfg.misc.amp) and device.type == "cuda"
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.misc.amp_dtype]
    scaler = torch.amp.GradScaler(device.type, enabled=(use_amp and amp_dtype == torch.float16))
    if use_amp:
        print("warning: AMP is enabled; EquiformerV3's Wigner-D/SO2 path is "
              "fp16-sensitive. Prefer cfg.misc.amp=false unless using fp16+scaler.",
              flush=True)

    # loss functions (MSE). In irrep mode both objectives average the MSE over
    # irrep types; the identity diagnostic always uses the l=0 invariant.
    if predict_irreps:
        l_slices = irrep_slices(eqv3_cfg.lmax)
        loss_fn = lambda p, t: mse_loss_irreps(p, t, l_slices)
        print(f"predicting FULL irreps (lmax={eqv3_cfg.lmax}): per-type MSE loss", flush=True)
    else:
        loss_fn = F.mse_loss

    print(f"using mse loss (symmetric={symmetric})", flush=True)
    print(f"objectives: graph={use_graph_loss}(w={graph_loss_weight}) "
          f"tgt_atom={use_tgt_atom_loss}(w={tgt_atom_loss_weight})", flush=True)

    # Parameter groups: the encoder (frozen during the predictor warmup, then its own
    # LR warmup), the newly initialized heads (predictor + supervised head), and the
    # UW log-variances (no weight decay, which would pull every s_t towards 0).
    head_params = list(predictor.parameters()) + (list(sup_head.parameters()) if use_sup else [])
    param_groups = [{"name": "encoder", "params": online_params},
                    {"name": "heads", "params": head_params}]
    if use_sup:
        param_groups.append({"name": "uw", "params": list(uw.parameters()), "no_decay": True})
    params = [p for g in param_groups for p in g["params"]]
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay_start)

    # gradient synchronization across ranks. Overlapped (default) hides the
    # all-reduce behind the backward pass; the blocking fallback is one coalesced
    # all-reduce after backward (useful for debugging / if hooks misbehave).
    grad_sync = None
    if is_dist:
        if bool(cfg.optim.get("overlap_grad_sync", True)):
            grad_sync = OverlappedGradSync(
                params, world_size, bucket_mb=float(cfg.optim.get("grad_bucket_mb", 25.0)))
            print("distributed: overlapping gradient all-reduce with backward", flush=True)
        else:
            print("distributed: blocking coalesced gradient all-reduce", flush=True)

    total_steps = cfg.optim.epochs * len(loader)
    warmup_steps = int(cfg.optim.warmup_frac * total_steps)

    # Predictor warmup: the encoder is frozen for the first unfreeze_step steps. Its
    # LR then follows its own warmup + cosine over the remaining steps, so it does not
    # jump straight to the peak LR. With no warmup this is the original schedule.
    unfreeze_step = int(cfg.init.get("predictor_warmup_steps", 0) or 0)
    if unfreeze_step >= total_steps:
        raise ValueError(f"init.predictor_warmup_steps={unfreeze_step} leaves no steps to train "
                         f"the encoder (total_steps={total_steps})")
    enc_lr = cfg.optim.lr * float(cfg.optim.get("encoder_lr_scale", 1.0))
    enc_warmup_steps = int(cfg.optim.warmup_frac * (total_steps - unfreeze_step))
    if unfreeze_step:
        print(f"predictor warmup: encoder frozen for the first {unfreeze_step} steps "
              f"({unfreeze_step / len(loader):.2f} epochs)", flush=True)

    def group_lrs(step):
        head_lr = cosine_lr(step, total_steps, cfg.optim.lr, warmup_steps)
        if step < unfreeze_step:
            return {"encoder": 0.0, "heads": head_lr, "uw": head_lr}
        return {"encoder": cosine_lr(step - unfreeze_step, total_steps - unfreeze_step,
                                     enc_lr, enc_warmup_steps),
                "heads": head_lr, "uw": head_lr}

    # checkpoint path + saver. Used by the periodic evaluation (overwriting the same
    # file each time) and by the final save at the end of training.
    # When resuming, all output files get a _resume{N} suffix so they never
    # overwrite checkpoints from the original (or an earlier resumed) run.
    os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)
    # names the checkpoint files and the wandb run; null -> jepa_<dataset>
    run_name = cfg.wandb.run_name or f"jepa_{cfg.data.dataset}"
    _ckpt_suffix = ""
    if do_resume:
        _n = 1
        while os.path.exists(os.path.join(
                cfg.misc.checkpoint_dir,
                f"context_encoder_{run_name}_resume{_n}.pt")):
            _n += 1
        _ckpt_suffix = f"_resume{_n}"
    ckpt_path = os.path.join(cfg.misc.checkpoint_dir, f"context_encoder_{run_name}{_ckpt_suffix}.pt")
    state_path = os.path.join(cfg.misc.checkpoint_dir, f"train_state_{run_name}{_ckpt_suffix}.pt")
    best_ckpt_path = os.path.join(cfg.misc.checkpoint_dir, f"context_encoder_best_{run_name}{_ckpt_suffix}.pt")

    def encoder_payload(epoch, **extra):
        """Downstream-facing checkpoint: the context encoder plus what is needed to
        rebuild it (loadable with atom_jepa.load_pretrained_encoder). The supervised
        head rides along for reference."""
        return {
            "context_encoder": online.state_dict(),
            "eqv3_cfg": eqv3_cfg.__dict__,
            "supervised_head": sup_head.state_dict() if use_sup else None,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "epoch": epoch,
            **extra,
        }

    def save_context_encoder(epoch):
        if is_main:
            torch.save(encoder_payload(epoch), ckpt_path)

    def save_full_state(epoch, step, epoch_complete=False):
        """Full resumable training state, written alongside the (lightweight,
        downstream-facing) context-encoder checkpoint.

        PCBA continuation requires a completed epoch. Includes:
        the online and EMA target encoders, the predictor, the supervised head and
        UW log-variances, the AdamW optimizer, the AMP GradScaler, the schedule
        counters (epoch / step / total_steps), and the CPU + CUDA RNG states.
        eqv3_cfg / cfg are stored too so the models can be rebuilt before loading.
        (Resume loading is handled by the resume block near the top of train().)
        """
        if not is_main:
            return
        destination = state_path
        if use_sup and epoch_complete:
            stem, ext = os.path.splitext(state_path)
            destination = f"{stem}_epoch{epoch + 1:04d}{ext}"
        torch.save(
            {
                "online": online.state_dict(),
                "target": target.state_dict(),
                "predictor": predictor.state_dict(),
                "supervised_head": sup_head.state_dict() if use_sup else None,
                "uw": uw.state_dict() if use_sup else None,
                "epoch_complete": epoch_complete,
                "assay_ids": dataset.assay_ids if use_sup else None,
                "world_size": world_size,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "step": step,
                "total_steps": total_steps,
                "eqv3_cfg": eqv3_cfg.__dict__,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (torch.cuda.get_rng_state_all()
                                   if torch.cuda.is_available() else None),
                "wandb_run_id": (wandb.run.id if use_wandb and wandb.run is not None else None),
                "best_metric_value": best_metric_value,
            },
            destination,
        )
        return destination

    # validation on the held-out split of the supervised dataset (rank 0 only)
    val_cfg = cfg.val
    val_loader = None
    if use_sup and bool(val_cfg.enabled) and is_main:
        val_dataset = build_pretraining_dataset(OmegaConf.merge(
            cfg.data, {"split": val_cfg.split, "limit": val_cfg.get("limit", None)}))
        if val_dataset.assay_ids != dataset.assay_ids:
            raise ValueError("PCBA training and validation assay order differs")
        val_loader = DataLoader(
            val_dataset, batch_size=int(val_cfg.batch_size), shuffle=False,
            num_workers=int(val_cfg.num_workers), pin_memory=(device.type == "cuda"),
            collate_fn=GraphCollator(cfg.model.cutoff, eqv3_cfg.max_num_elements))

    # "Best" encoder tracking: keep a separate checkpoint of the encoder with the
    # best value of one validation metric so far (val.best_metric / val.best_mode).
    best_metric_name = val_cfg.get("best_metric", "val/auroc_mean")
    best_mode = str(val_cfg.get("best_mode", "max")).lower()
    if best_mode not in ("min", "max"):
        raise ValueError(f"cfg.val.best_mode must be 'min' or 'max', got '{best_mode}'")
    best_metric_value = None

    def evaluate_and_save(epoch, step, epoch_complete=False):
        """rank-0: run validation (if enabled) + save the checkpoints, then restore
        train() so a MID-EPOCH call is safe. Non-main ranks return immediately; the
        caller is responsible for the barrier that keeps ranks aligned."""
        nonlocal best_metric_value
        if not is_main:
            return
        if val_loader is not None:
            metrics = evaluate_supervised(
                online, sup_head, val_loader, device,
                gamma=float(sup_cfg.focal_gamma), alpha=sup_cfg.focal_alpha,
                min_positives=int(val_cfg.min_positives))
            msg = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"[val] epoch {epoch} step {step}: {msg}", flush=True)
            wlog(metrics, step)

            mval = float(metrics.get(best_metric_name, float("nan")))
            if math.isnan(mval):
                print(f"warning: best_metric '{best_metric_name}' unavailable in "
                      f"{sorted(metrics)}; skipping best-checkpoint update", flush=True)
            elif best_metric_value is None or (
                    mval < best_metric_value if best_mode == "min" else mval > best_metric_value):
                prev = best_metric_value
                best_metric_value = mval
                torch.save(encoder_payload(epoch, step=step, val_metrics=dict(metrics),
                                           best_metric=best_metric_name,
                                           best_metric_value=mval), best_ckpt_path)
                wlog({f"{best_metric_name}_best": mval}, step)
                prev_str = "none" if prev is None else f"{prev:.4f}"
                print(f"[val] new best {best_metric_name}={mval:.4f} (prev {prev_str}); "
                      f"saved best encoder to {best_ckpt_path}", flush=True)
        save_context_encoder(epoch)
        saved_state = save_full_state(epoch, step, epoch_complete=epoch_complete)
        print(f"saved context encoder to {ckpt_path} and full state to {saved_state} "
              f"(epoch {epoch}, step {step})", flush=True)
        online.train(); predictor.train()
        if use_sup:
            sup_head.train()

    # Step-cadence evaluation/checkpoint interval (in optimizer steps). 0/null
    # disables; evaluation + checkpointing also run at the end of every epoch.
    eval_every_steps = int(cfg.misc.get("eval_every_steps", 0) or 0)
    grad_balance_every = int(cfg.misc.get("grad_balance_every", 0) or 0)
    if eval_every_steps:
        print(f"evaluation/checkpoint step cadence: every {eval_every_steps} steps", flush=True)

    # ---------------------------------------------------------------------- #
    # RESUME from a previous train_state checkpoint
    # ---------------------------------------------------------------------- #
    if use_sup and do_resume and not resume_path_cfg:
        raise ValueError("PCBA resume requires an explicit completed-epoch state")
    if resume_path_cfg is None:
        _base = os.path.join(cfg.misc.checkpoint_dir, f"train_state_{run_name}")
        resume_path_cfg = f"{_base}.pt"
        for _k in range(100, 0, -1):
            _candidate = f"{_base}_resume{_k}.pt"
            if os.path.exists(_candidate):
                resume_path_cfg = _candidate
                break
    start_epoch = 0
    start_step = 0
    if do_resume and os.path.exists(resume_path_cfg):
        rs = torch.load(resume_path_cfg, map_location=device, weights_only=False)
        if use_sup:
            if not rs.get("epoch_complete", False):
                raise ValueError("PCBA resume requires a completed epoch; use init for weights only")
            if rs.get("assay_ids") != dataset.assay_ids:
                raise ValueError("PCBA resume requires the same ordered assays")
            if rs.get("world_size") != world_size or rs["total_steps"] != total_steps:
                raise ValueError("PCBA resume requires the same world size and schedule")
            resolved = OmegaConf.to_container(cfg, resolve=True)
            for key in ("model", "mask", "optim", "supervised", "init"):
                if rs["cfg"].get(key) != resolved.get(key):
                    raise ValueError(f"PCBA resume requires unchanged {key}")
            for key in ("root", "split", "limit", "subset_seed"):
                if rs["cfg"]["data"].get(key) != resolved["data"].get(key):
                    raise ValueError(f"PCBA resume requires unchanged data.{key}")
            for key, default in (("precision_policy", "fp32"), ("amp", False),
                                 ("optimize_grid_mlp", False), ("optimize_packed_so2", False),
                                 ("optimize_rocm_grid", False)):
                if rs["cfg"]["misc"].get(key, default) != cfg.misc.get(key, default):
                    raise ValueError(f"PCBA resume requires unchanged misc.{key}")
        online.load_state_dict(rs["online"])
        target.load_state_dict(rs["target"])
        predictor.load_state_dict(rs["predictor"])
        if use_sup:
            sup_head.load_state_dict(rs["supervised_head"])
            uw.load_state_dict(rs["uw"])
        optimizer.load_state_dict(rs["optimizer"])
        scaler.load_state_dict(rs["scaler"])
        start_epoch = int(rs["epoch"]) + 1
        start_step = int(rs["step"])
        torch.set_rng_state(rs["torch_rng_state"].cpu())
        if rs.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rs["cuda_rng_state"]])
        target_params = list(target.parameters())
        online_params = list(online.parameters())
        if rs.get("best_metric_value") is not None:
            best_metric_value = rs["best_metric_value"]
        del rs
        if is_dist:
            broadcast_module(online, src=0)
            broadcast_module(target, src=0)
            broadcast_module(predictor, src=0)
            if use_sup:
                broadcast_module(sup_head, src=0)
                broadcast_module(uw, src=0)
        print(f"RESUMED from {resume_path_cfg}: start_epoch={start_epoch} "
              f"step={start_step}", flush=True)
    elif do_resume:
        if use_sup:
            raise FileNotFoundError(resume_path_cfg)
        print(f"resume=true but {resume_path_cfg} not found; starting from scratch",
              flush=True)
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project, entity=cfg.wandb.entity,
            name=run_name, mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    tprint("startup complete, entering training loop")
    step = start_step
    for epoch in range(start_epoch, cfg.optim.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)   # reshuffle shards each epoch

        online.train(); predictor.train()
        if use_sup:
            sup_head.train()
        running = torch.zeros((), device=device)  # on-device accumulator (no per-step sync)
        # Optional per-step timing diagnostic (cfg.misc.timing): data_wait vs.
        # compute vs. grad_sync. Off by default -- it's just for profiling.
        timing = bool(cfg.misc.get("timing", False))
        dw_acc = cmp_acc = gs_acc = 0.0
        n_timed = 0
        t_prev = time.perf_counter()
        for view_a, view_b, full_target in loader:
            data_wait = (time.perf_counter() - t_prev) if timing else 0.0
            view_a = move_batch(view_a, device)
            view_b = move_batch(view_b, device)

            full_batch = move_batch(full_target["full"], device)
            idx_a = full_target["index_a"].to(device, non_blocking=True)
            idx_b = full_target["index_b"].to(device, non_blocking=True)

            lrs = group_lrs(step)
            lr = lrs["heads"]
            tau = linear_schedule(step, total_steps, cfg.optim.ema_tau_start, cfg.optim.ema_tau_end)
            wd = linear_schedule(step, total_steps, cfg.optim.weight_decay_start, cfg.optim.weight_decay_end)
            for pg in optimizer.param_groups:
                pg["lr"] = lrs[pg["name"]]
                pg["weight_decay"] = 0.0 if pg.get("no_decay") else wd
            # during the predictor warmup the encoder runs without autograd: only the
            # predictor and the supervised head (fed its frozen features) train
            encoder_frozen = step < unfreeze_step

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                # Target-encoder node features of the FULL graph (l=0 or full
                # irreps), computed once. Each view's graph target pools them over
                # that view's atoms; the tgt_atom objective predicts them per atom.
                with torch.no_grad():
                    h_full = encode_target_nodes(full_batch)

                loss = None
                tgt_atom_loss = var_loss = cov_loss = None
                o_a = o_b = t_a = t_b = None     # symmetric view reps (identity diag)

                if symmetric:
                    with torch.no_grad():
                        t_a = target.pool(h_full[idx_a], view_a["node_graph_index"], view_a.get("num_graphs"))
                        t_b = target.pool(h_full[idx_b], view_b["node_graph_index"], view_b.get("num_graphs"))

                    with torch.set_grad_enabled(not encoder_frozen):
                        ha, sa = online.encode_nodes(view_a)
                        hb, sb = online.encode_nodes(view_b)
                    o_a = online.pool(ha, view_a["node_graph_index"], view_a.get("num_graphs"))
                    o_b = online.pool(hb, view_b["node_graph_index"], view_b.get("num_graphs"))

                    # Run only the heads we need: graph readout iff graph loss
                    # is on, per-node features iff tgt_atom loss is on.
                    out_a = predictor(view_a, sa, return_graph=use_graph_loss,
                                      return_node_full=use_tgt_atom_loss)
                    out_b = predictor(view_b, sb, return_graph=use_graph_loss,
                                      return_node_full=use_tgt_atom_loss)

                    if use_graph_loss:
                        graph_loss = 0.5 * (loss_fn(out_a["graph"], t_b) + loss_fn(out_b["graph"], t_a))
                        loss = add_loss(loss, graph_loss_weight * graph_loss)

                    if use_tgt_atom_loss:
                        fc = full_target["full"]["node_coordinates"].to(device, non_blocking=True)
                        ng = view_a.get("num_graphs")
                        ga, gb = view_a["node_graph_index"], view_b["node_graph_index"]
                        d_b = query_delta(fc, full_batch, full_target, idx_a, ga, idx_b, gb, ng, "anchor_a")
                        d_a = query_delta(fc, full_batch, full_target, idx_b, gb, idx_a, ga, ng, "anchor_b")
                        pred_b = predictor.predict_target_atoms(out_a["node_full"], ga, ng, d_b, gb)
                        pred_a = predictor.predict_target_atoms(out_b["node_full"], gb, ng, d_a, ga)
                        tgt_atom_loss = 0.5 * (loss_fn(pred_b, h_full[idx_b]) +
                                               loss_fn(pred_a, h_full[idx_a]))
                        loss = add_loss(loss, tgt_atom_loss_weight * tgt_atom_loss)

                    online_rep = torch.cat([o_a, o_b], dim=0)
                    target_rep = torch.cat([t_a, t_b], dim=0)
                else:
                    if torch.rand(()).item() < 0.5:
                        ctx_batch, tgt_batch, tgt_idx, ctx_idx = view_a, view_b, idx_b, idx_a
                        ctx_anchor = "anchor_a"
                    else:
                        ctx_batch, tgt_batch, tgt_idx, ctx_idx = view_b, view_a, idx_a, idx_b
                        ctx_anchor = "anchor_b"

                    with torch.no_grad():
                        target_rep = target.pool(h_full[tgt_idx], tgt_batch["node_graph_index"], tgt_batch.get("num_graphs"))

                    with torch.set_grad_enabled(not encoder_frozen):
                        h, s = online.encode_nodes(ctx_batch)
                    online_rep = online.pool(h, ctx_batch["node_graph_index"], ctx_batch.get("num_graphs"))
                    out_ctx = predictor(ctx_batch, s, return_graph=use_graph_loss, return_node_full=use_tgt_atom_loss)

                    if use_graph_loss:
                        loss = add_loss(loss, graph_loss_weight * loss_fn(out_ctx["graph"], target_rep))
                    if use_tgt_atom_loss:
                        fc = full_target["full"]["node_coordinates"].to(device, non_blocking=True)
                        ng = ctx_batch.get("num_graphs")
                        gc, gt = ctx_batch["node_graph_index"], tgt_batch["node_graph_index"]
                        d_t = query_delta(fc, full_batch, full_target, ctx_idx, gc, tgt_idx, gt, ng, ctx_anchor)
                        pred_t = predictor.predict_target_atoms(out_ctx["node_full"], gc, ng, d_t, gt)
                        tgt_atom_loss = loss_fn(pred_t, h_full[tgt_idx])
                        loss = add_loss(loss, tgt_atom_loss_weight * tgt_atom_loss)

                if cfg.optim.vicreg_var_weight > 0:
                    var_loss = vicreg_variance_loss(online_rep, cfg.optim.vicreg_var_gamma)
                    loss = add_loss(loss, cfg.optim.vicreg_var_weight * var_loss)
                if cfg.optim.vicreg_cov_weight > 0:
                    cov_loss = vicreg_covariance_loss(online_rep)
                    loss = add_loss(loss, cfg.optim.vicreg_cov_weight * cov_loss)

                # L = jepa_weight * L_jepa + supervised.weight * L_uw
                jepa_loss = loss
                loss = jepa_weight * jepa_loss
                sup_loss = task_losses = present = None
                if use_sup:
                    with torch.set_grad_enabled(not encoder_frozen):
                        node_full, _ = online.encode_nodes_full(full_batch)
                    logits = sup_head(node_full, full_batch["node_graph_index"], full_batch["num_graphs"])
                    task_losses, n_labels = focal_loss_per_task(
                        logits, full_batch["y"], float(sup_cfg.focal_gamma), sup_cfg.focal_alpha)
                    present = n_labels > 0
                    sup_loss = uw(task_losses, present)
                    loss = loss + float(sup_cfg.weight) * sup_loss

            # Balance of the two objectives in the ENCODER's gradients (every
            # misc.grad_balance_every steps, rank 0, once the encoder trains): the norms
            # of the gradients of jepa_weight * L_jepa and supervised.weight * L_uw
            # w.r.t. the encoder parameters, and their cosine similarity. Extra backward
            # passes through the retained graph; torch.autograd.grad does not touch
            # .grad, so the training step below is unaffected.
            gb = None
            if (use_sup and grad_balance_every and is_main and not encoder_frozen
                    and (step + 1) % grad_balance_every == 0):
                g_jepa = torch.autograd.grad(jepa_weight * jepa_loss, online_params,
                                             retain_graph=True, allow_unused=True)
                g_sup = torch.autograd.grad(float(sup_cfg.weight) * sup_loss, online_params,
                                            retain_graph=True, allow_unused=True)
                gb = gradient_balance(g_jepa, g_sup)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # gradient sync across ranks. With the overlapped reducer most of the
            # all-reduce already ran during backward; finish() only waits on the
            # tail. Reducing the (still-scaled) grads here then unscaling is
            # equivalent to unscaling then reducing, since both are linear.
            sync_t = 0.0
            if is_dist:
                t_sync = time.perf_counter() if timing else 0.0
                if grad_sync is not None:
                    grad_sync.finish()
                else:
                    average_gradients(params, world_size)
                if timing:
                    sync_t = time.perf_counter() - t_sync
            if encoder_frozen:
                # the overlapped all-reduce zero-fills grads it never saw; drop them
                # so AdamW's decoupled weight decay cannot touch the frozen encoder
                for p in online_params:
                    p.grad = None
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema_update(target_params, online_params, tau)

            running += loss.detach()
            step += 1
            if gb is not None:
                print(f"[grad balance] step {step}: |g_jepa| {gb['grad_balance/jepa_norm']:.3e} "
                      f"|g_sup| {gb['grad_balance/sup_norm']:.3e} "
                      f"sup/jepa {gb['grad_balance/sup_over_jepa']:.3f} "
                      f"cos {gb['grad_balance/cosine']:.3f}", flush=True)
                wlog(gb, step)

            # step-cadence evaluation + checkpoint (mid-epoch). step advances in
            # lockstep across ranks (DistributedSampler + drop_last), so every
            # rank enters this block together; non-main ranks return immediately
            # and wait at the barrier while rank 0 evaluates, so `online` is never
            # mutated underneath the evaluation. The last step of an epoch is
            # left to the end-of-epoch evaluation.
            if eval_every_steps and step % eval_every_steps == 0 and step % len(loader) != 0:
                evaluate_and_save(epoch, step)
                sync_ranks()

            # rank/spectrum diagnostics on the l=0 invariant. Gather target reps
            # across ranks first, so RankMe / a-ReQ see the EFFECTIVE batch
            # (world_size * local rows) instead of just rank 0's shard. The
            # all_gather is collective -> it MUST run on every rank, so it lives
            # OUTSIDE the is_main guard. `step` advances in lockstep across ranks
            # (DistributedSampler + drop_last), so this condition fires
            # simultaneously everywhere and never deadlocks.
            rm = ar = None
            if step % cfg.misc.log_every == 0 and step % cfg.misc.metric_every == 0:
                with torch.no_grad():
                    tr0 = (target_rep[:, 0, :] if predict_irreps else target_rep).detach()
                    if is_dist:
                        tr0 = gather_representations(tr0, world_size)
                    if is_main:
                        rm = rankme(tr0); ar = alpha_req(tr0)

            if step % cfg.misc.log_every == 0 and is_main:
                std = online_output_std(online_rep)
                with torch.no_grad():
                    # identity diagnostic always on the l=0 invariant. Symmetric
                    # runs use the per-view online/target reps.
                    if symmetric:
                        ta0 = t_a[:, 0, :] if predict_irreps else t_a
                        tb0 = t_b[:, 0, :] if predict_irreps else t_b
                        identity_loss = 0.5 * (F.mse_loss(o_a, tb0) + F.mse_loss(o_b, ta0))
                    else:
                        tr_id = target_rep[:, 0, :] if predict_irreps else target_rep
                        identity_loss = F.mse_loss(online_rep, tr_id)
                log_dict = {
                    "train/loss": loss.item(), "train/jepa_loss": jepa_loss.item(),
                    "train/lr": lr, "train/lr_encoder": lrs["encoder"],
                    "train/encoder_frozen": int(encoder_frozen),
                    "train/ema_tau": tau, "train/weight_decay": wd,
                    "train/online_out_std": std,
                    "train/identity_loss": identity_loss.item(),
                    "epoch": epoch,
                }
                msg = (f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} "
                       f"jepa {jepa_loss.item():.4f} identity {identity_loss.item():.4f} "
                       f"lr {lr:.2e} lr_enc {lrs['encoder']:.2e} tau {tau:.5f} "
                       f"wd {wd:.4f} out_std {std:.4f}")
                if sup_loss is not None:
                    w = uw.weights().detach()
                    focal = task_losses[present].mean().item() if present.any() else float("nan")
                    # sup_loss includes the s_t/2 regularizers (negative for s_t < 0);
                    # the weighted focal part alone is what drives the encoder
                    weighted = uw.reduce(task_losses * w, present).item()
                    log_dict.update({
                        "train/sup_loss": sup_loss.item(), "train/focal_loss": focal,
                        "train/uw_weighted_focal": weighted,
                        "train/tasks_present_frac": present.float().mean().item(),
                        "train/uw_weight_mean": w.mean().item(), "train/uw_weight_max": w.max().item(),
                        "train/uw_weight_min": w.min().item(),
                    })
                    msg += f" sup {sup_loss.item():.4f} (weighted focal {weighted:.4f}) focal {focal:.4f}"
                if timing and n_timed > 0:
                    dw_ms = 1000.0 * dw_acc / n_timed
                    cmp_ms = 1000.0 * cmp_acc / n_timed
                    gs_ms = 1000.0 * gs_acc / n_timed
                    total = dw_acc + cmp_acc + gs_acc
                    log_dict["train/data_wait_ms"] = dw_ms
                    log_dict["train/compute_ms"] = cmp_ms
                    log_dict["train/grad_sync_ms"] = gs_ms
                    log_dict["train/data_wait_frac"] = dw_acc / max(1e-9, total)
                    log_dict["train/grad_sync_frac"] = gs_acc / max(1e-9, total)
                    msg += (f" | data_wait {dw_ms:.0f}ms compute {cmp_ms:.0f}ms "
                            f"grad_sync {gs_ms:.0f}ms")
                    dw_acc = cmp_acc = gs_acc = 0.0
                    n_timed = 0
                if rm is not None:                       # computed above (gathered)
                    log_dict["train/rankme"] = rm; log_dict["train/alpha_req"] = ar
                    msg += f" rankme {rm:.1f} a-ReQ {ar:.3f}"
                if tgt_atom_loss is not None:
                    log_dict["train/tgt_atom_loss"] = tgt_atom_loss.item(); msg += f" tgt_atom {tgt_atom_loss.item():.4f}"
                print(msg, flush=True)
                if var_loss is not None:
                    log_dict["train/vicreg_var_loss"] = var_loss.item()
                if cov_loss is not None:
                    log_dict["train/vicreg_cov_loss"] = cov_loss.item()
                wlog(log_dict, step)

            if timing:
                t_end = time.perf_counter()
                dw_acc += data_wait
                gs_acc += sync_t
                cmp_acc += (t_end - t_prev) - data_wait - sync_t
                n_timed += 1
                t_prev = t_end
        if is_dist:
            dist.all_reduce(running, op=dist.ReduceOp.SUM)
            denom = max(1, len(loader)) * world_size
        else:
            denom = max(1, len(loader))
        mean_loss = (running / denom).item()
        print(f"== epoch {epoch} mean loss {mean_loss:.4f}", flush=True)
        wlog({"train/epoch_loss": mean_loss, "epoch": epoch}, step)

        # end-of-epoch validation + checkpoint (the headless online encoder, its
        # best-so-far copy, and the full resumable training state)
        evaluate_and_save(epoch, step, epoch_complete=True)
        sync_ranks()                   # wait for rank 0's validation/checkpoint

    if use_wandb:
        wandb.finish()

    if is_dist:
        sync_ranks()
        dist.destroy_process_group()


@hydra.main(version_base=None, config_path="../conf", config_name="pretrain2")
def main(cfg: DictConfig):
    if int(os.environ.get("RANK", "0")) == 0:
        print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
