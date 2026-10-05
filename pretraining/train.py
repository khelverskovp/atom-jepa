"""
JEPA pretraining with an EquiformerV3 encoder and an EquiformerV3 predictor, on
molecules or periodic crystals.
"""

import builtins
import copy
from datetime import timedelta
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
from atom_jepa.models.jepa_equiformer import (
    EquiformerV3Config, EquiformerV3Encoder, EquiformerV3Predictor, irrep_slices,
)
from atom_jepa.data.collate import JEPACollator
from atom_jepa.data.graphs import min_image_delta
from data.datasets import build_pretraining_dataset
from pretraining.probe import run_crystal_probe, run_qm9_probe
from pretraining.metrics import rankme, alpha_req, vicreg_variance_loss, vicreg_covariance_loss


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def mse_loss_irreps(pred, target, l_slices):
    total = pred.new_zeros(())
    for s, e in l_slices:
        total = total + F.mse_loss(pred[:, s:e, :], target[:, s:e, :])
    return total / len(l_slices)


def add_loss(acc, term):
    return term if acc is None else acc + term


@torch.no_grad()
def ema_update(target_params, online_params, tau):
    torch._foreach_mul_(target_params, tau)
    torch._foreach_add_(target_params, online_params, alpha=1.0 - tau)


@torch.no_grad()
def gather_representations(rep, world_size):
    rep = rep.contiguous()
    out = [torch.empty_like(rep) for _ in range(world_size)]
    dist.all_gather(out, rep)
    return torch.cat(out, dim=0)


@torch.no_grad()
def online_output_std(s_x):
    z = F.normalize(s_x, dim=-1)
    return z.std(dim=0).mean().item()


def cosine_lr(step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * p))


def linear_schedule(step, total_steps, start, end):
    p = step / max(1, total_steps - 1)
    p = min(max(p, 0.0), 1.0)
    return start + (end - start) * p


_SKIP_KEYS = {"node_coordinates"} 


def move_batch(batch, device):
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
        if k not in _SKIP_KEYS
    }


def _scatter_mean_coords(x, gidx, ng):
    out = x.new_zeros(ng, x.shape[1]); out.index_add_(0, gidx, x)
    cnt = x.new_zeros(ng, 1);          cnt.index_add_(0, gidx, torch.ones_like(x[:, :1]))
    return out / cnt.clamp_min(1.0)

def centroid_delta(coords, idx_ctx, gidx_ctx, idx_tgt, gidx_tgt, ng):
    cen = _scatter_mean_coords(coords[idx_ctx], gidx_ctx, ng)
    return coords[idx_tgt] - cen[gidx_tgt]

def centroid_delta_pbc(coords, cell, idx_ctx, gidx_ctx, idx_tgt, gidx_tgt, ng,
                       anchor_idx, mode="anchor"):
    ref = coords[anchor_idx]                                     # [G, 3]
    if mode == "centroid":
        d_ctx = min_image_delta(coords[idx_ctx] - ref[gidx_ctx], cell, gidx_ctx)
        ref = ref + _scatter_mean_coords(d_ctx, gidx_ctx, ng)
    elif mode != "anchor":
        raise ValueError(f"unknown tgt_atom_pbc_ref '{mode}', expected 'anchor' or 'centroid'")
    return min_image_delta(coords[idx_tgt] - ref[gidx_tgt], cell, gidx_tgt)

def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


# --------------------------------------------------------------------------- #
# distributed (multi-GPU) helpers
# --------------------------------------------------------------------------- #
def setup_distributed(cfg):
    """Initialize the process group from torchrun env vars.

    Multi-node: torchrun sets RANK (global), LOCAL_RANK (within node) and
    WORLD_SIZE (global) for every worker, so the only node-aware bit is that
    LOCAL_RANK selects the GPU while RANK selects the data shard.

    Also creates a gloo side-group used for barriers guarding rank-0-only work
    (probe + checkpoint). An NCCL barrier spins a GPU kernel for the whole wait
    and its watchdog counts against the collective timeout -- exactly wrong when
    the other 8N-1 ranks may sit there for the length of a 100-epoch probe.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"torchrun launched {world_size} processes but CUDA is not "
                f"available. This usually means a driver/environment issue on "
                f"the node (check `nvidia-smi`). Without CUDA every rank falls "
                f"back to rank=0/world_size=1, causing duplicate wandb runs, "
                f"duplicate probes, and no distributed training."
            )
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://",
                                timeout=timedelta(hours=2))
        cpu_group = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
                # NCCL allocates buffers lazily per message-size/protocol class. Touch
        # the tiny and bulk paths now, while the card is empty, so the first
        # epoch-boundary all_reduce doesn't cudaMalloc into a full GPU.
        _dev = torch.device(f"cuda:{local_rank}")
        _s = torch.zeros((), device=_dev)
        dist.all_reduce(_s)
        _v = torch.zeros(1024, device=_dev)
        dist.all_gather([torch.empty_like(_v) for _ in range(world_size)], _v)
        del _s, _v
        torch.cuda.empty_cache()
        return True, rank, world_size, torch.device(f"cuda:{local_rank}"), local_rank, cpu_group
    return False, 0, 1, torch.device(resolve_device(cfg.misc.device)), 0, None


@torch.no_grad()
def broadcast_module(module, src=0):
    """Broadcast a module's parameters + buffers from `src` so every rank starts
    from identical weights (belt-and-suspenders on top of identical seeding).

    NCCL requires contiguous tensors, and some state-dict entries here aren't
    (e.g. views from the Wigner-D / SO2 path), so broadcast a contiguous copy and
    copy it back into the original (possibly non-contiguous) storage.
    """
    for t in module.state_dict().values():
        if not torch.is_tensor(t):
            continue
        if t.is_contiguous():
            dist.broadcast(t, src=src)
        else:
            tmp = t.contiguous()
            dist.broadcast(tmp, src=src)
            t.copy_(tmp)


@torch.no_grad()
def average_gradients(params, world_size):
    """Average gradients across ranks in one coalesced, BLOCKING all-reduce.

    Fallback path (no overlap with backward). Mirrors DDP's gradient semantics
    (mean of per-rank gradients).
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat)                       # SUM
    flat.div_(world_size)
    for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        g.copy_(synced)


class OverlappedGradSync:
    """Average gradients across ranks, overlapping the all-reduce with backward.

    Gradients are grouped into fixed buckets in REVERSE parameter order (an
    approximation of the order backward produces them). When every parameter in
    a bucket has its gradient, the bucket is flattened into one contiguous
    buffer and reduced with a single async all-reduce. finish() waits on the
    outstanding buckets, divides by world_size, and scatters back.

    Single-node NVLink hides per-tensor latency; an inter-node fabric does not,
    so one collective per parameter is what makes multi-node scale poorly.
    """

    def __init__(self, params, world_size, bucket_mb=25.0):
        self.world_size = world_size
        params = [p for p in params if p.requires_grad]

        cap = max(1, int(bucket_mb * 1024 * 1024 / 4))   # elements, fp32 grads
        self.buckets, cur, cur_n = [], [], 0
        for p in reversed(params):
            cur.append(p)
            cur_n += p.numel()
            if cur_n >= cap:
                self.buckets.append(cur); cur, cur_n = [], 0
        if cur:
            self.buckets.append(cur)

        self._bucket_of = {p: bi for bi, b in enumerate(self.buckets) for p in b}
        self._ready = [0] * len(self.buckets)
        self._launched = [False] * len(self.buckets)
        self._handles = []
        for p in params:
            p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, param):
        bi = self._bucket_of[param]
        self._ready[bi] += 1
        if self._ready[bi] == len(self.buckets[bi]):
            self._launch(bi)

    def _launch(self, bi):
        grads = []
        for p in self.buckets[bi]:
            if p.grad is None:                       # keep the bucket layout fixed
                p.grad = torch.zeros_like(p)
            elif not p.grad.is_contiguous():
                p.grad = p.grad.contiguous()
            grads.append(p.grad)
        flat = torch._utils._flatten_dense_tensors(grads)
        h = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
        self._handles.append((h, flat, grads))
        self._launched[bi] = True

    def finish(self):
        # Any bucket containing a parameter that got no gradient this step never
        # filled; launch it now (zero-filled) so all ranks reduce the same shapes.
        for bi, done in enumerate(self._launched):
            if not done:
                self._launch(bi)
        for h, flat, grads in self._handles:
            h.wait()
            flat.div_(self.world_size)
            for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
                g.copy_(synced)
        self._handles.clear()
        self._ready = [0] * len(self.buckets)
        self._launched = [False] * len(self.buckets)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def train(cfg: DictConfig):
    is_dist, rank, world_size, device, local_rank, cpu_group = setup_distributed(cfg)
    is_main = (rank == 0)

    # Route all print() calls in train() (and its closures) through rank 0 only,
    # so multi-GPU logs aren't duplicated. builtins.print is the real print.
    def print(*args, **kwargs):
        if is_main:
            builtins.print(*args, **kwargs)

    def sync_ranks():
        """Barrier every rank must reach. Uses the gloo group so waiting ranks
        block on the CPU instead of holding a GPU kernel for the duration of
        rank 0's probe."""
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

    # model config
    tprint("building model")
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

    collator = JEPACollator(
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

    params = list(online.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay_start)

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

    # checkpoint path + saver. Used by the periodic probe (overwriting the same
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

    def save_context_encoder(epoch):
        if not is_main:
            return
        torch.save(
            {
                "context_encoder": online.state_dict(),
                "eqv3_cfg": eqv3_cfg.__dict__,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "epoch": epoch,
            },
            ckpt_path,
        )

    def save_full_state(epoch, step):
        """Full resumable training state, written alongside the (lightweight,
        downstream-facing) context-encoder checkpoint.

        Holds everything needed to continue the run from exactly this point:
        the online and EMA target encoders, the predictor, the AdamW optimizer,
        the AMP GradScaler, the schedule counters (epoch / step / total_steps),
        and the CPU + CUDA RNG states. eqv3_cfg / cfg are stored too
        so the models can be rebuilt before loading. (Resume loading is handled
        by the resume block near the top of train().)
        """
        if not is_main:
            return
        torch.save(
            {
                "online": online.state_dict(),
                "target": target.state_dict(),
                "predictor": predictor.state_dict(),
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
            state_path,
        )
    
    def save_best_context_encoder(epoch, step, metrics, metric_name, metric_value):
        """Downstream-facing checkpoint for the BEST probe result so far.

        Same lightweight payload as save_context_encoder (the online encoder + the
        bits needed to rebuild it), plus the probe metrics and which metric/value
        made it the best, for traceability. Overwrites one *_best_*.pt file.
        """
        if not is_main:
            return
        torch.save(
            {
                "context_encoder": online.state_dict(),
                "eqv3_cfg": eqv3_cfg.__dict__,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "epoch": epoch,
                "step": step,
                "probe_metrics": metrics,
                "best_metric": metric_name,
                "best_metric_value": metric_value,
            },
            best_ckpt_path,
        )

    probe_enabled = bool(cfg.probe.enabled)
    probe_dataset = None
    if probe_enabled and is_main:
        if is_crystal:
            from data.datasets.materials_project import MPProbeDataset
            probe_dataset = MPProbeDataset(
                target=cfg.probe.get("target", "e_form"),
                source=cfg.probe.get("mp_source", "matminer"),
                dataset_name=cfg.probe.get("mp_dataset", None),
                api_key=cfg.probe.get("mp_api_key", None),
                min_atoms=cfg.data.min_atoms,
                max_atoms=cfg.probe.get("max_atoms", cfg.data.get("max_atoms", None)),
                limit=cfg.probe.get("limit", None),
                file_path=cfg.probe.get("mp_file", None),
            )
        else:
            from data.datasets.qm9 import QM9Dataset
            probe_targets = list(cfg.probe.get("targets", None) or [cfg.probe.target])
            probe_dataset = {
                t: QM9Dataset(
                    root=cfg.probe.get("qm9_root", "data/qm9"),
                    target=t,
                    limit=cfg.probe.get("limit", None),
                )
                for t in probe_targets
            }
            print(f"probe targets: {probe_targets}", flush=True)

    # "Best" encoder tracking: watch one probe metric and keep a separate
    # checkpoint of the encoder that achieved its best value so far. Default:
    # watch the FIRST metric the probe returns and MINIMIZE it (right for
    # MAE/RMSE/loss). Override via cfg.probe.best_metric (key to watch) and
    # cfg.probe.best_mode in {"min","max"}.
    best_metric_name = cfg.probe.get("best_metric", None)
    best_mode = str(cfg.probe.get("best_mode", "min")).lower()
    if best_mode not in ("min", "max"):
        raise ValueError(f"cfg.probe.best_mode must be 'min' or 'max', got '{best_mode}'")
    best_metric_value = None

    def _do_probe_and_save(epoch, step):
        """rank-0: run the probe (if a probe dataset exists) + save both
        checkpoints, then restore train() so a MID-EPOCH call is safe.

        Shared by the epoch-cadence path (maybe_probe) and the step-cadence path
        inside the training loop. Non-main ranks return immediately; the caller is
        responsible for any barrier needed to keep ranks aligned.
        """
        nonlocal best_metric_value
        if not is_main:
            return
        if probe_dataset is not None:
            online.eval()
            if is_crystal:
                metrics = run_crystal_probe(online, probe_dataset, cfg, device)
            else:
                metrics = run_qm9_probe(online, probe_dataset, cfg, device)
            msg = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"[probe] epoch {epoch} step {step}: {msg}", flush=True)
            wlog(metrics, step)

            # Update the best-encoder checkpoint if this probe improved. Watch
            # best_metric_name (default: the probe's first metric); compare per
            # best_mode. rank-0-only -- only rank 0 runs the probe and touches
            # best_metric_value, and there are no collective ops here.
            mname = best_metric_name if best_metric_name is not None else next(iter(metrics), None)
            if mname is not None and mname not in metrics:
                print(f"warning: best_metric '{mname}' not in probe metrics "
                      f"{sorted(metrics)}; skipping best-checkpoint update", flush=True)
            elif mname is not None:
                mval = float(metrics[mname])
                improved = (best_metric_value is None) or (
                    mval < best_metric_value if best_mode == "min" else mval > best_metric_value
                )
                if improved:
                    prev = best_metric_value
                    best_metric_value = mval
                    save_best_context_encoder(epoch, step, dict(metrics), mname, mval)
                    wlog({f"probe/best_{mname}": mval}, step)
                    prev_str = "none" if prev is None else f"{prev:.4f}"
                    print(f"[probe] new best {mname}={mval:.4f} (prev {prev_str}); "
                          f"saved best encoder to {best_ckpt_path}", flush=True)
        save_context_encoder(epoch)
        save_full_state(epoch, step)
        print(f"saved context encoder to {ckpt_path} and full state to {state_path} "
              f"(epoch {epoch}, step {step})", flush=True)
        # restore training mode (no-op for the epoch-boundary caller, which sets
        # train() right after anyway, but required for the mid-epoch caller).
        online.train(); predictor.train()

    def maybe_probe(epoch, step):
        # Epoch-cadence probe + checkpoint. Tied to cfg.probe.enabled +
        # cfg.probe.every (in epochs). Rank 0 only -- other ranks wait at the
        # barrier after this call (in the epoch loop).
        if not is_main:
            return
        if not cfg.probe.enabled:
            return
        if epoch % cfg.probe.every != 0:
            return
        _do_probe_and_save(epoch, step)

    # Step-cadence probe/checkpoint interval (in optimizer steps). 0/null disables.
    # Essential at large scale: one epoch over Uni-Mol is millions of steps, so
    # the epoch cadence alone would almost never save.
    probe_every_steps = int(cfg.probe.get("every_steps", 0) or 0) if probe_enabled else 0
    if probe_every_steps:
        print(f"probe/checkpoint step cadence: every {probe_every_steps} steps", flush=True)

    # ---------------------------------------------------------------------- #
    # RESUME from a previous train_state checkpoint
    # ---------------------------------------------------------------------- #
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
        online.load_state_dict(rs["online"])
        target.load_state_dict(rs["target"])
        predictor.load_state_dict(rs["predictor"])
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
        print(f"RESUMED from {resume_path_cfg}: start_epoch={start_epoch} "
              f"step={start_step}", flush=True)
    elif do_resume:
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
    last_probe_step = -1
    for epoch in range(start_epoch, cfg.optim.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)   # reshuffle shards each epoch
        maybe_probe(epoch, step)
        sync_ranks()                   # wait for rank 0's probe/checkpoint
        last_probe_step = step

        online.train(); predictor.train()
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

            lr = cosine_lr(step, total_steps, cfg.optim.lr, warmup_steps)
            tau = linear_schedule(step, total_steps, cfg.optim.ema_tau_start, cfg.optim.ema_tau_end)
            wd = linear_schedule(step, total_steps, cfg.optim.weight_decay_start, cfg.optim.weight_decay_end)
            for pg in optimizer.param_groups:
                pg["lr"] = lr; pg["weight_decay"] = wd

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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema_update(target_params, online_params, tau)

            running += loss.detach()
            step += 1

            # step-cadence probe + checkpoint (mid-epoch). step advances in
            # lockstep across ranks (DistributedSampler + drop_last), so every
            # rank enters this block together; non-main ranks return immediately
            # and wait at the barrier while rank 0 probes, so `online` is never
            # mutated underneath the probe. Skip if the epoch-cadence probe
            # already ran at this step (avoids duplicate probes at boundaries).
            if probe_every_steps and step % probe_every_steps == 0 and step != last_probe_step:
                _do_probe_and_save(epoch, step)
                last_probe_step = step
                if is_dist:
                    dist.barrier()
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
                    "train/loss": loss.item(), "train/lr": lr,
                    "train/ema_tau": tau, "train/weight_decay": wd,
                    "train/online_out_std": std,
                    "train/identity_loss": identity_loss.item(),
                    "epoch": epoch,
                }
                msg = (f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} "
                       f"identity {identity_loss.item():.4f} lr {lr:.2e} tau {tau:.5f} "
                       f"wd {wd:.4f} out_std {std:.4f}")
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

    maybe_probe(cfg.optim.epochs, step) if cfg.optim.epochs % cfg.probe.every != 0 else None

    # save the headless online encoder (representation = pool(encode_nodes)) plus
    # the full resumable training state.
    save_context_encoder(cfg.optim.epochs - 1)
    save_full_state(cfg.optim.epochs - 1, step)
    print(f"saved context encoder to {ckpt_path} and full state to {state_path}", flush=True)

    if use_wandb:
        wandb.finish()

    if is_dist:
        sync_ranks()
        dist.destroy_process_group()


@hydra.main(version_base=None, config_path="../conf", config_name="pretrain")
def main(cfg: DictConfig):
    if int(os.environ.get("RANK", "0")) == 0:
        print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()