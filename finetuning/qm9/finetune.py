"""
Downstream full fine-tuning of the JEPA-pretrained EquiformerV3 context encoder on QM9.

"""

import copy
import os
from dataclasses import asdict, replace
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

import wandb
from atom_jepa.models.jepa_equiformer import EquiformerV3Encoder
from atom_jepa.data.collate import GraphCollator
from data.datasets.qm9 import QM9Dataset
from atom_jepa.data.splits import split_indices
from finetuning.common import load_pretrained_encoder, move_batch, resolve_device
from finetuning.qm9.readouts import build_readout
from finetuning.results_csv import append_row, encoder_tag
from atom_jepa.execution import configure_execution
from finetuning.optim import build_adamw, ema_update

import contextlib   


# native torch_geometric QM9 units -- used only for nicer log labels
QM9_UNITS = {
    "mu": "D", "alpha": "a0^3", "homo": "eV", "lumo": "eV", "gap": "eV",
    "r2": "a0^2", "zpve": "eV", "U0": "eV", "U": "eV", "H": "eV",
    "G": "eV", "Cv": "cal/(mol K)",
}

# -----------------------------
# small utilities 
# -----------------------------

def resolve_amp_dtype(precision, device):
    """QM9 uses BF16 on supported CUDA devices and FP32 on the CPU fallback."""
    if precision not in ("bf16", "fp32"):
        raise ValueError("finetune.precision must be 'bf16' or 'fp32'")
    if precision == "fp32":
        return None
    if device.type != "cuda":
        print("[finetune] BF16 requested; using FP32 on the non-CUDA device", flush=True)
        return None
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16; set finetune.precision=fp32")
    return torch.bfloat16


def make_split(n, ftc, split_seed) -> Tuple[List[int], List[int], List[int]]:
    """Train/val/test indices. """
    train_size = ftc.get("train_size", None)
    val_size = ftc.get("val_size", None)
    if train_size is None or val_size is None:
        return split_indices(n, float(ftc.val_frac), float(ftc.test_frac), split_seed)

    train_size, val_size = int(train_size), int(val_size)
    if train_size + val_size > n:
        raise ValueError(
            f"train_size + val_size = {train_size + val_size} exceeds the dataset "
            f"size {n}; lower them or clear data.limit / the atom-count filters."
        )
    if n != 130831:
        print(f"[finetune] WARNING: dataset has {n} molecules, not the full QM9 "
              f"130831. The absolute split still works, but the indices no longer "
              f"correspond to GotenNet's -- their split is NOT reproduced.",
              flush=True)

    perm = np.random.default_rng(int(split_seed)).permutation(n).tolist()
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]
    return train_idx, val_idx, test_idx


def save_full_state(path, *, model, ema_model, optimizer, scheduler, epoch, step,
                    best_monitor, best_mae, best_test, best_epoch, epochs_no_improve,
                    y_mean, y_std, eqv3_cfg, target, unit, pool,
                    use_atom_ref, standardize, use_ema, monitor, split_seed, freeze_encoder_epochs, encoder_warmup_epochs):
    """Everything needed to resume an interrupted run faithfully."""
    torch.save(
        {
            "model": model.state_dict(),                # LIVE training weights
            "ema_model": (ema_model.state_dict() if ema_model is not None else None),
            "optimizer": optimizer.state_dict(),        # AdamW moments
            "scheduler": scheduler.state_dict(),        # plateau best / num_bad_epochs / lr
            "epoch": epoch,                             # last COMPLETED epoch
            "step": step,                               # global optimizer step (warmup)
            "best_monitor": best_monitor,               # early-stop / LR metric
            "best_mae": best_mae,                       # checkpoint-selection metric
            "best_test": best_test,
            "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "y_mean": y_mean, "y_std": y_std,
            "eqv3_cfg": asdict(eqv3_cfg),
            "target": target, "unit": unit, "pool": pool,
            "use_atom_ref": use_atom_ref, "standardize": standardize,
            "ema": use_ema, "monitor": monitor, "split_seed": split_seed,
            "freeze_encoder_epochs": freeze_encoder_epochs,
            "encoder_warmup_epochs": encoder_warmup_epochs,
        },
        path,
    )


# -----------------------------
# model: pretrained encoder body + per-target readout head
# -----------------------------
class FineTuneModel(nn.Module):
    def __init__(self, encoder: EquiformerV3Encoder, F_in: int, target: str,
                 pool: str = "sum", n_layers: int = 2, n_hidden=None, amp_dtype=None):
        super().__init__()
        self.encoder = encoder
        self.amp_dtype = amp_dtype
        self.head = build_readout(target, F_in, pool=pool,
                                  n_layers=n_layers, n_hidden=n_hidden)
        self.encoder_frozen = False      # set by set_encoder_frozen()

    def set_encoder_frozen(self, frozen: bool, eval_mode: bool = True):
        """Freeze/unfreeze the encoder body.

        `requires_grad_(False)` alone still builds the graph through the encoder;
        combined with the no_grad forward below, backward stops at the head's
        input. Call this AFTER model.train(), which resets child train/eval flags."""
        self.encoder_frozen = frozen
        for p in self.encoder.parameters():
            p.requires_grad_(not frozen)
        if frozen and eval_mode:
            self.encoder.eval()          # frozen == fixed feature extractor

    def forward(self, batch):
        device_type = batch["atomic_numbers"].device.type
        ctx = torch.no_grad() if self.encoder_frozen else contextlib.nullcontext()
        with ctx, torch.autocast(device_type=device_type, dtype=self.amp_dtype,
                            enabled=self.amp_dtype is not None):
            if self.head.needs_vectors:
                node_full, _ = self.encoder.encode_nodes_full(batch)
            else:
                node_scalar, _ = self.encoder.encode_nodes(batch)
        # Keep the small readout, physical pooling, and regression output FP32.
        with torch.autocast(device_type=device_type, enabled=False):
            if self.head.needs_vectors:
                node_full = node_full.float()
                return self.head(node_full[:, 0, :], batch, node_vec=node_full)
            return self.head(node_scalar.float(), batch)






# -----------------------------
# target normalization (TRAIN split only)
# -----------------------------
def fit_target_stats(loader, use_atom_ref, standardize) -> Tuple[float, float]:
    """Mean/std of the (optionally atom-referenced) target over the train split.
    Returns (0, 1) when standardize is off, i.e. predict
    the (referenced) target directly in native units."""
    if not standardize:
        return 0.0, 1.0
    ys = []
    for batch in loader:
        y = batch["y"][:, 0]
        if use_atom_ref:
            y = y - batch["atom_ref"]
        ys.append(y)
    r = torch.cat(ys, 0)
    return float(r.mean()), float(r.std().clamp_min(1e-8))


@torch.no_grad()
def evaluate(model, loader, device, use_atom_ref, y_mean, y_std, loss_type
             ) -> Tuple[float, float]:
    """Return (MAE in the target's native unit, mean training-space loss). """
    model.eval()
    total_abs = torch.zeros((), device=device, dtype=torch.float64)
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    n = 0
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch)                                  # training space
        y = batch["y"][:, 0]

        y_ref = y - batch["atom_ref"] if use_atom_ref else y
        yb = (y_ref - y_mean) / y_std
        if loss_type == "mse":
            total_loss.add_(F.mse_loss(pred, yb, reduction="sum"))
        else:
            total_loss.add_(F.l1_loss(pred, yb, reduction="sum"))

        pred_native = pred * y_std + y_mean
        if use_atom_ref:
            pred_native = pred_native + batch["atom_ref"]
        total_abs.add_((pred_native - y).abs().sum())
        n += y.numel()
    metrics = torch.stack((total_abs, total_loss)) / max(1, n)
    mae, loss = metrics.cpu().tolist()
    return mae, loss


# -----------------------------
# main fine-tuning routine
# -----------------------------
def finetune(cfg: DictConfig) -> Dict[str, float]:
    device = torch.device(resolve_device(cfg.misc.device))

    ftc = cfg.finetune
    amp_dtype = resolve_amp_dtype(str(ftc.get("precision", "bf16")), device)
    target = ftc.target
    pool = ftc.pool                              # "sum" | "mean"  (atomwise readout choice)
    use_atom_ref = bool(ftc.use_atom_ref)        # subtract QM9 atom reference
    standardize = bool(ftc.standardize)          
    unit = QM9_UNITS.get(target, "")

    seed = int(ftc.get("seed", cfg.misc.seed))
    split_seed = int(ftc.get("split_seed", seed))
    torch.manual_seed(seed)

    epochs = int(ftc.epochs)
    base_lr = float(ftc.lr)
    bs = int(ftc.batch_size)
    weight_decay = float(ftc.weight_decay)
    grad_clip = float(ftc.grad_clip)             # <= 0 disables clipping
    head_lr_mult = float(ftc.head_lr_mult)
    use_ema = bool(ftc.ema)
    ema_decay = float(ftc.ema_decay)
    loss_type = str(ftc.get("loss", "mse")).lower()      # "mae" | "mse"
    adam_eps = float(ftc.get("adam_eps", 1e-7))        
    warmup_steps = int(ftc.get("lr_warmup_steps", 0))   # linear LR warmup
    head_n_layers = int(ftc.get("head_n_layers", 2))
    head_n_hidden = ftc.get("head_n_hidden", None)      # null -> match encoder width

    # which metric drives the LR schedule and early stopping.
    monitor = str(ftc.get("monitor", "loss")).lower()
    if monitor not in ("loss", "mae"):
        raise ValueError(f"finetune.monitor must be 'loss' or 'mae', got {monitor!r}")

    # 'plateau' scheduler + early-stopping knobs
    lr_factor = float(ftc.get("lr_factor", 0.8))
    lr_patience = int(ftc.get("lr_patience", 15))
    min_lr = float(ftc.get("min_lr", 1e-7))
    patience = int(ftc.get("patience", 150))

    # resume / checkpointing knobs
    do_resume = bool(ftc.get("resume", False))
    save_last = bool(ftc.get("save_last", True))

    # train-from-scratch baseline: build the SAME model but skip loading pretrained
    # encoder weights, so finetune-vs-scratch is a clean controlled comparison.
    train_from_scratch = bool(ftc.get("train_from_scratch", False))
    ckpt_tag = "_scratch" if train_from_scratch else ""   # keep run artifacts separate

    # --- encoder freezing / gradual unfreeze
    freeze_epochs = int(ftc.get("freeze_encoder_epochs", 0))
    enc_warmup_epochs = float(ftc.get("encoder_warmup_epochs", 0.0))
    freeze_eval_mode = bool(ftc.get("freeze_encoder_eval", True))
    reset_on_unfreeze = bool(ftc.get("reset_schedule_on_unfreeze", True))

    if freeze_epochs > 0 and train_from_scratch:
        print("[finetune] WARNING: freeze_encoder_epochs > 0 with "
              "train_from_scratch=true freezes a RANDOM encoder -- the head is "
              "being fit to random features.", flush=True)
    if freeze_epochs >= epochs:
        print(f"[finetune] WARNING: freeze_encoder_epochs={freeze_epochs} >= "
              f"epochs={epochs}; the encoder is never unfrozen (this is a probe, "
              f"and finetuning/probing/qm9_frozen_encoder_probe.py does it far faster with cached features).",
              flush=True)

    # wandb (optional)
    use_wandb = bool(cfg.wandb.enabled) and cfg.wandb.mode != "disabled"
    if use_wandb:
        scratch_tag = "-scratch" if train_from_scratch else ""
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=((cfg.wandb.run_name + "-finetune" + scratch_tag)
                  if cfg.wandb.run_name else f"finetune-{target}{scratch_tag}"),
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def wlog(d, step):
        if use_wandb:
            wandb.log(d, step=step)

    # --- load pretrained encoder ---
    ckpt_path = ftc.get("ckpt_path", None) or os.path.join(
        cfg.misc.checkpoint_dir, "context_encoder.pt"
    )
    encoder, eqv3_cfg, ckpt = load_pretrained_encoder(
        ckpt_path, device, load_weights=not train_from_scratch
    )
    # Pretraining checkpoints predate this field, so _cfg_from_ckpt fills the
    # dataclass default (True); this is what actually decides it for fine-tuning.
    grad_ckpt = bool(ftc.get("grad_checkpointing", True))
    eqv3_cfg = replace(eqv3_cfg, grad_checkpointing=grad_ckpt)
    encoder.set_grad_checkpointing(grad_ckpt)
    optimize_grid_mlp = bool(ftc.get("optimize_grid_mlp", True))
    encoder.body.set_grid_mlp_optimization(optimize_grid_mlp)
    F_in = eqv3_cfg.num_channels
    cutoff = eqv3_cfg.max_radius  # match the encoder's pretraining graph exactly
    pretrain_epoch = ckpt.get("epoch", "?")

    collate = GraphCollator(cutoff=cutoff)

    # --- data ---
    dataset = QM9Dataset(
        root=cfg.data.root,
        target=target,
        min_atoms=cfg.data.min_atoms,
        max_atoms=cfg.data.max_atoms,
        limit=cfg.data.limit,
    )
    if use_atom_ref and not dataset.has_atomref:
        print(f"[finetune] note: target {target!r} has no QM9 atom reference; "
              f"use_atom_ref=true is a no-op here (atom_ref=0 for every molecule). "
              f"GotenNet behaves the same way -- dataset_meta.atomref is None or "
              f"all-zeros for exactly these targets.", flush=True)

    train_idx, val_idx, test_idx = make_split(len(dataset), ftc, split_seed)

    num_workers = cfg.data.num_workers
    pin = (device.type == "cuda")

    def make_loader(indices, shuffle, drop_last):
        return DataLoader(
            Subset(dataset, indices),
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
            drop_last=drop_last,
            pin_memory=pin,
            persistent_workers=(num_workers > 0),
        )

    train_loader = make_loader(train_idx, shuffle=True, drop_last=False)
    val_loader = make_loader(val_idx, shuffle=False, drop_last=False)
    test_loader = make_loader(test_idx, shuffle=False, drop_last=False)

    # --- target normalization (TRAIN split only) ---
    if standardize:
        stats_loader = make_loader(train_idx, shuffle=False, drop_last=False)
        y_mean, y_std = fit_target_stats(stats_loader, use_atom_ref, standardize)
    else:
        y_mean, y_std = 0.0, 1.0

    if train_from_scratch:
        print(f"[finetune] TRAIN FROM SCRATCH: rebuilt encoder architecture from "
              f"{ckpt_path} (eqv3_cfg only) with RANDOM init; pretrained weights ignored. "
              f"C={F_in} cutoff={cutoff}", flush=True)
    else:
        print(f"[finetune] loaded encoder from {ckpt_path} (pretrain epoch {pretrain_epoch}); "
              f"C={F_in} cutoff={cutoff}", flush=True)
    print(f"[finetune] target={target} [{unit}] pool={pool} use_atom_ref={use_atom_ref} "
          f"standardize={standardize} (mean={y_mean:.4f} std={y_std:.4f}) | "
          f"split train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
          f"(seed={seed} split_seed={split_seed})", flush=True)
    print(f"[finetune] loss={loss_type} monitor={monitor} lr={base_lr:g} "
          f"warmup={warmup_steps} wd={weight_decay:g} eps={adam_eps:g} "
          f"clip={grad_clip:g} bs={bs} ema={use_ema}", flush=True)

    # --- model / EMA ---
    model = FineTuneModel(encoder, F_in, target, pool=pool,
                          n_layers=head_n_layers, n_hidden=head_n_hidden,
                          amp_dtype=amp_dtype).to(device)
    print(f"[finetune] readout head: {type(model.head).__name__} for target {target!r} "
          f"(needs_vectors={model.head.needs_vectors})", flush=True)
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)

    # --- optimizer: AdamW  ---
    optimizer = build_adamw(
        [
            {"params": list(model.encoder.parameters()), "lr": base_lr},
            {"params": list(model.head.parameters()), "lr": base_lr * head_lr_mult},
        ],
        lr=base_lr,
        weight_decay=weight_decay,
        eps=adam_eps,
    )
    # Remember each group's target LR so warmup can scale them independently.
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]

    def apply_warmup(global_step) -> bool:
        """Linear LR warmup over the first `warmup_steps` optimizer steps.
        """
        if warmup_steps <= 0 or global_step >= warmup_steps:
            return False
        scale = min(1.0, float(global_step + 1) / float(warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = scale * group["base_lr"]
        return True

    # --- scheduler: ReduceLROnPlateau, stepped once per epoch on `monitor` ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr,
    )

    all_params = list(model.parameters())

    # Encoder LR multiplier.
    steps_per_epoch = max(1, len(train_loader))
    freeze_steps = freeze_epochs * steps_per_epoch
    enc_warmup_total = int(round(enc_warmup_epochs * steps_per_epoch))

    def encoder_lr_scale(global_step: int) -> float:
        if global_step < freeze_steps:
            return 0.0
        if enc_warmup_total <= 0:
            return 1.0
        return min(1.0, float(global_step - freeze_steps + 1) / float(enc_warmup_total))

    start_epoch = 0
    step = 0
    last_epoch = start_epoch - 1
    best_monitor = float("inf")   # drives LR schedule + early stopping
    best_mae = float("inf")       # selects the saved checkpoint
    best_test = float("nan")
    best_epoch = -1
    epochs_no_improve = 0

    # ---------------------------------------------------------------
    # RESUME
    # ---------------------------------------------------------------
    def _check_config_match(meta, where):
        for key, cur in (("target", target),
                         ("standardize", standardize),
                         ("use_atom_ref", use_atom_ref),
                         ("pool", pool),
                         ("monitor", monitor),
                         ("freeze_encoder_epochs", freeze_epochs),
                         ("encoder_warmup_epochs", enc_warmup_epochs),
                         ("split_seed", split_seed)):
            old = meta.get(key, None)
            if old is not None and old != cur:
                print(f"[finetune] WARNING: {where} has {key}={old!r} but current "
                      f"cfg has {key}={cur!r}. Continuing anyway -- make sure this "
                      f"is intended (mismatched normalization/target invalidates MAE).",
                      flush=True)

    resume_path = ftc.get("resume_path", None) or os.path.join(
        cfg.misc.checkpoint_dir, f"last_{target}{ckpt_tag}.pt"
    )

    if do_resume and os.path.exists(resume_path):
        rs = torch.load(resume_path, map_location=device, weights_only=False)
        _check_config_match(rs, resume_path)
        model.load_state_dict(rs["model"])
        if use_ema and rs.get("ema_model") is not None:
            ema_model.load_state_dict(rs["ema_model"])
        optimizer.load_state_dict(rs["optimizer"])
        # load_state_dict drops our custom key, so reinstate the warmup targets.
        # Per group: the head's target is base_lr * head_lr_mult, and
        # reset_schedule_on_unfreeze writes base_lr back into lr, so a wrong
        # value here would permanently lose the multiplier.
        for group, target_lr in zip(optimizer.param_groups,
                                    (base_lr, base_lr * head_lr_mult)):
            group.setdefault("base_lr", target_lr)
        scheduler.load_state_dict(rs["scheduler"])
        best_monitor = float(rs.get("best_monitor", rs.get("best_val", float("inf"))))
        best_mae = float(rs.get("best_mae", rs.get("best_val", float("inf"))))
        best_test = float(rs["best_test"])
        best_epoch = int(rs["best_epoch"])
        epochs_no_improve = int(rs["epochs_no_improve"])
        start_epoch = int(rs["epoch"]) + 1
        step = int(rs.get("step", 0))
        # prefer the stats the checkpoint was trained with
        y_mean = float(rs.get("y_mean", y_mean))
        y_std = float(rs.get("y_std", y_std))
        # restore RNG streams (tensors must be CPU ByteTensors)
        torch.set_rng_state(rs["torch_rng_state"].cpu())
        if rs.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rs["cuda_rng_state"]])
        print(f"[finetune] RESUMED from {resume_path}: start_epoch={start_epoch} "
              f"step={step} best_{monitor}={best_monitor:.4f} "
              f"best_mae={best_mae:.4f} @ {best_epoch} (test {best_test:.4f})", flush=True)

    enc_group = optimizer.param_groups[0]      # encoder group is built first
    model.head.set_target_stats(y_mean, y_std)
    if use_ema:
        ema_model.head.set_target_stats(y_mean, y_std)

    # In-place compilation preserves checkpoint keys and optimizer references.
    # Compile after deepcopy/resume; geometry and FP32 readout remain eager.
    compile_blocks = bool(ftc.get("compile_blocks", True)) and device.type == "cuda"
    configure_execution(ftc, model, ema_model, device)
    print(f"[finetune] precision={'bf16' if amp_dtype is not None else 'fp32'} "
          f"(readout/loss fp32) optimize_grid_mlp={optimize_grid_mlp} "
          f"compile_blocks={compile_blocks}", flush=True)

    for epoch in range(start_epoch, epochs):
        last_epoch = epoch
        frozen = epoch < freeze_epochs
        model.train()
        model.set_encoder_frozen(frozen, eval_mode=freeze_eval_mode)

        # Unfreeze boundary. Also fires correctly when resuming exactly here,
        # since a run that had already passed it saved the post-reset scheduler.
        if freeze_epochs > 0 and epoch == freeze_epochs:
            print(f"[finetune] UNFREEZING encoder at epoch {epoch} "
                  f"(encoder LR warmup over {enc_warmup_epochs:g} epochs = "
                  f"{enc_warmup_total} steps)", flush=True)
            if reset_on_unfreeze:
                # The head converged against fixed features, so plateau may have
                # already cut the LR and the early-stop counter may be well into
                # its patience. Both are stale the moment the body starts moving.
                for g in optimizer.param_groups:
                    g["lr"] = g["base_lr"]
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=lr_factor,
                    patience=lr_patience, min_lr=min_lr,
                )
                best_monitor = float("inf")
                epochs_no_improve = 0
                print("[finetune] plateau scheduler + early-stop counter reset; "
                      "LR restored to base (best_mae/checkpoint kept)", flush=True)

        running = torch.zeros((), device=device, dtype=torch.float64)
        warming = False
        enc_scale = 1.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = move_batch(batch, device)

            y = batch["y"][:, 0]
            if use_atom_ref:
                y = y - batch["atom_ref"]
            yb = (y - y_mean) / y_std

            pred = model(batch)
            loss = F.mse_loss(pred, yb) if loss_type == "mse" else F.l1_loss(pred, yb)

            loss.backward()
            if grad_clip > 0:
                # frozen encoder params have .grad is None; clip_grad_norm_ skips them
                torch.nn.utils.clip_grad_norm_(
                    all_params, grad_clip, error_if_nonfinite=bool(ftc.get("cuequivariance", False)))
            warming = apply_warmup(step)

            # Encoder scale is applied as a TEMPORARY override so ReduceLROnPlateau
            # keeps operating on the unscaled LR it wrote.
            enc_scale = encoder_lr_scale(step)
            if enc_scale < 1.0:
                enc_lr_full = enc_group["lr"]
                enc_group["lr"] = enc_lr_full * enc_scale
                optimizer.step()
                enc_group["lr"] = enc_lr_full
            else:
                optimizer.step()

            if use_ema:
                ema_update(ema_model, model, ema_decay)

            running.add_(loss.detach())
            step += 1
            if step % cfg.misc.log_every == 0:
                loss_value = loss.item()
                lr_enc = enc_group["lr"] * enc_scale
                lr_head = optimizer.param_groups[1]["lr"]
                tag = " (warmup)" if warming else ""
                tag += " (frozen)" if frozen else (" (enc-warmup)" if enc_scale < 1.0 else "")
                print(f"epoch {epoch} step {step} loss {loss_value:.4f} "
                      f"lr_enc {lr_enc:.2e} lr_head {lr_head:.2e}{tag}", flush=True)
                wlog({"finetune/train_loss": loss_value,
                      "finetune/lr": lr_head,
                      "finetune/lr_encoder": lr_enc,
                      "finetune/encoder_frozen": float(frozen),
                      "epoch": epoch}, step)

        mean_loss = running.item() / max(1, len(train_loader))

        eval_model = ema_model if use_ema else model
        val_mae, val_loss = evaluate(eval_model, val_loader, device, use_atom_ref,
                                     y_mean, y_std, loss_type)
        monitored = val_loss if monitor == "loss" else val_mae

        # step the plateau scheduler on the monitored metric, once per epoch
        scheduler.step(monitored)

        # --- checkpoint selection: val MAE (GotenNet's ModelCheckpoint metric) ---
        if val_mae < best_mae:
            best_mae = val_mae
            best_test, _ = evaluate(eval_model, test_loader, device, use_atom_ref,
                                    y_mean, y_std, loss_type)
            best_epoch = epoch
            if bool(ftc.get("save_best", True)):
                os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)
                out_path = os.path.join(cfg.misc.checkpoint_dir, f"finetuned_{target}{ckpt_tag}.pt")
                torch.save(
                    {
                        "model": eval_model.state_dict(),
                        "eqv3_cfg": asdict(eqv3_cfg),
                        "target": target, "unit": unit, "pool": pool,
                        "use_atom_ref": use_atom_ref, "standardize": standardize,
                        "y_mean": y_mean, "y_std": y_std,
                        "val_mae": best_mae, "test_mae": best_test,
                        "val_loss": val_loss, "epoch": best_epoch, "ema": use_ema,
                    },
                    out_path,
                )

        # --- early stopping: the monitored metric  ---
        if monitored < best_monitor:
            best_monitor = monitored
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # --- write a full-state checkpoint EVERY epoch so the next runtime cutoff
        #     is cleanly resumable ---
        if save_last:
            os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)
            last_path = os.path.join(cfg.misc.checkpoint_dir, f"last_{target}{ckpt_tag}.pt")
            tmp_path = last_path + ".tmp"
            save_full_state(
                tmp_path,
                model=model, ema_model=ema_model, optimizer=optimizer,
                scheduler=scheduler, epoch=epoch, step=step,
                best_monitor=best_monitor, best_mae=best_mae,
                best_test=best_test, best_epoch=best_epoch,
                epochs_no_improve=epochs_no_improve, y_mean=y_mean, y_std=y_std,
                eqv3_cfg=eqv3_cfg, target=target, unit=unit, pool=pool,
                use_atom_ref=use_atom_ref, standardize=standardize,
                use_ema=use_ema, monitor=monitor, split_seed=split_seed,
                freeze_encoder_epochs=freeze_epochs,
                encoder_warmup_epochs=enc_warmup_epochs,
            )
            os.replace(tmp_path, last_path)  # atomic: a kill mid-write won't corrupt it

        lr_now = optimizer.param_groups[1]["lr"]                 # head / scheduler LR
        lr_enc_now = enc_group["lr"] * encoder_lr_scale(step)    # effective encoder LR
        print(f"== epoch {epoch} mean_loss {mean_loss:.4f} val_loss {val_loss:.4f} "
              f"val_mae {val_mae:.4f} {unit} (best val_mae {best_mae:.4f} @ epoch "
              f"{best_epoch}, test {best_test:.4f} {unit}) lr {lr_now:.2e}", flush=True)
        wlog({
            "finetune/epoch_loss": mean_loss,
            f"finetune/{target}_val_loss": val_loss,
            f"finetune/{target}_val_mae": val_mae,
            f"finetune/{target}_best_val_mae": best_mae,
            f"finetune/{target}_best_test_mae": best_test,
            "finetune/lr": lr_now,
            "finetune/lr_encoder": lr_enc_now,
            "epoch": epoch,
        }, step)

        # early stopping on no improvement in the monitored metric
        if epochs_no_improve >= patience:
            print(f"[finetune] early stopping at epoch {epoch} "
                  f"(no val-{monitor} improvement for {patience} epochs)", flush=True)
            break

    results = {
        f"finetune/{target}_val_mae": best_mae,
        f"finetune/{target}_test_mae": best_test,
        f"finetune/{target}_best_epoch": float(best_epoch),
    }
    print(f"[finetune] done ({unit}): " + " ".join(f"{k}={v:.4f}" for k, v in results.items()),
          flush=True)
    wlog(results, step)

    csv_path = append_row(cfg.misc.get("results_csv", None), {
        "script": ("finetune_scratch" if train_from_scratch else "finetune"),
        "encoder": encoder_tag(ckpt_path),
        "target": target, "unit": unit,
        "val_mae": best_mae, "test_mae": best_test, "best_epoch": best_epoch,
        "epochs_run": last_epoch + 1,
        "n_head_params": sum(p.numel() for p in model.head.parameters()),
        "layers": "final", "f_in": F_in,
        "f_vec": (F_in if model.head.needs_vectors else ""),
        "pool": pool, "loss": loss_type, "monitor": monitor,
        "standardize": standardize, "use_atom_ref": use_atom_ref,
        "seed": seed, "split_seed": split_seed,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "run_name": (cfg.wandb.run_name or ""), "ckpt_path": str(ckpt_path),
    })
    if csv_path:
        print(f"[finetune] results row appended to {csv_path}", flush=True)

    if use_wandb:
        wandb.finish()
    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="finetune_qm9")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    finetune(cfg)


if __name__ == "__main__":
    main()
