"""
Downstream full fine-tuning of the JEPA-pretrained EquiformerV3 context encoder on Matbench
"""

import copy
import json
import math
import os
from dataclasses import asdict, replace
from typing import Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

import wandb

from finetuning.common import load_pretrained_encoder, move_batch, resolve_device
from finetuning.matbench.readouts import MatBenchModel
from data.datasets.matbench import (
    MATBENCH_TASKS,
    PAPER_TASK_ORDER,
    MatBenchCollator,
    MatBenchDataset,
    MatBenchTaskSpec,
    load_matbench_fold,
)
from data.core.splits import split_indices

from finetuning.execution import configure_execution
from finetuning.optim import build_adamw, ema_update
from finetuning.admet.training.regularization import build_l2sp


# -----------------------------
# checkpointing
# -----------------------------
def save_full_state(path, *, model, ema_model, optimizer, scheduler, epoch,
                    best_val, best_test, best_epoch, epochs_no_improve,
                    y_mean, y_std, eqv3_cfg, target, unit, pool,
                    use_atom_ref, standardize, use_ema):
    """Everything needed to resume an interrupted run faithfully."""
    torch.save(
        {
            "model": model.state_dict(),                # LIVE training weights
            "ema_model": (ema_model.state_dict() if ema_model is not None else None),
            "optimizer": optimizer.state_dict(),        # AdamW moments
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,                             # last COMPLETED epoch
            "best_val": best_val,
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
            "use_atom_ref": use_atom_ref, "standardize": standardize, "ema": use_ema,
        },
        path,
    )


# -----------------------------
# metrics
# -----------------------------
def _mae(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred - y).abs().mean())


def _f1(prob: torch.Tensor, y: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (prob > threshold).float()
    tp = float((pred * y).sum())
    fp = float((pred * (1 - y)).sum())
    fn = float(((1 - pred) * y).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def _balanced_accuracy(prob: torch.Tensor, y: torch.Tensor,
                       threshold: float = 0.5) -> float:
    """Mean of the two per-class recalls. Insensitive to class imbalance, which
    is why it drives selection rather than plain accuracy."""
    n_pos, n_neg = float(y.sum()), float((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pred = (prob > threshold).float()
    tpr = float((pred * y).sum()) / n_pos
    tnr = float(((1 - pred) * (1 - y)).sum()) / n_neg
    return 0.5 * (tpr + tnr)


def _roc_auc(score: torch.Tensor, y: torch.Tensor) -> float:
    """Rank-based ROC-AUC; ties get the average rank. No sklearn dependency."""
    n_pos, n_neg = float(y.sum()), float((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(score)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=torch.float64)
    sorted_vals = score[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    sum_pos = float(ranks[y.bool()].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def compute_metrics(out: torch.Tensor, y: torch.Tensor,
                    spec: MatBenchTaskSpec) -> Dict[str, float]:
    """`out` is [G] denormalised predictions, or [G, 2] logits for classification."""
    if spec.classification:
        prob = torch.softmax(out.double(), dim=-1)[:, 1]
        return {
            "balanced_accuracy": _balanced_accuracy(prob, y),
            "roc_auc": _roc_auc(prob, y),
            "f1": _f1(prob, y),
        }
    return {"mae": _mae(out, y) * spec.scale}


def regression_loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    """The training loss for a regression task, on the STANDARDIZED target.
    """
    if kind == "huber":
        return F.smooth_l1_loss(pred, target)
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "mae":
        return F.l1_loss(pred, target)
    raise ValueError(f"loss must be 'huber', 'mse' or 'mae', got {kind!r}")


# metric key -> higher-is-better
METRIC_DIRECTION: Dict[str, bool] = {
    "mae": False, "f1": True, "roc_auc": True, "balanced_accuracy": True,
}


def selection_metric(spec: MatBenchTaskSpec, c: Optional[DictConfig] = None) -> str:
    if not spec.classification:
        return "mae"
    key = str(c.get("classification_metric", "f1")) if c is not None else "f1"
    if key not in METRIC_DIRECTION:
        raise ValueError(f"classification_metric must be one of "
                         f"{sorted(METRIC_DIRECTION)}, got {key!r}")
    return key


def is_better(new: float, old: float, key: str, min_delta: float) -> bool:
    if math.isnan(new):
        return False
    return (new > old + min_delta) if METRIC_DIRECTION[key] else (new < old - min_delta)


def _fmt(metrics: Dict[str, float]) -> str:
    return " ".join(f"{k} {v:.4f}" for k, v in metrics.items())


def resolve_reduce(c: DictConfig, spec: MatBenchTaskSpec) -> str:
    """How per-atom predictions combine into one structure prediction.
    """
    return "max" if spec.name in set(c.get("max_pool_tasks", [])) else str(c.reduce)


# -----------------------------
# parameter groups
# -----------------------------
def _decay_split(module, wd: float, no_decay_bias_norm: bool):
    if not no_decay_bias_norm:
        return [(list(module.parameters()), wd)]
    decay, no_decay = [], []
    for p in module.parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [pair for pair in ((decay, wd), (no_decay, 0.0)) if pair[0]]


def build_param_groups(model, *, base_lr: float, head_lr_mult: float,
                       encoder_wd: float, head_wd: float,
                       no_decay_bias_norm: bool):
    """Return (param_groups, group_base_lrs, lr_index).

    lr_index maps "encoder"/"head" to the FIRST group index of each block. The
    encoder occupies [lr_index["encoder"], lr_index["head"]) -- a range, not a
    single index, because no_decay_bias_norm splits each block in two.
    """
    groups, base_lrs = [], []
    lr_index = {"encoder": 0}
    for params, wd in _decay_split(model.encoder, encoder_wd, no_decay_bias_norm):
        groups.append({"params": params, "lr": base_lr, "weight_decay": wd})
        base_lrs.append(base_lr)
    lr_index["head"] = len(groups)
    head_lr = base_lr * head_lr_mult
    for params, wd in _decay_split(model.head, head_wd, no_decay_bias_norm):
        groups.append({"params": params, "lr": head_lr, "weight_decay": wd})
        base_lrs.append(head_lr)

    seen = {id(p) for g in groups for p in g["params"]}
    missing = [n for n, p in model.named_parameters() if id(p) not in seen]
    if missing:
        # A parameter outside model.encoder / model.head would silently never
        # be optimised. Loud, because the failure mode is "trains, scores badly".
        raise ValueError(f"parameters outside encoder/head are unoptimised: {missing}")
    return groups, base_lrs, lr_index


def grad_norm(params: List[torch.Tensor]) -> float:
    """Total gradient norm WITHOUT clipping: max_norm=inf computes and returns
    the norm as a side effect but never rescales (inf is never < the norm)."""
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf")))


# -----------------------------
# learning rate schedule
# -----------------------------
class WarmupCosine:
    """Linear warmup, then cosine decay to `min_factor * lr` over the 
    training run.

    One schedule, one horizon: `total_steps` is `epochs * steps_per_epoch`, so
    the LR trajectory is fully determined by the epoch budget and nothing else.
    Measured in optimiser steps rather than epochs, so it means the same thing
    at any `accum_steps`.

    `encoder_groups` is the [lo, hi) group range carrying an extra multiplier,
    the encoder-only ramp -- see encoder_warmup_factor in run_fold.
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 start_factor: float, min_factor: float,
                 encoder_groups: Tuple[int, int] = (0, 0)):
        self.optimizer = optimizer
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.start_factor = start_factor
        self.min_factor = min_factor
        self.enc_lo, self.enc_hi = encoder_groups
        self.encoder_factor = 1.0
        self.set_lr(0)

    def _lr_at(self, step: int, base_lr: float) -> float:
        floor = self.min_factor * base_lr
        if step < self.warmup_steps:
            start = self.start_factor * base_lr
            return start + (base_lr - start) * (step / max(1, self.warmup_steps))
        progress = min(1.0, (step - self.warmup_steps)
                       / max(1, self.total_steps - self.warmup_steps))
        return floor + 0.5 * (base_lr - floor) * (1 + math.cos(math.pi * progress))

    def set_encoder_factor(self, factor: float):
        """Set once per epoch, before that epoch's steps."""
        self.encoder_factor = factor

    def set_lr(self, step: int):
        """Call before every optimiser step."""
        for i, (group, base_lr) in enumerate(zip(self.optimizer.param_groups,
                                                 self.base_lrs)):
            lr = self._lr_at(step, base_lr)
            if self.enc_lo <= i < self.enc_hi:
                lr *= self.encoder_factor
            group["lr"] = lr

    def state_dict(self):
        return {"base_lrs": self.base_lrs}

    def load_state_dict(self, sd):
        self.base_lrs = sd["base_lrs"]


# -----------------------------
# eval
# -----------------------------
@torch.no_grad()
def predict(model, loader, device, y_mean: float, y_std: float,
            classification: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (out, y) on CPU in full precision.

    Regression: `out` is [G] in the target's native units.
    Classification: `out` is [G, 2] raw logits.
    """
    model.eval()
    outs, ys = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch).float()
        if not classification:
            out = out * y_std + y_mean
        outs.append(out.detach().cpu())
        ys.append(batch["y"][:, 0].detach().float().cpu())
    return torch.cat(outs), torch.cat(ys)


# -----------------------------
# one fold
# -----------------------------
def run_fold(cfg: DictConfig, spec: MatBenchTaskSpec, fold: int, device,
             wlog) -> Dict[str, object]:
    c = cfg.matbench
    reduce = resolve_reduce(c, spec)
    torch.manual_seed(c.seed)

    tag = f"{spec.name}_fold{fold}" + ("_scratch" if c.train_from_scratch else "")

    # --- encoder (architecture from the checkpoint) ---
    ckpt_path = c.ckpt_path or os.path.join(cfg.misc.checkpoint_dir,
                                            "context_encoder.pt")
    encoder, eqv3_cfg, ckpt = load_pretrained_encoder(
        ckpt_path, device, load_weights=not c.train_from_scratch
    )
    grad_ckpt = bool(c.get("grad_checkpointing", False))
    eqv3_cfg = replace(eqv3_cfg, grad_checkpointing=grad_ckpt)
    encoder.set_grad_checkpointing(grad_ckpt)
    F_in = eqv3_cfg.num_channels
    cutoff = eqv3_cfg.max_radius

    # --- data ---
    tr_s, tr_y, te_s, te_y = load_matbench_fold(
        spec.name, fold, prefer_official=c.official_folds
    )
    train_full = MatBenchDataset(tr_s, tr_y, tag=f"{spec.name}/f{fold}/train")
    test_ds = MatBenchDataset(te_s, te_y, tag=f"{spec.name}/f{fold}/test")

    # Validation comes out of the training fold, with a seed fixed across folds
    # and runs so the split is reproducible.
    tr_idx, val_idx, _ = split_indices(len(train_full), c.val_frac, 0.0, c.split_seed)

    collate = MatBenchCollator(cutoff=cutoff)
    pin = (device.type == "cuda")

    def make_loader(ds, indices, shuffle, batch_size):
        subset = ds if indices is None else Subset(ds, indices)
        return DataLoader(
            subset, batch_size=batch_size, shuffle=shuffle,
            num_workers=cfg.data.num_workers, collate_fn=collate,
            drop_last=False, pin_memory=pin,
            persistent_workers=(cfg.data.num_workers > 0),
        )

    train_loader = make_loader(train_full, tr_idx, True, c.batch_size)
    val_loader = make_loader(train_full, val_idx, False, c.eval_batch_size)
    test_loader = make_loader(test_ds, None, False, c.eval_batch_size)

    # --- target normalization ---
    if spec.classification or not c.standardize:
        y_mean, y_std = 0.0, 1.0
    else:
        stats_loader = make_loader(train_full, None, False, c.eval_batch_size)
        ys = torch.cat([b["y"][:, 0] for b in stats_loader], 0)
        y_mean, y_std = float(ys.mean()), float(ys.std().clamp_min(1e-8))

    # --- model ---
    model = MatBenchModel(
        encoder, F_in,
        out_dim=(2 if spec.classification else 1),
        reduce=reduce,
        readout=c.readout,
        head_hidden=c.head_hidden,
        use_higher_order=c.use_higher_order_invariants,
        num_layers=c.num_layers,
        activation=c.activation,
        dropout=c.dropout,
    ).to(device)
    model.head.set_target_stats(y_mean, y_std)

    freeze_epochs = int(c.freeze_encoder_epochs)

    def _set_encoder_trainable(flag: bool):
        for p in model.encoder.parameters():
            p.requires_grad_(flag)

    if freeze_epochs > 0:
        _set_encoder_trainable(False)

    use_ema = bool(c.ema)
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model).to(device)      # after set_target_stats
        for p in ema_model.parameters():
            p.requires_grad_(False)

    l2sp = build_l2sp(model, c.get("l2sp", {}),
                      train_from_scratch=bool(c.train_from_scratch),
                      have_pretrained_weights=not bool(c.train_from_scratch))
    l2sp_log_every = int(c.get("l2sp", {}).get("log_every", 500)) if l2sp else 0

    class_weights = None
    if spec.classification and c.get("class_weights", None):
        class_weights = torch.tensor(list(c.class_weights), dtype=torch.float,
                                     device=device)

    scaler = configure_execution(c, model, ema_model, device)
    use_amp = model.amp_dtype is not None
    accum_steps = max(1, int(c.accum_steps))

    print(f"[matbench] {tag}: readout={c.readout} reduce={reduce} "
          f"higher_order={model.use_higher_order} C={F_in} cutoff={cutoff} "
          f"bs={c.batch_size}x{accum_steps} lr={c.lr:.2e} amp={use_amp} "
          f"ema={use_ema} l2sp={'on' if l2sp else 'off'} freeze={freeze_epochs} "
          f"mean={y_mean:.4f} std={y_std:.4f} "
          f"train={len(tr_idx)} val={len(val_idx)} test={len(test_ds)}", flush=True)

    # --- optimizer ---
    param_groups, group_base_lrs, lr_index = build_param_groups(
        model, base_lr=c.lr, head_lr_mult=c.head_lr_mult,
        encoder_wd=c.encoder_weight_decay, head_wd=c.head_weight_decay,
        no_decay_bias_norm=c.no_decay_bias_norm,
    )
    optimizer = build_adamw(param_groups, lr=c.lr, betas=tuple(c.betas),
                                  eps=c.adam_eps)

    sel = selection_metric(spec, c)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum_steps))
    warmup_steps = int(c.lr_warmup_epochs) * steps_per_epoch
    total_steps = c.epochs * steps_per_epoch
    print(f"[matbench] {tag}: {steps_per_epoch} optimizer steps/epoch, "
          f"{total_steps} total ({c.epochs} epochs), warmup {warmup_steps} steps "
          f"({c.lr_warmup_epochs} epochs)",
          flush=True)
    if total_steps < 4 * warmup_steps:
        print(f"[matbench] {tag}: WARNING warmup is {warmup_steps} of "
              f"{total_steps} total steps; lower lr_warmup_epochs or raise epochs.",
              flush=True)

    encoder_warmup_epochs = int(c.encoder_warmup_epochs)

    def encoder_warmup_factor(ep: int) -> float:
        if encoder_warmup_epochs <= 0:
            return 1.0
        since_unfreeze = ep - freeze_epochs
        if since_unfreeze < 0 or since_unfreeze >= encoder_warmup_epochs:
            return 1.0
        return (since_unfreeze + 1) / float(encoder_warmup_epochs)

    if freeze_epochs + encoder_warmup_epochs >= c.epochs:
        print(f"[matbench] {tag}: WARNING the encoder ramp ends at epoch "
              f"{freeze_epochs + encoder_warmup_epochs}, at or past the {c.epochs}-epoch "
              f"budget; lower freeze_encoder_epochs/encoder_warmup_epochs.", flush=True)

    scheduler = WarmupCosine(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        start_factor=c.warmup_start_factor,
        min_factor=min(1.0, float(c.min_lr) / float(c.lr)),
        encoder_groups=(lr_index["encoder"], lr_index["head"]),
    )

    params = list(model.parameters())
    encoder_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters())
    best_val = float("-inf") if METRIC_DIRECTION[sel] else float("inf")
    best_val_metrics: Dict[str, float] = {}
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    step = 0

    for epoch in range(c.epochs):
        if freeze_epochs > 0 and epoch == freeze_epochs:
            _set_encoder_trainable(True)
            print(f"[matbench] {tag}: unfroze encoder @ epoch {epoch}", flush=True)
        scheduler.set_encoder_factor(encoder_warmup_factor(epoch))

        model.train()
        if c.encoder_eval_mode:
            model.encoder.eval()

        running = torch.zeros((), device=device, dtype=torch.float64)
        n_batches = len(train_loader)
        batches_done = 0
        enc_norm = head_norm = 0.0
        optimizer.zero_grad(set_to_none=True)   

        for batch in train_loader:
            batch = move_batch(batch, device)
            y = batch["y"][:, 0]
            scheduler.set_lr(step)

            out = model(batch)
            if spec.classification:
                loss = F.cross_entropy(out, y.long(), weight=class_weights)
            else:
                loss = regression_loss(out, (y - y_mean) / y_std, c.loss)

            scaler.scale(loss / accum_steps).backward()

            is_boundary = ((batches_done + 1) % accum_steps == 0
                           or batches_done + 1 == n_batches)
            if is_boundary:
                will_log = (step % cfg.misc.log_every_steps == 0)
                if c.grad_clip > 0 or will_log:
                    scaler.unscale_(optimizer)
                if c.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        params, c.grad_clip,
                        error_if_nonfinite=bool(c.get("cuequivariance", False)) and not scaler.is_enabled())
                if will_log:
                    enc_norm, head_norm = grad_norm(encoder_params), grad_norm(head_params)

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= scale_before

                if stepped and l2sp is not None and epoch >= freeze_epochs:
                    lr_enc_now = optimizer.param_groups[lr_index["encoder"]]["lr"]
                    do_l2sp_log = will_log and (step % l2sp_log_every == 0)
                    l2sp_metrics = l2sp.step_(lr_enc_now, want_metrics=do_l2sp_log)
                    if do_l2sp_log:
                        l2sp_penalty, l2sp_dist = l2sp_metrics
                        wlog({f"matbench/{spec.name}/fold{fold}/l2sp_penalty": l2sp_penalty,
                              f"matbench/{spec.name}/fold{fold}/l2sp_dist": l2sp_dist})

                if stepped and use_ema:
                    ema_update(ema_model, model, c.ema_decay)

                if will_log:
                    lr_enc = optimizer.param_groups[lr_index["encoder"]]["lr"]
                    lr_head = optimizer.param_groups[lr_index["head"]]["lr"]
                    print(f"[{tag}] ep {epoch} step {step} "
                          f"[{batches_done + 1}/{n_batches}] loss {loss.item():.4f} "
                          f"lr(enc/head) {lr_enc:.2e}/{lr_head:.2e} "
                          f"grad_norm(enc/head) {enc_norm:.2f}/{head_norm:.2f}",
                          flush=True)

                optimizer.zero_grad(set_to_none=True)
                step += 1

            running.add_(loss.detach())
            batches_done += 1

        mean_loss = running.item() / max(1, n_batches)

        eval_model = ema_model if use_ema else model
        vo, vy = predict(eval_model, val_loader, device, y_mean, y_std,
                         spec.classification)
        val_metrics = compute_metrics(vo, vy, spec)
        val_metric = val_metrics[sel]

        if is_better(val_metric, best_val, sel, c.min_delta):
            best_val = val_metric
            best_val_metrics = dict(val_metrics)
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = copy.deepcopy(eval_model.state_dict())
        else:
            epochs_no_improve += 1

        lr_now = optimizer.param_groups[lr_index["encoder"]]["lr"]
        if epoch % cfg.misc.log_every_epochs == 0:
            print(f"== [{tag}] epoch {epoch} loss {mean_loss:.4f} "
                  f"val {_fmt(val_metrics)} {spec.unit} "
                  f"(best {sel} {best_val:.4f} @ {best_epoch}) "
                  f"lr {lr_now:.2e}", flush=True)
            payload = {f"matbench/{spec.name}/fold{fold}/val_{k}": v
                       for k, v in val_metrics.items()}
            payload[f"matbench/{spec.name}/fold{fold}/train_loss"] = mean_loss
            payload[f"matbench/{spec.name}/fold{fold}/lr"] = lr_now
            payload[f"matbench/{spec.name}/fold{fold}/lr_head"] = (
                optimizer.param_groups[lr_index["head"]]["lr"])
            payload["matbench/epoch"] = epoch
            wlog(payload)

        if c.save_last:
            os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)
            last_path = os.path.join(cfg.misc.checkpoint_dir, f"last_{tag}.pt")
            tmp = last_path + ".tmp"
            save_full_state(
                tmp, model=model, ema_model=ema_model, optimizer=optimizer,
                scheduler=scheduler, epoch=epoch, best_val=best_val,
                best_test=float("nan"), best_epoch=best_epoch,
                epochs_no_improve=epochs_no_improve, y_mean=y_mean, y_std=y_std,
                eqv3_cfg=eqv3_cfg, target=spec.name, unit=spec.unit,
                pool=reduce, use_atom_ref=False,
                standardize=(y_std != 1.0), use_ema=use_ema,
            )
            os.replace(tmp, last_path)

        if c.patience is not None and epochs_no_improve >= c.patience:
            print(f"[matbench] {tag}: early stop at epoch {epoch} "
                  f"({c.patience} epochs without improvement)", flush=True)
            break

    # --- ONE test pass, with the best-val weights ---
    eval_model = ema_model if use_ema else model
    if best_state is not None:
        eval_model.load_state_dict(best_state)
    to, ty = predict(eval_model, test_loader, device, y_mean, y_std,
                     spec.classification)
    test_metrics = compute_metrics(to, ty, spec)

    preds_out = (torch.softmax(to.double(), dim=-1)[:, 1] if spec.classification
                 else to).tolist()

    print(f"[matbench] {tag}: TEST {_fmt(test_metrics)} {spec.unit} "
          f"(best val {sel} {best_val:.4f} @ epoch {best_epoch})", flush=True)

    return {
        "task": spec.name, "fold": fold, "unit": spec.unit,
        "best_epoch": best_epoch, "select_on": sel,
        "val": best_val, "val_metrics": best_val_metrics,
        "test": test_metrics[sel],          # kept for back-compat
        "test_metrics": test_metrics,
        "predictions": preds_out,
    }


# -----------------------------
# driver
# -----------------------------
def matbench_finetune(cfg: DictConfig) -> Dict[str, Dict[str, object]]:
    device = torch.device(resolve_device(cfg.misc.device))
    mbc = cfg.matbench

    tasks = list(mbc.tasks)
    folds = list(mbc.folds)

    use_wandb = bool(cfg.wandb.enabled) and cfg.wandb.mode != "disabled"
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project, entity=cfg.wandb.entity,
            name=(f"{cfg.wandb.run_name}-matbench" if cfg.wandb.run_name else None),
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        # x-axis for every per-fold series: each fold's curve starts at 0
        wandb.define_metric("matbench/epoch")
        wandb.define_metric("matbench/*", step_metric="matbench/epoch")

    def wlog(payload):
        if use_wandb:
            wandb.log(payload)

    summary: Dict[str, Dict[str, object]] = {}
    all_records: List[Dict[str, object]] = []

    for task_name in tasks:
        spec = MATBENCH_TASKS[task_name]
        fold_recs = [run_fold(cfg, spec, fold, device, wlog) for fold in folds]
        all_records.extend(fold_recs)

        entry: Dict[str, object] = {
            "unit": spec.unit,
            "n_folds": len(fold_recs),
            "select_on": fold_recs[0]["select_on"],
            "metrics": {},
        }
        for k in sorted({k for r in fold_recs for k in r["test_metrics"]}):
            vals = [float(r["test_metrics"][k]) for r in fold_recs]
            t = torch.tensor(vals, dtype=torch.float64)
            n_nan = int(torch.isnan(t).sum())
            if n_nan:
                print(f"[matbench] WARNING: {task_name}/{k} is NaN on {n_nan} "
                      f"fold(s); reporting the mean over the remaining folds.",
                      flush=True)
                t = t[~torch.isnan(t)]
            entry["metrics"][k] = {
                "mean": float(t.mean()) if t.numel() else float("nan"),
                "std": float(t.std(unbiased=False)) if t.numel() else float("nan"),
                "folds": vals,
            }
        summary[task_name] = entry

        shown = "  ".join(f"{k} {m['mean']:.4f} +/- {m['std']:.4f}"
                          for k, m in entry["metrics"].items())
        print(f"\n[matbench] {task_name}: {shown} {spec.unit} "
              f"over {len(fold_recs)} folds\n", flush=True)

    out_dir = mbc.results_dir or cfg.misc.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "matbench_results.json"), "w") as f:
        json.dump({"summary": summary,
                   "folds": [{k: v for k, v in r.items() if k != "predictions"}
                             for r in all_records]}, f, indent=2)
    with open(os.path.join(out_dir, "matbench_predictions.json"), "w") as f:
        json.dump([{"task": r["task"], "fold": r["fold"],
                    "predictions": r["predictions"]} for r in all_records], f)

    print("\n=== MatBench (mean +/- std over folds) ===", flush=True)
    for task_name in PAPER_TASK_ORDER:
        if task_name not in summary:
            continue
        s = summary[task_name]
        parts = "  ".join(f"{k}={m['mean']:.4f}+/-{m['std']:.4f}"
                          for k, m in s["metrics"].items())
        unit = f" [{s['unit']}]" if s["unit"] else ""
        print(f"  {task_name:24s} {parts}{unit}", flush=True)

    if use_wandb:
        flat = {}
        for task_name, s in summary.items():
            for k, m in s["metrics"].items():
                flat[f"matbench/{task_name}_{k}_mean"] = m["mean"]
                flat[f"matbench/{task_name}_{k}_std"] = m["std"]
        wandb.log(flat)
        wandb.finish()
    return summary


@hydra.main(version_base=None, config_path="../../conf", config_name="finetune_matbench")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    matbench_finetune(cfg)


if __name__ == "__main__":
    main()