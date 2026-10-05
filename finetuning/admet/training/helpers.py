"""Reusable tensor, preprocessing, logging, scheduling, and checkpoint helpers."""

import math
from dataclasses import asdict
from typing import Dict, Sequence

import numpy as np
import torch


def grad_norm(params: Sequence[torch.Tensor]) -> float:
    """Compute the total gradient norm without clipping; parameter groups share a separate
    clipping budget."""
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf")))

def _wandb_key(name: str) -> str:
    """Sanitize slashes in task names for W&B keys; other outputs keep the original names."""
    return name.replace("/", "-")

def _wandb_metric_summary(report: Dict, prefix: str = "test") -> Dict:
    """Write numeric report entries to W&B summary keys, skipping raw arrays.

    Per-task entries become prefix_metric/task keys; scalar entries become prefix_key.
    Nested ensemble curves are logged separately."""
    out: Dict[str, float] = {}
    for k, v in report.items():
        if k in ("per_mol_preds", "labels", "curve"):
            continue
        if isinstance(v, dict):
            sub = "mae" if k == "per_task" else k.replace("per_task_", "")
            for t, x in v.items():
                out[f"{prefix}_{sub}/{_wandb_key(t)}"] = float(x)
        elif isinstance(v, (int, float, np.floating, np.integer)):
            out[f"{prefix}_{k}"] = float(v)
    return out

def _fit_apply_fusion_feature_scaler(fusion_cfg, train_ds, other_dss) -> None:
    """Fit fusion feature scaling on valid training rows and apply it to all splits.

    Support none, standard, and quantile scaling. Preserve failed-feature rows and guard
    against rescaling dataset objects reused across seeds."""
    kind = str(fusion_cfg.get("feature_scaling", "none")).lower()
    if kind == "none":
        return
    if kind not in ("standard", "quantile"):
        raise ValueError(
            f"Unknown finetune.fusion.feature_scaling {kind!r}; expected none|standard|quantile")
    if int(getattr(train_ds, "feature_dim", 0)) == 0:
        return   # no features on this dataset at all -- nothing to scale
    if getattr(train_ds, "_fusion_scaler_applied", False):
        return

    X_train = train_ds.feature_matrix()
    if X_train.shape[0] == 0:
        return   # every molecule failed featurization; MultiTaskFinetuneDataset already raised if n_molecules>0
    if kind == "standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_train)
    else:
        from sklearn.preprocessing import QuantileTransformer
        n_quantiles = min(int(fusion_cfg.get("n_quantiles", 1000)), X_train.shape[0])
        scaler = QuantileTransformer(
            n_quantiles=n_quantiles, output_distribution="uniform",
            subsample=X_train.shape[0], random_state=0,
        ).fit(X_train)

    for ds in [train_ds, *other_dss]:
        if int(getattr(ds, "feature_dim", 0)) == 0:
            continue
        ds.apply_feature_transform(scaler.transform)
        ds._fusion_scaler_applied = True

def _per_mol(t: torch.Tensor, batch) -> torch.Tensor:
    """Average [G,T] conformer rows into [B,T] molecule rows; identical labels/masks
    are preserved. Identity when ungrouped."""
    mi = batch.get("mol_index") if isinstance(batch, dict) else None
    if mi is None:
        return t
    n_mol = int(mi.max().item()) + 1
    counts = torch.zeros(n_mol, device=t.device, dtype=t.dtype)
    counts.index_add_(0, mi, torch.ones_like(mi, dtype=t.dtype))
    out = torch.zeros(n_mol, t.shape[1], device=t.device, dtype=t.dtype)
    out.index_add_(0, mi, t)
    return out / counts.clamp_min(1.0).unsqueeze(1)

def _json_safe(report):
    return {k: v for k, v in report.items()
            if k not in ("per_mol_preds", "labels", "per_graph_preds", "mol_index")}

def save_full_state(path, *, model, ema_model, optimizer, scheduler, epoch,
                    best_val, best_test, best_epoch, epochs_no_improve,
                    y_mean, y_std, eqv3_cfg, name, task, pool, standardize,
                    log_transform, use_ema):
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": (ema_model.state_dict() if ema_model is not None else None),
            "optimizer": optimizer.state_dict(),
            "scheduler": (scheduler.state_dict() if scheduler is not None else None),
            "epoch": epoch,
            "best_val": best_val, "best_test": best_test, "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "y_mean": y_mean, "y_std": y_std,
            "eqv3_cfg": asdict(eqv3_cfg),
            "dataset": name, "task_kind": task.kind, "metric": task.metric_name,
            "pool": pool, "standardize": standardize, "log_transform": log_transform,
            "ema": use_ema,
        },
        path,
    )


def cosine_factor(ep: int, *, warmup_epochs: int, epochs: int,
                  min_lr: float, base_lr: float) -> float:
    """Warm up linearly, then decay to min_lr on the original epoch schedule."""
    if warmup_epochs > 0 and ep < warmup_epochs:
        return (ep + 1) / float(warmup_epochs)
    denom = max(1, epochs - warmup_epochs)
    prog = min(max((ep - warmup_epochs) / denom, 0.0), 1.0)
    min_ratio = min_lr / max(base_lr, 1e-12)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))


def encoder_warmup_factor(ep: int, *, freeze_epochs: int, warmup_epochs: int) -> float:
    """Ramp encoder LR after unfreezing, independently of the global schedule."""
    if warmup_epochs <= 0:
        return 1.0
    since_unfreeze = ep - freeze_epochs
    if since_unfreeze < 0 or since_unfreeze >= warmup_epochs:
        return 1.0
    return (since_unfreeze + 1) / float(warmup_epochs)


def set_encoder_trainable(encoder: torch.nn.Module, flag: bool) -> None:
    """Freeze or unfreeze encoder parameters without changing its train/eval mode."""
    for parameter in encoder.parameters():
        parameter.requires_grad_(flag)
