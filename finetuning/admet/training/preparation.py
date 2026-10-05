"""Prepare multi-task loaders, training-set statistics, and task metadata."""

from dataclasses import dataclass
from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader, DistributedSampler

from finetuning.admet.training import distributed as ddp
from finetuning.admet.training.helpers import _fit_apply_fusion_feature_scaler
from finetuning.admet.training.loss import binary_positive_weights, fit_target_stats_mt


@dataclass
class PreparedData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    y_mean: torch.Tensor
    y_std: torch.Tensor
    reg_task_idx: list[int]
    pos_weight: torch.Tensor | None
    feature_dim: int
    report_scale: Any
    plot_task_idx: list[int]


def make_loader(dataset, *, batch_size, collate_fn, workers, device, dist_state,
                training=False):
    """Shard only training; validation/test retain complete conformer groups."""
    sampler = (DistributedSampler(dataset, num_replicas=dist_state.world_size,
                                  rank=dist_state.rank, shuffle=True, drop_last=False)
               if training and dist_state.enabled else None)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=training and sampler is None,
        sampler=sampler, num_workers=workers, collate_fn=collate_fn,
        drop_last=False, pin_memory=device.type == "cuda", persistent_workers=workers > 0)


def log_label_coverage(train_ds, valid_ds, test_ds, task_cols):
    counts = [ds.mask.sum(dim=0) for ds in (train_ds, valid_ds, test_ds)]
    table = wandb.Table(
        columns=["task", "n_train", "n_val", "n_test"],
        data=[[task, *(int(count[i]) for count in counts)] for i, task in enumerate(task_cols)])
    wandb.log({"coverage_table": table})


def prepare_data(cfg, train_ds, valid_ds, test_ds, task_cols, collate_fn, *,
                 device, dist_state, log_mask=None, task_kinds=None,
                 use_wandb=False, run_label="") -> PreparedData:
    """Fit transforms on training only and prepare loaders and per-task metadata."""
    ftc = cfg.finetune
    _fit_apply_fusion_feature_scaler(ftc.get("fusion", {}), train_ds, [valid_ds, test_ds])
    workers = ddp.loader_workers(int(cfg.data.get("num_workers", 4)), dist_state)
    batch_size = int(ftc.get("batch_size", 64))
    loaders = [make_loader(ds, batch_size=batch_size, collate_fn=collate_fn,
                           workers=workers, device=device, dist_state=dist_state,
                           training=i == 0)
               for i, ds in enumerate((train_ds, valid_ds, test_ds))]
    if dist_state.enabled and dist_state.is_main:
        print(f"[{run_label}] DDP: world_size={dist_state.world_size} "
              f"batch_size={batch_size}/rank (effective {batch_size * dist_state.world_size}) "
              f"num_workers={workers}/rank", flush=True)
    y_mean, y_std = fit_target_stats_mt(
        train_ds.labels, bool(ftc.get("standardize", True)), log_mask, task_kinds=task_kinds)
    reg_task_idx = ([i for i, kind in enumerate(task_kinds) if kind != "binary"]
                    if task_kinds is not None else list(range(len(task_cols))))
    report_scale = getattr(train_ds, "report_scale", None)
    if report_scale is not None and dist_state.is_main:
        print(f"[{run_label}] secondary report scale -- {report_scale.describe()}", flush=True)
    pos_weight = binary_positive_weights(
        train_ds.labels, task_kinds, ftc.get("pos_weight", "auto"), device=device)
    if use_wandb and bool(ftc.get("per_task_logging", True)):
        log_label_coverage(train_ds, valid_ds, test_ds, task_cols)
    counts = valid_ds.mask.sum(dim=0).numpy()
    ranked = sorted((i for i in range(len(task_cols)) if counts[i] > 0), key=lambda i: counts[i])
    n_plot_tasks = int(ftc.get("n_plot_tasks", 3))
    return PreparedData(
        loaders[0], loaders[1], loaders[2], y_mean, y_std, reg_task_idx, pos_weight,
        int(getattr(train_ds, "feature_dim", 0)), report_scale,
        sorted(set(ranked[-n_plot_tasks:] + ranked[:n_plot_tasks])))
