"""Coordinate multi-task data preparation, training, selection, and final evaluation."""

from functools import partial
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import wandb
from torch import nn

import finetuning.admet.training.distributed as ddp
from finetuning.admet.metrics import evaluate_multitask
from finetuning.admet.training.checkpoints import (
    initialize_selection,
    model_metadata,
    save_final_checkpoint,
    save_last_checkpoint,
    update_selection,
)
from finetuning.admet.training.epoch import train_epoch
from atom_jepa.execution import configure_execution
from finetuning.admet.training.evaluation import (
    build_result,
    evaluate_final,
    log_training_epoch,
    log_validation_scatter,
    validate_epoch,
)
from finetuning.admet.training.loss import LogMask
from finetuning.admet.training.preparation import prepare_data
from finetuning.admet.training.state import prepare_model, resume_training


def log_epoch_metrics(cfg, state, data, metrics, val_report, val_macro, primary, *,
                      epoch, device, dist_state, task_cols, dataset_name, run_tag, wlog):
    if dist_state.is_main:
        log_training_epoch(
            cfg.finetune, epoch=epoch, mean_loss=metrics.mean_loss, val_report=val_report,
            val_macro=val_macro, selection=primary, train_macro=metrics.macro,
            train_rs=metrics.report_scale, train_maes=metrics.maes, train_pred_std=metrics.pred_std,
            epoch_task_loss=metrics.task_loss, epoch_label_counts=metrics.label_counts,
            task_cols=task_cols, reg_task_idx=data.reg_task_idx, mtl_loss=state.mtl_loss,
            device=device, dist_state=dist_state, bs=int(cfg.finetune.get("batch_size", 64)),
            lr_now=state.optimizer.param_groups[state.lr_index["encoder"]]["lr"],
            epoch_time=metrics.seconds, dataset_name=dataset_name, run_tag=run_tag, wlog=wlog)


def train_multitask(cfg, train_ds, valid_ds, test_ds, task_cols: List[str],
                    encoder_factory: Callable[[], nn.Module], F_in: int, collate_fn,
                    device, use_wandb, *, dataset_name: str, run_tag: str,
                    seed: int, log_mask: LogMask = None,
                    encoder_config: Optional[dict] = None, trial=None,
                    task_kinds: Optional[List[str]] = None,
                    return_test_preds: bool = False,
                    return_val_preds: bool = False) -> Dict:
    """Train one run; validation selects checkpoints and test metrics only report them.

    The caller merges validation into training for train_on_val and owns the W&B run.
    Existing dataset, encoder_factory, collate_fn, and prediction-return contracts apply.
    """
    ftc = cfg.finetune
    no_select = bool(ftc.get("train_on_val", False)) or not bool(ftc.get("select_best", True))
    mode = str(ftc.get("conformer_eval_mode", "avg_error")).lower()
    if mode not in ("avg_error", "ensemble"):
        raise ValueError(f"finetune.conformer_eval_mode must be 'avg_error' or 'ensemble', got {mode!r}")
    epochs = int(ftc.get("epochs", 100))
    if epochs <= 0:
        raise ValueError("Training requires at least one epoch")
    torch.manual_seed(seed)
    dist_state = ddp.init_distributed(str(cfg.misc.get("device", "cuda")))
    device = dist_state.device if dist_state.enabled else device
    use_wandb = bool(use_wandb) and dist_state.is_main
    data = prepare_data(cfg, train_ds, valid_ds, test_ds, task_cols, collate_fn,
                        device=device, dist_state=dist_state, log_mask=log_mask,
                        task_kinds=task_kinds, use_wandb=use_wandb,
                        run_label=f"{dataset_name} {run_tag}")
    state = prepare_model(cfg, data, encoder_factory, F_in, len(task_cols), device=device,
                          dist_state=dist_state, task_kinds=task_kinds,
                          run_label=f"{dataset_name} {run_tag}")
    selection = initialize_selection(ftc)
    primary_name = next(iter(selection))
    ckpt_tag = "_scratch" if bool(ftc.get("train_from_scratch", False)) else ""
    start_epoch = resume_training(cfg, state, data, selection, dataset_name=dataset_name,
                                  run_tag=run_tag, ckpt_tag=ckpt_tag, dist_state=dist_state)
    state.scaler = configure_execution(ftc, state.core, state.ema_model, device)
    assert data.y_mean.device.type == data.y_std.device.type == "cpu", "Target statistics must stay on CPU"
    metadata = model_metadata(cfg, state, data, task_cols, task_kinds, encoder_config, run_tag)
    score = partial(evaluate_multitask, device=device, y_mean=data.y_mean, y_std=data.y_std,
                    log_mask=log_mask, task_kinds=task_kinds,
                    compute_mcc=bool(ftc.get("compute_mcc", True)), conformer_eval_mode=mode,
                    curve_max_trials=int(ftc.get("ensemble_curve_max_trials", 256)),
                    curve_rng=np.random.default_rng(seed))
    if use_wandb:
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    def wlog(payload):
        if use_wandb:
            wandb.log({**payload, "step": state.step})

    epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        metrics = train_epoch(cfg, state, data, epoch=epoch, device=device, dist_state=dist_state,
                              task_cols=task_cols, task_kinds=task_kinds, log_mask=log_mask,
                              dataset_name=dataset_name, run_tag=run_tag, wlog=wlog)
        val_report, val_macro, values = validate_epoch(
            score, state.eval_model, data.val_loader, dist_state=dist_state,
            selection=selection, skip=no_select, trial=trial, epoch=epoch)
        plot_every = int(ftc.get("plot_every", 20))
        if (use_wandb and val_report is not None and plot_every > 0
                and (epoch % plot_every == 0 or epoch == epochs - 1)):
            log_validation_scatter(val_report, task_cols, data.plot_task_idx, epoch, wlog)
        if not no_select:
            update_selection(cfg, selection, values, state, data, score, metadata,
                             epoch=epoch, val_macro=val_macro, dist_state=dist_state,
                             dataset_name=dataset_name, run_tag=run_tag)
        save_last_checkpoint(cfg, state, data, selection, metadata, epoch=epoch,
                             is_main=dist_state.is_main, dataset_name=dataset_name, run_tag=run_tag)
        log_epoch_metrics(cfg, state, data, metrics, val_report, val_macro, selection[primary_name],
                          epoch=epoch, device=device, dist_state=dist_state, task_cols=task_cols,
                          dataset_name=dataset_name, run_tag=run_tag, wlog=wlog)
        if not no_select and min(spec["no_improve"] for spec in selection.values()) >= int(ftc.get("patience", 20)):
            print(f"[{dataset_name}] {run_tag}: early stop @ epoch {epoch}", flush=True)
            break

    best_epoch = epoch if no_select else selection[primary_name]["epoch"]
    if no_select:
        save_final_checkpoint(cfg, state, metadata, epoch=epoch, is_main=dist_state.is_main,
                              dataset_name=dataset_name, run_tag=run_tag)
    best_state = selection[primary_name]["state"]
    if best_state is not None:
        state.eval_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_val, final_test = evaluate_final(
        score, state.eval_model, data.val_loader, data.test_loader, ftc=ftc,
        is_main=dist_state.is_main, use_wandb=use_wandb, dataset_name=dataset_name,
        run_tag=run_tag, return_val_preds=return_val_preds)
    result = build_result(final_val, final_test, is_main=dist_state.is_main, run_tag=run_tag,
                          best_epoch=best_epoch, selection=selection, primary_name=primary_name,
                          dataset_name=dataset_name, ckpt_tag=ckpt_tag, use_wandb=use_wandb,
                          return_test_preds=return_test_preds, return_val_preds=return_val_preds)
    result = ddp.broadcast_obj(result, dist_state)
    ddp.barrier(dist_state)
    return result
