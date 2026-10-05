"""Evaluation and result assembly for multi-task training."""

from typing import Callable, Dict, Optional

import numpy as np
import torch
import wandb

import finetuning.admet.training.distributed as ddp
from finetuning.admet.metrics import (
    format_per_task,
    per_task_mae,
    per_task_pred_std,
    report_scale_metrics,
)
from finetuning.admet.training.helpers import (
    _json_safe,
    _wandb_key,
    _wandb_metric_summary,
)
from finetuning.admet.training.loss import inv_target_mt

Score = Callable[..., Dict]


def validate_epoch(score: Score, model, loader, *, dist_state, selection,
                   skip: bool, trial=None, epoch: int = 0):
    """Evaluate on rank 0 and share selection scores; optionally report/prune a trial."""
    report = score(model, loader) if dist_state.is_main and not skip else None
    macro = report["macro"] if report is not None else float("nan")
    macro = ddp.broadcast_obj(macro, dist_state)
    values = ({name: float(report.get(spec["key"], float("nan")))
               for name, spec in selection.items()} if report is not None else None)
    values = ddp.broadcast_obj(values, dist_state)
    if trial is not None:
        import optuna
        trial.report(macro, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return report, macro, values


def evaluate_test(score: Score, model, loader, *, is_main: bool,
                  final: bool = False) -> Optional[Dict]:
    """Score test data on rank 0; final evaluation also computes conformer curves."""
    if not is_main:
        return None
    return score(model, loader, compute_curve=final)


def evaluate_final(score: Score, model, val_loader, test_loader, *, ftc,
                   is_main: bool, use_wandb: bool, dataset_name: str, run_tag: str,
                   return_val_preds: bool = False):
    """Report final validation and test metrics, preserving their evaluation order."""
    if is_main:
        final_val = score(model, val_loader, compute_curve=True,
                          return_per_graph=return_val_preds)
        final_test = evaluate_test(score, model, test_loader, is_main=True, final=True)
        assert final_test is not None
        _val_tag = ("FINAL val (IN-SAMPLE, refit on train+val)" if bool(ftc.get('train_on_val', False))
                    else "FINAL val (held out, last epoch, no selection):" if not bool(ftc.get('select_best', True))
                    else "FINAL val: ")
        print(f"[{dataset_name}] {run_tag} {_val_tag} {format_per_task(final_val)}", flush=True)
        print(f"[{dataset_name}] {run_tag} FINAL test: {format_per_task(final_test)}", flush=True)
        for split_name, rep in (("val", final_val), ("test", final_test)):
            macro_curve = rep.get("curve", {}).get("macro")
            if macro_curve:
                curve_str = " ".join(f"k={k}:{v['mean']:.4f}" for k, v in sorted(macro_curve.items()))
                print(f"[{dataset_name}] {run_tag} {split_name} ensemble-size curve (macro): {curve_str}",
                      flush=True)
        if use_wandb:
            wandb.summary.update(_wandb_metric_summary(final_test, "test"))
            test_macro_curve = final_test.get("curve", {}).get("macro")
            if test_macro_curve:
                for k, v in test_macro_curve.items():
                    wandb.summary[f"test_curve_k{k}"] = v["mean"]
    else:
        final_val = final_test = None

    return final_val, final_test


def build_result(final_val, final_test, *, is_main: bool, run_tag: str,
                 best_epoch: int, selection, primary_name: str,
                 dataset_name: str, ckpt_tag: str, use_wandb: bool,
                 return_test_preds: bool, return_val_preds: bool):
    """Assemble the rank-0 report and optional raw prediction arrays."""
    sel = selection
    result = ({"run_tag": run_tag, "val": _json_safe(final_val), "test": _json_safe(final_test),
               "best_epoch": best_epoch} if is_main else None)
    if result is not None and final_test is not None and return_test_preds:
        result["test_per_mol_preds"] = final_test["per_mol_preds"]
    if result is not None and final_val is not None and return_val_preds:
        result["val_per_graph_preds"] = final_val["per_graph_preds"]
        result["val_mol_index"] = final_val["mol_index"]
        result["val_per_mol_preds"] = final_val["per_mol_preds"]
        result["val_labels"] = final_val["labels"]
    if result is not None and len(sel) > 1:
        result["selections"] = {
            name: {"key": sp["key"], "mode": sp["mode"],
                   "value": (float(sp["best"]) if np.isfinite(sp["best"]) else None),
                   "epoch": sp["epoch"],
                   "checkpoint": f"finetuned_{dataset_name}_{run_tag}"
                                 f"{('_' + name) if name else ''}{ckpt_tag}.pt",
                   "test": _json_safe(sp["test_report"]) if sp["test_report"] else None}
            for name, sp in sel.items()}
        if use_wandb:
            for name, sp in sel.items():
                if name and name != primary_name and sp["test_report"]:
                    wandb.summary.update(
                        _wandb_metric_summary(sp["test_report"], f"test_{name}"))
    return result


def summarize_training_predictions(predictions, masks, labels, *, y_mean, y_std,
                                   log_mask, task_cols, reg_task_idx, report_scale):
    """Score cached rank-local training predictions in native target units."""
    train_preds_full = torch.cat(predictions).cpu().numpy()                    # [n_items,T]
    train_masks_full = torch.cat(masks).cpu().numpy().astype(bool)
    masked_train_preds = np.where(train_masks_full, train_preds_full, np.nan)
    train_pred_std = per_task_pred_std(masked_train_preds, masked_train_preds)  # [T]

    # Train metrics use changing, non-EMA weights and sampled conformers; under DDP
    # they are rank-local.
    train_native = inv_target_mt(
        torch.from_numpy(train_preds_full) * y_std + y_mean, log_mask).numpy()   # [n_items,T]
    train_labels_full = torch.cat(labels).cpu().numpy()               # [n_items,T]
    masked_train_labels = np.where(train_masks_full, train_labels_full, np.nan)
    train_maes = per_task_mae(train_native, masked_train_labels)            # [T] native
    with np.errstate(invalid="ignore"):
        train_macro = (float(np.nanmean(train_maes[reg_task_idx]))
                       if reg_task_idx else float("nan"))
    train_rs = (report_scale_metrics(report_scale, train_native, masked_train_labels,
                                     task_cols, reg_task_idx)
                if report_scale is not None and reg_task_idx else {})

    return train_pred_std, train_maes, train_macro, train_rs


def log_validation_scatter(val_report, task_cols, task_indices, epoch: int, log):
    """Log configured validation scatter plots using cached predictions."""
    preds_arr, labels_arr = val_report["per_mol_preds"], val_report["labels"]
    scatter_log = {}
    for i in task_indices:
        idx = ~np.isnan(labels_arr[:, i])
        if idx.sum() < 2:
            continue
        tbl = wandb.Table(columns=["actual", "predicted"],
                          data=list(zip(labels_arr[idx, i].tolist(), preds_arr[idx, i].tolist())))
        scatter_log[f"scatter/{_wandb_key(task_cols[i])}"] = wandb.plot.scatter(
            tbl, "actual", "predicted", title=f"{task_cols[i]} (val, epoch {epoch})")
    if scatter_log:
        log(scatter_log)


def log_training_epoch(ftc, *, epoch, mean_loss, val_report, val_macro, selection,
                       train_macro, train_rs, train_maes, train_pred_std,
                       epoch_task_loss, epoch_label_counts, task_cols, reg_task_idx,
                       mtl_loss, device, dist_state, bs, lr_now, epoch_time,
                       dataset_name, run_tag, wlog):
    """Print epoch metrics and log task-level training/validation summaries on rank 0."""
    n_tasks = len(task_cols)
    train_macro_s = f"train_macro {train_macro:.4f}" + "".join(
        f" ({k[len('macro_mae_'):]} {v:.4f})" for k, v in sorted(train_rs.items())
        if k.startswith("macro_mae_"))
    val_s = (f"val {format_per_task(val_report)} (best {selection['best']:.4f} @ {selection['epoch']}) "
             if val_report is not None else "val skipped (no selection) ")
    print(f"== [{dataset_name} {run_tag}] ep {epoch} loss {mean_loss:.4f} "
          f"{train_macro_s} "
          f"{val_s}"
          f"lr {lr_now:.2e} ({epoch_time:.1f}s/epoch)", flush=True)

    epoch_log = {"epoch_loss": mean_loss, "val_macro": val_macro, "best_val": selection['best'],
                "lr": lr_now, "sec_per_epoch": epoch_time,
                "train_macro": train_macro}
    for k, v in train_rs.items():
        if k.startswith("macro_"):
            epoch_log[f"train_{k}"] = float(v)
    for k, v in (val_report or {}).items():
        if k.startswith("macro_mae_") and isinstance(v, float):
            epoch_log[f"val_{k}"] = float(v)
    # Test metrics follow validation-selected checkpoints and never drive
    # selection.
    if selection["test_report"] is not None:
        epoch_log["test_macro_at_best"] = float(selection["test_report"]["macro"])
        for k, v in selection["test_report"].items():
            if k.startswith("macro_mae_") and isinstance(v, float):
                epoch_log[f"test_{k}_at_best"] = float(v)
    if device.type == "cuda":
        epoch_log["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    if dist_state.enabled:
        epoch_log["world_size"] = dist_state.world_size
        epoch_log["effective_batch_size"] = bs * dist_state.world_size
    log_sigma_np = precision_np = mtl_weighted_np = np.zeros(n_tasks)
    if mtl_loss is not None:
        log_sigma_np = mtl_loss.log_sigma.detach().cpu().numpy()
        precision_np = mtl_loss.get_precisions().detach().cpu().numpy()
        mtl_weighted_np = precision_np * epoch_task_loss
    for i, t in enumerate(task_cols) if bool(ftc.get('per_task_logging', True)) else []:
        st = _wandb_key(t)
        epoch_log[f"train_loss/{st}"] = float(epoch_task_loss[i])
        epoch_log[f"n_labels/{st}"] = float(epoch_label_counts[i].item())
        epoch_log[f"train_pred_std/{st}"] = float(train_pred_std[i])
        if i in reg_task_idx:
            epoch_log[f"train_mae/{st}"] = float(train_maes[i])
            for k, v in train_rs.items():
                if k.startswith("per_task_mae_") and t in v:
                    epoch_log[f"train_{k[len('per_task_'):]}/{st}"] = float(v[t])
        if val_report is not None:
            for k, v in val_report.items():
                if k.startswith("per_task_mae_") and t in v:
                    epoch_log[f"val_{k[len('per_task_'):]}/{st}"] = float(v[t])
            epoch_log[f"val_mae/{st}"] = val_report["per_task"].get(t, float("nan"))
            epoch_log[f"val_corr/{st}"] = val_report["per_task_corr"].get(t, float("nan"))
            epoch_log[f"val_pred_std/{st}"] = val_report["per_task_pred_std"].get(t, float("nan"))
        if mtl_loss is not None:
            epoch_log[f"log_sigma/{st}"] = float(log_sigma_np[i])
            epoch_log[f"precision/{st}"] = float(precision_np[i])
            epoch_log[f"mtl_weighted_loss/{st}"] = float(mtl_weighted_np[i])
    wlog(epoch_log)

