"""Fit independent LightGBM models on each task's labelled feature rows.

Reports follow the PyTorch multi-task schema. Regression targets share the log
transforms but are not standardized; prediction std is in native units. best_epoch
summarizes the best boosting iterations across tasks."""

from typing import Dict, List, Optional

import numpy as np
import torch
import wandb

from finetuning.admet.config import config_dict
from finetuning.admet.metrics import (
    per_task_corr,
    per_task_mae,
    per_task_pred_std,
    report_scale_metrics,
)
from finetuning.admet.metrics.core import _pr_auc, _roc_auc, _spearman
from finetuning.admet.training.helpers import _wandb_key, _wandb_metric_summary
from finetuning.admet.training.loss import LogMask, fwd_target_mt, inv_target_mt


def _lgbm_params(cfg) -> Dict:
    """cfg.baseline.lgbm -> LGBMRegressor kwargs, minus the two knobs that are
    fit-time callbacks rather than constructor args."""
    from omegaconf import OmegaConf
    params = config_dict(cfg.baseline.get("lgbm", {}) or {})
    params.pop("early_stopping_rounds", None)
    return params


def _report(preds: np.ndarray, labels: np.ndarray, task_cols: List[str],
            report_scale=None, task_kinds: Optional[List[str]] = None) -> Dict:
    """Build a multi-task metric report from native regression values or binary
    probabilities.

    macro is restricted to regression MAE. Include optional report-scale metrics and
    classification metrics for the specified task_kinds."""
    maes = per_task_mae(preds, labels)
    corrs = per_task_corr(preds, labels)
    stds = per_task_pred_std(preds, labels)
    binary = ([k == "binary" for k in task_kinds] if task_kinds is not None
              else [False] * len(task_cols))
    reg_idx = [i for i, b in enumerate(binary) if not b]
    with np.errstate(invalid="ignore"):
        macro = float(np.nanmean(maes[reg_idx])) if reg_idx else float("nan")
    out = {
        "per_task": {t: float(v) for t, v in zip(task_cols, maes)},
        "per_task_corr": {t: float(v) for t, v in zip(task_cols, corrs)},
        "per_task_pred_std": {t: float(v) for t, v in zip(task_cols, stds)},
        "macro": macro,
        "macro_mae": macro,
    }
    if task_kinds is not None and reg_idx:
        # Include task-specific selection metrics when task kinds are available.
        spearmans = {}
        for i in reg_idx:
            idx = ~np.isnan(labels[:, i])
            spearmans[task_cols[i]] = (_spearman(labels[idx, i], preds[idx, i])
                                       if idx.sum() >= 2 else float("nan"))
        if spearmans:
            with np.errstate(invalid="ignore"):
                out["macro_spearman"] = float(np.nanmean(list(spearmans.values())))
            out["per_task_spearman"] = spearmans

    bin_idx = [i for i, b in enumerate(binary) if b]
    if bin_idx:
        from finetuning.admet.metrics.classification import mcc_from_proba, best_mcc_threshold
        mccs, aucs, praucs, ths = {}, {}, {}, {}
        for i in bin_idx:
            name = task_cols[i]
            yt, yp = labels[:, i], preds[:, i]
            idx = ~np.isnan(yt)
            mccs[name] = mcc_from_proba(yt, yp, 0.5)
            ths[name] = best_mcc_threshold(yt, yp)[0]
            aucs[name] = _roc_auc(yt[idx], yp[idx])
            praucs[name] = _pr_auc(yt[idx], yp[idx])
        with np.errstate(invalid="ignore"):
            out["macro_mcc"] = float(np.nanmean(list(mccs.values()))) if mccs else float("nan")
            out["macro_roc_auc"] = float(np.nanmean(list(aucs.values()))) if aucs else float("nan")
            out["macro_pr_auc"] = float(np.nanmean(list(praucs.values()))) if praucs else float("nan")
        out["per_task_mcc"] = mccs
        out["per_task_roc_auc"] = aucs
        out["per_task_pr_auc"] = praucs
        out["per_task_mcc_threshold"] = ths

    if report_scale is not None:
        out.update(report_scale_metrics(report_scale, preds, labels, task_cols, reg_idx))
    return out


def _ap_feval(y_true, y_pred):
    """Return LightGBM's (name, value, higher_is_better) tuple for average precision."""
    from sklearn.metrics import average_precision_score
    if len(np.unique(y_true)) < 2:
        return "average_precision", 0.0, True
    return "average_precision", float(average_precision_score(y_true, y_pred)), True


def train_multitask_gbm(cfg, train_ds, valid_ds, test_ds, task_cols: List[str], *,
                        dataset_name: str, run_tag: str, seed: int,
                        log_mask: LogMask = None, use_wandb: bool = False,
                        trial=None, task_kinds: Optional[List[str]] = None,
                        classifier_eval_metric: Optional[List[str]] = None,
                        return_test_preds: bool = False, refit_train_ds=None,
                        impute_ds=None) -> Dict:
    """Fit one LightGBM model per task and return run_tag, val, test, and best_epoch.

    Binary tasks use raw 0/1 labels and predict_proba; classifier_eval_metric selects
    ROC-AUC or PR-AUC for early stopping. Regression tasks use the shared target
    transforms. With refit_train_ds, select each task's round count on held-out
    validation, then refit without early stopping on train+validation.

    Optuna trials store per-task validation MAEs but do not report pruning steps.
    return_test_preds adds native-space predictions in test_ds order."""
    import lightgbm as lgb

    n_tasks = len(task_cols)
    early_stopping_rounds = int(cfg.baseline.get("lgbm", {}).get("early_stopping_rounds", 50))
    params = _lgbm_params(cfg)
    params.setdefault("random_state", seed)
    is_binary = ([k == "binary" for k in task_kinds] if task_kinds is not None
                 else [False] * n_tasks)

    X = {"train": train_ds.feature_matrix(), "val": valid_ds.feature_matrix(),
         "test": test_ds.feature_matrix()}
    # Imputation uses features only; labels do not influence it.
    if impute_ds is not None:
        X["impute"] = impute_ds.feature_matrix()
    native_labels = {k: ds.labels.numpy()
                     for k, ds in (("train", train_ds), ("val", valid_ds), ("test", test_ds))}
    # Transform regression targets; leave binary labels unchanged.
    Y = {k: fwd_target_mt(ds.labels, log_mask).numpy()
         for k, ds in (("train", train_ds), ("val", valid_ds), ("test", test_ds))}

    # Kept OUT of X/Y/native_labels: those drive the `preds` loop below, and the
    # refit split is a training set only -- it is never scored or predicted for.
    X_refit = native_refit = Y_refit = None
    if refit_train_ds is not None:
        X_refit = refit_train_ds.feature_matrix()
        native_refit = refit_train_ds.labels.numpy()
        Y_refit = fwd_target_mt(refit_train_ds.labels, log_mask).numpy()

    preds = {k: np.full((X[k].shape[0], n_tasks), np.nan, dtype=np.float64) for k in X}
    best_iters, n_skipped = [], 0
    n_refit = 0

    for t, task in enumerate(task_cols):
        binary_t = is_binary[t]
        y_train_col = native_labels["train"][:, t] if binary_t else Y["train"][:, t]
        tr_rows = ~np.isnan(y_train_col)
        if int(tr_rows.sum()) < 2:
            # Not a crash: a task can legitimately have ~no labels in a small
            # subset/unlucky split. Reported, and its metrics fall out as NaN.
            print(f"[gbm {dataset_name} {run_tag}] task {task!r}: "
                  f"{int(tr_rows.sum())} labeled train rows -- skipped", flush=True)
            n_skipped += 1
            continue
        if binary_t and len(np.unique(y_train_col[tr_rows])) < 2:
            # A classifier cannot fit a single-class TRAIN split -- report NaN
            # rather than let LightGBM raise deep inside .fit().
            print(f"[gbm {dataset_name} {run_tag}] task {task!r}: "
                  f"only one class in {int(tr_rows.sum())} labeled train rows "
                  f"-- skipped", flush=True)
            n_skipped += 1
            continue

        y_val_col = native_labels["val"][:, t] if binary_t else Y["val"][:, t]
        va_rows = ~np.isnan(y_val_col)
        fit_kwargs = {}
        if binary_t:
            model = lgb.LGBMClassifier(**params)
            metric_name = (classifier_eval_metric[t]
                           if classifier_eval_metric is not None else "roc-auc")
            eval_metric = (_ap_feval if str(metric_name).lower() in ("pr-auc", "average_precision")
                          else "auc")
            if int(va_rows.sum()) >= 2 and early_stopping_rounds > 0 and \
                    len(np.unique(y_val_col[va_rows])) >= 2:
                fit_kwargs = {
                    "eval_X": X["val"][va_rows], "eval_y": y_val_col[va_rows],
                    "eval_metric": eval_metric,
                    "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False),
                                  lgb.log_evaluation(0)],
                }
        else:
            model = lgb.LGBMRegressor(**params)
            if int(va_rows.sum()) >= 2 and early_stopping_rounds > 0:
                # Select boosting rounds on held-out validation.
                fit_kwargs = {
                    "eval_X": X["val"][va_rows], "eval_y": y_val_col[va_rows],
                    "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False),
                                  lgb.log_evaluation(0)],
                }
        model.fit(X["train"][tr_rows], y_train_col[tr_rows], **fit_kwargs)
        n_rounds = int(getattr(model, "best_iteration_", 0) or model.n_estimators)
        best_iters.append(n_rounds)

        if X_refit is not None and native_refit is not None and Y_refit is not None:
            # Refit on merged data without evaluation-based stopping.
            y_refit_col = native_refit[:, t] if binary_t else Y_refit[:, t]
            rf_rows = ~np.isnan(y_refit_col)
            can_refit = int(rf_rows.sum()) >= 2 and (
                not binary_t or len(np.unique(y_refit_col[rf_rows])) >= 2)
            if can_refit:
                refit_params = dict(params)
                refit_params["n_estimators"] = max(1, n_rounds)
                model = (lgb.LGBMClassifier(**refit_params) if binary_t
                         else lgb.LGBMRegressor(**refit_params))
                model.fit(X_refit[rf_rows], y_refit_col[rf_rows])
                n_refit += 1
            # Retain the first model if refitting cannot proceed.

        for split in preds:
            if isinstance(model, lgb.LGBMClassifier):
                probabilities = np.asarray(model.predict_proba(X[split]))
                preds[split][:, t] = probabilities[:, 1]
            else:
                preds[split][:, t] = np.asarray(model.predict(X[split]))

    if n_skipped == n_tasks:
        raise ValueError(f"[gbm {dataset_name} {run_tag}] every task was skipped "
                         f"(no task had >=2 labeled train rows)")

    # Invert regression transforms only; binary predictions are probabilities.
    if log_mask is not None and not isinstance(log_mask, bool) and any(is_binary):
        safe_log_mask = log_mask.clone()
        safe_log_mask[torch.tensor(is_binary)] = False
    else:
        safe_log_mask = log_mask
    reports = {}
    native_preds = {}          # populated for val/test; used by return_test_preds below
    for split, ds in (("val", valid_ds), ("test", test_ds)):
        native = inv_target_mt(torch.from_numpy(preds[split]), safe_log_mask).numpy()
        native_preds[split] = native
        reports[split] = _report(native, ds.labels.numpy(), task_cols,
                                 report_scale=getattr(ds, "report_scale", None),
                                 task_kinds=task_kinds)

    best_epoch = int(np.mean(best_iters)) if best_iters else 0
    if refit_train_ds is not None:
        print(f"[gbm {dataset_name} {run_tag}] train_on_val: refit {n_refit}/"
              f"{n_tasks - n_skipped} tasks on train+val at their own pass-1 "
              f"round counts (mean {best_epoch})", flush=True)
    print(f"[gbm {dataset_name} {run_tag}] {n_tasks - n_skipped}/{n_tasks} tasks fit, "
          f"mean best_iteration={best_epoch} | val macro-MAE={reports['val']['macro']:.4f} "
          f"test macro-MAE={reports['test']['macro']:.4f}", flush=True)

    if trial is not None:
        from finetuning.admet.hpo import _record_per_task_val
        _record_per_task_val(trial, reports["val"]["per_task"])

    if use_wandb:
        # No per-epoch curve exists here, so these go to the run SUMMARY rather
        # than as logged timeseries -- the caller owns the run either way.
        summary = {"final_val_macro": reports["val"]["macro"],
                   "final_test_macro": reports["test"]["macro"],
                   "mean_best_iteration": best_epoch, "n_tasks_skipped": n_skipped}
        for t in task_cols:
            st = _wandb_key(t)
            summary[f"val_mae/{st}"] = reports["val"]["per_task"][t]
            summary[f"val_corr/{st}"] = reports["val"]["per_task_corr"][t]
        # Use the shared flattened W&B metric schema.
        summary.update(_wandb_metric_summary(reports["test"], "test"))
        for k, v in summary.items():
            wandb.summary[k] = v

    out = {"run_tag": run_tag, "val": reports["val"], "test": reports["test"],
           "best_epoch": best_epoch}
    if return_test_preds:
        out["test_per_mol_preds"] = native_preds["test"]
    if impute_ds is not None:
        # Same model-space -> native inverse the scored splits get, so the
        # imputations land on the label ruler rather than the model's.
        out["impute_per_mol_preds"] = inv_target_mt(
            torch.from_numpy(preds["impute"]), safe_log_mask).numpy()
    return out
