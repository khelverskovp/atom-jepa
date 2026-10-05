"""Interval-aware regression metrics.

ST-RAE is a reconstruction of the CYP challenge description, not an official
implementation; retain it for compatibility with interval-labelled datasets."""

from typing import List, Optional, Sequence, Tuple

import numpy as np


def soft_threshold_error(y_pred: np.ndarray, conf_low: np.ndarray,
                         conf_high: np.ndarray) -> np.ndarray:
    """Per-point soft-threshold error: 0 inside [conf_low, conf_high],
    otherwise the distance to the nearer bound. This is what "explicitly
    accounts for ground-truth uncertainty" means -- a prediction anywhere
    inside a measurement's credible interval is treated as exactly correct,
    so a wide (badly determined) measurement cannot punish a model for
    disagreeing with its point estimate."""
    y_pred = np.asarray(y_pred, float)
    lo = np.asarray(conf_low, float)
    hi = np.asarray(conf_high, float)
    below = np.clip(lo - y_pred, 0.0, None)      # >0 only when pred < lo
    above = np.clip(y_pred - hi, 0.0, None)      # >0 only when pred > hi
    return below + above


def soft_threshold_rae(y_true: np.ndarray, y_pred: np.ndarray,
                       conf_low: np.ndarray, conf_high: np.ndarray) -> float:
    """ST-RAE for ONE task. NaNs in y_true are dropped (unmeasured compounds).

    Where a bound is missing the interval collapses to the point estimate, so
    the term degrades gracefully to a plain absolute error rather than being
    silently skipped -- a missing interval must not make a compound free.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    lo = np.asarray(conf_low, float)
    hi = np.asarray(conf_high, float)

    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if m.sum() == 0:
        return float("nan")
    yt, yp = y_true[m], y_pred[m]
    lo, hi = lo[m], hi[m]
    lo = np.where(np.isnan(lo), yt, lo)
    hi = np.where(np.isnan(hi), yt, hi)

    denom = np.sum(np.abs(yt - np.mean(yt)))
    if denom == 0:
        return float("nan")
    return float(np.sum(soft_threshold_error(yp, lo, hi)) / denom)


def ma_st_rae(y_true: np.ndarray, y_pred: np.ndarray, conf_low: np.ndarray,
              conf_high: np.ndarray, task_idx: Optional[Sequence[int]] = None
              ) -> Tuple[float, List[float]]:
    """Macro-averaged ST-RAE over the selected task columns of [n_mol, T]
    arrays. Returns (macro, per_task). `task_idx` restricts to the direct
    -inhibition columns; the binary TDI columns have no intervals and must
    not be included. Tasks with no measurements contribute NaN and are
    excluded from the macro via nanmean, matching evaluate_multitask's
    convention for empty tasks."""
    T = y_true.shape[1]
    idx = list(range(T)) if task_idx is None else list(task_idx)
    per_task = [soft_threshold_rae(y_true[:, t], y_pred[:, t],
                                   conf_low[:, t], conf_high[:, t]) for t in idx]
    with np.errstate(invalid="ignore"):
        macro = float(np.nanmean(per_task)) if per_task else float("nan")
    return macro, per_task
