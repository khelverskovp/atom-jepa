"""Shared binary classification metrics."""

from typing import List, Optional, Sequence, Tuple

import numpy as np


def mcc(y_true: np.ndarray, y_pred_label: np.ndarray) -> float:
    """Matthews correlation coefficient on hard 0/1 labels. NaN truths are
    dropped; a single-class subset returns NaN rather than sklearn's 0.0,
    matching this repo's `_roc_auc` convention (finetuning/admet/metrics/core.py) -- 0.0 would
    be indistinguishable from a genuinely uninformative model."""
    y_true = np.asarray(y_true, float)
    y_pred_label = np.asarray(y_pred_label, float)
    m = ~np.isnan(y_true) & ~np.isnan(y_pred_label)
    if m.sum() == 0 or len(np.unique(y_true[m])) < 2:
        return float("nan")
    from sklearn.metrics import matthews_corrcoef
    return float(matthews_corrcoef(y_true[m].astype(int), y_pred_label[m].astype(int)))


def mcc_from_proba(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Compute MCC after thresholding probabilities (default: 0.5)."""
    return mcc(y_true, (np.asarray(y_prob, float) >= threshold).astype(float))


def best_mcc_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                       grid: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Find (threshold, MCC) on validation data; keep the threshold fixed for testing."""
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    m = ~np.isnan(y_true) & ~np.isnan(y_prob)
    if m.sum() == 0 or len(np.unique(y_true[m])) < 2:
        return 0.5, float("nan")
    grid = np.linspace(0.05, 0.95, 91) if grid is None else np.asarray(grid, float)
    scores = [mcc_from_proba(y_true[m], y_prob[m], t) for t in grid]
    scores = np.array([s if np.isfinite(s) else -np.inf for s in scores])
    b = int(np.argmax(scores))
    return float(grid[b]), float(scores[b])
