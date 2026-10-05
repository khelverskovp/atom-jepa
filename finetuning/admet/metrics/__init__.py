"""
Shared evaluation metrics and reporting scales.

Re-export the core evaluation API for training and dataset drivers.
Classification, interval-aware regression, and reporting-scale helpers live
in their respective submodules. Run aggregation lives in the reporting package.
"""

from finetuning.admet.metrics.core import (
    TaskSpec, resolve_task,
    evaluate, evaluate_full, evaluate_multitask,
    ensemble_size_curve, predict_dataframe,
    single_conformer_draws, tdc_leaderboard,
    per_task_mae, per_task_corr, per_task_pred_std,
    report_scale_metrics, format_per_task,
    _aggregate_curves, _format_curve,
)

__all__ = [
    "TaskSpec", "resolve_task",
    "evaluate", "evaluate_full", "evaluate_multitask",
    "ensemble_size_curve", "predict_dataframe",
    "single_conformer_draws", "tdc_leaderboard",
    "per_task_mae", "per_task_corr", "per_task_pred_std",
    "report_scale_metrics", "format_per_task",
]
