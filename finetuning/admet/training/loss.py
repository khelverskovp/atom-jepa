"""Shared single-task and masked multi-task losses and target transforms."""

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

LogMask = Union[None, bool, torch.Tensor]


# Target transforms.
def fwd_target(y: torch.Tensor, log_transform: bool) -> torch.Tensor:
    """Transform native targets with log1p when enabled; otherwise return them unchanged."""
    return torch.log1p(y.clamp_min(-1 + 1e-6)) if log_transform else y


def inv_target(y: torch.Tensor, log_transform: bool) -> torch.Tensor:
    """Model space -> native (inverse of fwd_target)."""
    return torch.expm1(y) if log_transform else y


def fit_target_stats(labels: torch.Tensor, standardize: bool,
                     log_transform: bool) -> Tuple[float, float]:
    """Mean/std of the (optionally log-transformed) TRAIN target. (0,1) if not
    standardizing. Stats are computed in the SAME space the loss operates in."""
    if not standardize:
        return 0.0, 1.0
    t = fwd_target(labels, log_transform)
    return float(t.mean()), float(t.std().clamp_min(1e-8))


def regression_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    """Return mean single-task regression loss: huber, mse, or mae."""
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type in ("mae", "l1"):
        return F.l1_loss(pred, target)
    return F.smooth_l1_loss(pred, target)                 # "huber" / "smooth_l1"


# Multi-task loss.
class MTLLoss(nn.Module):
    """Learn task uncertainty weights following Kendall, Gal & Cipolla (2018).

    Sum precision * loss + log_sigma per task. Regression precision is 0.5 *
    exp(-2*log_sigma); binary precision is exp(-2*log_sigma) in paper mode or
    exp(-log_sigma) in legacy mode. Add log_sigma to the optimizer explicitly: it
    belongs to this loss module, not the prediction model."""

    _is_binary: torch.Tensor

    def __init__(self, num_tasks: int, task_kinds: Optional[Sequence[str]] = None,
                classification_precision: str = "legacy"):
        super().__init__()
        assert num_tasks > 1, "MTLLoss needs more than one task"
        if classification_precision not in ("legacy", "paper"):
            raise ValueError(f"MTLLoss: classification_precision must be 'legacy' or "
                             f"'paper', got {classification_precision!r}")
        self.num_tasks = num_tasks
        self.classification_precision = classification_precision
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))
        # Regression precision is 0.5*exp(-2*s); binary uses exp(-2*s) in paper mode or
        # exp(-s) in legacy mode.
        is_binary = ([k == "binary" for k in task_kinds] if task_kinds is not None
                     else [False] * num_tasks)
        assert len(is_binary) == num_tasks, "task_kinds length must match num_tasks"
        self.register_buffer("_is_binary", torch.tensor(is_binary, dtype=torch.bool))

    def get_precisions(self) -> torch.Tensor:
        reg = 0.5 * torch.exp(-2.0 * self.log_sigma)
        cls_exponent = -2.0 if self.classification_precision == "paper" else -1.0
        cls = torch.exp(cls_exponent * self.log_sigma)
        return torch.where(self._is_binary, cls, reg)

    def forward(self, task_losses: torch.Tensor,
               present: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Combine task losses, optionally excluding absent tasks via a [T] boolean mask.

        Exclusion also removes their log_sigma term, preventing weight updates driven
        solely by the regularizer when a batch contains no labels for that task."""
        assert task_losses.numel() == self.num_tasks
        precisions = self.get_precisions()
        terms = precisions * task_losses + self.log_sigma
        if present is not None:
            terms = terms[present]
        return terms.sum()


def masked_per_task_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                         loss_type: str = "huber",
                         task_kinds: Optional[Sequence[str]] = None,
                         pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return [T] losses from [G,T] predictions, targets, and masks.

    Normalize each task by its valid-label count. Targets must be finite even at masked
    positions. Binary tasks use BCE-with-logits on raw 0/1 targets, optionally with per-
    task pos_weight; regression tasks use loss_type."""
    if loss_type == "mse":
        elem = F.mse_loss(pred, target, reduction="none")
    elif loss_type in ("mae", "l1"):
        elem = F.l1_loss(pred, target, reduction="none")
    else:                                                # "huber" / "smooth_l1"
        elem = F.smooth_l1_loss(pred, target, reduction="none")

    if task_kinds is not None:
        binary = torch.tensor([k == "binary" for k in task_kinds],
                              device=pred.device, dtype=torch.bool)
        if any(k == "binary" for k in task_kinds):
            pw = None
            if pos_weight is not None:
                # broadcast per-task pos_weight over the [G,T] grid
                pw = pos_weight.to(pred.device).unsqueeze(0).expand_as(pred)
            bce = F.binary_cross_entropy_with_logits(
                pred, target, reduction="none", pos_weight=pw)
            elem = torch.where(binary.unsqueeze(0), bce, elem)

    elem = elem * mask
    counts = mask.sum(dim=0).clamp_min(1.0)
    return elem.sum(dim=0) / counts                       # [T]


# Masked task losses.
def fwd_target_mt(y: torch.Tensor, log_mask: LogMask) -> torch.Tensor:
    """Apply log1p uniformly for a bool log_mask or selectively for a [T] mask.

    None disables the transform. Exact zero targets remain valid under log1p."""
    if log_mask is None or isinstance(log_mask, bool):
        return fwd_target(y, bool(log_mask) if log_mask is not None else False)
    mask = log_mask.to(dtype=torch.bool, device=y.device)
    return torch.where(mask, fwd_target(y, True), y)


def inv_target_mt(y: torch.Tensor, log_mask: LogMask) -> torch.Tensor:
    """Model space -> native, per task (inverse of fwd_target_mt)."""
    if log_mask is None or isinstance(log_mask, bool):
        return inv_target(y, bool(log_mask) if log_mask is not None else False)
    mask = log_mask.to(dtype=torch.bool, device=y.device)
    return torch.where(mask, inv_target(y, True), y)


def fit_target_stats_mt(labels: torch.Tensor, standardize: bool,
                        log_mask: LogMask = None,
                        task_kinds: Optional[Sequence[str]] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute train-only per-task mean/std after optional log transforms, ignoring NaNs.

    Binary tasks and unstandardized targets use mean 0/std 1. Empty tasks receive finite
    fallback statistics so masked losses do not propagate NaNs."""
    n_tasks = labels.shape[1]
    if not standardize:
        return torch.zeros(n_tasks), torch.ones(n_tasks)
    t = fwd_target_mt(labels, log_mask)                    # NaN-preserving
    valid = ~torch.isnan(t)
    n_valid = valid.sum(dim=0).clamp_min(1)
    # Zero-fill masked targets and normalize by valid counts to prevent NaN propagation.
    safe_t = torch.where(valid, t, torch.zeros_like(t))
    mean = safe_t.sum(dim=0) / n_valid
    var = ((safe_t - mean.unsqueeze(0)) ** 2 * valid).sum(dim=0) / n_valid
    std = var.clamp_min(1e-8).sqrt()
    if task_kinds is not None:
        keep = torch.tensor([k == "binary" for k in task_kinds], dtype=torch.bool)
        mean = torch.where(keep, torch.zeros_like(mean), mean)
        std = torch.where(keep, torch.ones_like(std), std)
    return mean, std


def binary_positive_weights(labels: torch.Tensor, task_kinds, setting: str | float | None = "auto", *,
                            device=None) -> Optional[torch.Tensor]:
    """Weight positive binary labels by the train-only negative/positive count ratio.

    A numeric setting fixes binary weights; other settings disable weighting.
    Missing labels are excluded. Tasks with no positives receive weight 1.
    """
    if task_kinds is None or not any(kind == "binary" for kind in task_kinds):
        return None
    weights = torch.ones(labels.shape[1])
    if setting == "auto":
        for i, kind in enumerate(task_kinds):
            if kind == "binary":
                observed = labels[:, i][~torch.isnan(labels[:, i])]
                positives = float((observed == 1).sum())
                negatives = float((observed == 0).sum())
                weights[i] = negatives / positives if positives > 0 else 1.0
    elif isinstance(setting, (int, float)):
        for i, kind in enumerate(task_kinds):
            if kind == "binary":
                weights[i] = float(setting)
    else:
        return None
    return weights.to(device)
