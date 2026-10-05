"""Representation health metrics and VICReg regularization for JEPA training."""
import functools
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def safe_metric(fn):
    """Wrap a diagnostic metric so a numerical failure logs a warning and
    returns NaN instead of crashing the training run.

    These metrics are for monitoring only (no gradients), so it is safe to
    skip a step when the underlying linalg routine fails to converge --
    which typically happens exactly when the representation is collapsing.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except torch._C._LinAlgError as e:
            logger.warning(
                "Metric '%s' skipped: linalg failed to converge (%s)", fn.__name__, e
            )
            return float("nan")
        except Exception as e:  # noqa: BLE001 - never let a metric kill training
            logger.warning(
                "Metric '%s' skipped: unexpected error (%s)", fn.__name__, e
            )
            return float("nan")
    return wrapper


@safe_metric
@torch.no_grad()
def rankme(embeddings: torch.Tensor) -> float:
    """Effective rank via Shannon entropy of SVD singular value distribution.

    Returns a value in [1, min(N, D)].  High = full capacity utilization,
    low = dimensional collapse.  Embeddings are cast to FP32 before SVD
    to avoid numerical issues with half-precision backends.
    """
    Z = embeddings.float()
    sigma = torch.linalg.svdvals(Z)
    p = sigma / sigma.sum()
    p = p[p > 0]
    entropy = -(p * p.log()).sum()
    return entropy.exp().item()


@safe_metric
@torch.no_grad()
def alpha_req(embeddings: torch.Tensor) -> float:
    """Power-law eigenspectrum decay coefficient (alpha-ReQ).

    Fits  lambda_i ~ i^{-alpha}  on a log-log scale via least-squares
    regression.  High alpha -> rapid decay (dimensional collapse risk).
    Very low alpha -> flat spectrum (noise-dominated).
    """
    Z = embeddings.float()
    Z = Z - Z.mean(dim=0)
    n = Z.shape[0]
    cov = (Z.T @ Z) / (n - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = eigvals.flip(0)  # descending order
    eigvals = eigvals[eigvals > 0]
    if len(eigvals) < 2:
        return 0.0
    log_idx = torch.log(
        torch.arange(1, len(eigvals) + 1, dtype=torch.float32, device=eigvals.device)
    )
    log_eig = torch.log(eigvals)
    # Least-squares:  log_eig = -alpha * log_idx + c
    k = len(log_idx)
    sum_x = log_idx.sum()
    sum_y = log_eig.sum()
    sum_xy = (log_idx * log_eig).sum()
    sum_xx = (log_idx ** 2).sum()
    alpha = -(k * sum_xy - sum_x * sum_y) / (k * sum_xx - sum_x ** 2)
    return alpha.item()


# not used 
def vicreg_variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Hinge loss on per-dimension std: penalises dims with std < gamma."""
    std = z.std(dim=0)
    return F.relu(gamma - std).mean()

# not used
def vicreg_covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Off-diagonal covariance penalty: forces decorrelated dimensions."""
    z = z - z.mean(dim=0)
    n = z.shape[0]
    cov = (z.T @ z) / (n - 1)
    d = cov.shape[0]
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    return off_diag / d