"""
Supervised multi-task objective for second-stage pretraining (e.g. PCBA bioassays),
following MONROE (github.com/blazejba/monroe):

    context encoder (full molecule) -> per-atom invariants [h_l0, |h_l1|, ..., |h_lmax|]
        -> multi-head gated attention pooling -> Linear(n_degrees * C, K) -> K logits
    loss: masked focal loss per task, combined across tasks by uncertainty weighting

Labels are y [G, K] with NaN for unmeasured (molecule, task) pairs.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as segment_softmax

from atom_jepa.models.jepa_equiformer import irrep_slices

NORM_EPS = 1e-8     # sqrt(sum_m x_m^2 + eps): finite gradient at a zero l>0 feature


class AttentionPooling(nn.Module):
    """Multi-head gated attention pooling (MONROE's MultiHeadAttentionPooling).

    Per head, a gate scores every atom and a softmax over each molecule's atoms
    turns the scores into weights. A value projection is split into heads of
    width C // num_heads, each head takes its weighted sum over atoms, and the
    heads are concatenated back to width C.
    """

    def __init__(self, dim: int, num_heads: int = 8, gate_dropout: float = 0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"pool dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.gate = nn.Linear(dim, num_heads)
        self.gate_dropout = nn.Dropout(gate_dropout) if gate_dropout > 0 else nn.Identity()
        self.value_proj = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.value_proj.weight)

    def forward(self, x, gidx, num_graphs):                      # x: [N, C]
        alpha = segment_softmax(self.gate_dropout(self.gate(x)), gidx, num_nodes=num_graphs, dim=0)
        v = self.value_proj(x).view(-1, self.num_heads, self.head_dim)
        out = x.new_zeros(num_graphs, self.num_heads, self.head_dim)
        out.index_add_(0, gidx, v * alpha.unsqueeze(-1))
        return out.flatten(1)                                    # [G, C]


def invariant_features(node_full, lmax, degrees):
    """Per-atom rotation-invariant features from the encoder's full irreps.

    node_full: [N, (lmax+1)^2, C]. Returns [N, len(degrees) * C]: the l=0 channels as
    they are and, for each l > 0 in degrees, the per-channel norm over its 2l+1
    components. Norms are taken per atom, BEFORE pooling, so oppositely oriented
    vectors on different atoms do not cancel.
    """
    slices = irrep_slices(lmax)
    parts = []
    for l in degrees:
        s, e = slices[l]
        x = node_full[:, s:e, :]
        parts.append(x[:, 0, :] if l == 0 else (x.pow(2).sum(dim=1) + NORM_EPS).sqrt())
    return torch.cat(parts, dim=-1)


class SupervisedHead(nn.Module):
    """Invariant per-atom features -> attention pooling -> one Linear(n_degrees * C, K).

    Each row of the linear weight matrix is one task's logistic-regression head (a
    linear probe, as in MONROE). The l>0 norms are non-negative and on a different
    scale from the l=0 channels, so they get a LayerNorm before pooling.
    """

    def __init__(self, dim: int, num_tasks: int, lmax: int, degrees=(0, 1, 2),
                 pool_heads: int = 8, pool_dropout: float = 0.0):
        super().__init__()
        self.lmax = lmax
        self.degrees = sorted(int(l) for l in degrees)
        if not self.degrees or any(l < 0 or l > lmax for l in self.degrees):
            raise ValueError(f"supervised.degrees must be in 0..{lmax}, got {list(degrees)}")
        n_high = sum(l > 0 for l in self.degrees)
        self.n_l0 = dim if 0 in self.degrees else 0
        self.norm_ln = nn.LayerNorm(n_high * dim) if n_high else None
        in_dim = len(self.degrees) * dim
        self.pool = AttentionPooling(in_dim, pool_heads, pool_dropout)
        self.linear = nn.Linear(in_dim, num_tasks)

    def forward(self, node_full, gidx, num_graphs):
        x = invariant_features(node_full, self.lmax, self.degrees)
        if self.norm_ln is not None:
            x = torch.cat([x[:, :self.n_l0], self.norm_ln(x[:, self.n_l0:])], dim=-1)
        return self.linear(self.pool(x, gidx, num_graphs))            # [G, K] logits


def focal_loss_per_task(logits, y, gamma: float = 2.0, alpha: float = 0.25):
    """Masked binary focal loss (Lin et al., 2017), averaged per task.

    L = -alpha_t (1 - p_t)^gamma log p_t over the labeled entries of each task, with
    alpha_t = alpha for positives and 1 - alpha for negatives (alpha=None disables).
    Returns (loss [K], n_labels [K]); a task without labels in the batch has loss 0.
    """
    mask = ~torch.isnan(y)
    y0 = torch.where(mask, y, torch.zeros_like(y))
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, y0, reduction="none")
    weight = (1.0 - torch.exp(-bce)).pow(gamma)
    if alpha is not None:
        weight = weight * (y0 * alpha + (1.0 - y0) * (1.0 - alpha))
    loss = torch.where(mask, weight * bce, torch.zeros_like(bce))
    n = mask.sum(dim=0)
    return loss.sum(dim=0) / n.clamp(min=1), n


class UncertaintyWeighting(nn.Module):
    """Uncertainty weighting (Kendall et al., 2018), as in MONROE.

    s_t = log sigma_t^2 is learned per task, and each task contributes

        L_t / (2 exp(s_t)) + s_t / 2

    mask_absent=False (MONROE): every task's term is included every step, so a task
    with no labels in the batch (L_t = 0) still has s_t pushed down by its s_t/2 term,
    which raises its weight. mask_absent=True includes only tasks with at least one
    label in the batch, so s_t learns from steps where its task is present.

    reduction="sum" (MONROE) adds the included terms; "mean" averages them. At UW's
    equilibrium each weighted term L_t / (2 exp(s_t)) is ~1/2, so the sum grows with
    the number of tasks in the batch (and drifts upwards while the s_t converge),
    whereas the mean stays O(1) and keeps the loss weight against other objectives
    meaningful. Within the tasks both give the same relative weights.
    """

    def __init__(self, num_tasks: int, init: float = -0.5, mask_absent: bool = True,
                 reduction: str = "mean"):
        super().__init__()
        if reduction not in ("sum", "mean"):
            raise ValueError(f"reduction must be 'sum' or 'mean', got {reduction!r}")
        self.mask_absent = mask_absent
        self.reduction = reduction
        self.log_var = nn.Parameter(torch.full((num_tasks,), float(init)))

    def weights(self):
        return 0.5 * torch.exp(-self.log_var)

    def included(self, present):
        """Tasks whose terms enter the loss this step."""
        return present if self.mask_absent else torch.ones_like(present)

    def reduce(self, terms, present):
        terms = terms[self.included(present)]
        return terms.mean() if self.reduction == "mean" and terms.numel() else terms.sum()

    def forward(self, task_losses, present):
        return self.reduce(task_losses * self.weights() + 0.5 * self.log_var, present)


@torch.no_grad()
def evaluate_supervised(encoder, head, loader, device, gamma, alpha, min_positives=10) -> Dict[str, float]:
    """Validation metrics: mean focal loss and AUROC per task, averaged over tasks.

    AUROC is computed only for tasks with >= min_positives positives and at least
    one negative in the evaluated split (MONROE uses min_positives=10).
    """
    from sklearn.metrics import roc_auc_score

    encoder.eval(); head.eval()
    loss_sum = n_sum = None
    tasks, scores, labels = [], [], []
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        node_full, _ = encoder.encode_nodes_full(batch)
        logits = head(node_full, batch["node_graph_index"], batch["num_graphs"]).float()
        y = batch["y"]
        l, n = focal_loss_per_task(logits, y, gamma, alpha)
        loss_sum = l * n if loss_sum is None else loss_sum + l * n
        n_sum = n if n_sum is None else n_sum + n
        rows, cols = torch.nonzero(~torch.isnan(y), as_tuple=True)
        tasks.append(cols.cpu()); scores.append(logits[rows, cols].cpu()); labels.append(y[rows, cols].cpu())

    if n_sum is None:
        raise ValueError("PCBA validation split is empty")
    tasks = torch.cat(tasks).numpy(); scores = torch.cat(scores).numpy(); labels = torch.cat(labels).numpy()
    order = np.argsort(tasks, kind="stable")
    tasks, scores, labels = tasks[order], scores[order], labels[order]
    bounds = np.searchsorted(tasks, np.arange(len(n_sum) + 1))
    aurocs = []
    for t in range(len(n_sum)):
        yt = labels[bounds[t]:bounds[t + 1]]
        n_pos = int(yt.sum())
        if n_pos >= min_positives and n_pos < len(yt):
            aurocs.append(roc_auc_score(yt, scores[bounds[t]:bounds[t + 1]]))

    present = n_sum > 0
    task_loss = loss_sum[present] / n_sum[present]
    return {
        "val/focal_loss": task_loss.mean().item(),
        "val/auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
        "val/auroc_median": float(np.median(aurocs)) if aurocs else float("nan"),
        "val/auroc_tasks": len(aurocs),
    }
