"""Encoder wrappers and single-task or multi-task readout heads for ADMET."""

import torch
import torch.nn as nn

from atom_jepa.models.jepa_equiformer import irrep_slices, pool_nodes

_HO_NORM_EPS = 1e-8


def per_degree_invariant_norms(node_full: torch.Tensor, lmax: int) -> torch.Tensor:
    """Convert [N, sphere, C] irreps to [N, lmax*C] invariant norms for l=1..lmax.

    For each degree/channel, compute sqrt(sum_m x_m**2 + eps). Take norms per atom
    before graph pooling to avoid cancellation of oppositely aligned components."""
    if lmax < 1:
        raise ValueError(f"per_degree_invariant_norms: lmax must be >= 1, got {lmax}")
    slices = irrep_slices(lmax)[1:]                       # skip l=0
    norms = [(node_full[:, s:e, :].pow(2).sum(dim=1) + _HO_NORM_EPS).sqrt()
            for s, e in slices]                            # each [N, C]
    return torch.cat(norms, dim=-1)                        # [N, lmax*C]


# Readout heads.
class ADMETReadout(nn.Module):
    """Pool [N,C] scalar features to [G,C] and predict one logit or regression value per
    graph. Target transforms are handled externally."""

    def __init__(self, F_in: int, pool: str = "mean", dropout: float = 0.0):
        super().__init__()
        self.pool = pool
        self.mlp = nn.Sequential(
            nn.Linear(F_in, F_in), nn.SiLU(), nn.Dropout(dropout), nn.Linear(F_in, 1)
        )

    def set_target_stats(self, mean, std):
        self._mean, self._std = float(mean), float(std)

    def forward(self, node_scalar, batch):
        pooled = pool_nodes(
            node_scalar, batch["node_graph_index"], batch.get("num_graphs"),
            reduce=self.pool,
        )                                              # [G, C]
        return self.mlp(pooled).squeeze(-1)            # [G]


class FineTuneModel(nn.Module):
    """Pretrained EquiformerV3 encoder (trainable) + pooled readout head."""

    def __init__(self, encoder: nn.Module, F_in: int,
                 pool: str = "mean", dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.amp_dtype: torch.dtype | None = None
        self.head = ADMETReadout(F_in, pool=pool, dropout=dropout)

    def forward(self, batch):
        device_type = batch["atomic_numbers"].device.type
        with torch.autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            node_scalar, _ = getattr(self.encoder, "encode_nodes")(batch)
        with torch.autocast(device_type, enabled=False):
            return self.head(node_scalar.float(), batch)


# Multi-task readout.
FUSION_MODES = ("none", "early_concat", "projected_concat", "film", "late")


class _EnsembleMember(nn.Module):
    """One ensemble member with its own trunk and per-task heads; the encoder is shared."""

    def __init__(self, trunk_in: int, F_in: int, dropout: float, head_hidden: int,
                 n_tasks: int, shared_head: bool):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(trunk_in, F_in), nn.SiLU(), nn.Dropout(dropout))
        if shared_head:
            self.wide_head = nn.Sequential(
                nn.Linear(F_in, head_hidden), nn.SiLU(), nn.Linear(head_hidden, n_tasks))
            self.heads = nn.ModuleList()
        else:
            self.wide_head = None
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Linear(F_in, head_hidden), nn.SiLU(), nn.Linear(head_hidden, 1))
                for _ in range(n_tasks)
            ])

    def forward(self, x):
        h = self.trunk(x)
        if self.wide_head is not None:
            return self.wide_head(h)
        return torch.cat([head(h) for head in self.heads], dim=1)


class MultiTaskReadout(nn.Module):
    """Pool encoder features into a trunk and task heads, with optional feature fusion.

    Fusion modes: early_concat normalizes and concatenates features; projected_concat
    projects them first; film modulates pooled features; late adds task predictions.
    FiLM and late branches start at zero contribution. mol_features_valid gates failed
    feature rows off. fusion="none" constructs no feature branch.

    Optional higher-order invariants extend the pooled input. shared_head selects a
    single wide output head; n_members replicates trunk/heads while sharing the encoder.
    Predictions are averaged across members."""

    def __init__(self, F_in: int, n_tasks: int, pool: str = "mean",
                 dropout: float = 0.0, head_hidden: int = 64,
                 feature_dim: int = 0, fusion: str = "none",
                 fusion_hidden: int = 64, fusion_dropout: float = 0.0,
                 lmax: int = 0, use_higher_order: bool = False,
                 shared_head: bool = False, n_members: int = 1):
        super().__init__()
        if fusion not in FUSION_MODES:
            raise ValueError(f"MultiTaskReadout: fusion must be one of {FUSION_MODES}, "
                             f"got {fusion!r}")
        self.pool = pool
        # A zero feature dimension disables fusion.
        self.fusion = fusion if feature_dim > 0 else "none"

        # Higher-order norms widen pooled features; lmax=0 leaves the width unchanged.
        self.use_higher_order = bool(use_higher_order) and lmax > 0
        self.lmax = lmax if self.use_higher_order else 0
        ho_dim = self.lmax * F_in
        self.ho_norm = nn.LayerNorm(ho_dim) if self.use_higher_order else None
        pooled_dim = F_in + ho_dim

        trunk_in = pooled_dim
        self.feature_norm = None       # early_concat
        self.feature_proj = None       # projected_concat
        self.film_mlp = None           # film
        self.late_mlp = None           # late

        if self.fusion == "early_concat":
            self.feature_norm = nn.LayerNorm(feature_dim)
            trunk_in = pooled_dim + feature_dim
        elif self.fusion == "projected_concat":
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, fusion_hidden), nn.LayerNorm(fusion_hidden),
                nn.SiLU(), nn.Dropout(fusion_dropout),
            )
            trunk_in = pooled_dim + fusion_hidden
        elif self.fusion == "film":
            self.film_mlp = nn.Sequential(
                nn.Linear(feature_dim, fusion_hidden), nn.SiLU(), nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden, 2 * pooled_dim),
            )
            # Zero initialization makes FiLM an identity transform initially.
            output_layer = self.film_mlp[-1]
            assert isinstance(output_layer, nn.Linear)
            assert output_layer.bias is not None
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
        elif self.fusion == "late":
            self.late_mlp = nn.Sequential(
                nn.Linear(feature_dim, fusion_hidden), nn.SiLU(), nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden, n_tasks),
            )
            output_layer = self.late_mlp[-1]
            assert isinstance(output_layer, nn.Linear)
            assert output_layer.bias is not None
            nn.init.zeros_(output_layer.weight)  # identity at init, same reasoning as film
            nn.init.zeros_(output_layer.bias)

        # The trunk outputs F_in; heads can share a bottleneck or remain independent.
        self.shared_head = bool(shared_head)
        # Build only active branches to preserve checkpoint keys and avoid unused DDP
        # parameters.
        self.n_members = max(1, int(n_members or 1))
        # Cache member predictions [M,G,T] for losses and diagnostics.
        self.member_preds = None
        if self.n_members > 1:
            self.trunk = None
            self.wide_head = None
            self.heads = nn.ModuleList()
            self.members = nn.ModuleList([
                _EnsembleMember(trunk_in, F_in, dropout, head_hidden, n_tasks,
                                self.shared_head)
                for _ in range(self.n_members)
            ])
        else:
            self.members = nn.ModuleList()
            self.trunk = nn.Sequential(nn.Linear(trunk_in, F_in), nn.SiLU(),
                                       nn.Dropout(dropout))
            if self.shared_head:
                self.wide_head = nn.Sequential(
                    nn.Linear(F_in, head_hidden), nn.SiLU(), nn.Linear(head_hidden, n_tasks))
                self.heads = nn.ModuleList()      # empty: no params, no state_dict keys
            else:
                self.wide_head = None
                self.heads = nn.ModuleList([
                    nn.Sequential(nn.Linear(F_in, head_hidden), nn.SiLU(),
                                  nn.Linear(head_hidden, 1))
                    for _ in range(n_tasks)
                ])

    def _predict(self, h):
        """[G, F_in] trunk output -> [G, n_tasks]."""
        if self.wide_head is not None:
            return self.wide_head(h)
        return torch.cat([head(h) for head in self.heads], dim=1)

    def forward(self, node_scalar, batch, node_full=None):
        pooled = pool_nodes(
            node_scalar, batch["node_graph_index"], batch.get("num_graphs"),
            reduce=self.pool,
        )                                              # [G, F_in]

        if self.use_higher_order:
            if node_full is None or self.ho_norm is None:
                raise ValueError("Higher-order readout requires full encoder features")
            # Compute per-atom norms before pooling to avoid cancellation.
            ho = per_degree_invariant_norms(node_full, self.lmax)          # [N, lmax*F_in]
            ho_pooled = pool_nodes(ho, batch["node_graph_index"],
                                   batch.get("num_graphs"), reduce=self.pool)  # [G, lmax*F_in]
            pooled = torch.cat([pooled, self.ho_norm(ho_pooled)], dim=-1)   # [G, pooled_dim]

        if self.fusion == "none":
            return self._run_members(pooled, None)   # [G, n_tasks]

        feat = batch["mol_features"]                    # [G, D]
        valid = batch.get("mol_features_valid")
        # Gate failed feature rows with the validity mask.
        valid_col = (valid.unsqueeze(-1) if valid is not None
                    else torch.ones(feat.shape[0], 1, device=feat.device, dtype=feat.dtype))

        if self.fusion == "early_concat":
            assert self.feature_norm is not None
            f = self.feature_norm(feat) * valid_col
            return self._run_members(torch.cat([pooled, f], dim=-1), None)

        if self.fusion == "projected_concat":
            assert self.feature_proj is not None
            f = self.feature_proj(feat) * valid_col
            return self._run_members(torch.cat([pooled, f], dim=-1), None)

        if self.fusion == "film":
            assert self.film_mlp is not None
            gamma_beta = self.film_mlp(feat)             # [G, 2*F_in]
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = torch.tanh(gamma) * valid_col        # bounded: pooled is unnormalized
            beta = beta * valid_col
            return self._run_members(pooled * (1.0 + gamma) + beta, None)

        # Add late-fusion predictions to every member for its individual loss.
        assert self.late_mlp is not None
        return self._run_members(pooled, self.late_mlp(feat) * valid_col)

    def _run_members(self, trunk_input, late_term):
        """[G, trunk_in] -> [G, n_tasks], the ensemble MEAN when n_members > 1.

        Per-member predictions are stashed on self.member_preds ([M, G, n_tasks])
        for the training loop; the return value is always [G, n_tasks] so callers,
        metrics and checkpoints are unchanged. Returning the mean is also the right
        inference rule -- averaging members is what makes this an ensemble rather
        than M separately-reported models."""
        if self.n_members == 1:
            self.member_preds = None
            assert self.trunk is not None
            out = self._predict(self.trunk(trunk_input))
            return out if late_term is None else out + late_term
        preds = torch.stack([m(trunk_input) for m in self.members], dim=0)  # [M, G, T]
        if late_term is not None:
            preds = preds + late_term.unsqueeze(0)
        self.member_preds = preds
        return preds.mean(dim=0)


class MultiTaskFineTuneModel(nn.Module):
    """Wrap an encoder with a multi-task readout. Optional higher-order norms use lmax from
    the encoder config."""

    def __init__(self, encoder: nn.Module, F_in: int, n_tasks: int,
                 pool: str = "mean", dropout: float = 0.0, head_hidden: int = 64,
                 feature_dim: int = 0, fusion: str = "none",
                 fusion_hidden: int = 64, fusion_dropout: float = 0.0,
                 use_higher_order: bool = False, shared_head: bool = False,
                 n_members: int = 1):
        super().__init__()
        self.encoder = encoder
        self.amp_dtype: torch.dtype | None = None
        self.use_higher_order = bool(use_higher_order)
        # Feature encoders may lack encoder config and lmax.
        lmax = int(getattr(getattr(encoder, "cfg", None), "lmax", 0))
        self.head = MultiTaskReadout(F_in, n_tasks, pool=pool, dropout=dropout,
                                     head_hidden=head_hidden, feature_dim=feature_dim,
                                     fusion=fusion, fusion_hidden=fusion_hidden,
                                     fusion_dropout=fusion_dropout,
                                     lmax=lmax, use_higher_order=use_higher_order,
                                     shared_head=shared_head, n_members=n_members)
        # Use the resolved higher-order head setting.
        self.use_higher_order = self.head.use_higher_order

    def forward(self, batch):
        device_type = next(self.parameters()).device.type
        with torch.autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            if self.use_higher_order:
                node_full, _ = getattr(self.encoder, "encode_nodes_full")(batch)
                node_scalar = node_full[:, 0, :]
            else:
                node_scalar, _ = getattr(self.encoder, "encode_nodes")(batch)
                node_full = None
        with torch.autocast(device_type, enabled=False):
            return self.head(node_scalar.float(), batch,
                             node_full=node_full.float() if node_full is not None else None)
