"""
MatBench readout heads.

"""

from typing import List, Optional

import torch
import torch.nn as nn

from finetuning.qm9.readouts import _StandardizingHead, _num_graphs, _segment_sum

from finetuning.admet.models.model import per_degree_invariant_norms


_ACTIVATIONS = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": lambda: nn.LeakyReLU(0.2),
}


def _activation(name: str):
    act_cls = _ACTIVATIONS.get(name.lower())
    if act_cls is None:
        raise ValueError(f"unknown activation {name!r}; "
                         f"expected one of {sorted(_ACTIVATIONS)}")
    return act_cls


def _make_mlp(width: int, num_layers: int, out_dim: int,
              activation: str = "silu") -> nn.Sequential:
    """`num_layers` Linear layers: (num_layers - 1) of width->width, then
    width->out_dim. Activation after every layer but the last.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    act_cls = _activation(activation)

    dims = [width] * num_layers + [out_dim]
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=(dims[i + 1] != 1)))
        if i < len(dims) - 2:
            layers.append(act_cls())
    return nn.Sequential(*layers)


def _pool(x: torch.Tensor, idx: torch.Tensor, num_graphs: int,
          reduce: str) -> torch.Tensor:
    """Reduce per-atom values [N, D] to per-structure values [G, D].
    """
    if reduce == "sum":
        return _segment_sum(x, idx, num_graphs)
    if reduce == "mean":
        total = _segment_sum(x, idx, num_graphs)
        counts = _segment_sum(torch.ones_like(x[:, :1]), idx, num_graphs).clamp_min(1.0)
        return total / counts
    if reduce == "max":
        # in fp32: scatter_reduce's amax is not reliable under autocast
        xf = x.float()
        out = xf.new_zeros((num_graphs, xf.shape[-1]))
        out = out.scatter_reduce(0, idx.unsqueeze(-1).expand_as(xf), xf,
                                 reduce="amax", include_self=False)
        return out.to(x.dtype)
    raise ValueError(f"reduce must be 'sum', 'mean' or 'max', got {reduce!r}")


class PooledReadout(_StandardizingHead):
    """Pool over atoms, then trunk -> head MLP.
    """

    def __init__(self, f_in: int, out_dim: int = 1, reduce: str = "mean",
                 head_hidden: int = 64, activation: str = "silu",
                 dropout: float = 0.0, lmax: int = 0,
                 use_higher_order: bool = False):
        super().__init__()
        if reduce not in {"mean", "sum", "max"}:
            raise ValueError(f"reduce must be 'mean', 'sum' or 'max', got {reduce!r}")
        act_cls = _activation(activation)
        self.reduce = reduce
        self.out_dim = out_dim

        self.use_higher_order = bool(use_higher_order) and lmax > 0
        self.lmax = lmax if self.use_higher_order else 0
        ho_dim = self.lmax * f_in
        self.ho_norm = nn.LayerNorm(ho_dim) if self.use_higher_order else None
        pooled_dim = f_in + ho_dim

        self.trunk = nn.Sequential(
            nn.Linear(pooled_dim, f_in), act_cls(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(f_in, head_hidden), act_cls(),
            nn.Linear(head_hidden, out_dim))

    def forward(self, node_scalar, batch, node_full=None):
        idx = batch["node_graph_index"]
        g = _num_graphs(batch, idx)

        pooled = _pool(node_scalar, idx, g, self.reduce)          # [G, f_in]
        if self.use_higher_order:
            if node_full is None:
                raise ValueError("PooledReadout(use_higher_order=True) needs "
                                 "node_full from encoder.encode_nodes_full")
            ho = per_degree_invariant_norms(node_full, self.lmax)  # [N, lmax*f_in]
            ho_pooled = _pool(ho, idx, g, self.reduce)             # [G, lmax*f_in]
            pooled = torch.cat([pooled, self.ho_norm(ho_pooled)], dim=-1)

        x = self.head(self.trunk(pooled))                          # [G, out_dim]
        return x.squeeze(-1) if self.out_dim == 1 else x


class AtomwiseReadout(_StandardizingHead):
    """Per-atom MLP, then pool over atoms.

    out_dim=1 for regression; the result is squeezed to [G].
    out_dim=k for k-way classification, returning raw logits [G, k].

    reduce:
      "mean" 
      "max"   
      "sum"   

    dropout, if set, is applied to the atom features before the MLP.
    """

    def __init__(self, f_in: int, out_dim: int = 1, reduce: str = "mean",
                 num_layers: int = 5, activation: str = "silu",
                 dropout: float = 0.0):
        super().__init__()
        if reduce not in {"mean", "sum", "max"}:
            raise ValueError(f"reduce must be 'mean', 'sum' or 'max', got {reduce!r}")
        self.reduce = reduce
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.mlp = _make_mlp(f_in, num_layers, out_dim, activation=activation)

    def forward(self, node_scalar, batch, node_full=None):
        idx = batch["node_graph_index"]
        g = _num_graphs(batch, idx)

        x = node_scalar
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.mlp(x)                            # [N, out_dim]
        x = _pool(x, idx, g, self.reduce)          # [G, out_dim]
        return x.squeeze(-1) if self.out_dim == 1 else x


READOUTS = ("pooled", "atomwise")


class MatBenchModel(nn.Module):
    """Pretrained EquiformerV3 encoder + a readout head.

    Returns [G] for regression, [G, out_dim] for classification.
    """

    def __init__(self, encoder, f_in: int, out_dim: int = 1, reduce: str = "mean",
                 num_layers: int = 5, activation: str = "silu",
                 dropout: float = 0.0, readout: str = "pooled",
                 head_hidden: int = 64, use_higher_order: bool = False):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
        self.encoder = encoder
        self.amp_dtype = None
        self.readout = readout

        if readout == "pooled":
            lmax = int(getattr(getattr(encoder, "cfg", None), "lmax", 0))
            self.head = PooledReadout(
                f_in, out_dim=out_dim, reduce=reduce, head_hidden=head_hidden,
                activation=activation, dropout=dropout, lmax=lmax,
                use_higher_order=use_higher_order,
            )
        else:
            self.head = AtomwiseReadout(
                f_in, out_dim=out_dim, reduce=reduce, num_layers=num_layers,
                activation=activation, dropout=dropout,
            )
        self.use_higher_order = bool(getattr(self.head, "use_higher_order", False))

    def forward(self, batch):
        device_type = batch["node_coordinates"].device.type
        with torch.autocast(device_type=device_type, dtype=self.amp_dtype,
                            enabled=self.amp_dtype is not None):
            if self.use_higher_order:
                node_full, _ = self.encoder.encode_nodes_full(batch)
                node_scalar = node_full[:, 0, :]
            else:
                node_scalar, _ = self.encoder.encode_nodes(batch)
                node_full = None
        with torch.autocast(device_type=device_type, enabled=False):
            return self.head(node_scalar.float(), batch,
                             node_full=node_full.float() if node_full is not None else None)
