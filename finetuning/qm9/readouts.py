"""
Per-target QM9 readout heads for the EquiformerV3 finetuner.

"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- QM9 target -> readout family -------------------------------------------
DIPOLE_TARGETS = {"mu"}
SPATIAL_TARGETS = {"r2"}
# everything else QM9 ("alpha","homo","lumo","gap","zpve","U0","U","H","G","Cv")
# falls through to the atomwise head.


# Standard atomic weights indexed by atomic number Z (Z=0 is a padding slot).
# Values match ase.data.atomic_masses (IUPAC 2016) for every element QM9 uses
# (H, C, N, O, F); kept as a local table so this module needs no `ase` import
# and the registered buffer has a fixed size across environments.
_ATOMIC_MASS = (
    0.0,     1.008,   4.0026,  6.94,    9.0122,  10.81,   12.011,  14.007,
    15.999,  18.998,  20.180,  22.990,  24.305,  26.982,  28.085,  30.974,
    32.06,   35.45,   39.948,  39.098,  40.078,  44.956,  47.867,  50.942,
    51.996,  54.938,  55.845,  58.933,  58.693,  63.546,  65.38,   69.723,
    72.630,  74.922,  78.971,  79.904,  83.798,  85.468,  87.62,   88.906,
    91.224,  92.906,  95.95,   98.0,    101.07,  102.91,  106.42,  107.87,
    112.41,  114.82,  118.71,  121.76,  127.60,  126.90,  131.29,
)

_LOG2 = math.log(2.0)

def shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    """softplus(x) - log(2): smooth and zero-valued at x=0."""
    return F.softplus(x) - _LOG2


class Dense(nn.Linear):
    """Linear layer with an optional activation and Xavier-uniform init """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation=None):
        self.activation = activation
        super().__init__(in_features, out_features, bias=bias)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        y = super().forward(x)
        if self.activation is not None:
            y = self.activation(y)
        return y


class SchnetMLP(nn.Module):
    """schnetpack `MLP` """

    def __init__(self, n_in: int, n_out: int = 1, n_hidden: Optional[int] = None,
                 n_layers: int = 2, activation=shifted_softplus):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if n_hidden is None:
            c = n_in
            widths = []
            for _ in range(n_layers):
                widths.append(c)
                c = max(c // 2, n_out)
            widths.append(n_out)
        else:
            widths = [n_in] + [n_hidden] * (n_layers - 1) + [n_out]

        layers = [
            Dense(widths[i], widths[i + 1], activation=activation)
            for i in range(n_layers - 1)
        ]
        layers.append(Dense(widths[-2], widths[-1], activation=None))
        self.out_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.out_net(x)


class GatedEquivariantBlock(nn.Module):
    """PaiNN gated equivariant block: mixes invariant scalars [N, S]
    with equivariant vectors [N, 3, V], returns (scalars_out, vectors_out).
    """

    def __init__(self, n_sin: int, n_vin: int, n_sout: int, n_vout: int,
                 n_hidden: int, activation=F.silu, sactivation=None):
        super().__init__()
        self.n_sin = n_sin
        self.n_vin = n_vin
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.n_hidden = n_hidden
        self.mix_vectors = Dense(n_vin, 2 * n_vout, activation=None, bias=False)
        self.scalar_net = nn.Sequential(
            Dense(n_sin + n_vout, n_hidden, activation=activation),
            Dense(n_hidden, n_sout + n_vout, activation=None),
        )
        self.sactivation = sactivation

    def forward(self, scalars: torch.Tensor, vectors: torch.Tensor):
        vmix = self.mix_vectors(vectors)                       # [N, 3, 2*n_vout]
        vectors_V, vectors_W = torch.split(vmix, self.n_vout, dim=-1)
        vectors_Vn = torch.norm(vectors_V, dim=-2)             # [N, n_vout] (invariant)

        ctx = torch.cat([scalars, vectors_Vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, x = torch.split(x, [self.n_sout, self.n_vout], dim=-1)
        v_out = x.unsqueeze(-2) * vectors_W                    # [N, 3, n_vout]

        if self.sactivation is not None:
            s_out = self.sactivation(s_out)
        return s_out, v_out


# -----------------------------------------------------------------------------
# batching / irrep helpers (plain torch, no torch_scatter dependency)
# -----------------------------------------------------------------------------
def _segment_sum(src: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Sum rows of `src` [N, ...] into `num_segments` bins given by `index` [N].

    Plain-torch replacement for GotenNet's torch_scatter.scatter(reduce="sum")."""
    out = src.new_zeros((num_segments,) + tuple(src.shape[1:]))
    out.index_add_(0, index, src)
    return out


def _num_graphs(batch, index: torch.Tensor) -> int:
    n = batch.get("num_graphs", None)
    if n is None:
        return int(index.max().item()) + 1 if index.numel() else 0
    return int(n)


def l1_from_packed(node_vec: torch.Tensor) -> torch.Tensor:
    """Reduce EquiformerV3's packed irreps to the l=1 (vector) block.

    Accepts the packed state [N, (lmax+1)**2, C] (sphere dim 4, 9, 16, ...) and
    slices [:, 1:4, :], or passes through an already-sliced [N, 3, C]."""
    if node_vec is None:
        raise RuntimeError(
            "this head needs the encoder's l=1 features but got node_vec=None. "
            "Feed it from EquiformerV3Encoder.encode_nodes_full(batch), or build "
            "the head with use_vector_dipole=False for the charge-only fallback."
        )
    if node_vec.dim() != 3:
        raise RuntimeError(
            f"expected node_vec [N, sphere, C], got shape {tuple(node_vec.shape)}"
        )
    s = node_vec.shape[1]
    if s == 3:
        return node_vec
    if s >= 4:
        return node_vec[:, 1:4, :]
    raise RuntimeError(
        f"node_vec has sphere dim {s}: the encoder was built with lmax=0 and "
        f"carries no l=1 features, so the vector dipole head is unusable."
    )


# -----------------------------------------------------------------------------
# heads
# -----------------------------------------------------------------------------
class _StandardizingHead(nn.Module):
    """Base for heads that produce a native-unit physical quantity and must be
    mapped into the loop's standardized space. Stores (mean, std) as buffers so
    they travel with state_dict / resume.

    `needs_vectors` tells the wrapper model whether to call encode_nodes_full."""

    needs_vectors: bool = False

    def __init__(self):
        super().__init__()
        self.register_buffer("y_mean", torch.tensor(0.0))
        self.register_buffer("y_std", torch.tensor(1.0))

    @torch.no_grad()
    def set_target_stats(self, y_mean: float, y_std: float):
        self.y_mean.fill_(float(y_mean))
        self.y_std.fill_(float(y_std))

    def _standardize(self, native: torch.Tensor) -> torch.Tensor:
        return (native - self.y_mean) / self.y_std

    def forward(self, node_scalar, batch, node_vec=None):  # pragma: no cover
        raise NotImplementedError


class AtomwiseReadout(_StandardizingHead):
    """per-atom SchnetMLP, ScaleShift, then aggregation.

    pool:
      "sum"        plain sum of per-atom contributions   (default)
      "mean"       mean over atoms
      "scaled_sum" sum / sqrt(avg_num_nodes)             
    """

    def __init__(self, f_in: int, pool: str = "sum", avg_num_nodes: float = 18.0,
                 n_layers: int = 2, n_hidden: Optional[int] = None,
                 activation=F.silu):
        super().__init__()
        self.pool = pool
        self.avg_num_nodes = float(avg_num_nodes)
        self.out_net = SchnetMLP(f_in, 1, n_hidden, n_layers, activation)

    def forward(self, node_scalar, batch, node_vec=None):
        idx = batch["node_graph_index"]
        g = _num_graphs(batch, idx)

        yi = self.out_net(node_scalar) * self.y_std          # [N, 1], native units
        summed = _segment_sum(yi, idx, g).squeeze(-1)        # [G]

        if self.pool == "mean":
            counts = _segment_sum(torch.ones_like(yi), idx, g).squeeze(-1)
            pooled = summed / counts.clamp_min(1.0)
        elif self.pool == "scaled_sum":
            pooled = summed / (self.avg_num_nodes ** 0.5)
        else:
            pooled = summed

        native = pooled + self.y_mean
        # exact identity with the scaling above; kept for a uniform contract
        return self._standardize(native)


class _MassCentered(_StandardizingHead):
    """Shared machinery for the position-aware heads: mass-weighted centering
    from `atomic_numbers`"""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "atomic_mass", torch.tensor(_ATOMIC_MASS, dtype=torch.float32)
        )

    def _center_of_mass(self, pos, z, idx, g):
        z = z.clamp(min=0, max=self.atomic_mass.numel() - 1)
        mass = self.atomic_mass[z].unsqueeze(-1)                 # [N, 1]
        sum_mass = _segment_sum(mass, idx, g)                    # [G, 1]
        sum_mass_pos = _segment_sum(mass * pos, idx, g)          # [G, 3]
        com = sum_mass_pos / sum_mass.clamp_min(1e-8)            # [G, 3]
        return pos - com[idx]                                    # centered [N, 3]


class DipoleReadout(_MassCentered):
    """
    Two GatedEquivariantBlocks reduce (l0 [N, C], l1 [N, 3, C]) to a per-atom
    charge l0 [N, 1] and a per-atom dipole vector l1 [N, 3, 1]:

        y  = atomic_dipoles + positions * charges     # [N, 3]
        mu = || sum_atoms y ||                        # [G]
    """

    needs_vectors = True

    def __init__(self, f_in: int, n_hidden: Optional[int] = None,
                 f_vec: Optional[int] = None,
                 center_positions: bool = False, charge_neutral: bool = False):
        super().__init__()
        self.f_in = f_in
        self.f_vec = f_in if f_vec is None else f_vec
        self.center_positions = bool(center_positions)
        self.charge_neutral = bool(charge_neutral)
        n_hidden = f_in if n_hidden is None else n_hidden
        self.equivariant_layers = nn.ModuleList([
            GatedEquivariantBlock(
                n_sin=f_in, n_vin=self.f_vec, n_sout=n_hidden, n_vout=n_hidden,
                n_hidden=n_hidden, activation=F.silu, sactivation=F.silu,
            ),
            GatedEquivariantBlock(
                n_sin=n_hidden, n_vin=n_hidden, n_sout=1, n_vout=1,
                n_hidden=n_hidden, activation=F.silu,
            ),
        ])

    def forward(self, node_scalar, batch, node_vec=None):
        idx = batch["node_graph_index"]
        g = _num_graphs(batch, idx)
        pos = batch["node_coordinates"]

        l0 = node_scalar                                     # [N, C]
        l1 = l1_from_packed(node_vec)                        # [N, 3, C], Cartesian
        if l1.shape[-1] != self.f_vec:
            raise RuntimeError(
                f"node_vec has {l1.shape[-1]} channels but the head was built for "
                f"f_vec={self.f_vec}"
            )
        for eqlayer in self.equivariant_layers:
            l0, l1 = eqlayer(l0, l1)

        atomic_dipoles = l1.squeeze(-1)                      # [N, 3]
        charges = l0                                         # [N, 1]

        if self.charge_neutral:
            sum_q = _segment_sum(charges, idx, g)
            counts = _segment_sum(torch.ones_like(charges), idx, g)
            charges = charges - (sum_q / counts.clamp_min(1.0))[idx]
        if self.center_positions:
            pos = self._center_of_mass(pos, batch["atomic_numbers"], idx, g)

        y = atomic_dipoles + pos * charges                   # [N, 3]
        dipole = _segment_sum(y, idx, g)                     # [G, 3]
        mu = torch.linalg.norm(dipole, dim=1)                # [G]
        return self._standardize(mu)


class ChargeDipoleReadout(_MassCentered):
    """Charge-only dipole fallback (no l=1 features required).
    """

    needs_vectors = False

    def __init__(self, f_in: int, n_layers: int = 2, n_hidden: Optional[int] = None):
        super().__init__()
        self.charge_net = SchnetMLP(f_in, 1, n_hidden, n_layers, shifted_softplus)

    def forward(self, node_scalar, batch, node_vec=None):
        idx = batch["node_graph_index"]
        g = _num_graphs(batch, idx)
        pos = batch["node_coordinates"]
        centered = self._center_of_mass(pos, batch["atomic_numbers"], idx, g)

        charges = self.charge_net(node_scalar)                   # [N, 1]
        sum_q = _segment_sum(charges, idx, g)                    # [G, 1]
        counts = _segment_sum(torch.ones_like(charges), idx, g)  # [G, 1]
        charges = charges - (sum_q / counts.clamp_min(1.0))[idx]

        dipole = _segment_sum(charges * centered, idx, g)        # [G, 3]
        mu = torch.linalg.norm(dipole, dim=1)                    # [G]
        return self._standardize(mu)


class SpatialExtentReadout(_MassCentered):
    """(target `r2`).

        x  = out_net(h)                                   # per-atom scalar
        c  = mass-weighted center of mass
        R2 = sum_i ||r_i - c||^2 * x_i
    """

    def __init__(self, f_in: int, n_layers: int = 2, n_hidden: Optional[int] = None):
        super().__init__()
        self.out_net = SchnetMLP(f_in, 1, n_hidden, n_layers, shifted_softplus)

    def forward(self, node_scalar, batch, node_vec=None):
        idx = batch["node_graph_index"]
        g = _num_graphs(batch, idx)
        pos = batch["node_coordinates"]
        centered = self._center_of_mass(pos, batch["atomic_numbers"], idx, g)

        x = self.out_net(node_scalar)                                # [N, 1]
        r2_atom = (centered ** 2).sum(dim=1, keepdim=True) * x       # [N, 1]
        r2 = _segment_sum(r2_atom, idx, g).squeeze(-1)               # [G]
        return self._standardize(r2)


def build_readout(target: str, f_in: int, pool: str = "sum",
                  avg_num_nodes: float = 18.0, n_layers: int = 2,
                  n_hidden: Optional[int] = None,
                  use_vector_dipole: bool = True,
                  f_vec: Optional[int] = None,
                  dipole_center_positions: bool = False,
                  dipole_charge_neutral: bool = False) -> _StandardizingHead:
    if n_hidden is None:
        n_hidden = f_in
    elif n_hidden == "pyramidal":
        n_hidden = None
    if target in DIPOLE_TARGETS:
        if use_vector_dipole:
            return DipoleReadout(f_in, n_hidden=n_hidden, f_vec=f_vec,
                                 center_positions=dipole_center_positions,
                                 charge_neutral=dipole_charge_neutral)
        return ChargeDipoleReadout(f_in, n_layers=n_layers, n_hidden=n_hidden)
    if target in SPATIAL_TARGETS:
        return SpatialExtentReadout(f_in, n_layers=n_layers, n_hidden=n_hidden)
    return AtomwiseReadout(f_in, pool=pool, avg_num_nodes=avg_num_nodes,
                           n_layers=n_layers, n_hidden=n_hidden)