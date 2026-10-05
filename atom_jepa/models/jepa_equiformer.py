"""
JEPA encoder / predictor built on the headless EquiformerV3 backbone.

"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eqv3_backbone import HeadlessEquiformerV3
from .equiformer_v3.so3 import SO3Rotation
from .equiformer_v3.radial_function import GaussianSmearing
from .equiformer_v3.edge_rot_mat import init_edge_rot_mat


# --------------------------------------------------------------------------- #
# parameter-free pooling
# --------------------------------------------------------------------------- #
def scatter_add(src, index, dim_size):
    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


def pool_nodes(node_scalar, gidx, num_graphs, reduce="mean"):
    """Parameter-free mean/sum over nodes; works for [N, C] and [N, sphere, C]."""
    if num_graphs is None:
        num_graphs = int(gidx.max().item()) + 1 if gidx.numel() > 0 else 0
    pooled = scatter_add(node_scalar, gidx, num_graphs)
    if reduce == "mean":
        ones = node_scalar.new_ones(node_scalar.size(0), 1)
        counts = scatter_add(ones, gidx, num_graphs).clamp_min(1.0)
        counts = counts.view(num_graphs, *([1] * (pooled.dim() - 1)))  # [G,1] or [G,1,1]
        pooled = pooled / counts
    return pooled


def mlp(d_in, d_hidden, d_out, dropout=0.0):
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_hidden, d_out)
    )

def irrep_slices(lmax):
    """[(start, end)] index ranges into the packed sphere dim, one per l = 0..lmax."""
    slices, start = [], 0
    for l in range(lmax + 1):
        n = 2 * l + 1
        slices.append((start, start + n))
        start += n
    return slices


class IrrepProjection(nn.Module):
    """Equivariant per-irrep projection C -> C_out on a packed state [.., sphere, C].

    l=0 (invariant): full MLP or Linear (bias allowed, nonlinearity allowed).
    l>0: one Linear(C, C_out, bias=False) per l, applied identically to all 2l+1
         m-components (the only SO(3)-equivariant linear map). No bias, no nonlinearity.
    """
    def __init__(self, lmax, c_in, c_out, l0_mlp=True, hidden=None, dropout=0.0):
        super().__init__()
        self.slices = irrep_slices(lmax)
        hidden = hidden or c_in
        self.l0 = mlp(c_in, hidden, c_out, dropout) if l0_mlp else nn.Linear(c_in, c_out)
        self.l_high = nn.ModuleList(
            [nn.Linear(c_in, c_out, bias=False) for _ in range(1, lmax + 1)]
        )

    def forward(self, x):                                  # x: [.., sphere, C]
        outs = [self.l0(x[..., 0, :]).unsqueeze(-2)]       # [.., 1, C_out]
        for li, (s, e) in enumerate(self.slices[1:]):
            outs.append(self.l_high[li](x[..., s:e, :]))   # [.., 2l+1, C_out]
        return torch.cat(outs, dim=-2)                      # [.., sphere, C_out]

def invariant_query_features(ctx_full, q, lmax):
    """ctx_full, q: [Q, sphere, C] in the SAME real-SH basis.
    -> invariant [Q, C*(lmax+2)]: l=0 context, l=0 query (pure distance), and
    <ctx_l, q_l> summed over m for each l>=1 (each rotation-invariant since the
    real Wigner-D blocks are orthogonal)."""
    feats = [ctx_full[:, 0, :], q[:, 0, :]]
    for (s, e) in irrep_slices(lmax)[1:]:
        feats.append((ctx_full[:, s:e, :] * q[:, s:e, :]).sum(dim=1))
    return torch.cat(feats, dim=-1)


class CentroidQueryEmbedding(nn.Module):
    """Δ = (context centroid -> target atom) -> equivariant query token
    [Q, sphere, C], built the SAME way EdgeDegreeEmbedding builds its edge token:
    per-degree m=0 coefficients in the edge-aligned frame, lifted to the global
    frame by wigner_inv[:, :, :lmax+1]. This guarantees the token is Y_l(Δ̂) in the
    node irrep basis, so <ctx_l, q_l> is a meaningful invariant.

    Owns a PRIVATE SO3Rotation (mmax=lmax, full fidelity) so it never mutates the
    backbone/predictor-body shared wigner state. No cutoff envelope: a
    centroid->atom vector is routinely longer than the mp cutoff, and an envelope
    would zero the whole token (direction included) for those atoms. query_radius
    only sets the RBF range -- distance resolution degrades gracefully past it,
    direction is always preserved.
    """
    def __init__(self, lmax, num_channels, num_rbf, query_radius):
        super().__init__()
        self.lmax = lmax
        self.rbf = GaussianSmearing(0.0, query_radius, num_rbf, 2.0)
        self.so3 = SO3Rotation(lmax, lmax, use_rotation_mask=False)
        self.radial = nn.ModuleList(
            [nn.Linear(int(self.rbf.num_output), num_channels) for _ in range(lmax + 1)]
        )

    def forward(self, delta):                                   # delta: [Q, 3]
        r = delta.norm(dim=1)
        rbf = self.rbf(r)                                        # [Q, num_rbf]
        x_m0 = torch.stack([lin(rbf) for lin in self.radial], dim=1)  # [Q, lmax+1, C]
        # this query's Wigner-D on the PRIVATE rotation, then edge-frame m=0 ->
        # global full sphere, EXACTLY as EdgeDegreeEmbedding (wigner_inv[:,:,:L+1]).
        self.so3.set_wigner(init_edge_rot_mat(delta, use_rotation_mask=False))
        return torch.bmm(self.so3.wigner_inv.narrow(2, 0, self.lmax + 1), x_m0)  # [Q,sphere,C]

@dataclass
class EquiformerV3Config:
    """Shared backbone hyper-parameters. The encoder uses num_layers at width
    num_channels; the predictor uses pred_num_layers at width pred_num_channels
    (None -> num_channels, i.e. same width as the encoder)."""
    num_channels: int = 128
    pred_num_channels: Optional[int] = None   # predictor body width; None -> num_channels
    num_layers: int = 6
    pred_num_layers: int = 2
    lmax: int = 2
    mmax: int = 2
    max_radius: float = 6.0
    num_radial_basis: int = 64
    max_num_elements: int = 128
    num_heads: int = 8
    attn_hidden_channels: int = 64
    attn_alpha_channels: int = 32
    attn_value_channels: int = 16
    ffn_hidden_channels: int = 128
    edge_channels: int = 128
    norm_type: str = "merge_layer_norm"
    attn_activation: str = "sep-merge_gates2_swiglu"
    ffn_activation: str = "sep-merge_gates2_swiglu"
    attn_grid_resolution_list: List[int] = field(default_factory=lambda: [18, 1])
    ffn_grid_resolution_list: List[int] = field(default_factory=lambda: [18, 18])
    use_envelope: bool = True
    use_grid_mlp: bool = True
    avg_degree: float = 16.0
    readout_pool: str = "mean"          # parameter-free encoder pool: "mean"|"sum"
    dropout: float = 0.0
    drop_path_rate: float = 0.0
    predict_irreps: bool = False
    query_radius: Optional[float] = None   # centroid->atom query RBF range; None -> 2*max_radius
    grad_checkpointing: bool = True
    pred_grad_checkpointing: Optional[bool] = None 

    def __post_init__(self):
        if self.pred_num_channels is None:
            self.pred_num_channels = self.num_channels
        if self.query_radius is None:
            self.query_radius = 2.0 * self.max_radius
        if self.pred_grad_checkpointing is None:
            self.pred_grad_checkpointing = self.grad_checkpointing


def _backbone(cfg: EquiformerV3Config, num_layers: int,
              num_channels: Optional[int] = None, 
              grad_checkpointing: Optional[bool] = None) -> HeadlessEquiformerV3:
    """Build a headless backbone at `num_layers` and (optionally) an overridden
    residual-stream width. `num_channels=None` uses cfg.num_channels (encoder
    width); the predictor passes cfg.pred_num_channels. The per-head attention
    widths and ffn_hidden_channels / edge_channels are taken from cfg unchanged --
    they are independent of the residual-stream width."""
    return HeadlessEquiformerV3(
        num_channels=num_channels if num_channels is not None else cfg.num_channels,
        num_layers=num_layers,
        lmax=cfg.lmax,
        mmax=cfg.mmax,
        max_radius=cfg.max_radius,
        num_radial_basis=cfg.num_radial_basis,
        max_num_elements=cfg.max_num_elements,
        num_heads=cfg.num_heads,
        attn_hidden_channels=cfg.attn_hidden_channels,
        attn_alpha_channels=cfg.attn_alpha_channels,
        attn_value_channels=cfg.attn_value_channels,
        ffn_hidden_channels=cfg.ffn_hidden_channels,
        edge_channels=cfg.edge_channels,
        norm_type=cfg.norm_type,
        attn_activation=cfg.attn_activation,
        ffn_activation=cfg.ffn_activation,
        attn_grid_resolution_list=cfg.attn_grid_resolution_list,
        ffn_grid_resolution_list=cfg.ffn_grid_resolution_list,
        use_envelope=cfg.use_envelope,
        use_grid_mlp=cfg.use_grid_mlp,
        alpha_drop=cfg.dropout,
        ffn_drop=cfg.dropout,
        drop_path_rate=cfg.drop_path_rate,
        avg_degree=cfg.avg_degree,
        grad_checkpointing=(cfg.grad_checkpointing if grad_checkpointing is None
                            else bool(grad_checkpointing)),
    )


# --------------------------------------------------------------------------- #
# encoder (online context encoder; EMA-copied into the target)
# --------------------------------------------------------------------------- #
class EquiformerV3Encoder(nn.Module):
    """Headless EquiformerV3 encoder.

    encode_nodes(batch) -> (node_scalar [N,C], node_state [N,sphere,C])
        node_scalar : post-final-norm l=0 invariant features (pooled into the
                      representation; also the per-atom target for tgt_atom).
        node_state  : residual stream (pre final-norm) -> predictor handoff.
    encode_nodes_full(batch) -> (node_full [N,sphere,C], node_state)
        node_full   : post-final-norm full packed irreps of the LAST block.
    encode_nodes_all_layers(batch) -> (node_scalars [N,num_layers,C], node_state)
        node_scalars: pre-final-norm l=0 scalar slice of EVERY block, depth-ordered.
    pool(node_scalar, gidx, num_graphs) -> [G, C]   (parameter-free)
    forward(batch) -> [G, C]   (pool o encode_nodes); used by EMA target and probe.
    """

    def __init__(self, cfg: EquiformerV3Config):
        super().__init__()
        self.cfg = cfg
        self.reduce = cfg.readout_pool
        self.body = _backbone(cfg, cfg.num_layers)   # encoder width = cfg.num_channels

    def encode_nodes(self, batch):
        node_scalar, node_state = self.body.encode(batch, node_state=None, add_atom_embedding=True)
        return node_scalar, node_state

    def encode_nodes_full(self, batch):
        """Like encode_nodes but returns the full normed irreps [N, sphere, C]."""
        _, node_state, node_full = self.body.encode(
            batch, node_state=None, add_atom_embedding=True, return_full=True
        )
        return node_full, node_state

    def encode_nodes_all_layers(self, batch):
        """l=0 scalar slice of EVERY transformer block's output, depth-ordered:
        [N, num_layers, C]. self.body.blocks already runs num_layers times per
        call (see HeadlessEquiformerV3.run_blocks); this only captures, via a
        forward hook on each block, what each call already produces and
        normally discards -- not a second forward pass.

        NOT the same tensor as encode_nodes's node_scalar at the last layer:
        final_norm is applied exactly ONCE, to run_blocks' final output,
        AFTER the block loop -- so these are the RAW pre-final-norm residual
        stream at each depth, including the last. Consistent treatment across
        every layer (rather than norming only the last) is deliberate: the
        equivariant norm module's learned scale/bias were only ever trained on
        the final block's output distribution, so applying it to intermediate
        layers would distort them under an out-of-distribution norm rather
        than genuinely make them comparable; a plain train-fit scaler
        downstream (e.g. finetuning.admet.baseline.fit_apply_feature_scaler's "standard"
        option) is the more principled way to put every depth on one scale.

        Degenerate-graph caveat: if a batch's edges are entirely empty (e.g. a
        single free atom), run_blocks takes a special path that calls each
        block's submodules directly (_block_ffn_only) rather than the block's
        own forward/__call__, so these hooks would not fire. Not handled here
        -- does not occur for real molecules under the encoder's radius-graph
        cutoff.
        """
        acts: List[torch.Tensor] = []

        def _hook(_module, _inputs, out):
            acts.append(out[:, 0, :])

        handles = [blk.register_forward_hook(_hook) for blk in self.body.blocks]
        try:
            _, node_state = self.encode_nodes(batch)
        finally:
            for h in handles:
                h.remove()
        return torch.stack(acts, dim=1), node_state   # [N, num_layers, C]

    def set_grad_checkpointing(self, enabled: bool):
        """Toggle activation checkpointing after construction (e.g. load a
        pretrained encoder saved with checkpointing on, then finetune without)."""
        self.body.grad_checkpointing = bool(enabled)
        return self
    
    def pool(self, node_scalar, gidx, num_graphs=None):
        return pool_nodes(node_scalar, gidx, num_graphs, reduce=self.reduce)

    def forward(self, batch):
        node_scalar, _ = self.encode_nodes(batch)
        return self.pool(node_scalar, batch["node_graph_index"], batch.get("num_graphs"))


# --------------------------------------------------------------------------- #
# predictor (online-only; learned readout)
# --------------------------------------------------------------------------- #
class EquiformerV3Predictor(nn.Module):
    """Online predictor with a graph head and a target-atom head.

    Each head is constructed only when its objective is active (graph_pred /
    tgt_atom_pred). forward() runs the shared predictor body once and returns a
    dict with "graph" (if return_graph) and/or the raw body output "node_full"
    (if return_node_full), which predict_target_atoms() consumes. With
    return_graph=False the graph pool + readout is never executed.

    Width: the body runs at pred_C = cfg.pred_num_channels. The incoming
    node_state (enc_C) is projected enc_C -> pred_C by an equivariant in_proj
    before the body, and every head maps pred_C -> enc_C so outputs land in the
    EMA target's space. in_proj is None when enc_C == pred_C (identity handoff).
    """

    def __init__(self, cfg: EquiformerV3Config, graph_pred=True, tgt_atom_pred=False,
                 readout_use_mlp=True, readout_use_ln=True):
        super().__init__()
        self.cfg = cfg
        self.reduce = cfg.readout_pool
        self.predict_irreps = bool(getattr(cfg, "predict_irreps", False))

        enc_C = cfg.num_channels          # encoder / EMA-target width = prediction target
        pred_C = cfg.pred_num_channels    # predictor body width
        self.body = _backbone(cfg, cfg.pred_num_layers, num_channels=pred_C,
                      grad_checkpointing=cfg.pred_grad_checkpointing)

        # Equivariant handoff projection enc_C -> pred_C on the packed state
        # [N, sphere, enc_C]. Identity (None) when the widths match. Built
        # regardless of predict_irreps -- the body always runs on full irreps, so
        # the l>0 components must be projected too. l0_mlp=False -> a plain linear
        # bridge (no nonlinearity at the input); set True to give l=0 a small MLP.
        self.in_proj = (IrrepProjection(cfg.lmax, enc_C, pred_C, l0_mlp=False)
                        if enc_C != pred_C else None)

        self.graph_pred = bool(graph_pred)

        if not self.predict_irreps:
            self.readout_use_mlp = readout_use_mlp
            # graph readout modules exist only if the graph objective is active.
            # Body emits pred_C; heads terminate at enc_C (the target space).
            self.graph_mlp = mlp(pred_C, pred_C, enc_C, cfg.dropout) if (self.graph_pred and readout_use_mlp) else None
            self.graph_ln = nn.LayerNorm(pred_C) if (self.graph_pred and readout_use_ln and not readout_use_mlp) else None
            self.graph_out = nn.Linear(pred_C, enc_C) if self.graph_pred else None
        else:
            # full-irrep equivariant readout (readout_use_ln ignored here; an LN
            # over the full sphere dim would not be equivariant). pred_C -> enc_C.
            self.graph_proj = (IrrepProjection(cfg.lmax, pred_C, enc_C, l0_mlp=readout_use_mlp, dropout=cfg.dropout)
                               if self.graph_pred else None)
        self.tgt_atom_pred = bool(tgt_atom_pred)
        if tgt_atom_pred:
            self.query_embed = CentroidQueryEmbedding(
                cfg.lmax, pred_C, cfg.num_radial_basis, cfg.query_radius)
            if not self.predict_irreps:
                # invariant_query_features -> pred_C*(lmax+2); head -> enc_C (target space)
                self.tgt_atom_head = mlp(pred_C * (cfg.lmax + 2), pred_C, enc_C, cfg.dropout)
            else:
                self.query_proj   = IrrepProjection(cfg.lmax, pred_C, pred_C, l0_mlp=False)
                self.tgt_atom_head = IrrepProjection(cfg.lmax, pred_C, enc_C, l0_mlp=True)

    def _readout(self, node_scalar, gidx, num_graphs):
        pooled = pool_nodes(node_scalar, gidx, num_graphs, reduce=self.reduce)
        if self.graph_mlp is not None:
            return self.graph_mlp(pooled)
        if self.graph_ln is not None:
            pooled = self.graph_ln(pooled)
        return self.graph_out(pooled)

    def set_grad_checkpointing(self, enabled: bool):
        """Toggle activation checkpointing after construction (e.g. load a
        pretrained encoder saved with checkpointing on, then finetune without)."""
        self.body.grad_checkpointing = bool(enabled)
        return self

    def forward(self, batch, node_state, return_graph=True, return_node_full=False):
        if not (return_graph or return_node_full):
            raise ValueError("predictor.forward called with no outputs requested")

        # Project the encoder handoff enc_C -> pred_C before the body, since the
        # body's edge-degree re-injection adds a pred_C-wide term to node_state.
        if self.in_proj is not None:
            node_state = self.in_proj(node_state)

        node_scalar, _, node_full = self.body.encode(
            batch, node_state=node_state, add_atom_embedding=False,
            reinject_edge_degree=True, return_full=True,
        )
        gidx, ng = batch["node_graph_index"], batch.get("num_graphs")

        out = {}
        if return_graph:
            if self.predict_irreps:
                pooled = pool_nodes(node_full, gidx, ng, reduce=self.reduce)  # [G, sphere, pred_C]
                out["graph"] = self.graph_proj(pooled)                        # [G, sphere, enc_C]
            else:
                out["graph"] = self._readout(node_scalar, gidx, ng)           # [G, enc_C]
        if return_node_full:
            out["node_full"] = node_full
        return out

    def predict_target_atoms(self, node_full_ctx, gidx_ctx, ng_ctx, delta, tgt_gidx):
        """node_full_ctx: [Nc, sphere, pred_C] predictor-body output on the CONTEXT
        view (from forward(..., return_node_full=True)). delta: [Nt, 3]
        centroid->target-atom. tgt_gidx: [Nt] graph id of each target atom, in the
        SAME graph numbering as the context. Pools context as FULL irreps even
        when the target is l=0, so direction enters via the l>0 contraction."""
        ctx_pool = pool_nodes(node_full_ctx, gidx_ctx, ng_ctx, reduce=self.reduce)  # [G,sphere,pred_C]
        q     = self.query_embed(delta)                                             # [Nt,sphere,pred_C]
        ctx_b = ctx_pool[tgt_gidx]                                                   # [Nt,sphere,pred_C]
        if not self.predict_irreps:
            return self.tgt_atom_head(invariant_query_features(ctx_b, q, self.cfg.lmax))  # [Nt,enc_C]
        return self.tgt_atom_head(ctx_b + self.query_proj(q))                        # [Nt,sphere,enc_C]