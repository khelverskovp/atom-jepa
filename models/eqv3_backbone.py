"""
Headless EquiformerV3 backbone.

This assembles the EquiformerV3 components (taken from
models/equiformer_v3/) into a body with no energy/force/stress heads. 
It turns a molecular graph into a per-atom
equivariant state and exposes the rotation-invariant (l=0) slice for pooling.

Node state layout:
    x : [N, (lmax+1)**2, C]   packed irreps; x[:, 0, :] is the l=0 invariant part

The forward is factored exactly like the official equiformerv3 model so the predictor can
reuse the same body while skipping atom embedding (it is handed an already
embedded node state) and only re-deriving edge geometry for its view:

    embed_edges(...)   -> set Wigner-D for this graph's edges, return (rbf, env)
    embed_nodes(...)   -> initial x = atom-embedding + edge-degree-embedding
    run_blocks(...)    -> residual stream after all transformer blocks (pre final-norm)
    final_norm(x)      -> normed x;  scalar = normed[:, 0, :]

We bypass fairchem's data layer entirely: edges (index, length, unit vector) are
precomputed by the dataset, so `generate_graph` / PBC machinery is never needed.

Edge convention. The dataset emits edge_index [E,2] = (i=center, j=neighbor),
meaning message j -> i, with unit vector u pointing i -> j and length r. We map
this to EquiformerV3's [2,E] convention (row0=source, row1=target, aggregation
onto row1=target) as source=j, target=i, so messages aggregate onto the center i
exactly as before. edge_distance_vec = pos[source]-pos[target] = r * u. Trained
from scratch, the model is SE(3)-equivariant for any consistent sign choice.
"""

import math
import torch
import torch.nn as nn
import torch._functorch.config as functorch_config

from torch.utils.checkpoint import checkpoint

from .equiformer_v3.so3 import SO3Rotation
from .equiformer_v3.radial_function import GaussianSmearing
from .equiformer_v3.envelope import PolynomialEnvelope
from .equiformer_v3.input_block import EdgeDegreeEmbedding
from .equiformer_v3.transformer_block import TransBlockV3
from .equiformer_v3.layer_norm import get_normalization_layer
from .equiformer_v3.edge_rot_mat import init_edge_rot_mat


# Normalization constants for sum-aggregation. 
# The official model uses OC20 statistics; we use a fixed value tuned for small molecules.
# avg_degree rescales the edge-degree embedding; avg_num_nodes is unused here (no energy head).
_DEFAULT_AVG_DEGREE = 16.0


def edges_from_batch(batch):
    """Map a dataset batch dict to EquiformerV3 edge tensors.

    Returns:
        edge_index_v3 : [2, E] long   (row0=source=j, row1=target=center=i)
        edge_distance : [E]   float   (r)
        edge_vec      : [E, 3] float  (pos[source]-pos[target] = r*u)
    """
    edge = batch["edge_index"]                 # [E, 2] = (i_center, j_neighbor)
    r = batch["edge_lengths"]                  # [E]
    u = batch["edge_vectors"]                  # [E, 3], points i -> j
    i = edge[:, 0]
    j = edge[:, 1]
    edge_index_v3 = torch.stack([j, i], dim=0).contiguous()   # [2, E]
    edge_vec = r.unsqueeze(-1) * u                              # [E, 3]
    return edge_index_v3, r, edge_vec


class HeadlessEquiformerV3(nn.Module):
    """EquiformerV3 body (embedding + transformer blocks + final norm), no heads.
    """

    def __init__(
        self,
        num_channels=128,
        num_layers=6,
        lmax=2,
        mmax=2,
        max_radius=6.0,
        num_radial_basis=64,
        max_num_elements=128,
        num_heads=8,
        attn_hidden_channels=64,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=128,
        edge_channels=128,
        norm_type="merge_layer_norm",
        attn_activation="sep-merge_gates2_swiglu",
        ffn_activation="sep-merge_gates2_swiglu",
        attn_grid_resolution_list=(18, 1),
        ffn_grid_resolution_list=(18, 18),
        use_atom_edge_embedding=True,
        use_envelope=True,
        use_grid_mlp=True,
        use_attn_renorm=True,
        use_add_merge=False,
        use_rad_l_parametrization=True,
        softcap=None,
        attn_eps=1e-16,
        alpha_drop=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        drop_path_rate=0.0,
        avg_degree=_DEFAULT_AVG_DEGREE,
        # If False, the body is built without the atom-type / edge-degree input
        # path (predictor mode: it consumes a node state instead of embedding).
        with_embedding=True,
        grad_checkpointing=True
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.lmax = lmax
        self.mmax = mmax
        self.cutoff = max_radius
        self.with_embedding = with_embedding
        self.avg_degree = avg_degree
        self.sphere_dim = (lmax + 1) ** 2
        self.grad_checkpointing = grad_checkpointing
        self._blocks_compiled = False

        # mmax cannot exceed lmax
        assert mmax <= lmax

        # Radial basis + envelope (shared, parameter-free except RBF buffer)
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, num_radial_basis, 2.0)
        edge_input_channels = int(self.distance_expansion.num_output)
        self.edge_channels_list = [edge_input_channels] + [edge_channels] * 2
        self.envelope_func = PolynomialEnvelope(cutoff=self.cutoff, exponent=5) if use_envelope else None

        # Wigner-D rotation helper. direct_prediction=True everywhere here
        # (we never take gradients wrt positions), so no rotation mask.
        self.so3_rotation = SO3Rotation(lmax, mmax, use_rotation_mask=False)

        # Input path (atom embedding + edge-degree embedding). The predictor that
        # consumes a node state still re-injects view geometry through the
        # edge-degree embedding, so we always build it; with_embedding only
        # controls whether the atom-type embedding is added to the node state.
        self.sphere_embedding = nn.Embedding(max_num_elements, num_channels)
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            num_channels=num_channels,
            lmax=lmax,
            mmax=mmax,
            so3_rotation=self.so3_rotation,
            max_num_elements=max_num_elements,
            edge_channels_list=self.edge_channels_list,
            use_atom_edge_embedding=use_atom_edge_embedding,
            rescale_factor=avg_degree,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(TransBlockV3(
                num_in_channels=num_channels,
                attn_hidden_channels=attn_hidden_channels,
                num_heads=num_heads,
                attn_alpha_channels=attn_alpha_channels,
                attn_value_channels=attn_value_channels,
                ffn_hidden_channels=ffn_hidden_channels,
                num_out_channels=num_channels,
                lmax=lmax,
                mmax=mmax,
                so3_rotation=self.so3_rotation,
                attn_grid_resolution_list=list(attn_grid_resolution_list),
                ffn_grid_resolution_list=list(ffn_grid_resolution_list),
                max_num_elements=max_num_elements,
                edge_channels_list=self.edge_channels_list,
                use_atom_edge_embedding=use_atom_edge_embedding,
                attn_activation=attn_activation,
                use_attn_renorm=use_attn_renorm,
                use_add_merge=use_add_merge,
                use_rad_l_parametrization=use_rad_l_parametrization,
                softcap=softcap,
                attn_eps=attn_eps,
                ffn_activation=ffn_activation,
                use_grid_mlp=use_grid_mlp,
                norm_type=norm_type,
                alpha_drop=alpha_drop,
                attn_mask_rate=0.0,
                attn_weights_drop=attn_weights_drop,
                value_drop=value_drop,
                drop_path_rate=drop_path_rate,
                proj_drop=proj_drop,
                ffn_drop=ffn_drop,
            ))

        self.norm = get_normalization_layer(norm_type, lmax=lmax, num_channels=num_channels)
        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    # factored forward pieces
    # ------------------------------------------------------------------ #
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _set_wigner(self, edge_vec):
        """Set Wigner-D matrices on so3_rotation for this graph's edges.

        Handles the empty-edge case (single-atom views) without crashing
        init_edge_rot_mat, by installing empty wigner tensors of correct rank.
        """
        if edge_vec.numel() > 0:
            with torch.autocast(device_type=edge_vec.device.type, enabled=False):
                if edge_vec.dtype in (torch.float16, torch.bfloat16):
                    edge_vec = edge_vec.float()
                rot = init_edge_rot_mat(edge_vec, use_rotation_mask=False)
            self.so3_rotation.set_wigner(rot)
        else:
            # zero edges: install empty wigners so per-edge bmm is a no-op
            dim = self.sphere_dim
            dev, dt = edge_vec.device, edge_vec.dtype
            self.so3_rotation.wigner = torch.zeros(0, dim, dim, device=dev, dtype=dt)
            self.so3_rotation.wigner_inv = torch.zeros(0, dim, dim, device=dev, dtype=dt)

    def embed_edges(self, edge_distance, edge_vec):
        """Set Wigner-D, return (rbf_expanded_distance, envelope_weight)."""
        self._set_wigner(edge_vec)
        with torch.autocast(device_type=edge_distance.device.type, enabled=False):
            if edge_distance.dtype in (torch.float16, torch.bfloat16):
                edge_distance = edge_distance.float()
            env = self.envelope_func(edge_distance) if self.envelope_func is not None else None
            rbf = self.distance_expansion(edge_distance)
        return rbf, env

    def embed_nodes(self, atomic_numbers, rbf, edge_index_v3, env, add_atom_embedding=True):
        """Initial node state x = (atom embedding) + edge-degree embedding."""
        n = atomic_numbers.shape[0]
        x = torch.zeros(n, self.sphere_dim, self.num_channels,
                        device=atomic_numbers.device, dtype=rbf.dtype)
        if add_atom_embedding:
            x[:, 0, :] = self.sphere_embedding(atomic_numbers)
        if edge_index_v3.shape[1] > 0:
            x = x + self._edge_degree(atomic_numbers, rbf, edge_index_v3, env)
        return x

    def _edge_degree(self, atomic_numbers, rbf, edge_index_v3, env):
        if getattr(self, "admet_cue_enabled", False) and self._blocks_compiled:
            with functorch_config.patch(backward_pass_autocast="off"):
                return self.edge_degree_embedding(atomic_numbers, rbf, edge_index_v3, env)
        return self.edge_degree_embedding(atomic_numbers, rbf, edge_index_v3, env)

    def set_grid_mlp_optimization(self, enabled=True):
        for blk in self.blocks:
            blk.ffn.optimize_grid_mlp = bool(enabled)

    def compile_blocks(self, *, mode="default", dynamic=True, backend="inductor",
                       fullgraph=False):
        """Compile in place: state_dict keys and optimizer parameter identity stay intact.

        Geometry is built eagerly and passed as explicit tensors to each block.
        Call after loading weights, configuring the FFN, and creating any EMA copy.
        Compilation is lazy; the first training/evaluation batches pay its cost.
        """
        if not hasattr(functorch_config, "backward_pass_autocast"):
            raise RuntimeError(
                "Block compilation requires PyTorch with backward_pass_autocast support; "
                "upgrade PyTorch or set finetune.compile_blocks=false."
            )
        for blk in self.blocks:
            blk.compile(mode=mode, dynamic=dynamic, backend=backend, fullgraph=fullgraph)
        self._blocks_compiled = True

    def _call_block(self, blk, *args):
        if self._blocks_compiled:
            # AOTAutograd otherwise assumes backward uses forward's autocast.
            # Our backward runs outside AMP. Scope this to the actual (lazy)
            # compile invocation, including activation-checkpoint recomputation.
            with functorch_config.patch(backward_pass_autocast="off"):
                return blk(*args)
        return blk(*args)

    def run_blocks(self, x, atomic_numbers, rbf, edge_index_v3, env, node_graph_index,
                   num_graphs=None):
        ckpt = self.grad_checkpointing and x.requires_grad

        if edge_index_v3.shape[1] == 0:
            for blk in self.blocks:
                if ckpt:
                    x = checkpoint(self._block_ffn_only, blk, x, node_graph_index, num_graphs,
                                use_reentrant=False)
                else:
                    x = self._block_ffn_only(blk, x, node_graph_index, num_graphs)
            return x

        if getattr(self, "admet_cue_enabled", False):
            src_z = tgt_z = atomic_numbers
        else:
            src_z = atomic_numbers[edge_index_v3[0]]
            tgt_z = atomic_numbers[edge_index_v3[1]]

        if ckpt:
            # so3_rotation.wigner is mutable state shared across every encode() call
            # on this backbone. Checkpoint recompute during backward would otherwise
            # read whichever graph's wigner was set LAST (the other view / noisy
            # graph), not this graph's -> bmm size mismatch. Capture this graph's
            # matrices and pass them as explicit checkpoint inputs so they're saved
            # and reinstalled before each recompute.
            wigner = self.so3_rotation.wigner
            wigner_inv = self.so3_rotation.wigner_inv
            for blk in self.blocks:
                x = checkpoint(self._block_with_wigner, blk, wigner, wigner_inv,
                            x, src_z, tgt_z, rbf, edge_index_v3, env,
                            node_graph_index, num_graphs, use_reentrant=False)
            return x

        for blk in self.blocks:
            x = self._call_block(
                blk, x, src_z, tgt_z, rbf, edge_index_v3, env, node_graph_index,
                num_graphs, self.so3_rotation.wigner, self.so3_rotation.wigner_inv,
            )
        return x

    def _block_with_wigner(self, blk, wigner, wigner_inv, x, src_z, tgt_z, rbf,
                        edge_index_v3, env, node_graph_index, num_graphs=None):
        # Reinstall this graph's Wigner-D matrices on both the original forward and
        # the checkpointed recompute, so the block always rotates with the matrices
        # that match its own edge set.
        self.so3_rotation.wigner = wigner
        self.so3_rotation.wigner_inv = wigner_inv
        return self._call_block(blk, x, src_z, tgt_z, rbf, edge_index_v3, env,
                                node_graph_index, num_graphs, wigner, wigner_inv)

    @staticmethod
    def _block_ffn_only(blk, x, batch, num_graphs=None):
        """A TransBlockV3 forward with the attention sublayer as identity."""
        x_res = x
        out = blk.norm_2(x)
        out = blk.ffn(out)
        if blk.drop_path is not None:
            out = blk.drop_path(out, batch, num_graphs)
        if blk.proj_drop is not None:
            out = blk.proj_drop(out)
        if blk.ffn_shortcut is not None:
            x_res = blk.ffn_shortcut(x_res)
        return out + x_res

    def final_norm(self, x):
        """Apply final equivariant norm; return (normed_x, invariant_scalar)."""
        x = self.norm(x)
        scalar = x.narrow(1, 0, 1).reshape(x.shape[0], self.num_channels)
        return x, scalar

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    def encode(self, batch, node_state=None, add_atom_embedding=True,
           reinject_edge_degree=True, return_full=False):
        atomic_numbers = batch["atomic_numbers"]
        node_graph_index = batch["node_graph_index"]
        edge_index_v3, edge_distance, edge_vec = edges_from_batch(batch)

        rbf, env = self.embed_edges(edge_distance, edge_vec)

        if node_state is None:
            x = self.embed_nodes(atomic_numbers, rbf, edge_index_v3, env,
                                add_atom_embedding=add_atom_embedding)
        else:
            x = node_state
            if reinject_edge_degree and edge_index_v3.shape[1] > 0:
                x = x + self._edge_degree(atomic_numbers, rbf, edge_index_v3, env)

        x = self.run_blocks(x, atomic_numbers, rbf, edge_index_v3, env,
                            node_graph_index, batch.get("num_graphs"))
        normed_x, node_scalar = self.final_norm(x)
        if return_full:
            # node_scalar == normed_x[:, 0, :];  x is the pre-norm handoff
            return node_scalar, x, normed_x
        return node_scalar, x
