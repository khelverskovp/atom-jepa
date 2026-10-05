"""Enable the tested cuEquivariance contractions on individual Equiformer encoders."""

import torch


def enable_cue(body, *, device, encoder_eval_mode):
    """Install derived operators without changing learned weights or checkpoint keys."""
    if getattr(body, "admet_cue_enabled", False):
        return
    from atom_jepa.models.equiformer_v3.activation import SeparableGateS2Activation_SwiGLU_Merge
    from atom_jepa.models.equiformer_v3.transformer_block import GatedSwiGLUGridMLP
    from atom_jepa.models.equiformer_v3.so2_ops import SO2MLinear

    rotation = body.so3_rotation
    if rotation.lmax != 2 or rotation.mmax != 2:
        raise ValueError("cuEquivariance requires lmax=mmax=2; set cuequivariance=false")
    for block in body.blocks:
        ga, ffn = block.ga, block.ffn
        if not ga.use_rad_l_parametrization or ga.use_add_merge:
            raise ValueError("cuEquivariance requires degree-wise radial weights and concatenated messages")
        if type(ga.act) is not SeparableGateS2Activation_SwiGLU_Merge or type(ffn.grid_mlp) is not GatedSwiGLUGridMLP:
            raise ValueError("cuEquivariance requires the tested gated SwiGLU grid activations")
        if not encoder_eval_mode and (ffn.grid_mlp.dropout > 0 or not isinstance(ga.act.grid_drop, torch.nn.Identity)):
            raise ValueError("cuEquivariance requires inactive grid dropout")
    try:
        from atom_jepa.models.equiformer_v3.admet_cue_ops import (
            CompactGatherScaleRotate, WeightedRotateReduce, GatedGridProduct,
            InitialRotateReduce, derived_operator,
        )
    except ImportError as exc:
        raise ImportError(
            "Install the CUDA-matching kernels (pip install \"atom-jepa[cu12]\" or [cu13]) "
            "or set cuequivariance=false."
        ) from exc

    # Reuse only operators with identical geometry and widths within this encoder.
    products = {}
    for block in body.blocks:
        ga, ffn = block.ga, block.ffn
        ga.fused_gather_rotate = derived_operator(CompactGatherScaleRotate(rotation, ga.num_in_channels, device=device))
        ga.fused_return = derived_operator(WeightedRotateReduce(rotation, ga.num_heads, ga.attn_value_channels, device=device))
        ga.rad_func.use_expand = False
        ga.node_embeddings = True
        for layer, grid, channels in (
            (ga.act, ga.act.so3_grid, ga.num_hidden_channels // 2),
            (ffn.grid_mlp, ffn.so3_grid, ffn.grid_mlp.num_hidden_channels),
        ):
            key = (channels, grid.to_grid_mat.shape, grid.from_grid_mat.shape)
            cached = products.get(key)
            if cached is not None and torch.equal(grid.to_grid_mat, cached[0]) and torch.equal(grid.from_grid_mat, cached[1]):
                op = cached[2]
            else:
                op = derived_operator(GatedGridProduct(grid, channels, device=device))
                products[key] = (grid.to_grid_mat, grid.from_grid_mat, op)
            layer.gated_product = op
    initial = body.edge_degree_embedding
    initial.fused_initial = derived_operator(InitialRotateReduce(initial.num_channels, initial.rescale_factor, device=device))
    for module in body.modules():
        if isinstance(module, SO2MLinear):
            module.packed_gemm = True
    body.admet_cue_enabled = True
    print("[atom-jepa] Enabled cuEquivariance grid/rotation fusions and packed SO2 GEMMs", flush=True)
