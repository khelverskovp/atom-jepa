"""Encoder precision and in-place Equiformer execution optimizations."""

import torch
from torch.amp.grad_scaler import GradScaler


def configure_ema_buffers(model, ema_model):
    """Sync fixed Equiformer bases once; keep copying mutable/unknown buffers."""
    if ema_model is None:
        return
    from atom_jepa.models.equiformer_v3.layer_norm import EquivariantMergeLayerNorm
    from atom_jepa.models.equiformer_v3.radial_function import GaussianSmearing, RadialFunction
    from atom_jepa.models.equiformer_v3.so3 import SO3Grid, SO3Linear, SO3Rotation

    immutable = {
        GaussianSmearing: {"offset"},
        RadialFunction: {"expand_index"},
        SO3Rotation: {"wigner_index_to_m_array", "wigner_inv_rescale"},
        SO3Grid: {"to_grid_mat", "from_grid_mat"},
        SO3Linear: {"expand_index"},
        EquivariantMergeLayerNorm: {"expand_index", "balance_degree_weight"},
    }
    derived_prefixes = tuple(name + "." for name, module in model.named_modules()
                             if getattr(module, "_admet_derived_operator", False))
    mutable_names = []
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name.startswith(derived_prefixes):
                continue
            parent, _, local_name = name.rpartition(".")
            module = model.get_submodule(parent)
            ema_module = ema_model.get_submodule(parent)
            if (type(module) is type(ema_module)
                    and local_name in immutable.get(type(module), set())):
                ema_model.get_buffer(name).copy_(buffer)
            else:
                mutable_names.append(name)
    # Names survive device moves and buffer replacement; no duplicate state keys.
    ema_model._ema_mutable_buffer_names = tuple(mutable_names)


def resolve_amp_dtype(ftc, device):
    precision = ftc.get("precision")
    if precision is None:
        precision = ftc.get("amp_dtype", "bfloat16") if ftc.get("amp", False) else "fp32"
    dtypes = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
              "fp16": torch.float16, "float16": torch.float16, "fp32": None}
    if precision not in dtypes:
        raise ValueError("precision must be fp32, bf16, or fp16")
    dtype = dtypes[precision]
    if device.type != "cuda":
        return None
    if dtype == torch.bfloat16:
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("This GPU does not support BF16; use precision=fp32")
    return dtype


def configure_execution(ftc, model, ema_model, device):
    """Configure after EMA copying and resume; preserve parameters and state-dict keys."""
    dtype = resolve_amp_dtype(ftc, device)
    for current in (model, ema_model):
        if current is None:
            continue
        current.amp_dtype = dtype
        body = getattr(current.encoder, "body", None)
        if body is None or not hasattr(body, "set_grid_mlp_optimization"):
            continue
        use_cue = device.type == "cuda" and bool(ftc.get("cuequivariance", False))
        if use_cue:
            if not bool(ftc.get("optimize_grid_mlp", True)):
                raise ValueError("cuEquivariance requires optimize_grid_mlp=true")
            from finetuning.cue import enable_cue
            enable_cue(body, device=device, encoder_eval_mode=bool(ftc.get("encoder_eval_mode", False)))
        body.set_grid_mlp_optimization(bool(ftc.get("optimize_grid_mlp", True)))
        if device.type == "cuda" and bool(ftc.get("compile_blocks", False)):
            if use_cue:
                body.edge_degree_embedding.compile(
                    mode=str(ftc.get("compile_mode", "default")),
                    dynamic=bool(ftc.get("compile_dynamic", True)))
            body.compile_blocks(mode=str(ftc.get("compile_mode", "default")),
                                dynamic=bool(ftc.get("compile_dynamic", True)))
    configure_ema_buffers(model, ema_model)
    return GradScaler(device.type, enabled=dtype == torch.float16)
