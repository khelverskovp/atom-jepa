"""Selective BF16 projections with FP32 parameters and outputs."""
from types import MethodType
import torch


def _float_output(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(_float_output(v) for v in value)
    if isinstance(value, list):
        return [_float_output(v) for v in value]
    return value


def _forward(self, *args, **kwargs):
    if not self._mixed_enabled:
        return self._mixed_original_forward(*args, **kwargs)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = self._mixed_original_forward(*args, **kwargs)
    return _float_output(result)


def enable_bf16_projections(model, policy="no_radial"):
    """Keep radial networks in FP32; 'off' restores original forwards."""
    if policy not in ("off", "no_radial"):
        raise ValueError("policy must be off or no_radial")
    if policy != "off" and any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Selective BF16 requires FP32 master parameters")
    selected = []
    for name, module in model.named_modules():
        if not (isinstance(module, torch.nn.Linear) or type(module).__name__ == "SO2Linear"):
            continue
        enabled = policy == "no_radial" and "rad_func" not in name
        if not hasattr(module, "_mixed_original_forward"):
            module._mixed_original_forward = module.forward
            module.forward = MethodType(_forward, module)
        module._mixed_enabled = enabled
        if enabled:
            selected.append(name)
    return selected
