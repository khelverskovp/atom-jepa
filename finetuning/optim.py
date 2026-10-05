"""Batched optimizer and EMA updates shared by fine-tuning workflows."""

import torch


def build_adamw(param_groups, *, lr, **kwargs):
    """Use fused CUDA AdamW, retaining the execution policy when resuming."""
    params = [p for group in param_groups for p in group["params"]]
    fused = True if params and all(p.is_cuda for p in params) else None
    optimizer = torch.optim.AdamW(param_groups, lr=lr, fused=fused, **kwargs)

    def restore_execution_policy(optimizer, state_dict):
        # Set this before loading: AdamW uses it to place step counters correctly.
        return {**state_dict, "param_groups": [
            {**group, "fused": fused, "foreach": None}
            for group in state_dict["param_groups"]
        ]}

    optimizer.register_load_state_dict_pre_hook(restore_execution_policy)
    return optimizer


@torch.no_grad()
def ema_update(ema_model, model, decay):
    targets = list(ema_model.parameters())
    if targets:
        torch._foreach_lerp_(targets, list(model.parameters()), 1.0 - decay)  # pyright: ignore[reportPrivateImportUsage]
    mutable_names = getattr(ema_model, "_ema_mutable_buffer_names", None)
    if mutable_names is None:
        for be, bm in zip(ema_model.buffers(), model.buffers()):
            be.copy_(bm)
    else:
        for name in mutable_names:
            ema_model.get_buffer(name).copy_(model.get_buffer(name))


