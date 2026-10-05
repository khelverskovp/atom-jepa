"""AdamW parameter groups and decoupled L2-SP anchoring for ADMET training."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from finetuning.optim import build_adamw as build_adamw


def _split_decay_nodecay(params: List[nn.Parameter]) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Split parameters into decay and no-decay groups; tensors with ndim <= 1 receive no
    decay."""
    decay, no_decay = [], []
    for p in params:
        (no_decay if p.ndim <= 1 else decay).append(p)
    return decay, no_decay


def build_param_groups(core: nn.Module, mtl_loss: Optional[nn.Module], *,
                       base_lr: float, head_lr_mult: float, mtl_lr: float,
                       enc_wd: float, trunk_wd: float, mtl_wd: float,
                       no_decay_bias_norm: bool = False,
                       ) -> Tuple[List[Dict], List[float], Dict[str, int]]:
    """Return (param_groups, group_base_lrs, lr_index) for AdamW.

    Encoder, head/trunk, and optional MTLLoss have independent learning rates and weight
    decay. lr_index identifies each role's first group. no_decay_bias_norm splits
    encoder/head parameters by dimensionality and disables decay for MTLLoss; empty
    groups are omitted."""
    encoder = core.get_submodule("encoder")
    head = core.get_submodule("head")
    encoder_params = list(encoder.parameters())
    head_params = list(head.parameters())
    mtl_params = list(mtl_loss.parameters()) if mtl_loss is not None else []

    param_groups: List[Dict] = []
    lr_index: Dict[str, int] = {}

    lr_index["encoder"] = len(param_groups)
    if not no_decay_bias_norm:
        param_groups.append({"params": encoder_params, "lr": base_lr, "weight_decay": enc_wd})
    else:
        enc_decay, enc_nodecay = _split_decay_nodecay(encoder_params)
        if enc_decay:
            param_groups.append({"params": enc_decay, "lr": base_lr, "weight_decay": enc_wd})
        if enc_nodecay:
            param_groups.append({"params": enc_nodecay, "lr": base_lr, "weight_decay": 0.0})

    head_lr = base_lr * head_lr_mult
    lr_index["head"] = len(param_groups)
    if not no_decay_bias_norm:
        param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": trunk_wd})
    else:
        head_decay, head_nodecay = _split_decay_nodecay(head_params)
        if head_decay:
            param_groups.append({"params": head_decay, "lr": head_lr, "weight_decay": trunk_wd})
        if head_nodecay:
            param_groups.append({"params": head_nodecay, "lr": head_lr, "weight_decay": 0.0})

    if mtl_loss is not None:
        lr_index["mtl"] = len(param_groups)
        # Optionally exclude the task-weight vector from weight decay.
        wd = 0.0 if no_decay_bias_norm else mtl_wd
        param_groups.append({"params": mtl_params, "lr": mtl_lr, "weight_decay": wd})

    group_base_lrs = [g["lr"] for g in param_groups]
    return param_groups, group_base_lrs, lr_index


class L2SP:
    """Anchor encoder parameters to pretrained weights with a decoupled update.

    Apply p -= lr * lambda * (p - theta_0) after optimizer.step(), before EMA. The
    update does not affect gradients or Adam moments. Snapshot fp32 anchors before
    training; identical parameters and anchors require no DDP reduction."""

    def __init__(self, encoder: nn.Module, lam: float):
        self.lam = float(lam)
        # Match anchor parameters by name, skipping missing parameters.
        self.pairs: List[Tuple[nn.Parameter, torch.Tensor]] = []
        n_skipped = 0
        for name, p in encoder.named_parameters():
            anchor = p.detach().clone().to(dtype=torch.float32)
            if anchor.shape != p.shape:
                n_skipped += 1
                continue
            self.pairs.append((p, anchor))
        if n_skipped:
            print(f"[L2SP] WARNING: skipped {n_skipped} encoder parameter(s) with a "
                  f"shape mismatch against their own snapshot -- this should be "
                  f"impossible for a same-encoder anchor and indicates a bug.",
                  flush=True)

    @torch.no_grad()
    def step_(self, lr: float, want_metrics: bool = False) -> Optional[Tuple[float, float]]:
        """Apply the anchor update using the current encoder LR; skip frozen parameters.

        Call after optimizer.step() and before EMA. With want_metrics, return the post-
        update (0.5 * lambda * squared_distance, distance), requiring GPU-to-CPU
        synchronization. Otherwise return None without that synchronization."""
        params, anchors = [], []
        for p, anchor in self.pairs:
            if not p.requires_grad:
                continue
            if p.dtype == torch.float32:
                params.append(p)
                anchors.append(anchor)
            else:
                # Keep the original FP32 subtraction for non-FP32 parameters.
                delta = p.detach().to(torch.float32) - anchor
                p.add_(delta.to(p.dtype), alpha=-lr * self.lam)
        if params:
            torch._foreach_lerp_(params, anchors, lr * self.lam)
        if not want_metrics:
            return None
        if not self.pairs:
            return 0.0, 0.0
        dist_sq = torch.zeros((), device=self.pairs[0][0].device)
        for p, anchor in self.pairs:
            dist_sq = dist_sq + (p.detach().to(torch.float32) - anchor).pow(2).sum()
        dist = float(dist_sq.sqrt().item())
        return float(0.5 * self.lam * dist_sq.item()), dist


def build_l2sp(core: nn.Module, cfg_l2sp, *, train_from_scratch: bool,
              have_pretrained_weights: bool) -> Optional[L2SP]:
    """Build encoder-only L2-SP, or return None when disabled.

    Require pretrained weights and reject train_from_scratch. Randomly initialized heads
    are not anchored; their regularization uses separate weight decay."""
    if not bool(cfg_l2sp.get("enabled", False)):
        return None
    if train_from_scratch or not have_pretrained_weights:
        raise ValueError(
            "finetune.l2sp.enabled=true with train_from_scratch=true (or no "
            "pretrained encoder weights were loaded): the L2-SP anchor would be "
            "a random init, not a pretrained solution, so this combination is "
            "always a configuration mistake. Set l2sp.enabled=false or "
            "train_from_scratch=false."
        )
    lam = float(cfg_l2sp.get("lambda", 0.0))
    return L2SP(core.get_submodule("encoder"), lam)
