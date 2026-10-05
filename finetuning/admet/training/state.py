"""Model, optimizer, and resume state for multi-task training."""

import copy
import os
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler

from finetuning.admet.models.model import MultiTaskFineTuneModel
from finetuning.admet.training import distributed as ddp
from finetuning.admet.training.helpers import (
    cosine_factor,
    encoder_warmup_factor,
    set_encoder_trainable,
)
from finetuning.admet.training.loss import MTLLoss
from finetuning.admet.training.regularization import (
    L2SP,
    build_adamw,
    build_l2sp,
    build_param_groups,
)


@dataclass
class TrainingState:
    core: MultiTaskFineTuneModel
    model: nn.Module
    ema_model: MultiTaskFineTuneModel | None
    mtl_loss: MTLLoss | None
    optimizer: torch.optim.AdamW
    group_base_lrs: list[float]
    lr_index: Dict[str, int]
    l2sp: L2SP | None
    scaler: GradScaler
    step: int = 0
    micro: int = 0

    @property
    def eval_model(self):
        return self.ema_model if self.ema_model is not None else self.core

    @property
    def encoder_params(self):
        return list(self.core.encoder.parameters())

    @property
    def head_params(self):
        return list(self.core.head.parameters())

    @property
    def mtl_params(self):
        return list(self.mtl_loss.parameters()) if self.mtl_loss is not None else []

    @property
    def all_params(self):
        return list(self.core.parameters()) + self.mtl_params


def build_model(cfg, data, encoder_factory, F_in, n_tasks, *, device, dist_state, task_kinds):
    """Build the encoder/readout, regularizer, DDP wrapper, and optional EMA/task loss."""
    ftc = cfg.finetune
    fusion_cfg = ftc.get("fusion", {})
    fusion_mode = str(fusion_cfg.get("mode", "none")) if fusion_cfg.get("enabled", False) else "none"
    pool = str(ftc.get("pool", "mean"))
    if pool not in ("mean", "sum"):
        raise ValueError(f"finetune.pool must be 'mean' or 'sum', got {pool!r}")
    feature_dim = data.feature_dim
    use_higher_order = bool(ftc.get("use_higher_order_invariants", False))
    freeze_encoder_epochs = int(ftc.get("freeze_encoder_epochs", 0))
    n_members = max(1, int(ftc.get("n_members", 1) or 1))
    encoder = encoder_factory()
    core = MultiTaskFineTuneModel(encoder, F_in, n_tasks, pool=pool,
                                  dropout=float(ftc.get('head_dropout', 0.1)), head_hidden=int(ftc.get('head_hidden', 64)),
                                  feature_dim=feature_dim, fusion=fusion_mode,
                                  fusion_hidden=int(fusion_cfg.get('hidden', 64)),
                                  fusion_dropout=float(fusion_cfg.get('dropout', 0.0)),
                                  use_higher_order=use_higher_order,
                                  shared_head=bool(ftc.get('shared_head', False)),
                                  n_members=n_members).to(device)

    if freeze_encoder_epochs > 0:
        set_encoder_trainable(core.encoder, False)

    l2sp_cfg = ftc.get("l2sp", {})
    l2sp = build_l2sp(core, l2sp_cfg, train_from_scratch=bool(ftc.get('train_from_scratch', False)),
                      have_pretrained_weights=not bool(ftc.get('train_from_scratch', False)))

    model = ddp.wrap_ddp(core, dist_state)

    ema_model = None
    if bool(ftc.get('ema', True)):
        ema_model = copy.deepcopy(core).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)

    mtl_loss = (MTLLoss(n_tasks, task_kinds=task_kinds,
                       classification_precision=str(ftc.get('mtl_classification_precision', 'legacy'))).to(device)
               if bool(ftc.get('mtl_loss', True)) else None)
    return core, model, ema_model, mtl_loss, l2sp


def build_optimizer(cfg, core, mtl_loss, feature_dim):
    """Create AdamW groups and their base learning rates."""
    ftc = cfg.finetune
    base_lr = float(ftc.get("lr", 5e-5))
    fusion_lr_mult = ftc.get("fusion", {}).get("lr_mult", None)
    fusion_lr_mult = float(fusion_lr_mult) if fusion_lr_mult is not None else None
    param_groups, group_base_lrs, lr_index = build_param_groups(
        core, mtl_loss, base_lr=base_lr, head_lr_mult=float(ftc.get('head_lr_mult', 10.0)), mtl_lr=float(ftc.get('mtl_lr', base_lr)),
        enc_wd=float(ftc.get('encoder_weight_decay', 0.0)),
        trunk_wd=float(ftc.get('trunk_weight_decay', 0.1)),
        mtl_wd=float(ftc.get('mtl_weight_decay', 0.0)),
        no_decay_bias_norm=bool(ftc.get('no_decay_bias_norm', False)))

    if fusion_lr_mult is not None and feature_dim > 0:
        fusion_submodule = (core.head.feature_norm or core.head.feature_proj
                            or core.head.film_mlp or core.head.late_mlp)
        if fusion_submodule is not None:
            fusion_param_ids = {id(p) for p in fusion_submodule.parameters()}
            for g in param_groups:
                g["params"] = [p for p in g["params"] if id(p) not in fusion_param_ids]
            fusion_lr = base_lr * float(ftc.get('head_lr_mult', 10.0)) * fusion_lr_mult
            param_groups.append({"params": list(fusion_submodule.parameters()),
                                 "lr": fusion_lr, "weight_decay": float(ftc.get('trunk_weight_decay', 0.1))})
            lr_index["fusion"] = len(param_groups) - 1
            group_base_lrs.append(fusion_lr)

    optimizer = build_adamw(param_groups, lr=base_lr)
    return optimizer, group_base_lrs, lr_index


def prepare_model(cfg, data, encoder_factory, F_in, n_tasks, *, device, dist_state,
                  task_kinds, run_label) -> TrainingState:
    """Initialize training state without taking an optimizer step."""
    core, model, ema_model, mtl_loss, l2sp = build_model(
        cfg, data, encoder_factory, F_in, n_tasks, device=device,
        dist_state=dist_state, task_kinds=task_kinds)
    optimizer, base_lrs, lr_index = build_optimizer(cfg, core, mtl_loss, data.feature_dim)
    scaler = GradScaler(device.type, enabled=False)
    return TrainingState(core, model, ema_model, mtl_loss, optimizer, base_lrs,
                         lr_index, l2sp, scaler)


def apply_learning_rate(cfg, state: TrainingState, epoch: int):
    """Apply the global schedule and the encoder's post-unfreeze warmup."""
    ftc = cfg.finetune
    factor = cosine_factor(epoch, warmup_epochs=int(ftc.get("lr_warmup_epochs", 0)),
                           epochs=int(ftc.get("epochs", 100)),
                           min_lr=float(ftc.get("min_lr", 1e-7)),
                           base_lr=float(ftc.get("lr", 5e-5)))
    encoder_factor = encoder_warmup_factor(
        epoch, freeze_epochs=int(ftc.get("freeze_encoder_epochs", 0)),
        warmup_epochs=int(ftc.get("encoder_warmup_epochs", 0)))
    first = state.lr_index.get("encoder", 0)
    last = state.lr_index.get("head", first)
    for i, (group, base_lr) in enumerate(zip(state.optimizer.param_groups, state.group_base_lrs)):
        group["lr"] = base_lr * factor * (encoder_factor if first <= i < last else 1.0)


def resume_training(cfg, state, data, sel, *, dataset_name, run_tag, ckpt_tag, dist_state):
    """Restore model/optimizer state and CPU normalization statistics."""
    ftc = cfg.finetune
    is_main = dist_state.is_main
    freeze_encoder_epochs = int(ftc.get("freeze_encoder_epochs", 0))
    epochs = int(ftc.get("epochs", 100))
    start_epoch = 0
    do_resume = bool(ftc.get("resume", False))
    resume_path = os.path.join(cfg.misc.checkpoint_dir,
                               f"last_{dataset_name}_{run_tag}{ckpt_tag}.pt")
    if do_resume and not os.path.exists(resume_path):
        raise SystemExit(
            f"ERROR: finetune.resume=true but {resume_path} does not exist. Refusing "
            f"to silently restart from epoch 0 -- pass finetune.resume=false to start "
            f"a fresh run, or point misc.checkpoint_dir at the run you meant.")
    if do_resume:
        # Load on CPU to preserve CPU statistics and reduce GPU memory use.
        rs = torch.load(resume_path, map_location="cpu", weights_only=False)
        state.core.load_state_dict(rs["model"])
        if state.ema_model is not None and rs.get("ema_model") is not None:
            state.ema_model.load_state_dict(rs["ema_model"])
        state.optimizer.load_state_dict(rs["optimizer"])
        if state.mtl_loss is not None and rs.get("mtl_loss") is not None:
            state.mtl_loss.load_state_dict(rs["mtl_loss"])
        start_epoch = int(rs["epoch"]) + 1
        # Keep target statistics on CPU, as in a fresh run.
        data.y_mean = torch.as_tensor(rs.get("y_mean", data.y_mean)).cpu()
        data.y_std = torch.as_tensor(rs.get("y_std", data.y_std)).cpu()
        # Resume restores the best metric, but older checkpoints may lack the
        # corresponding best weights.
        if "" in sel:
            sel[""]["best"] = float(rs.get("best_val", sel[""]["best"]))
            sel[""]["epoch"] = int(rs.get("best_epoch", -1))
            sel[""]["no_improve"] = int(rs.get("epochs_no_improve", 0))
        set_encoder_trainable(state.core.encoder, start_epoch >= freeze_encoder_epochs)
        if is_main:
            print(f"[{dataset_name} {run_tag}] RESUMED from {resume_path} @ epoch "
                  f"{start_epoch}/{epochs} (encoder trainable: "
                  f"{start_epoch >= freeze_encoder_epochs})", flush=True)
        if start_epoch >= epochs:
            print(f"[{dataset_name} {run_tag}] already at epoch {start_epoch} of "
                  f"{epochs}; nothing left to train.", flush=True)
    return start_epoch

