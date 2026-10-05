"""Run one multi-task training epoch, including accumulation, AMP, and EMA."""

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DistributedSampler

from finetuning.admet.training import distributed as ddp
from finetuning.admet.training.evaluation import summarize_training_predictions
from finetuning.admet.training.helpers import _per_mol, grad_norm, set_encoder_trainable
from finetuning.admet.training.loss import fwd_target_mt, masked_per_task_loss
from finetuning.admet.training.state import TrainingState, apply_learning_rate
from finetuning.admet.training.utils import ema_update, move_batch


@dataclass
class EpochMetrics:
    mean_loss: float
    task_loss: np.ndarray
    label_counts: torch.Tensor
    pred_std: np.ndarray
    maes: np.ndarray
    macro: float
    report_scale: dict
    seconds: float


def forward_loss(cfg, state, data, batch, *, epoch, device, y_mean_dev, y_std_dev,
                 log_mask, task_kinds):
    """Apply training feature gating and compute masked supervised losses in fp32."""
    ftc = cfg.finetune
    fusion_cfg = ftc.get("fusion", {})
    loss_type = str(ftc.get("loss", "huber")).lower()
    batch = move_batch(batch, device)
    if "mol_features_valid" in batch:
        if epoch < int(fusion_cfg.get('warmup_epochs', 0)):
            batch["mol_features_valid"] = torch.zeros_like(batch["mol_features_valid"])
        elif float(fusion_cfg.get('feature_dropout', 0.0)) > 0:
            keep = (torch.rand_like(batch["mol_features_valid"]) >= float(fusion_cfg.get('feature_dropout', 0.0))).float()
            batch["mol_features_valid"] = batch["mol_features_valid"] * keep
    y = batch["y"]                                  # [G,T], 0.0 under mask=0
    mask = batch["y_mask"]                           # [G,T]
    pred = state.model(batch)
    # Compute targets and losses in fp32.
    pred = pred.float()

    yt = fwd_target_mt(y, log_mask)
    yb = (yt - y_mean_dev) / y_std_dev
    member_preds = getattr(state.core.head, "member_preds", None)
    if member_preds is not None:
        mp = member_preds.float()                                   # [M,G,T]
        task_losses = torch.stack([
            masked_per_task_loss(mp[i], yb, mask, loss_type,
                                 task_kinds=task_kinds,
                                 pos_weight=data.pos_weight)
            for i in range(mp.shape[0])
        ], dim=0).mean(dim=0)                                       # [T]
    else:
        task_losses = masked_per_task_loss(pred, yb, mask, loss_type,
                                           task_kinds=task_kinds,
                                           pos_weight=data.pos_weight)        # [T]
    if state.mtl_loss is not None:
        present = (mask.sum(dim=0) > 0) if bool(ftc.get('mtl_mask_absent_tasks', False)) else None
        loss = state.mtl_loss(task_losses, present=present)
    else:
        present = (mask.sum(dim=0) > 0) if bool(ftc.get('mtl_mask_absent_tasks', False)) else None
        loss = (torch.where(present.any(),
                            (task_losses * present).sum() / present.sum().clamp_min(1),
                            task_losses.mean()) if present is not None else task_losses.mean())
    return batch, pred, y, mask, task_losses, loss


def optimizer_step(cfg, state: TrainingState, *, dist_state, will_log):
    """Reduce gradients, step the optimizer, apply L2-SP, then update EMA."""
    ftc = cfg.finetune
    is_main = dist_state.is_main
    l2sp_log_every = int(ftc.get("l2sp", {}).get("log_every", 500))
    enc_norm = head_norm = mtl_norm = 0.0
    do_l2sp_log = False
    l2sp_penalty = l2sp_dist = 0.0
    if will_log:
        # read BEFORE the real clip (below) touches gradients; max_norm=inf
        # never rescales anything, see grad_norm's docstring.
        enc_norm = grad_norm(state.encoder_params)
        head_norm = grad_norm(state.head_params)
        mtl_norm = grad_norm(state.mtl_params) if state.mtl_params else 0.0

    # Explicitly reduce task-weight gradients outside the DDP wrapper.
    ddp.average_gradients_(state.mtl_params, dist_state)

    # Unscale gradients before clipping.
    if state.scaler.is_enabled():
        state.scaler.unscale_(state.optimizer)
    if float(ftc.get('grad_clip', 10.0)) > 0:
        torch.nn.utils.clip_grad_norm_(
            state.all_params, float(ftc.get('grad_clip', 10.0)),
            error_if_nonfinite=(bool(ftc.get('cuequivariance', False))
                                and not state.scaler.is_enabled()))
    if state.scaler.is_enabled():
        state.scaler.step(state.optimizer)      # skips the step if it found inf/nan
        state.scaler.update()
    else:
        state.optimizer.step()

    # Apply L2-SP after the optimizer and before EMA, using the current LR.
    if state.l2sp is not None:
        do_l2sp_log = will_log and is_main and (state.step % l2sp_log_every == 0)
        lr_enc_now = state.optimizer.param_groups[state.lr_index["encoder"]]["lr"]
        l2sp_metrics = state.l2sp.step_(lr_enc_now, want_metrics=do_l2sp_log)
        if l2sp_metrics is not None:
            l2sp_penalty, l2sp_dist = l2sp_metrics

    if bool(ftc.get('ema', True)):
        ema_update(state.ema_model, state.core, float(ftc.get('ema_decay', 0.99)))
    return enc_norm, head_norm, mtl_norm, do_l2sp_log, l2sp_penalty, l2sp_dist


def log_training_step(state, diagnostics, *, epoch, loss, running, batches_done,
                      n_train_batches, epoch_t0, dataset_name, run_tag, dist_state, wlog):
    """Report optimizer-step progress and gradient/regularization diagnostics."""
    enc_norm, head_norm, mtl_norm, do_l2sp_log, l2sp_penalty, l2sp_dist = diagnostics
    is_main = dist_state.is_main
    # Use role-based LR indices because parameter-group counts can vary.
    lr_enc = state.optimizer.param_groups[state.lr_index["encoder"]]["lr"]
    lr_head = state.optimizer.param_groups[state.lr_index["head"]]["lr"]
    avg_loss = running.item() / max(1, batches_done)
    elapsed = time.time() - epoch_t0
    rate = batches_done / max(elapsed, 1e-9)
    eta_min = (n_train_batches - batches_done) / max(rate, 1e-9) / 60.0
    if is_main:
        print(f"[{dataset_name} {run_tag}] ep {epoch} step {state.step} "
              f"[{batches_done}/{n_train_batches}] loss {loss.item():.4f} "
              f"avg {avg_loss:.4f} lr {lr_enc:.2e} "
              f"grad_norm(enc/head/mtl) {enc_norm:.2f}/{head_norm:.2f}/{mtl_norm:.2f} "
              f"| {rate:.2f} batch/s, epoch ETA {eta_min:.1f} min", flush=True)
    step_log = {"train_loss": loss.item(), "train_loss_avg": avg_loss,
               "lr_encoder": lr_enc, "lr_head": lr_head,
               "epoch": epoch, "grad_norm_encoder": enc_norm,
               "grad_norm_head": head_norm, "batches_per_sec": rate,
               "epoch_progress": batches_done / max(1, n_train_batches)}
    if state.mtl_loss is not None:
        step_log["lr_mtl"] = state.optimizer.param_groups[state.lr_index["mtl"]]["lr"]
        step_log["grad_norm_mtl"] = mtl_norm
    if state.l2sp is not None and do_l2sp_log:
        step_log["l2sp_penalty"] = l2sp_penalty
        step_log["l2sp_dist"] = l2sp_dist
    wlog(step_log)


def train_epoch(cfg, state: TrainingState, data, *, epoch, device, dist_state,
                task_cols, task_kinds, log_mask, dataset_name, run_tag, wlog) -> EpochMetrics:
    """Train once over the loader and return reduced loss and cached-prediction metrics."""
    ftc = cfg.finetune
    n_tasks = len(task_cols)
    is_main = dist_state.is_main
    freeze_encoder_epochs = int(ftc.get("freeze_encoder_epochs", 0))
    grad_accum_steps = max(1, int(ftc.get("grad_accum_steps", 1)))
    log_every_n = int(cfg.misc.get("log_every", 500))
    train_sampler = data.train_loader.sampler if dist_state.enabled else None
    y_mean_dev, y_std_dev = data.y_mean.to(device), data.y_std.to(device)
    if freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs:
        set_encoder_trainable(state.core.encoder, True)
        # Rebuild DDP after unfreezing parameters.
        state.model = ddp.wrap_ddp(state.core, dist_state)
        if is_main:
            print(f"[{dataset_name}] {run_tag}: unfroze encoder @ epoch {epoch}"
                  + (" (DDP wrapper rebuilt)" if dist_state.enabled else ""), flush=True)
    apply_learning_rate(cfg, state, epoch)
    if isinstance(train_sampler, DistributedSampler):
        train_sampler.set_epoch(epoch)

    state.model.train()
    if bool(ftc.get('encoder_eval_mode', False)):
        # Encoder eval mode controls stochastic layers separately from
        # requires_grad.
        state.core.encoder.eval()
    running = torch.zeros((), device=device, dtype=torch.float64)
    epoch_t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epoch_task_loss_sum = torch.zeros(n_tasks, device=device)
    epoch_label_counts = torch.zeros(n_tasks, device=device)
    train_preds_accum, train_masks_accum, train_labels_accum = [], [], []

    n_train_batches = len(data.train_loader)     # per rank
    batches_done = 0

    for batch in data.train_loader:
        if state.micro == 0:
            state.optimizer.zero_grad(set_to_none=True)
        batch, pred, y, mask, task_losses, loss = forward_loss(
            cfg, state, data, batch, epoch=epoch, device=device,
            y_mean_dev=y_mean_dev, y_std_dev=y_std_dev,
            log_mask=log_mask, task_kinds=task_kinds)
        is_step = (state.micro + 1) >= grad_accum_steps
        if (not is_step) and isinstance(state.model, nn.parallel.DistributedDataParallel):
            with state.model.no_sync():   # skip DDP's all-reduce on non-final micro-steps
                state.scaler.scale(loss / grad_accum_steps).backward()
        else:
            state.scaler.scale(loss / grad_accum_steps).backward()

        if not is_step:
            state.micro += 1
            epoch_task_loss_sum += task_losses.detach()
            epoch_label_counts += mask.sum(dim=0)
            train_preds_accum.append(_per_mol(pred, batch).detach())
            train_masks_accum.append(_per_mol(mask, batch).detach())
            train_labels_accum.append(_per_mol(y, batch).detach())
            running.add_(loss.detach())
            continue
        state.micro = 0

        will_log = (batches_done == 0) or ((state.step + 1) % log_every_n == 0)
        diagnostics = optimizer_step(cfg, state, dist_state=dist_state, will_log=will_log)
        epoch_task_loss_sum += task_losses.detach()
        epoch_label_counts += _per_mol(mask, batch).sum(dim=0)
        train_preds_accum.append(_per_mol(pred, batch).detach())
        train_masks_accum.append(_per_mol(mask, batch).detach())
        train_labels_accum.append(_per_mol(y, batch).detach())

        running.add_(loss.detach())
        state.step += 1
        batches_done += 1
        if will_log:
            log_training_step(
                state, diagnostics, epoch=epoch, loss=loss, running=running,
                batches_done=batches_done, n_train_batches=n_train_batches,
                epoch_t0=epoch_t0, dataset_name=dataset_name, run_tag=run_tag,
                dist_state=dist_state, wlog=wlog)

    mean_loss = running / max(1, len(data.train_loader))
    if dist_state.enabled:
        ddp.all_reduce_mean_(mean_loss, dist_state)
        ddp.all_reduce_sum_(epoch_task_loss_sum, dist_state)
        ddp.all_reduce_sum_(epoch_label_counts, dist_state)
    mean_loss = mean_loss.item()
    epoch_time = time.time() - epoch_t0
    n_batches = max(1, len(data.train_loader)) * (dist_state.world_size if dist_state.enabled else 1)
    epoch_task_loss = (epoch_task_loss_sum / n_batches).cpu().numpy()          # [T]
    train_pred_std, train_maes, train_macro, train_rs = summarize_training_predictions(
        train_preds_accum, train_masks_accum, train_labels_accum,
        y_mean=data.y_mean, y_std=data.y_std, log_mask=log_mask, task_cols=task_cols,
        reg_task_idx=data.reg_task_idx, report_scale=data.report_scale)
    return EpochMetrics(mean_loss, epoch_task_loss, epoch_label_counts, train_pred_std,
                        train_maes, train_macro, train_rs, epoch_time)


