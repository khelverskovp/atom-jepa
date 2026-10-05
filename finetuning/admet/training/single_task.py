"""Single-task ADMET training, validation selection, and train+validation refitting."""

import copy
import os
from dataclasses import asdict, replace
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from data.datasets.admet.admet_finetune import (
    ADMETFinetuneDataset,
    admet_collate,
    get_train_valid,
)
from finetuning.admet.metrics import (
    TaskSpec,
    _format_curve,
    evaluate,
    evaluate_full,
    predict_dataframe,
)
from finetuning.admet.models.model import FineTuneModel
from finetuning.admet.training.regularization import build_adamw
from finetuning.admet.training.helpers import (
    cosine_factor,
    save_full_state,
    set_encoder_trainable,
)
from finetuning.execution import configure_execution
from finetuning.admet.training.loss import fit_target_stats, fwd_target, regression_loss
from finetuning.admet.training.utils import _better, _worst, ema_update, move_batch
from atom_jepa.models.jepa_equiformer import EquiformerV3Encoder


def train_seed(cfg, group, cname, seed, task: TaskSpec, S, conformers, eqv3_cfg, encoder_ckpt,
               train_val_df, test_df, device, use_wandb, refit_epochs: Optional[int] = None,
               probe_only: bool = False) -> Dict:
    """Train one seed and return aligned test predictions and evaluation reports.

    S contains resolved per-dataset settings. probe_only selects the best epoch on held-
    out validation without testing; refit_epochs trains on train+validation for that
    many epochs on the original cosine schedule, without selection. The default is a
    single train/validation pass. Per-seed full-state checkpoints support resume. The
    caller owns the W&B run and prediction cache."""
    ftc = cfg.finetune
    pool = S["pool"]
    standardize = bool(S["standardize"])
    log_transform = bool(S["log_transform"])
    loss_type = str(S["loss"]).lower()
    head_dropout = float(S["head_dropout"])
    base_lr = float(S["lr"])
    bs = int(S["batch_size"])
    grad_clip = float(S["grad_clip"])
    head_lr_mult = float(S["head_lr_mult"])
    freeze_encoder_epochs = int(S["freeze_encoder_epochs"])
    lr_warmup_epochs = int(S["lr_warmup_epochs"])
    use_nc = ftc.get("num_conformers", None)
    use_nc = None if use_nc in (None, "all", -1, 0) else int(use_nc)
    conformer_eval_mode = str(ftc.get("conformer_eval_mode", "avg_error")).lower()
    ensemble_curve_max_trials = int(ftc.get("ensemble_curve_max_trials", 256))

    epochs = int(ftc.epochs)
    encoder_weight_decay = float(ftc.get("encoder_weight_decay", 0.0))
    trunk_weight_decay = float(ftc.get("trunk_weight_decay", 0.1))
    use_ema = bool(ftc.ema)
    ema_decay = float(ftc.ema_decay)
    lr_factor = float(ftc.get("lr_factor", 0.5))
    lr_patience = int(ftc.get("lr_patience", 10))
    min_lr = float(ftc.get("min_lr", 1e-12))
    patience = int(ftc.get("patience", 25))
    do_resume = bool(ftc.get("resume", False))
    save_last = bool(ftc.get("save_last", True))
    train_from_scratch = bool(ftc.get("train_from_scratch", False))
    ckpt_tag = "_scratch" if train_from_scratch else ""

    torch.manual_seed(seed)

    cutoff_override = S.get("cutoff", None)
    if cutoff_override is not None:
        eqv3_cfg = replace(eqv3_cfg, max_radius=float(cutoff_override))

    eqv3_cfg = replace(eqv3_cfg, grad_checkpointing=bool(S["grad_checkpointing"]))

    cutoff = eqv3_cfg.max_radius
    F_in = eqv3_cfg.num_channels
    max_z = int(getattr(eqv3_cfg, "max_num_elements", 128))

    split_type = "scaffold" if bool(ftc.get("scaffold_split", True)) else "random"
    train_df, valid_df = get_train_valid(group, cname, seed, split_type=split_type)
    # two-pass refit, pass 2: fold the validation rows into training. The validation
    # split is then in-sample, so it is neither scored per epoch nor used to select.
    refit = refit_epochs is not None
    if refit:
        if str(ftc.get("scheduler", "cosine")).lower() == "plateau":
            raise ValueError("finetune.train_on_val (two-pass refit) needs scheduler=cosine: "
                             "ReduceLROnPlateau steps on a validation metric the refit no longer has.")
        train_df = pd.concat([train_df, valid_df], ignore_index=True)

    def make_ds(df, tag, sample, num_conformers):
        return ADMETFinetuneDataset(
            df, conformers, cutoff, max_z=max_z,
            num_conformers=num_conformers, sample_conformers=sample, tag=tag,
        )

    train_ds = make_ds(train_df, f"{cname}/train/s{seed}", True, use_nc)   # sample one conformer/epoch
    valid_ds = make_ds(valid_df, f"{cname}/valid/s{seed}", False, None)    # expand + average, full ensemble
    test_ds = make_ds(test_df, f"{cname}/test/s{seed}", False, None)       # full ensemble

    num_workers = cfg.data.num_workers
    pin = (device.type == "cuda")

    def make_loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers,
            collate_fn=admet_collate, drop_last=False, pin_memory=pin,
            persistent_workers=(num_workers > 0),
        )

    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(valid_ds, False)

    fallback = float(train_ds.labels.mean())              # TRAIN mean / positive rate
    y_mean, y_std = fit_target_stats(train_ds.labels, standardize, log_transform)

    pos_weight = None
    if task.is_classification:
        pw = S.get("pos_weight", "auto")
        if isinstance(pw, str) and pw.lower() == "auto":
            n_pos = float(train_ds.labels.sum())
            n_neg = float(len(train_ds.labels) - n_pos)
            pos_weight = torch.tensor(n_neg / max(1.0, n_pos), device=device)
        elif isinstance(pw, (int, float)):
            pos_weight = torch.tensor(float(pw), device=device)

    encoder = EquiformerV3Encoder(eqv3_cfg).to(device)
    if not train_from_scratch:
        try:
            encoder.load_state_dict(encoder_ckpt, strict=True)
        except RuntimeError:
            encoder.body.load_state_dict(encoder_ckpt, strict=True)
    model = FineTuneModel(encoder, F_in, pool=pool, dropout=head_dropout).to(device)

    if freeze_encoder_epochs > 0:
        set_encoder_trainable(model.encoder, False)

    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)

    optimizer = build_adamw(
        [
            {"params": list(model.encoder.parameters()), "lr": base_lr,
             "weight_decay": encoder_weight_decay},
            {"params": list(model.head.parameters()), "lr": base_lr * head_lr_mult,
             "weight_decay": trunk_weight_decay},
        ],
        lr=base_lr,
    )
    group_base_lrs = [base_lr, base_lr * head_lr_mult]   # for the cosine schedule

    sched_type = str(ftc.get("scheduler", "cosine")).lower()
    plateau = None
    if sched_type == "plateau":
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=task.mode, factor=lr_factor, patience=lr_patience, min_lr=min_lr,
        )

    def apply_cosine(ep: int):
        f = cosine_factor(ep, warmup_epochs=lr_warmup_epochs, epochs=epochs,
                          min_lr=min_lr, base_lr=base_lr)
        for g, blr in zip(optimizer.param_groups, group_base_lrs):
            g["lr"] = blr * f

    all_params = list(model.parameters())

    if use_wandb:
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    def wlog(d):
        if use_wandb:
            payload = dict(d)
            payload["step"] = step
            wandb.log(payload)

    best_val = _worst(task.mode)
    best_test = float("nan")
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    start_epoch = 0
    step = 0

    resume_path = os.path.join(cfg.misc.checkpoint_dir, f"last_{cname}_seed{seed}{ckpt_tag}.pt")
    if do_resume and os.path.exists(resume_path):
        rs = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(rs["model"])
        if ema_model is not None and rs.get("ema_model") is not None:
            ema_model.load_state_dict(rs["ema_model"])
        optimizer.load_state_dict(rs["optimizer"])
        if plateau is not None and rs.get("scheduler") is not None:
            plateau.load_state_dict(rs["scheduler"])
        best_val = float(rs["best_val"]); best_test = float(rs["best_test"])
        best_epoch = int(rs["best_epoch"]); epochs_no_improve = int(rs["epochs_no_improve"])
        start_epoch = int(rs["epoch"]) + 1
        y_mean = float(rs.get("y_mean", y_mean)); y_std = float(rs.get("y_std", y_std))
        torch.set_rng_state(rs["torch_rng_state"].cpu())
        if rs.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rs["cuda_rng_state"]])
        set_encoder_trainable(model.encoder, start_epoch >= freeze_encoder_epochs)
        print(f"[finetune] {cname} s{seed}: RESUMED @ epoch {start_epoch} "
              f"(best_val {best_val:.4f} @ {best_epoch})", flush=True)

    model.head.set_target_stats(y_mean, y_std)
    if ema_model is not None:
        ema_model.head.set_target_stats(y_mean, y_std)

    scaler = configure_execution(ftc, model, ema_model, device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    test_loader = make_loader(test_ds, False)             # for the my-metric test number

    # refit: stop after pass 1's best epoch; the cosine schedule above still spans `epochs`
    n_run = int(refit_epochs) if refit else epochs
    if n_run <= 0:
        raise ValueError("Training requires at least one epoch")
    eval_model = ema_model if ema_model is not None else model
    for epoch in range(start_epoch, n_run):
        # unfreeze the encoder once the head-only warmup is over
        if freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs:
            set_encoder_trainable(model.encoder, True)
            print(f"[finetune] {cname} s{seed}: unfroze encoder @ epoch {epoch}", flush=True)
        if plateau is None:
            apply_cosine(epoch)

        model.train()
        running = torch.zeros((), device=device, dtype=torch.float64)
        for batch in train_loader:
            batch = move_batch(batch, device)
            y = batch["y"][:, 0]
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            if task.is_classification:
                loss = bce(pred, y)
            else:
                yt = fwd_target(y, log_transform)
                yb = (yt - y_mean) / y_std
                loss = regression_loss(pred, yb, loss_type)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    all_params, grad_clip,
                    error_if_nonfinite=(bool(ftc.get('cuequivariance', False))
                                        and not scaler.is_enabled()))
            scaler.step(optimizer)
            scaler.update()
            if use_ema:
                ema_update(ema_model, model, ema_decay)
            running.add_(loss.detach())
            step += 1
            if step % cfg.misc.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"[{cname} s{seed}] ep {epoch} step {step} loss {loss.item():.4f} "
                      f"lr {lr_now:.2e}", flush=True)
                wlog({"train_loss": loss.item(), "lr": lr_now, "epoch": epoch})

        mean_loss = running.item() / max(1, len(train_loader))
        eval_model = ema_model if ema_model is not None else model
        if refit:
            val_metric = float("nan")
            best_epoch = epoch
        else:
            val_metric = evaluate(eval_model, val_loader, device, task, y_mean, y_std, log_transform,
                                  conformer_eval_mode=conformer_eval_mode)
            if plateau is not None:
                plateau.step(val_metric)

        if not refit and _better(val_metric, best_val, task.mode):
            best_val = val_metric
            if not probe_only:          # pass 1 only measures the stopping epoch
                best_test = evaluate(eval_model, test_loader, device, task, y_mean, y_std, log_transform,
                                     conformer_eval_mode=conformer_eval_mode)
            best_epoch = epoch
            epochs_no_improve = 0
            # snapshot the best eval weights to CPU for the final test prediction
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
            if bool(ftc.get("save_best", True)) and not probe_only:
                os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)
                torch.save(
                    {
                        "model": best_state, "eqv3_cfg": asdict(eqv3_cfg),
                        "dataset": cname, "seed": seed, "task_kind": task.kind,
                        "metric": task.metric_name, "pool": pool, "loss": loss_type,
                        "standardize": standardize, "log_transform": log_transform,
                        "y_mean": y_mean, "y_std": y_std,
                        "val_metric": best_val, "test_metric": best_test,
                        "epoch": best_epoch, "ema": use_ema,
                    },
                    os.path.join(cfg.misc.checkpoint_dir,
                                 f"finetuned_{cname}_seed{seed}{ckpt_tag}.pt"),
                )
        else:
            epochs_no_improve += 1

        if save_last:
            os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)
            last_path = os.path.join(cfg.misc.checkpoint_dir, f"last_{cname}_seed{seed}{ckpt_tag}.pt")
            tmp = last_path + ".tmp"
            save_full_state(
                tmp, model=model, ema_model=ema_model, optimizer=optimizer,
                scheduler=plateau, epoch=epoch, best_val=best_val, best_test=best_test,
                best_epoch=best_epoch, epochs_no_improve=epochs_no_improve,
                y_mean=y_mean, y_std=y_std, eqv3_cfg=eqv3_cfg, name=cname, task=task,
                pool=pool, standardize=standardize, log_transform=log_transform,
                use_ema=use_ema,
            )
            os.replace(tmp, last_path)

        lr_now = optimizer.param_groups[0]["lr"]
        if refit:
            print(f"== [{cname} s{seed}] refit ep {epoch} ({epoch + 1}/{n_run}) loss {mean_loss:.4f} "
                  f"(train+val, nothing selected) lr {lr_now:.2e}", flush=True)
            wlog({"epoch_loss": mean_loss, "lr": lr_now, "epoch": epoch})
        else:
            print(f"== [{cname} s{seed}] ep {epoch} loss {mean_loss:.4f} "
                  f"val_{task.metric_name} {val_metric:.4f} "
                  f"(best {best_val:.4f} @ {best_epoch}, test {best_test:.4f}) lr {lr_now:.2e}",
                  flush=True)
            wlog({"epoch_loss": mean_loss, f"val_{task.metric_name}": val_metric,
                  "best_val": best_val, "best_test": best_test, "lr": lr_now, "epoch": epoch})

        if not refit and epochs_no_improve >= patience:
            print(f"[finetune] {cname} s{seed}: early stop @ epoch {epoch}", flush=True)
            break

    if probe_only:
        # two-pass refit, pass 1: only the stopping point survives (weights discarded)
        return {"best_epoch": best_epoch, "val": best_val}

    if best_state is not None:
        eval_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    if refit:
        best_test = evaluate(eval_model, test_loader, device, task, y_mean, y_std, log_transform,
                             conformer_eval_mode=conformer_eval_mode)
        if bool(ftc.get("save_best", True)):
            os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    "model": {k: v.detach().cpu() for k, v in eval_model.state_dict().items()},
                    "eqv3_cfg": asdict(eqv3_cfg), "dataset": cname, "seed": seed,
                    "task_kind": task.kind, "metric": task.metric_name, "pool": pool,
                    "loss": loss_type, "standardize": standardize, "log_transform": log_transform,
                    "y_mean": y_mean, "y_std": y_std, "test_metric": best_test,
                    "epoch": best_epoch, "ema": use_ema, "refit_on_train_val": True,
                },
                os.path.join(cfg.misc.checkpoint_dir, f"finetuned_{cname}_seed{seed}{ckpt_tag}.pt"),
            )
    prediction_result = predict_dataframe(
        eval_model, test_df, conformers, cutoff, max_z, device, task,
        y_mean, y_std, log_transform, fallback, bs, num_conformers=use_nc,
        return_per_conformer=True,
    )

    assert len(prediction_result) == 3
    preds, n_fallback, preds_conf = prediction_result

    curve_rng = np.random.default_rng(seed)
    # refit: the validation split was trained on, so its report would be in-sample -- skipped
    val_full = (None if refit else
                evaluate_full(eval_model, val_loader, device, task, y_mean, y_std, log_transform,
                              max_trials=ensemble_curve_max_trials, rng=curve_rng))
    test_full = evaluate_full(eval_model, test_loader, device, task, y_mean, y_std, log_transform,
                              max_trials=ensemble_curve_max_trials, rng=curve_rng)
    if val_full is not None:
        print(f"[finetune] {cname} s{seed} FINAL val:  avg_error={val_full['avg_error']:.4f} "
              f"ensemble={val_full['ensemble']:.4f}", flush=True)
    print(f"[finetune] {cname} s{seed} FINAL test: avg_error={test_full['avg_error']:.4f} "
          f"ensemble={test_full['ensemble']:.4f}", flush=True)
    if val_full is not None:
        print(f"[finetune] {cname} s{seed} val  ensemble curve: {_format_curve(val_full['curve'])}", flush=True)
    print(f"[finetune] {cname} s{seed} test ensemble curve: {_format_curve(test_full['curve'])}", flush=True)
    wlog({
        **({"val_avg_error": val_full["avg_error"], "val_ensemble": val_full["ensemble"],
            **{f"val_curve_k{k}": v["mean"] for k, v in val_full["curve"]["per_k"].items()}}
           if val_full is not None else {}),
        "test_avg_error": test_full["avg_error"], "test_ensemble": test_full["ensemble"],
        **{f"test_curve_k{k}": v["mean"] for k, v in test_full["curve"]["per_k"].items()},
    })

    return {"preds": preds, "preds_conf": preds_conf, "val": best_val, "test": best_test,
            "best_epoch": best_epoch, "n_fallback": int(n_fallback),
            "val_full": val_full, "test_full": test_full}
