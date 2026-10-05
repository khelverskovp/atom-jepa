"""Validation-selected checkpoints and resume artifacts for multi-task training."""

import os

import numpy as np
import torch

from finetuning.admet.training.evaluation import evaluate_test
from finetuning.admet.training.utils import _better, _worst


def initialize_selection(ftc):
    specs = ftc.get("select", None) or [{"name": "", "key": "macro", "mode": "min"}]
    return {str(spec["name"]): {
        "best": _worst(str(spec.get("mode", "min"))), "mode": str(spec.get("mode", "min")),
        "key": str(spec["key"]), "epoch": -1, "state": None,
        "test_report": None, "no_improve": 0,
    } for spec in specs}


def checkpoint_path(cfg, dataset_name, run_tag, *, kind="finetuned", selection=""):
    scratch = "_scratch" if bool(cfg.finetune.get("train_from_scratch", False)) else ""
    suffix = f"_{selection}" if selection else ""
    os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)
    return os.path.join(cfg.misc.checkpoint_dir, f"{kind}_{dataset_name}_{run_tag}{suffix}{scratch}.pt")


def model_metadata(cfg, state, data, task_cols, task_kinds, encoder_config, run_tag):
    ftc = cfg.finetune
    fusion = ftc.get("fusion", {})
    return {
        "encoder_config": encoder_config, "run_tag": run_tag, "task_cols": task_cols,
        "task_kinds": task_kinds, "pool": str(ftc.get("pool", "mean")),
        "loss": str(ftc.get("loss", "huber")).lower(),
        "standardize": bool(ftc.get("standardize", True)),
        "use_higher_order": state.core.use_higher_order,
        "shared_head": bool(ftc.get("shared_head", False)),
        "fusion": str(fusion.get("mode", "none")) if fusion.get("enabled", False) else "none",
        "feature_dim": data.feature_dim, "fusion_hidden": int(fusion.get("hidden", 64)),
        "fusion_dropout": float(fusion.get("dropout", 0.0)),
        "y_mean": data.y_mean.tolist(), "y_std": data.y_std.tolist(),
        "ema": bool(ftc.get("ema", True)),
    }


def update_selection(cfg, selection, values, state, data, score, metadata, *,
                     epoch, val_macro, dist_state, dataset_name, run_tag):
    """Select on validation everywhere; evaluate/save selected weights on rank 0."""
    for name, spec in selection.items():
        value = values.get(name, float("nan"))
        if not np.isfinite(value) or not _better(value, spec["best"], spec["mode"]):
            spec["no_improve"] += 1
            continue
        spec.update(best=value, epoch=epoch, no_improve=0)
        if dist_state.is_main:
            spec["test_report"] = evaluate_test(score, state.eval_model, data.test_loader, is_main=True)
            spec["state"] = {k: v.detach().cpu().clone() for k, v in state.eval_model.state_dict().items()}
            if bool(cfg.finetune.get("save_best", True)):
                torch.save({**metadata, "model": spec["state"], "select_name": name,
                            "select_key": spec["key"], "select_mode": spec["mode"],
                            "select_value": spec["best"], "val_macro": val_macro,
                            "test_report": spec["test_report"], "epoch": spec["epoch"]},
                           checkpoint_path(cfg, dataset_name, run_tag, selection=name))


def save_last_checkpoint(cfg, state, data, selection, metadata, *, epoch,
                         is_main, dataset_name, run_tag):
    if not is_main or not bool(cfg.finetune.get("save_last", True)):
        return
    primary = next(iter(selection.values()))
    architecture = {key: metadata[key] for key in (
        "use_higher_order", "shared_head", "fusion", "feature_dim", "fusion_hidden", "fusion_dropout")}
    torch.save({**architecture, "model": state.core.state_dict(),
                "ema_model": state.ema_model.state_dict() if state.ema_model is not None else None,
                "optimizer": state.optimizer.state_dict(),
                "mtl_loss": state.mtl_loss.state_dict() if state.mtl_loss is not None else None,
                "epoch": epoch, "best_val": primary["best"], "best_epoch": primary["epoch"],
                "epochs_no_improve": min(spec["no_improve"] for spec in selection.values()),
                "y_mean": data.y_mean, "y_std": data.y_std},
               checkpoint_path(cfg, dataset_name, run_tag, kind="last"))


def save_final_checkpoint(cfg, state, metadata, *, epoch, is_main, dataset_name, run_tag):
    """Save the last model when validation-based selection is disabled."""
    if is_main and bool(cfg.finetune.get("save_best", True)):
        torch.save({**metadata,
                    "model": {k: v.detach().cpu().clone() for k, v in state.eval_model.state_dict().items()},
                    "select_name": "", "select_key": None, "select_mode": None, "select_value": None,
                    "train_on_val": bool(cfg.finetune.get("train_on_val", False)),
                    "select_best": bool(cfg.finetune.get("select_best", True)), "epoch": epoch},
                   checkpoint_path(cfg, dataset_name, run_tag))
