"""Fine-tune EquiformerV3 on the 22 TDC ADMET regression/classification benchmarks.

Each benchmark uses its task-specific metric and optimization direction.
By default, seeds 1–5 generate scaffold train/validation splits with a fixed
held-out test set. Report mean and standard deviation across seeds.

With train_on_val, select the epoch count on validation before refitting on
train+validation. Cache per-seed predictions and report both single-conformer
and conformer-ensemble scores.

Run: python -m finetuning.admet.finetune_admet finetune.ckpt_path=/path/to/encoder.pt
"""

import copy
import gc
import json
import os
import traceback
from typing import Dict, List

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from data.datasets.admet.admet_conformers import (
    DEFAULT_PRUNE_RMS,
    build_or_load_conformers,
)
from data.datasets.admet.admet_finetune import get_test_and_trainval, load_admet_group
from finetuning.admet.config import config_dict
from finetuning.admet.metrics import (
    TaskSpec,
    _aggregate_curves,
    resolve_task,
    tdc_leaderboard,
)
from finetuning.admet.training.single_task import train_seed
from finetuning.admet.training.utils import load_pretrained_encoder, resolve_device


def _as_dict(x) -> dict:
    return config_dict(x)


def resolve_settings(ftc, task: TaskSpec, cname: str) -> dict:
    """Merge base settings, task-type defaults, and dataset overrides in that order."""
    s = {
        "pool": ftc.get("pool", "mean"),
        "standardize": bool(ftc.get("standardize", True)),
        "loss": str(ftc.get("loss", "huber")).lower(),
        "log_transform": bool(ftc.get("log_transform", False)),
        "lr": float(ftc.get("lr", 5e-5)),
        "head_lr_mult": float(ftc.get("head_lr_mult", 10.0)),
        "batch_size": int(ftc.get("batch_size", 32)),
        "head_dropout": float(ftc.get("head_dropout", 0.1)),
        "grad_clip": float(ftc.get("grad_clip", 10.0)),
        "freeze_encoder_epochs": int(ftc.get("freeze_encoder_epochs", 0)),
        "lr_warmup_epochs": int(ftc.get("lr_warmup_epochs", 0)),
        "pos_weight": ftc.get("pos_weight", "auto"),     # "auto" | "none" | float
        "cutoff": ftc.get("cutoff", None),               # None -> encoder's pretrained max_radius
        "grad_checkpointing": bool(ftc.get("grad_checkpointing", False)),
    }
    key = "defaults_classification" if task.is_classification else "defaults_regression"
    s.update(_as_dict(ftc.get(key, None)))
    per_ds = _as_dict(ftc.get("per_dataset", None))
    for k, v in per_ds.items():
        if str(k).lower() == cname.lower():
            s.update(_as_dict(v))
            break
    # Disable regression-only transforms for classification.
    s["standardize"] = bool(s["standardize"]) and not task.is_classification
    s["log_transform"] = bool(s["log_transform"]) and not task.is_classification
    return s


def _train_seed_maybe_two_pass(cfg, group, cname, seed, task, S, conformers, eqv3_cfg, encoder_ckpt,
                               train_val_df, test_df, device, use_wandb, ckpt_tag) -> Dict:
    """Train one seed, optionally refitting on train+validation.

    With train_on_val, first select the epoch count on held-out validation,
    then refit on train+validation for that many epochs using the same cosine
    LR schedule. Cache the first pass's result unless force_retrain is set."""
    if not bool(cfg.finetune.get("train_on_val", False)):
        return train_seed(cfg, group, cname, seed, task, S, conformers, eqv3_cfg, encoder_ckpt,
                          train_val_df, test_df, device, use_wandb)

    probe_path = os.path.join(cfg.misc.checkpoint_dir, f"pass1_{cname}_seed{seed}{ckpt_tag}.json")
    if os.path.exists(probe_path) and not bool(cfg.finetune.get("force_retrain", False)):
        with open(probe_path) as f:
            probe = json.load(f)
        print(f"[finetune] {cname} s{seed}: pass 1 cached ({probe_path})", flush=True)
    else:
        # Discard probe weights; cache the selected epoch and validation score.
        cfg1 = copy.deepcopy(cfg)
        OmegaConf.set_struct(cfg1, False)
        for key, val in (("save_best", False), ("save_last", False), ("resume", False)):
            OmegaConf.update(cfg1, f"finetune.{key}", val, merge=True)
        OmegaConf.set_struct(cfg1, True)
        p1 = train_seed(cfg1, group, cname, seed, task, S, conformers, eqv3_cfg, encoder_ckpt,
                        train_val_df, test_df, device, False, probe_only=True)
        probe = {"best_epoch": int(p1["best_epoch"]), "val": float(p1["val"])}
        with open(probe_path, "w") as f:
            json.dump(probe, f, indent=2)
        torch.cuda.empty_cache()

    refit_epochs = max(1, int(probe["best_epoch"]) + 1)              # best_epoch is 0-indexed
    print(f"[finetune] {cname} s{seed}: train_on_val pass 1 best epoch {probe['best_epoch']} "
          f"(held-out val {task.metric_name} {probe['val']:.4f}) -> refitting on train+val "
          f"for {refit_epochs} epoch(s)", flush=True)
    r = train_seed(cfg, group, cname, seed, task, S, conformers, eqv3_cfg, encoder_ckpt,
                   train_val_df, test_df, device, use_wandb, refit_epochs=refit_epochs)
    r.update(best_epoch=int(probe["best_epoch"]), val=float(probe["val"]), refit_epochs=refit_epochs)
    return r


def finetune_one(cfg: DictConfig, group, name: str) -> Dict:
    device = torch.device(resolve_device(cfg.misc.device))
    ftc = cfg.finetune
    seeds = [int(s) for s in ftc.get("seeds", [1, 2, 3, 4, 5])]
    ckpt_tag = "_scratch" if bool(ftc.get("train_from_scratch", False)) else ""

    # Reuse conformers across seeds; keep the test partition fixed.
    cname, train_val_df, test_df = get_test_and_trainval(group, name)
    all_smiles = train_val_df["Drug"].tolist() + test_df["Drug"].tolist()
    conformers = build_or_load_conformers(
        cname, all_smiles, cfg.data.conformer_cache_dir,
        seed=int(cfg.data.get("conformer_seed", 42)),
        n_conf=int(cfg.data.get("num_conformers", 10)),
        strip_fragments=bool(cfg.data.get("strip_multi_fragment_smiles", True)),
        n_workers=int(cfg.data.get("conformer_workers", 1)),
        prune_rms=float(cfg.data.get("conformer_prune_rms", DEFAULT_PRUNE_RMS)),
    )
    task = resolve_task(cname, train_val_df["Y"].to_numpy(),
                        metric_override=ftc.get("metric_override", None))
    S = resolve_settings(ftc, task, cname)

    # Load the checkpoint architecture and weights once; reinitialize each seed.
    ckpt_path = ftc.get("ckpt_path", None) or os.path.join(
        cfg.misc.checkpoint_dir, "context_encoder.pt"
    )
    _, eqv3_cfg, ckpt = load_pretrained_encoder(ckpt_path, device, load_weights=False)
    encoder_ckpt = None
    if not bool(ftc.get("train_from_scratch", False)):
        for key in ("context_encoder", "encoder", "model"):
            if key in ckpt and ckpt[key] is not None:
                encoder_ckpt = ckpt[key]
                break
        if encoder_ckpt is None:
            raise KeyError("checkpoint has no encoder weights; expected 'context_encoder'.")

    print(f"[finetune] {cname}: task={task.kind} metric={task.metric_name} "
          f"(mode={task.mode}) seeds={seeds}", flush=True)
    cutoff_str = (f"{float(S['cutoff']):.1f}" if S.get("cutoff") is not None
                  else f"{eqv3_cfg.max_radius:.1f}(pretrained)")
    print(f"[finetune] {cname}: settings -> cutoff={cutoff_str} pool={S['pool']} "
          f"standardize={S['standardize']} log_transform={S['log_transform']} "
          f"loss={S['loss']} lr={S['lr']:.2e} head_lr_mult={S['head_lr_mult']} "
          f"bs={S['batch_size']} grad_ckpt={S['grad_checkpointing']} "
          f"sched={str(ftc.get('scheduler', 'cosine'))} "
          f"num_conformers={ftc.get('num_conformers', None)} "
          f"conformer_eval_mode={str(ftc.get('conformer_eval_mode', 'avg_error'))}", flush=True)

    os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)
    force = bool(ftc.get("force_retrain", False))
    train_on_val = bool(ftc.get("train_on_val", False))
    print(f"[finetune] {cname}: train_on_val={train_on_val} "
          f"({'two-pass: best epoch on val, then refit on train+val' if train_on_val else 'train only, best val epoch'})",
          flush=True)

    # Group separate W&B runs for each dataset/seed.
    use_wandb = bool(cfg.wandb.enabled) and cfg.wandb.mode != "disabled"
    wandb_group = f"admet-{cname}{ckpt_tag}"
    base = (cfg.wandb.run_name + "-") if cfg.wandb.run_name else ""

    per_seed = []
    for seed in seeds:
        pred_path = os.path.join(
            cfg.misc.checkpoint_dir, f"preds_{cname}_seed{seed}{ckpt_tag}.npz"
        )
        if use_wandb:
            run_cfg = OmegaConf.to_container(cfg, resolve=True)
            if not isinstance(run_cfg, dict):
                raise TypeError("Expected the resolved run configuration to be a dictionary")
            run_cfg = {str(key): value for key, value in run_cfg.items()}
            run_cfg["seed"] = seed
            wandb.init(
                project=cfg.wandb.project, entity=cfg.wandb.entity,
                name=f"{base}admet-{cname}{ckpt_tag}-s{seed}",
                group=wandb_group, job_type="seed", mode=cfg.wandb.mode, reinit=True,
                config=run_cfg,
            )
        use_cache = os.path.exists(pred_path) and not force
        d = None
        if use_cache:
            d = np.load(pred_path, allow_pickle=False)
            # Reject caches from a different train_on_val mode; legacy caches are train-only.
            cached_tov = bool(d["train_on_val"]) if "train_on_val" in d else False
            if cached_tov != train_on_val:
                print(f"[finetune] {cname} s{seed}: cached predictions were made with "
                      f"train_on_val={cached_tov}, this run has {train_on_val} -- retraining.",
                      flush=True)
                use_cache = False
        if use_cache and d is not None:
            refit_cached = int(d["refit_epochs"]) if "refit_epochs" in d else -1
            per_seed.append({"seed": seed, "preds": d["preds"], "val": float(d["val"]),
                             "test": float(d["test"]), "best_epoch": int(d["best_epoch"]),
                             "n_fallback": int(d["n_fallback"]),
                             "preds_conf": d["preds_conf"] if "preds_conf" in d else None,
                             "refit_epochs": refit_cached if refit_cached >= 0 else None})
            print(f"[finetune] {cname} s{seed}: loaded cached predictions "
                  f"(val {float(d['val']):.4f})", flush=True)
            if use_wandb:                       # cached seeds have no curve; record finals
                wandb.summary["best_val"] = float(d["val"])
                wandb.summary["best_test"] = float(d["test"])
                wandb.finish()
            continue
        r = _train_seed_maybe_two_pass(cfg, group, cname, seed, task, S, conformers, eqv3_cfg,
                                       encoder_ckpt, train_val_df, test_df, device, use_wandb,
                                       ckpt_tag)
        # Release cached GPU memory before the next seed.
        torch.cuda.empty_cache()
        # Save ensemble predictions and NaN-padded per-conformer predictions.
        np.savez(pred_path, preds=r["preds"], preds_conf=r["preds_conf"], val=r["val"],
                 test=r["test"], best_epoch=r["best_epoch"], n_fallback=r["n_fallback"],
                 refit_epochs=(r.get("refit_epochs") or -1), train_on_val=train_on_val)
        per_seed.append({"seed": seed, **r})
        if use_wandb:
            wandb.finish()

    # Report both conformer modes; conformer_eval_mode selects the headline score.
    # avg_error averages scores across conformer draws; ensemble scores averaged predictions.
    predictions_list = [{cname: ps["preds"]} for ps in per_seed]
    lb = tdc_leaderboard(group, cname, test_df["Y"].to_numpy(),
                         [ps["preds"] for ps in per_seed], [ps.get("preds_conf") for ps in per_seed])
    ens = lb["ensemble"]
    if ens is None:
        # Fall back to stored per-seed metrics if TDC aggregation is unavailable.
        print(f"[finetune] {cname}: evaluate_many unavailable; ensemble number = mean of our "
              f"own per-seed test metric.", flush=True)
        vals = np.array([ps["test"] for ps in per_seed], dtype=np.float64)
        ens = (float(np.nanmean(vals)), float(np.nanstd(vals)))
    avg = lb["avg_error"]
    test_mode = str(ftc.get("conformer_eval_mode", "avg_error")).lower()
    if test_mode == "avg_error" and avg is None:
        print(f"[finetune] {cname}: WARNING: not every seed has per-conformer test predictions "
              f"(cache from before they were saved) -- headline falls back to the ensemble "
              f"number.", flush=True)
        test_mode = "ensemble"
    mean, std = avg if test_mode == "avg_error" and avg is not None else ens
    print(f"[finetune] {cname}: test single-conformer (avg_error) = "
          + (f"{avg[0]:.4f} ± {avg[1]:.4f}" if avg else "n/a")
          + f" | conformer ensemble = {ens[0]:.4f} ± {ens[1]:.4f}  [headline: {test_mode}]", flush=True)

    # Check that validation selection and TDC evaluation use the same metric.
    try:
        one = group.evaluate(predictions_list[0])          # {cname: {metric_name: val}}
        tdc_metric = next(iter(one[cname].keys()))
        norm = lambda s: str(s).lower().replace("_", "-").replace(" ", "-")
        if norm(tdc_metric) != norm(task.metric_name):
            print(f"[finetune] WARNING: TDC scores {cname} with {tdc_metric!r} but "
                  f"val-selection used {task.metric_name!r}. Set "
                  f"finetune.metric_override={tdc_metric!r} so selection matches.",
                  flush=True)
        else:
            print(f"[finetune] {cname}: metric check OK (TDC + val both "
                  f"{task.metric_name}).", flush=True)
    except Exception as e:
        print(f"[finetune] {cname}: metric cross-check skipped ({e!r}).", flush=True)

    total_fallback = int(sum(ps["n_fallback"] for ps in per_seed)) // max(1, len(per_seed))
    print(f"[finetune] {cname}: {task.metric_name} = {mean:.4f} ± {std:.4f} "
          f"over {len(seeds)} seeds (test n={len(test_df)}, "
          f"~{total_fallback} fallback/seed)", flush=True)

    # Aggregate conformer reports across seeds; skip caches without full reports.
    val_entries = [ps["val_full"] for ps in per_seed if ps.get("val_full") is not None]
    test_entries = [ps["test_full"] for ps in per_seed if ps.get("test_full") is not None]
    val_agg = _aggregate_curves(val_entries) if val_entries else None
    test_agg = _aggregate_curves(test_entries) if test_entries else None
    if test_agg and len(test_entries) < len(per_seed):
        print(f"[finetune] {cname}: ensemble-curve report covers {len(test_entries)}/"
              f"{len(per_seed)} seeds (rest loaded from a prediction cache built "
              f"before this report existed).", flush=True)
    if val_agg and test_agg:
        print(f"[finetune] {cname}: val  avg_error={val_agg['avg_error']['mean']:.4f}"
              f"±{val_agg['avg_error']['std']:.4f} ensemble={val_agg['ensemble']['mean']:.4f}"
              f"±{val_agg['ensemble']['std']:.4f} ({val_agg['n_seeds']} seeds)", flush=True)
        print(f"[finetune] {cname}: test avg_error={test_agg['avg_error']['mean']:.4f}"
              f"±{test_agg['avg_error']['std']:.4f} ensemble={test_agg['ensemble']['mean']:.4f}"
              f"±{test_agg['ensemble']['std']:.4f} ({test_agg['n_seeds']} seeds)", flush=True)
        curve_str = " ".join(f"k={k}:{v['mean']:.4f}±{v['std']:.4f}"
                             for k, v in sorted(test_agg["curve"].items()))
        print(f"[finetune] {cname}: test ensemble curve (seed-avg): {curve_str}", flush=True)

    # Log aggregate results in a separate W&B summary run.
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project, entity=cfg.wandb.entity,
            name=f"{base}admet-{cname}{ckpt_tag}-summary",
            group=wandb_group, job_type="summary", mode=cfg.wandb.mode, reinit=True,
            config={"dataset": cname, "n_seeds": len(seeds), "seeds": seeds},
        )
        wandb.summary[f"{task.metric_name}_mean"] = mean
        wandb.summary[f"{task.metric_name}_std"] = std
        wandb.summary["test_conformer_eval_mode"] = test_mode
        wandb.summary[f"{task.metric_name}_ensemble_mean"] = ens[0]
        if avg:
            wandb.summary[f"{task.metric_name}_avg_error_mean"] = avg[0]
        wandb.summary["n_seeds"] = len(seeds)
        if test_agg:
            wandb.summary["test_avg_error"] = test_agg["avg_error"]["mean"]
            wandb.summary["test_ensemble"] = test_agg["ensemble"]["mean"]
            for k, v in test_agg["curve"].items():
                wandb.summary[f"test_curve_k{k}"] = v["mean"]
        wandb.finish()

    return {
        "dataset": cname, "metric": task.metric_name, "kind": task.kind,
        "mean": mean, "std": std, "seeds": seeds,
        "per_seed_val": [ps["val"] for ps in per_seed],
        "per_seed_test": [ps["test"] for ps in per_seed],
        "val_ensemble_report": val_agg, "test_ensemble_report": test_agg,
        # Record the headline mode and both conformer scores.
        "conformer_eval_mode": test_mode,
        "test_avg_error": ({"mean": avg[0], "std": avg[1]} if avg else None),
        "test_ensemble": {"mean": ens[0], "std": ens[1]},
        "train_on_val": train_on_val,
        "per_seed_best_epoch": [ps.get("best_epoch") for ps in per_seed],
        "per_seed_refit_epochs": [ps.get("refit_epochs") for ps in per_seed],
    }


def resolve_dataset_names(cfg, group) -> List[str]:
    """Resolve data.datasets: null selects all benchmarks; a string selects one."""
    requested = cfg.data.get("datasets", None)
    if requested is None:
        return list(group.dataset_names)
    if isinstance(requested, str):
        return [requested]
    return [str(x) for x in requested]


def finetune(cfg: DictConfig) -> List[Dict]:
    group = load_admet_group(cfg.data.tdc_path)
    names = resolve_dataset_names(cfg, group)
    print(f"[finetune] running {len(names)} ADMET benchmark(s): {names}", flush=True)

    results: List[Dict] = []
    for name in names:
        try:
            results.append(finetune_one(cfg, group, name))
        except Exception as e:
            print(f"[finetune] ERROR on {name}: {e!r} -- skipping.", flush=True)
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            try:
                wandb.finish()
            except Exception:
                pass

    print("\n==================== ADMET leaderboard summary ====================")
    for r in results:
        print(f"  {r['dataset']:<32} {r['metric']:<9} "
              f"{r['mean']:.4f} ± {r['std']:.4f}")
    os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)
    summary_path = os.path.join(cfg.misc.checkpoint_dir, "admet_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[finetune] wrote summary -> {summary_path}", flush=True)
    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="finetune_admet")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    finetune(cfg)


if __name__ == "__main__":
    main()
