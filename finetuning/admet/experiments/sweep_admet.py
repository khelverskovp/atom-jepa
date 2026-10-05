"""Sweep one TDC benchmark, selecting by mean validation score across seeds.

Trial overrides take precedence over dataset and global settings. Trial seed and epoch
budgets are configurable; final_full optionally evaluates the winner."""

import argparse
import copy
import itertools
import json
import os
from typing import Dict, List

import numpy as np
from omegaconf import DictConfig, OmegaConf

from finetuning.admet.config import config_dict
from finetuning.admet.finetune_admet import finetune_one, load_admet_group

# knobs that resolve_settings reads per-dataset -> must be written into per_dataset
PER_DATASET_KEYS = {
    "pool", "standardize", "log_transform", "loss", "head_dropout", "lr",
    "head_lr_mult", "batch_size", "grad_clip", "freeze_encoder_epochs",
    "lr_warmup_epochs", "pos_weight", "cutoff",
}

DEFAULT_SPACE = {
    "lr": [1e-4, 5e-5, 2e-5],
    "head_lr_mult": [1.0, 10.0, 50.0],
    "pool": ["mean", "sum"],
    "head_dropout": [0.0, 0.1, 0.2],
}


def mode_for_metric(metric: str) -> str:
    return "min" if str(metric).lower() == "mae" else "max"


def make_trials(space: Dict[str, list], mode: str, n_trials: int, seed: int) -> List[Dict]:
    keys = list(space.keys())
    grid = [dict(zip(keys, combo)) for combo in itertools.product(*[space[k] for k in keys])]
    if mode == "grid":
        return grid
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(grid), size=min(n_trials, len(grid)), replace=False)
    return [grid[int(i)] for i in idx]


def apply_overrides(cfg, dataset: str, overrides: Dict):
    """Write trial overrides at the right precedence layer."""
    # per-dataset-tunable knobs -> highest-precedence per_dataset[dataset] block
    pd = config_dict(cfg.finetune.get("per_dataset", {}) or {})
    existing = {}
    for k, v in pd.items():
        if str(k).lower() == dataset.lower():
            existing = v or {}
            break
    per_ds_over = {k: v for k, v in overrides.items() if k in PER_DATASET_KEYS}
    merged = {**existing, **per_ds_over}
    cfg.finetune.per_dataset = {dataset: merged}
    # global knobs -> set directly on finetune
    for k, v in overrides.items():
        if k not in PER_DATASET_KEYS:
            cfg.finetune[k] = v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="TDC ADMET benchmark name")
    ap.add_argument("--config", default="conf/finetune_admet.yaml")
    ap.add_argument("--space", default=None,
                    help="JSON dict of {knob: [values]}; default is a small built-in grid")
    ap.add_argument("--mode", choices=["grid", "random"], default="grid")
    ap.add_argument("--n-trials", type=int, default=12, help="for --mode random")
    ap.add_argument("--seeds", type=int, nargs="*", default=[1],
                    help="seeds per trial (few = fast); the final run uses the cfg seeds")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs for trials")
    ap.add_argument("--sweep-dir", default="checkpoint/sweeps")
    ap.add_argument("--wandb", action="store_true", help="keep wandb on for trials (noisy)")
    ap.add_argument("--final-full", action="store_true",
                    help="retrain the best config with the cfg's full seed set")
    ap.add_argument("--sample-seed", type=int, default=0)
    args = ap.parse_args()

    base = OmegaConf.load(args.config)
    if not isinstance(base, DictConfig):
        raise TypeError("Sweep configuration must be a mapping")
    space = json.loads(args.space) if args.space else DEFAULT_SPACE
    trials = make_trials(space, args.mode, args.n_trials, args.sample_seed)
    print(f"[sweep] {args.dataset}: {len(trials)} trial(s) over {list(space.keys())} "
          f"(seeds/trial={args.seeds})", flush=True)

    group = load_admet_group(base.data.tdc_path)

    results = []
    for ti, override in enumerate(trials):
        cfg = copy.deepcopy(base)
        cfg.data.datasets = args.dataset
        cfg.finetune.seeds = list(args.seeds)
        if args.epochs is not None:
            cfg.finetune.epochs = int(args.epochs)
        if not args.wandb:
            cfg.wandb.enabled = False
            cfg.wandb.mode = "disabled"
        # isolate each trial's checkpoints / prediction caches
        cfg.misc.checkpoint_dir = os.path.join(args.sweep_dir, args.dataset, f"trial_{ti}")
        apply_overrides(cfg, args.dataset, override)

        print(f"\n[sweep] trial {ti}/{len(trials)-1}: {override}", flush=True)
        res = finetune_one(cfg, group, args.dataset)
        val_score = float(np.nanmean(res["per_seed_val"]))
        results.append({
            "trial": ti, "override": override, "val": val_score,
            "test_mean": res["mean"], "test_std": res["std"],
            "metric": res["metric"], "kind": res["kind"],
        })
        print(f"[sweep] trial {ti}: val={val_score:.4f} "
              f"test={res['mean']:.4f}±{res['std']:.4f} {res['metric']}", flush=True)

    metric = results[0]["metric"]
    mode = mode_for_metric(metric)
    results.sort(key=lambda r: r["val"], reverse=(mode == "max"))

    print(f"\n==================== sweep ranking ({args.dataset}, "
          f"select on val {metric}, {mode}) ====================")
    for r in results:
        print(f"  val={r['val']:.4f}  test={r['test_mean']:.4f}±{r['test_std']:.4f}  "
              f"{r['override']}")

    best = results[0]
    print(f"\n[sweep] BEST by val: {best['override']} "
          f"(val {best['val']:.4f}, test {best['test_mean']:.4f}±{best['test_std']:.4f})")

    os.makedirs(os.path.join(args.sweep_dir, args.dataset), exist_ok=True)
    out = os.path.join(args.sweep_dir, args.dataset, "sweep_results.json")
    with open(out, "w") as f:
        json.dump({"dataset": args.dataset, "metric": metric, "mode": mode,
                   "results": results}, f, indent=2)
    print(f"[sweep] wrote {out}")

    if args.final_full:
        print(f"\n[sweep] retraining BEST config with full seeds "
              f"{list(base.finetune.seeds)} ...", flush=True)
        cfg = copy.deepcopy(base)
        cfg.data.datasets = args.dataset
        apply_overrides(cfg, args.dataset, best["override"])
        res = finetune_one(cfg, group, args.dataset)
        print(f"[sweep] FINAL {args.dataset}: {res['metric']} = "
              f"{res['mean']:.4f} ± {res['std']:.4f}", flush=True)


if __name__ == "__main__":
    main()