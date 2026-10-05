"""Combine per-seed Biogen summaries and recompute aggregates; optionally report the four
dense tasks excluding PPB."""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

from data.datasets.admet.biogen_adme import TASK_COLS
from finetuning.admet.finetune_biogen_adme import _aggregate_per_task

# Dense-task macro excludes sparse PPB measurements.
LARGE_TASKS = [
    "LOG HLM_CLint (mL/min/kg)",
    "LOG RLM_CLint (mL/min/kg)",
    "LOG MDR1-MDCK ER (B-A/A-B)",
    "LOG SOLUBILITY PH 6.8 (ug/mL)",
]


def _macro_subset(results: List[Dict], task_subset: List[str], split: str) -> Optional[Dict]:
    """Compute each run's macro over the selected tasks, then aggregate across runs."""
    macro_vals = []
    for r in results:
        per_task = r[split]["per_task"]
        vals = [per_task[t] for t in task_subset if t in per_task and np.isfinite(per_task[t])]
        if vals:
            macro_vals.append(float(np.mean(vals)))
    if not macro_vals:
        return None
    return {"mean": float(np.mean(macro_vals)), "std": float(np.std(macro_vals)), "n_seeds": len(macro_vals)}


def aggregate(seed_dirs: List[str], out_dir: str) -> Dict:
    all_results = []
    seen_seeds = set()
    for d in seed_dirs:
        path = os.path.join(d, "biogen_adme_summary.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found -- did this seed's array task finish?")
        with open(path) as f:
            s = json.load(f)
        for r in s["per_seed"]:
            if r["seed"] in seen_seeds:
                raise ValueError(
                    f"seed {r['seed']} appears in more than one --seed-dir -- these "
                    f"must be a clean partition (each seed run exactly once)."
                )
            seen_seeds.add(r["seed"])
            all_results.append(r)
    all_results.sort(key=lambda r: r["seed"])

    val_agg = _aggregate_per_task(all_results, TASK_COLS, "val")
    test_agg = _aggregate_per_task(all_results, TASK_COLS, "test")

    val_macro_large = _macro_subset(all_results, LARGE_TASKS, "val")
    test_macro_large = _macro_subset(all_results, LARGE_TASKS, "test")
    if val_macro_large is not None:
        val_agg["_macro_large"] = val_macro_large
    if test_macro_large is not None:
        test_agg["_macro_large"] = test_macro_large

    seeds = [r["seed"] for r in all_results]
    # Do not infer scaffold splitting when summary metadata is absent.
    print(f"\n==================== Biogen ADME summary (mean+-std over "
          f"{len(all_results)} seeds: {seeds}) ====================")
    print(f"  val  macro-MAE = {val_agg['_macro']['mean']:.4f} +- {val_agg['_macro']['std']:.4f}")
    print(f"  test macro-MAE = {test_agg['_macro']['mean']:.4f} +- {test_agg['_macro']['std']:.4f}")
    if test_macro_large is not None:
        print(f"  test MAE (large tasks only: HLM/RLM CLint, MDR1-MDCK ER, solubility) "
              f"= {test_macro_large['mean']:.4f} +- {test_macro_large['std']:.4f}")
    for t in TASK_COLS:
        if t in test_agg:
            print(f"    {t}: test MAE = {test_agg[t]['mean']:.4f} +- {test_agg[t]['std']:.4f} "
                  f"(n_seeds={test_agg[t]['n_seeds']})")

    os.makedirs(out_dir, exist_ok=True)
    summary = {"per_seed": all_results, "val_agg": val_agg, "test_agg": test_agg}
    out_path = os.path.join(out_dir, "biogen_adme_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[aggregate_biogen_adme_seeds] wrote combined summary -> {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed-dir", action="append", required=True, dest="seed_dirs",
                    help="repeatable: one per single-seed checkpoint dir")
    ap.add_argument("--out", required=True,
                    help="output dir for the combined biogen_adme_summary.json "
                         "(the dir finetuning.admet.compare_models should point at for this model variant)")
    args = ap.parse_args()
    aggregate(args.seed_dirs, args.out)


if __name__ == "__main__":
    main()
