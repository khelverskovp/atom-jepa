"""Combine saved seed predictions with the official TDC evaluator.

Require at least five seeds. Report both conformer modes when per-conformer predictions
exist; otherwise report ensemble results only."""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np


def _seed_files(run_dir, bench):
    """Find seed prediction files in per_seed_preds, flat benchmark-prefixed, or benchmark-
    subdirectory layouts."""
    patterns = [
        os.path.join(run_dir, "per_seed_preds", f"{bench}_seed*.npz"),
        os.path.join(run_dir, f"preds_{bench}_seed*.npz"),
        os.path.join(run_dir, bench, f"preds_{bench}_seed*.npz"),
    ]
    out = {}
    for pat in patterns:
        for p in glob.glob(pat):
            m = re.search(r"_seed(\d+)\.npz$", os.path.basename(p))
            if m:
                out.setdefault(int(m.group(1)), p)
    return dict(sorted(out.items()))


def reduce_one(run_dir, bench, group, dry_run=False, mode="avg_error"):
    files = _seed_files(run_dir, bench)
    if len(files) < 5:
        print(f"  {bench:<34} SKIP -- {len(files)}/5 seeds present "
              f"({sorted(files) or 'none'})")
        return None

    # Order by seed so the list handed to evaluate_many is deterministic.
    predictions_list, per_seed_meta, per_seed_conf = [], [], []
    for seed, path in files.items():
        d = np.load(path)
        predictions_list.append({bench: d["preds"]})
        # every conformer's own prediction (finetune_admet.py, since the avg_error test
        # mode); absent for older caches and for the one-prediction-per-molecule baselines
        per_seed_conf.append(d["preds_conf"] if "preds_conf" in d else None)
        # Read seeds from either filename layout and retain optional fallback counts.
        per_seed_meta.append({"seed": int(seed),
                              "n_fallback": int(d["n_fallback"]) if "n_fallback" in d else 0})

    # Resolve the metric as in training.
    from finetuning.admet.metrics import resolve_task, tdc_leaderboard
    from finetuning.admet.baseline import get_test_and_trainval
    _cname, train_val_df, _test_df = get_test_and_trainval(group, bench)
    task = resolve_task(_cname, train_val_df["Y"].to_numpy())

    # both test conformer modes, TDC's convention (see metrics.core.tdc_leaderboard)
    lb = tdc_leaderboard(group, bench, _test_df["Y"].to_numpy(),
                         [p[bench] for p in predictions_list], per_seed_conf)
    ens, avg = lb["ensemble"], lb["avg_error"]
    if ens is None:                             # evaluate_many returned / raised an error
        print(f"  {bench:<34} FAILED -- group.evaluate_many gave no result")
        return None
    used, note = mode, ""
    if mode == "avg_error" and avg is None:
        used, note = "ensemble", ("  (no per-conformer predictions saved -- an older fine-tune "
                                  "cache or a one-prediction-per-molecule baseline)")
    mean, std = avg if used == "avg_error" and avg is not None else ens

    result = {"dataset": bench, "metric": task.metric_name, "kind": task.kind,
              "mean": mean, "std": std,
              "conformer_eval_mode": used,
              "test_avg_error": ({"mean": avg[0], "std": avg[1]} if avg else None),
              "test_ensemble": {"mean": ens[0], "std": ens[1]},
              "seeds": [m["seed"] for m in per_seed_meta],
              "per_seed": per_seed_meta,
              "n_fallback_total": sum(m["n_fallback"] for m in per_seed_meta),
              "reduced_from_per_seed_preds": True}
    out_path = os.path.join(run_dir, f"admet_tdc_{bench}_summary.json")
    if dry_run:
        print(f"  {bench:<34} {mean:.4f} +- {std:.4f} [{used}]{note}   (dry-run, not written)")
    else:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  {bench:<34} {mean:.4f} +- {std:.4f} [{used}]{note}   -> {os.path.basename(out_path)}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="a model's TDC checkpoint dir (contains per_seed_preds/)")
    ap.add_argument("--benchmark", default=None,
                    help="one benchmark name; default = every benchmark found")
    ap.add_argument("--tdc-path", default="data/admet_group",
                    help="TDC group download/cache dir (default: data/admet_group)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be written without writing it")
    ap.add_argument("--conformer-eval-mode", choices=["avg_error", "ensemble"], default="avg_error",
                    help="which test conformer mode becomes the summary's headline mean/std "
                         "(both are always stored): avg_error = average single-conformer "
                         "prediction (default; falls back to ensemble when a run saved no "
                         "per-conformer predictions), ensemble = conformer-averaged prediction")
    args = ap.parse_args()

    if args.benchmark:
        benches = [args.benchmark]
    else:
        # Discover across BOTH layouts (see _seed_files). "preds_" is stripped
        # because finetune_admet.py prefixes it and baseline.py does not.
        found = set()
        for pat in (os.path.join(args.run_dir, "per_seed_preds", "*_seed*.npz"),
                    os.path.join(args.run_dir, "preds_*_seed*.npz"),
                    os.path.join(args.run_dir, "*", "preds_*_seed*.npz")):
            for f in glob.glob(pat):
                stem = re.sub(r"_seed\d+\.npz$", "", os.path.basename(f))
                found.add(stem[len("preds_"):] if stem.startswith("preds_") else stem)
        benches = sorted(found)
    if not benches:
        sys.exit(f"no per-seed prediction files under {args.run_dir} -- looked for "
                 f"per_seed_preds/<bench>_seed<N>.npz (finetuning.admet.baseline) and "
                 f"[<bench>/]preds_<bench>_seed<N>.npz (finetuning.admet.finetune_admet).")

    # Imported here, not at module scope: instantiating the group downloads /
    # reads the TDC cache, which is pointless for --help or a bad path.
    from tdc.benchmark_group import admet_group
    group = admet_group(path=args.tdc_path)

    print(f"[aggregate_admet_tdc_seeds] {args.run_dir}  ({len(benches)} benchmark(s))")
    done = sum(1 for b in benches
               if reduce_one(args.run_dir, b, group, args.dry_run, mode=args.conformer_eval_mode))
    print(f"  -> {done}/{len(benches)} reduced"
          + ("" if done == len(benches) else "; the rest need all 5 seeds first"))


if __name__ == "__main__":
    main()
