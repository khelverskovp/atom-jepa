"""Score external KERMT predictions in the shared summary schema, optionally using the
ExpansionRX log10 reporting scale."""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data.datasets.admet.expansionrx import KERMT_LOG10_TASKS
from finetuning.admet.metrics.core import (
    per_task_corr,
    per_task_mae,
    report_scale_metrics,
)
from finetuning.admet.metrics.report_scale import make_log10_report_scale

# dataset -> (split subdirectory for a run_tag, summary filename our drivers use)
SPLIT_DIR = {
    "biogen_adme": lambda tag: f"biogen_adme/{tag}",
    # same protocol, our full 6-endpoint task set instead of their published 4
    "biogen_adme_all6": lambda tag: f"biogen_adme_all6/{tag}",
    # one shipped split shared by every seed
    "expansionrx": lambda tag: "expansionrx/fixed",
    # Same split and molecules, but exported with LOG10 labels so KERMT trains in the
    # space it is scored in (see finetuning.admet.export_splits_for_kermt.export_expansionrx).
    "expansionrx_log10": lambda tag: "expansionrx_log10/fixed",
    # run_tag is f<fold>_s<seed>; the split depends on the fold only
    "chembl_mt": lambda tag: f"chembl_mt_public/{tag.split('_')[0]}",
}
SUMMARY_NAME = {"biogen_adme": "biogen_adme_summary.json",
                "biogen_adme_all6": "biogen_adme_summary.json",
                "expansionrx": "expansionrx_summary.json",
                "expansionrx_log10": "expansionrx_summary.json",
                "chembl_mt": "chembl_mt_public_summary.json"}


def _aligned(preds_csv: str, test_csv: str, tasks: List[str]):
    """Align predictions and labels by SMILES and return [n_mol,n_task] arrays."""
    pred = pd.read_csv(preds_csv)
    test = pd.read_csv(test_csv)
    if "smiles" not in pred.columns:
        # Accept KERMT's unnamed SMILES index only when it matches the split.
        first = str(pred.columns[0])
        overlap = (len(set(pred[first].astype(str)) & set(test["smiles"].astype(str)))
                   if pred[first].dtype == object else 0)
        if overlap >= 0.9 * len(test):
            pred = pred.rename(columns={first: "smiles"})
            print(f"  (took unnamed column {first!r} as SMILES: {overlap}/{len(test)} matched)")
        else:
            raise SystemExit(f"ERROR: {preds_csv} has no `smiles` column and its first "
                             f"column matches only {overlap}/{len(test)} test molecules "
                             f"(got {list(pred.columns)[:5]})")
    missing = [t for t in tasks if t not in pred.columns]
    if missing:
        raise SystemExit(f"ERROR: {preds_csv} is missing predicted task column(s) {missing[:3]}")
    merged = test[["smiles"] + tasks].merge(pred[["smiles"] + tasks], on="smiles",
                                            how="left", suffixes=("_true", "_pred"))
    if len(merged) != len(test):
        raise SystemExit(f"ERROR: {preds_csv} does not cover {test_csv} one-to-one "
                         f"({len(merged)} rows vs {len(test)})")
    labels = merged[[f"{t}_true" for t in tasks]].to_numpy(dtype=float)
    preds = merged[[f"{t}_pred" for t in tasks]].to_numpy(dtype=float)
    if np.isnan(preds).all(axis=0).any():
        bad = [t for t, allnan in zip(tasks, np.asarray(np.isnan(preds).all(axis=0)).reshape(-1)) if allnan]
        raise SystemExit(f"ERROR: {preds_csv} has no finite prediction for {bad[:3]}")
    return preds, labels


def score_run(run_dir: str, split_dir: str, dataset: str) -> Optional[Dict]:
    preds_csv = os.path.join(run_dir, "test_preds.csv")
    test_csv = os.path.join(split_dir, "test.csv")
    if not os.path.exists(preds_csv):
        print(f"  skip {run_dir}: no test_preds.csv (did the predict pass run?)")
        return None
    tasks = [c for c in pd.read_csv(test_csv, nrows=0).columns if c != "smiles"]
    preds, labels = _aligned(preds_csv, test_csv, tasks)

    maes = per_task_mae(preds, labels)
    corrs = per_task_corr(preds, labels)
    with np.errstate(invalid="ignore"):
        macro = float(np.nanmean(maes))
    report = {"per_task": {t: float(v) for t, v in zip(tasks, maes)},
              "per_task_corr": {t: float(v) for t, v in zip(tasks, corrs)},
              "macro": macro, "macro_mae": macro}

    if dataset == "expansionrx":
        # floors from TRAIN labels only, matching finetuning/admet/finetune_expansionrx.py
        train = pd.read_csv(os.path.join(split_dir, "train.csv"))
        rs = make_log10_report_scale(tasks, list(KERMT_LOG10_TASKS),
                                     train[tasks].to_numpy(dtype=float), name="kermt")
        report.update(report_scale_metrics(rs, preds, labels, tasks, list(range(len(tasks)))))
    elif dataset == "expansionrx_log10":
        # Predictions already in log10 space must not be transformed again.
        report["per_task_mae_kermt"] = dict(report["per_task"])
        report["macro_mae_kermt"] = float(report["macro"])
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(SPLIT_DIR))
    ap.add_argument("--runs-root", default="external/contrastive-kermt/runs")
    ap.add_argument("--splits-root", default="external/contrastive-kermt/splits")
    ap.add_argument("--out", default=None,
                    help="directory for the summary JSON (default: <runs-root>/<dataset>), "
                         "which is what --model points at in compare_models/compare_ranks")
    args = ap.parse_args()

    root = os.path.join(args.runs_root, args.dataset)
    if not os.path.isdir(root):
        raise SystemExit(f"ERROR: {root} not found -- run scripts/kermt/submit_finetune_kermt.sh first.")
    out_dir = args.out or root
    rows: List[Dict] = []
    for tag in sorted(os.listdir(root)):
        run_dir = os.path.join(root, tag)
        if not os.path.isdir(run_dir):
            continue
        report = score_run(run_dir, os.path.join(args.splits_root, SPLIT_DIR[args.dataset](tag)),
                           args.dataset)
        if report is None:
            continue
        row = {"run_tag": tag, "test": report}
        # the key our drivers carry alongside run_tag, so aggregation code that groups
        # by fold (ChEMBL) or seed (the others) keeps working
        if tag.startswith("f") and "_s" in tag:
            row["fold"] = int(tag.split("_")[0][1:])
            row["seed"] = int(tag.split("_s")[1])
        else:
            row["seed"] = int(tag.replace("seed", ""))
        rows.append(row)
        print(f"  {tag}: macro={report['macro']:.4f}"
              + (f"  kermt={report['macro_mae_kermt']:.4f}" if "macro_mae_kermt" in report else ""))

    if not rows:
        raise SystemExit("ERROR: no scored runs -- nothing written.")
    rows.sort(key=lambda r: r["run_tag"])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, SUMMARY_NAME[args.dataset])
    payload = rows
    if args.dataset.startswith("biogen_adme"):
        # Biogen's summary is a {"per_seed": [...]} wrapper, not a flat list
        payload = {"per_seed": rows, "val_agg": {}, "test_agg": {}}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {len(rows)} run(s) -> {path}\n"
          f"  use it as: --model ckermt={out_dir}")


if __name__ == "__main__":
    main()
