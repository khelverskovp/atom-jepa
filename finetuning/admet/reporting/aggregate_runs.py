"""Combine ChEMBL-MT or ExpansionRX run summaries and report per-fold aggregates.

Support secondary reporting scales and optional LaTeX tables. Ensemble summaries use the
largest available k on the modal-conformer subset, omit unrelated metrics, and are
written separately. Biogen has its own aggregation helper."""

import argparse
import copy
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

from finetuning.admet.reporting.aggregate_admet_tdc import (
    _latex_cell,
    _latex_mean_only_cell,
)

# Fixed ChEMBL-MT task labels.
CHEMBL_MT_TASK_LABELS = [
    ("CL_microsome_human", "Mic-H"), ("CL_microsome_mouse", "Mic-M"),
    ("CL_microsome_rat", "Mic-R"), ("CL_total_dog", "Tot-D"),
    ("CL_total_human", "Tot-H"), ("CL_total_monkey", "Tot-M"),
    ("CL_total_rat", "Tot-R"), ("CYP2C8_inhibition", "2C8"),
    ("CYP2C9_inhibition", "2C9"), ("CYP2D6_inhibition", "2D6"),
    ("CYP3A4_inhibition", "3A4"), ("Dog_fraction_unbound_plasma", "fu-D"),
    ("Human_fraction_unbound_plasma", "fu-H"), ("LogD_pH_7.4", "logD"),
    ("Monkey_fraction_unbound_plasma", "fu-M"), ("Papp_Caco2", "Papp"),
    ("Pgp_human", "P-gp"), ("Rat_fraction_unbound_plasma", "fu-R"),
    ("VDss_dog", "VD-D"), ("VDss_human", "VD-H"), ("VDss_monkey", "VD-M"),
    ("VDss_rat", "VD-R"), ("hERG_binding", "hERG"),
    ("kinetic_logSaq", "Kin-S"), ("thermo_logSaq", "Therm-S"),
]
assert len(CHEMBL_MT_TASK_LABELS) == 25

# Visual column groups for the table.
CHEMBL_MT_WIDE_GROUPS = [7, 4, 4, 3, 4, 4]
assert sum(CHEMBL_MT_WIDE_GROUPS) == len(CHEMBL_MT_TASK_LABELS) + 1  # +1 for the trailing Mean

# ExpansionRX task labels follow loader order.
from data.datasets.admet.expansionrx import LABEL_COLS as _EXPANSIONRX_LABEL_COLS
EXPANSIONRX_TASK_LABELS = list(zip(_EXPANSIONRX_LABEL_COLS, [
    "LogD", "KSOL", "HLM", "MLM", "Papp", "Efflux", "MPPB", "MBPB", "MGMB",
]))
assert len(EXPANSIONRX_TASK_LABELS) == len(_EXPANSIONRX_LABEL_COLS) == 9

BIOGEN_SUMMARY = "biogen_adme_summary.json"


def _find_summary(run_dir: str) -> str:
    """Require exactly one summary JSON in run_dir."""
    if not os.path.isdir(run_dir):
        raise SystemExit(f"ERROR: not a directory: {run_dir}")
    hits = sorted(glob.glob(os.path.join(run_dir, "*_summary.json")))
    if not hits:
        raise SystemExit(
            f"ERROR: no *_summary.json in {run_dir}\n"
            "  Did that array task actually finish? Check its job log."
        )
    if len(hits) > 1:
        raise SystemExit(
            f"ERROR: {len(hits)} summary files in {run_dir}: {[os.path.basename(h) for h in hits]}\n"
            "  Pass --summary-name to disambiguate."
        )
    return hits[0]


def _run_id(row: Dict) -> str:
    """Identify a run by its fold and/or seed."""
    bits = [f"{k}={row[k]}" for k in ("fold", "seed") if k in row]
    return ", ".join(bits) if bits else "<no fold/seed key>"


def _to_ensemble(rows: List[Dict]) -> List[Dict]:
    """Copy rows with MAE blocks replaced by the largest-k conformer-ensemble results."""
    out = []
    for r in rows:
        r = copy.deepcopy(r)
        for split in ("val", "test"):
            blk = r.get(split)
            if not blk:
                continue
            curve = blk.get("curve") or {}
            if not curve.get("per_task"):
                raise SystemExit(
                    f"ERROR: run {_run_id(r)} has no {split} ensemble-size curve -- a "
                    "single-conformer run (nothing to ensemble) or a summary from before "
                    "curves were saved. Use --conformer-eval-mode summary.")
            per_task, ks, excluded = {}, {}, {}
            for t, c in curve["per_task"].items():
                k = max(c["per_k"], key=int)
                per_task[t] = float(c["per_k"][k]["mean"])
                ks[t], excluded[t] = int(k), int(c.get("n_mol_excluded", 0))
            k_max = max(ks.values())
            short = {t: k for t, k in ks.items() if k < k_max}
            if short:
                print(f"  NOTE run {_run_id(r)} {split}: {len(short)} task(s) have fewer than "
                      f"{k_max} conformers in their curve, using their own largest k: {short}")
            macro = float(np.mean(list(per_task.values())))
            for key in [k for k in blk if k.startswith(("per_task_", "macro_"))]:
                del blk[key]            # correlations / secondary scales are single-conformer
            blk.update(per_task=per_task, macro=macro, macro_mae=macro,
                       conformer_eval_mode="ensemble", ensemble_k=k_max,
                       ensemble_n_mol_excluded=excluded)
            print(f"  run {_run_id(r)} {split}: {k_max}-conformer ensemble macro-MAE = {macro:.4f} "
                  f"(up to {max(excluded.values())} molecule(s) per task excluded: fewer "
                  "conformers than the modal count)")
        out.append(r)
    return out


def _task_cols(rows: List[Dict], split: str) -> List[str]:
    """Return the union of task names in first-seen order."""
    seen: Dict[str, None] = {}
    for r in rows:
        for t in r.get(split, {}).get("per_task", {}):
            seen.setdefault(t, None)
    return list(seen)


def _aggregate_per_task(rows: List[Dict], task_cols: List[str], split: str,
                        per_task_key: str = "per_task",
                        macro_key: Optional[str] = None) -> Dict:
    """Aggregate the requested per-task metric block across runs as mean/std."""
    out = {}
    for t in task_cols:
        vals = [r[split].get(per_task_key, {}).get(t, float("nan"))
                for r in rows if split in r]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[t] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    if macro_key is None:
        macro_key = "macro_mae" if any("macro_mae" in r.get(split, {}) for r in rows) else "macro"
    macro_vals = [r[split][macro_key] for r in rows if split in r and macro_key in r[split]]
    # isfinite() throws on anything non-numeric, and a summary may legitimately
    # carry string-valued metadata next to the metrics -- drop by type first.
    macro_vals = [float(v) for v in macro_vals
                  if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v)]
    if macro_vals:
        out["_macro"] = {"mean": float(np.mean(macro_vals)), "std": float(np.std(macro_vals)),
                         "n": len(macro_vals)}
    return out


def _report_scales(rows: List[Dict], split: str) -> List[str]:
    """Find secondary scales with both macro_mae_<name> and per_task_mae_<name> entries."""
    seen: Dict[str, None] = {}
    for r in rows:
        blk = r.get(split, {})
        for k, v in blk.items():
            # A reporting scale requires both macro and per-task values.
            if (k.startswith("macro_mae_")
                    and isinstance(v, (int, float, np.floating, np.integer))
                    and f"per_task_mae_{k[len('macro_mae_'):]}" in blk):
                seen.setdefault(k[len("macro_mae_"):], None)
    return list(seen)


def _print_group_summary(label: str, rows: List[Dict]) -> None:
    ids = [_run_id(r) for r in rows]
    print(f"\n==================== {label} (mean+-std over {len(rows)} run(s): "
          f"{', '.join(ids)}) ====================")
    for split in ("val", "test"):
        if not any(split in r for r in rows):
            continue
        cols = _task_cols(rows, split)
        blocks = [(None, "per_task", None)]
        # one extra block per secondary reporting scale, printed after the
        # native-unit one it is a re-scoring of -- never instead of it.
        blocks += [(n, f"per_task_mae_{n}", f"macro_mae_{n}")
                   for n in _report_scales(rows, split)]
        for name, ptk, mk in blocks:
            agg = _aggregate_per_task(rows, cols, split, per_task_key=ptk, macro_key=mk)
            if not agg:
                continue
            unit = "native units" if name is None else f"{name} scale"
            macro = agg.get("_macro")
            if macro is not None:
                print(f"  {split:4s} macro-MAE [{unit}] = "
                      f"{macro['mean']:.4f} +- {macro['std']:.4f}")
            for t in cols:
                if t in agg:
                    print(f"    {t}: {split} MAE = {agg[t]['mean']:.4f} +- {agg[t]['std']:.4f} "
                          f"(n={agg[t]['n']})")


def _print_summary(combined: List[Dict]) -> None:
    """Print mean/std groups separately for each fold when folds are present."""
    has_fold = any("fold" in r for r in combined)
    if not has_fold:
        _print_group_summary("summary", combined)
        return

    folds = sorted({r.get("fold") for r in combined}, key=lambda x: (x is None, x))
    for fold in folds:
        group = [r for r in combined if r.get("fold") == fold]
        _print_group_summary(f"fold {fold}", group)
    if len(folds) > 1:
        print(f"\nNOTE: {len(folds)} folds printed SEPARATELY, not collapsed into one "
              "mean+-std. ChEMBL-MT's folds are overlapping train/val resamples of one "
              "pool, not independent partitions -- pooling them would understate real "
              "uncertainty. Compare folds side by side above, or hand all rows to "
              "finetuning/admet/compare_models.py, which treats each as its own run-level sample.")


def _group_macro(rows: List[Dict], split: str) -> Optional[Dict]:
    """Return native macro-MAE mean, std, and count, or None if unavailable."""
    cols = _task_cols(rows, split)
    agg = _aggregate_per_task(rows, cols, split)
    return agg.get("_macro")


def _latex_summary(combined: List[Dict], split: str, model_name: str,
                   pool_folds: bool = False, fold_mean_only: bool = False) -> Optional[str]:
    """Format a LaTeX summary row; missing values become dashes.

    Default: one cell per fold. poolfold pools runs; foldmean averages fold means
    without uncertainty. Return None when no values are available."""
    has_fold = any("fold" in r for r in combined)

    if fold_mean_only and has_fold:
        folds = sorted({r.get("fold") for r in combined}, key=lambda x: (x is None, x))
        fold_means = []
        for fold in folds:
            macro = _group_macro([r for r in combined if r.get("fold") == fold], split)
            if macro is not None:
                fold_means.append(macro["mean"])
        if not fold_means:
            return None
        return (f"  \\quad {model_name} & "
                f"{_latex_mean_only_cell(float(np.mean(fold_means)))} \\\\")

    if has_fold and not pool_folds:
        folds = sorted({r.get("fold") for r in combined}, key=lambda x: (x is None, x))
        groups = [[r for r in combined if r.get("fold") == fold] for fold in folds]
    else:
        groups = [combined]

    cells = []
    any_value = False
    for group in groups:
        macro = _group_macro(group, split)
        if macro is not None:
            cells.append(_latex_cell(macro["mean"], macro["std"]))
            any_value = True
        else:
            cells.append("--")
    if not any_value:
        return None
    return f"  \\quad {model_name} & " + " & ".join(cells) + " \\\\"


def _wrap_latex_cells(cells: List[str], groups: List[int]) -> List[str]:
    """Split table cells into visual groups; group sizes must sum to the cell count."""
    assert sum(groups) == len(cells), (sum(groups), len(cells))
    lines, i = [], 0
    for g in groups:
        lines.append("    & " + " & ".join(cells[i:i + g]))
        i += g
    return lines


def _chembl_mt_wide_header() -> str:
    """Format fixed ChEMBL-MT task labels and a mean column."""
    cols = ["\\textbf{Model}"]
    for _, label in CHEMBL_MT_TASK_LABELS:
        cols.append("$\\boldsymbol{\\log D}$" if label == "logD" else f"\\textbf{{{label}}}")
    cols.append("\\textbf{Mean}")
    return "\n& ".join(cols) + " \\\\\n\\midrule"


def _chembl_mt_task_fold_mean(combined: List[Dict], task: str, split: str) -> Optional[float]:
    """Average a task's fold means with equal fold weight, or return None."""
    folds = sorted({r.get("fold") for r in combined if "fold" in r}, key=lambda x: (x is None, x))
    vals = []
    for fold in folds:
        group = [r for r in combined if r.get("fold") == fold]
        agg = _aggregate_per_task(group, [task], split)
        if task in agg:
            vals.append(agg[task]["mean"])
    return float(np.mean(vals)) if vals else None


def _chembl_mt_wide_row(combined: List[Dict], split: str, model_name: str) -> Optional[str]:
    """Format 25 task values and their mean to three decimals, with dashes for missing
    values."""
    cells = []
    for task, _ in CHEMBL_MT_TASK_LABELS:
        v = _chembl_mt_task_fold_mean(combined, task, split)
        cells.append(_latex_mean_only_cell(v, decimals=3) if v is not None else "--")

    folds = sorted({r.get("fold") for r in combined if "fold" in r}, key=lambda x: (x is None, x))
    fold_macro_means = []
    for fold in folds:
        m = _group_macro([r for r in combined if r.get("fold") == fold], split)
        if m is not None:
            fold_macro_means.append(m["mean"])
    if fold_macro_means:
        cells.append(_latex_mean_only_cell(float(np.mean(fold_macro_means)), decimals=3))
    elif cells and any(c != "--" for c in cells):
        cells.append("--")
    else:
        return None

    lines = [f"  \\quad {model_name}"] + _wrap_latex_cells(cells, CHEMBL_MT_WIDE_GROUPS)
    lines[-1] += " \\\\"
    return "\n".join(lines)


def _expansionrx_wide_header() -> str:
    """Format the nine ExpansionRX task labels and a mean column."""
    cols = ["\\textbf{Model}"] + [f"\\textbf{{{lbl}}}" for _, lbl in EXPANSIONRX_TASK_LABELS] \
         + ["\\textbf{Mean}"]
    return "\n& ".join(cols) + " \\\\\n\\midrule"


def _expansionrx_wide_row(combined: List[Dict], split: str, model_name: str) -> Optional[str]:
    """Format KERMT-scale task means/stds across seeds, or return None if unavailable."""
    agg = _aggregate_per_task(combined, [t for t, _ in EXPANSIONRX_TASK_LABELS], split,
                              per_task_key="per_task_mae_kermt", macro_key="macro_mae_kermt")
    if not agg:
        return None

    cells = []
    for task, _ in EXPANSIONRX_TASK_LABELS:
        v = agg.get(task)
        cells.append(_latex_cell(v["mean"], v["std"], decimals=3) if v is not None else "--")
    macro = agg.get("_macro")
    cells.append(_latex_cell(macro["mean"], macro["std"], decimals=3) if macro is not None else "--")

    return f"  \\quad {model_name} & " + " & ".join(cells) + " \\\\"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", dest="run_dirs", action="append", required=True,
                    help="A single-run checkpoint dir (repeat once per array task), OR a "
                         "direct path to one *_summary.json file (e.g. you already have one "
                         "complete summary -- from a sequential, non-array run -- and just "
                         "want --latex output from it, with nothing to combine).")
    ap.add_argument("--out", default=None,
                    help="Directory to write the combined summary into. Required only when "
                         "combining more than one --run-dir (there is genuinely no single "
                         "obvious destination then); with exactly one --run-dir this defaults "
                         "to leaving that file untouched -- pass --out to write a copy anyway.")
    ap.add_argument("--summary-name", default=None,
                    help="Summary filename, if auto-detection is ambiguous.")
    ap.add_argument("--latex", action="store_true",
                    help="also print a LaTeX row, '\\quad <model> & $mean \\err{std}$ "
                         "[& $mean \\err{std}$ ...] \\\\' -- one cell per fold when this "
                         "dataset has folds (ChEMBL-MT), never pooled into one number "
                         "(see _latex_summary's docstring); one cell otherwise (ExpansionRX)")
    ap.add_argument("--latex-split", default="test", choices=["val", "test"],
                    help="which split's macro-MAE to put in the --latex cell(s) (default: test)")
    ap.add_argument("--model-name", default=None,
                    help="row label for --latex (default: --out's basename)")
    ap.add_argument("--latex-pool-folds", action="store_true",
                    help="ChEMBL-MT only: pool every fold's runs into ONE --latex cell instead "
                         "of one cell per fold (matches how the Contrastive KERMT paper quotes "
                         "a single ChEMBL-MT number in prose -- arXiv:2606.11508; its own "
                         "results table does not report mean+-std at all, only a Tukey-HSD "
                         "best/indistinguishable count). Off by default: the folds are "
                         "overlapping resamples of one pool, not independent partitions, so "
                         "pooling understates real uncertainty regardless of what one paper's "
                         "prose does with it -- opt in only when you need a number comparable "
                         "to that specific paper. No effect on ExpansionRX (no fold key).")
    ap.add_argument("--latex-fold-mean", action="store_true",
                    help="ChEMBL-MT only: average the two folds' OWN MEANS together (mean of "
                         "means, n=2) and print that ONE cell with NO uncertainty at all -- no "
                         "\\err{}, since a std over 2 fold-means is not a quantity worth "
                         "reporting ('it is okay that no uncertainty is reported, as that is "
                         "not allowed' -- your words). Different from --latex-pool-folds, which "
                         "pools the 4 raw fold*seed values and DOES report a std over those; "
                         "the two means can disagree slightly if the folds have unequal seed "
                         "counts. Mutually exclusive with --latex-pool-folds. No effect on "
                         "ExpansionRX (no fold key) -- falls back to the default per-cell "
                         "mean+-std there, since those seeds ARE independent.")
    ap.add_argument("--latex-wide", action="store_true",
                    help="ChEMBL-MT only: print this run as ONE ROW of a wide, one-column-"
                         "per-TASK table (25 task cells + a trailing Mean cell), each "
                         "$value$ with NO uncertainty -- the mean-of-fold-means for that task, "
                         "3 decimals, leading '0' dropped (\"$.357$\"). See "
                         "CHEMBL_MT_TASK_LABELS for the column order/labels and "
                         "--latex-wide-header for the matching header row.")
    ap.add_argument("--latex-wide-header", action="store_true",
                    help="print the --latex-wide column-label header + \\midrule. Independent "
                         "of --run-dir/--latex-wide -- print this ONCE for a table, not once "
                         "per model.")
    ap.add_argument("--latex-wide-expansionrx", action="store_true",
                    help="ExpansionRX only: print this run as ONE ROW of a wide, one-column-"
                         "per-TASK table (9 task cells + a trailing Mean cell), each "
                         "'$mean \\err{std}$' -- WITH uncertainty, unlike ChEMBL-MT's "
                         "--latex-wide: ExpansionRX's 5 seeds are independent (no fold key), "
                         "so a std over them is a real, reportable quantity. Reads the KERMT "
                         "log10 reporting scale (per_task_mae_kermt/macro_mae_kermt), the "
                         "number comparable to data.datasets.admet.expansionrx.KERMT_REFERENCE_MAE and the "
                         "Contrastive KERMT paper's own ExpansionRX table -- NOT native units, "
                         "which are dominated by the raw-scale tasks. See "
                         "EXPANSIONRX_TASK_LABELS for the column order/labels and "
                         "--latex-wide-header-expansionrx for the matching header row.")
    ap.add_argument("--conformer-eval-mode", default="summary", choices=["summary", "ensemble"],
                    help="summary (default): each run's headline numbers as its driver wrote them "
                         "(finetune.conformer_eval_mode, default avg_error = one conformer). "
                         "ensemble: the full conformer ensemble, read from each run's "
                         "ensemble-size curve at its largest k; --out then writes "
                         "*_summary_ensemble.json. See the module docstring.")
    ap.add_argument("--latex-wide-header-expansionrx", action="store_true",
                    help="print the --latex-wide-expansionrx column-label header + \\midrule. "
                         "Independent of --run-dir/--latex-wide-expansionrx -- print this ONCE "
                         "for a table, not once per model.")
    args = ap.parse_args()

    if args.latex_pool_folds and args.latex_fold_mean:
        raise SystemExit(
            "ERROR: --latex-pool-folds and --latex-fold-mean compute genuinely different "
            "numbers (pooled-raw-values mean+-std vs. mean-of-fold-means with no "
            "uncertainty) -- pick one."
        )
    # LaTeX-specific options require LaTeX output.
    if (args.latex_pool_folds or args.latex_fold_mean or args.model_name
            or args.latex_split != "test") and not args.latex:
        print("[aggregate_runs] NOTE: a --latex-* option was given without --latex -- "
              "implying --latex.")
        args.latex = True

    if len(args.run_dirs) > 1 and not args.out:
        raise SystemExit(
            f"ERROR: --out is required when combining {len(args.run_dirs)} --run-dir values "
            "-- there is no single obvious destination for the merged summary. Pass --out, "
            "or drop down to one --run-dir if you only wanted --latex from an already-"
            "complete summary."
        )

    combined: List[Dict] = []
    name = args.summary_name
    for d in args.run_dirs:
        # Accept a summary file directly.
        if os.path.isfile(d):
            path = d
        else:
            path = os.path.join(d, name) if name else _find_summary(d)
        if not os.path.isfile(path):
            raise SystemExit(f"ERROR: {path} not found. Did that array task finish?")
        base = os.path.basename(path)
        if base == BIOGEN_SUMMARY:
            raise SystemExit(
                f"ERROR: {path} is a Biogen ADME summary, which is a "
                '{"per_seed": [...], "val_agg": ..., "test_agg": ...} wrapper, '
                "not a flat list.\n"
                "  Use finetuning/admet/reporting/aggregate_biogen_adme_seeds.py for Biogen instead."
            )
        if name is None:
            name = base
        elif base != name:
            raise SystemExit(
                f"ERROR: mixed summary filenames: {name!r} vs {base!r} (in {d}).\n"
                "  These dirs are from different datasets/configs -- combining "
                "them would produce a meaningless file."
            )
        with open(path) as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise SystemExit(
                f"ERROR: {path} is a {type(rows).__name__}, expected a list of "
                "per-run dicts. This script only handles the flat-list datasets "
                "(chembl_mt, expansionrx)."
            )
        print(f"  {path}: {len(rows)} run(s) -> {'; '.join(_run_id(r) for r in rows)}")
        combined += rows

    seen: Dict[str, int] = {}
    for r in combined:
        seen[_run_id(r)] = seen.get(_run_id(r), 0) + 1
    dupes = {k: v for k, v in seen.items() if v > 1}
    if dupes:
        # Duplicate runs would receive extra weight in aggregates.
        print(f"\nWARNING: {len(dupes)} duplicated run identifier(s): {dupes}\n"
              "  compare_models would treat each copy as an independent sample "
              "and double-weight it. Drop the stale --run-dir and re-run.")

    if args.conformer_eval_mode == "ensemble":
        print("\n[aggregate_runs] conformer_eval_mode=ensemble -- every number below is the "
              "full conformer ensemble, not the headline:")
        combined = _to_ensemble(combined)
        name = name.replace("_summary.json", "_summary_ensemble.json")

    _print_summary(combined)

    if args.model_name:
        model_name = args.model_name
    elif args.out:
        model_name = os.path.basename(os.path.normpath(args.out))
    else:
        # Derive the model label from the containing directory.
        d0 = args.run_dirs[0]
        model_name = os.path.basename(os.path.normpath(
            os.path.dirname(d0) if os.path.isfile(d0) else d0))

    if args.latex:
        row = _latex_summary(combined, args.latex_split, model_name,
                             pool_folds=args.latex_pool_folds,
                             fold_mean_only=args.latex_fold_mean)
        if row is None:
            print(f"\n[aggregate_runs] NOTE: --latex requested but no {args.latex_split!r} "
                  "macro value found in any group -- nothing to print.")
        else:
            has_fold = (any("fold" in r for r in combined)
                       and not args.latex_pool_folds and not args.latex_fold_mean)
            mode = ("mean of fold-means, no uncertainty" if args.latex_fold_mean
                    else "folds pooled" if args.latex_pool_folds
                    else "one cell per fold" if has_fold else "")
            print(f"\n==================== LaTeX row ({args.latex_split}"
                  f"{', ' + mode if mode else ''}) ====================")
            print(row)

    if args.latex_wide_header:
        print("\n==================== LaTeX wide-table header (ChEMBL-MT) ====================")
        print(_chembl_mt_wide_header())

    if args.latex_wide:
        if not any("fold" in r for r in combined):
            print("\n[aggregate_runs] NOTE: --latex-wide is ChEMBL-MT-specific (needs a "
                  "'fold' key -- CHEMBL_MT_TASK_LABELS is a fixed 25-task mapping) -- this "
                  "summary has none, skipping.")
        else:
            row = _chembl_mt_wide_row(combined, args.latex_split, model_name)
            if row is None:
                print(f"\n[aggregate_runs] NOTE: --latex-wide requested but no {args.latex_split!r} "
                      "value found for any task or the macro -- nothing to print.")
            else:
                print(f"\n==================== LaTeX wide-table row ({args.latex_split}"
                      ") ====================")
                print(row)

    if args.latex_wide_header_expansionrx:
        print("\n==================== LaTeX wide-table header (ExpansionRX) ====================")
        print(_expansionrx_wide_header())

    if args.latex_wide_expansionrx:
        if any("fold" in r for r in combined):
            print("\n[aggregate_runs] NOTE: --latex-wide-expansionrx is ExpansionRX-specific "
                  "(no 'fold' key expected) -- this summary has one (looks like ChEMBL-MT?), "
                  "skipping. Use --latex-wide for ChEMBL-MT instead.")
        else:
            row = _expansionrx_wide_row(combined, args.latex_split, model_name)
            if row is None:
                print(f"\n[aggregate_runs] NOTE: --latex-wide-expansionrx requested but no "
                      f"{args.latex_split!r} kermt-scale value found -- either this run has no "
                      "report_scale attached (an old pre-report_scale run, or "
                      "finetune.report_scale disabled), or no seed has finished yet.")
            else:
                print(f"\n==================== LaTeX wide-table row, ExpansionRX "
                      f"({args.latex_split}, kermt scale) ====================")
                print(row)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, name)
        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nwrote {len(combined)} run(s) from {len(args.run_dirs)} dir(s) -> {out_path}")
    else:
        print(f"\n[aggregate_runs] --out not given -- nothing written (only one --run-dir, "
              "so there is nothing to combine; its file is unchanged).")


if __name__ == "__main__":
    main()
