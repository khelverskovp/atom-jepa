"""Collect per-benchmark TDC summaries, report missing benchmarks, and optionally format
LaTeX tables."""

import argparse
import glob
import json
import os
from typing import Dict, List

# Fixed TDC benchmark order for table alignment.
TDC_ADMET_NAMES = [
    "caco2_wang", "hia_hou", "pgp_broccatelli", "bioavailability_ma",
    "lipophilicity_astrazeneca", "solubility_aqsoldb", "bbb_martins", "ppbr_az",
    "vdss_lombardo", "cyp2c9_veith", "cyp2d6_veith", "cyp3a4_veith",
    "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels", "half_life_obach",
    "clearance_microsome_az", "clearance_hepatocyte_az", "herg", "ames",
    "dili", "ld50_zhu",
]

# Labels follow the benchmark order.
TDC_ADMET_LABELS = [
    "Caco2", "HIA", "P-gp", "Bioav.", "Lipo.", "Solub.", "BBB", "PPBR", "VDss",
    "2C9-I", "2D6-I", "3A4-I", "2C9-S", "2D6-S", "3A4-S", "Half-life",
    "Cl-Mic", "Cl-Hep", "hERG", "AMES", "DILI", "LD50",
]
assert len(TDC_ADMET_LABELS) == len(TDC_ADMET_NAMES)


def _latex_num(x: float, decimals: int = 4) -> str:
    """Format fixed decimals, retaining the leading zero."""
    return f"{x:.{decimals}f}"


def _latex_strip_zero(s: str) -> str:
    """Remove the leading zero for values in [0,1)."""
    return s[1:] if s.startswith("0.") else s


def _latex_err(x: float, decimals: int = 4) -> str:
    """Format uncertainty with fixed decimals and no leading zero for values in [0,1)."""
    return _latex_strip_zero(_latex_num(x, decimals))


def _latex_cell(mean: float, std: float, decimals: int = 4) -> str:
    """Format a mean and uncertainty cell using the paper-defined err macro."""
    return f"${_latex_num(mean, decimals)} \\err{{{_latex_err(std, decimals)}}}$"


def _latex_mean_only_cell(mean: float, decimals: int = 4) -> str:
    """Format a mean-only cell, omitting the leading zero for values in [0,1)."""
    return f"${_latex_strip_zero(_latex_num(mean, decimals))}$"


def _latex_row(name: str, mean: float, std: float) -> str:
    """Format one benchmark as a LaTeX table row."""
    return f"  {name} & {_latex_cell(mean, std)} \\\\"


def _latex_wide_header() -> str:
    """Format fixed benchmark labels and the mean column."""
    cols = ["\\textbf{Model}"] + [f"\\textbf{{{lbl}}}" for lbl in TDC_ADMET_LABELS] \
         + ["\\textbf{Mean}"]
    return "\n& ".join(cols) + " \\\\\n\\midrule"


def _latex_wide_section(title: str) -> str:
    """Format a section divider spanning all table columns."""
    n_cols = len(TDC_ADMET_NAMES) + 2
    return f"\n\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{{title}}}}} \\\\"


def _latex_wide_row(model_name: str, by_name: Dict[str, Dict], mean_col: str = "--") -> str:
    """Format benchmarks in fixed order, using dashes for missing values. The mean must be
    explicit because metrics differ."""
    cells = [f"\\quad {model_name}"]
    for name in TDC_ADMET_NAMES:
        r = by_name.get(name)
        cells.append(_latex_cell(r["mean"], r["std"]) if r is not None else "--")
    cells.append(mean_col)
    return "\n& ".join(cells) + " \\\\"


def _default_model_name(run_dir: str) -> str:
    """Derive the model label from the directory name by stripping known prefixes/suffixes."""
    name = os.path.basename(os.path.normpath(run_dir))
    if name.startswith("admet_tdc_baseline_"):
        name = name[len("admet_tdc_baseline_"):]
    for suffix in ("_biogenhpo_trainval", "_default_trainval", "_biogenhpo",
                  "_default", "_trainval"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def _find_summaries(run_dir: str) -> Dict[str, str]:
    """Map benchmark names to available summary paths."""
    if not os.path.isdir(run_dir):
        raise SystemExit(f"ERROR: not a directory: {run_dir}")
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "admet_tdc_*_summary.json"))):
        base = os.path.basename(path)
        name = base[len("admet_tdc_"):-len("_summary.json")]
        out[name] = path
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="checkpoint dir holding this model's admet_tdc_<name>_summary.json files")
    ap.add_argument("--out", default=None,
                    help="write the combined leaderboard JSON here (default: <run-dir>/admet_tdc_summary.json)")
    ap.add_argument("--check-against-tdc", action="store_true",
                    help="also load the live TDC group and assert TDC_ADMET_NAMES above "
                         "still matches sorted(group.dataset_names) -- needs PyTDC + the "
                         "downloaded group; off by default so this script stays usable offline")
    ap.add_argument("--latex", action="store_true",
                    help="also print each benchmark as a LaTeX tabular row, "
                         "'<name> & $<mean> \\err{<std>}$ \\\\', ready to paste into a table")
    ap.add_argument("--out-latex", default=None,
                    help="write the --latex rows to this file too (only with --latex; "
                         "default: <run-dir>/admet_tdc_leaderboard.tex)")
    ap.add_argument("--latex-wide", action="store_true",
                    help="print this run's results as ONE ROW of a wide table (one column "
                         "per benchmark, 'Model & $mean \\err{std}$ & ... & --  \\\\') instead "
                         "of --latex's one-row-per-benchmark layout -- for stacking multiple "
                         "models' rows under a shared header (see --latex-wide-header)")
    ap.add_argument("--latex-wide-header", action="store_true",
                    help="print the column-label header + \\midrule for the --latex-wide "
                         "layout. Independent of --run-dir/--latex-wide -- print this ONCE "
                         "for a table, not once per model")
    ap.add_argument("--latex-wide-section", default=None, metavar="TITLE",
                    help="print a \\multicolumn{...}{l}{\\textit{TITLE}} section-divider row "
                         "for the --latex-wide layout, e.g. --latex-wide-section 'Our models'")
    ap.add_argument("--model-name", default=None,
                    help="row label for --latex-wide (default: --run-dir's basename with the "
                         "admet_tdc_baseline_/_biogenhpo/_default/_trainval scaffolding stripped)")
    ap.add_argument("--conformer-eval-mode", choices=["summary", "avg_error", "ensemble"],
                    default="summary",
                    help="which test conformer mode to report: summary (default) = each "
                         "summary's own headline mean/std; avg_error / ensemble = that mode's "
                         "stored numbers (summaries without them keep their headline, flagged)")
    ap.add_argument("--mean-col", default="--",
                    help="value to print in --latex-wide's trailing Mean column (default: "
                         "'--', since averaging MAE/Spearman/ROC-AUC/PR-AUC into one number "
                         "is not a real quantity -- see _latex_wide_row's docstring)")
    args = ap.parse_args()

    found = _find_summaries(args.run_dir)
    missing = [n for n in TDC_ADMET_NAMES if n not in found]
    extra = [n for n in found if n not in TDC_ADMET_NAMES]

    if args.check_against_tdc:
        from data.datasets.admet.admet_finetune import load_admet_group
        live = sorted(load_admet_group("data/admet_group").dataset_names)
        if live != sorted(TDC_ADMET_NAMES):
            raise SystemExit(
                f"ERROR: TDC_ADMET_NAMES in this script has drifted from the live "
                f"group.\n  in TDC_ADMET_NAMES but not live: {sorted(set(TDC_ADMET_NAMES) - set(live))}\n"
                f"  in live but not TDC_ADMET_NAMES: {sorted(set(live) - set(TDC_ADMET_NAMES))}\n"
                "  Update the list here AND the hardcoded list in "
                "scripts/submit_test_fingerprints_{lightgbm,mlp}_all.sh together.")
        print("[aggregate_admet_tdc] TDC_ADMET_NAMES matches the live group: OK")

    results: List[Dict] = []
    for name in TDC_ADMET_NAMES:
        if name not in found:
            continue
        with open(found[name]) as f:
            results.append(json.load(f))

    if args.conformer_eval_mode != "summary":
        # report the other stored test conformer mode instead of each summary's headline
        swapped = []
        for r in results:
            alt = r.get(f"test_{args.conformer_eval_mode}")
            if alt:
                r = {**r, "mean": alt["mean"], "std": alt["std"],
                     "conformer_eval_mode": args.conformer_eval_mode}
            else:
                print(f"  NOTE: {r['dataset']}: no stored test_{args.conformer_eval_mode} -- "
                      f"keeping its headline ({r.get('conformer_eval_mode', 'unrecorded')})")
            swapped.append(r)
        results = swapped

    print(f"[aggregate_admet_tdc] {args.run_dir}: {len(results)}/22 benchmark(s) found")
    if missing:
        print(f"  missing (array element not finished / not submitted): {missing}")
    if extra:
        print(f"  WARNING: {len(extra)} summary file(s) not among the 22 official names "
              f"(stale run under an old benchmark name?): {extra}")

    print("\n==================== TDC ADMET leaderboard summary ====================")
    for r in results:
        # conformer mode: avg_error (single conformer) / ensemble; "-" = summary predates the field
        print(f"  {r['dataset']:<32} {r['metric']:<9} {r['mean']:.4f} +- {r['std']:.4f}"
              f"   [{r.get('conformer_eval_mode', '-')}]")
    if results:
        n_fb = sum(r.get("n_fallback_total", 0) for r in results)
        if n_fb:
            print(f"\n  {n_fb} total unembeddable-molecule fallback(s) across all "
                  "benchmarks/seeds above (see each benchmark's own JSON for where).")

    if args.latex and results:
        print("\n==================== LaTeX rows (paste into a tabular) ====================")
        # The paper defines the LaTeX err macro.
        latex_lines = [_latex_row(r["dataset"], r["mean"], r["std"]) for r in results]
        print("\n".join(latex_lines))
        latex_out = args.out_latex or os.path.join(args.run_dir, "admet_tdc_leaderboard.tex")
        with open(latex_out, "w") as f:
            f.write("\n".join(latex_lines) + "\n")
        print(f"\n[aggregate_admet_tdc] wrote {len(latex_lines)} LaTeX row(s) -> {latex_out}")
    elif args.out_latex and not args.latex:
        print("\n[aggregate_admet_tdc] NOTE: --out-latex given without --latex -- ignored "
              "(nothing to write).")

    # Header and section output do not require result rows.
    if args.latex_wide_header:
        print("\n==================== LaTeX wide-table header ====================")
        print(_latex_wide_header())
    if args.latex_wide_section is not None:
        print("\n==================== LaTeX wide-table section ====================")
        print(_latex_wide_section(args.latex_wide_section))
    if args.latex_wide:
        if not results:
            print("\n[aggregate_admet_tdc] NOTE: --latex-wide requested but 0/22 benchmarks "
                  "found in --run-dir -- nothing to build a row from.")
        else:
            model_name = args.model_name or _default_model_name(args.run_dir)
            by_name = {r["dataset"]: r for r in results}
            print("\n==================== LaTeX wide-table row ====================")
            print(_latex_wide_row(model_name, by_name, mean_col=args.mean_col))

    out_path = args.out or os.path.join(args.run_dir, "admet_tdc_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[aggregate_admet_tdc] wrote {len(results)} benchmark(s) -> {out_path}")


if __name__ == "__main__":
    main()
