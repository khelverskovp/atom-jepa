"""Tukey HSD figures and the summary table of model-validation-central's atom-jepa study,
with our own arms added next to the study's.

Reproduces 06_report.py's `tukey_<metric>.png` and `summary.csv` / `summary.md` (mean
+/- sd over folds, tagged with the Tukey grouping against the best method), plus a
paired head-to-head of each of our arms against the study's fine-tuned Atom-JEPA and
Monroe + TabPFN 3. The test and the plot are the study's own: `model_comparison.py`
is loaded from the study checkout, not reimplemented.

Our fold metrics are computed from eval_mvc's prediction CSVs exactly as
05_collect_metrics.py does, counting only folds whose diagnostics JSON (the
completion marker eval_mvc writes last) exists. The study's arms come from its
published fold_metrics.csv.

Tukey needs every method on the same folds (model_comparison.assert_balanced), so
each endpoint is restricted to the folds every method shares. An arm with fewer than
--min-folds complete folds is left out rather than shrinking everyone's folds to its.

    python -m finetuning.admet.report_mvc --dataset biogen
    python -m finetuning.admet.report_mvc --dataset biogen --min-folds 23   # a run in progress
    python -m finetuning.admet.report_mvc --dataset biogen --arms atomjepa_bio atomjepa_bio_tabpfn

Arms fine-tuned elsewhere (e.g. on another cluster) come in from their saved test
predictions (--preds) or from their wandb fold runs (--wandb, MAE and Spearman only):

    python -m finetuning.admet.report_mvc --dataset biogen --arms \\
        --preds atomjepa_bio=/path/to/predictions --out /path/to/report
    python -m finetuning.admet.report_mvc --dataset biogen --arms \\
        --wandb atomjepa_bio=GROUP/PREFIX --wandb-project ENTITY/PROJECT
"""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from finetuning.admet.eval_mvc import DATASETS, N_REPEATS, N_SPLITS, STUDY_DIR, out_dir

METRICS = ["r2", "spearman", "mae"]
HIGHER_IS_BETTER = {"r2": True, "spearman": True, "mae": False}
METRIC_LABELS = {"r2": "$R^2$", "spearman": "Spearman $\\rho$", "mae": "MAE"}

# 06_report.py compares these and atomjepa_tabpfn, which is left out here; the two
# from-scratch ChemProp arms are only carried for the site overview.
STUDY_METHODS = ["lgbm", "chemeleon", "monroe", "atomjepa_ft"]
STUDY_LABELS = {
    "lgbm": "LightGBM + Morgan",
    "chemprop_st": "ChemProp single-task",
    "chemprop": "ChemProp multi-task",
    "chemeleon": "ChemProp + CheMeleon",
    "monroe": "Monroe + TabPFN 3",
    "atomjepa_tabpfn": "Atom-JEPA + TabPFN 3",
    "atomjepa_ft": "Atom-JEPA FT",
}
# display names for our arms; any other arm is labelled from its name
ARM_LABELS = {
    "atomjepa_bio": "Atom-JEPA-Bio FT",
    "atomjepa_bio_tabpfn": "Atom-JEPA-Bio + TabPFN 3",
}
# each of our arms is compared, paired by fold, against these
HEAD_TO_HEAD = ["atomjepa_ft", "monroe"]

# 06_report.py's layout: three panels to a row, Tukey panels grow with the method count
N_COLS = 3
PANEL_SIZE = (17 / 3, 14 / 3)
TUKEY_PAD_IN = PANEL_SIZE[1] - 0.5 * 7
TUKEY_ROW_H_IN = 0.5
SUPTITLE_IN = 0.5


def load_study_module(name: str):
    spec = importlib.util.spec_from_file_location(f"mvc_{name}", STUDY_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mc = load_study_module("model_comparison")


def arm_label(arm: str) -> str:
    return STUDY_LABELS.get(arm) or ARM_LABELS.get(arm) or "Atom-JEPA " + arm.removeprefix("atomjepa_").replace("_", " ")


def our_fold_metrics(dataset: str, arms: list[str]) -> pd.DataFrame:
    """05_collect_metrics.py fold_metrics, for the folds eval_mvc finished."""
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    out = out_dir(dataset)
    targets = DATASETS[dataset]["targets"]
    rows = []
    for arm in arms:
        done = 0
        for repeat in range(N_REPEATS):
            for fold in range(N_SPLITS):
                diag = out / "diagnostics" / f"{arm}_r{repeat}_f{fold}.json"
                if not diag.exists():
                    continue
                saved = json.loads(diag.read_text())
                if (saved.get("arm"), saved.get("repeat"), saved.get("fold")) != (arm, repeat, fold):
                    raise SystemExit(f"{diag} does not describe {arm} r{repeat} f{fold}")
                done += 1
                for t in targets:
                    g = pd.read_csv(out / "predictions" / arm / f"{t}_r{repeat}_f{fold}.csv")
                    rows.append({"endpoint": t, "method": arm, "repeat": repeat, "fold": fold,
                                 "n": len(g),
                                 "r2": r2_score(g["y_true"], g["y_pred"]),
                                 "spearman": spearmanr(g["y_true"], g["y_pred"]).statistic,
                                 "mae": mean_absolute_error(g["y_true"], g["y_pred"])})
        print(f"{arm}: {done} of {N_REPEATS * N_SPLITS} folds complete")
    return pd.DataFrame(rows)


def wandb_fold_metrics(arm: str, project: str, group: str, prefix: str,
                       targets: list[str]) -> pd.DataFrame:
    """Per-fold test MAE and Spearman of an arm run elsewhere, from wandb.

    The fold runs are named <prefix>-r<repeat>-f<fold> in <group>, and log the test
    metrics of the best-validation epoch to their summary once, at the end. R^2 is not
    logged, so it comes back NaN. Where a fold was run more than once (a short smoke
    test next to the real run), the run with the most fine-tuning epochs is kept.
    """
    import re

    import wandb

    pattern = re.compile(rf"^{re.escape(prefix)}-r(\d+)-f(\d+)$")
    best = {}
    for run in wandb.Api(timeout=60).runs(project, filters={"group": group}, per_page=200):
        m = pattern.match(run.name)
        if not m or run.state != "finished":
            continue
        key = (int(m[1]), int(m[2]))
        epochs = run.config.get("finetune", {}).get("epochs", 0)
        if key not in best or epochs > best[key][0]:
            best[key] = (epochs, run)
    rows = []
    for (repeat, fold), (epochs, run) in sorted(best.items()):
        for t in targets:
            rows.append({"endpoint": t, "method": arm, "repeat": repeat, "fold": fold, "n": None,
                         "r2": float("nan"),
                         "spearman": run.summary.get(f"test_spearman/{t}", float("nan")),
                         "mae": run.summary.get(f"test_mae/{t}", float("nan"))})
    epochs = sorted({e for e, _ in best.values()})
    print(f"{arm}: {len(best)} folds from wandb {group}/{prefix} (fine-tuning epochs {epochs})")
    return pd.DataFrame(rows)


def dir_fold_metrics(arm: str, root: Path, targets: list[str]) -> pd.DataFrame:
    """Fold metrics of an arm run elsewhere from its saved test predictions,
    root/r<repeat>-f<fold>/predictions.csv: one row per test molecule, the measured
    <endpoint> columns next to prediction_<endpoint>. Computed as 05_collect_metrics.py
    does, per endpoint over the molecules that have a measurement."""
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    rows = []
    for repeat in range(N_REPEATS):
        for fold in range(N_SPLITS):
            path = root / f"r{repeat}-f{fold}" / "predictions.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for t in targets:
                g = df[df[t].notna()]
                y, p = g[t].to_numpy(), g[f"prediction_{t}"].to_numpy()
                rows.append({"endpoint": t, "method": arm, "repeat": repeat, "fold": fold,
                             "n": len(g), "r2": r2_score(y, p),
                             "spearman": spearmanr(y, p).statistic,
                             "mae": mean_absolute_error(y, p)})
    out = pd.DataFrame(rows)
    n = len(out[["repeat", "fold"]].drop_duplicates()) if len(out) else 0
    print(f"{arm}: {n} folds of test predictions from {root}")
    return out


def shared_folds(metrics: pd.DataFrame, methods: list[str], targets: list[str]) -> pd.DataFrame:
    """06_report.py usable_metrics: per endpoint, only the folds every method has."""
    keep = []
    for endpoint in targets:
        sub = metrics[metrics["endpoint"] == endpoint]
        by_method = {m: set(g["fold_id"]) for m, g in sub.groupby("method")}
        missing = set(methods) - set(by_method)
        if missing:
            raise SystemExit(f"{endpoint}: no folds for {sorted(missing)}")
        shared = set.intersection(*(by_method[m] for m in methods))
        keep.append(sub[sub["fold_id"].isin(shared)])
    return pd.concat(keep, ignore_index=True)


def save(fig, out: Path, name: str, reserve_title: bool = False) -> None:
    top = 1 - SUPTITLE_IN / fig.get_size_inches()[1] if reserve_title else 1
    fig.tight_layout(rect=(0, 0, 1, top))
    for suffix in ("png", "svg"):
        fig.savefig(out / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.png")


def tukey_figures(metrics, mets, methods, targets, labels, out, dataset_label, n_folds) -> None:
    """06_report.py tukey_figures: blue best, grey indistinguishable, red worse."""
    rows = (len(targets) + N_COLS - 1) // N_COLS
    panel_w = PANEL_SIZE[0]
    panel_h = TUKEY_PAD_IN + TUKEY_ROW_H_IN * len(methods)
    for metric in mets:
        fig, axes = plt.subplots(rows, N_COLS, squeeze=False)
        axes = axes.flatten()
        for ax in axes[len(targets):]:
            ax.set_visible(False)
        xlims = []
        for ax, endpoint in zip(axes, targets):
            sub = metrics[metrics["endpoint"] == endpoint]
            mc.make_tukey_plot(sub, metric, higher_is_better=HIGHER_IS_BETTER[metric], ax=ax,
                               title=endpoint, xlabel=METRIC_LABELS[metric])
            ax.set_yticklabels([labels.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()])
            xlims.append(ax.get_xlim())
        lo, hi = min(x[0] for x in xlims), max(x[1] for x in xlims)
        bottom_row = len(targets) - N_COLS
        for i, ax in enumerate(axes[:len(targets)]):
            ax.set_xlim(lo, hi)
            if i % N_COLS:
                ax.set_yticklabels([])
            if i < bottom_row:
                ax.set_xticklabels([])
                ax.set_xlabel("")
        fig.suptitle(f"{dataset_label}, {n_folds} folds: Tukey HSD on {METRIC_LABELS[metric]} -- "
                     "blue: best, grey: indistinguishable, red: significantly worse", fontsize=14)
        # statsmodels' plot_simultaneous resizes the figure on every panel, so size it last
        fig.set_size_inches(panel_w * N_COLS, panel_h * rows)
        save(fig, out, f"tukey_{metric}", reserve_title=True)


def summary_table(metrics, mets, methods, targets, labels, out, n_folds) -> pd.DataFrame:
    """06_report.py summary_table: mean +/- sd, tagged with the Tukey grouping."""
    rows = []
    for endpoint in targets:
        sub = metrics[metrics["endpoint"] == endpoint]
        for metric in mets:
            groups = mc.tukey_groups(sub, metric, HIGHER_IS_BETTER[metric])
            stats = sub.groupby("method")[metric].agg(["mean", "std"])
            for method in methods:
                rows.append({"endpoint": endpoint, "metric": metric, "method": method,
                             "mean": stats.loc[method, "mean"], "sd": stats.loc[method, "std"],
                             "tukey_group": groups[method],
                             "n_folds": int((sub["method"] == method).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)

    tag = {"best": " (best)", "equivalent": " (=)", "worse": ""}
    cells = summary.assign(cell=[f"{m:.3f} +/- {s:.3f}{tag[g]}" for m, s, g in
                                 zip(summary["mean"], summary["sd"], summary["tukey_group"])])
    lines = []
    for metric in mets:
        wide = (cells[cells["metric"] == metric]
                .pivot(index="endpoint", columns="method", values="cell")
                .reindex(index=targets, columns=methods).rename(columns=labels))
        lines.append(f"\n## {metric}\n")
        lines.append(wide.to_markdown())
    (out / "summary.md").write_text(
        "# 5x5 cross-validated comparison\n"
        f"\nMean +/- sd over {n_folds} folds. `(best)` marks the best mean; `(=)` a method "
        "Tukey HSD cannot distinguish from it at alpha = 0.05; unmarked methods are "
        "significantly worse.\n" + "\n".join(lines) + "\n")
    print(f"wrote {out / 'summary.csv'} and summary.md")
    return summary


def best_counts(summary, mets, methods, labels, out) -> pd.DataFrame:
    """Per method, over every endpoint x metric: best alone, tied for best, or worse.

    Best alone: the best mean, and Tukey separates it from every other method. Tied
    for best: the best mean with at least one method Tukey cannot separate from it,
    or one of those methods. Worse: significantly worse than the best.
    """
    def outcome(g: pd.DataFrame) -> pd.Series:
        tied = (g["tukey_group"] == "equivalent").any()
        return g["tukey_group"].map({"best": "tied for best" if tied else "best alone",
                                     "equivalent": "tied for best", "worse": "worse"})

    cols = ["best alone", "tied for best", "worse"]
    per_cell = summary.assign(outcome=summary.groupby(["endpoint", "metric"], group_keys=False)
                              .apply(outcome))

    def count(df: pd.DataFrame) -> pd.DataFrame:
        return (pd.crosstab(df["method"], df["outcome"]).reindex(index=methods, columns=cols)
                .fillna(0).astype(int).rename(index=labels))

    table = count(per_cell)
    table.to_csv(out / "best_counts.csv")
    n = summary.groupby(["endpoint", "metric"]).ngroups
    lines = ["# Tukey outcome counts\n",
             f"Over {n} endpoint x metric combinations. Best alone: the best mean and "
             "significantly ahead of every other method. Tied for best: the best mean with a "
             "method Tukey HSD cannot separate from it, or one of those methods. Worse: "
             "significantly worse than the best.\n", table.to_markdown()]
    for metric in mets:
        lines += [f"\n## {metric}\n", count(per_cell[per_cell["metric"] == metric]).to_markdown()]
    (out / "best_counts.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'best_counts.csv'} and best_counts.md")
    return table


def head_to_head(metrics, mets, arms, targets, out) -> pd.DataFrame:
    """06_report.py head_to_head: mean difference, fold win rate, paired t-test p."""
    rows = []
    for arm in arms:
        for left in HEAD_TO_HEAD:
            for endpoint in targets:
                for metric in mets:
                    sub = metrics[(metrics["endpoint"] == endpoint) & metrics["method"].isin([left, arm])]
                    diff, pval = mc.paired_test(sub, metric, left, arm)
                    wide = sub.pivot(index="fold_id", columns="method", values=metric)
                    wins = wide[arm] > wide[left] if HIGHER_IS_BETTER[metric] else wide[arm] < wide[left]
                    rows.append({"endpoint": endpoint, "metric": metric, "left": left, "right": arm,
                                 "mean_diff": diff, "right_wins": int(wins.sum()),
                                 "n_folds": len(wide), "p_value": pval})
    table = pd.DataFrame(rows)
    table.to_csv(out / "head_to_head.csv", index=False)
    print(f"wrote {out / 'head_to_head.csv'}")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--arms", nargs="*",
                    help="our local arms (default: every arm with predictions; give none for none)")
    ap.add_argument("--wandb", nargs="+", default=[], metavar="ARM=GROUP/PREFIX",
                    help="arms read from wandb fold runs <PREFIX>-r<i>-f<j> in GROUP; "
                         "MAE and Spearman only")
    ap.add_argument("--wandb-project", metavar="ENTITY/PROJECT", help="wandb project of --wandb")
    ap.add_argument("--preds", nargs="+", default=[], metavar="ARM=DIR",
                    help="arms read from saved test predictions DIR/r<i>-f<j>/predictions.csv")
    ap.add_argument("--study-methods", nargs="+", default=STUDY_METHODS,
                    choices=list(STUDY_LABELS), help="the study's arms to compare against")
    ap.add_argument("--min-folds", type=int, default=N_REPEATS * N_SPLITS,
                    help="leave out arms with fewer complete folds; lower it for a run in progress")
    ap.add_argument("--out", type=Path, help="default: <MVC_ROOT>/runs/<dataset>/report")
    a = ap.parse_args()
    if a.wandb and not a.wandb_project:
        ap.error("--wandb needs --wandb-project")

    targets = DATASETS[a.dataset]["targets"]
    pred_root = out_dir(a.dataset) / "predictions"
    arms = a.arms if a.arms is not None else sorted(p.name for p in pred_root.iterdir() if p.is_dir())
    ours = [our_fold_metrics(a.dataset, arms)]
    for spec in a.wandb:
        arm, _, rest = spec.partition("=")
        group, _, prefix = rest.partition("/")
        ours.append(wandb_fold_metrics(arm, a.wandb_project, group, prefix, targets))
        arms.append(arm)
    for spec in a.preds:
        arm, _, root = spec.partition("=")
        ours.append(dir_fold_metrics(arm, Path(root), targets))
        arms.append(arm)
    ours = pd.concat([o for o in ours if len(o)], ignore_index=True)
    counts = ours.groupby("method")[["repeat", "fold"]].apply(lambda g: len(g.drop_duplicates()))
    dropped = [arm for arm in arms if counts.get(arm, 0) < a.min_folds]
    if dropped:
        print(f"leaving out (fewer than {a.min_folds} folds): {', '.join(dropped)}")
    arms = [arm for arm in arms if arm not in dropped]
    if not arms:
        raise SystemExit("no arm has enough complete folds")

    study = pd.read_csv(STUDY_DIR / "results" / a.dataset / "fold_metrics.csv")
    methods = list(a.study_methods) + arms
    metrics = pd.concat([study[study["method"].isin(a.study_methods)], ours[ours["method"].isin(arms)]],
                        ignore_index=True)
    metrics["fold_id"] = "r" + metrics["repeat"].astype(str) + "f" + metrics["fold"].astype(str)
    metrics = shared_folds(metrics, methods, targets)
    n_folds = int(metrics.groupby(["endpoint", "method"]).size().min())
    mets = [m for m in METRICS if metrics[m].notna().all()]
    if len(mets) < len(METRICS):
        print(f"not every arm has {sorted(set(METRICS) - set(mets))}; reporting {mets} only")
    print(f"reporting on {len(targets)} endpoints x {len(methods)} methods x {n_folds} shared folds")

    out = a.out or out_dir(a.dataset) / "report"
    out.mkdir(parents=True, exist_ok=True)
    metrics.drop(columns="fold_id").to_csv(out / "fold_metrics.csv", index=False)
    labels = {m: arm_label(m) for m in methods}
    tukey_figures(metrics, mets, methods, targets, labels, out, a.dataset, n_folds)
    summary = summary_table(metrics, mets, methods, targets, labels, out, n_folds)
    counts = best_counts(summary, mets, methods, labels, out)
    h2h = head_to_head(metrics, mets, arms, targets, out)

    mae = (metrics.groupby(["endpoint", "method"])["mae"].mean().unstack("method")
           .reindex(index=targets, columns=methods).rename(columns=labels))
    print(f"\nmean MAE over {n_folds} folds:\n{mae.round(3).to_string()}")
    print(f"\nTukey outcomes over every endpoint x metric:\n{counts.to_string()}")
    print("\nhead to head (right minus left; right_wins of n_folds):")
    print(h2h[h2h["metric"] == "mae"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
