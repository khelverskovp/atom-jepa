"""Shared single-task and multi-task ADMET evaluation."""

import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch

from data.datasets.admet.admet_finetune import admet_collate, featurize_mol
from finetuning.admet.metrics.classification import best_mcc_threshold, mcc_from_proba
from finetuning.admet.training.loss import LogMask, inv_target, inv_target_mt
from finetuning.admet.training.utils import move_batch


# Classification metrics.
def _mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def _spearman(y, p):
    from scipy.stats import spearmanr
    rho = np.asarray(spearmanr(np.asarray(y), np.asarray(p))[0], dtype=float).item()
    return float(rho) if np.isfinite(rho) else 0.0


def _roc_auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:          # a split with one class -> AUROC undefined
        return float("nan")
    return float(roc_auc_score(y, np.asarray(p)))


def _pr_auc(y, p):
    from sklearn.metrics import average_precision_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, np.asarray(p)))


@dataclass
class TaskSpec:
    name: str
    kind: str          # "regression" | "classification"
    metric_name: str   # "mae" | "spearman" | "roc-auc" | "pr-auc"
    mode: Literal["min", "max"]
    metric_fn: Callable

    @property
    def is_classification(self) -> bool:
        return self.kind == "classification"


def resolve_task(name: str, y_train, metric_override: Optional[str] = None) -> TaskSpec:
    """Build the TaskSpec for a benchmark.

    Metric precedence: explicit override -> TDC's recommended metric
    (tdc.metadata.admet_metrics, the SAME dict group.evaluate scores with) ->
    inferred from the labels. The inference is a last resort and is WRONG for the
    AUPRC (CYP) and Spearman (VDss/half-life/clearance) sets, so we warn loudly if
    we ever reach it -- a silent fallback would make us select/early-stop on a
    different metric than the leaderboard scores."""
    metric = metric_override
    source = "override"
    if metric is None:
        try:
            from tdc.metadata import admet_metrics
            lut = {str(k).lower(): v for k, v in admet_metrics.items()}
            metric = lut.get(name.lower())
            source = "tdc.metadata.admet_metrics"
        except Exception as e:
            print(f"[finetune] WARNING: could not read tdc.metadata.admet_metrics "
                  f"({e!r}); will infer the metric for {name!r}.", flush=True)
            metric = None
    if metric is None:
        uniq = set(np.unique(np.asarray(y_train)).astype(float).tolist())
        metric = "roc-auc" if uniq <= {0.0, 1.0} else "mae"
        source = "INFERRED"
        print(f"[finetune] WARNING: no recommended metric found for {name!r}; "
              f"INFERRED {metric!r}. If this dataset is AUPRC (CYP*) or Spearman "
              f"(VDss/half-life/clearance), set finetune.metric_override.", flush=True)

    m = str(metric).lower().replace("_", "-").replace(" ", "-")
    if m == "mae":
        return TaskSpec(name, "regression", "mae", "min", _mae)
    if m in ("spearman", "spearmanr"):
        return TaskSpec(name, "regression", "spearman", "max", _spearman)
    if m in ("roc-auc", "auroc", "auc"):
        return TaskSpec(name, "classification", "roc-auc", "max", _roc_auc)
    if m in ("pr-auc", "auprc", "prauc", "average-precision", "ap"):
        return TaskSpec(name, "classification", "pr-auc", "max", _pr_auc)
    print(f"[finetune] WARNING: unrecognized metric {metric!r} (from {source}) for "
          f"{name!r}; defaulting to MAE/regression.", flush=True)
    return TaskSpec(name, "regression", "mae", "min", _mae)


def _segment_mean(values: np.ndarray, seg_index: np.ndarray, n_seg: int) -> np.ndarray:
    """Mean of `values` grouped by `seg_index` into `n_seg` bins (empty bins -> 0)."""
    s = np.zeros(n_seg, dtype=np.float64)
    c = np.zeros(n_seg, dtype=np.float64)
    np.add.at(s, seg_index, np.asarray(values, dtype=np.float64))
    np.add.at(c, seg_index, 1.0)
    return s / np.maximum(c, 1.0)


@torch.no_grad()
def _forward_per_graph(model, loader, device, task: TaskSpec, y_mean, y_std,
                       log_transform: bool):
    """Return native predictions [G], optional molecule indices [G], and molecule labels in
    one loader pass."""
    model.eval()
    ds = loader.dataset
    per_graph = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        if task.kind == "regression":
            out = inv_target(out * y_std + y_mean, log_transform)
        else:
            out = torch.sigmoid(out)
        per_graph.append(out.detach())
    per_graph = torch.cat(per_graph).cpu().numpy()
    mol_index = getattr(ds, "mol_index", None)
    mol_index = mol_index.numpy() if mol_index is not None else None
    return per_graph, mol_index, ds.labels.numpy()


def evaluate(model, loader, device, task: TaskSpec, y_mean, y_std,
             log_transform: bool, conformer_eval_mode: str = "avg_error") -> float:
    """Compute the native task metric across conformers.

    avg_error scores flattened conformer rows; ensemble first averages predictions per
    molecule. Classification probabilities are computed before pooling."""
    per_graph, mol_index, labels = _forward_per_graph(
        model, loader, device, task, y_mean, y_std, log_transform)

    if mol_index is None:                               # sample mode: already 1/molecule
        return task.metric_fn(labels, per_graph)

    if conformer_eval_mode == "ensemble":
        per_mol = _segment_mean(per_graph, mol_index, len(labels))
        return task.metric_fn(labels, per_mol)
    if conformer_eval_mode != "avg_error":
        raise ValueError(
            f"finetune.conformer_eval_mode must be 'avg_error' or 'ensemble', "
            f"got {conformer_eval_mode!r}"
        )
    labels_per_graph = labels[mol_index]                 # repeat each label per conformer
    return task.metric_fn(labels_per_graph, per_graph)


def _dense_conformer_matrix(per_graph: np.ndarray, mol_index: np.ndarray, n_mol: int):
    """Group graph predictions by molecule into a NaN-padded [n_mol,max_n] matrix and
    counts."""
    order = np.argsort(mol_index, kind="stable")
    sorted_idx = mol_index[order]
    sorted_val = per_graph[order]
    counts = np.bincount(sorted_idx, minlength=n_mol)
    max_n = int(counts.max()) if len(counts) else 0
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    positions = np.arange(len(sorted_idx)) - starts[sorted_idx]
    P = np.full((n_mol, max_n), np.nan, dtype=np.float64)
    P[sorted_idx, positions] = sorted_val
    return P, counts


def _combo_indices(n: int, k: int, max_trials: int, rng: np.random.Generator) -> np.ndarray:
    """Enumerate k-of-n subsets when within max_trials; otherwise sample subsets without
    replacement within each subset."""
    if math.comb(n, k) <= max_trials:
        return np.array(list(itertools.combinations(range(n), k)), dtype=np.int64)
    keys = rng.random((max_trials, n))
    return np.argpartition(keys, k - 1, axis=1)[:, :k]


def ensemble_size_curve(per_graph: np.ndarray, mol_index: np.ndarray, labels: np.ndarray,
                        metric_fn: Callable, max_trials: int = 256,
                        rng: Optional[np.random.Generator] = None) -> Dict:
    """Score ensemble sizes 1..N using exact subsets or Monte Carlo sampling.

    Include only molecules with the modal conformer count; report excluded counts.
    Return per_k means/stds, n_mol_ref, and n_mol_excluded. At k=1, equivalence to
    avg_error holds for decomposable metrics such as MAE, not rank metrics."""
    n_mol = len(labels)
    if mol_index is None or n_mol == 0:                  # nothing to ensemble over
        v = float(metric_fn(labels, per_graph)) if n_mol else float("nan")
        return {"per_k": {1: {"mean": v, "std": 0.0, "n_trials": 1}},
                "n_mol_ref": n_mol, "n_mol_excluded": 0}

    rng = rng or np.random.default_rng(0)
    P, counts = _dense_conformer_matrix(np.asarray(per_graph, dtype=np.float64),
                                        mol_index, n_mol)
    n_ref = int(np.bincount(counts).argmax())            # modal conformer count
    ref_mask = counts == n_ref
    P_ref = P[ref_mask][:, :n_ref]                        # [n_ref_mol, n_ref], no NaNs
    labels_ref = np.asarray(labels, dtype=np.float64)[ref_mask]
    n_excluded = int((~ref_mask).sum())

    per_k = {}
    for k in range(1, n_ref + 1):
        idx = _combo_indices(n_ref, k, max_trials, rng)              # [T, k]
        ensembled = P_ref[:, idx].mean(axis=-1)                       # [n_ref_mol, T]
        scores = np.array([metric_fn(labels_ref, ensembled[:, t])
                           for t in range(idx.shape[0])])
        per_k[k] = {"mean": float(scores.mean()), "std": float(scores.std()),
                    "n_trials": int(idx.shape[0])}
    return {"per_k": per_k, "n_mol_ref": int(ref_mask.sum()), "n_mol_excluded": n_excluded}


@torch.no_grad()
def evaluate_full(model, loader, device, task: TaskSpec, y_mean, y_std,
                  log_transform: bool, max_trials: int = 256,
                  rng: Optional[np.random.Generator] = None) -> Dict:
    """Compute both conformer evaluation modes and the ensemble-size curve in one forward
    pass."""
    per_graph, mol_index, labels = _forward_per_graph(
        model, loader, device, task, y_mean, y_std, log_transform)

    if mol_index is None:
        v = float(task.metric_fn(labels, per_graph))
        return {"avg_error": v, "ensemble": v,
                "curve": {"per_k": {1: {"mean": v, "std": 0.0, "n_trials": 1}},
                          "n_mol_ref": len(labels), "n_mol_excluded": 0}}

    avg_error = float(task.metric_fn(labels[mol_index], per_graph))
    ensemble = float(task.metric_fn(labels, _segment_mean(per_graph, mol_index, len(labels))))
    curve = ensemble_size_curve(per_graph, mol_index, labels, task.metric_fn,
                                max_trials=max_trials, rng=rng)
    return {"avg_error": avg_error, "ensemble": ensemble, "curve": curve}


def _format_curve(curve: Dict) -> str:
    """One-line human-readable render of an ensemble_size_curve result."""
    parts = [f"k={k}:{v['mean']:.4f}±{v['std']:.4f}({v['n_trials']})"
             for k, v in sorted(curve["per_k"].items())]
    excl = curve["n_mol_excluded"]
    suffix = f" [{excl} partial-conformer mol excluded]" if excl else ""
    return " ".join(parts) + suffix


def _aggregate_curves(entries: List[Dict]) -> Dict:
    """Aggregate ensemble-size curves across seeds, reporting mean and standard deviation."""
    avg_error = np.array([e["avg_error"] for e in entries], dtype=np.float64)
    ensemble = np.array([e["ensemble"] for e in entries], dtype=np.float64)
    all_ks = sorted(set().union(*(e["curve"]["per_k"].keys() for e in entries)))
    per_k = {}
    for k in all_ks:
        vals = [e["curve"]["per_k"][k]["mean"] for e in entries if k in e["curve"]["per_k"]]
        per_k[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_seeds": len(vals)}
    return {
        "avg_error": {"mean": float(avg_error.mean()), "std": float(avg_error.std())},
        "ensemble": {"mean": float(ensemble.mean()), "std": float(ensemble.std())},
        "curve": per_k, "n_seeds": len(entries),
    }


@torch.no_grad()
def predict_dataframe(model, df, conformers, cutoff, max_z, device, task,
                      y_mean, y_std, log_transform, fallback, bs,
                      num_conformers=None, return_per_conformer: bool = False):
    """Return predictions in original dataframe order, averaging conformers per molecule.

    Use native regression values or classification probabilities and fallback values for
    failed rows. Return predictions and fallback count; optionally include a NaN-padded
    [len(df),nmax] conformer matrix."""
    model.eval()
    preds = np.full(len(df), float(fallback), dtype=np.float64)

    rows, cols, mols = [], [], []
    for i, smi in enumerate(df["Drug"].tolist()):
        conf = conformers.get(smi)
        if conf is None:
            continue
        z_np, pos_list = conf
        if int(z_np.max()) >= max_z:
            continue
        for j, pos_np in enumerate(pos_list[:num_conformers] if num_conformers else pos_list):
            rows.append(i)
            cols.append(j)
            mols.append(featurize_mol(z_np, pos_np, cutoff))

    out_all = []
    for start in range(0, len(mols), bs):
        batch = move_batch(admet_collate(mols[start:start + bs]), device)
        out = model(batch)
        if task.kind == "regression":
            out = inv_target(out * y_std + y_mean, log_transform)
        else:
            out = torch.sigmoid(out)
        out_all.append(out.detach())

    if rows:
        rows = np.asarray(rows)
        sums = np.zeros(len(df), dtype=np.float64)
        cnts = np.zeros(len(df), dtype=np.float64)
        np.add.at(sums, rows, torch.cat(out_all).cpu().numpy().astype(np.float64))
        np.add.at(cnts, rows, 1.0)
        mask = cnts > 0
        preds[mask] = sums[mask] / cnts[mask]
        n_fallback = int((~mask).sum())
    else:
        n_fallback = len(df)
    if not return_per_conformer:
        return preds, n_fallback
    per_conf = np.full((len(df), (max(cols) + 1) if cols else 1), np.nan, dtype=np.float64)
    if len(out_all):
        per_conf[np.asarray(rows), np.asarray(cols)] = torch.cat(out_all).cpu().numpy().astype(np.float64)
    return preds, n_fallback, per_conf


def single_conformer_draws(preds_conf: np.ndarray, preds: np.ndarray) -> list:
    """Build whole-dataset prediction vectors, one per conformer index.

    Cycle indices for molecules with fewer conformers; use fallback when none exist.
    Averaging these metrics differs from scoring flattened rows for rank metrics."""
    preds_conf = np.asarray(preds_conf, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    n_mol, n_max = preds_conf.shape
    counts = (~np.isnan(preds_conf)).sum(axis=1)
    has = counts > 0
    rows = np.arange(n_mol)[has]
    draws = []
    for c in range(n_max):
        p = preds.copy()
        p[has] = preds_conf[rows, c % counts[has]]
        draws.append(p)
    return draws


def tdc_leaderboard(group, name: str, y_test, per_seed_preds, per_seed_conf=None) -> Dict:
    """Aggregate both conformer modes with TDC rounding and population standard deviation.

    Predictions must follow test dataframe order. Ensemble uses evaluate_many; avg_error
    requires per-conformer predictions for every seed."""
    out: Dict[str, Any] = {"metric": None, "ensemble": None, "avg_error": None, "avg_error_per_seed": None}
    per_seed_preds = [np.asarray(p, dtype=np.float64) for p in per_seed_preds]
    if not per_seed_preds:
        return out
    try:
        one = group.evaluate({name: per_seed_preds[0]})              # {bench: {metric: value}}
        out["metric"] = next(iter(next(iter(one.values())).keys()))
    except Exception as e:
        print(f"[tdc_leaderboard] {name}: group.evaluate failed ({e!r})", flush=True)
    try:
        official = group.evaluate_many([{name: p} for p in per_seed_preds])
        if not isinstance(official, ValueError):
            v = official.get(name) or next(iter(official.values()))
            out["ensemble"] = (float(v[0]), float(v[1]))
    except Exception as e:
        print(f"[tdc_leaderboard] {name}: group.evaluate_many failed ({e!r})", flush=True)
    if (out["metric"] is not None and per_seed_conf is not None
            and len(per_seed_conf) == len(per_seed_preds)
            and all(c is not None for c in per_seed_conf)):
        from tdc import Evaluator
        ev = Evaluator(name=out["metric"])
        y = np.asarray(y_test, dtype=np.float64)
        per_seed = [round(float(np.mean([np.asarray(ev(y, d), dtype=float).item() for d in single_conformer_draws(pc, p)])), 3)
                    for p, pc in zip(per_seed_preds, per_seed_conf)]
        out["avg_error"] = (round(float(np.mean(per_seed)), 3), round(float(np.std(per_seed)), 3))
        out["avg_error_per_seed"] = per_seed
    return out


# Conformer aggregation.
def _segment_mean_2d(values: np.ndarray, seg_index: np.ndarray, n_seg: int) -> np.ndarray:
    """[G,T] values grouped by seg_index[G] into [n_seg,T] means (empty bins -> 0)."""
    n_tasks = values.shape[1]
    s = np.zeros((n_seg, n_tasks), dtype=np.float64)
    c = np.zeros(n_seg, dtype=np.float64)
    np.add.at(s, seg_index, values)
    np.add.at(c, seg_index, 1.0)
    return s / np.maximum(c, 1.0)[:, None]


def per_task_mae(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """[n_mol,T] preds/labels (labels NaN=missing) -> [T] MAE per task (NaN for
    a task with zero labeled molecules in this split)."""
    mask = ~np.isnan(labels)
    diff = np.where(mask, np.abs(preds - labels), 0.0)
    counts = mask.sum(axis=0)
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, diff.sum(axis=0) / np.maximum(counts, 1), np.nan)


def per_task_corr(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """[n_mol,T] preds/labels (labels NaN=missing) -> [T] Pearson r per task
    (NaN if <2 labeled molecules or zero variance in either side). MAE alone
    can't tell "learned real structure with a scale/offset error" apart from
    "learned nothing, MAE is just the label spread" -- correlation can."""
    n_tasks = labels.shape[1]
    out = np.full(n_tasks, np.nan)
    mask = ~np.isnan(labels)
    for t in range(n_tasks):
        idx = mask[:, t]
        if idx.sum() < 2:
            continue
        p, y = preds[idx, t], labels[idx, t]
        if np.std(p) < 1e-12 or np.std(y) < 1e-12:
            continue
        out[t] = np.corrcoef(p, y)[0, 1]
    return out


def per_task_pred_std(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """[n_mol,T] preds/labels (labels NaN=missing, used only to select which
    rows count for each task) -> [T] std of PREDICTIONS at that task's labeled
    rows. A task whose predictions have collapsed toward a constant (std -> 0)
    has effectively stopped learning that task -- the multi-task analogue of
    the embedding collapse this project already watches for in pretraining."""
    n_tasks = labels.shape[1]
    out = np.full(n_tasks, np.nan)
    mask = ~np.isnan(labels)
    for t in range(n_tasks):
        idx = mask[:, t]
        if idx.sum() > 0:
            out[t] = float(np.std(preds[idx, t]))
    return out


def report_scale_metrics(rs, preds: np.ndarray, labels: np.ndarray,
                         task_cols: Sequence[str], reg_idx: Sequence[int]) -> Dict:
    """Return secondary per-task and macro regression MAE keys without replacing native
    metrics."""
    p = rs.transform(preds)
    l = rs.transform(labels)
    maes = per_task_mae(p, l)
    per_task = {task_cols[i]: float(maes[i]) for i in reg_idx}
    with np.errstate(invalid="ignore"):
        macro = float(np.nanmean(maes[list(reg_idx)]))
    return {f"per_task_mae_{rs.name}": per_task, f"macro_mae_{rs.name}": macro}


@torch.no_grad()
def evaluate_multitask(model, loader, device, y_mean: torch.Tensor, y_std: torch.Tensor,
                       log_mask: LogMask, task_kinds: Optional[Sequence[str]] = None,
                       compute_mcc: bool = True, conformer_eval_mode: str = "avg_error",
                       compute_curve: bool = False, curve_max_trials: int = 256,
                       curve_rng: Optional[np.random.Generator] = None,
                       return_per_graph: bool = False) -> Dict:
    """Evaluate native per-task/macro metrics and model-space prediction spread.

    avg_error changes MAE aggregation only; other metrics use molecule ensembles.
    compute_curve enables ensemble-size curves. return_per_graph adds prediction, label,
    and molecule-index arrays to the report."""
    model.eval()
    ds = loader.dataset
    kinds = task_kinds if task_kinds is not None else getattr(ds, "task_kinds", None)
    binary = (np.array([k == "binary" for k in kinds], dtype=bool)
              if kinds is not None else None)

    per_graph, per_graph_std = [], []
    mean_dev, std_dev = y_mean.to(device), y_std.to(device)
    bmask = torch.as_tensor(binary, device=device) if binary is not None and binary.any() else None
    for batch in loader:
        batch = move_batch(batch, device)
        out_std = model(batch)                                       # [G,T] standardized model space
        out = inv_target_mt(out_std * std_dev + mean_dev, log_mask)
        if bmask is not None:
            # Convert logits to probabilities before averaging conformers.
            out = torch.where(bmask.unsqueeze(0), torch.sigmoid(out_std), out)
        per_graph.append(out.detach())
        per_graph_std.append(out_std.detach())
    per_graph = torch.cat(per_graph).cpu().numpy()                         # [G,T] native / probability
    per_graph_std = torch.cat(per_graph_std).cpu().numpy()                 # [G,T] standardized

    mol_index = getattr(ds, "mol_index", None)
    labels = ds.labels.numpy()                                       # [n_mol,T], NaN=missing
    if mol_index is not None:
        per_mol = _segment_mean_2d(per_graph, mol_index.numpy(), len(labels))
        per_mol_std = _segment_mean_2d(per_graph_std, mol_index.numpy(), len(labels))
    else:
        per_mol, per_mol_std = per_graph, per_graph_std

    # Both MAE modes use the same predictions and labels.
    if mol_index is None or conformer_eval_mode == "ensemble":
        maes = per_task_mae(per_mol, labels)                          # [T] native, ensembled
    elif conformer_eval_mode == "avg_error":
        # Repeat molecule labels for conformer-level scoring, preserving missing-label
        # masks.
        maes = per_task_mae(per_graph, labels[mol_index.numpy()])     # [T] native, per-conformer mean
    else:
        raise ValueError(
            f"conformer_eval_mode must be 'avg_error' or 'ensemble', got {conformer_eval_mode!r}"
        )
    corrs = per_task_corr(per_mol, labels)                            # [T]
    stds = per_task_pred_std(per_mol_std, labels)                     # [T] standardized
    per_task = {name: float(v) for name, v in zip(ds.task_cols, maes)}
    per_task_corr_d = {name: float(v) for name, v in zip(ds.task_cols, corrs)}
    per_task_std_d = {name: float(v) for name, v in zip(ds.task_cols, stds)}

    # Macro MAE includes regression tasks only.
    reg_idx = ([i for i, b in enumerate(binary) if not b] if binary is not None
               else list(range(len(maes))))
    with np.errstate(invalid="ignore"):
        macro = float(np.nanmean(maes[reg_idx])) if reg_idx else float("nan")
    report = {"per_task": per_task, "per_task_corr": per_task_corr_d,
              "per_task_pred_std": per_task_std_d, "macro": macro,
              "macro_mae": macro,
              "per_mol_preds": per_mol, "labels": labels}
    if return_per_graph:
        report["per_graph_preds"] = per_graph
        report["mol_index"] = mol_index.numpy() if mol_index is not None else None

    # Include TDC selection metrics for the relevant task types.
    if reg_idx:
        spearmans = {}
        for i in reg_idx:
            idx = ~np.isnan(labels[:, i])
            spearmans[ds.task_cols[i]] = (_spearman(labels[idx, i], per_mol[idx, i])
                                          if idx.sum() >= 2 else float("nan"))
        with np.errstate(invalid="ignore"):
            report["macro_spearman"] = float(np.nanmean(list(spearmans.values())))
        report["per_task_spearman"] = spearmans

    if binary is not None and binary.any():
        bin_idx = [i for i, b in enumerate(binary) if b]
        mccs, aucs, praucs, ths = {}, {}, {}, {}
        # Threshold search can be disabled to avoid its evaluation cost.
        for i in bin_idx:
            name = ds.task_cols[i]
            yt, yp = labels[:, i], per_mol[:, i]
            idx = ~np.isnan(yt)
            if compute_mcc:
                mccs[name] = mcc_from_proba(yt, yp, 0.5)
                ths[name] = best_mcc_threshold(yt, yp)[0]
            # Exclude single-class tasks from ROC-AUC and report n_scored.
            aucs[name] = _roc_auc(yt[idx], yp[idx])
            praucs[name] = _pr_auc(yt[idx], yp[idx])
        with np.errstate(invalid="ignore"):
            if compute_mcc:
                report["macro_mcc"] = float(np.nanmean(list(mccs.values()))) if mccs else float("nan")
            report["macro_roc_auc"] = float(np.nanmean(list(aucs.values()))) if aucs else float("nan")
            report["macro_pr_auc"] = float(np.nanmean(list(praucs.values()))) if praucs else float("nan")
        report["n_scored_roc_auc"] = int(sum(1 for v in aucs.values() if np.isfinite(v)))
        report["n_binary_tasks"] = len(bin_idx)
        if compute_mcc:
            report["per_task_mcc"] = mccs
            report["per_task_mcc_threshold"] = ths
        report["per_task_roc_auc"] = aucs
        report["per_task_pr_auc"] = praucs

    # Optional secondary reporting scale.
    rs = getattr(ds, "report_scale", None)
    if rs is not None and reg_idx:
        report.update(report_scale_metrics(rs, per_mol, labels, ds.task_cols, reg_idx))

    # Interval metrics require attached confidence bounds.
    lo = getattr(ds, "conf_low", None)
    hi = getattr(ds, "conf_high", None)
    if lo is not None and hi is not None and reg_idx:
        from finetuning.admet.metrics.interval import ma_st_rae
        macro_st, per_st = ma_st_rae(labels, per_mol, lo, hi, task_idx=reg_idx)
        report["macro_st_rae"] = macro_st
        report["per_task_st_rae"] = {ds.task_cols[i]: float(v)
                                     for i, v in zip(reg_idx, per_st)}

    if compute_curve and mol_index is not None and reg_idx:
        report["curve"] = _multitask_ensemble_curve(
            per_graph, mol_index.numpy(), labels, ds.task_cols, reg_idx,
            max_trials=curve_max_trials, rng=curve_rng)
    return report


def _multitask_ensemble_curve(per_graph: np.ndarray, mol_index: np.ndarray, labels: np.ndarray,
                              task_cols: Sequence[str], reg_idx: Sequence[int],
                              max_trials: int = 256,
                              rng: Optional[np.random.Generator] = None) -> Dict:
    """Compute NaN-aware regression MAE curves on molecules with the modal conformer count.

    Return per_task curves and a macro curve averaging per-task means at each k."""
    def _nan_mae(y, p):
        return float(np.nanmean(np.abs(np.asarray(y, dtype=np.float64)
                                       - np.asarray(p, dtype=np.float64))))

    per_task_curves = {}
    for i in reg_idx:
        per_task_curves[task_cols[i]] = ensemble_size_curve(
            per_graph[:, i], mol_index, labels[:, i], _nan_mae,
            max_trials=max_trials, rng=rng)

    all_ks = sorted(set().union(*(c["per_k"].keys() for c in per_task_curves.values())))
    macro = {}
    for k in all_ks:
        vals = [c["per_k"][k]["mean"] for c in per_task_curves.values() if k in c["per_k"]]
        macro[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_tasks": len(vals)}
    return {"per_task": per_task_curves, "macro": macro}


def format_per_task(report: Dict) -> str:
    # Print classification metrics when no regression macro is available.
    macro = report.get("macro", float("nan"))
    label = "macro"
    if not np.isfinite(macro):
        for k in ("macro_roc_auc", "macro_pr_auc", "macro_mcc"):
            if np.isfinite(report.get(k, float("nan"))):
                macro, label = report[k], k
                break
    per_task = report.get("per_task", {})
    if not any(np.isfinite(v) for v in per_task.values()) and "per_task_roc_auc" in report:
        per_task = report["per_task_roc_auc"]
    parts = [f"{k}={v:.3f}" for k, v in per_task.items() if np.isfinite(v)]
    # any secondary reporting scale's macro rides alongside the native one, so
    # the per-epoch console line shows both rulers (see finetuning/admet/metrics/report_scale.py).
    extra = "".join(f" {k[len('macro_mae_'):]}-macro={v:.4f}"
                    for k, v in sorted(report.items())
                    if k.startswith("macro_mae_") and isinstance(v, float) and np.isfinite(v))
    # cap the per-task list: ToxCast would otherwise build a 617-entry string
    # on every epoch line
    if len(parts) > 12:
        parts = parts[:12] + [f"... +{len(parts) - 12} more"]
    return f"{label}={macro:.4f}{extra}  (" + ", ".join(parts) + ")"
