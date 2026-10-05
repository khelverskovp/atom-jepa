"""Optuna HPO for ADMET feature-based models; currently supports Biogen ADME.

Uses the baseline Hydra config and the selected scaffold or cluster split.
Trials optimize validation macro-MAE without evaluating held-out test data.
Features are rebuilt per trial when featurizer parameters change. Trials use
finetune.epochs; pruning can stop them early without changing the LR schedule.

Set hpo.final_full=true to evaluate the winning config with finetune.seeds.
The HPO validation score is not a final test result.

Run: python -m finetuning.admet.hpo dataset=biogen_adme model=morgan_rdkit_lgbm
"""

import copy
import json
import os
from typing import Callable, Dict, List

import hydra
import numpy as np
import optuna
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from data.datasets.admet.biogen_adme import SMILES_COL, BiogenADMEData, get_split
from data.datasets.admet.biogen_adme import load as load_biogen_adme
from data.datasets.admet.feature_finetune import MolecularFeatureDataset
from finetuning.admet.baseline import (
    _build_encoder_ctx,
    _build_features,
    _fit_and_report,
)
from finetuning.admet.config import config_dict, split_sizes
from finetuning.admet.training.utils import resolve_device


def _suggest(trial: optuna.trial.Trial, name: str, spec: Dict):
    kind = str(spec["type"])
    if kind == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]),
                                   log=bool(spec.get("log", False)), step=spec.get("step"))
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]),
                                 log=bool(spec.get("log", False)), step=int(spec.get("step", 1)))
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unknown search-space type {kind!r} for {name!r}")


def _in_domain(value, spec: Dict) -> bool:
    """Check bounds, category membership, and positivity for log distributions."""
    kind = str(spec["type"])
    if kind == "categorical":
        return value in list(spec["choices"])
    low, high = float(spec["low"]), float(spec["high"])
    if not (low <= float(value) <= high):
        return False
    return not (bool(spec.get("log", False)) and float(value) <= 0)


def _default_params_from_cfg(cfg, space: Dict[str, Dict]) -> Dict[str, object]:
    """Collect typed warm-start values, skipping missing or out-of-domain defaults."""
    params = {}
    for dotted_path, spec in space.items():
        value = OmegaConf.select(cfg, dotted_path)
        if value is None:
            continue
        kind = str(spec["type"])
        value = float(value) if kind == "float" else (int(value) if kind == "int" else value)
        if not _in_domain(value, spec):
            print(f"[hpo] warm start: skipping {dotted_path}={value!r} -- outside its own "
                  f"search-space spec {spec}; optuna will sample it instead", flush=True)
            continue
        params[dotted_path] = value
    return params


def _apply_search_space(cfg, trial: optuna.trial.Trial, space: Dict[str, Dict]):
    """Update cfg in place with trial suggestions at the configured dotted paths."""
    for dotted_path, spec in space.items():
        value = _suggest(trial, dotted_path, spec)
        OmegaConf.update(cfg, dotted_path, value, merge=True)
    return cfg


def _apply_fixed_params(cfg, fixed: Dict[str, object]):
    """Update cfg in place with parameters held constant across trials."""
    for dotted_path, value in fixed.items():
        OmegaConf.update(cfg, dotted_path, value, merge=True)
    return cfg


def _build_sampler(hpo_cfg) -> optuna.samplers.BaseSampler:
    kind = str(hpo_cfg.get("sampler", "tpe")).lower()
    seed = int(hpo_cfg.get("sampler_seed", 0))
    # Explore before switching to model-based suggestions.
    n_startup = int(hpo_cfg.get("n_startup_trials", 10))
    if kind == "tpe":
        # Model correlations between parameters.
        return optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True,
                                          n_startup_trials=n_startup)
    if kind == "gp":
        return optuna.samplers.GPSampler(seed=seed, n_startup_trials=n_startup)
    if kind == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unknown hpo.sampler {kind!r}; expected tpe|gp|random")


def _build_pruner(hpo_cfg, max_resource: int) -> optuna.pruners.BasePruner:
    kind = str(hpo_cfg.get("pruner", "hyperband")).lower()
    if kind == "none":
        return optuna.pruners.NopPruner()
    if kind == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=int(hpo_cfg.get("pruner_min_resource", 10)))
    if kind == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=int(hpo_cfg.get("pruner_min_resource", 10)),
            max_resource=max(1, max_resource),
            reduction_factor=int(hpo_cfg.get("pruner_reduction_factor", 3)),
        )
    raise ValueError(f"Unknown hpo.pruner {kind!r}; expected hyperband|median|none")


_PER_TASK_ATTR_PREFIX = "val_mae/"


def _record_per_task_val(trial: optuna.trial.Trial, per_task: Dict[str, float]):
    """Persist finite per-task validation MAEs in the trial user attributes."""
    for task, mae in per_task.items():
        if np.isfinite(mae):
            trial.set_user_attr(f"{_PER_TASK_ATTR_PREFIX}{task}", float(mae))


def _objective_biogen_adme(cfg_base: DictConfig, trial: optuna.trial.Trial,
                           data: BiogenADMEData, all_smiles: List[str], task_cols: List[str],
                           device) -> float:
    cfg = copy.deepcopy(cfg_base)
    space = config_dict(cfg.hpo.search_space)
    _apply_search_space(cfg, trial, space)

    # Disable trial checkpoints; only validation metrics are needed.
    cfg.finetune.save_best = False
    cfg.finetune.save_last = False

    # Featurizer parameters can vary by trial; reuse matching cached features.
    features = _build_features(cfg, "biogen_adme", all_smiles)

    hpo_seed = int(cfg.hpo.seed)
    sizes = split_sizes(cfg.data.get("split_sizes", [0.8, 0.1, 0.1]))
    train_df, val_df, _test_df = get_split(data, seed=hpo_seed, sizes=sizes)
    # Do not build or evaluate a test dataset during HPO.

    def make_ds(df, tag):
        return MolecularFeatureDataset(df, features, task_cols, smiles_col=SMILES_COL, tag=tag)

    train_ds = make_ds(train_df, f"biogen_adme_hpo/train/trial{trial.number}")
    val_ds = make_ds(val_df, f"biogen_adme_hpo/valid/trial{trial.number}")

    # Use validation in both evaluation slots to avoid scoring held-out test data.
    # Reuse the baseline driver's scaling and model-family dispatch.
    r = _fit_and_report(cfg, train_ds, val_ds, val_ds, task_cols,
                        dataset_name="biogen_adme_hpo", run_tag=f"trial{trial.number}",
                        seed=hpo_seed, log_mask=None,
                        encoder_ctx=_build_encoder_ctx(cfg, features, device),
                        device=device, use_wandb=False, trial=trial)
    _record_per_task_val(trial, r["val"]["per_task"])
    return float(r["val"]["macro"])


def _per_task_val_history_figure(study: optuna.Study):
    """Plot stored per-task validation MAEs; return None if no data is available."""
    trials = sorted(
        (t for t in study.trials if any(k.startswith(_PER_TASK_ATTR_PREFIX) for k in t.user_attrs)),
        key=lambda t: t.number,
    )
    if not trials:
        return None
    tasks = sorted({k[len(_PER_TASK_ATTR_PREFIX):] for t in trials for k in t.user_attrs
                    if k.startswith(_PER_TASK_ATTR_PREFIX)})

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab20" if len(tasks) > 10 else "tab10")
    for i, task in enumerate(tasks):
        xs, ys = [], []
        for t in trials:
            v = t.user_attrs.get(f"{_PER_TASK_ATTR_PREFIX}{task}")
            if v is not None:
                xs.append(t.number)
                ys.append(v)
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.2,
                    color=cmap(i % cmap.N), label=task)
    ax.set_xlabel("HPO trial")
    ax.set_ylabel("Validation MAE")
    ax.set_title("Per-task validation error across HPO trials")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def optuna_figures(study: optuna.Study):
    """Build study plots, skipping individual plots that fail.

    The caller must close the returned matplotlib figures."""
    import optuna.visualization.matplotlib as opt_mpl
    figs = {}
    for plot_name, plot_fn in [
        ("optimization_history", opt_mpl.plot_optimization_history),
        ("param_importances", opt_mpl.plot_param_importances),
        ("parallel_coordinate", opt_mpl.plot_parallel_coordinate),
    ]:
        try:
            ax = plot_fn(study)
            figs[plot_name] = ax.get_figure()
        except (ValueError, RuntimeError) as e:
            print(f"[hpo] skipped {plot_name} plot: {e}", flush=True)
    try:
        fig = _per_task_val_history_figure(study)
        if fig is not None:
            figs["per_task_val_history"] = fig
    except (ValueError, RuntimeError) as e:
        print(f"[hpo] skipped per_task_val_history plot: {e}", flush=True)
    return figs


def _log_wandb_summary(cfg: DictConfig, study: optuna.Study, study_name: str):
    """Log study metrics, trials, and plots in a single W&B summary run."""
    use_wandb = bool(cfg.wandb.enabled) and cfg.wandb.mode != "disabled"
    if not use_wandb:
        return
    wandb.init(
        project=cfg.wandb.project, entity=cfg.wandb.entity,
        name=f"{study_name}-hpo-summary", group=f"{study_name}-hpo",
        job_type="hpo_summary", mode=cfg.wandb.mode, reinit=True,
        config={"hpo": config_dict(cfg.hpo)},
    )
    wandb.summary["n_trials"] = len(study.trials)
    wandb.summary["n_pruned"] = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    wandb.summary["n_complete"] = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    wandb.summary["best_value"] = study.best_value
    for k, v in study.best_params.items():
        wandb.summary[f"best_params/{k}"] = v

    # Convert datetime columns for W&B table serialization.
    df = study.trials_dataframe()
    for col in df.columns:
        if df[col].dtype.kind in "Mm":   # datetime64 / timedelta64
            df[col] = df[col].astype(str)
    wandb.log({"trials_table": wandb.Table(dataframe=df)})

    import matplotlib.pyplot as plt
    for plot_name, fig in optuna_figures(study).items():
        wandb.log({f"plots/{plot_name}": wandb.Image(fig)})
        plt.close(fig)

    wandb.finish()


def _run_hpo_biogen_adme(cfg: DictConfig) -> optuna.Study:
    device = torch.device(resolve_device(cfg.misc.device))
    hpo_cfg = cfg.hpo

    data = load_biogen_adme(
        cfg.data.biogen_adme_path, split=str(cfg.data.get("split", "scaffold")),
        cluster_dir=str(cfg.data.get("cluster_path", "data/chembl_mt")),
        cluster_fold=int(cfg.data.get("cluster_fold", 0)))
    task_cols = data.tasks
    print(f"[hpo] biogen_adme: tasks={len(task_cols)} molecules={len(data.df)} "
          f"hpo_seed={hpo_cfg.seed} (disjoint from finetune.seeds={list(cfg.finetune.get('seeds', []))}) "
          f"n_trials={hpo_cfg.n_trials}", flush=True)

    # Build features inside each trial to respect its featurizer parameters.
    all_smiles = data.df[SMILES_COL].tolist()

    # Keep split protocols in separate studies via run_suffix.
    study_name = f"{cfg.dataset}{cfg.get('run_suffix', '')}_{cfg.baseline.kind}"
    out_dir = os.path.join("checkpoint/hpo", study_name)
    os.makedirs(out_dir, exist_ok=True)
    storage = str(hpo_cfg.get("storage") or f"sqlite:///{out_dir}/study.db")

    max_resource = int(cfg.finetune.get("epochs", 100))
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        direction=str(hpo_cfg.get("direction", "minimize")),
        sampler=_build_sampler(hpo_cfg), pruner=_build_pruner(hpo_cfg, max_resource),
    )

    # Enqueue valid config defaults once, including when resuming a study.
    if bool(hpo_cfg.get("warm_start", True)):
        space = config_dict(cfg.hpo.search_space)
        default_params = _default_params_from_cfg(cfg, space)
        if default_params:
            study.enqueue_trial(default_params, skip_if_exists=True)
            print(f"[hpo] warm start: enqueued the current default config as a trial: "
                  f"{default_params}", flush=True)

    def objective(trial: optuna.trial.Trial) -> float:
        return _objective_biogen_adme(cfg, trial, data, all_smiles, task_cols, device)

    study.optimize(objective, n_trials=int(hpo_cfg.n_trials))

    print(f"\n[hpo] study {study_name}: {len(study.trials)} trial(s), "
          f"best value={study.best_value:.4f}", flush=True)
    print(f"[hpo] best params: {study.best_params}", flush=True)

    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump({"study_name": study_name, "value": study.best_value,
                   "params": study.best_params}, f, indent=2)

    # Include model= so the suggested command loads the matching config.
    override_str = " ".join(f"{k}={v}" for k, v in study.best_params.items())
    print(f"\n[hpo] final 5-seed evaluation:\n"
          f"  python -m finetuning.admet.baseline dataset={cfg.dataset} model={cfg.baseline.kind} "
          f"{override_str}\n", flush=True)

    _log_wandb_summary(cfg, study, study_name)

    if bool(hpo_cfg.get("final_full", False)):
        print("[hpo] hpo.final_full=true -- retraining the winning config with the "
              "real finetune.seeds via finetuning.admet.baseline._run_biogen_adme ...", flush=True)
        final_cfg = copy.deepcopy(cfg)
        for k, v in study.best_params.items():
            OmegaConf.update(final_cfg, k, v, merge=True)
        from finetuning.admet.baseline import _run_biogen_adme
        _run_biogen_adme(final_cfg)

    return study


_HPO_OBJECTIVES: Dict[str, Callable] = {
    "biogen_adme": _run_hpo_biogen_adme,
}


def hpo(cfg: DictConfig):
    # Export defaults without running a study: +dump_defaults=path/to/params.json
    dump_path = cfg.get("dump_defaults", None)
    if dump_path:
        space = config_dict(cfg.hpo.search_space)
        defaults = _default_params_from_cfg(cfg, space)
        os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
        with open(dump_path, "w") as f:
            json.dump({"study_name": f"{cfg.dataset}_{cfg.baseline.kind}", "params": defaults},
                      f, indent=2)
        print(f"[hpo] wrote default reference params -> {dump_path}", flush=True)
        return None

    # Keep validation held out during HPO; refitting belongs to final evaluation.
    if bool(cfg.finetune.get("train_on_val", False)):
        raise ValueError(
            "finetune.train_on_val=true is invalid during HPO: validation must stay a "
            "clean, held-out selection signal. Drop the override -- it belongs only on "
            "the TEST-phase scripts (scripts/**/submit_test_*.sh), which refit on "
            "train+val AFTER hyperparameters have been chosen."
        )

    name = str(cfg.dataset)
    if name not in _HPO_OBJECTIVES:
        raise NotImplementedError(
            f"HPO not implemented for dataset={name!r} yet; only {sorted(_HPO_OBJECTIVES)} "
            f"today (ChEMBL-MT/ExpansionRX ship fixed splits, so a 'dedicated HPO seed' "
            f"doesn't apply the same way -- see finetuning/admet/hpo.py's module docstring). Add a "
            f"new `_objective_<dataset>`/`_run_hpo_<dataset>` pair and register it in "
            f"_HPO_OBJECTIVES to support it."
        )
    return _HPO_OBJECTIVES[name](cfg)


@hydra.main(version_base=None, config_path="../../conf", config_name="baseline")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    hpo(cfg)


if __name__ == "__main__":
    main()
