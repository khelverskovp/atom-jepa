"""Optimize Biogen encoder fine-tuning with Optuna and the shared HPO helpers.

Cache conformers once. Validation occupies the test slot during trials; train for the
configured epochs unless pruned. final_full optionally evaluates the winner."""

import copy
import json
import os
from typing import List

import hydra
import optuna
import torch
from omegaconf import DictConfig, OmegaConf

from data.datasets.admet.admet_conformers import build_or_load_conformers
from data.datasets.admet.biogen_adme import SMILES_COL, BiogenADMEData, get_split
from data.datasets.admet.biogen_adme import load as load_biogen_adme
from data.datasets.admet.multitask_finetune import (
    MultiTaskFinetuneDataset,
    multitask_collate,
)
from finetuning.admet.config import config_dict, split_sizes
from finetuning.admet.hpo import (
    _apply_fixed_params,
    _apply_search_space,
    _build_pruner,
    _build_sampler,
    _default_params_from_cfg,
    _log_wandb_summary,
    _record_per_task_val,
)
from finetuning.admet.training.train import train_multitask
from finetuning.admet.training.utils import (
    build_equiformer_multitask_factory,
    load_pretrained_encoder,
    resolve_device,
)


def _objective_jepa_biogen_adme(cfg_base: DictConfig, trial: optuna.trial.Trial,
                                data: BiogenADMEData, conformers, eqv3_cfg, encoder_ckpt,
                                task_cols: List[str], device) -> float:
    cfg = copy.deepcopy(cfg_base)
    space = config_dict(cfg.hpo.search_space)
    _apply_search_space(cfg, trial, space)
    fixed = config_dict(cfg.hpo.get("fixed_params", {}))
    _apply_fixed_params(cfg, fixed)

    # no per-trial checkpoints -- only the metric matters, and there are
    # potentially dozens of trials
    cfg.finetune.save_best = False
    cfg.finetune.save_last = False

    hpo_seed = int(cfg.hpo.seed)
    sizes = split_sizes(cfg.data.get("split_sizes", [0.8, 0.1, 0.1]))
    train_df, val_df, _test_df = get_split(data, seed=hpo_seed, sizes=sizes)
    # _test_df is intentionally unused -- see module docstring, LEAKAGE DISCIPLINE.

    encoder_factory, F_in, encoder_config = build_equiformer_multitask_factory(
        cfg, eqv3_cfg, encoder_ckpt, device)

    max_z = int(getattr(eqv3_cfg, "max_num_elements", 128))
    use_nc = cfg.finetune.get("num_conformers", None)
    use_nc = None if use_nc in (None, "all", -1, 0) else int(use_nc)
    cutoff = eqv3_cfg.max_radius   # finetune.cutoff is NOT in the search space -- fixed

    def make_ds(df, tag, sample):
        return MultiTaskFinetuneDataset(
            df, conformers, cutoff, task_cols, max_z=max_z,
            num_conformers=use_nc, sample_conformers=sample,
            smiles_col=SMILES_COL, tag=tag,
        )

    train_ds = make_ds(train_df, f"biogen_adme_jepa_hpo/train/trial{trial.number}", True)
    val_ds = make_ds(val_df, f"biogen_adme_jepa_hpo/valid/trial{trial.number}", False)

    # Use validation in the test slot during HPO.
    r = train_multitask(cfg, train_ds, val_ds, val_ds, task_cols, encoder_factory, F_in,
                        multitask_collate, device, use_wandb=False,
                        dataset_name="biogen_adme_jepa_hpo", run_tag=f"trial{trial.number}",
                        seed=hpo_seed, log_mask=None, encoder_config=encoder_config, trial=trial)
    _record_per_task_val(trial, r["val"]["per_task"])
    return float(r["val"]["macro"])


def _run_hpo_jepa_biogen_adme(cfg: DictConfig) -> optuna.Study:
    device = torch.device(resolve_device(cfg.misc.device))
    hpo_cfg = cfg.hpo

    ckpt_path = cfg.finetune.get("ckpt_path", None)
    if not ckpt_path:
        raise ValueError("finetune.ckpt_path must point at a pretrained encoder checkpoint.")

    data = load_biogen_adme(
        cfg.data.biogen_adme_path, split=str(cfg.data.get("split", "scaffold")),
        cluster_dir=str(cfg.data.get("cluster_path", "data/chembl_mt")),
        cluster_fold=int(cfg.data.get("cluster_fold", 0)))
    task_cols = data.tasks
    print(f"[hpo_finetune] biogen_adme: tasks={len(task_cols)} molecules={len(data.df)} "
          f"split={data.split_mode} hpo_seed={hpo_cfg.seed} "
          f"(disjoint from finetune.seeds={list(cfg.finetune.get('seeds', []))}) "
          f"n_trials={hpo_cfg.n_trials}", flush=True)

    all_smiles = data.df[SMILES_COL].tolist()
    conformers = build_or_load_conformers(
        "biogen_adme", all_smiles, cfg.data.conformer_cache_dir,
        seed=int(cfg.data.get("conformer_seed", 42)),
        n_conf=int(cfg.data.get("num_conformers", 1)),
        strip_fragments=bool(cfg.data.get("strip_multi_fragment_smiles", True)),
        n_workers=int(cfg.data.get("conformer_workers", 1)),
    )

    _, eqv3_cfg, ckpt = load_pretrained_encoder(ckpt_path, device, load_weights=False)
    encoder_ckpt = None
    if not bool(cfg.finetune.get("train_from_scratch", False)):
        for key in ("context_encoder", "encoder", "model"):
            if key in ckpt and ckpt[key] is not None:
                encoder_ckpt = ckpt[key]
                break
        if encoder_ckpt is None:
            raise KeyError("checkpoint has no encoder weights; expected 'context_encoder'.")

    # see finetuning/admet/hpo.py: run_suffix keeps split protocols in separate studies
    study_name = f"biogen_adme{cfg.get('run_suffix', '')}_jepa_finetune"
    out_dir = os.path.join("checkpoint/hpo", study_name)
    os.makedirs(out_dir, exist_ok=True)
    storage = str(hpo_cfg.get("storage") or f"sqlite:///{out_dir}/study.db")

    max_resource = int(cfg.finetune.get("epochs", 100))
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        direction=str(hpo_cfg.get("direction", "minimize")),
        sampler=_build_sampler(hpo_cfg), pruner=_build_pruner(hpo_cfg, max_resource),
    )

    # Warm-start defaults.
    if bool(hpo_cfg.get("warm_start", True)):
        space = config_dict(cfg.hpo.search_space)
        default_params = _default_params_from_cfg(cfg, space)
        if default_params:
            study.enqueue_trial(default_params, skip_if_exists=True)
            print(f"[hpo_finetune] warm start: enqueued the current default config as a trial: "
                  f"{default_params}", flush=True)

    def objective(trial: optuna.trial.Trial) -> float:
        return _objective_jepa_biogen_adme(cfg, trial, data, conformers, eqv3_cfg, encoder_ckpt,
                                           task_cols, device)

    study.optimize(objective, n_trials=int(hpo_cfg.n_trials))

    print(f"\n[hpo_finetune] study {study_name}: {len(study.trials)} trial(s), "
          f"best value={study.best_value:.4f}", flush=True)
    print(f"[hpo_finetune] best params: {study.best_params}", flush=True)

    # Export fixed settings alongside sampled parameters for reproducibility.
    fixed_now = config_dict(hpo_cfg.get("fixed_params", {}))
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump({"study_name": study_name, "value": study.best_value,
                   "split": str(cfg.data.get("split", "scaffold")),
                   "params": study.best_params, "fixed_params": fixed_now}, f, indent=2)

    fixed_params = config_dict(hpo_cfg.get("fixed_params", {}))
    override_str = " ".join(f"{k}={v}" for k, v in {**fixed_params, **study.best_params}.items())
    print(f"\n[hpo_finetune] final 5-seed evaluation:\n"
          f"  python -m finetuning.admet.finetune_biogen_adme finetune.ckpt_path={ckpt_path} {override_str}\n",
          flush=True)

    _log_wandb_summary(cfg, study, study_name)

    if bool(hpo_cfg.get("final_full", False)):
        print("[hpo_finetune] hpo.final_full=true -- retraining the winning config with the "
              "real finetune.seeds via finetuning.admet.finetune_biogen_adme ...", flush=True)
        final_cfg = copy.deepcopy(cfg)
        _apply_fixed_params(final_cfg, fixed_params)
        for k, v in study.best_params.items():
            OmegaConf.update(final_cfg, k, v, merge=True)
        from finetuning.admet.finetune_biogen_adme import finetune_biogen_adme
        finetune_biogen_adme(final_cfg)

    return study


def hpo_finetune(cfg: DictConfig):
    # Dump defaults without training.
    dump_path = cfg.get("dump_defaults", None)
    if dump_path:
        space = config_dict(cfg.hpo.search_space)
        defaults = _default_params_from_cfg(cfg, space)
        os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
        with open(dump_path, "w") as f:
            json.dump({"study_name": f"biogen_adme{cfg.get('run_suffix', '')}_jepa_finetune",
                       "params": defaults}, f, indent=2)
        print(f"[hpo_finetune] wrote default reference params -> {dump_path}", flush=True)
        return None
    return _run_hpo_jepa_biogen_adme(cfg)


@hydra.main(version_base=None, config_path="../../../conf", config_name="finetune_biogen_adme")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    hpo_finetune(cfg)


if __name__ == "__main__":
    main()
