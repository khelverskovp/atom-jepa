"""Run one validation-only fine-tuning grid cell and write its metrics as JSON.

Hydra overrides select parameters and hpo.seed. Validation occupies the test slot; no
held-out test is scored. Rerun the selected configuration for final testing."""

import json
import os

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig

import finetuning.admet.training.distributed
from data.datasets.admet.admet_conformers import (
    DEFAULT_PRUNE_RMS,
    build_or_load_conformers,
)
from data.datasets.admet.biogen_adme import SMILES_COL, get_split
from data.datasets.admet.biogen_adme import load as load_biogen_adme
from data.datasets.admet.multitask_finetune import (
    MultiTaskFinetuneDataset,
    multitask_collate,
)
from finetuning.admet.config import config_dict, split_sizes
from finetuning.admet.training.train import train_multitask
from finetuning.admet.training.utils import (
    build_equiformer_multitask_factory,
    build_fusion_features,
    load_pretrained_encoder,
    resolve_device,
)


def run_cell(cfg: DictConfig) -> dict:
    device = torch.device(resolve_device(cfg.misc.device))

    ckpt_path = cfg.finetune.get("ckpt_path", None)
    if not ckpt_path:
        raise ValueError("finetune.ckpt_path must point at a pretrained encoder checkpoint.")

    # cell identity: whatever the array script varied, recorded verbatim so the
    # output file is self-describing rather than relying on its own filename.
    cell = config_dict(cfg.get("cell", {}) or {})
    cell_id = str(cfg.get("cell_id", "cell"))

    # no per-cell checkpoints -- only the val metric matters here
    cfg.finetune.save_best = False
    cfg.finetune.save_last = False

    data = load_biogen_adme(
        cfg.data.biogen_adme_path, split=str(cfg.data.get("split", "scaffold")),
        cluster_dir=str(cfg.data.get("cluster_path", "data/chembl_mt")),
        cluster_fold=int(cfg.data.get("cluster_fold", 0)))
    task_cols = data.tasks

    # SAME seed the HPO study selects on -- see LEAKAGE DISCIPLINE above.
    val_seed = int(cfg.hpo.seed)
    sizes = split_sizes(cfg.data.get("split_sizes", [0.8, 0.1, 0.1]))
    train_df, val_df, _test_df = get_split(data, seed=val_seed, sizes=sizes)
    del _test_df                      # never read; deleted so it cannot be

    print(f"[val_grid] {cell_id}: split={data.split_mode} val_seed={val_seed} "
          f"train={len(train_df)} val={len(val_df)} tasks={len(task_cols)} "
          f"cell={cell}", flush=True)

    all_smiles = data.df[SMILES_COL].tolist()
    conformers = build_or_load_conformers(
        "biogen_adme", all_smiles, cfg.data.conformer_cache_dir,
        seed=int(cfg.data.get("conformer_seed", 42)),
        n_conf=int(cfg.data.get("num_conformers", 1)),
        strip_fragments=bool(cfg.data.get("strip_multi_fragment_smiles", True)),
        n_workers=int(cfg.data.get("conformer_workers", 1)),
        prune_rms=float(cfg.data.get("conformer_prune_rms", DEFAULT_PRUNE_RMS)),
    )
    # Build fusion features so sampled fusion settings affect the model.
    features = build_fusion_features(cfg, "biogen_adme_grid", all_smiles)

    _, eqv3_cfg, ckpt = load_pretrained_encoder(ckpt_path, device, load_weights=False)
    encoder_ckpt = None
    if not bool(cfg.finetune.get("train_from_scratch", False)):
        for key in ("context_encoder", "encoder", "model"):
            if key in ckpt and ckpt[key] is not None:
                encoder_ckpt = ckpt[key]
                break
        if encoder_ckpt is None:
            raise KeyError("checkpoint has no encoder weights; expected 'context_encoder'.")

    encoder_factory, F_in, encoder_config = build_equiformer_multitask_factory(
        cfg, eqv3_cfg, encoder_ckpt, device)

    max_z = int(getattr(eqv3_cfg, "max_num_elements", 128))
    use_nc = cfg.finetune.get("num_conformers", None)
    use_nc = None if use_nc in (None, "all", -1, 0) else int(use_nc)
    # Keep evaluation conformer limits independent of training limits.
    eval_nc = cfg.finetune.get("eval_num_conformers", None)
    eval_nc = use_nc if eval_nc in (None, "all") else (None if int(eval_nc) <= 0 else int(eval_nc))
    cutoff = eqv3_cfg.max_radius

    def make_ds(df, tag, sample, nc):
        return MultiTaskFinetuneDataset(
            df, conformers, cutoff, task_cols, max_z=max_z,
            num_conformers=nc, sample_conformers=sample,
            smiles_col=SMILES_COL, tag=tag, features=features,
        )

    train_ds = make_ds(train_df, f"biogen_adme_grid/train/{cell_id}", True, use_nc)
    val_ds = make_ds(val_df, f"biogen_adme_grid/valid/{cell_id}", False, eval_nc)

    # Use one W&B run per grid cell, with swept settings at the top level.
    use_wandb = (bool(cfg.wandb.get("enabled", False))
                 and str(cfg.wandb.get("mode", "online")) != "disabled"
                 and finetuning.admet.training.distributed.is_main_process())
    if use_wandb:
        run_cfg = config_dict(cfg)
        if not isinstance(run_cfg, dict):
            raise TypeError("Expected the resolved run configuration to be a dictionary")
        run_cfg = {str(key): value for key, value in run_cfg.items()}
        run_cfg["cell_id"] = cell_id
        run_cfg.update({f"cell.{k}": v for k, v in cell.items()})
        base = cfg.wandb.get("run_name", None) or "biogen_adme_grid"
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                   name=f"{base}-{cell_id}",
                   group=str(cfg.get("grid_group", base)),
                   job_type="grid_cell", mode=cfg.wandb.mode,
                   reinit=True, config=run_cfg)

    # Optionally retain validation predictions.
    save_val_preds = bool(cfg.get("save_val_preds", False))

    # val_ds again in the test_ds slot -- see LEAKAGE DISCIPLINE above.
    r = train_multitask(cfg, train_ds, val_ds, val_ds, task_cols, encoder_factory, F_in,
                        multitask_collate, device, use_wandb=use_wandb,
                        dataset_name="biogen_adme_grid", run_tag=cell_id,
                        seed=val_seed, log_mask=None, encoder_config=encoder_config,
                        return_val_preds=save_val_preds)

    out = {
        "cell_id": cell_id,
        "cell": cell,
        "val_macro": float(r["val"]["macro"]),
        "val_per_task": r["val"]["per_task"],
        "val_per_task_corr": r["val"]["per_task_corr"],
        "val_macro_spearman": r["val"].get("macro_spearman"),
        # Record the ensemble-size curve.
        "val_curve": r["val"].get("curve"),
        "best_epoch": r["best_epoch"],
        "num_conformers_train": use_nc,
        "num_conformers_eval": eval_nc,
        # False -> best_epoch is simply the LAST epoch: nothing was selected on val
        # (finetune.select_best=false), so val_* above is not selection-biased.
        "select_best": bool(cfg.finetune.get("select_best", True)),
        "val_seed": val_seed,
        "split": data.split_mode,
        # Omit duplicate test metrics from validation-only runs.
    }
    grid_dir = str(cfg.misc.checkpoint_dir)
    os.makedirs(grid_dir, exist_ok=True)
    path = os.path.join(grid_dir, f"{cell_id}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[val_grid] {cell_id}: val_macro={out['val_macro']:.4f} "
          f"(best_epoch={out['best_epoch']}) -> {path}", flush=True)

    if save_val_preds:
        # Map graph rows to cached molecule/conformer order, retaining surviving rows.
        mol_index = r["val_mol_index"]
        npz_path = os.path.join(grid_dir, f"{cell_id}_val_preds.npz")
        np.savez_compressed(
            npz_path,
            smiles=np.array([val_df[SMILES_COL].iloc[i] for i in val_ds.keep_index]),
            task_cols=np.array(task_cols),
            labels=r["val_labels"],                         # [n_mol,T] native, NaN = missing
            per_mol_preds=r["val_per_mol_preds"],           # [n_mol,T] all-conformer ensemble
            per_graph_preds=r["val_per_graph_preds"],       # [G,T] one row per (mol, conformer)
            mol_index=(mol_index if mol_index is not None
                       else np.arange(len(val_ds.keep_index))),
            conf_index=np.array([c for (_, c) in val_ds._index] if mol_index is not None
                                else np.zeros(len(val_ds.keep_index), dtype=int)),
        )
        print(f"[val_grid] {cell_id}: per-conformer val predictions -> {npz_path}", flush=True)

    if use_wandb:
        # Flat summary scalars so the grid is sortable in wandb's runs table
        # without opening each run.
        wandb.summary["cell_val_macro"] = out["val_macro"]
        wandb.summary["cell_best_epoch"] = out["best_epoch"]
        for k, v in cell.items():
            wandb.summary[f"cell_{k}"] = v
        wandb.finish()
    return out


@hydra.main(version_base=None, config_path="../../../conf", config_name="finetune_biogen_adme")
def main(cfg: DictConfig):
    run_cell(cfg)


if __name__ == "__main__":
    main()
