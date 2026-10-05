"""Fine-tune EquiformerV3 on Biogen ADME's six log-scale regression tasks.

Scaffold mode creates a new partition per seed; cluster mode uses a fixed
partition. Report per-task and macro MAE across seeds without further log transforms.

Download: python -m data.datasets.admet.download.biogen_adme
Run: python -m finetuning.admet.finetune_biogen_adme finetune.ckpt_path=/path/to/encoder.pt
"""

import json
import os
from typing import Dict, List

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

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
from finetuning.admet.training.train import train_multitask
from finetuning.admet.training.utils import (
    build_equiformer_multitask_factory,
    build_fusion_features,
    load_pretrained_encoder,
    maybe_merge_val_into_train,
    resolve_device,
)


def train_seed(cfg, data, task_cols: List[str], seed: int, conformers, eqv3_cfg,
               encoder_factory, F_in: int, encoder_config: dict, device, use_wandb, sizes,
               features=None) -> Dict:
    """Build the selected split and train one seed, reusing cached features."""
    ftc = cfg.finetune
    if int(ftc.get("group_conformers", 0) or 0) > 0:
        raise NotImplementedError(
            "Grouped conformers are not supported by MultiTaskFinetuneDataset. "
            "Use finetune.group_conformers=0 for Biogen fine-tuning."
        )
    max_z = int(getattr(eqv3_cfg, "max_num_elements", 128))
    use_nc = ftc.get("num_conformers", None)
    use_nc = None if use_nc in (None, "all", -1, 0) else int(use_nc)
    # Evaluation cap: null/"all" inherits the training cap; <= 0 uses all cached conformers.
    eval_nc = ftc.get("eval_num_conformers", None)
    eval_nc = use_nc if eval_nc in (None, "all") else (None if int(eval_nc) <= 0 else int(eval_nc))
    cutoff = eqv3_cfg.max_radius

    train_df, val_df, test_df = get_split(data, seed=seed, sizes=sizes)

    def make_ds(d, tag, sample, nc):
        return MultiTaskFinetuneDataset(
            d, conformers, cutoff, task_cols, max_z=max_z,
            num_conformers=nc, sample_conformers=sample,
            smiles_col=SMILES_COL, tag=tag, features=features,
        )

    train_df = maybe_merge_val_into_train(cfg, train_df, val_df, f"biogen_adme seed{seed}")
    train_ds = make_ds(train_df, f"biogen_adme/train/seed{seed}", True, use_nc)
    valid_ds = make_ds(val_df, f"biogen_adme/valid/seed{seed}", False, eval_nc)
    test_ds = make_ds(test_df, f"biogen_adme/test/seed{seed}", False, eval_nc)

    r = train_multitask(cfg, train_ds, valid_ds, test_ds, task_cols, encoder_factory, F_in,
                        multitask_collate, device, use_wandb, dataset_name="biogen_adme",
                        run_tag=f"seed{seed}", seed=seed, log_mask=None,
                        encoder_config=encoder_config)
    return {"seed": seed, **r}


def _aggregate_per_task(results: List[Dict], task_cols: List[str], split: str) -> Dict:
    """Aggregate MAE across seeds, excluding non-finite per-task scores."""
    out = {}
    for t in task_cols:
        vals = [r[split]["per_task"].get(t, float("nan")) for r in results]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[t] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_seeds": len(vals)}
    macro_vals = [r[split]["macro"] for r in results]
    out["_macro"] = {"mean": float(np.mean(macro_vals)), "std": float(np.std(macro_vals)),
                     "n_seeds": len(macro_vals)}
    return out


def finetune_biogen_adme(cfg: DictConfig) -> Dict:
    device = torch.device(resolve_device(cfg.misc.device))
    ftc = cfg.finetune
    seeds = [int(s) for s in ftc.get("seeds", [1, 2, 3, 4, 5])]
    sizes = tuple(float(s) for s in cfg.data.get("split_sizes", [0.8, 0.1, 0.1]))

    data = load_biogen_adme(
        cfg.data.biogen_adme_path, split=str(cfg.data.get("split", "scaffold")),
        cluster_dir=str(cfg.data.get("cluster_path", "data/chembl_mt")),
        cluster_fold=int(cfg.data.get("cluster_fold", 0)))
    task_cols = data.tasks
    print(f"[biogen_adme] tasks={len(task_cols)} molecules={len(data.df)} seeds={seeds} "
          f"split={data.split_mode}"
          + (" (FIXED partition; seeds vary model init only)" if data.split_mode == "cluster"
             else f" sizes={sizes}"), flush=True)

    all_smiles = data.df["SMILES"].tolist()
    conformers = build_or_load_conformers(
        "biogen_adme", all_smiles, cfg.data.conformer_cache_dir,
        seed=int(cfg.data.get("conformer_seed", 42)),
        n_conf=int(cfg.data.get("num_conformers", 1)),
        strip_fragments=bool(cfg.data.get("strip_multi_fragment_smiles", True)),
        n_workers=int(cfg.data.get("conformer_workers", 1)),
        prune_rms=float(cfg.data.get("conformer_prune_rms", DEFAULT_PRUNE_RMS)),
    )
    features = build_fusion_features(cfg, "biogen_adme", all_smiles)

    ckpt_path = ftc.get("ckpt_path", None)
    if not ckpt_path:
        raise ValueError("finetune.ckpt_path must point at a pretrained encoder checkpoint.")
    _, eqv3_cfg, ckpt = load_pretrained_encoder(ckpt_path, device, load_weights=False)
    encoder_ckpt = None
    if not bool(ftc.get("train_from_scratch", False)):
        for key in ("context_encoder", "encoder", "model"):
            if key in ckpt and ckpt[key] is not None:
                encoder_ckpt = ckpt[key]
                break
        if encoder_ckpt is None:
            raise KeyError("checkpoint has no encoder weights; expected 'context_encoder'.")
    encoder_factory, F_in, encoder_config = build_equiformer_multitask_factory(
        cfg, eqv3_cfg, encoder_ckpt, device)

    os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)

    # Group separate W&B runs for each seed.
    use_wandb = bool(cfg.wandb.enabled) and cfg.wandb.mode != "disabled"
    wandb_group = "biogen-adme"
    base = (cfg.wandb.run_name + "-") if cfg.wandb.run_name else ""

    results = []
    for seed in seeds:
        if use_wandb:
            run_cfg = OmegaConf.to_container(cfg, resolve=True)
            if not isinstance(run_cfg, dict):
                raise TypeError("Expected the resolved run configuration to be a dictionary")
            run_cfg = {str(key): value for key, value in run_cfg.items()}
            run_cfg["seed"] = seed
            wandb.init(
                project=cfg.wandb.project, entity=cfg.wandb.entity,
                name=f"{base}biogen-adme-seed{seed}", group=wandb_group,
                job_type="seed", mode=cfg.wandb.mode, reinit=True, config=run_cfg,
            )
        r = train_seed(cfg, data, task_cols, seed, conformers, eqv3_cfg,
                       encoder_factory, F_in, encoder_config, device, use_wandb, sizes,
                       features=features)
        results.append(r)
        if use_wandb:
            wandb.summary["final_val_macro"] = r["val"]["macro"]
            wandb.summary["final_test_macro"] = r["test"]["macro"]
            wandb.finish()

    val_agg = _aggregate_per_task(results, task_cols, "val")
    test_agg = _aggregate_per_task(results, task_cols, "test")

    # Cluster seeds vary model initialization, not the partition.
    split_desc = ("cluster-protocol seeds (FIXED partition; seeds vary model "
                  "init only)" if data.split_mode == "cluster"
                  else f"{data.split_mode}-split seeds")
    print("\n==================== Biogen ADME summary (mean+-std over "
          f"{len(seeds)} {split_desc}) ====================")
    print(f"  val  macro-MAE = {val_agg['_macro']['mean']:.4f} +- {val_agg['_macro']['std']:.4f}")
    print(f"  test macro-MAE = {test_agg['_macro']['mean']:.4f} +- {test_agg['_macro']['std']:.4f}")
    for t in task_cols:
        if t in test_agg:
            print(f"    {t}: test MAE = {test_agg[t]['mean']:.4f} +- {test_agg[t]['std']:.4f} "
                  f"(n_seeds={test_agg[t]['n_seeds']})")

    summary = {"per_seed": results, "val_agg": val_agg, "test_agg": test_agg}
    summary_path = os.path.join(cfg.misc.checkpoint_dir, "biogen_adme_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[biogen_adme] wrote summary -> {summary_path}", flush=True)
    return summary


@hydra.main(version_base=None, config_path="../../conf", config_name="finetune_biogen_adme")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    finetune_biogen_adme(cfg)


if __name__ == "__main__":
    main()
