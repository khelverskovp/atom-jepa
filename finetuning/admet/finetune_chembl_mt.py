"""Fine-tune EquiformerV3 on ChEMBL-MT's 25 sparse, log-scale regression tasks.

The two folds share a test set and have overlapping training partitions.
Seeds vary model initialization and shuffling within each fixed fold.
Missing labels are handled by the shared masked loss.

Download: python -m data.datasets.admet.download.chembl_mt
Run: python -m finetuning.admet.finetune_chembl_mt finetune.ckpt_path=/path/to/encoder.pt
"""

import json
import os
from typing import Dict, List

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

import finetuning.admet.training.distributed
from data.datasets.admet.admet_conformers import (
    DEFAULT_PRUNE_RMS,
    build_or_load_conformers,
)
from data.datasets.admet.chembl_mt import get_train_valid
from data.datasets.admet.chembl_mt import load as load_chembl_mt
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


def train_fold(cfg, data, fold: int, seed_idx: int, task_cols: List[str], conformers, eqv3_cfg,
               encoder_factory, F_in: int, encoder_config: dict, device, use_wandb,
               features=None) -> Dict:
    """Build one fold and train a seed, reusing cached conformers and features."""
    ftc = cfg.finetune
    if int(ftc.get("group_conformers", 0) or 0) > 0:
        raise NotImplementedError(
            "Grouped conformers are not supported by MultiTaskFinetuneDataset. "
            "Use finetune.group_conformers=0 for ChEMBL-MT fine-tuning."
        )
    cutoff = eqv3_cfg.max_radius
    max_z = int(getattr(eqv3_cfg, "max_num_elements", 128))
    use_nc = ftc.get("num_conformers", None)
    use_nc = None if use_nc in (None, "all", -1, 0) else int(use_nc)

    train_df, valid_df = get_train_valid(data, fold)

    def make_ds(df, tag, sample):
        return MultiTaskFinetuneDataset(
            df, conformers, cutoff, task_cols, max_z=max_z,
            num_conformers=use_nc, sample_conformers=sample, tag=tag, features=features,
        )

    train_df = maybe_merge_val_into_train(cfg, train_df, valid_df, f"chembl_mt f{fold}_s{seed_idx}")
    train_ds = make_ds(train_df, f"chembl_mt/train/f{fold}_s{seed_idx}", True)
    valid_ds = make_ds(valid_df, f"chembl_mt/valid/f{fold}_s{seed_idx}", False)
    test_ds = make_ds(data.test_df, f"chembl_mt/test/f{fold}_s{seed_idx}", False)

    # Derive a reproducible seed for each fold/run pair.
    seed = int(cfg.misc.get("seed", 0)) + fold * 1000 + seed_idx
    log_transform = bool(ftc.get("log_transform", False))   # uniform: all 25 tasks already log-scale
    r = train_multitask(cfg, train_ds, valid_ds, test_ds, task_cols, encoder_factory, F_in,
                        multitask_collate, device, use_wandb, dataset_name="chembl_mt",
                        run_tag=f"f{fold}_s{seed_idx}", seed=seed, log_mask=log_transform,
                        encoder_config=encoder_config)
    return {"fold": fold, "seed": seed_idx, **r}


def finetune_chembl_mt(cfg: DictConfig) -> List[Dict]:
    device = torch.device(resolve_device(cfg.misc.device))
    ftc = cfg.finetune
    which = str(cfg.data.get("which", "public"))
    folds = [int(f) for f in ftc.get("folds", [0, 1])]
    seeds_per_fold = [int(s) for s in ftc.get("seeds_per_fold", [0, 1])]

    data = load_chembl_mt(cfg.data.chembl_mt_path, which=which)
    task_cols = data.tasks
    print(f"[chembl_mt] which={which} tasks={len(task_cols)} folds={folds} "
          f"seeds_per_fold={seeds_per_fold}", flush=True)

    all_smiles = list(dict.fromkeys(
        data.folds[0][0]["smiles"].tolist() + data.folds[0][1]["smiles"].tolist()
        + data.folds[1][0]["smiles"].tolist() + data.folds[1][1]["smiles"].tolist()
        + data.test_df["smiles"].tolist()
    ))
    conformers = build_or_load_conformers(
        f"chembl_mt_{which}", all_smiles, cfg.data.conformer_cache_dir,
        seed=int(cfg.data.get("conformer_seed", 42)),
        n_conf=int(cfg.data.get("num_conformers", 1)),
        strip_fragments=bool(cfg.data.get("strip_multi_fragment_smiles", True)),
        n_workers=int(cfg.data.get("conformer_workers", 1)),
        prune_rms=float(cfg.data.get("conformer_prune_rms", DEFAULT_PRUNE_RMS)),
    )
    features = build_fusion_features(cfg, f"chembl_mt_{which}", all_smiles)

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

    # Only rank 0 logs W&B runs and writes summaries.
    _is_main = finetuning.admet.training.distributed.is_main_process()
    use_wandb = bool(cfg.wandb.enabled) and cfg.wandb.mode != "disabled" and _is_main
    wandb_group = f"chembl-mt-{which}"
    base = (cfg.wandb.run_name + "-") if cfg.wandb.run_name else ""

    results = []
    for fold in folds:
        for seed_idx in seeds_per_fold:
            if use_wandb:
                run_cfg = OmegaConf.to_container(cfg, resolve=True)
                if not isinstance(run_cfg, dict):
                    raise TypeError("Expected the resolved run configuration to be a dictionary")
                run_cfg = {str(key): value for key, value in run_cfg.items()}
                run_cfg["fold"] = fold
                run_cfg["seed"] = seed_idx
                wandb.init(
                    project=cfg.wandb.project, entity=cfg.wandb.entity,
                    name=f"{base}chembl-mt-{which}-f{fold}_s{seed_idx}", group=wandb_group,
                    job_type="fold_seed", mode=cfg.wandb.mode, reinit=True, config=run_cfg,
                )
            r = train_fold(cfg, data, fold, seed_idx, task_cols, conformers, eqv3_cfg,
                           encoder_factory, F_in, encoder_config, device, use_wandb,
                           features=features)
            results.append(r)
            if use_wandb:
                wandb.summary["final_val_macro"] = r["val"]["macro"]
                wandb.summary["final_test_macro"] = r["test"]["macro"]
                wandb.finish()

    print("\n==================== ChEMBL-MT summary (per fold/seed; the 2 folds are NOT "
          "independent seeds themselves -- see data/chembl_mt.py) ====================")
    for r in results:
        print(f"  fold {r['fold']} seed {r['seed']}: val macro-MAE={r['val']['macro']:.4f}  "
              f"test macro-MAE={r['test']['macro']:.4f}  best_epoch={r['best_epoch']}")

    summary_path = os.path.join(cfg.misc.checkpoint_dir, f"chembl_mt_{which}_summary.json")
    if _is_main:
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[chembl_mt] wrote summary -> {summary_path}", flush=True)
    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="finetune_chembl_mt")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    finetune_chembl_mt(cfg)


if __name__ == "__main__":
    main()
