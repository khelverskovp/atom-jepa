"""Fine-tune EquiformerV3 on ExpansionRX's nine regression tasks.

Preserve the temporal train/test split and take validation from the training
set's tail. Seeds vary initialization and shuffling on this fixed partition.
Apply log1p to all targets except LogD; report MAE in native units and,
optionally, on the KERMT log10 scale.

Download: python -m data.datasets.admet.download.expansionrx
Run: python -m finetuning.admet.finetune_expansionrx finetune.ckpt_path=/path/to/encoder.pt
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
from data.datasets.admet.expansionrx import (
    KERMT_LOG10_TASKS,
    LOG_TRANSFORM_TASKS,
    get_train_valid,
)
from data.datasets.admet.expansionrx import load as load_expansionrx
from data.datasets.admet.multitask_finetune import (
    MultiTaskFinetuneDataset,
    multitask_collate,
)
from finetuning.admet.metrics.report_scale import make_log10_report_scale
from finetuning.admet.training.train import train_multitask
from finetuning.admet.training.utils import (
    build_equiformer_multitask_factory,
    build_fusion_features,
    load_pretrained_encoder,
    maybe_merge_val_into_train,
    resolve_device,
)


def _log_mask(task_cols: List[str], log_transform_tasks: List[str]) -> torch.Tensor:
    """Return a boolean mask for tasks requiring a log transform."""
    wanted = set(log_transform_tasks)
    return torch.tensor([t in wanted for t in task_cols], dtype=torch.bool)


def train_seed(cfg, train_ds, valid_ds, test_ds, task_cols: List[str], seed: int,
               encoder_factory, F_in: int, encoder_config: dict, device, use_wandb, log_mask) -> Dict:
    """Train one seed on the fixed split, reusing the prepared datasets."""
    r = train_multitask(cfg, train_ds, valid_ds, test_ds, task_cols, encoder_factory, F_in,
                        multitask_collate, device, use_wandb, dataset_name="expansionrx",
                        run_tag=f"seed{seed}", seed=seed, log_mask=log_mask,
                        encoder_config=encoder_config)
    return {"seed": seed, **r}


def finetune_expansionrx(cfg: DictConfig) -> List[Dict]:
    device = torch.device(resolve_device(cfg.misc.device))
    ftc = cfg.finetune

    data = load_expansionrx(cfg.data.expansionrx_path)
    task_cols = data.tasks
    train_df, valid_df = get_train_valid(data, val_frac=float(cfg.data.get("val_frac", 0.15)))
    test_df = data.test_df
    log_transform_tasks = list(ftc.get("log_transform_tasks", LOG_TRANSFORM_TASKS))
    log_mask = _log_mask(task_cols, log_transform_tasks)
    print(f"[expansionrx] tasks={len(task_cols)} train={len(train_df)} val={len(valid_df)} "
          f"test={len(test_df)} log_transform={log_transform_tasks}", flush=True)

    all_smiles = list(dict.fromkeys(
        train_df["SMILES"].tolist() + valid_df["SMILES"].tolist() + test_df["SMILES"].tolist()
    ))
    conformers = build_or_load_conformers(
        "expansionrx", all_smiles, cfg.data.conformer_cache_dir,
        seed=int(cfg.data.get("conformer_seed", 42)),
        n_conf=int(cfg.data.get("num_conformers", 1)),
        strip_fragments=bool(cfg.data.get("strip_multi_fragment_smiles", True)),
        n_workers=int(cfg.data.get("conformer_workers", 1)),
        prune_rms=float(cfg.data.get("conformer_prune_rms", DEFAULT_PRUNE_RMS)),
    )
    features = build_fusion_features(cfg, "expansionrx", all_smiles)

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
    cutoff = eqv3_cfg.max_radius
    max_z = int(getattr(eqv3_cfg, "max_num_elements", 128))
    use_nc = ftc.get("num_conformers", None)
    use_nc = None if use_nc in (None, "all", -1, 0) else int(use_nc)

    def make_ds(df, tag, sample):
        return MultiTaskFinetuneDataset(
            df, conformers, cutoff, task_cols, max_z=max_z,
            num_conformers=use_nc, sample_conformers=sample,
            smiles_col="SMILES", tag=tag, features=features,
        )

    train_df = maybe_merge_val_into_train(cfg, train_df, valid_df, "expansionrx")
    train_ds = make_ds(train_df, "expansionrx/train", True)
    valid_ds = make_ds(valid_df, "expansionrx/valid", False)
    test_ds = make_ds(test_df, "expansionrx/test", False)

    # Add log10 reporting metrics without changing training or model selection.
    # Use training labels to set floors, then share the scale across all splits.
    rs_name = ftc.get("report_scale", "kermt")
    if rs_name:
        report_scale = make_log10_report_scale(
            task_cols, list(ftc.get("report_scale_log10_tasks", KERMT_LOG10_TASKS)),
            train_ds.labels.numpy(), name=str(rs_name),
            floor=ftc.get("report_scale_floor", None),
        )
        for ds in (train_ds, valid_ds, test_ds):
            setattr(ds, "report_scale", report_scale)
        print(f"[expansionrx] report scale -- {report_scale.describe()}", flush=True)

    # Group separate W&B runs for each seed.
    seeds = [int(s) for s in ftc.get("seeds", [1, 2, 3, 4, 5])]
    # Only rank 0 logs W&B runs and writes summaries.
    _is_main = finetuning.admet.training.distributed.is_main_process()
    use_wandb = bool(cfg.wandb.enabled) and cfg.wandb.mode != "disabled" and _is_main
    wandb_group = "expansionrx"
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
                name=f"{base}expansionrx-seed{seed}", group=wandb_group,
                job_type="seed", mode=cfg.wandb.mode, reinit=True, config=run_cfg,
            )
        r = train_seed(cfg, train_ds, valid_ds, test_ds, task_cols, seed,
                       encoder_factory, F_in, encoder_config, device, use_wandb, log_mask)
        results.append(r)
        if use_wandb:
            wandb.summary["final_val_macro"] = r["val"]["macro"]
            wandb.summary["final_test_macro"] = r["test"]["macro"]
            wandb.finish()

    print("\n==================== ExpansionRX summary (per seed) ====================")
    for r in results:
        print(f"  seed {r['seed']}: val macro-MAE={r['val']['macro']:.4f}  "
              f"test macro-MAE={r['test']['macro']:.4f}  best_epoch={r['best_epoch']}")

    summary_path = os.path.join(cfg.misc.checkpoint_dir, "expansionrx_summary.json")
    if _is_main:
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[expansionrx] wrote summary -> {summary_path}", flush=True)
    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="finetune_expansionrx")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    finetune_expansionrx(cfg)


if __name__ == "__main__":
    main()
