"""Fine-tune on eight binary MoleculeNet benchmarks with scaffold seeds 0, 1, and 2.

Download data as needed. Use masked task heads and macro ROC-AUC; omit single-class
tasks from the macro and record n_scored."""

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
from data.datasets.admet.moleculenet import (
    DATASET_NAMES,
    SMILES_COL,
    get_split,
    load,
    split_diagnostics,
)
from data.datasets.admet.multitask_finetune import (
    MultiTaskFinetuneDataset,
    multitask_collate,
)
from finetuning.admet.training.train import train_multitask
from finetuning.admet.training.utils import (
    build_equiformer_multitask_factory,
    load_pretrained_encoder,
    maybe_merge_val_into_train,
    resolve_device,
)


def resolve_dataset_names(cfg) -> List[str]:
    """Resolve data.datasets: null selects all benchmarks, a string selects one, and a list
    selects several."""
    requested = cfg.data.get("datasets", None)
    if requested is None:
        return list(DATASET_NAMES)
    if isinstance(requested, str):
        return [requested]
    names = [str(x).lower() for x in requested]
    unknown = [n for n in names if n not in DATASET_NAMES]
    if unknown:
        raise ValueError(f"Unknown MoleculeNet dataset(s) {unknown}; "
                         f"expected from {DATASET_NAMES}")
    return names


def _force_classification_cfg(cfg, n_targets: int) -> None:
    """Force macro ROC-AUC maximization and disable uncertainty weighting for single-target
    datasets."""
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "finetune.select",
                     [{"name": "", "key": "macro_roc_auc", "mode": "max"}], merge=False)
    if n_targets == 1:
        OmegaConf.update(cfg, "finetune.mtl_loss", False, merge=False)
    OmegaConf.set_struct(cfg, True)


def _aggregate(results: List[Dict], task_cols: List[str], split: str) -> Dict:
    """Aggregate per-target and macro ROC-AUC across seeds, retaining n_scored."""
    out = {}
    for t in task_cols:
        vals = [r[split].get("per_task_roc_auc", {}).get(t, float("nan")) for r in results]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[t] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                      "n_splits": len(vals)}
    macro = [r[split]["macro_roc_auc"] for r in results]
    scored = [r[split].get("n_scored_roc_auc", float("nan")) for r in results]
    out["_macro"] = {"mean": float(np.mean(macro)), "std": float(np.std(macro)),
                     "n_splits": len(macro),
                     "n_scored_per_split": scored,
                     "n_targets": len(task_cols)}
    return out


def run_dataset(cfg: DictConfig, name: str, device) -> Dict:
    ftc = cfg.finetune
    data = load(name, root=str(cfg.data.get("moleculenet_root", "data/moleculenet")))
    task_cols, kinds = data.tasks, data.kinds
    n_targets = len(task_cols)
    _force_classification_cfg(cfg, n_targets)

    split_seeds = [int(s) for s in ftc.get("split_seeds", [0, 1, 2])]
    print(f"[moleculenet] {name}: {len(data.df)} molecules, {n_targets} targets, "
          f"split_seeds={split_seeds}, shared_head={bool(ftc.get('shared_head', False))}",
          flush=True)

    # Conformers over the WHOLE molecule pool, once -- independent of the split.
    all_smiles = data.df[SMILES_COL].tolist()
    conformers = build_or_load_conformers(
        f"moleculenet_{name}", all_smiles, cfg.data.conformer_cache_dir,
        seed=int(cfg.data.get("conformer_seed", 42)),
        n_conf=int(cfg.data.get("num_conformers", 10)),
        strip_fragments=bool(cfg.data.get("strip_multi_fragment_smiles", True)),
        n_workers=int(cfg.data.get("conformer_workers", 1)),
        prune_rms=float(cfg.data.get("conformer_prune_rms", DEFAULT_PRUNE_RMS)),
    )

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

    max_z = int(getattr(eqv3_cfg, "max_num_elements", 128))
    cutoff = eqv3_cfg.max_radius
    use_nc = ftc.get("num_conformers", None)
    use_nc = None if use_nc in (None, "all", -1, 0) else int(use_nc)
    # Evaluation conformer limits are independent of training limits.
    eval_nc = ftc.get("eval_num_conformers", None)
    eval_nc = use_nc if eval_nc in (None, "all") else (None if int(eval_nc) <= 0 else int(eval_nc))

    use_wandb = bool(cfg.wandb.enabled) and cfg.wandb.mode != "disabled"
    base = (cfg.wandb.run_name + "-") if cfg.wandb.run_name else ""

    results, diagnostics = [], []
    for seed in split_seeds:
        diag = split_diagnostics(data, seed)
        diagnostics.append(diag)
        print(f"[moleculenet] {name} split{seed}: "
              f"{diag['n_train']}/{diag['n_valid']}/{diag['n_test']} "
              f"scorable targets train/val/test="
              f"{diag['train_targets_scorable']}/{diag['valid_targets_scorable']}/"
              f"{diag['test_targets_scorable']} of {n_targets}", flush=True)

        train_df, val_df, test_df = get_split(data, seed)

        def make_ds(d, tag, sample, nc):
            ds = MultiTaskFinetuneDataset(
                d, conformers, cutoff, task_cols, max_z=max_z, num_conformers=nc,
                sample_conformers=sample, smiles_col=SMILES_COL, tag=tag)
            setattr(ds, "task_kinds", kinds)
            return ds

        train_df = maybe_merge_val_into_train(cfg, train_df, val_df, f"moleculenet_{name} s{seed}")
        train_ds = make_ds(train_df, f"moleculenet_{name}/train/s{seed}", True, use_nc)
        valid_ds = make_ds(val_df, f"moleculenet_{name}/valid/s{seed}", False, eval_nc)
        test_ds = make_ds(test_df, f"moleculenet_{name}/test/s{seed}", False, eval_nc)

        if use_wandb:
            run_cfg = OmegaConf.to_container(cfg, resolve=True)
            if not isinstance(run_cfg, dict):
                raise TypeError("Expected the resolved run configuration to be a dictionary")
            run_cfg = {str(key): value for key, value in run_cfg.items()}
            run_cfg["dataset"] = name
            run_cfg["split_seed"] = seed
            wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                       name=f"{base}moleculenet-{name}-s{seed}",
                       group=f"moleculenet-{name}", job_type="split",
                       mode=cfg.wandb.mode, reinit=True, config=run_cfg)

        r = train_multitask(cfg, train_ds, valid_ds, test_ds, task_cols,
                            encoder_factory, F_in, multitask_collate, device, use_wandb,
                            dataset_name=f"moleculenet_{name}", run_tag=f"s{seed}",
                            seed=seed, log_mask=None, encoder_config=encoder_config,
                            task_kinds=kinds)
        r["split_seed"] = seed
        # molecules the graph pipeline dropped, so the realized split sizes are
        # traceable (they are slightly under the nominal 80/10/10)
        r["n_used"] = {"train": train_ds.n_molecules, "valid": valid_ds.n_molecules,
                       "test": test_ds.n_molecules}
        results.append(r)

        if use_wandb:
            wandb.summary["final_val_roc_auc"] = r["val"]["macro_roc_auc"]
            wandb.summary["final_test_roc_auc"] = r["test"]["macro_roc_auc"]
            wandb.summary["n_scored_test"] = r["test"].get("n_scored_roc_auc", -1)
            wandb.finish()

    val_agg = _aggregate(results, task_cols, "val")
    test_agg = _aggregate(results, task_cols, "test")
    print(f"\n========== MoleculeNet {name} (mean+-std over {len(split_seeds)} "
          f"scaffold splits) ==========")
    print(f"  val  ROC-AUC = {val_agg['_macro']['mean']:.4f} +- {val_agg['_macro']['std']:.4f}")
    print(f"  test ROC-AUC = {test_agg['_macro']['mean']:.4f} +- {test_agg['_macro']['std']:.4f}"
          f"   (scored {test_agg['_macro']['n_scored_per_split']} of {n_targets} targets)")

    summary = {"dataset": name, "metric": "roc-auc", "n_targets": n_targets,
               "split_seeds": split_seeds, "per_split": results,
               "val_agg": val_agg, "test_agg": test_agg,
               "split_diagnostics": diagnostics}
    os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.misc.checkpoint_dir, f"moleculenet_{name}_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[moleculenet] {name}: wrote {path}", flush=True)
    return summary


def finetune_moleculenet(cfg: DictConfig) -> List[Dict]:
    device = torch.device(resolve_device(cfg.misc.device))
    names = resolve_dataset_names(cfg)
    print(f"[moleculenet] running {len(names)} dataset(s): {names}", flush=True)

    summaries = []
    for name in names:
        try:
            summaries.append(run_dataset(cfg, name, device))
        except Exception as e:                          # one bad dataset must not kill the rest
            print(f"[moleculenet] {name}: FAILED ({type(e).__name__}: {e})", flush=True)
            if len(names) == 1:
                raise
        torch.cuda.empty_cache()

    if len(summaries) > 1:
        print("\n==================== MoleculeNet summary ====================")
        for s in summaries:
            m = s["test_agg"]["_macro"]
            print(f"  {s['dataset']:<10} test ROC-AUC = {m['mean']:.4f} +- {m['std']:.4f}")
        path = os.path.join(cfg.misc.checkpoint_dir, "moleculenet_summary.json")
        with open(path, "w") as f:
            json.dump(summaries, f, indent=2, default=float)
        print(f"[moleculenet] wrote {path}", flush=True)
    return summaries


@hydra.main(version_base=None, config_path="../../../conf", config_name="finetune_moleculenet")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    finetune_moleculenet(cfg)


if __name__ == "__main__":
    main()
