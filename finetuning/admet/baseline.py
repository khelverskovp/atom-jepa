"""Train feature-based MLP or LightGBM models on ADMET benchmarks.

Select ChEMBL-MT, ExpansionRX, Biogen ADME, or TDC with dataset=; model= selects
featurizers, scaling, model family, and HPO config. Dataset drivers share feature
construction and train-only scaling, while retaining their own split/seed grids.
Classical fingerprints, frozen Atom-JEPA activations, and cached Mol-JEPA embeddings use
the same reporting interface.

Run: python -m finetuning.admet.baseline dataset=biogen_adme model=morgan_rdkit_lgbm"""

import copy
import json
import os
from typing import Callable, Dict, List, Optional

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from data.datasets.admet.admet_conformers import DEFAULT_PRUNE_RMS
from data.datasets.admet.admet_finetune import get_test_and_trainval, load_admet_group
from data.datasets.admet.admet_finetune import (
    get_train_valid as get_train_valid_admet_tdc,
)
from data.datasets.admet.biogen_adme import SMILES_COL as BIOGEN_SMILES_COL
from data.datasets.admet.biogen_adme import get_split as get_split_biogen_adme
from data.datasets.admet.biogen_adme import load as load_biogen_adme
from data.datasets.admet.chembl_mt import get_train_valid as get_train_valid_chembl_mt
from data.datasets.admet.chembl_mt import load as load_chembl_mt
from data.datasets.admet.expansionrx import KERMT_LOG10_TASKS
from data.datasets.admet.expansionrx import (
    LOG_TRANSFORM_TASKS as EXPANSIONRX_LOG_TRANSFORM_TASKS,
)
from data.datasets.admet.expansionrx import (
    get_train_valid as get_train_valid_expansionrx,
)
from data.datasets.admet.expansionrx import load as load_expansionrx
from data.datasets.admet.feature_finetune import (
    MolecularFeatureDataset,
    feature_collate,
)
from data.datasets.admet.mol_features import build_or_load_features
from finetuning.admet.config import config_dict, split_sizes
from finetuning.admet.features.datasets import concat_feature_datasets
from finetuning.admet.features.jepa_activations import build_or_load_jepa_activations
from finetuning.admet.features.moljepa_features import (
    MOLJEPA_FEATURIZERS,
    MOLJEPA_REVISION,
    load_moljepa_features,
)
from finetuning.admet.metrics import resolve_task
from finetuning.admet.metrics.report_scale import make_log10_report_scale
from finetuning.admet.models.baseline_model import build_baseline_encoder
from finetuning.admet.training.gbm import train_multitask_gbm
from finetuning.admet.training.train import train_multitask
from finetuning.admet.training.utils import load_pretrained_encoder, resolve_device

_JEPA_FEATURIZERS = {"jepa_last_layer": "last", "jepa_all_layers": "all"}


def _featurizer_names(cfg) -> List[str]:
    """Return featurizer names in concatenation order, accepting a list or legacy string."""
    spec = cfg.baseline.get("featurizers", None)
    if spec is None:
        spec = cfg.baseline.get("featurizer", "morgan")
    if isinstance(spec, str):
        return [spec]
    return [str(s) for s in spec]


def _build_features(cfg, name: str, all_smiles: List[str]) -> Dict[str, Optional[np.ndarray]]:
    """Return SMILES-to-feature vectors concatenated in configured order.

    A failed component makes the molecule's vector None. Classical featurizers use
    baseline.<name> options; Atom-JEPA uses encoder/conformer settings and loads the
    encoder at most once. Mol-JEPA embeddings must be generated beforehand in their own
    environment and are read from the revision-specific cache."""
    names = _featurizer_names(cfg)

    encoder_ctx = None   # (encoder, eqv3_cfg), built lazily and shared across all JEPA featurizers
    parts: List[Dict[str, Optional[np.ndarray]]] = []   # one dict per name, IN LISTED ORDER
    for fname in names:
        if fname in _JEPA_FEATURIZERS:
            if encoder_ctx is None:
                device = torch.device(resolve_device(cfg.misc.device))
                jepa_cfg = cfg.baseline.jepa
                encoder, eqv3_cfg, _ = load_pretrained_encoder(jepa_cfg.ckpt_path, device, load_weights=True)
                encoder.eval()
                encoder_ctx = (encoder, eqv3_cfg, device, jepa_cfg)
            encoder, eqv3_cfg, device, jepa_cfg = encoder_ctx
            parts.append(build_or_load_jepa_activations(
                name, all_smiles, encoder, eqv3_cfg, cfg.data.feature_cache_dir,
                layers=_JEPA_FEATURIZERS[fname], pool=str(jepa_cfg.get("pool", "mean")),
                higher_order=bool(jepa_cfg.get("higher_order", False)),
                conformer_cache_dir=cfg.data.conformer_cache_dir,
                conformer_seed=int(cfg.data.get("conformer_seed", 42)),
                num_conformers=int(cfg.data.get("num_conformers", 1)),
                conformer_prune_rms=float(cfg.data.get("conformer_prune_rms", DEFAULT_PRUNE_RMS)),
                strip_multi_fragment_smiles=bool(cfg.data.get("strip_multi_fragment_smiles", True)),
                conformer_workers=int(cfg.data.get("conformer_workers", 1)),
                batch_size=int(jepa_cfg.get("batch_size", 64)), device=device,
            ))
        elif fname in MOLJEPA_FEATURIZERS:
            mcfg = cfg.baseline.get("moljepa", None)
            revision = str(mcfg.get("revision", MOLJEPA_REVISION)) if mcfg is not None else MOLJEPA_REVISION
            parts.append(load_moljepa_features(name, all_smiles, cfg.data.feature_cache_dir,
                                               output=MOLJEPA_FEATURIZERS[fname], revision=revision))
        else:
            # Missing featurizer options use defaults; avoid passing a plain dict to
            # OmegaConf.
            node = cfg.baseline.get(fname, None)
            kwargs = config_dict(node) if node is not None else {}
            workers = int(cfg.data.get("feature_workers", 1))
            parts.append(build_or_load_features(name, all_smiles, cfg.data.feature_cache_dir,
                                                fname, kwargs, n_workers=workers))

    if len(parts) == 1:
        return parts[0]

    out: Dict[str, Optional[np.ndarray]] = {}
    for smi in dict.fromkeys(all_smiles):
        vecs = [d.get(smi) for d in parts]
        out[smi] = None if any(v is None for v in vecs) else np.concatenate([v for v in vecs if v is not None]).astype(np.float32)
    return out


def _feature_dim(cfg, features: Dict[str, Optional[np.ndarray]]) -> int:
    """Derive width from built features and validate baseline.in_dim when explicitly set."""
    dim = next((int(v.shape[0]) for v in features.values() if v is not None), 0)
    if dim == 0:
        raise ValueError("every molecule failed featurization -- no feature width to derive")
    declared = cfg.baseline.get("in_dim", None)
    if declared is not None and int(declared) != dim:
        raise ValueError(
            f"baseline.in_dim={int(declared)} disagrees with the built features' width {dim} "
            f"(featurizers={_featurizer_names(cfg)}). Set baseline.in_dim=null to derive it."
        )
    return dim


def fit_apply_feature_scaler(cfg, train_ds, other_dss) -> None:
    """Fit on training features only, then transform all splits.

    Support none, StandardScaler, and uniform QuantileTransformer. Quantile scaling
    bounds features to [0,1] but can change LightGBM histogram bins. Reused dataset
    objects are transformed only once."""
    kind = str(cfg.baseline.get("feature_scaling", "none")).lower()
    if kind == "none":
        return
    if kind not in ("standard", "quantile"):
        raise ValueError(
            f"Unknown baseline.feature_scaling {kind!r}; expected none|standard|quantile")

    # Reused datasets must not be scaled again across seeds.
    if getattr(train_ds, "_feature_scaler_applied", False):
        return

    X_train = train_ds.feature_matrix()
    if kind == "standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_train)
    else:
        from sklearn.preprocessing import QuantileTransformer
        # Quantiles cannot exceed the number of training rows.
        n_quantiles = min(int(cfg.baseline.get("n_quantiles", 1000)), X_train.shape[0])
        scaler = QuantileTransformer(
            n_quantiles=n_quantiles, output_distribution="uniform",
            subsample=X_train.shape[0], random_state=0,
        ).fit(X_train)

    for ds in [train_ds, *other_dss]:
        ds.apply_feature_transform(scaler.transform)
        ds._feature_scaler_applied = True


def _wandb_init(cfg, use_wandb: bool, group: str, name: str, job_type: str, run_cfg_extra: Dict):
    if not use_wandb:
        return
    run_cfg = config_dict(cfg)
    if not isinstance(run_cfg, dict):
        raise TypeError("Expected the resolved run configuration to be a dictionary")
    run_cfg = {str(key): value for key, value in run_cfg.items()}
    run_cfg.update(run_cfg_extra)
    wandb.init(
        project=cfg.wandb.project, entity=cfg.wandb.entity,
        name=name, group=group, job_type=job_type, mode=cfg.wandb.mode,
        reinit=True, config=run_cfg,
    )


def _wandb_finish(use_wandb: bool, r: Dict):
    if not use_wandb:
        return
    wandb.summary["final_val_macro"] = r["val"]["macro"]
    wandb.summary["final_test_macro"] = r["test"]["macro"]
    wandb.finish()


def _family(cfg) -> str:
    """Read the explicit baseline.family setting: mlp or lgbm."""
    fam = str(cfg.baseline.get("family", "mlp")).lower()
    if fam not in ("mlp", "lgbm"):
        raise ValueError(f"Unknown baseline.family {fam!r}; expected mlp|lgbm")
    return fam


def _fit_and_report(cfg, train_ds, valid_ds, test_ds, task_cols: List[str], *,
                    dataset_name: str, run_tag: str, seed: int, log_mask,
                    encoder_ctx, device, use_wandb: bool, trial=None,
                    task_kinds: Optional[List[str]] = None,
                    classifier_eval_metric: Optional[List[str]] = None,
                    return_test_preds: bool = False) -> Dict:
    """Apply train-only feature scaling and dispatch to MLP or LightGBM training.

    encoder_ctx is (factory, F_in, config) for MLP and None for LightGBM. task_kinds
    controls binary/regression losses; classifier_eval_metric is LightGBM-specific. Both
    families return the same metric report schema.

    With train_on_val, select epochs/boosting rounds on held-out validation, then refit
    on train+validation without selection. Keep the original train-fitted feature scaler
    for both passes. return_test_preds includes raw test predictions."""
    fit_apply_feature_scaler(cfg, train_ds, [valid_ds, test_ds])
    train_on_val = bool(cfg.finetune.get("train_on_val", False))
    refit_ds = (concat_feature_datasets(train_ds, valid_ds,
                                        tag=f"{dataset_name}/train+val/{run_tag}")
                if train_on_val else None)

    if _family(cfg) == "lgbm":
        # One call: train_multitask_gbm runs both passes per task internally,
        # since the round count is per-task and only it can see best_iteration_.
        return train_multitask_gbm(
            cfg, train_ds, valid_ds, test_ds, task_cols, dataset_name=dataset_name,
            run_tag=run_tag, seed=seed, log_mask=log_mask, use_wandb=use_wandb, trial=trial,
            task_kinds=task_kinds, classifier_eval_metric=classifier_eval_metric,
            return_test_preds=return_test_preds, refit_train_ds=refit_ds,
        )

    encoder_factory, F_in, encoder_config = encoder_ctx
    if not train_on_val:
        return train_multitask(
            cfg, train_ds, valid_ds, test_ds, task_cols, encoder_factory, F_in,
            feature_collate, device, use_wandb, dataset_name=dataset_name, run_tag=run_tag,
            seed=seed, log_mask=log_mask, encoder_config=encoder_config, trial=trial,
            task_kinds=task_kinds, return_test_preds=return_test_preds,
        )

    # Select the stopping epoch without refitting, saving, or reporting Optuna steps.
    cfg1 = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg1, False)
    OmegaConf.update(cfg1, "finetune.train_on_val", False, merge=True)
    OmegaConf.update(cfg1, "finetune.save_best", False, merge=True)
    OmegaConf.set_struct(cfg1, True)
    probe = train_multitask(
        cfg1, train_ds, valid_ds, test_ds, task_cols, encoder_factory, F_in,
        feature_collate, device, False, dataset_name=dataset_name,
        run_tag=f"{run_tag}_pass1", seed=seed, log_mask=log_mask,
        encoder_config=encoder_config, trial=None, task_kinds=task_kinds,
    )
    refit_epochs = max(1, int(probe["best_epoch"]) + 1)      # best_epoch is 0-indexed
    print(f"[{dataset_name}] {run_tag} train_on_val pass 1: best epoch "
          f"{probe['best_epoch']} (val macro={probe['val'].get('macro')}) -> "
          f"refitting on train+val for {refit_epochs} epochs", flush=True)

    # Refit on train+validation without selection; validation is now in-sample.
    cfg2 = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg2, False)
    OmegaConf.update(cfg2, "finetune.epochs", refit_epochs, merge=True)
    OmegaConf.update(cfg2, "finetune.train_on_val", True, merge=True)
    OmegaConf.set_struct(cfg2, True)
    out = train_multitask(
        cfg2, refit_ds, valid_ds, test_ds, task_cols, encoder_factory, F_in,
        feature_collate, device, use_wandb, dataset_name=dataset_name, run_tag=run_tag,
        seed=seed, log_mask=log_mask, encoder_config=encoder_config, trial=trial,
        task_kinds=task_kinds, return_test_preds=return_test_preds,
    )
    out["refit_epochs"] = refit_epochs
    out["pass1_best_epoch"] = int(probe["best_epoch"])
    out["pass1_val"] = probe["val"]
    return out


def _maybe_dump_test_preds(cfg, r: Dict, test_ds, test_df, task_cols: List[str],
                           smiles_col: str, run_tag: str) -> None:
    """Save optional test predictions aligned to surviving SMILES, then remove raw arrays
    from r.

    Use keep_index because failed featurization can drop rows. Summaries remain JSON-
    safe; LightGBM predictions must be saved during the run because boosters are not
    persisted."""
    if "test_per_mol_preds" not in r:
        return
    preds = np.asarray(r.pop("test_per_mol_preds"), dtype=float)
    kept = test_df.iloc[test_ds.keep_index]
    if len(kept) != preds.shape[0]:
        raise RuntimeError(f"{run_tag}: {len(kept)} kept molecules but predictions are "
                           f"{preds.shape} -- keep_index does not describe this array.")
    path = os.path.join(cfg.misc.checkpoint_dir, f"{run_tag}_test_preds.npz")
    np.savez_compressed(path, preds=preds, labels=kept[task_cols].to_numpy(dtype=float),
                        smiles=np.array(kept[smiles_col].tolist(), dtype=object),
                        task_cols=np.array(task_cols, dtype=object), run_tag=run_tag)
    print(f"[{cfg.baseline.kind}] {run_tag}: wrote {preds.shape[0]} test prediction(s) "
          f"-> {path}", flush=True)


def _build_encoder_ctx(cfg, features, device):
    """Return (encoder_factory, F_in, encoder_config) for MLP, or None for LightGBM."""
    if _family(cfg) == "lgbm":
        return None
    return build_baseline_encoder(cfg, in_dim=_feature_dim(cfg, features), device=device)


# ChEMBL-MT
def _train_chembl_mt_fold(cfg, data, fold: int, seed_idx: int, task_cols: List[str], features,
                          encoder_ctx, device, use_wandb) -> Dict:
    ftc = cfg.finetune
    train_df, valid_df = get_train_valid_chembl_mt(data, fold)

    def make_ds(df, tag):
        return MolecularFeatureDataset(df, features, task_cols, tag=tag)

    train_ds = make_ds(train_df, f"chembl_mt_baseline/train/f{fold}_s{seed_idx}")
    valid_ds = make_ds(valid_df, f"chembl_mt_baseline/valid/f{fold}_s{seed_idx}")
    test_ds = make_ds(data.test_df, f"chembl_mt_baseline/test/f{fold}_s{seed_idx}")

    seed = int(cfg.misc.get("seed", 0)) + fold * 1000 + seed_idx
    log_transform = bool(ftc.get("log_transform", False))
    r = _fit_and_report(cfg, train_ds, valid_ds, test_ds, task_cols,
                        dataset_name="chembl_mt_baseline", run_tag=f"f{fold}_s{seed_idx}",
                        seed=seed, log_mask=log_transform, encoder_ctx=encoder_ctx,
                        device=device, use_wandb=use_wandb,
                        return_test_preds=bool(ftc.get("dump_test_preds", False)))
    _maybe_dump_test_preds(cfg, r, test_ds, data.test_df, task_cols, "smiles",
                           f"f{fold}_s{seed_idx}")
    return {"fold": fold, "seed": seed_idx, **r}


def _run_chembl_mt(cfg: DictConfig) -> List[Dict]:
    device = torch.device(resolve_device(cfg.misc.device))
    ftc = cfg.finetune
    which = str(cfg.data.get("which", "public"))
    folds = [int(f) for f in ftc.get("folds", [0, 1])]
    seeds_per_fold = [int(s) for s in ftc.get("seeds_per_fold", [0, 1])]

    data = load_chembl_mt(cfg.data.chembl_mt_path, which=which)
    task_cols = data.tasks
    print(f"[chembl_mt_baseline] which={which} tasks={len(task_cols)} folds={folds} "
          f"seeds_per_fold={seeds_per_fold}", flush=True)

    all_smiles = list(dict.fromkeys(
        data.folds[0][0]["smiles"].tolist() + data.folds[0][1]["smiles"].tolist()
        + data.folds[1][0]["smiles"].tolist() + data.folds[1][1]["smiles"].tolist()
        + data.test_df["smiles"].tolist()
    ))
    features = _build_features(cfg, f"chembl_mt_{which}", all_smiles)
    encoder_ctx = _build_encoder_ctx(cfg, features, device)

    os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)
    use_wandb = bool(cfg.wandb.enabled) and cfg.wandb.mode != "disabled"
    wandb_group = f"chembl-mt-baseline-{which}"
    base = (cfg.wandb.run_name + "-") if cfg.wandb.run_name else ""

    results = []
    for fold in folds:
        for seed_idx in seeds_per_fold:
            _wandb_init(cfg, use_wandb, wandb_group,
                       f"{base}chembl-mt-baseline-{which}-f{fold}_s{seed_idx}", "fold_seed",
                       {"fold": fold, "seed": seed_idx})
            r = _train_chembl_mt_fold(cfg, data, fold, seed_idx, task_cols, features,
                                      encoder_ctx, device, use_wandb)
            results.append(r)
            _wandb_finish(use_wandb, r)

    print("\n==================== ChEMBL-MT baseline summary (per fold/seed) "
          "====================")
    for r in results:
        print(f"  fold {r['fold']} seed {r['seed']}: val macro-MAE={r['val']['macro']:.4f}  "
              f"test macro-MAE={r['test']['macro']:.4f}  best_epoch={r['best_epoch']}")

    summary_path = os.path.join(cfg.misc.checkpoint_dir, f"chembl_mt_{which}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[chembl_mt_baseline] wrote summary -> {summary_path}", flush=True)
    return results


# ExpansionRX
def _log_mask(task_cols: List[str], log_transform_tasks: List[str]) -> torch.Tensor:
    wanted = set(log_transform_tasks)
    return torch.tensor([t in wanted for t in task_cols], dtype=torch.bool)


def _train_expansionrx_seed(cfg, train_ds, valid_ds, test_ds, task_cols: List[str], seed: int,
                            encoder_ctx, device, use_wandb, log_mask, test_df=None) -> Dict:
    r = _fit_and_report(cfg, train_ds, valid_ds, test_ds, task_cols,
                        dataset_name="expansionrx_baseline", run_tag=f"seed{seed}",
                        seed=seed, log_mask=log_mask, encoder_ctx=encoder_ctx,
                        device=device, use_wandb=use_wandb,
                        return_test_preds=bool(cfg.finetune.get("dump_test_preds", False)))
    if test_df is not None:
        _maybe_dump_test_preds(cfg, r, test_ds, test_df, task_cols, "SMILES", f"seed{seed}")
    return {"seed": seed, **r}


def _run_expansionrx(cfg: DictConfig) -> List[Dict]:
    device = torch.device(resolve_device(cfg.misc.device))
    ftc = cfg.finetune

    data = load_expansionrx(cfg.data.expansionrx_path)
    task_cols = data.tasks
    train_df, valid_df = get_train_valid_expansionrx(data, val_frac=float(cfg.data.get("val_frac", 0.15)))
    test_df = data.test_df
    log_transform_tasks = list(ftc.get("log_transform_tasks", EXPANSIONRX_LOG_TRANSFORM_TASKS))
    log_mask = _log_mask(task_cols, log_transform_tasks)
    print(f"[expansionrx_baseline] tasks={len(task_cols)} train={len(train_df)} "
          f"val={len(valid_df)} test={len(test_df)} log_transform={log_transform_tasks}", flush=True)

    all_smiles = list(dict.fromkeys(
        train_df["SMILES"].tolist() + valid_df["SMILES"].tolist() + test_df["SMILES"].tolist()
    ))
    features = _build_features(cfg, "expansionrx", all_smiles)
    encoder_ctx = _build_encoder_ctx(cfg, features, device)

    os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)

    def make_ds(df, tag):
        return MolecularFeatureDataset(df, features, task_cols, smiles_col="SMILES", tag=tag)

    train_ds = make_ds(train_df, "expansionrx_baseline/train")
    valid_ds = make_ds(valid_df, "expansionrx_baseline/valid")
    test_ds = make_ds(test_df, "expansionrx_baseline/test")

    # Secondary log10 reporting uses train-fitted floors; training is unchanged.
    rs_name = ftc.get("report_scale", "kermt")
    if rs_name:
        report_scale = make_log10_report_scale(
            task_cols, list(ftc.get("report_scale_log10_tasks", KERMT_LOG10_TASKS)),
            train_ds.labels.numpy(), name=str(rs_name),
            floor=ftc.get("report_scale_floor", None),
        )
        for ds in (train_ds, valid_ds, test_ds):
            setattr(ds, "report_scale", report_scale)
        print(f"[expansionrx_baseline] report scale -- {report_scale.describe()}",
              flush=True)

    seeds = [int(s) for s in ftc.get("seeds", [1, 2, 3, 4, 5])]
    use_wandb = bool(cfg.wandb.enabled) and cfg.wandb.mode != "disabled"
    wandb_group = "expansionrx-baseline"
    base = (cfg.wandb.run_name + "-") if cfg.wandb.run_name else ""

    results = []
    for seed in seeds:
        _wandb_init(cfg, use_wandb, wandb_group, f"{base}expansionrx-baseline-seed{seed}",
                   "seed", {"seed": seed})
        r = _train_expansionrx_seed(cfg, train_ds, valid_ds, test_ds, task_cols, seed,
                                    encoder_ctx, device, use_wandb, log_mask,
                                    test_df=test_df)
        results.append(r)
        _wandb_finish(use_wandb, r)

    print("\n==================== ExpansionRX baseline summary (per seed) ====================")
    for r in results:
        print(f"  seed {r['seed']}: val macro-MAE={r['val']['macro']:.4f}  "
              f"test macro-MAE={r['test']['macro']:.4f}  best_epoch={r['best_epoch']}")

    summary_path = os.path.join(cfg.misc.checkpoint_dir, "expansionrx_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[expansionrx_baseline] wrote summary -> {summary_path}", flush=True)
    return results


# Biogen ADME
def _train_biogen_adme_seed(cfg, data, task_cols: List[str], seed: int, features,
                            encoder_ctx, device, use_wandb, sizes,
                            dump_test_preds: bool = False) -> Dict:
    # Preserve split-mode metadata.
    train_df, val_df, test_df = get_split_biogen_adme(data, seed=seed, sizes=sizes)

    def make_ds(d, tag):
        return MolecularFeatureDataset(d, features, task_cols, smiles_col=BIOGEN_SMILES_COL, tag=tag)

    train_ds = make_ds(train_df, f"biogen_adme_baseline/train/seed{seed}")
    valid_ds = make_ds(val_df, f"biogen_adme_baseline/valid/seed{seed}")
    test_ds = make_ds(test_df, f"biogen_adme_baseline/test/seed{seed}")

    r = _fit_and_report(cfg, train_ds, valid_ds, test_ds, task_cols,
                        dataset_name="biogen_adme_baseline", run_tag=f"seed{seed}",
                        seed=seed, log_mask=None, encoder_ctx=encoder_ctx,
                        device=device, use_wandb=use_wandb,
                        return_test_preds=dump_test_preds)
    _maybe_dump_test_preds(cfg, r, test_ds, test_df, task_cols, BIOGEN_SMILES_COL,
                           f"seed{seed}")
    return {"seed": seed, **r}


def _aggregate_per_task(results: List[Dict], task_cols: List[str], split: str) -> Dict:
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


def _run_biogen_adme(cfg: DictConfig) -> Dict:
    device = torch.device(resolve_device(cfg.misc.device))
    ftc = cfg.finetune
    seeds = [int(s) for s in ftc.get("seeds", [1, 2, 3, 4, 5])]
    sizes = split_sizes(cfg.data.get("split_sizes", [0.8, 0.1, 0.1]))

    data = load_biogen_adme(
        cfg.data.biogen_adme_path, split=str(cfg.data.get("split", "scaffold")),
        cluster_dir=str(cfg.data.get("cluster_path", "data/chembl_mt")),
        cluster_fold=int(cfg.data.get("cluster_fold", 0)))
    task_cols = data.tasks
    print(f"[biogen_adme_baseline] tasks={len(task_cols)} molecules={len(data.df)} "
          f"seeds={seeds} split={data.split_mode}"
          + (f" (FIXED partition; seeds vary model init only)" if data.split_mode == "cluster"
             else f" sizes={sizes}"), flush=True)

    all_smiles = data.df[BIOGEN_SMILES_COL].tolist()
    features = _build_features(cfg, "biogen_adme", all_smiles)
    encoder_ctx = _build_encoder_ctx(cfg, features, device)

    os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)
    use_wandb = bool(cfg.wandb.enabled) and cfg.wandb.mode != "disabled"
    wandb_group = "biogen-adme-baseline"
    base = (cfg.wandb.run_name + "-") if cfg.wandb.run_name else ""

    # Save predictions during training; LightGBM boosters are not persisted.
    dump_preds = bool(ftc.get("dump_test_preds", False))

    results = []
    for seed in seeds:
        _wandb_init(cfg, use_wandb, wandb_group, f"{base}biogen-adme-baseline-seed{seed}",
                   "seed", {"seed": seed})
        r = _train_biogen_adme_seed(cfg, data, task_cols, seed, features,
                                    encoder_ctx, device, use_wandb, sizes,
                                    dump_test_preds=dump_preds)
        results.append(r)
        _wandb_finish(use_wandb, r)

    val_agg = _aggregate_per_task(results, task_cols, "val")
    test_agg = _aggregate_per_task(results, task_cols, "test")

    # Cluster splits are fixed; seeds vary model initialization.
    split_desc = ("cluster-protocol seeds (FIXED partition; seeds vary model "
                  "init only)" if data.split_mode == "cluster"
                  else f"{data.split_mode}-split seeds")
    print("\n==================== Biogen ADME baseline summary (mean+-std over "
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
    print(f"[biogen_adme_baseline] wrote summary -> {summary_path}", flush=True)
    return summary


# TDC: train each benchmark independently.
_TDC_SELECT_KEY = {
    "mae": ("macro_mae", "min"), "spearman": ("macro_spearman", "max"),
    "roc-auc": ("macro_roc_auc", "max"), "pr-auc": ("macro_pr_auc", "max"),
}


def _resolve_tdc_names(cfg, group) -> List[str]:
    """Resolve data.benchmark: null selects all TDC benchmarks, a string selects one, and a
    list selects several."""
    requested = cfg.data.get("benchmark", None)
    if requested is None:
        return list(group.dataset_names)
    if isinstance(requested, str):
        return [requested]
    return [str(x) for x in requested]


def _run_admet_tdc_one(cfg: DictConfig, group, name: str, device) -> Dict:
    """Train independent single-task models for one TDC benchmark across seeds.

    Each seed has its own train/validation split and predicts the fixed test set in
    dataframe order. Aggregate predictions through group.evaluate_many."""
    ftc = cfg.finetune
    seeds = [int(s) for s in ftc.get("seeds", [1, 2, 3, 4, 5])]

    cname, train_val_df, test_df = get_test_and_trainval(group, name)
    task = resolve_task(cname, train_val_df["Y"].to_numpy(),
                        metric_override=ftc.get("metric_override", None))
    task_cols = ["Y"]
    task_kinds = ["binary" if task.is_classification else "regression"]
    sel_key, sel_mode = _TDC_SELECT_KEY[task.metric_name]
    # LightGBM-only: matches early-stopping's own objective to the TDC metric
    # (see finetuning.admet.training.gbm.train_multitask_gbm's docstring); unused on regression.
    classifier_eval_metric = [task.metric_name]

    # Resolve the benchmark-specific selection metric before restoring config structure.
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "finetune.select",
                     [{"name": "", "key": sel_key, "mode": sel_mode}], merge=True)
    OmegaConf.set_struct(cfg, True)

    all_smiles = list(dict.fromkeys(
        train_val_df["Drug"].tolist() + test_df["Drug"].tolist()))
    features = _build_features(cfg, f"admet_tdc_{cname}", all_smiles)
    encoder_ctx = _build_encoder_ctx(cfg, features, device)

    print(f"[admet_tdc] {cname}: kind={task.kind} metric={task.metric_name} "
          f"(select {sel_key}/{sel_mode}) seeds={seeds} "
          f"train_val={len(train_val_df)} test={len(test_df)}", flush=True)

    os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)
    use_wandb = bool(cfg.wandb.enabled) and cfg.wandb.mode != "disabled"
    wandb_group = f"admet-tdc-{cname}"
    base = (cfg.wandb.run_name + "-") if cfg.wandb.run_name else ""

    per_seed = []
    for seed in seeds:
        train_df, valid_df = get_train_valid_admet_tdc(group, cname, seed)

        def make_ds(d, tag):
            return MolecularFeatureDataset(d, features, task_cols, smiles_col="Drug", tag=tag)

        # Rebuild datasets per seed so scaling fits that seed's training split.
        train_ds = make_ds(train_df, f"admet_tdc_{cname}/train/s{seed}")
        valid_ds = make_ds(valid_df, f"admet_tdc_{cname}/valid/s{seed}")
        test_ds = make_ds(test_df, f"admet_tdc_{cname}/test/s{seed}")

        _wandb_init(cfg, use_wandb, wandb_group, f"{base}admet-tdc-{cname}-s{seed}",
                   "seed", {"seed": seed, "tdc_metric": task.metric_name})
        r = _fit_and_report(cfg, train_ds, valid_ds, test_ds, task_cols,
                            dataset_name=f"admet_tdc_{cname}", run_tag=f"seed{seed}",
                            seed=seed, log_mask=None, encoder_ctx=encoder_ctx,
                            device=device, use_wandb=use_wandb,
                            task_kinds=task_kinds,
                            classifier_eval_metric=classifier_eval_metric,
                            return_test_preds=True)
        if use_wandb:
            wandb.summary[f"final_val_{sel_key}"] = r["val"].get(sel_key, float("nan"))
            wandb.summary[f"final_test_{sel_key}"] = r["test"].get(sel_key, float("nan"))
            wandb.finish()

        # Restore test dataframe order; failed rows use training mean/prevalence.
        fallback = float(np.nanmean(train_ds.labels.numpy()[:, 0]))
        full = np.full(len(test_df), fallback, dtype=np.float64)
        survivors = r["test_per_mol_preds"][:, 0]
        full[test_ds.keep_index] = survivors
        n_fallback = len(test_df) - len(test_ds.keep_index)
        if n_fallback:
            print(f"[admet_tdc] {cname} s{seed}: {n_fallback} test molecule(s) "
                  f"unembeddable -- filled with train-derived fallback "
                  f"({fallback:.4g})", flush=True)

        per_seed.append({"seed": seed, "preds": full, "n_fallback": n_fallback,
                         "val": r["val"], "test": r["test"], "best_epoch": r["best_epoch"]})

    # Save seed predictions for later TDC aggregation across at least five seeds.
    preds_dir = os.path.join(cfg.misc.checkpoint_dir, "per_seed_preds")
    os.makedirs(preds_dir, exist_ok=True)
    for ps in per_seed:
        np.savez(os.path.join(preds_dir, f"{cname}_seed{ps['seed']}.npz"),
                 preds=np.asarray(ps["preds"]), seed=ps["seed"],
                 n_fallback=ps["n_fallback"])
    print(f"[admet_tdc] {cname}: wrote {len(per_seed)} per-seed prediction "
          f"file(s) -> {preds_dir}", flush=True)

    predictions_list = [{cname: ps["preds"]} for ps in per_seed]

    # Official TDC aggregation requires at least five seeds.
    if len(seeds) < 5:
        print(f"[admet_tdc] {cname}: PARTIAL run ({len(seeds)} seed(s): {seeds}). "
              f"No official number written -- TDC needs all 5 seeds in one "
              f"evaluate_many call. Once every seed has run, reduce with:\n"
              f"    python -m finetuning.admet.reporting.aggregate_admet_tdc_seeds "
              f"--run-dir {cfg.misc.checkpoint_dir} --benchmark {cname}", flush=True)
        return {"dataset": cname, "partial": True, "seeds": list(seeds),
                "per_seed": [{k: v for k, v in ps.items() if k != "preds"}
                             for ps in per_seed]}

    if len(predictions_list) < 5:
        raise ValueError(
            f"[admet_tdc] {cname}: group.evaluate_many needs >=5 seeds for a "
            f"real leaderboard number (TDC's own protocol minimum); got "
            f"{len(predictions_list)} ({seeds}). Use finetune.seeds with >=5 "
            f"entries, or read per-seed val/test metrics from the written "
            f"summary JSON directly for a quick check.")
    official = group.evaluate_many(predictions_list)
    mean, std = float(official[cname][0]), float(official[cname][1])

    # Verify the selection metric matches the benchmark.
    one = group.evaluate(predictions_list[0])
    tdc_metric = next(iter(one[cname].keys()))
    norm = lambda s: str(s).lower().replace("_", "-").replace(" ", "-")
    if norm(tdc_metric) != norm(task.metric_name):
        print(f"[admet_tdc] WARNING: {cname}: TDC scores with {tdc_metric!r} but "
              f"selection used {task.metric_name!r}. Set "
              f"finetune.metric_override={tdc_metric!r}.", flush=True)

    total_fallback = sum(ps["n_fallback"] for ps in per_seed)
    print(f"[admet_tdc] {cname}: {task.metric_name} = {mean:.4f} +- {std:.4f} "
          f"over {len(seeds)} seeds (test n={len(test_df)}, "
          f"{total_fallback} fallback total)", flush=True)

    if use_wandb:
        _wandb_init(cfg, use_wandb, wandb_group, f"{base}admet-tdc-{cname}-summary",
                   "summary", {"dataset": cname, "n_seeds": len(seeds), "seeds": seeds})
        wandb.summary[f"{task.metric_name}_mean"] = mean
        wandb.summary[f"{task.metric_name}_std"] = std
        wandb.finish()

    result = {"dataset": cname, "metric": task.metric_name, "kind": task.kind,
             "mean": mean, "std": std, "seeds": seeds,
             "per_seed": [{k: v for k, v in ps.items() if k != "preds"} for ps in per_seed],
             "n_fallback_total": total_fallback}
    summary_path = os.path.join(cfg.misc.checkpoint_dir, f"admet_tdc_{cname}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[admet_tdc] {cname}: wrote summary -> {summary_path}", flush=True)
    return result


def _run_admet_tdc(cfg: DictConfig) -> List[Dict]:
    device = torch.device(resolve_device(cfg.misc.device))
    group = load_admet_group(cfg.data.get("tdc_path", "data/admet_group"))
    names = _resolve_tdc_names(cfg, group)
    print(f"[admet_tdc] running {len(names)} TDC ADMET benchmark(s): {names}", flush=True)

    results: List[Dict] = []
    for name in names:
        results.append(_run_admet_tdc_one(cfg, group, name, device))

    if len(names) > 1:
        # Separate benchmark jobs are combined by the summary aggregator.
        print("\n==================== TDC ADMET leaderboard summary ====================")
        for r in results:
            print(f"  {r['dataset']:<32} {r['metric']:<9} {r['mean']:.4f} +- {r['std']:.4f}")
    return results


# Dispatch by dataset.
_DATASETS: Dict[str, Callable] = {
    "chembl_mt": _run_chembl_mt,
    "expansionrx": _run_expansionrx,
    "biogen_adme": _run_biogen_adme,
    "admet_tdc": _run_admet_tdc,
}


def baseline(cfg: DictConfig):
    name = str(cfg.dataset)
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {sorted(_DATASETS)}")
    return _DATASETS[name](cfg)


@hydra.main(version_base=None, config_path="../../conf", config_name="baseline")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    baseline(cfg)


if __name__ == "__main__":
    main()
