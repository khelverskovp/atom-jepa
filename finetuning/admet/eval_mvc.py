"""Test-set evaluation on the protocol of model-validation-central's atom-jepa study.

github.com/patwalters/model-validation-central/tree/main/studies/atom-jepa fine-tunes
Atom-JEPA on ExpansionRx and Biogen ADME over 25 folds and scores every fold on a fixed
test set. This reimplements its three steps for our own checkpoints, reading the source
repositories only as data:

  prepare  the molecule table and the 25 folds, as expansion-ml-comparison's
           01_make_folds.py makes them: drop unparseable SMILES and rows with no
           endpoint, then 5 repeats of 5-fold GroupKFold(shuffle=True,
           random_state=0xF00D + repeat) over the ds == train molecules, grouped by
           cluster. Checked fold by fold against the study's published
           diagnostics.csv (n_train, n_val, n_test).
  run      the study's 04_run_finetune.py: per fold, train on the four other folds,
           pick the best epoch on the held-out fold, predict every ds == test
           molecule (mean over its ten conformers), with the study's
           FINETUNE_OVERRIDES and seed 0xF00D + 100 * repeat + fold. Writes the same
           per-endpoint prediction CSVs and one diagnostics JSON per fold.
  metrics  the study's 05_collect_metrics.py: R^2, Spearman and MAE per (endpoint,
           arm, repeat, fold), written next to the study's own fold_metrics.csv rows.

    python -m finetuning.admet.eval_mvc prepare --dataset biogen
    python -m finetuning.admet.eval_mvc run --dataset biogen --arm base --ckpt molecules --repeat 0
    python -m finetuning.admet.eval_mvc metrics --dataset biogen
"""

import argparse
import glob
import json
import os
import socket
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# the study repositories (scripts/data/setup_external.sh) and the runs live here
MVC_ROOT = Path(os.environ.get("MVC_ROOT", "data/mvc"))
SOURCE_REPO = MVC_ROOT / "expansion-ml-comparison"
STUDY_DIR = MVC_ROOT / "model-validation-central" / "studies" / "atom-jepa"
CONFORMER_DIR = os.environ.get("MVC_CONFORMERS", "data/eval_conformers")

DATASETS = {
    "expansion": {"raw": "expansion_log_scaled.csv", "config": "finetune_expansionrx",
                  "cache": "expansionrx",
                  "targets": ["LogD", "LogS", "LOG_HLM", "LOG_MLM", "LOG_Caco_AB",
                              "LOG_Caco_Efflux", "LOG_MPPB", "LOG_MBPB", "LOG_MGMB"]},
    "biogen": {"raw": "biogen_adme_3521.csv", "config": "finetune_biogen_adme",
               "cache": "biogen_adme",
               "targets": ["LOG_SOL", "LOG_HLM", "LOG_RLM", "LOG_MDR1_ER", "LOG_HPPB",
                           "LOG_RPPB"]},
}
SMILES_COL, ID_COL, SET_COL, CLUSTER_COL = "SMILES", "Name", "ds", "cluster"
N_REPEATS, N_SPLITS, RANDOM_SEED = 5, 5, 0xF00D
N_CONF, CONF_SEED, PRUNE_RMS = 10, 42, -1.0          # the study's 01_conformers.py

# the study's config.FINETUNE_OVERRIDES; none of them changes what is trained
FINETUNE_OVERRIDES = [
    "++wandb.enabled=false",
    "++finetune.log_transform_tasks=[]",
    "++finetune.report_scale=null",
    "++finetune.conformer_eval_mode=ensemble",
    "++finetune.save_best=false",
    "++finetune.save_last=false",
    "++finetune.seeds=[0]",
    "++finetune.plot_every=0",
]
PRED_COLUMNS = ["method", "endpoint", "repeat", "fold", ID_COL, SMILES_COL, "y_true", "y_pred"]


def atomic_write(path: Path, write) -> None:
    """Publish a file only after writing and flushing it successfully."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as fh:
            tmp = Path(fh.name)
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def out_dir(dataset: str) -> Path:
    return MVC_ROOT / "runs" / dataset


def fold_seed(repeat: int, fold: int) -> int:
    return RANDOM_SEED + 100 * repeat + fold


def load_clean(dataset: str) -> pd.DataFrame:
    """01_make_folds.py load_clean_data."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    targets = DATASETS[dataset]["targets"]
    df = pd.read_csv(SOURCE_REPO / DATASETS[dataset]["raw"])
    missing = [c for c in [SMILES_COL, ID_COL, SET_COL, CLUSTER_COL, *targets] if c not in df.columns]
    if missing:
        raise SystemExit(f"missing expected columns: {missing}")
    parsed = df[SMILES_COL].map(lambda s: Chem.MolFromSmiles(s) is not None)
    df = df[parsed].reset_index(drop=True)
    df = df[~df[targets].isna().all(axis=1)].reset_index(drop=True)
    if df[ID_COL].duplicated().any():
        raise SystemExit(f"{df[ID_COL].duplicated().sum()} duplicate {ID_COL} values")
    df[SET_COL] = df[SET_COL].str.lower()
    return df


def assign_folds(df: pd.DataFrame) -> pd.DataFrame:
    """01_make_folds.py assign_folds."""
    from sklearn.model_selection import GroupKFold
    train_df = df[df[SET_COL] == "train"]
    records = []
    for repeat in range(N_REPEATS):
        splitter = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED + repeat)
        fold_of = np.empty(len(train_df), dtype=int)
        for fold, (_, val_idx) in enumerate(splitter.split(train_df, groups=train_df[CLUSTER_COL])):
            fold_of[val_idx] = fold
        records.append(pd.DataFrame({ID_COL: train_df[ID_COL].to_numpy(), "repeat": repeat,
                                     "fold": fold_of}))
    return pd.concat(records, ignore_index=True)


def prepare(dataset: str) -> None:
    df = load_clean(dataset)
    folds = assign_folds(df)
    cluster_of = df.set_index(ID_COL)[CLUSTER_COL]
    n_train_all = int((df[SET_COL] == "train").sum())
    for repeat, rep in folds.groupby("repeat"):
        assert len(rep) == n_train_all and rep[ID_COL].is_unique
        per_cluster = pd.Series(rep["fold"].to_numpy()).groupby(cluster_of.loc[rep[ID_COL]].to_numpy()).nunique()
        assert (per_cluster == 1).all(), f"repeat {repeat}: a cluster is split across folds"

    # fold sizes as 04_run_finetune.py counts them, against the study's diagnostics
    targets = DATASETS[dataset]["targets"]
    labelled = df[targets].notna().any(axis=1)
    is_test = df[SET_COL] == "test"
    ours = []
    for (repeat, fold) in [(r, f) for r in range(N_REPEATS) for f in range(N_SPLITS)]:
        fold_of = df[ID_COL].map(folds[folds["repeat"] == repeat].set_index(ID_COL)["fold"])
        ours.append({"repeat": repeat, "fold": fold,
                     "n_train": int((~is_test & labelled & (fold_of != fold)).sum()),
                     "n_val": int((~is_test & labelled & (fold_of == fold)).sum()),
                     "n_test": int((is_test & labelled).sum())})
    ours = pd.DataFrame(ours)
    theirs = pd.read_csv(STUDY_DIR / "results" / dataset / "diagnostics.csv")
    cmp = ours.merge(theirs[["repeat", "fold", "n_train", "n_val", "n_test"]],
                     on=["repeat", "fold"], suffixes=("", "_study"))
    bad = cmp[(cmp.n_train != cmp.n_train_study) | (cmp.n_val != cmp.n_val_study)
              | (cmp.n_test != cmp.n_test_study)]
    if len(cmp) != N_REPEATS * N_SPLITS or len(bad):
        raise SystemExit(f"fold sizes differ from the study's diagnostics:\n{bad.to_string()}")
    print(f"{dataset}: {len(df)} molecules ({n_train_all} train, {int(is_test.sum())} test); "
          f"all {len(cmp)} folds match the study's n_train/n_val/n_test", flush=True)

    out = out_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "master.csv", index=False)
    folds.to_csv(out / "fold_assignments.csv", index=False)
    cmp.to_csv(out / "fold_size_check.csv", index=False)

    # 01_conformers.py: every molecule needs a conformer, or the run refuses
    from data.datasets.admet.admet_conformers import build_or_load_conformers
    smiles = list(dict.fromkeys(df[SMILES_COL]))
    conformers = build_or_load_conformers(
        DATASETS[dataset]["cache"], smiles, CONFORMER_DIR, seed=CONF_SEED, n_conf=N_CONF,
        strip_fragments=True, n_workers=-2, prune_rms=PRUNE_RMS)
    failed = [s for s in smiles if conformers.get(s) is None]
    if failed:
        raise SystemExit(f"{len(failed)} molecules have no conformer")
    print(f"all {len(smiles)} molecules have conformers in {CONFORMER_DIR}", flush=True)


def run(dataset: str, arm: str, ckpt: str, repeats, folds_sel, extra, shard=(0, 1)) -> None:
    import pickle

    import torch
    from hydra import compose, initialize_config_dir

    from data.datasets.admet.multitask_finetune import MultiTaskFinetuneDataset, multitask_collate
    from finetuning.admet.training.train import train_multitask
    from finetuning.admet.training.utils import (
        build_equiformer_multitask_factory, load_pretrained_encoder, resolve_device)

    out = out_dir(dataset)
    pred_dir = out / "predictions" / arm if not extra else out / "scratch" / arm
    diag_dir = out / "diagnostics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    conf_dir = str(Path(__file__).resolve().parents[2] / "conf")
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        hcfg = compose(config_name=DATASETS[dataset]["config"], overrides=FINETUNE_OVERRIDES + [
            f"++finetune.ckpt_path={ckpt}",
            f"++misc.checkpoint_dir={out / 'checkpoints' / arm}",
        ] + list(extra))
    ftc = hcfg.finetune
    assert str(ftc.get("pool", "mean")) == "mean"
    print(f"recipe: lr={ftc.lr} epochs={ftc.epochs} freeze={ftc.freeze_encoder_epochs} "
          f"batch={ftc.batch_size} l2sp={ftc.l2sp['lambda']} conformers={ftc.num_conformers} "
          f"precision={ftc.precision} cueq={ftc.cuequivariance} eval={ftc.conformer_eval_mode} "
          f"pool={ftc.pool}", flush=True)

    df = pd.read_csv(out / "master.csv")
    folds = pd.read_csv(out / "fold_assignments.csv")
    tasks = DATASETS[dataset]["targets"]
    caches = glob.glob(os.path.join(CONFORMER_DIR, f"{DATASETS[dataset]['cache']}_seed{CONF_SEED}"
                                    f"_nconf{N_CONF}_prune{int(PRUNE_RMS)}.pkl"))
    with open(caches[0], "rb") as fh:
        conformers = pickle.load(fh)

    device = torch.device(resolve_device(hcfg.misc.device))
    _, eqv3_cfg, ck = load_pretrained_encoder(ckpt, device, load_weights=False)
    factory, F_in, enc_cfg = build_equiformer_multitask_factory(hcfg, eqv3_cfg, ck["context_encoder"], device)
    max_z = int(getattr(eqv3_cfg, "max_num_elements", 128))

    def make_ds(part, tag, sample):
        ds = MultiTaskFinetuneDataset(part, conformers, eqv3_cfg.max_radius, tasks, max_z=max_z,
                                      num_conformers=int(ftc.num_conformers), sample_conformers=sample,
                                      smiles_col=SMILES_COL, tag=tag, features=None)
        if len(ds.keep_index) != len(part):
            raise SystemExit(f"{tag}: {len(part) - len(ds.keep_index)} molecules dropped by the dataset")
        return ds

    labelled = df[tasks].notna().any(axis=1)
    is_test = df[SET_COL] == "test"
    test_df = df[is_test & labelled].reset_index(drop=True)
    test_ds = make_ds(test_df, f"{dataset}/test", False)

    # --shard I/N: this job takes every N-th (repeat, fold) pair, starting at the I-th
    pairs = [(r, f) for r in repeats for f in folds_sel]
    mine = set(pairs[shard[0]::shard[1]])
    for repeat in repeats:
        fold_of = df[ID_COL].map(folds[folds["repeat"] == repeat].set_index(ID_COL)["fold"])
        for fold in folds_sel:
            if (repeat, fold) not in mine:
                continue
            paths = {t: pred_dir / f"{t}_r{repeat}_f{fold}.csv" for t in tasks}
            diagnostic = diag_dir / f"{arm}_r{repeat}_f{fold}.json"
            # Diagnostics are committed only after every prediction CSV succeeds.
            # File existence alone can mistake a failed/partial write for a completed fold.
            if not extra and diagnostic.exists() and all(p.exists() for p in paths.values()):
                saved = json.loads(diagnostic.read_text())
                if (saved.get("arm"), saved.get("ckpt"), saved.get("repeat"), saved.get("fold")) == (arm, ckpt, repeat, fold):
                    print(f"=== r{repeat} f{fold}: done, skipped", flush=True)
                    continue
            # Remove an old completion marker before replacing any fold outputs.
            if not extra:
                diagnostic.unlink(missing_ok=True)
            train_df = df[~is_test & labelled & (fold_of != fold)].reset_index(drop=True)
            val_df = df[~is_test & labelled & (fold_of == fold)].reset_index(drop=True)
            train_ds = make_ds(train_df, f"{dataset}/train", True)
            val_ds = make_ds(val_df, f"{dataset}/valid", False)
            seed = fold_seed(repeat, fold)
            print(f"=== {arm} r{repeat} f{fold} seed {seed}: train {len(train_df)} val {len(val_df)} "
                  f"test {len(test_df)}", flush=True)
            start = time.time()
            torch.cuda.reset_peak_memory_stats()
            r = train_multitask(hcfg, train_ds, val_ds, test_ds, tasks, factory, F_in, multitask_collate,
                                device, False, dataset_name=dataset, run_tag=f"r{repeat}_f{fold}",
                                seed=seed, log_mask=None, encoder_config=enc_cfg, return_test_preds=True)
            seconds = time.time() - start
            preds = np.asarray(r["test_per_mol_preds"], dtype=float)
            if preds.shape != (len(test_df), len(tasks)):
                raise SystemExit(f"test predictions are {preds.shape}, expected {(len(test_df), len(tasks))}")
            for j, t in enumerate(tasks):
                has = test_df[t].notna().to_numpy()
                frame = pd.DataFrame({"method": arm, "endpoint": t, "repeat": repeat, "fold": fold,
                              ID_COL: test_df.loc[has, ID_COL].to_numpy(),
                              SMILES_COL: test_df.loc[has, SMILES_COL].to_numpy(),
                              "y_true": test_df.loc[has, t].to_numpy(), "y_pred": preds[has, j]},
                             columns=PRED_COLUMNS)
                atomic_write(paths[t], lambda fh: frame.to_csv(fh, index=False))
            diag = {"arm": arm, "ckpt": ckpt, "repeat": repeat, "fold": fold, "seed": seed,
                    "seconds": seconds, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(),
                    "best_epoch": r.get("best_epoch"), "val_macro_mae": r["val"]["macro"],
                    "test_macro_mae": r["test"]["macro"],
                    "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
                    "overrides": list(extra)}
            if not extra:
                atomic_write(diagnostic, lambda fh: json.dump(diag, fh, indent=1))
            print(f"--- {arm} r{repeat} f{fold}: best epoch {diag['best_epoch']}, val MAE "
                  f"{diag['val_macro_mae']:.4f}, test MAE {diag['test_macro_mae']:.4f}, "
                  f"{seconds / 60:.1f} min, {diag['peak_mem_gb']:.1f} GB", flush=True)


def metrics(dataset: str) -> None:
    """05_collect_metrics.py fold_metrics, for our arms, appended to the study's rows."""
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score
    out = out_dir(dataset)
    files = sorted(glob.glob(str(out / "predictions" / "*" / "*.csv")))
    preds = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    rows = []
    for (endpoint, method, repeat, fold), g in preds.groupby(["endpoint", "method", "repeat", "fold"]):
        rows.append({"endpoint": endpoint, "method": method, "repeat": repeat, "fold": fold, "n": len(g),
                     "r2": r2_score(g["y_true"], g["y_pred"]),
                     "spearman": spearmanr(g["y_true"], g["y_pred"]).statistic,
                     "mae": mean_absolute_error(g["y_true"], g["y_pred"])})
    ours = pd.DataFrame(rows)
    study = pd.read_csv(STUDY_DIR / "results" / dataset / "fold_metrics.csv")
    allm = pd.concat([study, ours], ignore_index=True)
    allm.to_csv(out / "fold_metrics.csv", index=False)
    print(ours.groupby("method").size().to_string())
    summary = allm.groupby(["endpoint", "method"])[["r2", "spearman", "mae"]].mean().unstack("method")
    print(summary.round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "run", "metrics"])
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--arm", help="method name for the predictions (run)")
    ap.add_argument("--ckpt", help="encoder checkpoint, or a released encoder: 'molecules' / 'bio' (run)")
    ap.add_argument("--repeat", nargs="+", type=int, default=list(range(N_REPEATS)))
    ap.add_argument("--fold", nargs="+", type=int, default=list(range(N_SPLITS)))
    ap.add_argument("--shard", default="0/1",
                    help="I/N: run only every N-th (repeat, fold) pair, starting at the I-th")
    ap.add_argument("--set", nargs="*", default=[],
                    help="extra Hydra overrides; a run with any is written to scratch/")
    a = ap.parse_args()
    if a.step == "prepare":
        prepare(a.dataset)
    elif a.step == "run":
        if not a.arm or not a.ckpt:
            ap.error("run needs --arm and --ckpt")
        i, n = map(int, a.shard.split("/"))
        if not 0 <= i < n:
            ap.error("--shard needs 0 <= I < N")
        run(a.dataset, a.arm, a.ckpt, a.repeat, a.fold, a.set, (i, n))
    else:
        metrics(a.dataset)


if __name__ == "__main__":
    main()
