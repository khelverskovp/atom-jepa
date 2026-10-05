"""
Biogen ADME (Fang et al. 2023) multi-task benchmark: data loading + scaffold
split (see data/datasets/admet/multitask_finetune.py for the encoder-ready Dataset).

Fang, C. et al., "Prospective Validation of Machine Learning Algorithms for
ADME Prediction: An Industrial Perspective", J. Chem. Inf. Model. 2023, 63,
3263-3274. Public data released at
github.com/molecularinformatics/Computational-ADME as a single CSV,
`ADME_public_set_3521.csv`. Run `python -m data.datasets.admet.download.biogen_adme` first.

Format: `SMILES` + 6 task columns, ALL already log-scale (literal `LOG ...`
column names; negative values present, confirming these are pre-logged, not
raw values needing log1p). Missing label = empty cell (NaN via pandas -- same
convention as data/datasets/admet/chembl_mt.py and data/datasets/admet/expansionrx.py). Coverage is very
uneven: HLM_CLint/RLM_CLint/MDR1-MDCK ER/solubility are dense (2100-3100 of
3521), human/rat plasma protein binding are sparse (168/194).

Two task columns contain a literal "/" (`LOG MDR1-MDCK ER (B-A/A-B)`,
`LOG SOLUBILITY PH 6.8 (ug/mL)`) -- harmless everywhere in this pipeline
(task names are only ever pandas column labels / dict keys, never Python
identifiers) except wandb metric keys, which treat "/" as a panel-hierarchy
separator; finetuning.admet.training.train's `_wandb_key` sanitizes just that key fragment.

NO SPLIT SHIPS WITH THIS DATASET. Verified against both the original repo's
own code and its most-used re-curation (Polaris's `biogen/adme-fang-v1`
multitask benchmark) rather than assuming: the repo's per-endpoint
MPNN/FCNN train/test CSVs come from a plain
`sklearn.train_test_split(random_state=84)` (not scaffold), independently per
endpoint (mismatched molecule membership across tasks -- unusable for one
shared multi-task split); Polaris's multitask mirror also uses a single fixed
RANDOM 80/20 split for exactly that reason. This repo deliberately splits
differently (a from-scratch BALANCED Murcko scaffold split, see
data/datasets/admet/scaffold_split.py) as a harder, more realistic generalization test,
consistent with this repo's own TDC ADMET default (finetune_admet.py's
scaffold_split=True) -- results will not be directly literature-comparable.

Since there's no shipped seed/fold either, get_split re-splits FRESH per
seed (the split partition itself changes with the seed, not just model
init) -- finetune_biogen_adme.py loops multiple seeds and reports mean+-std,
mirroring finetune_admet.py's TDC-seed convention, specifically so the
sparse PPB tasks' single-split variance is visible rather than hidden.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from data.datasets.admet.scaffold_split import scaffold_split

SMILES_COL = "SMILES"
TASK_COLS = [
    "LOG HLM_CLint (mL/min/kg)",
    "LOG MDR1-MDCK ER (B-A/A-B)",
    "LOG SOLUBILITY PH 6.8 (ug/mL)",
    "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)",
    "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)",
    "LOG RLM_CLint (mL/min/kg)",
]


# The Figshare release (10.6084/m9.figshare.30350548, the companion set to
# ChEMBL-MT) ships the SAME 3521 molecules and the SAME six endpoints under
# short column names, pre-partitioned by the Adrian et al. cluster split.
# VERIFIED byte-identical to the GitHub CSV: canonical-SMILES overlap 3521/3521
# and max |label difference| == 0.0 with correlation 1.0000 on every endpoint.
# So the two sources differ ONLY in column naming and in how they partition --
# which is what makes the split selectable as a plain config knob.
_FIGSHARE_TO_CANONICAL = {
    "HLM_CLint": "LOG HLM_CLint (mL/min/kg)",
    "MDR1-MDCK_ER": "LOG MDR1-MDCK ER (B-A/A-B)",
    "SOLY_6.8": "LOG SOLUBILITY PH 6.8 (ug/mL)",
    "Human_fraction_unbound_plasma": "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)",
    "Rat_fraction_unbound_plasma": "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)",
    "RLM_Clint": "LOG RLM_CLint (mL/min/kg)",
}
SPLIT_MODES = ("scaffold", "cluster")


@dataclass
class BiogenADMEData:
    tasks: List[str]
    df: pd.DataFrame
    split_mode: str = "scaffold"
    # (train, val, test) already renamed to TASK_COLS; only set for split_mode="cluster"
    cluster_split: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None


def _load_cluster_split(cluster_dir: str, fold: int
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The Adrian et al. K-means cluster split, renamed into our canonical
    column names so every downstream consumer (task order, wandb keys,
    compare_models, plots) is identical regardless of split mode."""
    base = os.path.join(cluster_dir, "export_biogen_cluster_split")
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"{base} not found -- data.split=cluster needs the Figshare biogen "
            f"companion set; run `python -m data.datasets.admet.download.chembl_mt` first."
        )
    if fold not in (0, 1):
        raise ValueError(f"data.cluster_fold must be 0 or 1, got {fold!r}")

    def _read(name):
        df = pd.read_csv(os.path.join(base, name))
        missing = [c for c in _FIGSHARE_TO_CANONICAL if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing expected column(s) {missing}")
        return df.rename(columns={**_FIGSHARE_TO_CANONICAL, "smiles": SMILES_COL})

    return (_read(f"all_train_fold_{fold}_cluster_morgan.csv"),
            _read(f"all_val_fold_{fold}_cluster_morgan.csv"),
            _read("all_test_cluster_morgan.csv"))


def load(data_dir: str, split: str = "scaffold",
         cluster_dir: str = "data/chembl_mt", cluster_fold: int = 0) -> BiogenADMEData:
    """Read the single combined CSV once, plus (for split="cluster") the
    Figshare pre-made partition.

    `data.df` is ALWAYS the full 3521-molecule pool regardless of split mode,
    because conformer/feature caches are built over the whole pool once and
    reused -- only the partitioning changes."""
    if split not in SPLIT_MODES:
        raise ValueError(f"data.split must be one of {SPLIT_MODES}, got {split!r}")
    path = os.path.join(data_dir, "ADME_public_set_3521.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python -m data.datasets.admet.download.biogen_adme` first."
        )
    df = pd.read_csv(path)
    missing = [c for c in [SMILES_COL] + TASK_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing expected column(s) {missing} -- got {list(df.columns)}. "
            f"This should be the 3521-row public 6-endpoint release; a different file "
            f"was loaded by mistake."
        )
    cluster = _load_cluster_split(cluster_dir, cluster_fold) if split == "cluster" else None
    return BiogenADMEData(tasks=TASK_COLS, df=df, split_mode=split, cluster_split=cluster)


def get_split(data: BiogenADMEData, seed: int,
              sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, val, test) for this seed, dispatched on `data.split_mode`.

    "scaffold" (DEFAULT, this repo's own protocol): a fresh balanced Murcko
    scaffold split per seed -- no split ships with the GitHub CSV, and the
    seed reshuffles the PARTITION, not just model init, so seeds are
    independent samples of split difficulty as well as of training noise.

    "cluster": the fixed Adrian et al. partition (K-means, k=5, on PCA-reduced
    2048-bit Morgan r=2 fingerprints; one whole cluster held out as test),
    i.e. the benchmark the Contrastive KERMT paper reports on. NOTE the
    semantics of `seed` change here: the partition is FIXED, so seeds vary
    only model init/shuffling, and `sizes` is ignored. Error bars from
    multi-seed runs are therefore narrower and mean something different than
    in scaffold mode -- do not compare the two protocols' numbers."""
    if data.split_mode == "cluster":
        if data.cluster_split is None:
            raise ValueError(
                "split_mode='cluster' but no cluster split was loaded -- call "
                "load(..., split='cluster') rather than constructing BiogenADMEData by hand."
            )
        return data.cluster_split

    smiles = data.df[SMILES_COL].tolist()
    train_idx, val_idx, test_idx = scaffold_split(smiles, sizes=sizes, balanced=True, seed=seed)
    train_df = data.df.iloc[train_idx].reset_index(drop=True)
    val_df = data.df.iloc[val_idx].reset_index(drop=True)
    test_df = data.df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df
