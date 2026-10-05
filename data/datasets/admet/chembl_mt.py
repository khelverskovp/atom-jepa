"""
ChEMBL-MT multi-task ADME benchmark: data loading only (see
data/datasets/admet/multitask_finetune.py for the encoder-ready Dataset).

ChEMBL-MT is the 25-task regression benchmark ("public_cluster_split") from
Adrian et al., "Multitask finetuning and acceleration of chemical pretrained
models for small molecule drug property prediction" (Figshare DOI
10.6084/m9.figshare.30350548.v2) -- the paper behind NVIDIA's KERMT /
Contrastive-KERMT models. The same release ships a companion 6-task "biogen"
split, much smaller and much denser, useful as a fast smoke test of this
pipeline before the large, extremely sparse ChEMBL-MT set. Run
`python -m data.datasets.admet.download.chembl_mt` first to download both into data/chembl_mt/.

Format: one wide CSV per split-file, `smiles` column + one column per task,
missing label = empty cell (parsed as NaN by pandas -- this IS the label mask).
ChEMBL-MT itself is EXTREMELY sparse: mean 1.12 of 25 labels per molecule,
92.9% of molecules have exactly one.

Splits: `all_train_fold_{0,1}` / `all_val_fold_{0,1}` are two OVERLAPPING
train/val partitions of the same pool (~89% of fold 0's train molecules also
appear in fold 1's train split -- verified, not independent resamples like
TDC's 5 seeds), plus one FIXED `all_test` set with zero SMILES overlap against
either fold's train split (also verified). `fold` plays the role TDC's `seed`
plays in data/datasets/admet/admet_finetune.py, but there are only 2 of them and they are
correlated -- report fold-averaged numbers with that caveat, not as an
independent-seed mean+-std.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

SMILES_COL = "smiles"

# which -> the Figshare zip's extracted subdirectory name (see data/datasets/admet/download/chembl_mt.py)
_SUBDIR = {
    "public": "export_public_cluster_split",   # ChEMBL-MT: 25 tasks, ~90k mol, extremely sparse
    "biogen": "export_biogen_cluster_split",   # companion: 6 tasks, ~3k mol, much denser
}


@dataclass
class ChEMBLMTData:
    which: str
    tasks: List[str]                                          # task column names
    folds: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]        # fold -> (train_df, val_df)
    test_df: pd.DataFrame


def load(data_dir: str, which: str = "public") -> ChEMBLMTData:
    """Read all 5 CSVs for one split ('public' == ChEMBL-MT, or 'biogen') once."""
    if which not in _SUBDIR:
        raise ValueError(f"which must be one of {list(_SUBDIR)}, got {which!r}")
    base = os.path.join(data_dir, _SUBDIR[which])

    def _read(name: str) -> pd.DataFrame:
        path = os.path.join(base, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- run `python -m data.datasets.admet.download.chembl_mt "
                f"--which {which}` first."
            )
        return pd.read_csv(path)

    train0 = _read("all_train_fold_0_cluster_morgan.csv")
    val0 = _read("all_val_fold_0_cluster_morgan.csv")
    train1 = _read("all_train_fold_1_cluster_morgan.csv")
    val1 = _read("all_val_fold_1_cluster_morgan.csv")
    test_df = _read("all_test_cluster_morgan.csv")

    tasks = [c for c in train0.columns if c != SMILES_COL]
    return ChEMBLMTData(which=which, tasks=tasks,
                        folds={0: (train0, val0), 1: (train1, val1)}, test_df=test_df)


def get_train_valid(data: ChEMBLMTData, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return data.folds[fold]
