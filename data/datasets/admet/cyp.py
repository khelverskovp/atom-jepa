"""
OpenADMET CYP inhibition blind-challenge loader.

Download first:  python -m finetuning.admet.fetch_cyp

SIX TASKS, TWO KINDS -- this is the repo's first MIXED regression+classification
multi-task dataset:
  * 4 REGRESSION targets: direct-inhibition pIC50 for CYP1A2/2C9/2D6/3A4,
    scored on the leaderboard by MA-ST-RAE (macro-averaged soft-threshold
    relative absolute error, zero inside the measurement's credible interval).
  * 2 BINARY targets: CYP2D6_is_TDI / CYP3A4_is_TDI (time-dependent
    inhibition, positive = >2-fold IC50 shift after NADPH preincubation),
    scored by MCC. TDI is assayed for these two isoforms only.

ONLY cyp-challenge-TRAIN_TDI.csv IS READ for training, even though the release
also ships cyp-challenge-TRAIN_inhibition.csv. Verified against the real files:
the TDI file is a strict SUPERSET -- all 4,905 inhibition SMILES appear among
its 6,145 rows, it carries every direct-inhibition column as well, and on the
overlap the values are byte-identical (max |difference| = 0.0, identical NaN
patterns, identical SMILES). Reading both and merging would add a join and a
reconciliation step for no information. The inhibition file is still fetched
so that claim stays checkable.

LABELS ARE VERY SPARSE, which is the normal case for this pipeline (NaN =
"not measured", never "measured as zero"). Measured coverage of the 6,145:
    CYP1A2 pIC50 1412 | CYP2C9 1285 | CYP2D6 1493 | CYP3A4 2335
    CYP2D6_is_TDI 1497 (21.6% positive) | CYP3A4_is_TDI 3584 (21.3% positive)

CREDIBLE INTERVALS are carried alongside the labels rather than as extra
tasks. ST-RAE is defined against them (error = distance to the nearer bound,
exactly zero inside), so they must reach evaluation -- but they are not
prediction targets and must never enter the loss or the label mask. Verified:
every direct-inhibition measurement has both bounds, lo <= value <= hi holds
for 100% of rows, median interval width 0.379 pIC50 units.

pIC50 IS ALREADY LOG-SCALE by definition (observed range 1.91-7.95), so
log_mask is None -- do NOT apply this repo's log1p transform on top.

SPLIT: no train/val split ships, and the blind test set was built by ANALOG
EXPANSION (top hits per isoform plus purchased close analogs), so held-out
compounds come in tight similarity series. get_split therefore uses
data/datasets/admet/cluster_split.py (K-means on Morgan fingerprints, whole clusters held
out) rather than a random or scaffold split. Measured on this pool at the
default k=10: val->train mean max-Tanimoto 0.348 vs 0.498 for a random split
of the same size -- i.e. genuinely harder, which is the point.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from data.datasets.admet.cluster_split import cluster_split, hit_expansion_split

SMILES_COL = "SMILES"
NAME_COL = "Molecule_Name"

ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
TDI_ISOFORMS = ["CYP2D6", "CYP3A4"]          # TDI is assayed for these only

# Fixed task order. Everything downstream (head index, wandb key, submission
# column) keys off this list, so it must not be reordered casually.
DIRECT_COLS = [f"{iso}_pIC50_direct_inhibition" for iso in ISOFORMS]
TDI_COLS = [f"{iso}_is_TDI" for iso in TDI_ISOFORMS]
TASK_COLS = DIRECT_COLS + TDI_COLS

# Parallel to TASK_COLS: which loss/metric family each head belongs to.
TASK_KINDS = ["regression"] * len(DIRECT_COLS) + ["binary"] * len(TDI_COLS)

# Parallel to TASK_COLS; None for the binary tasks, which have no interval.
CONF_LOW_COLS = [f"{c}_conf_low" for c in DIRECT_COLS] + [None] * len(TDI_COLS)
CONF_HIGH_COLS = [f"{c}_conf_high" for c in DIRECT_COLS] + [None] * len(TDI_COLS)

TRAIN_FILE = "cyp-challenge-TRAIN_TDI.csv"
TEST_FILE = "cyp-challenge-TEST-BLINDED.csv"


@dataclass
class CYPData:
    tasks: List[str]
    kinds: List[str]
    df: pd.DataFrame           # train pool, 6145 rows, TASK_COLS + conf bounds
    test_df: pd.DataFrame      # the blind 750: Molecule_Name + SMILES only


def _require(df: pd.DataFrame, cols: List[str], path: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing expected column(s) {missing}. This does not look "
            f"like the OpenADMET CYP challenge release -- re-fetch with "
            f"`python -m finetuning.admet.fetch_cyp --force`."
        )


def load(data_dir: str = "data/cyp") -> CYPData:
    train_path = os.path.join(data_dir, TRAIN_FILE)
    test_path = os.path.join(data_dir, TEST_FILE)
    for p in (train_path, test_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"{p} not found -- run `python -m finetuning.admet.fetch_cyp` first."
            )

    df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    _require(df, [SMILES_COL, NAME_COL] + TASK_COLS, train_path)
    _require(df, [c for c in CONF_LOW_COLS + CONF_HIGH_COLS if c], train_path)
    _require(test_df, [SMILES_COL, NAME_COL], test_path)

    # is_TDI arrives as object dtype holding True/False/NaN. Cast to float so
    # NaN survives as "not measured" -- data.datasets.admet.multitask_finetune's _build_labels
    # builds its mask from notna(), and a bool column would coerce NaN to
    # False, silently turning unmeasured compounds into negatives.
    for c in TDI_COLS:
        df[c] = df[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float)

    return CYPData(tasks=list(TASK_COLS), kinds=list(TASK_KINDS), df=df, test_df=test_df)


def get_split(data: CYPData, seed: int = 0,
              sizes: Tuple[float, float, float] = (0.85, 0.15, 0.0),
              n_clusters: int = 10, n_components: int = 50,
              split_mode: str = "hit_expansion", top_k: int = 25,
              candidate_mult: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(train_df, val_df). No local test split: the real test set is the blind
    750 in `data.test_df`.

    split_mode:
      "hit_expansion" (DEFAULT) -- reproduces the challenge's own test-set
          construction: hold out tight ANALOG SERIES around the most potent
          compounds, with the parent hits themselves left in TRAIN. This is
          the harder and more faithful proxy for the blind set; see
          data/datasets/admet/cluster_split.py::hit_expansion_split.
      "cluster" -- the earlier whole-K-means-cluster holdout. Activity-agnostic
          and measurably easier, kept so older runs stay reproducible.

    NOTE the two are NOT comparable: hit_expansion enriches validation for
    potency, which widens the label spread and therefore RAE's denominator.
    Switching modes resets the baseline for ST-RAE/RAE; MAE and the rank
    correlations are the metrics that stay interpretable across the change.
    """
    smiles = data.df[SMILES_COL].tolist()
    if split_mode == "hit_expansion":
        acts = data.df[DIRECT_COLS].to_numpy(dtype=float)
        train_idx, val_idx = hit_expansion_split(
            smiles, acts, val_frac=float(sizes[1]), seed=seed,
            top_k=top_k, candidate_mult=candidate_mult)
    elif split_mode == "cluster":
        train_idx, val_idx, test_idx = cluster_split(
            smiles, sizes=sizes, seed=seed,
            n_clusters=n_clusters, n_components=n_components)
        if test_idx:
            raise AssertionError(
                f"cluster_split returned {len(test_idx)} molecules for a local "
                f"test split; sizes={sizes} requested none.")
    else:
        raise ValueError(
            f"data.split_mode={split_mode!r}; expected 'hit_expansion' or 'cluster'")

    if set(train_idx) & set(val_idx):
        raise AssertionError("train/val overlap")
    if len(train_idx) + len(val_idx) != len(data.df):
        raise AssertionError("split does not partition the pool")
    return (data.df.iloc[train_idx].reset_index(drop=True),
            data.df.iloc[val_idx].reset_index(drop=True))


def conf_bounds(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """[n_mol, T] credible-interval bounds aligned to TASK_COLS, NaN where the
    task has none (the two binary tasks) or the measurement is missing. Fed to
    finetuning/admet/metrics/cyp.py::soft_threshold_rae, never to the loss."""
    n, T = len(df), len(TASK_COLS)
    lo = np.full((n, T), np.nan)
    hi = np.full((n, T), np.nan)
    for t, (lc, hc) in enumerate(zip(CONF_LOW_COLS, CONF_HIGH_COLS)):
        if lc is not None and lc in df.columns:
            lo[:, t] = df[lc].to_numpy(dtype=float)
        if hc is not None and hc in df.columns:
            hi[:, t] = df[hc].to_numpy(dtype=float)
    return lo, hi
