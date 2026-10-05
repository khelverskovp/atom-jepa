"""
MoleculeNet (Wu et al. 2018) -- the 8 BINARY-CLASSIFICATION benchmarks used as
the standard evaluation suite for self-supervised molecular pretraining.

    BBBP(1)  Tox21(12)  ToxCast(617)  SIDER(27)
    ClinTox(2)  MUV(17)  HIV(1)  BACE(1)

Metric is ROC-AUC, macro-averaged over a dataset's targets. ONE MODEL PER
DATASET: every target of a dataset is predicted jointly by a single wide head
(see finetuning.admet.models.model.MultiTaskReadout's `shared_head`), NOT one head per target --
617 separate heads for ToxCast would be ~10.2M parameters over ~8.6k molecules.

SPLIT PROTOCOL -- a verbatim port of C-FREE (arXiv:2509.22468,
github.com/ariguiba/C-FREE, preprocessing/utils/scaffolds.py::get_scaffold_split),
so our numbers are directly comparable to that paper's Table 2 rather than
merely adjacent to it. See scaffold_split_cfree below for the four properties
that must NOT be "cleaned up".

data/datasets/admet/scaffold_split.py::scaffold_split is a DIFFERENT algorithm (Chemprop's
balanced Murcko split, used by Biogen ADME). Do not substitute it here.

Data source is torch_geometric.datasets.MoleculeNet -- the same loader C-FREE
uses (their preprocessing/datasets/moleculenet.py wraps it). Going through PyG
rather than the raw DeepChem CSVs is deliberate: PyG drops molecules that
RDKit resolves to zero atoms, which changes both N and the row ORDER, and
C-FREE's algorithm depends on scaffold first-appearance order. Reading the CSVs
ourselves would silently produce a different partition.

This module holds NO torch-model code and NO conformer logic -- it returns
pandas DataFrames, exactly like data/datasets/admet/cyp.py and data/datasets/admet/biogen_adme.py. The
encoder-ready Dataset is data.datasets.admet.multitask_finetune.MultiTaskFinetuneDataset,
whose `y` + `y_mask` contract already matches MoleculeNet's NaN-for-missing
convention (ToxCast is 71% missing).
"""

import csv
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SMILES_COL = "smiles"


@dataclass(frozen=True)
class MoleculeNetSpec:
    """key -> how to load it and what to expect back.

    `n_molecules` is the count PyG actually yields on this stack, NOT the
    published MoleculeNet size -- PyG's process() skips any SMILES that RDKit
    resolves to zero atoms, so e.g. BBBP is 2039 here vs 2050 as published.
    Measured once and pinned as a warn-on-mismatch guard: a change means the
    upstream CSV or the RDKit version moved, which would silently change every
    split.
    """
    key: str
    pyg_name: str
    n_targets: int
    n_molecules: int
    metric: str = "roc-auc"


DATASETS: Dict[str, MoleculeNetSpec] = {
    "bace":    MoleculeNetSpec("bace",    "BACE",      1,  1513),
    "bbbp":    MoleculeNetSpec("bbbp",    "BBBP",      1,  2039),
    "clintox": MoleculeNetSpec("clintox", "ClinTox",   2,  1480),
    "sider":   MoleculeNetSpec("sider",   "SIDER",    27,  1427),
    "tox21":   MoleculeNetSpec("tox21",   "Tox21",    12,  7823),
    "toxcast": MoleculeNetSpec("toxcast", "ToxCast", 617,  8579),
    "hiv":     MoleculeNetSpec("hiv",     "HIV",       1, 41120),
    "muv":     MoleculeNetSpec("muv",     "MUV",      17, 93087),
}

DATASET_NAMES: List[str] = list(DATASETS)


@dataclass
class MoleculeNetData:
    name: str
    tasks: List[str]                 # target column names, len == spec.n_targets
    kinds: List[str]                 # all "binary"
    df: pd.DataFrame                 # SMILES_COL + one column per target, NaN = missing
    spec: MoleculeNetSpec


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _task_names(ds, spec: MoleculeNetSpec) -> List[str]:
    """Real target names from the raw CSV header.

    PyG throws the header away (`f.read().split('\\n')[1:-1]`), so the names are
    not recoverable from the dataset object. Read them back with csv.reader
    (SIDER's header contains quoted commas) and slice with the SAME y_idx that
    MoleculeNet.names uses. Falls back to positional names -- cosmetic only,
    nothing downstream depends on the strings except the per-target report and
    wandb keys (which finetuning.admet.training.train._wandb_key sanitizes anyway).
    """
    try:
        from torch_geometric.datasets import MoleculeNet
        # names entry is (pretty, filename, dirname, smiles_idx, y_slice); the
        # last element is a slice (or list) selecting the label columns.
        y_idx = MoleculeNet.names[spec.pyg_name.lower()][4]
        with open(ds.raw_paths[0], "r") as f:
            header = next(csv.reader(f))
        if isinstance(y_idx, slice):
            names = header[y_idx]
        elif isinstance(y_idx, int):          # bace=2, bbbp=-2, hiv=-1: one label column
            names = [header[y_idx]]
        else:
            names = [header[i] for i in y_idx]
        names = [str(n).strip() for n in names]
        if len(names) == spec.n_targets:
            return names
        print(f"[moleculenet] {spec.key}: header gave {len(names)} names but "
              f"{spec.n_targets} targets -- using positional names", flush=True)
    except Exception as e:                                   # pragma: no cover
        print(f"[moleculenet] {spec.key}: could not read header ({e}) -- "
              f"using positional names", flush=True)
    return [f"{spec.key}_t{i:03d}" for i in range(spec.n_targets)]


def load(name: str, root: str = "data/moleculenet") -> MoleculeNetData:
    """Load one MoleculeNet classification benchmark into a DataFrame.

    Rows are returned in PyG's own order and are NEVER deduplicated, dropped or
    reordered here: C-FREE's split bins scaffolds by first-appearance order, so
    any of those would produce a different partition than the paper's.
    MoleculeNet does ship exact duplicate SMILES (notably ClinTox and Tox21) --
    that is reported by split_diagnostics, not silently fixed.
    """
    key = str(name).lower()
    if key not in DATASETS:
        raise ValueError(f"Unknown MoleculeNet dataset {name!r}; "
                         f"expected one of {DATASET_NAMES}")
    spec = DATASETS[key]

    from torch_geometric.datasets import MoleculeNet   # lazy: torch_geometric is heavy
    ds = MoleculeNet(root=root, name=spec.pyg_name)

    # Bulk extraction via the public accessors. A `str` attribute collates to a
    # list and Data.__inc__ is 0 for both, so InMemoryDataset.__getattr__
    # returns the stacked attribute directly -- no per-item loop, no warning.
    smiles = list(ds.smiles)
    y = ds.y.numpy()
    if y.ndim == 1:                                   # defensive; process() view(1,-1)s
        y = y.reshape(-1, 1)

    # These guard a REAL failure mode: PyG's process() strips quoted CSV fields
    # with a greedy `re.sub(r'\".*\"', '', line)`, which on a row containing two
    # quoted fields deletes everything between the first and last quote --
    # including unquoted columns in between -- shifting the column count.
    if y.shape[1] != spec.n_targets:
        raise ValueError(f"[moleculenet] {key}: expected {spec.n_targets} targets, "
                         f"got y.shape={y.shape}. Upstream CSV or PyG parser changed.")
    if len(smiles) != y.shape[0]:
        raise ValueError(f"[moleculenet] {key}: {len(smiles)} SMILES vs {y.shape[0]} label rows")
    finite = np.isfinite(y)
    bad = set(np.unique(y[finite]).tolist()) - {0.0, 1.0}
    if bad:
        raise ValueError(f"[moleculenet] {key}: non-binary labels present: {sorted(bad)[:8]}")
    if len(ds) != spec.n_molecules:
        print(f"[moleculenet] {key}: WARNING got {len(ds)} molecules, pinned "
              f"{spec.n_molecules}. Upstream data or RDKit version changed -- "
              f"splits will NOT match previous runs.", flush=True)

    tasks = _task_names(ds, spec)
    # Built in one shot: 617 successive df[t] = ... inserts fragment the frame
    # badly enough that pandas warns (ToxCast).
    df = pd.concat(
        [pd.DataFrame({SMILES_COL: smiles}),
         pd.DataFrame(y, columns=tasks)],
        axis=1,
    )

    print(f"[moleculenet] {key}: {len(df)} molecules, {spec.n_targets} targets, "
          f"{100 * (~finite).mean():.1f}% labels missing", flush=True)
    return MoleculeNetData(name=key, tasks=tasks, kinds=["binary"] * spec.n_targets,
                           df=df, spec=spec)


# --------------------------------------------------------------------------- #
# split -- verbatim port of C-FREE's get_scaffold_split
# --------------------------------------------------------------------------- #
def _bm_scaffold(smi: str, include_chirality: bool = False) -> str:
    """Bemis-Murcko scaffold SMILES, or "" when RDKit cannot parse the input.

    Ported from C-FREE's bm_scaffold. NOTE include_chirality defaults to FALSE
    here, matching the function they actually call -- their two UNUSED variants
    (get_scaffold_split_old, random_scaffold_split) pass True, so copying from
    the wrong one silently changes the benchmark.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=include_chirality)


def scaffold_split_cfree(smiles_list: List[str], frac_train: float = 0.8,
                         frac_valid: float = 0.1, seed: int = 0
                         ) -> Tuple[List[int], List[int], List[int]]:
    """(train_idx, val_idx, test_idx) -- C-FREE's scaffold split, ported verbatim.

    FOUR PROPERTIES THAT MUST NOT BE "FIXED". Each is a real, load-bearing
    detail of the protocol we are reproducing, and each looks like a bug:

    1. include_chirality=False (see _bm_scaffold).

    2. "" IS A SCAFFOLD, AND A HUGE ONE. MurckoScaffoldSmiles returns "" for
       every ACYCLIC molecule (verified: "CCCCO" -> "", "[Na+].[Cl-]" -> ""),
       not just for parse failures. All acyclic molecules therefore collapse
       into ONE bin, which is typically among the largest, is sorted first, and
       lands whole in train. Do NOT disambiguate them the way
       data/datasets/admet/scaffold_split.py::_scaffold does with f"invalid:{smiles}" -- that
       is a different benchmark.

    3. THE SHUFFLE ONLY BREAKS TIES. `rng.shuffle(bins)` followed by
       `bins.sort(key=len, reverse=True)` relies on Python's sort being STABLE:
       the ordering among bins of EQUAL size is the shuffled one, and the
       ordering across sizes is always largest-first regardless of seed. So the
       seed moves far fewer molecules than a random split would. Must be
       random.Random (Mersenne Twister) -- numpy's RNG draws a different
       permutation for the same seed.

    4. THE TRIMS CAN SPLIT A SCAFFOLD GROUP. The two trim_to spills move the
       overflow of train into valid and of valid into test by POSITION, which
       can place part of one scaffold group in train and the rest in valid.
       Scaffold separation is therefore not strictly guaranteed at the two
       boundaries. split_diagnostics counts how often this fires.

    Their `task_idx` filtering path is not ported: the call site
    (preprocessing/datasets/moleculenet.py) always passes task_idx=None, which
    makes `list(compress(enumerate(smiles_list), non_null))` an identity
    re-index. Their `n_test` is likewise computed and never used.
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(smiles_list):
        groups[_bm_scaffold(s)].append(i)

    rng = random.Random(seed)
    bins = list(groups.values())
    rng.shuffle(bins)                  # stable tiebreak
    bins.sort(key=len, reverse=True)   # big scaffolds first

    n = len(smiles_list)
    n_train = int(round(frac_train * n))
    n_valid = int(round(frac_valid * n))

    train: List[int] = []
    valid: List[int] = []
    test: List[int] = []
    for b in bins:
        if len(train) + len(b) <= n_train:
            train += b
        elif len(valid) + len(b) <= n_valid:
            valid += b
        else:
            test += b

    train, spill = train[:n_train], train[n_train:]
    valid += spill
    valid, spill = valid[:n_valid], valid[n_valid:]
    test += spill
    return train, valid, test


def get_split(data: MoleculeNetData, seed: int, frac_train: float = 0.8,
              frac_valid: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train_df, val_df, test_df) for this split seed. The partition is
    redrawn per seed (C-FREE reports the mean over three scaffold splits)."""
    tr, va, te = scaffold_split_cfree(data.df[SMILES_COL].tolist(),
                                      frac_train=frac_train, frac_valid=frac_valid, seed=seed)
    d = data.df
    return (d.iloc[tr].reset_index(drop=True),
            d.iloc[va].reset_index(drop=True),
            d.iloc[te].reset_index(drop=True))


def split_diagnostics(data: MoleculeNetData, seed: int) -> Dict:
    """Facts about a split that are easy to get wrong silently and cheap to
    measure: how big the ""-scaffold (acyclic) bin is, how many scaffold groups
    the trim_to spills split across a boundary, how many SMILES appear in more
    than one split (MoleculeNet ships exact duplicates and C-FREE does not
    dedupe), and -- the important one for reporting -- how many targets are
    SINGLE-CLASS in each split, since those score NaN and are dropped by the
    macro nanmean, making the averaged ROC-AUC span a different target subset
    per split."""
    smiles = data.df[SMILES_COL].tolist()
    tr, va, te = scaffold_split_cfree(smiles, seed=seed)

    scaf = [_bm_scaffold(s) for s in smiles]
    where = {}
    for split, idx in (("train", tr), ("valid", va), ("test", te)):
        for i in idx:
            where.setdefault(scaf[i], set()).add(split)
    split_groups = sum(1 for v in where.values() if len(v) > 1)

    smi_where: Dict[str, set] = {}
    for split, idx in (("train", tr), ("valid", va), ("test", te)):
        for i in idx:
            smi_where.setdefault(smiles[i], set()).add(split)
    dup_across = sum(1 for v in smi_where.values() if len(v) > 1)

    y = data.df[data.tasks].to_numpy(dtype=float)
    out = {"seed": seed, "n_total": len(smiles),
           "n_scaffolds": len(set(scaf)),
           "n_acyclic_bin": sum(1 for s in scaf if s == ""),
           "scaffold_groups_split_across": split_groups,
           "smiles_in_multiple_splits": dup_across}
    for split, idx in (("train", tr), ("valid", va), ("test", te)):
        sub = y[idx]
        fin = np.isfinite(sub)
        n_single = sum(1 for t in range(sub.shape[1])
                       if len(np.unique(sub[fin[:, t], t])) < 2)
        out[f"n_{split}"] = len(idx)
        out[f"{split}_targets_single_class"] = n_single
        out[f"{split}_targets_scorable"] = sub.shape[1] - n_single
    return out
