"""
Generic balanced Murcko-scaffold train/val/test split (Chemprop's well-known
`scaffold_split(..., balanced=True)` algorithm, reimplemented directly since
chemprop isn't an installed dependency -- RDKit's MurckoScaffold is already a
project dependency, see data/datasets/admet/admet_conformers.py).

Not tied to any one dataset -- built for data/datasets/admet/biogen_adme.py (the first
dataset in this repo with no shipped split at all) but generic enough to
reuse for a future one.

Algorithm: group molecule indices by Bemis-Murcko scaffold, split the
scaffold groups into "big" (> len(smiles)/2 molecules) and "small", shuffle
each group independently with the given seed, concatenate big-then-small, and
greedily fill train/val/test index lists in that order up to each split's
target size. Putting big clusters first (rather than randomly interleaved)
is what makes this "balanced": it prevents a single huge scaffold cluster
from landing entirely in test (bad luck with a size-sorted split) while still
randomizing which of the many small clusters go where.
"""

import random
from typing import List, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def _scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES (no stereochemistry). Unparsable SMILES
    get a unique scaffold key of their own (`"invalid:<smiles>"`) so they end
    up as their own singleton group rather than being silently dropped or
    lumped together."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"invalid:{smiles}"
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return f"invalid:{smiles}"


def scaffold_split(smiles: List[str], sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   balanced: bool = True, seed: int = 0) -> Tuple[List[int], List[int], List[int]]:
    """[N] SMILES -> (train_idx, val_idx, test_idx) index lists partitioning
    range(N). `sizes` must sum to ~1.0. `balanced=True` disperses large
    scaffold clusters across all three splits instead of dumping them all
    into train (see module docstring)."""
    assert abs(sum(sizes) - 1.0) < 1e-6, f"sizes must sum to 1.0, got {sizes}"
    n = len(smiles)
    n_train = int(round(sizes[0] * n))
    n_val = int(round(sizes[1] * n))

    scaffold_to_indices = {}
    for i, smi in enumerate(smiles):
        scaffold_to_indices.setdefault(_scaffold(smi), []).append(i)

    index_sets = list(scaffold_to_indices.values())
    if balanced:
        big_sets = [s for s in index_sets if len(s) > n / 2]
        small_sets = [s for s in index_sets if len(s) <= n / 2]
        rng = random.Random(seed)
        rng.shuffle(big_sets)
        rng.shuffle(small_sets)
        index_sets = big_sets + small_sets
    else:
        index_sets = sorted(index_sets, key=len, reverse=True)

    train_idx, val_idx, test_idx = [], [], []
    for s in index_sets:
        if len(train_idx) + len(s) <= n_train:
            train_idx.extend(s)
        elif len(val_idx) + len(s) <= n_val:
            val_idx.extend(s)
        else:
            test_idx.extend(s)

    return train_idx, val_idx, test_idx
