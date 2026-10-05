"""
Generic K-means CLUSTER train/val/test split on Morgan fingerprints.

Sibling of data/datasets/admet/scaffold_split.py, same contract, harder split. Written for
the OpenADMET CYP blind challenge (data/datasets/admet/cyp.py), but tied to no dataset.

WHY A CLUSTER SPLIT RATHER THAN A SCAFFOLD ONE. The CYP challenge's blind
test set was built by ANALOG EXPANSION -- the top ~25 screening hits per
isoform, plus ~10 purchased close analogs of each -- so held-out compounds
arrive in tight similarity series. A random split leaves near-analogs of every
validation compound sitting in train; even a scaffold split only separates
Bemis-Murcko cores, and two analogs frequently share neither an exact scaffold
nor enough dissimilarity to be a fair test. Holding out whole fingerprint
clusters is the closest local proxy for "predict a series you have never
seen", which is what the leaderboard actually measures.

This repo had NO clustering code before this file. data/datasets/admet/biogen_adme.py's
"cluster" mode only READS a partition Figshare ships with that dataset
(all_{train,val}_fold_N_cluster_morgan.csv); it never computes one, and the
CYP release ships nothing equivalent.

Algorithm (the Adrian et al. protocol this repo already documents for Biogen):
Morgan fingerprint per molecule -> PCA to n_components -> KMeans into
n_clusters -> shuffle the clusters with `seed` and greedily assign WHOLE
clusters to train, then val, then test up to each split's target size. No
cluster is ever divided, which is the entire point.

n_clusters is deliberately larger than Adrian et al.'s k=5: at k=5 one cluster
is ~20% of the data, so the val fraction quantizes to multiples of 20% and
cannot hit a requested 15%. More, smaller clusters keep the sizes controllable
while still holding out whole series.

SEEDS ARE LOAD-BEARING here. Re-clustering per seed (KMeans init + cluster
shuffle) means an across-seed spread captures SPLIT DIFFICULTY as well as
training noise -- unlike data/datasets/admet/biogen_adme.py's cluster mode, where the
partition is fixed and seeds vary model init only. Report it accordingly.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")


def morgan_matrix(smiles: List[str], radius: int = 2, n_bits: int = 2048
                  ) -> Tuple[np.ndarray, List[int]]:
    """[n_valid, n_bits] float32 fingerprint matrix + the indices it covers.

    Unparsable SMILES are reported back via the index list rather than being
    silently dropped, so the caller can give them their own singleton clusters
    (matching data/datasets/admet/scaffold_split.py's `invalid:<smiles>` convention).
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows, keep = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        rows.append(gen.GetFingerprintAsNumPy(mol).astype(np.float32))
        keep.append(i)
    if not rows:
        return np.zeros((0, n_bits), dtype=np.float32), []
    return np.stack(rows), keep


def cluster_labels(smiles: List[str], seed: int = 0, n_clusters: int = 10,
                   n_components: int = 50, radius: int = 2, n_bits: int = 2048
                   ) -> np.ndarray:
    """[N] int cluster id per molecule. Unparsable SMILES each get their own
    negative id, so they can never be merged into a real cluster."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    n = len(smiles)
    labels = np.full(n, -1, dtype=np.int64)
    X, keep = morgan_matrix(smiles, radius=radius, n_bits=n_bits)
    if len(keep) == 0:
        return -np.arange(1, n + 1, dtype=np.int64)

    # PCA before KMeans: Euclidean k-means on raw high-dimensional sparse bit
    # vectors is dominated by bit-count differences; reducing first is what the
    # Adrian et al. protocol does and it also makes KMeans tractable.
    k_comp = int(min(n_components, X.shape[0], X.shape[1]))
    Xr = PCA(n_components=k_comp, random_state=seed).fit_transform(X)
    k = int(min(n_clusters, Xr.shape[0]))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xr)
    labels[keep] = km.labels_

    # unparsable -> unique negative ids (-1, -2, ... never collide with KMeans)
    bad = [i for i in range(n) if i not in set(keep)]
    for j, i in enumerate(bad):
        labels[i] = -(j + 1)
    return labels


def cluster_split(smiles: List[str], sizes: Tuple[float, float, float] = (0.85, 0.15, 0.0),
                  seed: int = 0, n_clusters: int = 10, n_components: int = 50,
                  radius: int = 2, n_bits: int = 2048
                  ) -> Tuple[List[int], List[int], List[int]]:
    """[N] SMILES -> (train_idx, val_idx, test_idx) partitioning range(N),
    with every cluster kept intact. `sizes` must sum to ~1.0; a zero entry
    yields an empty list (the CYP case, where the real test set is the blind
    750 and only train/val are wanted locally).

    Same signature shape and return contract as
    data/datasets/admet/scaffold_split.py::scaffold_split, so the two are interchangeable at
    a call site.
    """
    assert abs(sum(sizes) - 1.0) < 1e-6, f"sizes must sum to 1.0, got {sizes}"
    n = len(smiles)
    n_train = int(round(sizes[0] * n))
    n_val = int(round(sizes[1] * n))

    labels = cluster_labels(smiles, seed=seed, n_clusters=n_clusters,
                            n_components=n_components, radius=radius, n_bits=n_bits)
    groups = {}
    for i, c in enumerate(labels):
        groups.setdefault(int(c), []).append(i)

    # Largest cluster first, then shuffled -- same "balanced" reasoning as
    # scaffold_split: placing big clusters before small ones stops one huge
    # cluster from landing entirely in val by bad luck, while the shuffle
    # randomizes which of the many small ones go where.
    index_sets = list(groups.values())
    big = [s for s in index_sets if len(s) > n / 2]
    small = [s for s in index_sets if len(s) <= n / 2]
    rng = np.random.RandomState(seed)
    rng.shuffle(big)
    rng.shuffle(small)
    index_sets = big + small

    train_idx, val_idx, test_idx = [], [], []
    buckets = [(train_idx, n_train), (val_idx, n_val), (test_idx, n - n_train - n_val)]

    # Only splits with a NONZERO requested size may receive clusters. A naive
    # train->val->test cascade silently dumps overflow into `test` even when
    # sizes=(0.85, 0.15, 0.0) asked for no test split at all -- which is the
    # CYP case, where the real test set is the blind 750 and a local "test"
    # would just be stolen validation data. Overflow goes to the last bucket
    # that was actually requested.
    active = [(lst, cap) for lst, cap in buckets if cap > 0]
    if not active:
        raise ValueError(f"sizes={sizes} requests no non-empty split")

    for s in index_sets:
        for lst, cap in active:
            if len(lst) + len(s) <= cap:
                lst.extend(s)
                break
        else:
            active[-1][0].extend(s)   # doesn't fit anywhere: last requested split
    return train_idx, val_idx, test_idx


def max_tanimoto_to_train(smiles: List[str], train_idx: List[int], query_idx: List[int],
                          radius: int = 2, n_bits: int = 2048) -> Optional[float]:
    """Mean over `query_idx` of each molecule's MAX Tanimoto similarity to any
    training molecule -- the split-difficulty statistic this repo already
    quotes for the Biogen protocols (scaffold 0.431 vs cluster 0.386, lower =
    harder). Returns None if either side is empty.

    This is the check that a split is actually doing its job: if it is not
    materially below what a random split gives, the held-out set is full of
    near-analogs of training compounds and local validation will flatter the
    model relative to the blind leaderboard.
    """
    if not train_idx or not query_idx:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    def fps(idx):
        out = []
        for i in idx:
            mol = Chem.MolFromSmiles(smiles[i])
            if mol is not None:
                out.append(gen.GetFingerprint(mol))
        return out

    train_fps, q_fps = fps(train_idx), fps(query_idx)
    if not train_fps or not q_fps:
        return None
    return float(np.mean([max(DataStructs.BulkTanimotoSimilarity(f, train_fps))
                          for f in q_fps]))


def hit_expansion_split(smiles: List[str], activities: np.ndarray,
                        val_frac: float = 0.15, seed: int = 0, top_k: int = 25,
                        candidate_mult: int = 2, radius: int = 2, n_bits: int = 2048
                        ) -> Tuple[List[int], List[int]]:
    """(train_idx, val_idx) reproducing the CYP challenge's own test-set
    construction: hold out ANALOG SERIES around the most potent compounds.

    WHY, precisely. OpenADMET built the blind 750 by taking "the top 25 hits
    per CYP for 3 CYPs (75 compounds total) and purchasing the top 10
    chemisimilars" of each from Enamine. So the held-out chemistry is neither a
    random sample nor an arbitrary fingerprint cluster -- it is tight analog
    series clustered around the ACTIVE end of the range. A cluster split
    reproduces "unseen series" but not "unseen series around known hits", and
    is activity-agnostic, which makes it materially easier than the real task.

    THE PARENTS STAY IN TRAIN. This is the load-bearing detail. In the real
    challenge the top hits ARE part of the released training data; what is
    withheld are newly purchased neighbours of them. So the faithful split
    keeps each parent trainable and moves only its analogs to validation --
    mirroring "you have seen this hit, now predict its series". Holding the
    parents out too would be harsher than reality, not more realistic.

    `activities`: [n_mol, n_direct_tasks] with NaN for unmeasured. Parents are
    the top `top_k` per task, so a compound potent on several isoforms is
    counted once (measured: 100 draws -> 95 unique parents on the CYP pool).

    `seed` matters despite "top-k" being deterministic: parents are drawn at
    random from the top `candidate_mult * top_k`, so different seeds pick
    different-but-comparably-potent parents and the across-seed spread reflects
    which series were withheld. With candidate_mult=1 the parent set is fixed
    and only neighbour ties move.
    """
    n = len(smiles)
    n_val_target = int(round(val_frac * n))
    rng = np.random.RandomState(seed)

    # --- parents: potent compounds, sampled from the top of each task -------
    parents: set = set()
    A = np.asarray(activities, float)
    for t in range(A.shape[1]):
        col = A[:, t]
        have = np.where(~np.isnan(col))[0]
        if have.size == 0:
            continue
        ranked = have[np.argsort(-col[have])]
        pool_top = ranked[: max(top_k, min(candidate_mult * top_k, ranked.size))]
        take = min(top_k, pool_top.size)
        parents |= set(rng.choice(pool_top, size=take, replace=False).tolist())
    parents_l = sorted(parents)
    if not parents_l:
        raise ValueError("no measured activities -- cannot pick parent hits")

    # --- analogs: nearest neighbours of each parent, by Tanimoto ------------
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps: List[Optional[object]] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fps.append(gen.GetFingerprint(mol) if mol is not None else None)
    valid = [i for i, f in enumerate(fps) if f is not None]
    valid_fps = [fps[i] for i in valid]

    # (similarity, neighbour) over every parent, tightest analogs first -- so
    # the val set is filled with the closest series members, which is what
    # "purchase the top 10 chemisimilars" produces.
    cand: Dict[int, float] = {}
    parent_set = set(parents_l)
    for p in parents_l:
        if fps[p] is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fps[p], valid_fps)
        for j, s in zip(valid, sims):
            if j in parent_set or j == p:
                continue                      # parents stay in TRAIN
            if s > cand.get(j, -1.0):
                cand[j] = float(s)

    ordered = sorted(cand, key=lambda j: -cand[j])
    val_idx = sorted(ordered[:n_val_target])
    val_set = set(val_idx)
    train_idx = [i for i in range(n) if i not in val_set]
    return train_idx, val_idx
