"""PCBA bioassays: 3D molecules with sparse binary labels over ~1k assays, read from LMDB.

Built by scripts/data/pcba_monroe_prep.py (MONROE's PCBA preprocessing: dedup,
benchmark-aware train/val split, assay filter) and scripts/data/pcba_to_lmdb.py (one 3D
conformer per molecule). The LMDB shards of a split live in
<root>/lmdb/<split>/shard_*.lmdb; the assay ids, in label order, are in
<root>/manifest.json. Optional <root>/<split>-indices.npy (sorted, unique int64 row
indices) selects a fixed subset of a split.

Sample:
    atomic_numbers   [N] long
    node_coordinates [N, 3] float32 (Angstrom)
    y                [n_assays] float32: 1 active, 0 inactive, NaN missing

DEPENDENCY: lmdb (pip install lmdb), imported lazily.
"""

import bisect
import glob
import json
import os
import pickle
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

META_KEY = b"__meta__"


def _open_env(path):
    import lmdb
    return lmdb.open(path, subdir=os.path.isdir(path), readonly=True, lock=False,
                     readahead=False, meminit=False, max_readers=256)


class PCBADataset(Dataset):
    """One split of the PCBA LMDB shards.

    Args:
        root:   output dir of pcba_prep (holds manifest.json and lmdb/)
        split:  "train" | "val"
        limit:  use a random subset of this many molecules (fixed by subset_seed)
        subset_seed: seed of that subset, so every run and rank sees the same molecules
    """

    periodic = False

    def __init__(self, root: str, split: str = "train", limit: Optional[int] = None,
                 subset_seed: int = 0):
        self.root = root
        self.split = split
        with open(os.path.join(root, "manifest.json")) as f:
            self.assay_ids = json.load(f)["assay_ids"]

        self._paths = sorted(glob.glob(os.path.join(root, "lmdb", split, "shard_*.lmdb")))
        if not self._paths:
            raise FileNotFoundError(f"no LMDB shards in {os.path.join(root, 'lmdb', split)}")
        self._envs = None          # opened lazily (per worker; not picklable)

        self._cum, n = [], 0       # cumulative molecule counts across shards
        for path in self._paths:
            env = _open_env(path)
            with env.begin() as txn:
                n += pickle.loads(txn.get(META_KEY))["n"]
            env.close()
            self._cum.append(n)
        view_path = os.path.join(root, f"{split}-indices.npy")
        self._subset = np.load(view_path, allow_pickle=False) if os.path.exists(view_path) else None
        if self._subset is not None:
            if (self._subset.ndim != 1 or self._subset.dtype != np.int64
                    or np.any(self._subset < 0) or np.any(self._subset >= n)
                    or np.any(np.diff(self._subset) <= 0)):
                raise ValueError("invalid immutable PCBA dataset view")
        available = n if self._subset is None else len(self._subset)
        if limit is not None and int(limit) < available:
            rng = np.random.default_rng(subset_seed)
            chosen = np.sort(rng.choice(available, size=int(limit), replace=False))
            self._subset = chosen if self._subset is None else self._subset[chosen]
        self._length = n if self._subset is None else len(self._subset)
        print(f"[PCBADataset] {split}: {self._length} molecules, {self.num_tasks} assays, "
              f"{len(self._paths)} shards in {root}", flush=True)

    @classmethod
    def from_config(cls, data_cfg):
        return cls(root=data_cfg.root, split=data_cfg.get("split", "train"),
                   limit=data_cfg.get("limit", None),
                   subset_seed=int(data_cfg.get("subset_seed", 0)))

    @property
    def num_tasks(self) -> int:
        return len(self.assay_ids)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self._envs is None:
            self._envs = [_open_env(p) for p in self._paths]
        if self._subset is not None:
            idx = int(self._subset[idx])
        shard = bisect.bisect_right(self._cum, idx)
        pos = idx - (self._cum[shard - 1] if shard else 0)
        with self._envs[shard].begin() as txn:
            rec = pickle.loads(txn.get(str(pos).encode()))
        y = rec["y"].astype(np.float32)
        y[rec["y"] < 0] = np.nan
        return {
            "atomic_numbers": torch.from_numpy(rec["atomic_numbers"].astype(np.int64)),
            "node_coordinates": torch.from_numpy(rec["coordinates"]),
            "y": torch.from_numpy(y),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_envs"] = None      # LMDB handles don't survive pickling into workers
        return state
