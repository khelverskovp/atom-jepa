"""Uni-Mol pretraining molecules (drug-like 3D conformers), read from LMDB.

Each LMDB value is a pickled dict per MOLECULE (the official Uni-Mol format):
    atoms        list of N element symbols
    coordinates  list of [N, 3] arrays, one per conformer (a single [N, 3] or a
                 [C, N, 3] array is accepted too)
    scaffold / smi  optional strings, used only by `dedup_by`

Sample:
    atomic_numbers   [N] long
    node_coordinates [N, 3] float32 (Angstrom)

Examples per molecule, via `conformer` / `conformers_per_mol`:
    "first"   the stored first conformer(s); deterministic
    "random"  a freshly sampled conformer on every access (augmentation)
    "all"     every conformer becomes its own example
"first"/"random" without dedup or atom-count filters need no scan of the LMDB
(its length is read from metadata); everything else scans the records once.

DEPENDENCY: lmdb (pip install lmdb), imported lazily.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from data.core.elements import SYMBOL_TO_Z, Z_TO_SYMBOL, symbols_to_z


def _open_env(path):
    import lmdb
    return lmdb.open(
        path, subdir=os.path.isdir(path), readonly=True, lock=False,
        readahead=False, meminit=False, max_readers=512,
    )


def _decode_value(buf):
    """Unpickle a record, transparently handling gzip-compressed values."""
    try:
        return pickle.loads(buf)
    except Exception:
        import gzip
        return pickle.loads(gzip.decompress(buf))


def _norm_symbols(atoms) -> List[str]:
    """Normalize a record's `atoms` field to a list of element symbols."""
    out = []
    for a in atoms:
        if isinstance(a, bytes):
            out.append(a.decode())
        elif isinstance(a, int):              # atomic number stored instead of symbol
            out.append(Z_TO_SYMBOL.get(int(a), "X"))
        else:
            out.append(str(a))
    return out


def _norm_conformers(coords) -> List["torch.Tensor"]:
    """Normalize a record's coordinates to a list of [N,3] float arrays."""
    import numpy as np
    if isinstance(coords, np.ndarray):
        if coords.ndim == 3:
            return [coords[i] for i in range(coords.shape[0])]
        return [coords]                         # [N,3]
    if isinstance(coords, (list, tuple)) and len(coords) > 0:
        first = np.asarray(coords[0])
        if first.ndim == 2:                     # list of [N,3] conformers
            return [np.asarray(c) for c in coords]
        return [np.asarray(coords)]             # a single [N,3] given row-wise
    return []


class UniMolDataset(Dataset):
    """Uni-Mol LMDB molecules as samples.

    Args:
        lmdb_paths:  one path, or a list of paths, to Uni-Mol LMDB files/dirs
        conformer:   "first" | "random" | "all" (see module docstring)
        conformers_per_mol: conformers kept per molecule for first/random (>=1)
        remove_hs:   drop hydrogens
        dedup_by:    "none" | "scaffold" | "smi" -- keep the first record per key (scans)
        skip_unknown_elements: skip records containing a symbol that is not a real
                     element (corrupt records); False raises on them instead
        min_atoms/max_atoms: atom-count filtering (scans)
        limit:       cap the number of examples
    """

    periodic = False

    def __init__(
        self,
        lmdb_paths,
        conformer: str = "random",
        conformers_per_mol: int = 1,
        remove_hs: bool = False,
        dedup_by: str = "none",
        skip_unknown_elements: bool = True,
        min_atoms: int = 1,
        max_atoms: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        self.conformer = conformer.lower()
        self.conformers_per_mol = max(1, int(conformers_per_mol))
        self.remove_hs = remove_hs
        self.dedup_by = dedup_by.lower()
        self.skip_unknown_elements = skip_unknown_elements
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms

        self._paths = [lmdb_paths] if isinstance(lmdb_paths, str) else list(lmdb_paths)
        self._envs: Optional[list] = None     # opened lazily (per worker; not picklable)

        # Per-env metadata: (ascii_int_keys, n_entries, keys_or_None). Keys are only
        # read when they aren't the common contiguous-ascii-int scheme.
        self._meta: List[Tuple[bool, int, Optional[list]]] = []
        for path in self._paths:
            env = _open_env(path)
            with env.begin() as txn:
                entries = txn.stat()["entries"]
                ascii_int = (txn.get(b"0") is not None
                             and txn.get(str(entries - 1).encode()) is not None)
                keys = None
                if not ascii_int:
                    keys = [k for k, _ in txn.cursor() if not k.startswith(b"__")]
                    entries = len(keys)
            env.close()
            self._meta.append((ascii_int, entries, keys))

        self._cum = []  # cumulative record counts across envs (for the fast path)
        c = 0
        for _, n, _ in self._meta:
            c += n
            self._cum.append(c)
        self._n_records = c

        needs_scan = (self.conformer == "all") or (self.dedup_by != "none") \
            or (min_atoms > 1) or (max_atoms is not None)
        if not needs_scan:
            # FAST PATH: no record reads; (env, pos, conformer) resolved arithmetically.
            self._scanned = False
            self._index = None
            total = self._n_records * self.conformers_per_mol
            self._length = total if limit is None else min(limit, total)
        else:
            if self.conformer == "all":
                print("[UniMolDataset] conformer='all': scanning records to count "
                      "conformers (one-time, O(#records)). Use `limit` for dev runs.",
                      flush=True)
            self._scanned = True
            self._index = self._build_scanned_index(limit)
            self._length = len(self._index)

        self._close_envs()   # keep the dataset picklable for DataLoader workers
        print(f"[UniMolDataset] {self._length} examples from {self._paths} "
              f"(conformer={self.conformer}x{self.conformers_per_mol}, "
              f"remove_hs={remove_hs}, dedup_by={self.dedup_by}, "
              f"min_atoms={min_atoms}, max_atoms={max_atoms}, limit={limit})",
              flush=True)

    @classmethod
    def from_config(cls, data_cfg):
        return cls(
            lmdb_paths=data_cfg.lmdb_paths,
            conformer=data_cfg.get("conformer", "random"),
            conformers_per_mol=int(data_cfg.get("conformers_per_mol", 1)),
            remove_hs=bool(data_cfg.get("remove_hs", False)),
            dedup_by=data_cfg.get("dedup_by", "none"),
            skip_unknown_elements=bool(data_cfg.get("skip_unknown_elements", True)),
            min_atoms=data_cfg.get("min_atoms", 1),
            max_atoms=data_cfg.get("max_atoms", None),
            limit=data_cfg.get("limit", None),
        )

    # ----- LMDB plumbing (lazy, per-worker) ------------------------------- #
    def _ensure_envs(self):
        if self._envs is None:
            self._envs = [_open_env(p) for p in self._paths]

    def _close_envs(self):
        if self._envs is not None:
            for e in self._envs:
                try:
                    e.close()
                except Exception:
                    pass
            self._envs = None

    def _key_for(self, env_idx: int, pos: int) -> bytes:
        ascii_int, _, keys = self._meta[env_idx]
        return str(pos).encode() if ascii_int else keys[pos]

    def _read_record(self, env_idx: int, pos: int):
        self._ensure_envs()
        with self._envs[env_idx].begin() as txn:
            buf = txn.get(self._key_for(env_idx, pos))
        return _decode_value(buf)

    def _locate(self, rec_global: int) -> Tuple[int, int]:
        """Map a global record index to (env_idx, position-within-env)."""
        prev = 0
        for env_idx, cum in enumerate(self._cum):
            if rec_global < cum:
                return env_idx, rec_global - prev
            prev = cum
        raise IndexError(rec_global)

    def _build_scanned_index(self, limit) -> List[Tuple[int, int, int]]:
        index: List[Tuple[int, int, int]] = []
        seen = set()
        for env_idx, (_, n, _) in enumerate(self._meta):
            for pos in range(n):
                if limit is not None and len(index) >= limit:
                    self._close_envs()
                    return index
                try:
                    rec = self._read_record(env_idx, pos)
                    symbols = self._record_symbols(rec)
                except Exception:
                    continue
                na = len(symbols)
                if na < self.min_atoms:
                    continue
                if self.max_atoms is not None and na > self.max_atoms:
                    continue
                if self.skip_unknown_elements and any(s not in SYMBOL_TO_Z for s in symbols):
                    continue
                if self.dedup_by != "none":
                    key = rec.get(self.dedup_by)
                    if key is not None:
                        key = key.decode() if isinstance(key, bytes) else str(key)
                        if key in seen:
                            continue
                        seen.add(key)
                if self.conformer == "all":
                    n_conf = len(_norm_conformers(self._record_coords(rec)))
                    index.extend((env_idx, pos, c) for c in range(n_conf))
                else:
                    for c in range(self.conformers_per_mol):
                        index.append((env_idx, pos, c))
        self._close_envs()
        return index

    @staticmethod
    def _record_symbols(rec) -> List[str]:
        atoms = rec.get("atoms")
        if atoms is None:
            raise KeyError("record has no 'atoms' field")
        return _norm_symbols(atoms)

    @staticmethod
    def _record_coords(rec):
        co = rec.get("coordinates", rec.get("coords"))
        if co is None:
            raise KeyError("record has no 'coordinates' field")
        return co

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int, _depth: int = 0) -> Dict[str, torch.Tensor]:
        if self._scanned:
            env_idx, pos, conf = self._index[idx]
        else:
            rec_global = idx // self.conformers_per_mol
            conf = idx % self.conformers_per_mol
            env_idx, pos = self._locate(rec_global)

        try:
            rec = self._read_record(env_idx, pos)
            symbols = self._record_symbols(rec)
            conformers = _norm_conformers(self._record_coords(rec))
            if not conformers:
                raise ValueError("no conformers")
        except Exception:
            # skip an unreadable/empty record (rare); advance to the next index
            if _depth > 8:
                raise
            return self.__getitem__((idx + 1) % self._length, _depth + 1)

        if self.conformer == "random":
            ci = int(torch.randint(len(conformers), (1,)))
        elif self.conformer == "all":
            ci = conf if conf < len(conformers) else 0
        else:  # "first" (or the c-th of the first-k)
            ci = min(conf, len(conformers) - 1)

        coords = torch.as_tensor(conformers[ci], dtype=torch.float32)

        if self.remove_hs:
            keep = [k for k, s in enumerate(symbols) if s != "H"]
            symbols = [symbols[k] for k in keep]
            coords = coords[torch.tensor(keep, dtype=torch.long)] if keep else coords[:0]

        # The fast path can't pre-screen corrupt records, so skip them here.
        if self.skip_unknown_elements and any(s not in SYMBOL_TO_Z for s in symbols):
            if _depth > 8:
                raise KeyError("too many consecutive records with unknown elements")
            return self.__getitem__((idx + 1) % self._length, _depth + 1)

        return {"atomic_numbers": symbols_to_z(symbols), "node_coordinates": coords}
