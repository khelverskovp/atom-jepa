"""Alexandria (PBE) DFT-relaxed inorganic crystals (https://alexandria.icams.rub.de/).

Sample:
    atomic_numbers   [N] long
    node_coordinates [N, 3] float32 (Cartesian, Angstrom)
    cell             [3, 3] float32 (row-vector lattice)

INPUT FORMAT. The raw dumps (alexandria_*.json.bz2) are slow to read, so convert
them once to LMDB shards and point `src` at the output directory:

    python -m scripts.data.alexandria_to_lmdb --src data/alexandria/pbe --out data/alexandria_lmdb

`AlexandriaDataset(src, ...)` picks the backend: LMDB when `src` is an LMDB store
(a directory of `*.lmdb` shards, or a single one), raw JSON otherwise (slow fallback).

FILTERING. `e_above_hull` is stored per record, so filtering is an index operation
at load time:  0.0 -> on the convex hull (~115k), 0.05 -> within 50 meV/atom
(~770k), None -> everything (~4.5M).

DEPENDENCIES: lmdb, numpy (LMDB path); pymatgen (JSON fallback only).
"""

import bz2
import glob
import gzip
import json
import os
import struct
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# record codec (also used by scripts/data/alexandria_to_lmdb.py)
# ---------------------------------------------------------------------------
# layout, little-endian, chosen so every field lands on a 4-byte boundary:
#   [0 : 8]                header   int32 n_atoms, float32 e_above_hull (nan = unknown)
#   [8 : 8+12n]            pos      float32[n, 3]  Cartesian Angstrom
#   [8+12n : 44+12n]       cell     float32[3, 3]  row vectors
#   [44+12n : 44+14n]      Z        int16[n]
_HDR = struct.Struct("<if")
META_KEY = b"__meta__"
FORMAT_VERSION = 1


def encode_record(z, pos, cell, e_above_hull: float = float("nan")) -> bytes:
    n = len(z)
    return b"".join((
        _HDR.pack(n, float(e_above_hull)),
        np.ascontiguousarray(pos, dtype="<f4").tobytes(),
        np.ascontiguousarray(cell, dtype="<f4").tobytes(),
        np.ascontiguousarray(z, dtype="<i2").tobytes(),
    ))


def decode_record(buf) -> Dict[str, torch.Tensor]:
    n, _ = _HDR.unpack_from(buf, 0)
    o = _HDR.size
    pos = np.frombuffer(buf, "<f4", count=3 * n, offset=o).reshape(n, 3)
    o += 12 * n
    cell = np.frombuffer(buf, "<f4", count=9, offset=o).reshape(3, 3)
    o += 36
    z = np.frombuffer(buf, "<i2", count=n, offset=o)
    # copies are mandatory: `buf` is an LMDB memoryview only valid inside the txn.
    return {
        "atomic_numbers": torch.from_numpy(z.astype(np.int64)),
        "node_coordinates": torch.from_numpy(pos.copy()),
        "cell": torch.from_numpy(cell.copy()),
    }


def record_header(buf):
    """(n_atoms, e_above_hull) without decoding the payload."""
    return _HDR.unpack_from(buf, 0)


def idx_to_key(i: int) -> bytes:
    return b"%09d" % i


# ---------------------------------------------------------------------------
# LMDB dataset
# ---------------------------------------------------------------------------
def _is_lmdb(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.exists(os.path.join(path, "data.mdb")):
        return True
    return bool(glob.glob(os.path.join(path, "*.lmdb")))


# LMDB refuses to open the same path twice in one process, so environments are shared
# process-wide. The cache is keyed on the pid: a forked DataLoader worker drops the
# inherited handles and opens its own, the safe way to use a read-only env across fork.
_ENV_CACHE: Dict[str, "object"] = {}
_ENV_PID: Optional[int] = None


def _get_env(path: str):
    global _ENV_PID
    pid = os.getpid()
    if _ENV_PID != pid:
        _ENV_CACHE.clear()
        _ENV_PID = pid
    env = _ENV_CACHE.get(path)
    if env is None:
        import lmdb
        env = lmdb.open(path, readonly=True, lock=False, readahead=False,
                        meminit=False, subdir=True, max_readers=1024)
        _ENV_CACHE[path] = env
    return env


def _resolve_lmdb_shards(src) -> List[str]:
    """Expand a path / list / directory / glob into a sorted list of LMDB dirs."""
    srcs = [src] if isinstance(src, str) else list(src)
    out: List[str] = []
    for s in srcs:
        if os.path.isdir(s) and os.path.exists(os.path.join(s, "data.mdb")):
            out.append(s)                                     # a single store
        elif os.path.isdir(s):
            out.extend(glob.glob(os.path.join(s, "*.lmdb")))  # a dir of shards
        else:
            out.extend(glob.glob(s))
    return sorted(set(p for p in out if os.path.isdir(p)))


class AlexandriaLMDBDataset(Dataset):
    """Alexandria crystals stored as LMDB shards.

    Args:
        src:                 LMDB dir, dir of `*.lmdb` shards, list, or glob
        e_above_hull_max:    keep only entries within this eV/atom of the hull (None = all);
                             entries with an unknown hull distance are dropped when set
        min_atoms/max_atoms: atom-count filtering
        limit:               keep at most this many structures (after filtering)
    """

    periodic = True

    def __init__(
        self,
        src,
        e_above_hull_max: Optional[float] = 0.05,
        min_atoms: int = 1,
        max_atoms: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        import lmdb  # fail fast with a clear error if the dep is missing
        assert lmdb is not None

        self.shards = _resolve_lmdb_shards(src)
        if not self.shards:
            raise FileNotFoundError(f"no Alexandria LMDB shards matched src={src!r}")

        shard_ids: List[np.ndarray] = []
        local_ids: List[np.ndarray] = []
        n_total = 0
        for si, path in enumerate(self.shards):
            natoms, ehull = self._read_shard_meta(path)
            n_total += natoms.size

            keep = natoms >= min_atoms
            if max_atoms is not None:
                keep &= natoms <= max_atoms
            if e_above_hull_max is not None:
                keep &= np.isfinite(ehull) & (ehull <= float(e_above_hull_max))

            local = np.flatnonzero(keep).astype(np.int64)
            if limit is not None:
                room = limit - int(sum(a.size for a in local_ids))
                if room <= 0:
                    break
                local = local[:room]
            local_ids.append(local)
            shard_ids.append(np.full(local.size, si, dtype=np.int32))

        self._shard_id = np.concatenate(shard_ids) if shard_ids else np.zeros(0, np.int32)
        self._local_id = np.concatenate(local_ids) if local_ids else np.zeros(0, np.int64)

        print(f"[AlexandriaLMDBDataset] kept {len(self._shard_id)} / {n_total} structures "
              f"from {len(self.shards)} shard(s) (e_above_hull_max={e_above_hull_max}, "
              f"min_atoms={min_atoms}, max_atoms={max_atoms}, limit={limit})", flush=True)

    @staticmethod
    def _read_shard_meta(path: str):
        """(natoms[int32], ehull[float32]) for every record in a shard."""
        with _get_env(path).begin(write=False) as txn:
            raw = txn.get(META_KEY)
            if raw is not None:
                meta = json.loads(raw.decode())
                natoms = np.frombuffer(bytes.fromhex(meta["natoms"]), "<i4")
                ehull = np.frombuffer(bytes.fromhex(meta["ehull"]), "<f4")
                return natoms.copy(), ehull.copy()
            # legacy / interrupted store: rebuild the index by scanning headers.
            print(f"[AlexandriaLMDBDataset] no {META_KEY!r} in {path}, scanning...",
                  flush=True)
            na, eh = [], []
            with txn.cursor() as cur:
                for k, v in cur:
                    if k == META_KEY:
                        continue
                    n, e = record_header(v)
                    na.append(n)
                    eh.append(e)
            return np.asarray(na, np.int32), np.asarray(eh, np.float32)

    def __len__(self) -> int:
        return int(self._shard_id.size)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        si = int(self._shard_id[idx])
        key = idx_to_key(int(self._local_id[idx]))
        with _get_env(self.shards[si]).begin(write=False, buffers=True) as txn:
            buf = txn.get(key)
            if buf is None:
                raise KeyError(f"{key!r} missing from {self.shards[si]}")
            return decode_record(buf)


# ---------------------------------------------------------------------------
# raw-JSON fallback (slow; prefer converting to LMDB)
# ---------------------------------------------------------------------------
_EHULL_KEYS = ("e_above_hull", "energy_above_hull", "ehull", "e_above_hull_pbe")


def _resolve_json_files(src) -> List[str]:
    srcs = [src] if isinstance(src, str) else list(src)
    files: List[str] = []
    for s in srcs:
        if os.path.isdir(s):
            for ext in ("*.json", "*.json.bz2", "*.json.gz"):
                files.extend(glob.glob(os.path.join(s, ext)))
        elif any(ch in s for ch in "*?[]"):
            files.extend(glob.glob(s))
        else:
            files.append(s)
    return sorted(set(files))


def open_maybe_compressed(path: str):
    if path.endswith(".bz2"):
        return bz2.open(path, "rt")
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def _get_first(d: dict, keys: Sequence[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


class AlexandriaJSONDataset(Dataset):
    """In-memory reader for raw alexandria_*.json(.bz2/.gz). Same args as the LMDB one."""

    periodic = True

    def __init__(
        self,
        src,
        e_above_hull_max: Optional[float] = 0.05,
        min_atoms: int = 1,
        max_atoms: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        from pymatgen.core import Structure
        from data.core.structures import structure_to_sample

        files = _resolve_json_files(src)
        if not files:
            raise FileNotFoundError(f"no Alexandria json files matched src={src!r}")

        self._structs: List[Dict[str, torch.Tensor]] = []
        n_seen = n_skip_hull = n_skip_size = n_err = 0
        for path in files:
            if limit is not None and len(self._structs) >= limit:
                break
            with open_maybe_compressed(path) as f:
                obj = json.load(f)
            entries = obj.get("entries", obj.get("data", [])) if isinstance(obj, dict) else obj
            for entry in entries:
                if limit is not None and len(self._structs) >= limit:
                    break
                n_seen += 1
                try:
                    sd_raw = entry["structure"] if isinstance(entry, dict) and "structure" in entry else entry
                    sd = structure_to_sample(Structure.from_dict(sd_raw))
                    data = entry.get("data", {}) if isinstance(entry, dict) else {}
                except Exception:
                    n_err += 1
                    continue
                n = int(sd["atomic_numbers"].numel())
                if n < min_atoms or (max_atoms is not None and n > max_atoms):
                    n_skip_size += 1
                    continue
                if e_above_hull_max is not None:
                    eah = _get_first(data or {}, _EHULL_KEYS)
                    if eah is None or float(eah) > e_above_hull_max:
                        n_skip_hull += 1
                        continue
                self._structs.append(sd)

        print(f"[AlexandriaJSONDataset] kept {len(self._structs)} / {n_seen} structures "
              f"from {len(files)} file(s) (skipped: hull={n_skip_hull} "
              f"size={n_skip_size} parse_err={n_err}). Convert to LMDB for faster loads.",
              flush=True)

    def __len__(self) -> int:
        return len(self._structs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._structs[idx]


class AlexandriaDataset:
    """Backend-dispatching constructor: LMDB when `src` is an LMDB store, else JSON."""

    periodic = True

    def __new__(cls, src, *args, **kwargs):
        probe = src if isinstance(src, str) else (list(src)[0] if len(list(src)) else "")
        if _resolve_lmdb_shards(src) or _is_lmdb(probe):
            return AlexandriaLMDBDataset(src, *args, **kwargs)
        return AlexandriaJSONDataset(src, *args, **kwargs)

    @classmethod
    def from_config(cls, data_cfg):
        return cls(
            src=data_cfg.src,
            e_above_hull_max=data_cfg.get("e_above_hull_max", 0.05),
            min_atoms=data_cfg.get("min_atoms", 1),
            max_atoms=data_cfg.get("max_atoms", None),
            limit=data_cfg.get("limit", None),
        )
