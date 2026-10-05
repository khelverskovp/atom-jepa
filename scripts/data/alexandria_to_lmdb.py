"""
One-time converter: Alexandria json(.bz2/.gz) dumps -> LMDB shards.

    python -m scripts.data.alexandria_to_lmdb --src /data/alexandria --out /data/alexandria_lmdb
    python -m scripts.data.alexandria_to_lmdb --src '/data/alexandria/*.json.bz2' \
        --out /data/alexandria_lmdb --workers 8 --e-above-hull-max 0.1

Why this is much faster than reading the JSON at train time:
  * one output LMDB per input file, converted in parallel processes;
  * entries are streamed with an incremental JSON decoder, so a 20 GB shard never
    materialises as one Python object graph;
  * pymatgen is bypassed entirely -- `Structure.from_dict` costs ~1 ms/entry, which is
    over an hour of pure overhead across 4.5M structures. The site dicts are read
    directly instead;
  * multithreaded bz2 (lbzip2/pbzip2) is used automatically when available.

The result is read by data/alexandria_dataset.AlexandriaLMDBDataset. `e_above_hull` is
stored per record and mirrored into a per-shard `__meta__` blob, so hull/size filtering
happens at load time -- convert once with the widest settings you might want.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.datasets.alexandria import META_KEY, FORMAT_VERSION, encode_record, idx_to_key

# 1 TiB of *sparse* address space; LMDB only commits pages it actually writes.
MAP_SIZE = 1 << 40
COMMIT_EVERY = 4096

_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl "
    "Mc Lv Ts Og"
).split()
_Z = {s: i + 1 for i, s in enumerate(_SYMBOLS)}
_ELEM_RE = re.compile(r"^([A-Z][a-z]?)")
_EHULL_KEYS = ("e_above_hull", "energy_above_hull", "ehull", "e_above_hull_pbe")


# ---------------------------------------------------------------------------
# input streaming
# ---------------------------------------------------------------------------
def _open_text(path: str):
    """Text stream over a plain/bz2/gz file, using a parallel decompressor if present."""
    if path.endswith(".bz2"):
        for exe in ("lbzip2", "pbzip2"):
            if shutil.which(exe):
                proc = subprocess.Popen([exe, "-dc", path], stdout=subprocess.PIPE)
                return open(proc.stdout.fileno(), "rt", encoding="utf-8", closefd=True)
        import bz2
        return bz2.open(path, "rt", encoding="utf-8")
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


_WRAPPER_RE = re.compile(r'"(?:entries|data|structures)"\s*:\s*\[')


def _array_start(f, buf: str, chunk_size: int, max_prelude: int = 1 << 20):
    """Locate the start of the entry array, refilling `buf` as needed.

    -> (buf, index just past '[') or None if the file holds no array.
    """
    while True:
        lead = buf.lstrip()
        if lead.startswith("["):
            return buf, buf.index("[") + 1
        if lead[:1] == "{":
            m = _WRAPPER_RE.search(buf)
            if m:
                return buf, m.end()
        elif lead:
            raise ValueError(f"expected a JSON array or wrapper object, got {lead[:32]!r}")
        if len(buf) > max_prelude:
            raise ValueError("no entry array found in the first 1 MB of the file")
        more = f.read(chunk_size)
        if not more:
            return None
        buf += more


def iter_json_array(f, chunk_size: int = 1 << 22):
    """Yield the elements of a top-level JSON array without loading the whole file.

    Handles both a bare `[...]` and a `{"entries": [...]}` / `{"data": [...]}` wrapper
    by locking onto the first `[` in the stream.
    """
    dec = json.JSONDecoder()
    buf = f.read(chunk_size)
    if not buf:
        return
    idx = _array_start(f, buf, chunk_size)
    if idx is None:
        return
    buf, idx = idx[0], idx[1]
    while True:
        while True:  # skip separators, refilling as needed
            while idx < len(buf) and buf[idx] in " \t\r\n,":
                idx += 1
            if idx < len(buf):
                break
            more = f.read(chunk_size)
            if not more:
                return
            buf, idx = more, 0
        if buf[idx] == "]":
            return
        while True:  # grow the buffer until one full object decodes
            try:
                obj, end = dec.raw_decode(buf, idx)
                break
            except ValueError:
                more = f.read(chunk_size)
                if not more:
                    raise
                buf += more
        yield obj
        idx = end
        if idx > (1 << 23):  # drop the consumed prefix
            buf, idx = buf[idx:], 0


# ---------------------------------------------------------------------------
# entry -> arrays  (no pymatgen)
# ---------------------------------------------------------------------------
def _species_z(site: dict) -> int:
    sp = site.get("species")
    if sp:
        best = max(sp, key=lambda s: s.get("occu", 1.0))
        sym = best.get("element") or best.get("symbol") or ""
    else:
        sym = site.get("label", "")
    m = _ELEM_RE.match(str(sym))
    if not m or m.group(1) not in _Z:
        raise ValueError(f"unrecognised species {sym!r}")
    return _Z[m.group(1)]


def parse_entry(entry) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """-> (Z[n], pos[n,3] Cartesian, cell[3,3], e_above_hull)."""
    if isinstance(entry, dict) and "structure" in entry:
        struct, data = entry["structure"], (entry.get("data") or {})
        eah = _first(data, _EHULL_KEYS)
        if eah is None:
            eah = _first(entry, _EHULL_KEYS)
    else:
        struct, eah = entry, None

    cell = np.asarray(struct["lattice"]["matrix"], dtype=np.float32)
    if cell.shape != (3, 3):
        raise ValueError(f"bad lattice shape {cell.shape}")

    sites = struct["sites"]
    z = np.empty(len(sites), dtype=np.int16)
    pos = np.empty((len(sites), 3), dtype=np.float32)
    for i, site in enumerate(sites):
        z[i] = _species_z(site)
        xyz = site.get("xyz")
        pos[i] = xyz if xyz is not None else np.asarray(site["abc"], np.float64) @ cell
    return z, pos, cell, (float("nan") if eah is None else float(eah))


def _first(d: dict, keys):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# per-shard conversion
# ---------------------------------------------------------------------------
def convert_one(job) -> Dict:
    import lmdb

    path, out_dir, e_above_hull_max, min_atoms, max_atoms, limit, overwrite = job
    name = os.path.basename(path)
    for ext in (".json.bz2", ".json.gz", ".json"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    out = os.path.join(out_dir, name + ".lmdb")

    if os.path.exists(os.path.join(out, "data.mdb")) and not overwrite:
        return {"file": path, "out": out, "kept": None, "skipped_existing": True}
    if os.path.exists(out) and overwrite:
        shutil.rmtree(out)

    t0 = time.time()
    env = lmdb.open(out, map_size=MAP_SIZE, subdir=True, meminit=False, max_dbs=0)
    natoms: List[int] = []
    ehulls: List[float] = []
    n_seen = n_err = n_skip_size = n_skip_hull = 0
    kept = 0
    try:
        txn = env.begin(write=True)
        with _open_text(path) as f:
            for entry in iter_json_array(f):
                n_seen += 1
                try:
                    z, pos, cell, eah = parse_entry(entry)
                except Exception:
                    n_err += 1
                    continue
                n = int(z.size)
                if n < min_atoms or (max_atoms is not None and n > max_atoms):
                    n_skip_size += 1
                    continue
                if e_above_hull_max is not None and not (eah <= e_above_hull_max):
                    n_skip_hull += 1  # also drops NaN (unknown hull distance)
                    continue
                txn.put(idx_to_key(kept), encode_record(z, pos, cell, eah))
                natoms.append(n)
                ehulls.append(eah)
                kept += 1
                if kept % COMMIT_EVERY == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                if limit is not None and kept >= limit:
                    break
        meta = {
            "version": FORMAT_VERSION,
            "count": kept,
            "source": os.path.abspath(path),
            "natoms": np.asarray(natoms, "<i4").tobytes().hex(),
            "ehull": np.asarray(ehulls, "<f4").tobytes().hex(),
        }
        txn.put(META_KEY, json.dumps(meta).encode())
        txn.commit()
        env.sync()
    finally:
        env.close()

    return {"file": path, "out": out, "seen": n_seen, "kept": kept, "err": n_err,
            "skip_size": n_skip_size, "skip_hull": n_skip_hull,
            "secs": round(time.time() - t0, 1), "skipped_existing": False}


def resolve_inputs(src) -> List[str]:
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


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Convert Alexandria JSON dumps to LMDB.")
    p.add_argument("--src", nargs="+", required=True,
                   help="file(s), directory, or glob of alexandria_*.json(.bz2/.gz)")
    p.add_argument("--out", required=True, help="output directory for *.lmdb shards")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                   help="parallel shards; each holds one file's decoder in memory")
    p.add_argument("--e-above-hull-max", type=float, default=None,
                   help="drop entries above this eV/atom AT CONVERSION TIME "
                        "(default: keep all and filter at load time)")
    p.add_argument("--min-atoms", type=int, default=1)
    p.add_argument("--max-atoms", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="max structures per shard")
    p.add_argument("--overwrite", action="store_true",
                   help="rebuild shards that already exist (default: skip them)")
    args = p.parse_args(argv)

    files = resolve_inputs(args.src)
    if not files:
        print(f"no input files matched {args.src!r}", file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)

    jobs = [(f, args.out, args.e_above_hull_max, args.min_atoms, args.max_atoms,
             args.limit, args.overwrite) for f in files]
    print(f"converting {len(files)} file(s) -> {args.out} with {args.workers} worker(s)",
          flush=True)

    results = []
    if args.workers <= 1:
        for job in jobs:
            results.append(convert_one(job))
            print(" ", results[-1], flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(args.workers, len(jobs)), maxtasksperchild=1) as pool:
            for res in pool.imap_unordered(convert_one, jobs):
                results.append(res)
                print(" ", res, flush=True)

    total = sum(r["kept"] for r in results if r.get("kept") is not None)
    print(f"done: {total} structures across {len(results)} shard(s) in {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())