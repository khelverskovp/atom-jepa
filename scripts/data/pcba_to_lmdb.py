"""
PCBA molecules from MONROE's preprocessing -> one 3D conformer each -> LMDB shards.
Run after scripts/data/pcba_monroe_prep.py.

    python -m scripts.data.pcba_to_lmdb --monroe-dir data/pcba/monroe --out-dir data/pcba \
        --split train --shard 0 --workers 16

Each call converts one of MONROE's parquet shards of a split
(<monroe-dir>/<split>/shards/pcba_shard_*.parquet, in sorted order; --shard defaults
to $LSB_JOBINDEX - 1 or $SLURM_ARRAY_TASK_ID), so a job array runs the shards in
parallel: 20 train shards and 1 val shard with MONROE's defaults. A finished shard is
skipped on rerun.

Conformers use the ADMET fine-tuning recipe (data/datasets/admet/admet_conformers.py):
largest fragment, explicit Hs, ETKDGv3 + MMFF94, no 2D fallback. Molecules that fail
to embed are dropped and listed in the shard's metadata.

Output, read by data/datasets/pcba.py:
    <out-dir>/manifest.json            assay_ids in label order (MONROE's kept assays)
    <out-dir>/lmdb/<split>/shard_<k>.lmdb
Keys are ascii ints 0..n-1; each value is a pickled dict
    atomic_numbers  uint8 [N]
    coordinates     float32 [N, 3] (Angstrom)
    y               int8 [n_assays]: 1 active, 0 inactive, -1 missing
    smi, inchi      str
    row             int, the molecule's row in its MONROE parquet shard
plus a META_KEY entry {n, failed_rows, source, ...}.
"""

import argparse
import glob
import json
import os
import pickle
import shutil
import tempfile
import time
from multiprocessing import Pool

import lmdb
import numpy as np
import pandas as pd

from data.datasets.admet.admet_conformers import smi2coords

META_KEY = b"__meta__"
MAP_SIZE = 1 << 34          # 16 GiB cap per shard; sparse
COMMIT_EVERY = 4096


def _embed(args):
    row, smi, seed = args
    try:
        res = smi2coords(smi, n_conf=1, seed=seed)
    except Exception:
        res = None
    if res is None:
        return row, None
    z, confs = res
    return row, (z.astype(np.uint8), confs[0])


def _default_shard():
    if "LSB_JOBINDEX" in os.environ:
        return int(os.environ["LSB_JOBINDEX"]) - 1
    return int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))


def write_manifest(out_dir, monroe_dir, assay_ids):
    """<out-dir>/manifest.json with the assay order; every shard job writes or checks it."""
    path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(path):
        with open(path) as f:
            if json.load(f)["assay_ids"] != assay_ids:
                raise SystemExit(f"{path} lists other assays than {monroe_dir}/manifest.json")
        return
    manifest = {"source": "MONROE pcba_prep (scripts/data/pcba_monroe_prep.py)",
                "monroe_dir": os.path.abspath(monroe_dir), "assay_ids": assay_ids}
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=out_dir, delete=False, suffix=".tmp") as f:
        json.dump(manifest, f, indent=1)
    os.replace(f.name, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monroe-dir", required=True, help="output dir of pcba_monroe_prep")
    parser.add_argument("--out-dir", required=True, help="dataset root for data/datasets/pcba.py")
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--shard", type=int, default=_default_shard(),
                        help="index of the MONROE parquet shard of the split")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=42, help="RDKit embedding seed")
    parser.add_argument("--limit", type=int, default=None, help="convert only the first rows (testing)")
    args = parser.parse_args()

    with open(os.path.join(args.monroe_dir, "manifest.json")) as f:
        assay_ids = json.load(f)["assay_cols"]
    write_manifest(args.out_dir, args.monroe_dir, assay_ids)

    sources = sorted(glob.glob(os.path.join(args.monroe_dir, args.split, "shards", "pcba_shard_*.parquet")))
    if not 0 <= args.shard < len(sources):
        raise SystemExit(f"--shard {args.shard}: {args.split} has {len(sources)} MONROE shards")
    src = sources[args.shard]
    out_dir = os.path.join(args.out_dir, "lmdb", args.split)
    out_path = os.path.join(out_dir, f"shard_{args.shard:03d}.lmdb")
    if os.path.exists(out_path):
        print(f"{out_path} exists, skipping", flush=True)
        return

    df = pd.read_parquet(src, columns=["inchi", "SMILES", *assay_ids])
    if args.limit is not None:
        df = df.iloc[:args.limit]
    smiles, inchis = df["SMILES"].tolist(), df["inchi"].tolist()
    y = df[assay_ids].to_numpy(dtype=np.float32)
    labels = np.where(np.isnan(y), -1, y).astype(np.int8)       # 1 / 0 / -1 missing
    del df, y
    print(f"{args.split} shard {args.shard}: {len(smiles):,} molecules from {src} -> {out_path}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    tmp_path = out_path + ".tmp"
    shutil.rmtree(tmp_path, ignore_errors=True)
    env = lmdb.open(tmp_path, map_size=MAP_SIZE, subdir=True, meminit=False, map_async=True)

    t0 = time.time()
    n, failed = 0, []
    txn = env.begin(write=True)
    jobs = ((row, smiles[row], args.seed) for row in range(len(smiles)))
    with Pool(args.workers) as pool:
        for row, res in pool.imap(_embed, jobs, chunksize=16):
            if res is None:
                failed.append(int(row))
                continue
            z, pos = res
            rec = {"atomic_numbers": z, "coordinates": pos, "y": labels[row].copy(),
                   "smi": smiles[row], "inchi": inchis[row], "row": int(row)}
            txn.put(str(n).encode(), pickle.dumps(rec, protocol=pickle.HIGHEST_PROTOCOL))
            n += 1
            if n % COMMIT_EVERY == 0:
                txn.commit()
                txn = env.begin(write=True)
                rate = (n + len(failed)) / (time.time() - t0)
                print(f"  {n + len(failed):,}/{len(smiles):,} ({rate:.0f} mol/s, {len(failed)} failed)", flush=True)

    meta = {"n": n, "failed_rows": failed, "source": os.path.abspath(src), "split": args.split,
            "shard": args.shard, "seed": args.seed}
    txn.put(META_KEY, pickle.dumps(meta))
    txn.commit()
    env.sync()
    env.close()
    os.rename(tmp_path, out_path)
    print(f"done: {n:,} molecules, {len(failed)} failed, {time.time() - t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
