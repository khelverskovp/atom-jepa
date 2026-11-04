"""
PCBA preprocessing for second-stage pretraining: MONROE's own pipeline, run strictly.

Atom-JEPA-Bio was trained on PCBA as MONROE prepares it (github.com/blazejba/monroe,
monroe/preprocessing/pcba_prep.py): SMILES -> InChI, dedup by InChI, a seeded DuckDB
hash split into train/val in which every molecule of a Polaris or MoleculeACE test set
goes to val, and MONROE's assay filter on the train rows (>= 100 positives, >= 100
negatives, >= 1,000 labels), which keeps 1,089 of the 1,328 assays.

This script runs that pipeline unchanged, with two additions:
  * it stops if any Polaris or MoleculeACE benchmark fails to load. MONROE only logs a
    warning and carries on, which silently changes the split (Polaris needs a login,
    `polaris login`);
  * it writes <out-dir>/benchmark_inchis.json (the held-out InChIs with their count and
    SHA-256) and <out-dir>/prep_info.json (the MONROE commit and package versions).

It needs MONROE and its preprocessing/benchmark dependencies, best in their own
environment (Python 3.11: MoleculeACE needs rdkit-pypi, which has no 3.12 wheels):

    git clone https://github.com/blazejba/monroe data/mvc/monroe   # or scripts/data/setup_external.sh
    git -C data/mvc/monroe checkout de075eabe98e4adbbedbc86ca9e877f4bfd9b629
    pip install -e "data/mvc/monroe[preprocessing,eval]" "httpx<0.28" "authlib<1.6"
    polaris login

then, from the repository root (PCBA_1328_1564k.parquet is in pcba_1328.zip at
https://zenodo.org/records/8024997):

    python scripts/data/pcba_monroe_prep.py --pcba-parquet data/pcba/raw/PCBA_1328_1564k.parquet \
        --out-dir data/pcba/monroe

All arguments are MONROE's (python -m monroe.preprocessing.pcba_prep --help); its
defaults are the published dataset's. Output: MONROE's layout,
<out-dir>/{train,val}/shards/pcba_shard_*.parquet and manifest.json, which
scripts/data/pcba_to_lmdb.py turns into 3D conformers.
"""

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
from importlib import metadata


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _monroe_commit(pkg_dir):
    try:
        return subprocess.run(["git", "-C", pkg_dir, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out-dir", required=True)
    out_dir = parser.parse_known_args()[0].out_dir

    import monroe
    import monroe.preprocessing.pm6_prep as pm6

    # Every benchmark that fails to load is logged by MONROE as "Warning: Failed to
    # load ...". Collect them, and refuse to split until the list is complete.
    failures = []
    monroe_log = pm6.log

    def log(msg):
        monroe_log(msg)
        if "Failed to load" in str(msg):
            failures.append(str(msg).strip())

    monroe_benchmark_inchis = pm6.get_benchmark_inchis

    def get_benchmark_inchis():
        inchis = monroe_benchmark_inchis()
        if failures:
            raise SystemExit(f"{len(failures)} benchmark(s) failed to load, so the split would "
                             f"differ from MONROE's; fix this (e.g. `polaris login`) and rerun:\n  "
                             + "\n  ".join(failures))
        unique = sorted(set(inchis))
        record = {"n": len(unique),
                  "sha256": hashlib.sha256("\n".join(unique).encode()).hexdigest(),
                  "inchis": unique}
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "benchmark_inchis.json"), "w") as f:
            json.dump(record, f)
        monroe_log(f"benchmark InChIs held out of train: {record['n']:,} (sha256 {record['sha256']})")
        return inchis

    # pcba_prep imports these names from pm6_prep when it runs, and the benchmark
    # loaders look up `log` in pm6_prep, so patching the module covers both
    pm6.log = log
    pm6.get_benchmark_inchis = get_benchmark_inchis

    pkg_dir = os.path.dirname(os.path.abspath(monroe.__file__))
    info = {"monroe_commit": _monroe_commit(pkg_dir), "argv": sys.argv[1:],
            "versions": {p: _version(p) for p in ("duckdb", "datamol", "rdkit", "pandas",
                                                   "polaris-lib", "MoleculeACE")}}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "prep_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    sys.argv = ["monroe.preprocessing.pcba_prep", *sys.argv[1:]]
    runpy.run_module("monroe.preprocessing.pcba_prep", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
