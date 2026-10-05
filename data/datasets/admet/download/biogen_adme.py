"""
Download the Biogen ADME (Fang et al. 2023) public dataset.

Source: github.com/molecularinformatics/Computational-ADME,
ADME_public_set_3521.csv (3521 rows, 6 log-scale ADME endpoints). Pinned to a
specific commit (not `main`) so a future upstream change doesn't silently
alter what a cached conformer/split/training run was built against -- re-run
with --revision to pick up a new one deliberately.

Usage (from the repo root):
  python -m data.datasets.admet.download.biogen_adme
"""

import argparse
from pathlib import Path

import requests

REPO = "molecularinformatics/Computational-ADME"
REVISION = "685c9c828b23d8672572c21f82bb7e0780dfba2f"   # pinned, see module docstring
FILE = "ADME_public_set_3521.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/biogen_adme")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = out_dir / FILE
    if target.exists() and not args.force:
        print(f"[biogen_adme] {FILE}: {target} already exists, skipping (--force to re-fetch)")
        return
    url = f"https://raw.githubusercontent.com/{REPO}/{args.revision}/{FILE}"
    print(f"[biogen_adme] downloading {url}", flush=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    target.write_bytes(resp.content)
    print(f"[biogen_adme] wrote {target} ({len(resp.content)} bytes)")


if __name__ == "__main__":
    main()
