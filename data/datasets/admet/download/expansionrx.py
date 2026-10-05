"""
Download the ExpansionRX (OpenADMET) multi-task ADME challenge data.

Source: huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data
(CC-BY-4.0) -- the POST-CHALLENGE release with test labels. A sibling repo,
...-test-data-blinded, has the test labels stripped; that one is NOT what we
want here. Pinned to a specific commit (not `main`) so a future dataset
update doesn't silently change what a cached conformer/training run was built
against -- re-run with --revision to pick up a new one deliberately.

9 endpoints (LogD, KSOL, HLM CLint, MLM CLint, Caco-2 Permeability Papp A>B,
Caco-2 Permeability Efflux, MPPB, MBPB, MGMB), train 5,326 / test 2,282 rows,
temporal split (verified: Molecule Name's numeric ID is monotonically
increasing with row order, and every test ID exceeds every train ID -- do not
re-split). The repo ALSO has `expansion_data_raw.csv` (10 endpoints incl. RLM
CLint, plus censored >/< values) -- deliberately not fetched here; that file
is for a different analysis, not this pipeline's masked-regression setup.

Usage (from the repo root):
  python -m data.datasets.admet.download.expansionrx
"""

import argparse
from pathlib import Path

import requests

REPO = "openadmet/openadmet-expansionrx-challenge-data"
REVISION = "6b898ccc43d10d25b230fb09e22a6e30c30022b5"   # pinned, see module docstring
FILES = ["expansion_data_train.csv", "expansion_data_test.csv"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/expansionrx")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        target = out_dir / name
        if target.exists() and not args.force:
            print(f"[expansionrx] {name}: {target} already exists, skipping (--force to re-fetch)")
            continue
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{args.revision}/{name}"
        print(f"[expansionrx] downloading {url}", flush=True)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        target.write_bytes(resp.content)
        print(f"[expansionrx] wrote {target} ({len(resp.content)} bytes)")


if __name__ == "__main__":
    main()
