"""
Download + unpack the ChEMBL-MT multi-task ADME benchmark from Figshare.

ChEMBL-MT is the 25-task regression benchmark from Adrian et al., "Multitask
finetuning and acceleration of chemical pretrained models for small molecule drug
property prediction" (the paper behind NVIDIA's KERMT/Contrastive-KERMT models),
Figshare DOI 10.6084/m9.figshare.30350548.v2. The same release also ships a much
smaller, much denser companion 6-task set ("Biogen") used here as a fast smoke
test of the pipeline before the large, extremely sparse ChEMBL-MT set (25 tasks,
mean 1.12 labels/molecule, 92.9% of molecules have exactly ONE label).

The article's file IDs are resolved dynamically via the Figshare API (not
hardcoded download URLs) so a future dataset version bump doesn't silently break
this script, and each file's MD5 is checked against Figshare's own
`supplied_md5` before unzipping.

Usage (from the repo root):
  python -m data.datasets.admet.download.chembl_mt                    # both splits -> data/chembl_mt/
  python -m data.datasets.admet.download.chembl_mt --which public      # ChEMBL-MT only
  python -m data.datasets.admet.download.chembl_mt --which biogen      # Biogen only
  python -m data.datasets.admet.download.chembl_mt --force             # re-download even if present
"""

import argparse
import hashlib
import io
import zipfile
from pathlib import Path

import requests

FIGSHARE_ARTICLE_ID = 30350548
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"

# Figshare filename -> (short name, extracted subdirectory the zip contains)
SPLITS = {
    "public": ("export_public_cluster_split_figshare.zip", "export_public_cluster_split"),
    "biogen": ("export_biogen_cluster_split_figshare.zip", "export_biogen_cluster_split"),
}


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def fetch_article_files() -> dict:
    """{filename -> figshare file record (download_url, supplied_md5, ...)}."""
    resp = requests.get(FIGSHARE_API, timeout=30)
    resp.raise_for_status()
    return {f["name"]: f for f in resp.json()["files"]}


def download_and_extract(name: str, out_dir: Path, force: bool) -> Path:
    filename, subdir = SPLITS[name]
    target = out_dir / subdir
    if target.exists() and not force:
        print(f"[chembl_mt] {name}: {target} already exists, skipping (--force to re-fetch)")
        return target

    files = fetch_article_files()
    if filename not in files:
        raise KeyError(f"{filename!r} not found in Figshare article {FIGSHARE_ARTICLE_ID}; "
                        f"available: {list(files)}")
    record = files[filename]

    print(f"[chembl_mt] {name}: downloading {filename} ({record['size']} bytes) "
          f"from {record['download_url']}", flush=True)
    resp = requests.get(record["download_url"], timeout=300)
    resp.raise_for_status()
    data = resp.content

    got_md5 = _md5(data)
    want_md5 = record["supplied_md5"]
    if got_md5 != want_md5:
        raise ValueError(f"{filename}: MD5 mismatch (got {got_md5}, expected {want_md5}) -- "
                          f"download may be corrupted or the Figshare file changed.")
    print(f"[chembl_mt] {name}: MD5 verified ({got_md5})", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(out_dir)
    print(f"[chembl_mt] {name}: extracted -> {target}", flush=True)
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/chembl_mt")
    ap.add_argument("--which", choices=["public", "biogen", "both"], default="both")
    ap.add_argument("--force", action="store_true", help="re-download even if already extracted")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    names = ["public", "biogen"] if args.which == "both" else [args.which]
    for name in names:
        target = download_and_extract(name, out_dir, args.force)
        n_csv = len(list(target.glob("*.csv")))
        print(f"[chembl_mt] {name}: {n_csv} CSV file(s) in {target}")


if __name__ == "__main__":
    main()
