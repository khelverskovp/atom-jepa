"""Cache frozen Mol-JEPA CLS (512) and modality (12x512) embeddings by exact SMILES.

Run this file in the separate Mol-JEPA environment. Append to existing caches; retry
failed batches individually and store None for failed molecules."""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, REPO)
from finetuning.admet.features.moljepa_features import (  # noqa: E402
    MOLJEPA_REPO,
    MOLJEPA_REVISION,
    moljepa_cache_path,
)

# Match the baseline's chembl_mt_public cache name.
CACHE_DIRS = {"biogen_adme": "data/biogen_adme_features",
              "expansionrx": "data/expansionrx_features",
              "chembl_mt_public": "data/chembl_mt_features"}


def dataset_smiles(dataset: str, args) -> list:
    """Every SMILES string the baseline / HPO may look up for this dataset, de-duplicated."""
    if dataset == "biogen_adme":
        from data.datasets.admet.biogen_adme import load, SMILES_COL
        smiles = []
        for fold in (0, 1):
            d = load(args.biogen_path, split="cluster", cluster_dir=args.cluster_path, cluster_fold=fold)
            smiles += d.df[SMILES_COL].tolist()
            if d.cluster_split is None:
                raise ValueError("Biogen cluster split was not loaded")
            for part in d.cluster_split:
                smiles += part[SMILES_COL].tolist()
    elif dataset == "chembl_mt_public":
        # every molecule the baseline may look up: both folds' train/val pools plus the
        # shared test set, matching _run_chembl_mt's own all_smiles construction
        from data.datasets.admet.chembl_mt import load
        d = load(args.chembl_path, which="public")
        smiles = list(d.test_df["smiles"])
        for fold in (0, 1):
            for part in d.folds[fold]:
                smiles += part["smiles"].tolist()
    else:
        from data.datasets.admet.expansionrx import load, SMILES_COL
        d = load(args.expansionrx_path)
        smiles = d.train_df[SMILES_COL].tolist() + d.test_df[SMILES_COL].tolist()
    return list(dict.fromkeys(s for s in smiles if isinstance(s, str)))


def embed(model, smiles: list, batch_size: int):
    """{smiles: {"cls", "predictions"} | None}, and the number of failures."""
    out, failed, t0 = {}, 0, time.time()

    def run(chunk):
        with torch.no_grad():
            o = model(chunk)
        return o.cls.float().cpu().numpy(), o.predictions.float().cpu().numpy()

    for start in range(0, len(smiles), batch_size):
        chunk = smiles[start:start + batch_size]
        try:
            cls, pred = run(chunk)
            for s, c, p in zip(chunk, cls, pred):
                out[s] = {"cls": c.astype(np.float32), "predictions": p.astype(np.float32)}
        except Exception:
            for s in chunk:                     # one bad SMILES fails the whole batch
                try:
                    cls, pred = run([s])
                    out[s] = {"cls": cls[0].astype(np.float32), "predictions": pred[0].astype(np.float32)}
                except Exception as e:
                    out[s] = None
                    failed += 1
                    if failed <= 10:
                        print(f"[moljepa] cannot embed {s!r}: {type(e).__name__}: {str(e)[:150]}", flush=True)
        done = start + len(chunk)
        if (start // batch_size) % 20 == 0 or done == len(smiles):
            print(f"[moljepa] {done}/{len(smiles)} molecules, {failed} failed, {time.time() - t0:.0f}s",
                  flush=True)
    return out, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(CACHE_DIRS))
    ap.add_argument("--revision", default=MOLJEPA_REVISION, help="Hugging Face commit of the model")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--cache-dir", default=None, help="default: the dataset's feature_cache_dir")
    ap.add_argument("--biogen-path", default="data/biogen_adme")
    ap.add_argument("--cluster-path", default="data/chembl_mt")
    ap.add_argument("--chembl-path", default="data/chembl_mt")
    ap.add_argument("--expansionrx-path", default="data/expansionrx")
    args = ap.parse_args()
    os.chdir(REPO)                              # the data paths above are repo-relative

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    cache_dir = args.cache_dir or CACHE_DIRS[args.dataset]
    path = moljepa_cache_path(cache_dir, args.dataset, args.revision)
    smiles = dataset_smiles(args.dataset, args)

    cache = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            cache = pickle.load(f)
    todo = [s for s in smiles if s not in cache]
    print(f"[moljepa] {args.dataset}: {len(smiles)} molecules, {len(cache)} cached, {len(todo)} to embed "
          f"on {device} -> {path}", flush=True)
    if todo:
        from transformers import AutoModel  # pyright: ignore[reportMissingImports] -- separate Mol-JEPA environment
        model = AutoModel.from_pretrained(MOLJEPA_REPO, revision=args.revision,
                                          trust_remote_code=True).eval().to(device)
        new, failed = embed(model, todo, args.batch_size)
        cache.update(new)
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)                   # atomic publish
        first = next(v for v in new.values() if v is not None)
        print(f"[moljepa] {args.dataset}: wrote {len(cache)} entries ({failed} unfeaturizable -> None); "
              f"cls {first['cls'].shape}, predictions {first['predictions'].shape}", flush=True)


if __name__ == "__main__":
    main()
