"""The model-validation-central atom-jepa study's frozen arm for any encoder checkpoint:
its 02_embed.py and 03_run_tabpfn.py, on the molecules and 25 folds eval_mvc prepared.

  embed   conformer 0 of every molecule through the frozen encoder in fp32, on the plain
          PyTorch path (the study used the cuEquivariance kernels, which in fp32 give the
          same features, but they do not run on every GPU, e.g. V100); the l = 0 features
          of all eight blocks, mean-pooled over atoms, flattened to 8 x 256 = 2,048 numbers
          per molecule.
  tabpfn  per endpoint and fold: drop to the 2,000 highest-variance columns (variance
          over that fold's training rows), then Monroe's fit_predict_tabpfn with its
          default_ensemble_specs, output_type="mean" and the fold seed. Fitted on the
          four training folds only; scored on the fixed ds == test molecules.

Predictions go to runs/<dataset>/predictions/<arm>/ in the study's format, and a fold
counts as done once runs/<dataset>/diagnostics/<arm>_r<r>_f<f>.json exists, so
report_mvc reads the arm like a fine-tuned one.

Needs tabpfn 8.3 (which loads TabPFN 3; pip install tabpfn==8.3.0) and Monroe's code,
the checkout scripts/data/setup_external.sh makes (MONROE_HOME, default
<MVC_ROOT>/monroe):

    python -m finetuning.admet.eval_mvc_tabpfn embed  --dataset biogen --arm A --ckpt CKPT
    python -m finetuning.admet.eval_mvc_tabpfn tabpfn --dataset biogen --arm A
"""

import argparse
import json
import os
import pickle
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from finetuning.admet.eval_mvc import (CONF_SEED, CONFORMER_DIR, DATASETS, ID_COL, MVC_ROOT,
                                       N_CONF, N_REPEATS, N_SPLITS, PRUNE_RMS, SET_COL,
                                       SMILES_COL, fold_seed, out_dir)

EMBED_DIM = 8 * 256
MAX_FEATURES = 2000                 # TabPFN 3's limit
BATCH = 64
PRED_COLUMNS = ["method", "endpoint", "repeat", "fold", ID_COL, SMILES_COL, "y_true", "y_pred"]
MONROE_HOME = Path(os.environ.get("MONROE_HOME", MVC_ROOT / "monroe"))


def embed_path(dataset: str, arm: str) -> Path:
    return out_dir(dataset) / "embeddings" / f"{arm}.npz"


def embed(dataset: str, arm: str, ckpt: str) -> None:
    """02_embed.py."""
    import torch

    from atom_jepa import AtomJEPA

    out = out_dir(dataset)
    df = pd.read_csv(out / "master.csv")
    cache = Path(CONFORMER_DIR) / (f"{DATASETS[dataset]['cache']}_seed{CONF_SEED}"
                                   f"_nconf{N_CONF}_prune{int(PRUNE_RMS)}.pkl")
    with open(cache, "rb") as fh:
        conformers = pickle.load(fh)

    model = AtomJEPA.from_pretrained(ckpt, precision="fp32", cuequivariance=False)
    model.eval()
    smiles = list(dict.fromkeys(df[SMILES_COL]))
    lookup = {}
    start = time.time()
    with torch.no_grad():
        for i in range(0, len(smiles), BATCH):
            chunk = smiles[i:i + BATCH]
            structs = []
            for smi in chunk:
                z, confs = conformers[smi]
                structs.append((np.asarray(z), np.asarray(confs[0], dtype=np.float64)))
            x = model.embed(structs, layers="all").float().reshape(len(chunk), -1).cpu().numpy()
            lookup.update(zip(chunk, x))
            if (i // BATCH) % 20 == 0:
                print(f"  {i + len(chunk):,}/{len(smiles):,}  {time.time() - start:.0f}s", flush=True)

    X = np.stack([lookup[s] for s in df[SMILES_COL]]).astype(np.float32)
    if X.shape[1] != EMBED_DIM:
        raise SystemExit(f"expected {EMBED_DIM}-d embeddings, got {X.shape[1]}")
    if not np.isfinite(X).all():
        raise SystemExit("non-finite values in the embeddings")
    path = embed_path(dataset, arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, names=df[ID_COL].to_numpy().astype(str), ckpt=ckpt)
    print(f"wrote {path} {X.shape} in {time.time() - start:.0f}s", flush=True)


def top_variance(X_fit: np.ndarray, k: int) -> np.ndarray:
    """03_run_tabpfn.py: the k columns that vary most across the fold's training rows."""
    return np.sort(np.argsort(-X_fit.var(axis=0), kind="stable")[:k])


def tabpfn(dataset: str, arm: str, repeats, folds_sel) -> None:
    """03_run_tabpfn.py, all endpoints of a fold, then that fold's completion marker."""
    import tabpfn as tp
    import torch
    from tabpfn.settings import settings
    if settings.tabpfn.model_version.value != "v3":
        raise SystemExit(f"tabpfn {tp.__version__} would load TabPFN "
                         f"{settings.tabpfn.model_version.value}, not v3")
    sys.path.insert(0, str(MONROE_HOME))
    from monroe.eval.tabpfn import default_ensemble_specs, fit_predict_tabpfn

    out = out_dir(dataset)
    df = pd.read_csv(out / "master.csv")
    folds = pd.read_csv(out / "fold_assignments.csv")
    cached = np.load(embed_path(dataset, arm), allow_pickle=False)
    if not np.array_equal(cached["names"], df[ID_COL].to_numpy().astype(str)):
        raise SystemExit("embeddings do not line up with master.csv")
    X, ckpt = cached["X"], str(cached["ckpt"])
    is_test = (df[SET_COL] == "test").to_numpy()
    pred_dir, diag_dir = out / "predictions" / arm, out / "diagnostics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    print(f"tabpfn {tp.__version__} (TabPFN v3), {arm}: {X.shape} from {ckpt}", flush=True)

    for repeat in repeats:
        fold_of = df[ID_COL].map(folds[folds["repeat"] == repeat].set_index(ID_COL)["fold"]).to_numpy()
        for fold in folds_sel:
            diagnostic = diag_dir / f"{arm}_r{repeat}_f{fold}.json"
            if diagnostic.exists():
                print(f"=== r{repeat} f{fold}: done, skipped", flush=True)
                continue
            seed, start, n_fit = fold_seed(repeat, fold), time.time(), {}
            for endpoint in DATASETS[dataset]["targets"]:
                y = df[endpoint].to_numpy()
                measured = df[endpoint].notna().to_numpy()
                fit_mask, test_mask = measured & ~is_test & (fold_of != fold), measured & is_test
                cols = top_variance(X[fit_mask], MAX_FEATURES)
                t0 = time.time()
                pred = fit_predict_tabpfn(X[fit_mask][:, cols], y[fit_mask], X[test_mask][:, cols],
                                          is_classification=False,
                                          ensemble_specs=default_ensemble_specs(),
                                          seed=seed, output_type="mean")
                test_df = df.loc[test_mask]
                pd.DataFrame({"method": arm, "endpoint": endpoint, "repeat": repeat, "fold": fold,
                              ID_COL: test_df[ID_COL].to_numpy(),
                              SMILES_COL: test_df[SMILES_COL].to_numpy(),
                              "y_true": y[test_mask], "y_pred": np.asarray(pred, dtype=float)},
                             columns=PRED_COLUMNS).to_csv(pred_dir / f"{endpoint}_r{repeat}_f{fold}.csv",
                                                          index=False)
                n_fit[endpoint] = int(fit_mask.sum())
                print(f"{endpoint} r{repeat} f{fold}: {n_fit[endpoint]} fit, {int(test_mask.sum())} test, "
                      f"{time.time() - t0:.1f}s", flush=True)
            diagnostic.write_text(json.dumps({
                "arm": arm, "ckpt": ckpt, "repeat": repeat, "fold": fold, "seed": seed,
                "seconds": time.time() - start, "host": socket.gethostname(),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                "head": f"tabpfn {tp.__version__} v3", "n_fit": n_fit}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["embed", "tabpfn"])
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ckpt", help="encoder checkpoint, or a released encoder: 'molecules' / 'bio' (embed)")
    ap.add_argument("--repeat", nargs="+", type=int, default=list(range(N_REPEATS)))
    ap.add_argument("--fold", nargs="+", type=int, default=list(range(N_SPLITS)))
    a = ap.parse_args()
    if a.step == "embed":
        if not a.ckpt:
            ap.error("embed needs --ckpt")
        embed(a.dataset, a.arm, a.ckpt)
    else:
        tabpfn(a.dataset, a.arm, a.repeat, a.fold)


if __name__ == "__main__":
    main()
