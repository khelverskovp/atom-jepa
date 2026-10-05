"""Read revision-pinned Mol-JEPA caches; missing SMILES require cache regeneration."""

import os
import pickle
from typing import Dict, Iterable, Optional

import numpy as np

MOLJEPA_REPO = "Flogrammer/Mol-JEPA"
MOLJEPA_REVISION = "4c912b450175f31b5ba913a5dc921c03b27b985a"     # Hugging Face commit, 2026-08-19
MOLJEPA_FEATURIZERS = {"moljepa_cls": "cls", "moljepa_modalities": "modalities"}


def moljepa_cache_path(cache_dir: str, name: str, revision: str = MOLJEPA_REVISION) -> str:
    return os.path.join(cache_dir, f"{name}_moljepa_{revision[:10]}.pkl")


def load_moljepa_features(name: str, smiles: Iterable[str], cache_dir: str, output: str = "cls",
                          revision: str = MOLJEPA_REVISION) -> Dict[str, Optional[np.ndarray]]:
    """{smiles -> Mol-JEPA feature vector | None} for every SMILES, from the cache
    finetuning/admet/features/moljepa_embed.py wrote for dataset `name`. `output`: "cls" or "modalities"."""
    if output not in ("cls", "modalities"):
        raise ValueError(f"output must be 'cls' or 'modalities', got {output!r}")
    path = moljepa_cache_path(cache_dir, name, revision)
    rerun = (f"run `python finetuning/admet/features/moljepa_embed.py --dataset {name}` "
             f"in the Mol-JEPA environment")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no Mol-JEPA embedding cache {path} -- {rerun}")
    with open(path, "rb") as f:
        cache = pickle.load(f)
    wanted = list(dict.fromkeys(smiles))
    missing = [s for s in wanted if s not in cache]
    if missing:
        raise KeyError(f"{len(missing)} SMILES not in {path} (e.g. {missing[0]!r}) -- {rerun}")
    out: Dict[str, Optional[np.ndarray]] = {}
    for s in wanted:
        e = cache[s]
        if e is None:
            out[s] = None
        elif output == "cls":
            out[s] = np.asarray(e["cls"], dtype=np.float32)
        else:
            out[s] = np.asarray(e["predictions"], dtype=np.float32).reshape(-1)
    n_none = sum(v is None for v in out.values())
    print(f"[moljepa] {name}: {output} features for {len(out)} molecules from {path}"
          + (f" ({n_none} unfeaturizable -> None)" if n_none else ""), flush=True)
    return out
