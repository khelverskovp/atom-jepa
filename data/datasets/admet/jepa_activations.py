"""
Frozen-JEPA-encoder activation cache for shallow probes (finetuning.admet.baseline's
LightGBM/MLP battery, applied to the pretrained encoder instead of a
classical fingerprint/descriptor featurizer -- see
finetuning/admet/baseline.py's `_build_features`).

Mirrors data/datasets/admet/mol_features.py::build_or_load_features's cache contract
(`{smiles -> np.ndarray | None}`, atomic pickle write, hash-tagged filename,
never retries a cached failure) but the "compute" step is a GPU, batched, 3D-
conformer-dependent encoder forward pass rather than a per-SMILES CPU joblib
featurizer, so it cannot share build_or_load_features' dispatch itself. This
module's only job is what's genuinely new -- conformer lookup + graph/loader
construction + caching; the actual encoder call is
finetuning.admet.training.utils.pooled_node_features (which itself generalizes probe.py's
encode_nodes loop), not reimplemented here.

Layering: data/ must not import finetuning/admet/ (see
data.datasets.admet.multitask_finetune.MultiTaskFinetuneDataset's own comment on this), so
this module takes an already-loaded `encoder`/`eqv3_cfg` rather than a
ckpt_path -- the one `finetuning.admet.training.utils.load_pretrained_encoder(...)` call lives in
the caller (finetuning/admet/baseline.py), exactly as finetuning/admet/finetune_biogen_adme.py
already does before handing the encoder downstream.
"""

import hashlib
import os
import pickle
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data.datasets.admet.admet_conformers import build_or_load_conformers, DEFAULT_PRUNE_RMS
from data.datasets.admet.multitask_finetune import MultiTaskFinetuneDataset, multitask_collate


def build_or_load_jepa_activations(
    name: str,
    smiles: Iterable[str],
    encoder,
    eqv3_cfg,
    cache_dir: str,
    *,
    layers: str,                       # "last" | "all"
    pool: str = "mean",
    conformer_cache_dir: str,
    conformer_seed: int = 42,
    num_conformers: int = 1,
    conformer_prune_rms: float = DEFAULT_PRUNE_RMS,
    strip_multi_fragment_smiles: bool = True,
    conformer_workers: int = 1,
    batch_size: int = 64,
    device=None,
) -> Dict[str, Optional[np.ndarray]]:
    """Return {smiles -> pooled activation vector | None}, building+persisting
    misses. layers="last": the final block's l=0 scalar (encoder.encode_nodes,
    same quantity every other ADMET driver uses) -- [F]. layers="all": the
    l=0 scalar of EVERY transformer block, depth-concatenated
    (encoder.encode_nodes_all_layers) -- [num_layers*F]. Both mean/sum-pooled
    per molecule via finetuning.admet.training.utils.pooled_node_features -- see its docstring
    for why this is pooled (fixed-width rows, unlike probe.py's per-atom
    caching) and for the extraction itself, which this module does not
    reimplement.

    A molecule whose conformer lookup fails (unparsable SMILES, embed
    failure, unembeddable element) gets None -- same "never fabricate"
    convention as every other featurizer in data/datasets/admet/mol_features.py."""
    layers = str(layers)
    if layers not in ("last", "all"):
        raise ValueError(f"layers must be 'last' or 'all', got {layers!r}")
    pool = str(pool)

    tag = (f"layers={layers},pool={pool},conformer_seed={conformer_seed},"
           f"num_conformers={num_conformers},conformer_prune_rms={conformer_prune_rms},"
           f"strip_multi_fragment_smiles={strip_multi_fragment_smiles}")
    tag_hash = hashlib.sha1(tag.encode()).hexdigest()[:10]
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}_jepa_{layers}_{tag_hash}.pkl")

    cache: Dict[str, Optional[np.ndarray]] = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            cache = pickle.load(f)

    todo = [s for s in dict.fromkeys(smiles) if s not in cache]
    if todo:
        print(f"[jepa_activations] {name}: computing {layers}-layer JEPA activations "
              f"for {len(todo)} new molecule(s) (cache has {len(cache)}) -> {path}", flush=True)

        conformers = build_or_load_conformers(
            name, todo, conformer_cache_dir, seed=conformer_seed, n_conf=num_conformers,
            strip_fragments=strip_multi_fragment_smiles, n_workers=conformer_workers,
            prune_rms=conformer_prune_rms,
        )

        df_todo = pd.DataFrame({"smiles": todo})
        max_z = int(getattr(eqv3_cfg, "max_num_elements", 128))
        ds = MultiTaskFinetuneDataset(
            df_todo, conformers, eqv3_cfg.max_radius, task_cols=[], max_z=max_z,
            num_conformers=1, sample_conformers=False, smiles_col="smiles",
            tag=f"jepa_activation_cache/{name}/{layers}",
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=multitask_collate)

        n_failed = 0
        if ds.n_molecules > 0:
            from finetuning.admet.training.utils import pooled_node_features   # local: avoid data/ -> finetuning/admet/ layering at import time
            pooled = pooled_node_features(encoder, loader, device, layers=layers, pool=pool)
            for row_i, vec in zip(ds.keep_index, pooled):
                cache[todo[row_i]] = vec
        kept = {todo[i] for i in ds.keep_index}
        for smi in todo:
            if smi not in kept:
                cache[smi] = None
                n_failed += 1

        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        print(f"[jepa_activations] {name}: done ({n_failed}/{len(todo)} failed -> None)", flush=True)

    return cache
