"""Materials Project crystals with a scalar label -- the in-loop probe for crystal
pretraining (the periodic counterpart of the QM9 probe).

A frozen-encoder probe (pretraining.probe.run_crystal_probe) fits the encoder's
atom features to `y` and reports held-out MAE / RMSE / R^2.

Targets (cfg.probe.target):
    "e_form"    formation energy per atom, eV/atom  (matbench_mp_e_form, ~132k)
    "band_gap"  DFT PBE band gap, eV                (matbench_mp_gap, ~106k); zero-
                inflated (metals at 0 eV), so e_form is the cleaner headline metric

Access modes (cfg.probe.mp_source):
    "matminer"  (default, no API key) a matminer release, downloaded + cached
    "mpr"       the Materials Project API via mp_api.MPRester (needs an API key)
    "file"      a local json list of pymatgen Structure dicts, each optionally
                wrapped as {"structure": {...}, "<target>": value}

Sample: atomic_numbers [N], node_coordinates [N, 3], cell [3, 3], y [1].

DEPENDENCIES: pymatgen; plus matminer (matminer mode) or mp-api (mpr mode). All lazy.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from atom_jepa.data.structures import load_matminer, structure_to_sample

# probe-target name -> default matminer release that ships structures + that label.
_TARGET_TO_MATBENCH = {
    "e_form": "matbench_mp_e_form",
    "band_gap": "matbench_mp_gap",
}

# probe-target name -> Materials Project API field (mpr mode).
_MPR_FIELD = {
    "e_form": "formation_energy_per_atom",
    "band_gap": "band_gap",
    "e_above_hull": "energy_above_hull",
}


def _load_mpr(api_key: Optional[str], target: Optional[str], limit: Optional[int],
              search_kwargs: Optional[dict] = None):
    try:
        from mp_api.client import MPRester
    except Exception as e:  # pragma: no cover
        raise ImportError("mp-api is required for mp_source='mpr' (pip install mp-api)") from e
    tgt_field = _MPR_FIELD.get(target) if target else None
    fields = ["structure"] + ([tgt_field] if tgt_field else [])
    kwargs = dict(fields=fields)
    if search_kwargs:
        kwargs.update(search_kwargs)
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(**kwargs)
    pairs = []
    for i, d in enumerate(docs):
        if limit is not None and i >= limit:
            break
        y = getattr(d, tgt_field) if tgt_field else None
        pairs.append((d.structure, y))
    return pairs


def _load_file(file_path: str, target: Optional[str], limit: Optional[int]):
    import json
    from pymatgen.core import Structure
    with open(file_path) as f:
        obj = json.load(f)
    items = obj["structures"] if isinstance(obj, dict) and "structures" in obj else obj
    pairs = []
    for i, it in enumerate(items):
        if limit is not None and i >= limit:
            break
        if isinstance(it, dict) and "structure" in it:
            s = Structure.from_dict(it["structure"])
            y = it.get(target) if target else None
        else:
            s = Structure.from_dict(it)
            y = None
        pairs.append((s, y))
    return pairs


def load_mp_structures(
    source: str = "matminer",
    dataset_name: str = "matbench_mp_e_form",
    api_key: Optional[str] = None,
    target: Optional[str] = None,
    limit: Optional[int] = None,
    file_path: Optional[str] = None,
    search_kwargs: Optional[dict] = None,
) -> List[Tuple[object, Optional[float]]]:
    """Return [(pymatgen Structure, target_value_or_None), ...] for any access mode."""
    source = (source or "matminer").lower()
    if source == "matminer":
        return load_matminer(dataset_name, limit)
    if source in ("mpr", "mp_api", "api"):
        return _load_mpr(api_key, target, limit, search_kwargs)
    if source == "file":
        return _load_file(file_path, target, limit)
    raise ValueError(f"unknown mp_source {source!r}, expected 'matminer', 'mpr', or 'file'")


class MPProbeDataset(Dataset):
    """Labelled Materials Project crystals for the frozen-encoder probe.

    Args:
        target:       "e_form" (default) or "band_gap"
        source:       "matminer" (default) | "mpr" | "file"
        dataset_name: matminer release; defaults from `target` when source="matminer"
        api_key:      Materials Project API key (mpr mode)
        min_atoms/max_atoms/limit: filtering; `limit` caps the probe set size
        file_path:    local json (file mode)
    """

    periodic = True

    def __init__(
        self,
        target: str = "e_form",
        source: str = "matminer",
        dataset_name: Optional[str] = None,
        api_key: Optional[str] = None,
        min_atoms: int = 1,
        max_atoms: Optional[int] = None,
        limit: Optional[int] = None,
        file_path: Optional[str] = None,
    ):
        self.target = target

        if dataset_name is None and source == "matminer":
            dataset_name = _TARGET_TO_MATBENCH.get(target)
            if dataset_name is None:
                raise ValueError(
                    f"no default matminer release for probe target {target!r}; "
                    f"set cfg.probe.mp_dataset explicitly"
                )

        pairs = load_mp_structures(
            source=source, dataset_name=dataset_name, api_key=api_key,
            target=target, limit=None, file_path=file_path,
        )
        self._items: List[Dict[str, torch.Tensor]] = []
        for structure, y in pairs:
            if y is None:
                continue
            sd = structure_to_sample(structure)
            n = int(sd["atomic_numbers"].numel())
            if n < min_atoms or (max_atoms is not None and n > max_atoms):
                continue
            sd["y"] = torch.tensor([float(y)], dtype=torch.float32)
            self._items.append(sd)
            if limit is not None and len(self._items) >= limit:
                break

        print(f"[MPProbeDataset] {len(self._items)} labeled structures "
              f"(target={target}, source={source}, dataset={dataset_name}, "
              f"limit={limit})", flush=True)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._items[idx]
