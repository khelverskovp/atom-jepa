"""pymatgen / matminer helpers shared by the crystal datasets.

DEPENDENCIES: pymatgen; matminer for load_matminer. Both imported lazily.
"""

from typing import Dict, List, Optional, Tuple

import torch

# matminer release -> target column. Unlisted releases fall back to the single
# non-structure column (see load_matminer).
_MATMINER_TARGET_COL = {
    "matbench_mp_e_form": "e_form",        # eV/atom
    "matbench_mp_gap": "gap pbe",          # eV (PBE; zero-inflated -- metals at 0)
    "matbench_perovskites": "e_form",      # eV/atom
    "matbench_log_kvrh": "log10(K_VRH)",
    "matbench_log_gvrh": "log10(G_VRH)",
}


def structure_to_sample(structure) -> Dict[str, torch.Tensor]:
    """pymatgen Structure -> crystal sample (atomic_numbers, node_coordinates, cell)."""
    z = torch.as_tensor(list(structure.atomic_numbers), dtype=torch.long)
    coords = torch.as_tensor(structure.cart_coords, dtype=torch.float32)
    cell = torch.as_tensor(structure.lattice.matrix, dtype=torch.float32)
    return {"atomic_numbers": z, "node_coordinates": coords, "cell": cell}


def load_matminer(dataset_name: str, limit: Optional[int] = None
                  ) -> List[Tuple[object, Optional[float]]]:
    """[(pymatgen Structure, target), ...] for a matminer release (downloads + caches)."""
    try:
        from matminer.datasets import load_dataset
    except Exception as e:  # pragma: no cover
        raise ImportError("matminer is required here (pip install matminer)") from e
    df = load_dataset(dataset_name)

    struct_col = "structure" if "structure" in df.columns else None
    if struct_col is None:
        from pymatgen.core import Structure
        for c in df.columns:
            if len(df) and isinstance(df[c].iloc[0], Structure):
                struct_col = c
                break
    if struct_col is None:
        raise ValueError(f"no pymatgen-Structure column found in '{dataset_name}'")

    tgt_col = _MATMINER_TARGET_COL.get(dataset_name)
    if tgt_col is None or tgt_col not in df.columns:
        others = [c for c in df.columns if c != struct_col]
        tgt_col = others[0] if others else None

    structs = df[struct_col].tolist()
    tgts = df[tgt_col].tolist() if tgt_col is not None else [None] * len(structs)
    pairs = list(zip(structs, tgts))
    if limit is not None:
        pairs = pairs[:limit]
    return pairs
