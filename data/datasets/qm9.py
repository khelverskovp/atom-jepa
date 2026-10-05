"""QM9 (~131k small organic molecules, H/C/N/O/F), via torch_geometric.

One dataset class serves pretraining (no target), the in-loop probe and
fine-tuning (one target).

Sample:
    atomic_numbers   [N] long
    node_coordinates [N, 3] float32 (Angstrom)
    bond_edge_index  [B, 2] long
    y                [1] float32  (only with `target`; torch_geometric units, eV for energies)
    atom_ref         []  float32  (only with `target`; sum over atoms of QM9's
                                   per-element reference for that target, 0 if none)

DEPENDENCY: torch_geometric (downloads QM9 into `root` on first use).
"""

from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

# torch_geometric QM9 y-column indices (the 12 standard targets).
QM9_TARGETS: Dict[str, int] = {
    "mu": 0, "alpha": 1, "homo": 2, "lumo": 3, "gap": 4, "r2": 5,
    "zpve": 6, "U0": 7, "U": 8, "H": 9, "G": 10, "Cv": 11,
}


class QM9Dataset(Dataset):
    """QM9 molecules as samples, optionally labelled with one target.

    Args:
        root:        torch_geometric download/cache directory
        target:      QM9 property to attach as `y` (and its `atom_ref`); None for pretraining
        min_atoms/max_atoms: optional atom-count filtering
        limit:       keep only the first `limit` molecules (after filtering)
        cache:       keep built samples in memory
        raw_dataset: inject a preloaded torch_geometric QM9 (testing)
    """

    periodic = False

    def __init__(
        self,
        root: str = "data/qm9",
        target: Optional[str] = None,
        min_atoms: int = 1,
        max_atoms: Optional[int] = None,
        limit: Optional[int] = None,
        cache: bool = True,
        raw_dataset=None,
    ):
        if target is not None and target not in QM9_TARGETS:
            raise KeyError(f"unknown QM9 target {target!r}; choose from {sorted(QM9_TARGETS)}")
        self.target = target
        self.target_names = (target,) if target is not None else ()
        self.cache = cache

        if raw_dataset is None:
            from torch_geometric.datasets import QM9 as PyGQM9  # lazy import
            raw_dataset = PyGQM9(root=root)
        self.raw = raw_dataset

        # QM9's per-element reference for this target (length-100, indexed by Z);
        # only the atomization energies U0/U/H/G have a non-zero one.
        self.atomref = None
        if target is not None:
            ref = self.raw.atomref(QM9_TARGETS[target])
            self.atomref = ref.view(-1).float() if ref is not None else None
        self.has_atomref = self.atomref is not None and bool(self.atomref.abs().sum() > 0)

        if min_atoms > 1 or max_atoms is not None:
            self.indices = [
                i for i in range(len(self.raw))
                if min_atoms <= int(self.raw[i].z.shape[0])
                and (max_atoms is None or int(self.raw[i].z.shape[0]) <= max_atoms)
            ]
        else:
            self.indices = list(range(len(self.raw)))
        if limit is not None:
            self.indices = self.indices[:limit]

        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        print(f"[QM9Dataset] {len(self.indices)} molecules (target={target}, "
              f"min_atoms={min_atoms}, max_atoms={max_atoms}, limit={limit})", flush=True)

    @classmethod
    def from_config(cls, data_cfg):
        return cls(
            root=data_cfg.root,
            min_atoms=data_cfg.get("min_atoms", 1),
            max_atoms=data_cfg.get("max_atoms", None),
            limit=data_cfg.get("limit", None),
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache and idx in self._cache:
            return self._cache[idx]
        data = self.raw[self.indices[idx]]
        sample = {
            "atomic_numbers": data.z.long(),
            "node_coordinates": data.pos.to(torch.float32),
        }
        # torch_geometric stores bonds as [2, E] in data.z order; samples use [E, 2].
        ei = getattr(data, "edge_index", None)
        if ei is not None and ei.numel() > 0:
            sample["bond_edge_index"] = ei.t().contiguous()
        if self.target is not None:
            sample["y"] = data.y.view(-1)[QM9_TARGETS[self.target]].float().view(1)
            ref_sum = self.atomref[data.z.long()].sum() if self.atomref is not None else torch.zeros(())
            sample["atom_ref"] = ref_sum.float().view(())
        if self.cache:
            self._cache[idx] = sample
        return sample
