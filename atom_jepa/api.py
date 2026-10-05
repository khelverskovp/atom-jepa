"""AtomJEPA: embeddings from a pretrained Atom-JEPA encoder.

    from atom_jepa import AtomJEPA
    model = AtomJEPA.from_pretrained("molecules")      # or "crystals", or a local checkpoint
    emb = model.embed(["CCO", "c1ccccc1"])             # [2, 256]
"""

from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch

from atom_jepa.checkpoint import load_pretrained_encoder
from atom_jepa.data.collate import GraphCollator, move_batch

Sample = Dict[str, torch.Tensor]


def _tensor(x, dtype) -> torch.Tensor:
    """A fresh tensor (inputs such as pymatgen's lattice matrix can be read-only arrays)."""
    if torch.is_tensor(x):
        return x.detach().to(dtype=dtype, device="cpu")
    return torch.tensor(np.array(x), dtype=dtype)


def to_sample(structure: Any) -> Sample:
    """One structure -> {atomic_numbers [N], node_coordinates [N, 3] (Angstrom), cell [3, 3]}.

    Accepts a dict with those keys, a tuple (atomic_numbers, positions[, cell]), an
    ase.Atoms (periodic if any of its pbc flags is set), a pymatgen Structure (periodic)
    or Molecule, or a SMILES string (one ETKDGv3 + MMFF94 conformer, needs RDKit).
    The cell holds the lattice vectors as rows."""
    if isinstance(structure, str):
        from atom_jepa.conformers import smiles_to_samples
        return smiles_to_samples(structure)[0]
    if isinstance(structure, dict):
        z, pos, cell = structure["atomic_numbers"], structure["node_coordinates"], structure.get("cell")
    elif isinstance(structure, (tuple, list)):
        if len(structure) not in (2, 3):
            raise ValueError("a tuple structure is (atomic_numbers, positions[, cell])")
        z, pos, cell = structure[0], structure[1], structure[2] if len(structure) == 3 else None
    elif hasattr(structure, "get_atomic_numbers") and hasattr(structure, "get_positions"):  # ase
        z, pos = structure.get_atomic_numbers(), structure.get_positions()
        cell = structure.cell.array if any(structure.pbc) else None
    elif hasattr(structure, "atomic_numbers") and hasattr(structure, "cart_coords"):  # pymatgen
        z, pos = structure.atomic_numbers, structure.cart_coords
        cell = structure.lattice.matrix if hasattr(structure, "lattice") else None
    else:
        raise TypeError(f"cannot read a structure from {type(structure).__name__}")
    sample = {"atomic_numbers": _tensor(z, torch.long).reshape(-1),
              "node_coordinates": _tensor(pos, torch.float32).reshape(-1, 3)}
    if cell is not None:
        sample["cell"] = _tensor(cell, torch.float32).reshape(3, 3)
    if len(sample["atomic_numbers"]) != len(sample["node_coordinates"]):
        raise ValueError("atomic_numbers and positions have different lengths")
    return sample


class AtomJEPA:
    """A pretrained Atom-JEPA context encoder ready for inference."""

    def __init__(self, encoder, config, device="cpu"):
        self.encoder = encoder.to(device).eval().set_grad_checkpointing(False)
        self.config = config
        self.device = torch.device(device)
        self.collate = GraphCollator(cutoff=config.max_radius,
                                     max_num_elements=config.max_num_elements)

    @classmethod
    def from_pretrained(cls, name_or_path: str = "molecules", device=None) -> "AtomJEPA":
        """'molecules' / 'crystals' (downloaded from huggingface.co/atom-jepa/atom-jepa
        and cached), or a local checkpoint (.pt or a released-format folder).
        device defaults to CUDA when available."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder, config, _ = load_pretrained_encoder(name_or_path, device)
        return cls(encoder, config, device)

    @property
    def embedding_dim(self) -> int:
        return self.config.num_channels

    @torch.inference_mode()
    def embed(self, structures: Union[Any, Sequence[Any]], batch_size: int = 32,
              per_atom: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Embed one structure, or a list of structures.

        Any input to_sample accepts counts as one structure (a tuple is one
        (atomic_numbers, positions[, cell]) structure); pass a list for several.
        Returns the structure embedding [embedding_dim] (the mean of the per-atom
        invariant features), [S, embedding_dim] for a list, or with per_atom=True the
        per-atom features [n_atoms, embedding_dim] (a list of them for a list input).
        Outputs are float32 on the CPU."""
        single = not isinstance(structures, list)
        samples = [to_sample(s) for s in ([structures] if single else structures)]
        for s in samples:
            if s["atomic_numbers"].numel() and int(s["atomic_numbers"].max()) >= self.config.max_num_elements:
                raise ValueError(f"atomic number above the encoder's limit "
                                 f"({self.config.max_num_elements - 1})")
        out = []
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            batch = move_batch(self.collate(chunk), self.device)
            if per_atom:
                node_scalar, _ = self.encoder.encode_nodes(batch)
                sizes = [len(s["atomic_numbers"]) for s in chunk]
                out.extend(node_scalar.float().cpu().split(sizes))
            else:
                out.append(self.encoder(batch).float().cpu())
        if per_atom:
            return out[0] if single else out
        emb = torch.cat(out) if out else torch.zeros(0, self.embedding_dim)
        return emb[0] if single else emb
