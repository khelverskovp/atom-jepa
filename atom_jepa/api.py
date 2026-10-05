"""AtomJEPA: a pretrained Atom-JEPA encoder as a PyTorch module.

    from atom_jepa import AtomJEPA
    model = AtomJEPA.from_pretrained("molecules")      # or "crystals", or a local checkpoint
    emb = model.embed(["CCO", "c1ccccc1"])             # [2, 256]
"""

from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
from torch import nn

from atom_jepa.checkpoint import load_pretrained_encoder
from atom_jepa.data.collate import GraphCollator, move_batch
from atom_jepa.models.jepa_equiformer import pool_nodes

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


class AtomJEPA(nn.Module):
    """A pretrained Atom-JEPA encoder.

    model(batch, ...) returns differentiable features of a batch from model.collate, for
    training your own head or fine-tuning the encoder; embed() is the no-grad shortcut
    from structures. Both take the feature options described in embed()."""

    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder.set_grad_checkpointing(False)
        self.config = config
        self._collator = GraphCollator(cutoff=config.max_radius,
                                       max_num_elements=config.max_num_elements)
        self.amp_dtype = None
        self.execution = {"device": "cpu", "precision": "fp32", "cuequivariance": False,
                          "compile": False}
        self._execution_notes = {}
        self.eval()

    @classmethod
    def from_pretrained(cls, name_or_path: str = "molecules", device=None,
                        precision: str = "auto", cuequivariance: Union[str, bool] = "auto",
                        compile: bool = False, verbose: bool = True) -> "AtomJEPA":
        """'molecules' / 'crystals' (downloaded from huggingface.co/atom-jepa/atom-jepa
        and cached), or a local checkpoint. The model starts in eval mode; call .train()
        to fine-tune it.

        device defaults to CUDA when available. On a CUDA GPU, "auto" runs the encoder in
        bf16 (if supported) with the cuEquivariance kernels (if installed, see
        atom-jepa[cu12] / [cu13]); otherwise in plain fp32 PyTorch. compile=True compiles
        the transformer blocks (CUDA only). The settings are fixed at load time."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder, config, _ = load_pretrained_encoder(name_or_path, device)
        model = cls(encoder, config).to(device)
        model._configure_execution(torch.device(device), precision, cuequivariance, compile)
        if verbose:
            print(f"[atom-jepa] {name_or_path}: {model.execution_summary()}", flush=True)
        return model

    def _configure_execution(self, device, precision, cuequivariance, compile):
        from atom_jepa.cue import enable_cue
        cuda = device.type == "cuda"
        notes = {}

        if precision == "auto":
            precision = "bf16" if cuda and torch.cuda.is_bf16_supported() else "fp32"
            if precision == "fp32":
                notes["bf16"] = "needs a CUDA GPU with bf16 support"
        if precision not in ("bf16", "fp32"):
            raise ValueError(f"precision must be 'auto', 'bf16' or 'fp32', got {precision!r}")
        if precision == "bf16" and not cuda:
            raise ValueError("precision='bf16' needs a CUDA device")

        if cuequivariance not in ("auto", True, False):
            raise ValueError(f"cuequivariance must be 'auto', True or False, got {cuequivariance!r}")
        use_cue = False
        if cuequivariance is not False:
            reason = _cue_unavailable(cuda)
            if reason is None:
                try:
                    enable_cue(self.encoder.body, device=device, encoder_eval_mode=False)
                    use_cue = True
                except (ImportError, ValueError) as e:
                    reason = str(e)
            if reason is not None:
                if cuequivariance is True:
                    raise RuntimeError(f"cuequivariance=True but unavailable: {reason}")
                notes["cuequivariance"] = reason
        if cuda:
            self.encoder.body.set_grid_mlp_optimization(True)

        if compile and not cuda:
            raise ValueError("compile=True needs a CUDA device")
        if compile:
            if use_cue:
                self.encoder.body.edge_degree_embedding.compile(mode="default", dynamic=True)
            self.encoder.body.compile_blocks(mode="default", dynamic=True)

        self.amp_dtype = torch.bfloat16 if precision == "bf16" else None
        self.execution = {"device": str(device), "precision": precision,
                          "cuequivariance": use_cue, "compile": bool(compile)}
        self._execution_notes = notes

    def execution_summary(self) -> str:
        """The execution settings, as printed when loading."""
        e = self.execution
        out = (f"device={e['device']}, precision={e['precision']}, "
               f"cuequivariance={'on' if e['cuequivariance'] else 'off'}, "
               f"compile={'on' if e['compile'] else 'off'}")
        for key, note in self._execution_notes.items():
            out += f"\n  {key} off: {note}"
        if not e["compile"] and e["device"].startswith("cuda"):
            out += "\n  we recommend compile=True for larger jobs"
        return out

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def embedding_dim(self) -> int:
        return self.config.num_channels

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    @property
    def lmax(self) -> int:
        return self.config.lmax

    def set_grad_checkpointing(self, enabled: bool = True) -> "AtomJEPA":
        """Recompute activations in the backward pass to save memory when fine-tuning."""
        self.encoder.set_grad_checkpointing(enabled)
        return self

    def collate(self, samples: Sequence[Any]) -> Dict[str, torch.Tensor]:
        """Batch structures (anything to_sample accepts) or samples from to_sample.

        Extra tensors in a sample dict, e.g. a target "y", are stacked into the batch."""
        samples = [s if isinstance(s, dict) else to_sample(s) for s in samples]
        for s in samples:
            if s["atomic_numbers"].numel() and int(s["atomic_numbers"].max()) >= self.config.max_num_elements:
                raise ValueError(f"atomic number above the encoder's limit "
                                 f"({self.config.max_num_elements - 1})")
        return self._collator(samples)

    def forward(self, batch: Dict[str, torch.Tensor], layers: Union[str, int, Sequence] = "last",
                degrees: Union[str, int, Sequence[int]] = 0, invariant: bool = False,
                per_atom: bool = False) -> torch.Tensor:
        """Features of a batch: [num_structures, ...] mean-pooled over atoms, or
        [num_atoms, ...] with per_atom=True (batch["node_graph_index"] maps atoms to structures)."""
        batch = move_batch(batch, self.device)
        sel = _select_layers(layers, self.num_layers)
        ls = _select_degrees(degrees, self.lmax)

        blocks = sorted({t for t in sel if t != "last"})
        acts = {}
        hooks = [self.encoder.body.blocks[t - 1].register_forward_hook(
            lambda _m, _i, out, t=t: acts.__setitem__(t, out)) for t in blocks]
        try:
            with torch.autocast(self.device.type, dtype=self.amp_dtype,
                                enabled=self.amp_dtype is not None):
                normed, _ = self.encoder.encode_nodes_full(batch)        # [N, sphere, C]
        finally:
            for h in hooks:
                h.remove()
        if len(acts) != len(blocks):
            raise RuntimeError("per-block features need every structure to have edges "
                               "(at least two atoms within the cutoff)")

        sphere = torch.cat([torch.arange(l * l, (l + 1) ** 2) for l in ls]).to(normed.device)
        x = torch.stack([(normed if t == "last" else acts[t]).float() for t in sel], dim=1)[:, :, sphere]
        if not per_atom:
            x = pool_nodes(x, batch["node_graph_index"], batch["num_graphs"], reduce=self.encoder.reduce)
        if invariant:
            parts, start = [], 0
            for l in ls:
                block = x[:, :, start:start + 2 * l + 1]
                parts.append(block if l == 0 else block.norm(dim=2, keepdim=True))
                start += 2 * l + 1
            x = torch.cat(parts, dim=2)
        if layers == "last" or isinstance(layers, int):
            x = x.squeeze(1)
        if isinstance(degrees, int) and (degrees == 0 or invariant):
            x = x.squeeze(-2)
        return x

    @torch.inference_mode()
    def embed(self, structures: Union[Any, Sequence[Any]], layers: Union[str, int, Sequence] = "last",
              degrees: Union[str, int, Sequence[int]] = 0, invariant: bool = False,
              per_atom: bool = False, batch_size: int = 32) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Embed one structure, or a list of them (anything to_sample accepts), in eval mode.

        layers: "last" (block 8 after the final norm, default), a block 1..8 (before the
            final norm), a list of those, or "all" (blocks 1..8); a list or "all" adds a layer axis.
        degrees: 0 (default), an l, a list of them, or "all" (0..lmax); adds an axis with the
            2l+1 (equivariant) components of each l, which degrees=0 drops.
        invariant: replace the l>0 components by their per-channel norm (one entry per l).
        per_atom: per-atom features instead of the mean over atoms.

        Returns float32 CPU tensors: [..., embedding_dim] for one structure, [S, ...] for a
        list; with per_atom, [n_atoms, ...] per structure."""
        single = not isinstance(structures, list)
        samples = [to_sample(s) for s in ([structures] if single else structures)]
        was_training = self.training
        self.eval()
        try:
            out = []
            for start in range(0, len(samples), batch_size):
                chunk = samples[start:start + batch_size]
                x = self(self.collate(chunk), layers, degrees, invariant, per_atom).float().cpu()
                out.extend(x.split([len(s["atomic_numbers"]) for s in chunk]) if per_atom else [x])
        finally:
            self.train(was_training)
        if per_atom:
            return out[0] if single else out
        emb = torch.cat(out)
        return emb[0] if single else emb


def _cue_unavailable(cuda: bool) -> Union[str, None]:
    """Why the cuEquivariance kernels cannot be used here, or None."""
    if not cuda:
        return "needs a CUDA GPU"
    import importlib.util
    if importlib.util.find_spec("cuequivariance_ops_torch") is None:
        major = (torch.version.cuda or "12").split(".")[0]
        return f'kernels not installed (pip install "atom-jepa[cu{major}]")'
    try:
        import atom_jepa.models.equiformer_v3.admet_cue_ops  # noqa: F401
    except ImportError as e:
        return f"kernels failed to load ({e})"
    return None


def _select_layers(layers, num_layers) -> List:
    if layers == "all":
        return list(range(1, num_layers + 1))
    sel = [layers] if isinstance(layers, (str, int)) else list(layers)
    for t in sel:
        if t != "last" and not (isinstance(t, int) and 1 <= t <= num_layers):
            raise ValueError(f"layers must be 'last', 'all' or blocks 1..{num_layers}, got {t!r}")
    return sel


def _select_degrees(degrees, lmax) -> List[int]:
    ls = list(range(lmax + 1)) if degrees == "all" else (
        [degrees] if isinstance(degrees, int) else list(degrees))
    for l in ls:
        if not (isinstance(l, int) and 0 <= l <= lmax):
            raise ValueError(f"degrees must be 'all' or l's in 0..{lmax}, got {l!r}")
    return ls
