"""Atom-JEPA: Joint-Embedding Predictive Architecture for 3D Atomistic Systems.

    from atom_jepa import AtomJEPA
    model = AtomJEPA.from_pretrained("molecules")      # or "crystals"
    emb = model.embed(["CCO", "c1ccccc1"])             # [2, 256]
"""

from atom_jepa.api import AtomJEPA, to_sample
from atom_jepa.checkpoint import PRETRAINED_MODELS, load_pretrained_encoder

__version__ = "0.1.0"
__all__ = ["AtomJEPA", "to_sample", "load_pretrained_encoder", "PRETRAINED_MODELS", "__version__"]
