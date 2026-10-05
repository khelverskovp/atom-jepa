# Atom-JEPA

Embeddings from the pretrained encoders of **Atom-JEPA: Joint-Embedding Predictive
Architecture for 3D Atomistic Systems**.

Atom-JEPA is a self-supervised pretraining method for 3D atomistic systems. It learns
by predicting the latent representations of one part of a structure from another, and
the same recipe applies to both molecules and crystals. This package loads the
pretrained encoders and turns 3D structures into embeddings. Pretraining and
fine-tuning code is in the [GitHub repository](https://github.com/khelverskovp/atom-jepa).

## Installation

Install PyTorch first ([pytorch.org](https://pytorch.org/get-started/locally/)), then:

```bash
pip install atom-jepa            # structures as arrays or ase.Atoms
pip install "atom-jepa[rdkit]"   # + SMILES input
pip install "atom-jepa[all]"     # + SMILES and pymatgen input
```

## Usage

```python
from atom_jepa import AtomJEPA

model = AtomJEPA.from_pretrained("molecules")    # or "crystals"

emb = model.embed("CC(=O)Oc1ccccc1C(=O)O")       # one structure -> [256]
embs = model.embed(["CCO", "c1ccncc1"])          # a list -> [2, 256]
atoms = model.embed("CCO", per_atom=True)        # per-atom features -> [9, 256]
```

The weights (`molecules`, `crystals`) are downloaded from
[huggingface.co/atom-jepa/atom-jepa](https://huggingface.co/atom-jepa/atom-jepa) on
first use and cached locally. A path to a local checkpoint also works.

`embed` accepts any of these as one structure, or a list of them:

| input | notes |
|---|---|
| SMILES string | one RDKit ETKDGv3 + MMFF94 conformer, hydrogens added (needs `[rdkit]`) |
| `ase.Atoms` | periodic if any of its `pbc` flags is set |
| pymatgen `Structure` / `Molecule` | crystals / molecules (needs `[pymatgen]`) |
| `(atomic_numbers, positions[, cell])` | positions in Angstrom; cell with lattice vectors as rows |
| `{"atomic_numbers", "node_coordinates"[, "cell"]}` | the same as a dict |

The embedding is the mean of the per-atom invariant features of the encoder's last
layer. For several conformers per molecule, use
`atom_jepa.conformers.smiles_to_samples(smiles, num_conformers=10)` and embed the list.

## License

The code is released under the MIT License. The pretrained models are released under
CC BY 4.0.
