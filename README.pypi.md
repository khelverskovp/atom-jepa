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

For several conformers per molecule, use
`atom_jepa.conformers.smiles_to_samples(smiles, num_conformers=10)` and embed the list.

### Feature options

```python
model.embed(smiles, layers="all")                    # [S, 8, 256]     output of each of the 8 blocks
model.embed(smiles, layers=[4, 8])                   # [S, 2, 256]
model.embed(smiles, degrees="all")                   # [S, 9, 256]     l=0, l=1 (3), l=2 (5) components
model.embed(smiles, degrees="all", invariant=True)   # [S, 3, 256]     l=0 and per-channel norms of l=1, l=2
model.embed(smiles, layers="all", degrees="all")     # [S, 8, 9, 256]
model.embed(smiles, per_atom=True)                   # list of [n_atoms, 256]
```

- `layers="last"` (default) is the last block after the encoder's final norm; block
  numbers 1-8 and `"all"` are the block outputs before it.
- Structure features are the mean over atoms. The l>0 components are equivariant: l=1
  rotates like an (x, y, z) vector. `invariant=True` makes them rotation invariant.

### Fine-tuning

`AtomJEPA` is a `torch.nn.Module`: calling it on a batch returns differentiable features
(with the same options), so you can train your own head and fine-tune the encoder:

```python
import torch
from torch.utils.data import DataLoader
from atom_jepa import AtomJEPA, to_sample

model = AtomJEPA.from_pretrained("molecules").train()
head = torch.nn.Linear(model.embedding_dim, 1).to(model.device)

data = [{**to_sample(s), "y": torch.tensor([y])} for s, y in zip(train_smiles, train_y)]
loader = DataLoader(data, batch_size=32, shuffle=True, collate_fn=model.collate)
opt = torch.optim.AdamW([{"params": model.parameters(), "lr": 1e-5},
                         {"params": head.parameters(), "lr": 1e-3}])
for batch in loader:
    loss = torch.nn.functional.mse_loss(head(model(batch)), batch["y"].to(model.device))
    opt.zero_grad(); loss.backward(); opt.step()
```

`model.set_grad_checkpointing(True)` trades compute for memory on large structures.

The package runs the encoder in plain PyTorch (fp32, no cuEquivariance). The faster bf16 /
compiled / cuEquivariance execution and the paper's benchmark recipes are in the
fine-tuning scripts of the [GitHub repository](https://github.com/khelverskovp/atom-jepa#fine-tuning).

## License

The code is released under the MIT License. The pretrained models are released under
CC BY 4.0.
