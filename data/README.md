# Data

```
data/
  core/                 dataset-independent pipeline
    elements.py         element symbol <-> atomic number
    graphs.py           radius graphs (molecular and periodic), minimum-image folding
    masking.py          k-hop ego partitions for the JEPA views
    collate.py          samples -> batches: GraphCollator, JEPACollator
    splits.py           seeded random train/val/test splits
    structures.py       pymatgen / matminer helpers for crystals
  datasets/             one file per dataset
    __init__.py         pretraining registry (name -> class)
    qm9.py              QM9           molecules   pretraining, probe, fine-tuning
    unimol.py           Uni-Mol       molecules   pretraining
    alexandria.py       Alexandria    crystals    pretraining
    materials_project.py              crystals    probe during crystal pretraining
    matbench.py         MatBench      crystals    fine-tuning
    admet/              ADMET benchmarks (TDC, Biogen, ChEMBL-MT, CYP, ExpansionRx,
                        MoleculeNet): loaders, RDKit conformers, splits, features
scripts/data/           Alexandria download / filter / LMDB conversion
```

The ADMET loaders build their graphs in `__getitem__` from cached RDKit
conformers and batch with `data.core.collate.collate_graphs`.

## Sample format

A dataset's `__getitem__` returns a dict of tensors describing atoms and
geometry only. Graphs are built later, at batching time, with the model's cutoff.

| key | shape | dtype | |
|---|---|---|---|
| `atomic_numbers` | `[N]` | long | required |
| `node_coordinates` | `[N, 3]` | float32, Angstrom | required |
| `cell` | `[3, 3]` | float32, row-vector lattice | crystals only; its presence makes the sample periodic |
| `bond_edge_index` | `[B, 2]` | long | optional covalent bonds; used to cut the k-hop views |
| anything else, e.g. `y` | any | | graph-level labels, stacked by `GraphCollator` |

Molecules without bonds (and all crystals) cut their k-hop views on a radius
graph at `mask.ego_topology_cutoff` instead.

## Adding a pretraining dataset

1. Create `data/datasets/<name>.py` with a `torch.utils.data.Dataset` that
   returns samples in the format above, plus:
   - a class attribute `periodic = True` for crystals, `False` for molecules;
   - a `from_config(cls, data_cfg)` classmethod that builds it from the
     `data:` block of the Hydra config.
2. Register the class in `PRETRAINING_DATASETS` in `data/datasets/__init__.py`.
3. Add `conf/data/<name>.yaml` (copy `qm9.yaml` for molecules or
   `alexandria.yaml` for crystals) with `data.dataset: <name>`, your
   `from_config` arguments, and the `mask.ego_hops` / `mask.ego_topology_cutoff`
   for the views.
4. Train with `python -m pretraining.train data=<name>`.

Nothing else in the training script is dataset-specific: `JEPACollator` builds
the two views and the full-graph target for molecules and crystals alike, and
`dataset.periodic` switches the target-atom query to minimum-image vectors.
