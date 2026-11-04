# Data

```
atom_jepa/data/         dataset-independent pipeline (part of the atom_jepa package)
  elements.py           element symbol <-> atomic number
  graphs.py             radius graphs (molecular and periodic), minimum-image folding
  masking.py            k-hop ego partitions for the JEPA views
  collate.py            samples -> batches: GraphCollator, JEPACollator
  splits.py             seeded random train/val/test splits
  structures.py         pymatgen / matminer helpers for crystals
data/
  datasets/             one file per dataset
    __init__.py         pretraining registry (name -> class)
    qm9.py              QM9           molecules   pretraining, probe, fine-tuning
    unimol.py           Uni-Mol       molecules   pretraining
    alexandria.py       Alexandria    crystals    pretraining
    pcba.py             PCBA          molecules   second-stage pretraining (Atom-JEPA-Bio)
    materials_project.py              crystals    probe during crystal pretraining
    matbench.py         MatBench      crystals    fine-tuning
    admet/              ADMET benchmarks (TDC, Biogen, ChEMBL-MT, CYP, ExpansionRx,
                        MoleculeNet): loaders, RDKit conformers, splits, features
scripts/data/           Alexandria download / filter / LMDB conversion, PCBA preparation,
                        external repositories (setup_external.sh)
```

The ADMET loaders build their graphs in `__getitem__` from cached RDKit
conformers and batch with `atom_jepa.data.collate.collate_graphs`.

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

## PCBA (second-stage pretraining)

Atom-JEPA-Bio continues the molecular encoder on PCBA (1.56M molecules x 1,328
PubChem bioassays) prepared exactly as MONROE prepares it: InChI dedup, a seeded
train/val hash split, and MONROE's assay filter (>= 100 positives, >= 100 negatives and >= 1,000
labels in train), which keeps 1,089 assays. MONROE's own script does this; ours runs it
and then embeds one 3D conformer per molecule.

1. Clone MONROE (and the evaluation repositories) at the release commits, and install
   it in its own Python 3.11 environment:

   ```bash
   bash scripts/data/setup_external.sh                  # -> data/mvc/{monroe,...}
   pip install -e "data/mvc/monroe[preprocessing,eval]" "httpx<0.28" "authlib<1.6"
   polaris login                                         # the Polaris test sets need it
   ```

2. Download `pcba_1328.zip` from [Zenodo](https://zenodo.org/records/8024997), unpack
   `PCBA_1328_1564k.parquet`, and run MONROE's preprocessing in that environment:

   ```bash
   python scripts/data/pcba_monroe_prep.py \
       --pcba-parquet data/pcba/raw/PCBA_1328_1564k.parquet --out-dir data/pcba/monroe
   ```

   It stops if any benchmark test set fails to load (MONROE itself only warns, which
   would change the split), and records the held-out InChIs
   (`benchmark_inchis.json`) and the MONROE commit and package versions
   (`prep_info.json`).

3. Embed the conformers into LMDB shards, one job per MONROE shard (20 train, 1 val),
   in this repository's environment:

   ```bash
   python -m scripts.data.pcba_to_lmdb --monroe-dir data/pcba/monroe --out-dir data/pcba \
       --split train --shard 0 --workers 16          # ... --shard 19
   python -m scripts.data.pcba_to_lmdb --monroe-dir data/pcba/monroe --out-dir data/pcba \
       --split val --shard 0 --workers 16
   ```

`data/pcba` (`conf/data/pcba.yaml`) then holds `manifest.json` (the assays in label
order) and `lmdb/{train,val}/shard_*.lmdb`; samples carry `y` [1089] with NaN for
unmeasured assays. Train with `python -m pretraining.train2` (`conf/pretrain2.yaml`).
