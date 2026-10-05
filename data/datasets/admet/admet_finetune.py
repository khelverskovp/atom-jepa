"""
TDC ADMET dataset for SUPERVISED fine-tuning of the EquiformerV3 encoder.

Analogous to data.datasets.qm9.QM9Dataset, but for the Therapeutics Data
Commons ADMET benchmark group. Differences from the QM9 version:

  * Source is SMILES, not 3D structures -> coordinates come from the RDKit
    conformer cache (admet_conformers.build_or_load_conformers).
  * NO atom reference. The QM9 per-element reference energy is meaningless for
    ADMET endpoints, so there is no `atom_ref` here at all.
  * Atoms are carried as `atomic_numbers` (long [N]) taken DIRECTLY from RDKit.
    The encoder embeds by atomic number and ignores `node_features`, so we skip
    the one-hot vocabulary entirely (no make_idx_to_z / SYMBOL_TO_Z step) -- this
    makes element coverage exact and immune to vocab gaps for the wider element
    set in drug-like molecules (S, Cl, Br, I, P, ...). `node_features` is kept
    only as a structural placeholder so data.core.collate.collate_graphs can count nodes.
  * Splits come from TDC's scaffold split by default (train/valid partition of
    train_val is structurally disjoint), not a random index split -- toggled
    via `finetune.scaffold_split` in the Hydra config (see get_train_valid).

MULTI-CONFORMER (Uni-Mol style). The conformer cache now maps each SMILES to a
LIST of 3D conformers: {smiles -> (z [N], [pos_0 .. pos_{k-1}]) | None}. This
dataset consumes them in one of two modes:

  * sample_conformers=True  (TRAIN): one item per MOLECULE; __getitem__ returns a
    RANDOMLY sampled conformer each time it is accessed, so across epochs the model
    sees different geometries of the same molecule (Uni-Mol's per-epoch sampling).
  * sample_conformers=False (EVAL): the dataset is EXPANDED to one item per
    (molecule, conformer). Iterating it in order yields per-graph predictions that
    the caller averages back to one prediction per molecule via `mol_index`
    (Uni-Mol's inference-time averaging). `labels` and `mol_index` are exposed for
    that aggregation.

`num_conformers` caps how many of each molecule's cached conformers to use (None =
all). With num_conformers=1 the dataset reduces to the old single-conformer behavior.

Each emitted item is the encoder input dict plus:
    mol["atomic_numbers"] : [N]  long   true Z per atom (from RDKit)
    mol["y"]              : [1]  float  the label (regression value or 0/1 class)

The radius graph is built with the SAME cutoff the encoder was pretrained with
(passed in by the caller, read from the checkpoint's eqv3_cfg.max_radius).
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from data.core.collate import collate_graphs
from data.core.graphs import radius_graph
from data.datasets.admet.admet_conformers import build_or_load_conformers


def load_admet_group(tdc_path: str):
    """Return a TDC admet_group object (downloads on first use)."""
    try:
        from tdc.benchmark_group import admet_group
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "ADMET fine-tuning needs PyTDC. Install with `pip install PyTDC`."
        ) from e
    return admet_group(path=tdc_path)


def get_test_and_trainval(group, name: str):
    """(canonical_name, train_val_df, test_df) for one benchmark.

    The scaffold test set and the train_val pool are FIXED across split seeds; only
    the train/valid partition of train_val varies. Conformers and the task spec are
    therefore built once from these, before the seed loop."""
    bench = group.get(name)
    return bench["name"], bench["train_val"], bench["test"]


def get_train_valid(group, name: str, seed: int, split_type: str = "scaffold"):
    """TDC's seeded train/valid partition of the (fixed) train_val pool.

    split_type: "scaffold" (default) partitions by Murcko scaffold so the
    validation molecules are structurally distinct from training ones -- this
    is the TDC leaderboard protocol and matches every ADMET benchmark's own
    default split. "random" uses an i.i.d. index split instead (optimistic val
    estimates; useful only as an ablation baseline)."""
    return group.get_train_valid_split(benchmark=name, split_type=split_type, seed=seed)


def featurize_mol(z, pos, cutoff: float) -> Dict[str, torch.Tensor]:
    """Build the encoder input dict from atomic numbers + positions.

    Shared by the dataset and the test-prediction path so both build byte-identical
    radius graphs. `node_features` is a structural placeholder (the encoder embeds
    by `atomic_numbers`); no label is attached here."""
    coords = (pos if torch.is_tensor(pos) else torch.from_numpy(pos)).to(torch.float32)
    z = (z if torch.is_tensor(z) else torch.from_numpy(z)).long()
    edge_index, r, u = radius_graph(coords, float(cutoff))
    return {
        "node_features": torch.ones(z.size(0), 1, dtype=torch.float32),
        "node_coordinates": coords,
        "edge_index": edge_index,
        "edge_lengths": r,
        "edge_vectors": u,
        "atomic_numbers": z,
    }


def _cap(pos_list: List, num_conformers: Optional[int]) -> List:
    """Take the first `num_conformers` conformers (None / <=0 -> all)."""
    if num_conformers is None or num_conformers <= 0:
        return pos_list
    return pos_list[:num_conformers]


class ADMETFinetuneDataset(Dataset):
    """One TDC ADMET split (train/valid/test) as encoder-ready molecule dicts.

    Args:
        df:               TDC dataframe with columns 'Drug' (SMILES) and 'Y' (label)
        conformers:       {smiles -> (z, [pos_0..]) | None} from the conformer cache
        cutoff:           radius-graph cutoff; MUST match the encoder's max_radius
        max_z:            atoms with Z >= max_z are unembeddable; such molecules are
                          skipped + counted.
        num_conformers:   cap on conformers used per molecule (None -> all cached)
        sample_conformers:True  -> one item per molecule, random conformer per access (TRAIN)
                          False -> one item per (molecule, conformer); average via mol_index (EVAL)
        smiles_col / label_col: column names (TDC defaults 'Drug' / 'Y')

    Exposes:
        labels    : [n_molecules] float   per-MOLECULE labels (both modes)
        mol_index : [n_items]      long    molecule index per item (EVAL mode only)
    """

    def __init__(
        self,
        df,
        conformers: Dict[str, Optional[Tuple]],
        cutoff: float,
        max_z: int = 128,
        num_conformers: Optional[int] = None,
        sample_conformers: bool = True,
        smiles_col: str = "Drug",
        label_col: str = "Y",
        tag: str = "",
    ):
        self.cutoff = float(cutoff)
        self.sample_conformers = bool(sample_conformers)
        self.num_conformers = num_conformers

        # per-molecule storage
        self._z: List[torch.Tensor] = []           # [N] long, per molecule
        self._pos: List[List[torch.Tensor]] = []    # list of [N,3] f32 conformers, per molecule
        self._y: List[float] = []                    # label, per molecule

        n_embed_fail = n_oob = 0
        for smi, y in zip(df[smiles_col].tolist(), df[label_col].tolist()):
            conf = conformers.get(smi)
            if conf is None:                          # genuine embed/parse failure (now rare)
                n_embed_fail += 1
                continue
            z_np, pos_list = conf
            z = torch.from_numpy(z_np).long()         # [N]
            if int(z.max()) >= max_z:                 # unembeddable element
                n_oob += 1
                continue
            confs = _cap(pos_list, num_conformers)
            if not confs:                             # nothing usable
                n_embed_fail += 1
                continue
            self._z.append(z)
            self._pos.append([torch.from_numpy(p).to(torch.float32) for p in confs])
            self._y.append(float(y))

        self.labels = torch.tensor(self._y, dtype=torch.float32)   # PER MOLECULE
        self.n_molecules = len(self._z)

        # build the item index for the chosen mode
        if self.sample_conformers:
            self._index: List = list(range(self.n_molecules))      # one per molecule
            self.mol_index = None
            n_graphs = self.n_molecules
        else:
            pairs = [(m, c) for m in range(self.n_molecules)
                            for c in range(len(self._pos[m]))]      # one per (mol, conf)
            self._index = pairs
            self.mol_index = torch.tensor([m for (m, _) in pairs], dtype=torch.long)
            n_graphs = len(pairs)

        mode = "sample" if self.sample_conformers else "expand"
        print(f"[ADMETFinetuneDataset{(' ' + tag) if tag else ''}] "
              f"{self.n_molecules} molecules, {n_graphs} conformer-graphs ({mode}) "
              f"(dropped {n_embed_fail} embed-failed/None, {n_oob} out-of-range-Z)",
              flush=True)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if self.sample_conformers:
            m = self._index[i]
            confs = self._pos[m]
            # random conformer per access; with persistent DataLoader workers the
            # worker RNG advances across epochs, so the model sees varied geometries.
            c = int(torch.randint(len(confs), (1,)).item()) if len(confs) > 1 else 0
        else:
            m, c = self._index[i]
        mol = featurize_mol(self._z[m], self._pos[m][c], self.cutoff)
        mol["y"] = torch.tensor([self._y[m]], dtype=torch.float32)   # [1]
        return mol


def admet_collate(mols: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Base graph collation + the concatenated per-atom `atomic_numbers`.

    data.core.collate.collate_graphs handles node_features/coords/edges/graph_index and
    stacks mol["y"] into [G, 1]. It iterates `mols` in order, so concatenating
    atomic_numbers in the SAME order keeps them row-aligned with the batched
    nodes. We attach Z directly (no one-hot argmax) for exact element fidelity."""
    batch = collate_graphs(mols)
    batch["atomic_numbers"] = torch.cat([m["atomic_numbers"] for m in mols], dim=0)
    return batch