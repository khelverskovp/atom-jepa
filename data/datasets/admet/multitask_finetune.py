"""
Multi-task ADME dataset for SUPERVISED fine-tuning, for benchmarks with a wide
label MATRIX (one row per molecule, one column per task) rather than a single
scalar label -- currently ChEMBL-MT (data/datasets/admet/chembl_mt.py). A parallel sibling of
data/datasets/admet/admet_finetune.py's ADMETFinetuneDataset (not a modification of it, so the
22 established single-task TDC datasets are unaffected), reusing everything
that doesn't depend on the label shape: conformer lookup, featurize_mol, the
sample-mode (train, one random conformer/epoch) vs expand-mode (val/test, one
item per (molecule, conformer) + mol_index aggregation) split.

The one genuinely new piece is the label MASK: ChEMBL-MT is extremely sparse
(mean 1.12 of 25 labels per molecule, 92.9% of molecules have exactly one), so
every item carries both `y` (label, 0.0 where missing) and `y_mask` (1.0 where
that task's label is actually present for this molecule, 0.0 otherwise). NaN
is deliberately NOT put in `y` itself: `nan * mask` is `nan` regardless of
`mask`'s value (IEEE 754), so a masked multi-task loss needs a real, finite
placeholder under a zero mask weight, not NaN -- see finetune_chembl_mt.py's
masked loss, which relies on exactly this contract.

Each emitted item is the encoder input dict plus:
    mol["y"]      : [n_tasks]  float   label per task, 0.0 where missing
    mol["y_mask"] : [n_tasks]  float   1.0 where that task's label is present
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from atom_jepa.data.collate import collate_graphs
from data.datasets.admet.admet_finetune import featurize_mol, _cap


def _build_labels(y_rows: List[np.ndarray], n_tasks: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack a list of already-filtered [n_tasks] NaN-preserving label rows
    (one per surviving molecule) into (labels [n_mol,n_tasks] float, mask
    [n_mol,n_tasks] bool, True=present). Shared by MultiTaskFinetuneDataset
    below and data.datasets.admet.feature_finetune.MolecularFeatureDataset -- pure
    extraction from this class's former inline logic, no behavior change."""
    labels = torch.from_numpy(np.stack(y_rows)).float() if y_rows else torch.zeros(0, n_tasks)
    mask = ~torch.isnan(labels)
    return labels, mask


class MultiTaskFinetuneDataset(Dataset):
    """One ChEMBL-MT-style split (train/valid/test) as encoder-ready molecule dicts.

    Args:
        df:               dataframe with a SMILES column + one column per task
        conformers:       {smiles -> (z, [pos_0..]) | None} from the conformer cache
        cutoff:           radius-graph cutoff; MUST match the encoder's max_radius
        task_cols:        ordered list of label column names (defines task order
                          everywhere downstream: the model's output columns, the
                          loss's per-task terms, the metric report)
        max_z:            atoms with Z >= max_z are unembeddable; such molecules
                          are skipped + counted
        num_conformers:   cap on conformers used per molecule (None -> all cached)
        sample_conformers:True  -> one item per molecule, random conformer per
                                    access (TRAIN)
                          False -> one item per (molecule, conformer); average
                                    via mol_index (EVAL)
        smiles_col:       SMILES column name (chembl_mt.py uses "smiles")

    Exposes:
        labels    : [n_molecules, n_tasks] float  NaN where missing (per-task
                    train stats, e.g. standardization or a fallback prediction,
                    should use torch.nanmean/nanstd over this -- NOT mol["y"],
                    which has the zero-under-mask placeholder instead of NaN)
        mask      : [n_molecules, n_tasks] bool   True where a label is present
        mol_index : [n_items]               long  molecule index per item (EVAL mode only)
        feature_dim : int  width of mol_features (0 when `features` is None)
    """

    def __init__(
        self,
        df,
        conformers: Dict[str, Optional[Tuple]],
        cutoff: float,
        task_cols: List[str],
        max_z: int = 128,
        num_conformers: Optional[int] = None,
        sample_conformers: bool = True,
        smiles_col: str = "smiles",
        tag: str = "",
        features: Optional[Dict[str, Optional[np.ndarray]]] = None,
    ):
        """
        features: optional {smiles -> feature_vector | None} (see
        data.datasets.admet.mol_features.build_concat_features), for finetuning.admet.training.train's
        encoder-fusion path (finetune.fusion). None (default): every
        __getitem__ item is byte-identical to before this param existed.

        CRITICAL: a molecule whose FEATURE lookup fails (None, e.g. RDKit
        couldn't parse the SMILES) is NEVER dropped from the dataset here --
        only conformer-pipeline failures drop a molecule (see n_embed_fail/
        n_oob below). The conformer and featurizer pipelines can fail on
        DIFFERENT molecules; if a feature failure also dropped the molecule,
        a fusion run's train/val/test molecule set would silently diverge
        from the corresponding no-fusion run's, invalidating any comparison
        between them. Instead a feature failure emits a zero vector plus
        mol_features_valid=0.0 (see __getitem__), which finetuning.admet.models.model's fusion
        head gates to exactly zero contribution for that molecule."""
        self.cutoff = float(cutoff)
        self.task_cols = list(task_cols)
        self.sample_conformers = bool(sample_conformers)
        self.num_conformers = num_conformers

        self._z: List[torch.Tensor] = []            # [N] long, per molecule
        self._pos: List[List[torch.Tensor]] = []     # list of [N,3] f32 conformers, per molecule
        self._y: List[np.ndarray] = []               # [n_tasks] float, NaN where missing, per molecule
        self._feat: Optional[List[Optional[np.ndarray]]] = [] if features is not None else None

        y_matrix = df[self.task_cols].to_numpy(dtype=np.float64)   # [n_rows, n_tasks], NaN-preserving
        n_embed_fail = n_oob = n_feat_fail = 0
        # Positional index into the INPUT dataframe of every molecule that
        # survived filtering. Molecules are dropped silently here (embed
        # failure / unembeddable element), so anything a caller wants to carry
        # alongside `labels` -- e.g. data/datasets/admet/cyp.py's credible-interval bounds for
        # ST-RAE -- must be subset by this or it silently misaligns with the
        # label rows. Cheap to record and impossible to reconstruct afterwards.
        self.keep_index: List[int] = []
        for row_i, (smi, y_row) in enumerate(zip(df[smiles_col].tolist(), y_matrix)):
            conf = conformers.get(smi)
            if conf is None:                          # genuine embed/parse failure
                n_embed_fail += 1
                continue
            z_np, pos_list = conf
            z = torch.from_numpy(z_np).long()
            if int(z.max()) >= max_z:                 # unembeddable element
                n_oob += 1
                continue
            confs = _cap(pos_list, num_conformers)
            if not confs:                              # nothing usable
                n_embed_fail += 1
                continue
            self._z.append(z)
            self._pos.append([torch.from_numpy(p).to(torch.float32) for p in confs])
            self._y.append(y_row)
            self.keep_index.append(row_i)
            if self._feat is not None:
                f = features.get(smi)                 # None on a genuine featurization failure
                if f is None:
                    n_feat_fail += 1
                self._feat.append(f)

        self.labels, self.mask = _build_labels(self._y, len(self.task_cols))
        self.n_molecules = len(self._z)

        self.feature_dim = 0
        if self._feat is not None:
            self.feature_dim = next((int(v.shape[0]) for v in self._feat if v is not None), 0)
            if self.feature_dim == 0 and self.n_molecules > 0:
                raise ValueError(
                    f"MultiTaskFinetuneDataset{(' ' + tag) if tag else ''}: `features` was "
                    "provided but EVERY surviving molecule failed featurization -- there is "
                    "no feature width to derive. Check the featurizer/cache."
                )

        if self.sample_conformers:
            self._index: List = list(range(self.n_molecules))
            self.mol_index = None
            n_graphs = self.n_molecules
        else:
            pairs = [(m, c) for m in range(self.n_molecules)
                            for c in range(len(self._pos[m]))]
            self._index = pairs
            self.mol_index = torch.tensor([m for (m, _) in pairs], dtype=torch.long)
            n_graphs = len(pairs)

        mode = "sample" if self.sample_conformers else "expand"
        n_labels = int(self.mask.sum().item())
        # rank 0 only: under torchrun every rank builds its own copy of this
        # Dataset, so an unguarded print emits world_size identical lines.
        # RANK is read straight from the env rather than importing
        # finetuning.admet.training.distributed -- data/ must not depend on finetuning/admet/.
        if os.environ.get("RANK", "0") == "0":
            feat_msg = (f", {n_feat_fail}/{self.n_molecules} feature-failed->zero+invalid "
                       f"(dim={self.feature_dim})" if self._feat is not None else "")
            print(f"[MultiTaskFinetuneDataset{(' ' + tag) if tag else ''}] "
                  f"{self.n_molecules} molecules, {n_graphs} conformer-graphs ({mode}), "
                  f"{n_labels} labels over {len(self.task_cols)} tasks "
                  f"(dropped {n_embed_fail} embed-failed/None, {n_oob} out-of-range-Z)"
                  f"{feat_msg}",
                  flush=True)

    def feature_matrix(self) -> np.ndarray:
        """[n_valid, D] float32 -- ONLY molecules whose featurization
        succeeded, for fitting a train-only feature scaler
        (finetuning.admet.training.train.train_multitask's finetune.fusion.feature_scaling).
        Molecules with a None feature (a genuine featurization failure --
        see __init__'s docstring) are excluded, never imputed: a scaler fit
        should never be contaminated by a fabricated value, and those
        molecules stay zero+invalid regardless of scaling (the fusion head
        gates them to zero contribution either way)."""
        if self._feat is None:
            return np.zeros((0, 0), dtype=np.float32)
        rows = [f for f in self._feat if f is not None]
        if not rows:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        return np.stack(rows).astype(np.float32)

    def apply_feature_transform(self, fn) -> None:
        """Replace features in place with fn(feature_matrix()) -> [n_valid, D'],
        applied ONLY to molecules whose featurization succeeded. Mirrors
        data.datasets.admet.feature_finetune.MolecularFeatureDataset.apply_feature_transform,
        adapted for the None-preserving invariant this class needs (see
        __init__'s docstring on why a feature failure must never be dropped
        or fabricated here)."""
        if self._feat is None:
            return
        valid_idx = [i for i, f in enumerate(self._feat) if f is not None]
        if not valid_idx:
            return
        X = np.stack([self._feat[i] for i in valid_idx]).astype(np.float32)
        out = np.asarray(fn(X), dtype=np.float32)
        for pos, i in enumerate(valid_idx):
            self._feat[i] = out[pos]
        self.feature_dim = int(out.shape[1])

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if self.sample_conformers:
            m = self._index[i]
            confs = self._pos[m]
            c = int(torch.randint(len(confs), (1,)).item()) if len(confs) > 1 else 0
        else:
            m, c = self._index[i]
        mol = featurize_mol(self._z[m], self._pos[m][c], self.cutoff)
        y = self.labels[m]                        # [n_tasks], NaN where missing
        mask = self.mask[m]                        # [n_tasks], True where present
        mol["y"] = torch.where(mask, y, torch.zeros_like(y))   # real, finite placeholder under mask=0
        mol["y_mask"] = mask.float()
        if self._feat is not None:
            f = self._feat[m]
            if f is None:                          # genuine featurization failure -- see __init__
                mol["mol_features"] = torch.zeros(self.feature_dim, dtype=torch.float32)
                mol["mol_features_valid"] = torch.tensor(0.0)
            else:
                mol["mol_features"] = torch.from_numpy(f).to(torch.float32)
                mol["mol_features_valid"] = torch.tensor(1.0)
        return mol


def multitask_collate(mols: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Base graph collation + concatenated atomic_numbers (as admet_collate) + the
    stacked [G, n_tasks] label mask (atom_jepa.data.collate.collate_graphs already stacks `y`
    generically into [G, P], but has no notion of `y_mask`).

    mol_features/mol_features_valid are stacked CONDITIONALLY, present iff the
    Dataset was built with `features` -- so this collate's output is
    byte-identical to before those keys existed when fusion is off."""
    batch = collate_graphs(mols)
    batch["atomic_numbers"] = torch.cat([m["atomic_numbers"] for m in mols], dim=0)
    batch["y_mask"] = torch.stack([m["y_mask"] for m in mols], dim=0)   # [G, n_tasks]
    if "mol_features" in mols[0]:
        batch["mol_features"] = torch.stack([m["mol_features"] for m in mols], dim=0)          # [G, D]
        batch["mol_features_valid"] = torch.stack([m["mol_features_valid"] for m in mols], dim=0)  # [G]
    return batch
