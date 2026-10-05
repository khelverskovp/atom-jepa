"""
Multi-task ADME dataset for feature-vector baselines (Morgan fingerprint +
MLP today; more variants later -- see data/datasets/admet/mol_features.py). A parallel
sibling of data/datasets/admet/multitask_finetune.py's MultiTaskFinetuneDataset (not a
modification of it), for models that need a fixed-length 2D feature vector
per molecule instead of a 3D conformer graph -- no conformers, no radius
graph, no atomic numbers.

Reuses MultiTaskFinetuneDataset's label/mask construction verbatim via the
shared `_build_labels` helper, so `fit_target_stats_mt`/coverage-table calls
in finetuning.admet.training.train.train_multitask (which read `train_ds.labels`/`train_ds.mask`)
work completely unchanged regardless of which Dataset built them.

Each emitted item is:
    mol["features"] : [D]         float   the precomputed feature vector
    mol["y"]         : [n_tasks]  float   label per task, 0.0 where missing
    mol["y_mask"]    : [n_tasks]  float   1.0 where that task's label is present
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from data.datasets.admet.multitask_finetune import _build_labels


class MolecularFeatureDataset(Dataset):
    """One split (train/valid/test) as feature-vector + label dicts.

    Args:
        df:           dataframe with a SMILES column + one column per task
        features:     {smiles -> feature_vector[D] | None} from build_or_load_features
        task_cols:    ordered list of label column names (same task order
                      everywhere downstream, matching MultiTaskFinetuneDataset)
        smiles_col:   SMILES column name
        tag:          construction-summary print tag only

    Exposes:
        labels : [n_molecules, n_tasks] float  NaN where missing
        mask   : [n_molecules, n_tasks] bool   True where a label is present
    """

    def __init__(
        self,
        df,
        features: Dict[str, Optional[np.ndarray]],
        task_cols: List[str],
        smiles_col: str = "smiles",
        tag: str = "",
    ):
        self.task_cols = list(task_cols)

        self._features: List[torch.Tensor] = []   # [D] f32, per molecule
        self._y: List[np.ndarray] = []             # [n_tasks] float, NaN where missing

        y_matrix = df[self.task_cols].to_numpy(dtype=np.float64)   # [n_rows, n_tasks], NaN-preserving
        n_failed = 0
        # Positional index into the INPUT dataframe of every molecule that
        # survived filtering, mirroring MultiTaskFinetuneDataset.keep_index
        # (data/datasets/admet/multitask_finetune.py) exactly. Molecules are dropped silently
        # below (featurization failure), so a caller that needs to scatter
        # this dataset's predictions back into df order -- e.g.
        # finetuning/admet/baseline.py's dataset=admet_tdc path, which must hand TDC's
        # group.evaluate_many a len(test_df)-long array in df order -- cannot
        # do so correctly without this: n_molecules < len(df) whenever any
        # SMILES fails to featurize, so naive positional alignment silently
        # shifts every prediction after the first failure.
        self.keep_index: List[int] = []
        for row_i, (smi, y_row) in enumerate(zip(df[smiles_col].tolist(), y_matrix)):
            feat = features.get(smi)
            if feat is None:                       # unparsable SMILES / featurization failure
                n_failed += 1
                continue
            self._features.append(torch.from_numpy(np.asarray(feat, dtype=np.float32)))
            self._y.append(y_row)
            self.keep_index.append(row_i)

        self.labels, self.mask = _build_labels(self._y, len(self.task_cols))
        self.n_molecules = len(self._features)
        self.feature_dim = int(self._features[0].shape[0]) if self._features else 0

        n_labels = int(self.mask.sum().item())
        print(f"[MolecularFeatureDataset{(' ' + tag) if tag else ''}] "
              f"{self.n_molecules} molecules, {n_labels} labels over "
              f"{len(self.task_cols)} tasks (dropped {n_failed} featurization-failed)",
              flush=True)

    def __len__(self) -> int:
        return self.n_molecules

    def feature_matrix(self) -> np.ndarray:
        """[n_molecules, D] float32 dense matrix of this split's features.

        Two consumers, both of which would otherwise have to reach into the
        private `_features` list: the train-only feature scaler
        (finetuning.admet.baseline.fit_apply_feature_scaler) and the LightGBM path
        (finetuning.admet.training.gbm), which needs a plain dense X rather than a DataLoader."""
        if not self._features:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        return torch.stack(self._features, dim=0).numpy()

    def apply_feature_transform(self, fn) -> None:
        """Replace features in place with `fn(feature_matrix) -> [n_mol, D']`.

        Exists because a scaler must be FIT ON TRAIN ONLY and then applied to
        val/test -- so it cannot be a constructor argument (at construction
        time the train split's statistics don't exist yet). Updates
        `feature_dim` in case the transform changes width."""
        if not self._features:
            return
        out = np.asarray(fn(self.feature_matrix()), dtype=np.float32)
        if out.shape[0] != self.n_molecules:
            raise ValueError(
                f"feature transform changed the molecule count "
                f"({self.n_molecules} -> {out.shape[0]}); it must be row-preserving."
            )
        self._features = [torch.from_numpy(row.copy()) for row in out]
        self.feature_dim = int(out.shape[1])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        y = self.labels[i]                         # [n_tasks], NaN where missing
        mask = self.mask[i]                          # [n_tasks], True where present
        return {
            "features": self._features[i],
            "y": torch.where(mask, y, torch.zeros_like(y)),   # real, finite placeholder under mask=0
            "y_mask": mask.float(),
        }


def feature_collate(mols: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """[G] items -> a batch dict for finetuning.admet.models.baseline_model.FeatureVectorEncoder +
    the UNCHANGED finetuning.admet.models.model.MultiTaskReadout. `node_graph_index`/`num_graphs`
    make each molecule exactly one "node", so MultiTaskReadout's pool_nodes(...)
    call is a true no-op identity pool -- required by its calling convention
    even though there is nothing to pool here."""
    g = len(mols)
    return {
        "features": torch.stack([m["features"] for m in mols], dim=0),   # [G, D]
        "y": torch.stack([m["y"] for m in mols], dim=0),                  # [G, n_tasks]
        "y_mask": torch.stack([m["y_mask"] for m in mols], dim=0),        # [G, n_tasks]
        "node_graph_index": torch.arange(g, dtype=torch.long),
        "num_graphs": g,
    }
