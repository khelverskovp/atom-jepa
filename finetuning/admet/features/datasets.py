"""Feature-dataset helpers for baseline refitting."""

import copy

from data.datasets.admet.feature_finetune import MolecularFeatureDataset
from data.datasets.admet.multitask_finetune import _build_labels


def concat_feature_datasets(a: "MolecularFeatureDataset", b: "MolecularFeatureDataset",
                            tag: str = "") -> "MolecularFeatureDataset":
    """Combine already-scaled train/validation features without fitting new statistics."""
    if list(a.task_cols) != list(b.task_cols):
        raise ValueError(f"task_cols differ: {a.task_cols} vs {b.task_cols}")
    if a.n_molecules and b.n_molecules and a.feature_dim != b.feature_dim:
        raise ValueError(
            f"feature_dim differs ({a.feature_dim} vs {b.feature_dim}) -- both splits "
            "must have had the same feature transform applied before concatenation."
        )
    out = copy.copy(a)                       # keeps task_cols/report_scale/etc.
    out._features = list(a._features) + list(b._features)
    out._y = list(a._y) + list(b._y)
    out.keep_index = []  # Combined training rows have no single source dataframe.
    out.labels, out.mask = _build_labels(out._y, len(out.task_cols))
    out.n_molecules = len(out._features)
    out.feature_dim = int(out._features[0].shape[0]) if out._features else 0
    print(f"[MolecularFeatureDataset{(' ' + tag) if tag else ''}] "
          f"{out.n_molecules} molecules ({a.n_molecules}+{b.n_molecules}), "
          f"{int(out.mask.sum().item())} labels over {len(out.task_cols)} tasks "
          f"(concatenated)", flush=True)
    return out

