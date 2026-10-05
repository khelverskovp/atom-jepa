"""Secondary log10 reporting scales; training and model selection are unchanged.

Clamp labels and predictions to a train-derived positive floor before log10. The paper
does not publish its floor, so exact comparability is not guaranteed.
finetune.report_scale_floor overrides the default minimum positive train label."""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

IDENTITY = "identity"
LOG10 = "log10"
KINDS = (IDENTITY, LOG10)

DEFAULT_FLOOR = 1e-6      # only reached by a task with no positive TRAIN label at all


@dataclass
class ReportScale:
    """Task-aligned transforms and floors; name supplies the metric-key suffix."""

    name: str
    kinds: List[str]
    floors: List[float]

    def __post_init__(self):
        bad = [k for k in self.kinds if k not in KINDS]
        if bad:
            raise ValueError(f"ReportScale: unknown kind(s) {sorted(set(bad))}; "
                             f"expected one of {list(KINDS)}")
        if len(self.kinds) != len(self.floors):
            raise ValueError(f"ReportScale: {len(self.kinds)} kinds but "
                             f"{len(self.floors)} floors")

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transform native [N,T] values to reporting space, preserving NaNs."""
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != len(self.kinds):
            raise ValueError(f"ReportScale.transform expected [N,{len(self.kinds)}], "
                             f"got {y.shape}")
        out = y.copy()
        for t, kind in enumerate(self.kinds):
            if kind == LOG10:
                # Clamp labels and predictions to the same floor, preserving NaNs.
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[:, t] = np.log10(np.maximum(out[:, t], self.floors[t]))
        return out

    def describe(self) -> str:
        n_log = sum(1 for k in self.kinds if k == LOG10)
        return (f"{self.name}: log10 on {n_log}/{len(self.kinds)} task(s), "
                f"floors={[f'{f:g}' for f, k in zip(self.floors, self.kinds) if k == LOG10]}")


def make_log10_report_scale(task_cols: Sequence[str],
                            log10_tasks: Sequence[str],
                            train_labels: np.ndarray,
                            name: str = "kermt",
                            floor: Optional[float] = None) -> ReportScale:
    """Transform designated tasks with log10 and leave other tasks unchanged.

    train_labels has shape [N,T], with NaNs for missing labels. Use floor_override when
    supplied, otherwise each task's minimum positive train label or the fallback."""
    task_cols = list(task_cols)
    wanted = set(log10_tasks)
    unknown = sorted(wanted - set(task_cols))
    if unknown:
        raise ValueError(f"make_log10_report_scale: log10_tasks names not in "
                         f"task_cols: {unknown}")

    labels = np.asarray(train_labels, dtype=np.float64)
    if labels.ndim != 2 or labels.shape[1] != len(task_cols):
        raise ValueError(f"make_log10_report_scale: train_labels must be "
                         f"[N,{len(task_cols)}], got {labels.shape}")

    kinds, floors = [], []
    for t, col in enumerate(task_cols):
        if col not in wanted:
            kinds.append(IDENTITY)
            floors.append(DEFAULT_FLOOR)
            continue
        kinds.append(LOG10)
        if floor is not None:
            floors.append(float(floor))
            continue
        v = labels[:, t]
        pos = v[np.isfinite(v) & (v > 0)]
        floors.append(float(pos.min()) if pos.size else DEFAULT_FLOOR)
    return ReportScale(name=name, kinds=kinds, floors=floors)
