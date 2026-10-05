"""MatBench (v0.1) structure tasks for fine-tuning.

Samples are crystal samples (atomic_numbers, node_coordinates, cell) plus a
scalar label y [] (0/1 for the classification task).

FOLDS. MatBench's protocol is 5-fold CV over FIXED, published index sets that
live in the `matbench` package, so `load_matbench_fold` goes through
matbench.bench.MatbenchBenchmark by default.

DEPENDENCIES: pymatgen; matbench (preferred) or matminer (fallback). All lazy.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from data.core.collate import GraphCollator
from data.core.structures import load_matminer, structure_to_sample


# --------------------------------------------------------------------------- #
# task registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MatBenchTaskSpec:
    """Everything that differs between the eight MatBench structure tasks.

    column        : matminer/matbench target column name (only used by the fallback
                    loader; the matbench package hands us the target directly).
    unit          : unit AFTER `scale` is applied -- i.e. the unit the paper reports.
    scale         : multiply the native target by this before reporting. Only affects
                    the LOGGED metric, never the training target.
    pool          : "mean" for intensive targets, "sum" for extensive ones.
    classification: BCE-with-logits + F1/ROC-AUC instead of L1 + MAE.
    n_samples     : dataset size, for sanity-checking the load.
    epochs/batch_size/lr : per-task defaults; the 1.3k-sample phonons task and the
                    133k-sample e_form task need very different budgets.
    """

    name: str
    column: str
    unit: str
    pool: str
    n_samples: int
    epochs: int
    batch_size: int
    lr: float
    scale: float = 1.0
    classification: bool = False


MATBENCH_TASKS: Dict[str, MatBenchTaskSpec] = {
    # --- small tasks: heavy regularization, many epochs, small batches ---
    "matbench_phonons": MatBenchTaskSpec(
        name="matbench_phonons", column="last phdos peak", unit="cm^-1",
        pool="mean", n_samples=1_265, epochs=800, batch_size=16, lr=5e-4,
    ),
    "matbench_dielectric": MatBenchTaskSpec(
        name="matbench_dielectric", column="n", unit="",
        pool="mean", n_samples=4_764, epochs=600, batch_size=16, lr=5e-4,
    ),
    "matbench_log_gvrh": MatBenchTaskSpec(
        name="matbench_log_gvrh", column="log10(G_VRH)", unit="log10(GPa)",
        pool="mean", n_samples=10_987, epochs=400, batch_size=32, lr=5e-4,
    ),
    "matbench_log_kvrh": MatBenchTaskSpec(
        name="matbench_log_kvrh", column="log10(K_VRH)", unit="log10(GPa)",
        pool="mean", n_samples=10_987, epochs=400, batch_size=32, lr=5e-4,
    ),
    # perovskites: heat of formation of the WHOLE 5-atom cell -> extensive -> sum pool.
    # paper reports meV, native units are eV, hence scale=1000.
    "matbench_perovskites": MatBenchTaskSpec(
        name="matbench_perovskites", column="e_form", unit="meV",
        pool="sum", n_samples=18_928, epochs=300, batch_size=64, lr=1e-3, scale=1e3,
    ),
    # --- large tasks ---
    "matbench_mp_gap": MatBenchTaskSpec(
        name="matbench_mp_gap", column="gap pbe", unit="eV",
        pool="mean", n_samples=106_113, epochs=150, batch_size=128, lr=1e-3,
    ),
    # e_form is already eV/ATOM -> intensive -> mean pool. paper reports meV/atom.
    "matbench_mp_e_form": MatBenchTaskSpec(
        name="matbench_mp_e_form", column="e_form", unit="meV/atom",
        pool="mean", n_samples=132_752, epochs=150, batch_size=128, lr=1e-3, scale=1e3,
    ),
    "matbench_mp_is_metal": MatBenchTaskSpec(
        name="matbench_mp_is_metal", column="is_metal", unit="F1",
        pool="mean", n_samples=106_113, epochs=150, batch_size=128, lr=1e-3,
        classification=True,
    ),
}

# The eight tasks in the order the paper's Table 4 lists them.
PAPER_TASK_ORDER: Tuple[str, ...] = (
    "matbench_phonons",
    "matbench_dielectric",
    "matbench_log_gvrh",
    "matbench_log_kvrh",
    "matbench_perovskites",
    "matbench_mp_gap",
    "matbench_mp_e_form",
    "matbench_mp_is_metal",
)


# --------------------------------------------------------------------------- #
# fold loading
# --------------------------------------------------------------------------- #
def _load_fold_matbench(task_name: str, fold: int):
    """Official MatBench folds via the `matbench` package."""
    from matbench.bench import MatbenchBenchmark

    mb = MatbenchBenchmark(autoload=False, subset=[task_name])
    task = list(mb.tasks)[0]
    task.load()
    train_in, train_out = task.get_train_and_val_data(fold)
    test_in = task.get_test_data(fold, include_target=False)
    test_out = task.get_test_data(fold, include_target=True)[1]
    return (
        list(train_in), [float(v) for v in train_out],
        list(test_in), [float(v) for v in test_out],
    )


def _load_fold_fallback(task_name: str, fold: int, n_folds: int = 5, seed: int = 18012019):
    """Deterministic KFold over the matminer release. NOT the official split."""
    import numpy as np

    pairs = load_matminer(task_name, limit=None)
    structs = [p[0] for p in pairs]
    ys = [float(p[1]) for p in pairs]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(structs))
    bounds = np.linspace(0, len(structs), n_folds + 1).astype(int)
    test_idx = set(order[bounds[fold]: bounds[fold + 1]].tolist())

    tr = [i for i in range(len(structs)) if i not in test_idx]
    te = sorted(test_idx)
    print(f"[matbench] WARNING: `matbench` package not found -- using a deterministic "
          f"KFold fallback for {task_name}. These numbers are NOT comparable to the "
          f"published leaderboard or to Table 4 of the paper.", flush=True)
    return ([structs[i] for i in tr], [ys[i] for i in tr],
            [structs[i] for i in te], [ys[i] for i in te])


def load_matbench_fold(task_name: str, fold: int, prefer_official: bool = True):
    """Return (train_structs, train_y, test_structs, test_y) for one MatBench fold."""
    if task_name not in MATBENCH_TASKS:
        raise KeyError(f"unknown MatBench task {task_name!r}; "
                       f"expected one of {sorted(MATBENCH_TASKS)}")
    if prefer_official:
        try:
            return _load_fold_matbench(task_name, fold)
        except ImportError:
            pass
    return _load_fold_fallback(task_name, fold)


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class MatBenchDataset(Dataset):
    """Labelled crystal samples from pymatgen Structures.

    Args:
        structures: sequence of pymatgen Structures
        targets:    sequence of floats (bools are cast to 0.0/1.0)
        min_atoms/max_atoms: optional filtering. Filtering changes the test set and
                    invalidates the benchmark, so leave it off for reported numbers.
    """

    periodic = True

    def __init__(
        self,
        structures: Sequence[object],
        targets: Sequence[float],
        min_atoms: int = 1,
        max_atoms: Optional[int] = None,
        tag: str = "",
    ):
        if len(structures) != len(targets):
            raise ValueError(f"{len(structures)} structures vs {len(targets)} targets")

        self._items: List[Dict[str, torch.Tensor]] = []
        n_dropped = 0
        for structure, y in zip(structures, targets):
            sd = structure_to_sample(structure)
            n = int(sd["atomic_numbers"].numel())
            if n < min_atoms or (max_atoms is not None and n > max_atoms):
                n_dropped += 1
                continue
            sd["y"] = torch.tensor(float(y), dtype=torch.float32)
            self._items.append(sd)

        zs = torch.cat([it["atomic_numbers"] for it in self._items]) if self._items \
            else torch.zeros(0, dtype=torch.long)
        self.max_z = int(zs.max()) if zs.numel() else 0
        print(f"[MatBenchDataset{'/' + tag if tag else ''}] {len(self._items)} structures "
              f"(dropped {n_dropped}, max Z present = {self.max_z})", flush=True)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._items[idx]


class MatBenchCollator(GraphCollator):
    """Full periodic graph per structure, labels as y [G, 1]."""

    def __call__(self, samples):
        batch = super().__call__(samples)
        batch["y"] = batch["y"].view(-1, 1)
        return batch
