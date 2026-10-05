"""Seeded random train/val/test index splits."""

from typing import List, Tuple

import torch


def split_indices(n: int, val_frac: float, test_frac: float, seed: int
                  ) -> Tuple[List[int], List[int], List[int]]:
    """Deterministic (train, val, test) index split of range(n); test is taken first."""
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    test_idx = perm[:n_test].tolist()
    val_idx = perm[n_test:n_test + n_val].tolist()
    train_idx = perm[n_test + n_val:].tolist()
    return train_idx, val_idx, test_idx
