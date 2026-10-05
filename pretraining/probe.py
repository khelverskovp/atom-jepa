"""Frozen-encoder probes run during JEPA pretraining.

    molecular runs (QM9, Uni-Mol)  run_qm9_probe      QM9 properties (cfg.probe.targets)
    crystal runs (Alexandria)      run_crystal_probe  a Materials Project property (cfg.probe.target)

Both freeze the online encoder, cache its l=0 per-atom features for a fixed
train/val/test split, and fit a small per-atom readout on the standardized target. 

"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.core.collate import GraphCollator
from data.core.splits import split_indices

# QM9 single-atom reference energies in eV, keyed by atomic number (the values
# torch_geometric's QM9.atomref() ships). y_atomization = y - sum_i atomref[Z_i].
ATOMREF_EV: Dict[str, Dict[int, float]] = {
    "U0": {1: -13.61312172, 6: -1029.86312267, 7: -1485.30251237,
           8: -2042.61123593, 9: -2713.48485589},
    "U": {1: -13.57459040, 6: -1029.82456413, 7: -1485.26398105,
          8: -2042.57270460, 9: -2713.44632457},
    "H": {1: -13.54887564, 6: -1029.79887659, 7: -1485.23829350,
          8: -2042.54701705, 9: -2713.42063702},
    "G": {1: -13.90303183, 6: -1030.25891228, 7: -1485.71166277,
          8: -2043.01812778, 9: -2713.88796536},
}


@dataclass
class ProbeHParams:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    hidden_dim: Optional[int] = None      # None -> C
    dropout: float = 0.0
    pool: str = "sum"
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 0
    num_workers: int = 0

    @classmethod
    def from_cfg(cls, cfg, target: str, default_pool: str) -> "ProbeHParams":
        """cfg.probe.* over the defaults; cfg.probe.pool_per_target[target] over cfg.probe.pool."""
        p, d = cfg.probe, cls()
        pool = str(p.get("pool", default_pool))
        pool = dict(p.get("pool_per_target", {}) or {}).get(target, pool)
        return cls(
            epochs=int(p.get("epochs", d.epochs)),
            lr=float(p.get("lr", d.lr)),
            weight_decay=float(p.get("weight_decay", d.weight_decay)),
            batch_size=int(p.get("batch_size", d.batch_size)),
            hidden_dim=(int(p["hidden_dim"]) if p.get("hidden_dim", None) is not None else None),
            dropout=float(p.get("dropout", d.dropout)),
            pool=pool,
            val_frac=float(p.get("val_frac", d.val_frac)),
            test_frac=float(p.get("test_frac", d.test_frac)),
            seed=int(cfg.misc.get("seed", d.seed)),
            num_workers=int(p.get("num_workers", d.num_workers)),
        )

    def describe(self) -> str:
        return (f"epochs={self.epochs} lr={self.lr:g} wd={self.weight_decay:g} "
                f"bs={self.batch_size} hidden={self.hidden_dim or 'C'} "
                f"dropout={self.dropout:g} pool={self.pool} seed={self.seed}")


# --------------------------------------------------------------------------- #
# frozen features
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cache_node_features(encoder, dataset, indices, cfg, device, hp: ProbeHParams
                        ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """Frozen l=0 per-atom features of the samples in `indices`.

    Returns (feats, z, y): per-sample [n_atoms, C] features and [n_atoms] atomic
    numbers, and the [M] labels, all on CPU.
    """
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=hp.batch_size,
        shuffle=False,
        num_workers=hp.num_workers,
        collate_fn=GraphCollator(cutoff=float(cfg.model.cutoff),
                                 max_num_elements=int(cfg.model.get("max_num_elements", 128))),
        pin_memory=(device.type == "cuda"),
    )
    was_training = encoder.training
    encoder.eval()
    feats, zs, ys = [], [], []
    for batch in loader:
        y = batch.pop("y")
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        node_scalar, _ = encoder.encode_nodes(batch)                 # [N, C]
        # the collator keeps each sample's atoms contiguous
        counts = torch.bincount(batch["node_graph_index"], minlength=batch["num_graphs"]).tolist()
        feats.extend(torch.split(node_scalar.float().cpu(), counts))
        zs.extend(torch.split(batch["atomic_numbers"].cpu(), counts))
        ys.append(y.float().view(-1))
    if was_training:
        encoder.train()
    return feats, zs, torch.cat(ys)


# --------------------------------------------------------------------------- #
# readout
# --------------------------------------------------------------------------- #
class ScalarReadout(nn.Module):
    """EquiformerV3-style scalar head on frozen atom features, sum/mean pooled."""

    def __init__(self, num_channels: int, hidden: Optional[int] = None,
                 dropout: float = 0.0, pool: str = "sum"):
        super().__init__()
        assert pool in ("sum", "mean"), f"unknown pool {pool!r}"
        self.pool = pool
        hidden = hidden if hidden is not None else num_channels
        self.linear_1 = nn.Linear(num_channels, hidden, bias=True)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.linear_2 = nn.Linear(hidden, 1, bias=True)

    def forward(self, feats, gidx, num_graphs):
        y = self.linear_2(self.dropout(self.act(self.linear_1(feats))))   # [N, 1]
        out = feats.new_zeros(num_graphs, 1)
        out.index_add_(0, gidx, y)
        if self.pool == "mean":
            counts = feats.new_zeros(num_graphs, 1)
            counts.index_add_(0, gidx, feats.new_ones(feats.size(0), 1))
            out = out / counts.clamp_min(1.0)
        return out.squeeze(-1)                                            # [G]


def _assemble(feats: List[torch.Tensor], idx, device):
    """Concatenate the atom features of samples `idx` with a per-sample graph index."""
    chunks = [feats[k] for k in idx]
    sizes = torch.tensor([c.size(0) for c in chunks])
    x = torch.cat(chunks, 0).to(device, non_blocking=True)
    gidx = torch.repeat_interleave(torch.arange(len(idx), device=device), sizes.to(device))
    return x, gidx, len(idx)


@torch.no_grad()
def _evaluate(head, feats, y, device, bs, y_mean, y_std) -> Dict[str, float]:
    """MAE / RMSE / R^2 in raw target units (the head predicts standardized values)."""
    head.eval()
    preds = []
    for i in range(0, len(feats), bs):
        x, gidx, ng = _assemble(feats, range(i, min(i + bs, len(feats))), device)
        preds.append((head(x, gidx, ng) * y_std + y_mean).cpu())
    err = torch.cat(preds) - y
    denom = (y - y.mean()).pow(2).sum().clamp_min(1e-12)
    return {"mae": err.abs().mean().item(),
            "rmse": err.pow(2).mean().sqrt().item(),
            "r2": (1.0 - err.pow(2).sum() / denom).item()}


def fit_readout(splits, hp: ProbeHParams, device) -> Dict[str, float]:
    """Fit a ScalarReadout on splits = {"train"|"val"|"test": (feats, y)}.

    Returns val MAE and test MAE / RMSE / R^2 at the best-validation epoch.
    """
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits["train"], splits["val"], splits["test"]
    y_mean, y_std = float(ytr.mean()), float(ytr.std().clamp_min(1e-8))

    gen = torch.Generator().manual_seed(hp.seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(hp.seed)
        head = ScalarReadout(Xtr[0].size(1), hidden=hp.hidden_dim, dropout=hp.dropout, pool=hp.pool)
    head = head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    best = {"val_mae": float("inf"), "test_mae": float("nan"), "test_rmse": float("nan"),
            "test_r2": float("nan"), "best_epoch": -1.0}
    bs = hp.batch_size
    for epoch in range(hp.epochs):
        for pg in opt.param_groups:
            pg["lr"] = 0.5 * hp.lr * (1.0 + math.cos(math.pi * epoch / max(1, hp.epochs)))
        head.train()
        perm = torch.randperm(len(Xtr), generator=gen).tolist()
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            x, gidx, ng = _assemble(Xtr, idx, device)
            yb = ((ytr[idx] - y_mean) / y_std).to(device)
            loss = ((head(x, gidx, ng) - yb) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        val_mae = _evaluate(head, Xva, yva, device, bs, y_mean, y_std)["mae"]
        if val_mae < best["val_mae"]:
            test = _evaluate(head, Xte, yte, device, bs, y_mean, y_std)
            best = {"val_mae": val_mae, "test_mae": test["mae"], "test_rmse": test["rmse"],
                    "test_r2": test["r2"], "best_epoch": float(epoch)}
    return best


def _cache_splits(encoder, dataset, cfg, device, hp: ProbeHParams):
    tr, va, te = split_indices(len(dataset), hp.val_frac, hp.test_frac, hp.seed)
    if not va or not te:
        raise ValueError(f"probe split left an empty val/test set ({len(tr)}/{len(va)}/{len(te)} "
                         f"of {len(dataset)}); raise the fractions or the probe limit.")
    print(f"[probe] split(seed={hp.seed}): {len(tr)} train / {len(va)} val / {len(te)} test",
          flush=True)
    return {name: cache_node_features(encoder, dataset, idx, cfg, device, hp)
            for name, idx in (("train", tr), ("val", va), ("test", te))}


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def _subtract_atomref(y: torch.Tensor, zs: List[torch.Tensor], target: str) -> torch.Tensor:
    """y - sum_i atomref[Z_i] for the QM9 energies; in float64, since U0 ~ -1e4 eV
    and float32 only resolves ~1e-3 eV there."""
    table = ATOMREF_EV.get(target)
    if table is None:
        return y
    lut = torch.zeros(128, dtype=torch.float64)
    for z, v in table.items():
        lut[z] = v
    offsets = torch.stack([lut[z.long()].sum() for z in zs])
    return (y.double() - offsets).float()


def run_qm9_probe(encoder, datasets: Dict[str, object], cfg, device) -> Dict[str, float]:
    """Molecular pretraining probe. datasets = {QM9 target name: QM9Dataset(target=...)}."""
    metrics: Dict[str, float] = {}
    for target, dataset in datasets.items():
        hp = ProbeHParams.from_cfg(cfg, target, default_pool="sum")
        print(f"[probe] target={target!r} | {hp.describe()}", flush=True)
        caches = _cache_splits(encoder, dataset, cfg, device, hp)
        splits = {name: (feats, _subtract_atomref(y, zs, target))
                  for name, (feats, zs, y) in caches.items()}
        m = fit_readout(splits, hp, device)
        metrics[f"probe/{target}_val_mae"] = m["val_mae"]
        metrics[f"probe/{target}_test_mae"] = m["test_mae"]
        metrics[f"probe/{target}_best_epoch"] = m["best_epoch"]
    return metrics


def run_crystal_probe(encoder, dataset, cfg, device) -> Dict[str, float]:
    """Crystal pretraining probe on one labelled dataset (MPProbeDataset)."""
    if len(dataset) == 0:
        return {}
    target = str(cfg.probe.target)
    hp = ProbeHParams.from_cfg(cfg, target, default_pool="mean")
    print(f"[probe] target={target!r} | {hp.describe()}", flush=True)
    caches = _cache_splits(encoder, dataset, cfg, device, hp)
    m = fit_readout({name: (feats, y) for name, (feats, _, y) in caches.items()}, hp, device)
    return {f"probe/{target}_{k}": v for k, v in m.items()}
