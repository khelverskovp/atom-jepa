"""Device, checkpoint, feature, and dataset helpers shared by ADMET drivers."""

import os
from dataclasses import asdict, fields, replace
from typing import Callable, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from finetuning.common import read_checkpoint
from finetuning.optim import ema_update as ema_update

from models.jepa_equiformer import EquiformerV3Config, EquiformerV3Encoder, pool_nodes

# Omit placeholder node features and coordinates when using precomputed GPU edges.
_SKIP_KEYS = {"node_features", "node_coordinates"}


def move_batch(batch, device):
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
        if k not in _SKIP_KEYS
    }


def resolve_device(name):
    """Resolve CUDA to LOCAL_RANK under torchrun; fall back to CPU when CUDA is
    unavailable."""
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if name == "cuda":
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            return f"cuda:{int(local_rank)}"
    return name


def _better(new: float, best: float, mode: str) -> bool:
    if not np.isfinite(new):
        return False
    return new > best if mode == "max" else new < best


def _worst(mode: str) -> float:
    return float("-inf") if mode == "max" else float("inf")


def _cfg_from_ckpt(ckpt) -> EquiformerV3Config:
    raw = None
    for key in ("eqv3_cfg", "model_cfg", "cfg", "gnn_cfg"):
        if key in ckpt and ckpt[key] is not None:
            raw = ckpt[key]
            break
    if raw is None:
        raise KeyError(
            "checkpoint has no encoder config; expected one of "
            "'eqv3_cfg'/'model_cfg'/'cfg'."
        )
    if isinstance(raw, EquiformerV3Config):
        return raw
    valid = {f.name for f in fields(EquiformerV3Config)}
    return EquiformerV3Config(**{k: v for k, v in dict(raw).items() if k in valid})


def extract_encoder_state_dict(ckpt: dict, verbose: bool = True):
    """Extract encoder weights from a pretrained or fine-tuned checkpoint.

    For full model state dictionaries, retain encoder.* keys, remove their prefix, and
    discard the task head. Pretrained encoder state dictionaries pass through."""
    sd = None
    for key in ("context_encoder", "encoder", "model"):
        if key in ckpt and ckpt[key] is not None:
            sd = ckpt[key]
            break
    if sd is None:
        raise KeyError("checkpoint has no encoder weights; expected 'context_encoder'.")

    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    if enc:
        dropped = sorted({k.split(".")[0] for k in sd if not k.startswith("encoder.")})
        if verbose:
            print(f"[encoder] fine-tuned-model checkpoint: kept {len(enc)} encoder "
                  f"tensor(s), dropped module(s) {dropped} (encoder-only transfer)",
                  flush=True)
            # Retain source task metadata rather than inheriting unrelated
            # configuration.
            meta = {k: ckpt[k] for k in ("target", "unit", "pool", "standardize",
                                         "use_atom_ref") if k in ckpt}
            if meta:
                print(f"[encoder] source task metadata (NOT inherited; your "
                      f"finetune.* config wins): {meta}", flush=True)
        return enc
    return sd


def load_pretrained_encoder(ckpt_path: str, device, load_weights: bool = True
                            ) -> Tuple[EquiformerV3Encoder, EquiformerV3Config, dict]:
    """Rebuild the encoder from checkpoint architecture metadata; load_weights=False leaves
    random weights."""
    ckpt = read_checkpoint(ckpt_path, device)
    cfg = _cfg_from_ckpt(ckpt)
    encoder = EquiformerV3Encoder(cfg).to(device)
    if load_weights:
        sd = extract_encoder_state_dict(ckpt)
        try:
            encoder.load_state_dict(sd, strict=True)
        except RuntimeError:
            encoder.body.load_state_dict(sd, strict=True)   # bare backbone fallback
    return encoder, cfg, ckpt


def build_equiformer_multitask_factory(cfg, eqv3_cfg: EquiformerV3Config, encoder_ckpt,
                                       device) -> Tuple[Callable[[], nn.Module], int, dict]:
    """Return (encoder_factory, F_in, encoder_config) for multi-task training.

    Apply cutoff, checkpointing, and drop-path overrides. Each factory call builds a
    fresh encoder and loads pretrained weights unless train_from_scratch is enabled."""
    ftc = cfg.finetune
    cutoff_override = ftc.get("cutoff", None)
    if cutoff_override is not None:
        eqv3_cfg = replace(eqv3_cfg, max_radius=float(cutoff_override))
    eqv3_cfg = replace(eqv3_cfg, grad_checkpointing=bool(ftc.get("grad_checkpointing", False)))
    # Frozen encoders still have stochastic drop-path unless put in eval mode.
    drop_path_override = ftc.get("encoder_drop_path_rate", None)
    if drop_path_override is not None:
        eqv3_cfg = replace(eqv3_cfg, drop_path_rate=float(drop_path_override))
    train_from_scratch = bool(ftc.get("train_from_scratch", False))

    # Strip encoder prefixes from full-model checkpoints before rebuilding the encoder.
    enc_sd = (extract_encoder_state_dict({"model": encoder_ckpt}, verbose=False)
              if encoder_ckpt is not None else None)

    def encoder_factory() -> nn.Module:
        encoder = EquiformerV3Encoder(eqv3_cfg).to(device)
        if not train_from_scratch:
            if enc_sd is None:
                raise ValueError("Pretrained encoder weights are required unless train_from_scratch is enabled")
            try:
                encoder.load_state_dict(enc_sd, strict=True)
            except RuntimeError:
                encoder.body.load_state_dict(enc_sd, strict=True)
        return encoder

    return encoder_factory, eqv3_cfg.num_channels, asdict(eqv3_cfg)


def build_fusion_features(cfg, name: str, all_smiles):
    """Build concatenated fusion features, or return None when fusion is disabled.

    Read per-featurizer options from finetune.fusion and share feature caches with
    baseline models. Missing option blocks use the featurizer defaults."""
    fusion_cfg = cfg.finetune.get("fusion", {})
    if not bool(fusion_cfg.get("enabled", False)):
        return None
    from data.datasets.admet.mol_features import build_concat_features   # local: only needed when fusion is on
    names = [str(s) for s in fusion_cfg.get("featurizers", ["rdkit_2d_normalized"])]
    workers = int(cfg.data.get("feature_workers", 1))
    kwargs_by_name = {}
    for fname in names:
        node = fusion_cfg.get(fname, None)
        kwargs_by_name[fname] = OmegaConf.to_container(node, resolve=True) if node is not None else {}
    return build_concat_features(name, all_smiles, cfg.data.feature_cache_dir, names,
                                 kwargs_by_name, n_workers=workers)


def _with_higher_order(node_full: torch.Tensor) -> torch.Tensor:
    """Convert [N, L, sphere, C] irreps to [N, L, (1+lmax)*C] invariants.

    Concatenate l=0 scalars with per-degree norms. Derive lmax from the sphere size and
    reject non-square sizes."""
    from finetuning.admet.models.model import (
        per_degree_invariant_norms,  # local: avoids an
    )
    # import cycle at module scope if finetuning.admet.models.model ever imports this module.
    n, L, sphere, C = node_full.shape
    lmax = int(round(sphere ** 0.5)) - 1
    if (lmax + 1) ** 2 != sphere:
        raise ValueError(f"irrep dim {sphere} is not (lmax+1)**2 for any integer lmax")
    if lmax == 0:                                   # scalar-only encoder: nothing to add
        return node_full[:, :, 0, :]
    flat = node_full.reshape(n * L, sphere, C)
    ho = per_degree_invariant_norms(flat, lmax)      # [N*L, lmax*C]
    out = torch.cat([flat[:, 0, :], ho], dim=-1)     # [N*L, (1+lmax)*C]
    return out.reshape(n, L, (1 + lmax) * C)


@torch.no_grad()
def pooled_node_features(encoder, loader, device, layers: str = "last", pool: str = "mean",
                         higher_order: bool = False):
    """Return pooled molecule features in loader order as a float32 numpy array.

    layers="last" yields [n_mol, F]; "all" yields [n_mol, num_layers*F]. With
    higher_order, compute per-atom norms before pooling to avoid vector cancellation. No
    trainable normalization is applied; downstream scaling must fit on train only."""
    was_training = encoder.training
    encoder.eval()
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        if layers == "all":
            if higher_order:
                full, _ = encoder.encode_nodes_all_layers(batch, full=True)  # [N, L, sphere, C]
                node_feats = _with_higher_order(full)                        # [N, L, (1+lmax)*C]
            else:
                node_feats, _ = encoder.encode_nodes_all_layers(batch)       # [N, L, C]
        elif layers == "last":
            if higher_order:
                full, _ = encoder.encode_nodes_full(batch)                   # [N, sphere, C]
                node_feats = _with_higher_order(full.unsqueeze(1))           # [N, 1, (1+lmax)*C]
            else:
                node_scalar, _ = encoder.encode_nodes(batch)                 # [N, C]
                node_feats = node_scalar.unsqueeze(1)                        # [N, 1, C]
        else:
            raise ValueError(f"Unknown layers={layers!r}; expected 'last'|'all'")
        pooled = pool_nodes(node_feats, batch["node_graph_index"], batch.get("num_graphs"), reduce=pool)
        rows.append(pooled.reshape(pooled.size(0), -1).cpu().numpy())   # [G, L*C]
    if was_training:
        encoder.train()
    return np.concatenate(rows, axis=0).astype(np.float32) if rows else np.zeros((0, 0), dtype=np.float32)


def maybe_merge_val_into_train(cfg, train_df, val_df, tag: str = ""):
    """Append validation rows when finetune.train_on_val is enabled.

    The caller must still pass the original validation dataset to train_multitask, which
    disables selection and early stopping and reports the final model. Validation scores
    are then in-sample; do not use this mode for HPO."""
    if not bool(cfg.finetune.get("train_on_val", False)):
        return train_df
    import pandas as pd
    merged = pd.concat([train_df, val_df], ignore_index=True)
    print(f"[train_on_val]{(' ' + tag) if tag else ''}: folded {len(val_df)} validation "
          f"molecules into train ({len(train_df)} -> {len(merged)}). Best-epoch "
          f"selection and early stopping are DISABLED; the final-epoch model is "
          f"the deliverable.", flush=True)
    return merged
