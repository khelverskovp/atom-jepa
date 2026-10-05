"""
Frozen-encoder READOUT-HEAD probe on QM9 ("mlp_all", generalized).

Same protocol as finetuning/qm9/finetune.py in every respect that affects the numbers --
same dataset, same split, same seeds, same target transform, same loss, same
optimizer/schedule/early-stopping, same metric -- with exactly one difference:

    finetuning/qm9/finetune.py : encoder weights TRAIN
    this script     : encoder is FROZEN, its per-atom features are cached once,
                      and only the finetuning.qm9.readouts head is trained

so the two runs are directly comparable and the delta isolates "does adapting
the representation help beyond a head on the frozen one".

To keep them from drifting, the split and the checkpoint loading are IMPORTED
from finetuning.qm9.finetune rather than reimplemented, both batch with the same
atom_jepa.data.collate.GraphCollator, and both scripts read
the same conf/finetune_qm9.yaml (this one additionally reads an optional
`probe_head:` block for cache/head settings).

FEATURES ("mlp_all"): the l=0 invariant slice is cached at EVERY depth --

    embed          x[:, 0, :] after atom-embedding + edge-degree-embedding
    block0..blockN x[:, 0, :] after each TransBlockV3 (residual stream, pre-norm)
    final          post-final_norm scalar == encoder.encode_nodes()

and the head sees the selected layers concatenated, so f_in = len(layers) * C.
Intermediate slices are UNNORMALIZED, and their scale grows with depth, so they
are standardized (per layer+channel, fit on train atoms only) before the head.

VECTORS (target `mu` only): the dipole head also needs the l=1 block. It is
cached from the post-final-norm irreps only -- i.e. exactly what
FineTuneModel.forward hands the head -- so f_vec = C while f_in = len(layers)*C.
The l=1 features are NOT standardized: subtracting a per-channel mean from a
vector is not equivariant, and they are post-norm anyway.

Usage (hydra, same config as the finetuner):
    python -m finetuning.probing.qm9_frozen_encoder_probe finetune.target=alpha
    python -m finetuning.probing.qm9_frozen_encoder_probe finetune.target=mu probe_head.layers=[final]
    python -m finetuning.probing.qm9_frozen_encoder_probe probe_head.cache_dtype=float16 data.limit=20000
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

import wandb
from atom_jepa.models.eqv3_backbone import edges_from_batch
from atom_jepa.data.collate import GraphCollator
from data.datasets.qm9 import QM9Dataset
from finetuning.qm9.readouts import build_readout
from finetuning.results_csv import append_row, encoder_tag

# Imported, not reimplemented: any change to the finetuner's split/loading is
# picked up here automatically.
from finetuning.qm9.finetune import QM9_UNITS, make_split
from finetuning.common import load_pretrained_encoder, move_batch, resolve_device


# --------------------------------------------------------------------------- #
# feature extraction
# --------------------------------------------------------------------------- #
def layer_names(encoder) -> List[str]:
    """Names of the cached layers, in cache order."""
    n_blocks = len(encoder.body.blocks)
    return ["embed"] + [f"block{i}" for i in range(n_blocks)] + ["final"]


@torch.no_grad()
def encode_all_layers(encoder, batch, want_vectors: bool
                      ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Per-atom l=0 features at every depth -> [N, L, C] (+ final l=1 [N, 3, C]).

    Re-runs HeadlessEquiformerV3.encode()'s factored pieces (embed_edges ->
    embed_nodes -> blocks -> final_norm) so the per-block residual stream can be
    tapped. run_blocks() only takes its checkpointed path when x.requires_grad,
    which is False under no_grad, so this unrolled loop is numerically identical
    to encoder.encode_nodes(batch).

    x is packed irreps [N, (lmax+1)**2, C]; x[:, 0, :] is the invariant l=0 part
    and normed_x[:, 1:4, :] the Cartesian l=1 block (e3nn convention)."""
    body = encoder.body
    atomic_numbers = batch["atomic_numbers"]
    node_graph_index = batch["node_graph_index"]
    edge_index_v3, edge_distance, edge_vec = edges_from_batch(batch)

    rbf, env = body.embed_edges(edge_distance, edge_vec)
    x = body.embed_nodes(atomic_numbers, rbf, edge_index_v3, env, add_atom_embedding=True)

    outs = [x[:, 0, :]]
    if edge_index_v3.shape[1] == 0:
        # single-atom / no-neighbour graphs: attention sublayer is identity
        for blk in body.blocks:
            x = body._block_ffn_only(blk, x, node_graph_index)
            outs.append(x[:, 0, :])
    else:
        src_z = atomic_numbers[edge_index_v3[0]]
        tgt_z = atomic_numbers[edge_index_v3[1]]
        for blk in body.blocks:
            x = blk(x, src_z, tgt_z, rbf, edge_index_v3, env, node_graph_index)
            outs.append(x[:, 0, :])

    normed_x, node_scalar = body.final_norm(x)
    outs.append(node_scalar)
    scalars = torch.stack(outs, dim=1)                      # [N, L, C]
    vectors = normed_x[:, 1:4, :] if want_vectors else None  # [N, 3, C]
    return scalars, vectors


@torch.no_grad()
def cache_features(encoder, dataset, indices, collate, device, batch_size,
                   num_workers, cache_dtype, want_vectors, check_parity=False) -> Dict:
    """Cache frozen per-atom features plus everything the heads read from `batch`.

    Positions and atomic numbers are kept because the mu / r2 heads consume them
    (center of mass, atomic masses, q*r); the atom reference is kept so the
    target transform matches the finetuner exactly."""
    encoder.eval()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )
    feats: List[torch.Tensor] = []
    vecs: List[torch.Tensor] = []
    pos: List[torch.Tensor] = []
    zs: List[torch.Tensor] = []
    ys, refs = [], []

    for bi, batch in enumerate(loader):
        batch = move_batch(batch, device)
        scalars, vectors = encode_all_layers(encoder, batch, want_vectors)

        if check_parity and bi == 0:
            ref, _ = encoder.encode_nodes(batch)
            diff = (scalars[:, -1, :] - ref).abs().max().item()
            print(f"[check] |cached final layer - encode_nodes|_max = {diff:.3e}",
                  flush=True)

        gidx = batch["node_graph_index"]
        ng = int(batch.get("num_graphs") or (gidx.max().item() + 1))
        counts = torch.bincount(gidx, minlength=ng).tolist()

        feats.extend(t.contiguous() for t in
                     torch.split(scalars.to("cpu", dtype=cache_dtype), counts))
        if want_vectors:
            vecs.extend(t.contiguous() for t in
                        torch.split(vectors.to("cpu", dtype=cache_dtype), counts))
        pos.extend(t.contiguous() for t in
                   torch.split(batch["node_coordinates"].cpu(), counts))
        zs.extend(t.contiguous() for t in
                  torch.split(batch["atomic_numbers"].cpu(), counts))
        ys.append(batch["y"][:, 0].cpu())
        refs.append(batch["atom_ref"].cpu())

    return {
        "feats": feats,
        "vecs": vecs if want_vectors else None,
        "pos": pos,
        "z": zs,
        "y": torch.cat(ys, 0),
        "atom_ref": torch.cat(refs, 0),
    }


def resolve_layers(spec: Sequence[str], names: List[str]) -> List[int]:
    """Turn a layer spec into cache column indices.

    'all'      -> every cached layer          (the mlp_all default)
    'blocks'   -> block0..blockN only
    'no-embed' -> everything except embed
    otherwise  -> explicit layer names, in the order given
    """
    spec = list(spec)
    if len(spec) == 1 and spec[0] in ("all", "blocks", "no-embed"):
        if spec[0] == "all":
            return list(range(len(names)))
        if spec[0] == "blocks":
            return [i for i, n in enumerate(names) if n.startswith("block")]
        return [i for i, n in enumerate(names) if n != "embed"]
    out = []
    for s in spec:
        if s not in names:
            raise ValueError(f"unknown layer {s!r}; cached layers are {names}")
        out.append(names.index(s))
    return out


def describe_layers(sel: Sequence[int], names: List[str], maxn: int = 3) -> str:
    picked = [names[i] for i in sel]
    if len(picked) <= maxn:
        return ",".join(picked)
    return f"{picked[0]},{picked[1]},..,{picked[-1]} ({len(picked)})"


# --------------------------------------------------------------------------- #
# standardization of the concatenated features (train atoms only)
# --------------------------------------------------------------------------- #
def feature_stats(feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-(layer, channel) mean/std over all TRAIN atoms -> two [L, C] tensors."""
    n, s, ss = 0, None, None
    for f in feats:
        x = f.double()
        n += x.size(0)
        s = x.sum(0) if s is None else s + x.sum(0)
        ss = (x * x).sum(0) if ss is None else ss + (x * x).sum(0)
    mean = s / max(1, n)
    std = (ss / max(1, n) - mean * mean).clamp_min(0.0).sqrt().clamp_min(1e-6)
    return mean.float(), std.float()


class Standardizer:
    """Applies [L, C] mean/std to layer-selected, flattened features."""

    def __init__(self, mean, std, sel, device):
        sel_t = torch.as_tensor(sel, dtype=torch.long)
        self.mean = mean.index_select(0, sel_t).reshape(-1).to(device)
        self.std = std.index_select(0, sel_t).reshape(-1).to(device)

    def __call__(self, x):
        return (x - self.mean) / self.std


# --------------------------------------------------------------------------- #
# batching from the cache
# --------------------------------------------------------------------------- #
def assemble(cache, idx, sel, device, standardizer, want_vectors):
    """Build (feats, node_vec, batch-dict, y, atom_ref) for a list of molecules."""
    chunks = [cache["feats"][k] for k in idx]
    sizes = torch.tensor([c.size(0) for c in chunks])

    feats = torch.cat(chunks, 0).index_select(1, sel)
    feats = feats.reshape(feats.size(0), -1).to(device, torch.float32, non_blocking=True)
    if standardizer is not None:
        feats = standardizer(feats)

    node_vec = None
    if want_vectors:
        node_vec = torch.cat([cache["vecs"][k] for k in idx], 0).to(
            device, torch.float32, non_blocking=True)          # [N, 3, C]

    gidx = torch.repeat_interleave(
        torch.arange(len(idx), device=device), sizes.to(device))
    batch = {
        "node_graph_index": gidx,
        "num_graphs": len(idx),
        "node_coordinates": torch.cat([cache["pos"][k] for k in idx], 0).to(device),
        "atomic_numbers": torch.cat([cache["z"][k] for k in idx], 0).to(device),
    }
    y = cache["y"][idx].to(device)
    atom_ref = cache["atom_ref"][idx].to(device)
    return feats, node_vec, batch, y, atom_ref


@torch.no_grad()
def evaluate(head, cache, sel, device, bs, standardizer, want_vectors,
             use_atom_ref, y_mean, y_std, loss_type) -> Tuple[float, float]:
    """(MAE in native units, mean training-space loss) -- same definitions as
    finetuning.qm9.finetune.evaluate, so the two scripts' numbers are directly comparable."""
    head.eval()
    total_abs = total_loss = 0.0
    M = len(cache["feats"])
    for i in range(0, M, bs):
        idx = list(range(i, min(i + bs, M)))
        feats, node_vec, batch, y, atom_ref = assemble(
            cache, idx, sel, device, standardizer, want_vectors)
        pred = head(feats, batch, node_vec=node_vec)

        y_ref = y - atom_ref if use_atom_ref else y
        yb = (y_ref - y_mean) / y_std
        red = "sum"
        total_loss += (F.mse_loss(pred, yb, reduction=red) if loss_type == "mse"
                       else F.l1_loss(pred, yb, reduction=red)).item()

        pred_native = pred * y_std + y_mean
        if use_atom_ref:
            pred_native = pred_native + atom_ref
        total_abs += (pred_native - y).abs().sum().item()
    return total_abs / max(1, M), total_loss / max(1, M)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def probe_head(cfg: DictConfig) -> Dict[str, float]:
    device = torch.device(resolve_device(cfg.misc.device))
    ftc = cfg.finetune
    phc = cfg.get("probe_head", OmegaConf.create({}))

    target = ftc.target
    unit = QM9_UNITS.get(target, "")
    pool = ftc.pool
    use_atom_ref = bool(ftc.use_atom_ref)
    standardize = bool(ftc.standardize)
    loss_type = str(ftc.get("loss", "mse")).lower()

    # ---- seeds & split: identical to the finetuner ----
    seed = int(ftc.get("seed", cfg.misc.seed))
    split_seed = int(ftc.get("split_seed", seed))
    torch.manual_seed(seed)

    def pick(key, default=None):
        """probe_head.<key> if set, else finetune.<key>, else default.

        An explicit `null` in the probe_head block counts as 'not set' -- that is
        how the config expresses "inherit the finetune value" (OmegaConf's .get
        returns None for a present-but-null key, so a plain .get(key, fallback)
        would wrongly yield None)."""
        v = phc.get(key, None)
        if v is not None:
            return v
        return ftc.get(key, default)

    # ---- probe-specific knobs (all optional; defaults mirror the finetuner) ----
    layers_spec = list(phc.get("layers", None) or ["all"])
    cache_dtype = (torch.float16 if str(phc.get("cache_dtype", None) or "float32") == "float16"
                   else torch.float32)
    cache_bs = int(phc.get("cache_batch_size", None) or 128)
    epochs = int(pick("epochs"))
    lr = float(pick("lr"))
    bs = int(pick("batch_size"))
    weight_decay = float(pick("weight_decay", 0.0) or 0.0)
    warmup_steps = int(pick("lr_warmup_steps", 0) or 0)
    adam_eps = float(pick("adam_eps", 1e-7))
    grad_clip = float(pick("grad_clip", 0.0))
    head_n_layers = int(pick("head_n_layers", 2))
    head_n_hidden = pick("head_n_hidden", None)
    do_standardize_feats = bool(phc.get("standardize_features", None) is not False)
    monitor = str(ftc.get("monitor", "loss")).lower()
    lr_factor = float(ftc.get("lr_factor", 0.8))
    lr_patience = int(ftc.get("lr_patience", 15))
    min_lr = float(ftc.get("min_lr", 1e-7))
    patience = int(ftc.get("patience", 150))
    random_encoder = bool(phc.get("random_encoder", False))

    use_wandb = bool(cfg.wandb.enabled) and cfg.wandb.mode != "disabled"
    if use_wandb:
        wandb.init(
            project=cfg.wandb.project, entity=cfg.wandb.entity,
            name=((f"{cfg.wandb.run_name}-probehead" if cfg.wandb.run_name
                   else f"probehead-{target}")
                  + ("-randinit" if random_encoder else "")),
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def wlog(d, step):
        if use_wandb:
            wandb.log(d, step=step)

    # ---- frozen encoder (loaded exactly as the finetuner loads it) ----
    ckpt_path = ftc.get("ckpt_path", None) or os.path.join(
        cfg.misc.checkpoint_dir, "context_encoder.pt")
    encoder, eqv3_cfg, ckpt = load_pretrained_encoder(
        ckpt_path, device, load_weights=not random_encoder)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    C = eqv3_cfg.num_channels
    cutoff = eqv3_cfg.max_radius
    enc_tag = encoder_tag(ckpt_path) + ("-randinit" if random_encoder else "")
    if random_encoder:
        print(f"[probe] RANDOM-INIT encoder: architecture from {ckpt_path}, "
              f"weights NOT loaded (seed={seed}); C={C} cutoff={cutoff}", flush=True)
    else:
        print(f"[probe] frozen encoder from {ckpt_path} (pretrain epoch "
              f"{ckpt.get('epoch', '?')}); C={C} cutoff={cutoff}", flush=True)

    # ---- data: the SAME dataset class and split the finetuner uses ----
    collate = GraphCollator(cutoff=cutoff)
    dataset = QM9Dataset(
        root=cfg.data.root, target=target,
        min_atoms=cfg.data.min_atoms, max_atoms=cfg.data.max_atoms,
        limit=cfg.data.limit,
    )
    train_idx, val_idx, test_idx = make_split(len(dataset), ftc, split_seed)
    print(f"[probe] target={target} [{unit}] split train={len(train_idx)} "
          f"val={len(val_idx)} test={len(test_idx)} (seed={seed} "
          f"split_seed={split_seed})", flush=True)

    names = layer_names(encoder)
    sel_list = resolve_layers(layers_spec, names)
    sel = torch.tensor(sel_list, dtype=torch.long)
    f_in = len(sel_list) * C

    head_proto = build_readout(target, f_in, pool=pool)   # just to read needs_vectors
    want_vectors = bool(head_proto.needs_vectors)

    itemsize = 2 if cache_dtype == torch.float16 else 4
    per_atom = len(names) * C + (3 * C if want_vectors else 0)
    est_gb = len(dataset) * 18.0 * per_atom * itemsize / 1e9
    print(f"[probe] caching layers={len(names)} ({names[0]}..{names[-1]}) C={C} "
          f"vectors={want_vectors} dtype={cache_dtype} -> est. ~{est_gb:.2f} GB "
          f"for the full dataset", flush=True)
    print(f"[probe] head input: {describe_layers(sel_list, names)} -> f_in={f_in}"
          + (f", f_vec={C}" if want_vectors else ""), flush=True)

    Xtr = cache_features(encoder, dataset, train_idx, collate, device, cache_bs,
                         cfg.data.num_workers, cache_dtype, want_vectors, check_parity=True)
    Xva = cache_features(encoder, dataset, val_idx, collate, device, cache_bs,
                         cfg.data.num_workers, cache_dtype, want_vectors)
    Xte = cache_features(encoder, dataset, test_idx, collate, device, cache_bs,
                         cfg.data.num_workers, cache_dtype, want_vectors)
    print("[probe] cache done", flush=True)

    # ---- feature standardization (TRAIN atoms only) ----
    standardizer = None
    if do_standardize_feats:
        m, s = feature_stats(Xtr["feats"])
        standardizer = Standardizer(m, s, sel_list, device)

    # ---- target transform: identical to the finetuner ----
    if standardize:
        y = Xtr["y"] - Xtr["atom_ref"] if use_atom_ref else Xtr["y"]
        y_mean, y_std = float(y.mean()), float(y.std().clamp_min(1e-8))
    else:
        y_mean, y_std = 0.0, 1.0

    head = build_readout(target, f_in, pool=pool, n_layers=head_n_layers,
                         n_hidden=head_n_hidden,
                         f_vec=(C if want_vectors else None)).to(device)
    head.set_target_stats(y_mean, y_std)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[probe] head: {type(head).__name__} ({n_params/1e6:.2f}M trainable params) "
          f"| loss={loss_type} monitor={monitor} standardize={standardize} "
          f"(mean={y_mean:.4f} std={y_std:.4f})", flush=True)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr,
                                  weight_decay=weight_decay, eps=adam_eps)
    for g in optimizer.param_groups:
        g["base_lr"] = g["lr"]

    def apply_warmup(step):
        if warmup_steps <= 0 or step >= warmup_steps:
            return False
        scale = min(1.0, float(step + 1) / float(warmup_steps))
        for g in optimizer.param_groups:
            g["lr"] = scale * g["base_lr"]
        return True

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr)

    best_monitor = float("inf")
    best_mae = float("inf")
    best_test = float("nan")
    best_epoch = -1
    epochs_no_improve = 0
    step = 0
    last_epoch = -1
    M = len(Xtr["feats"])

    for epoch in range(epochs):
        last_epoch = epoch
        head.train()
        perm = torch.randperm(M).tolist()
        running = 0.0
        nb = 0
        for i in range(0, M, bs):
            idx = perm[i:i + bs]
            feats, node_vec, batch, y, atom_ref = assemble(
                Xtr, idx, sel, device, standardizer, want_vectors)

            y_ref = y - atom_ref if use_atom_ref else y
            yb = (y_ref - y_mean) / y_std

            pred = head(feats, batch, node_vec=node_vec)
            loss = (F.mse_loss(pred, yb) if loss_type == "mse"
                    else F.l1_loss(pred, yb))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            apply_warmup(step)
            optimizer.step()

            running += loss.item()
            nb += 1
            step += 1

        val_mae, val_loss = evaluate(head, Xva, sel, device, bs, standardizer,
                                     want_vectors, use_atom_ref, y_mean, y_std, loss_type)
        monitored = val_loss if monitor == "loss" else val_mae
        scheduler.step(monitored)

        if val_mae < best_mae:
            best_mae = val_mae
            best_test, _ = evaluate(head, Xte, sel, device, bs, standardizer,
                                    want_vectors, use_atom_ref, y_mean, y_std, loss_type)
            best_epoch = epoch
            if bool(phc.get("save_best", False)):
                os.makedirs(cfg.misc.checkpoint_dir, exist_ok=True)
                torch.save(
                    {"head": head.state_dict(), "target": target, "unit": unit,
                     "layers": [names[i] for i in sel_list], "f_in": f_in,
                     "f_vec": (C if want_vectors else None), "pool": pool,
                     "use_atom_ref": use_atom_ref, "standardize": standardize,
                     "y_mean": y_mean, "y_std": y_std, "val_mae": best_mae,
                     "test_mae": best_test, "epoch": best_epoch,
                     "random_encoder": random_encoder},
                    os.path.join(cfg.misc.checkpoint_dir,
                                 f"probehead_{target}"
                                 + ("_randinit" if random_encoder else "") + ".pt"))

        if monitored < best_monitor:
            best_monitor = monitored
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"== epoch {epoch} mean_loss {running/max(1,nb):.4f} "
              f"val_loss {val_loss:.4f} val_mae {val_mae:.4f} {unit} "
              f"(best val_mae {best_mae:.4f} @ epoch {best_epoch}, "
              f"test {best_test:.4f} {unit}) lr {lr_now:.2e}", flush=True)
        wlog({"probe_head/epoch_loss": running / max(1, nb),
              f"probe_head/{target}_val_loss": val_loss,
              f"probe_head/{target}_val_mae": val_mae,
              f"probe_head/{target}_best_val_mae": best_mae,
              f"probe_head/{target}_best_test_mae": best_test,
              "probe_head/lr": lr_now, "epoch": epoch}, step)

        if epochs_no_improve >= patience:
            print(f"[probe] early stopping at epoch {epoch} (no val-{monitor} "
                  f"improvement for {patience} epochs)", flush=True)
            break

    results = {
        f"probe_head/{target}_val_mae": best_mae,
        f"probe_head/{target}_test_mae": best_test,
        f"probe_head/{target}_best_epoch": float(best_epoch),
    }
    print(f"[probe] done ({unit}): "
          + " ".join(f"{k}={v:.4f}" for k, v in results.items()), flush=True)
    wlog(results, step)

    csv_path = append_row(cfg.misc.get("results_csv", None), {
        "script": "probe_head",
        "encoder": enc_tag,
        "random_encoder": random_encoder,
        "target": target, "unit": unit,
        "val_mae": best_mae, "test_mae": best_test, "best_epoch": best_epoch,
        "epochs_run": last_epoch + 1, "n_head_params": n_params,
        "layers": "+".join(names[i] for i in sel_list),
        "f_in": f_in, "f_vec": (C if want_vectors else ""),
        "pool": pool, "loss": loss_type, "monitor": monitor,
        "standardize": standardize, "use_atom_ref": use_atom_ref,
        "seed": seed, "split_seed": split_seed,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "run_name": (cfg.wandb.run_name or ""), "ckpt_path": str(ckpt_path),
    })
    if csv_path:
        print(f"[probe] results row appended to {csv_path}", flush=True)
    if use_wandb:
        wandb.finish()
    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="finetune_qm9")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    probe_head(cfg)


if __name__ == "__main__":
    main()