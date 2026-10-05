"""Load pretrained Atom-JEPA encoders: released ones from the Hugging Face Hub, or
local checkpoints from a pretraining run."""

import json
import os
from typing import Tuple
from dataclasses import fields

import torch

from atom_jepa.models.jepa_equiformer import EquiformerV3Config, EquiformerV3Encoder


# Released pretrained encoders on the Hugging Face Hub: <name>/model.safetensors + config.json
HF_REPO = "atom-jepa/atom-jepa"
HF_REVISION = "main"
PRETRAINED_MODELS = ("molecules", "crystals")


def read_checkpoint(ckpt_path: str, device="cpu") -> dict:
    """Read a pretraining checkpoint as a dict with 'context_encoder' and 'eqv3_cfg'.

    ckpt_path is a local .pt checkpoint, a local folder in the released format
    (model.safetensors + config.json), or the name of a released encoder
    ('molecules' / 'crystals'), downloaded from the Hugging Face Hub on first use
    and cached locally."""
    if not os.path.exists(ckpt_path) and ckpt_path in PRETRAINED_MODELS:
        from huggingface_hub import hf_hub_download
        for f in ("config.json", "model.safetensors"):
            local = hf_hub_download(HF_REPO, f"{ckpt_path}/{f}", revision=HF_REVISION)
        ckpt_path = os.path.dirname(local)
    if os.path.isdir(ckpt_path):
        from safetensors.torch import load_file
        with open(os.path.join(ckpt_path, "config.json")) as f:
            config = json.load(f)
        return {"context_encoder": load_file(os.path.join(ckpt_path, "model.safetensors"),
                                             device=str(device)),
                "eqv3_cfg": config["eqv3_cfg"]}
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no checkpoint at {ckpt_path!r}; pass a local .pt / folder or "
                                f"one of the released encoders {PRETRAINED_MODELS}")
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def _cfg_from_ckpt(ckpt) -> EquiformerV3Config:
    """Recover an EquiformerV3Config from a pretraining checkpoint.

    Accepts the config stored as a dict (the usual asdict(cfg)) or as a live
    EquiformerV3Config. Unknown keys are filtered out so a superset dict still
    loads."""
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


def load_pretrained_encoder(ckpt_path: str, device, load_weights: bool = True
                            ) -> Tuple[EquiformerV3Encoder, EquiformerV3Config, dict]:
    """Rebuild the headless EquiformerV3 encoder from a pretraining checkpoint.

    The architecture (EquiformerV3Config) is ALWAYS taken from the checkpoint so
    the model is bit-for-bit identical regardless of init. When load_weights=True
    (default) the pretrained context-encoder weights are loaded; when False the
    encoder is left at fresh random init -- the 'train from scratch' baseline."""
    ckpt = read_checkpoint(ckpt_path, device)
    cfg = _cfg_from_ckpt(ckpt)
    encoder = EquiformerV3Encoder(cfg).to(device)
    if load_weights:
        sd = None
        for key in ("context_encoder", "encoder", "model"):
            if key in ckpt and ckpt[key] is not None:
                sd = ckpt[key]
                break
        if sd is None:
            raise KeyError(
                "checkpoint has no encoder weights; expected 'context_encoder'."
            )
        try:
            # JEPA context encoder is an EquiformerV3Encoder -> keys are 'body.*'
            encoder.load_state_dict(sd, strict=True)
        except RuntimeError:
            # fallback: checkpoint stored the bare HeadlessEquiformerV3 backbone
            encoder.body.load_state_dict(sd, strict=True)
    return encoder, cfg, ckpt
