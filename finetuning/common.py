"""Helpers shared by the fine-tuning scripts."""

import torch

from atom_jepa.checkpoint import (  # noqa: F401  (re-exported for the fine-tuning scripts)
    HF_REPO, HF_REVISION, PRETRAINED_MODELS, _cfg_from_ckpt, load_pretrained_encoder,
    read_checkpoint,
)
from atom_jepa.data.collate import move_batch  # noqa: F401


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name

