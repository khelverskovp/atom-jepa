"""Validated conversions for mapping-shaped Hydra settings."""

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def config_dict(value) -> dict[str, Any]:
    """Resolve a mapping config and normalize its keys for keyword/W&B use."""
    if value is None:
        return {}
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError("Expected a mapping configuration")
    return {str(key): item for key, item in value.items()}


def split_sizes(values) -> tuple[float, float, float]:
    """Require exactly three train/validation/test split fractions."""
    train, valid, test = (float(value) for value in values)
    return train, valid, test
