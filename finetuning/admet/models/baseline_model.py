"""Feature-vector encoders compatible with the shared ADMET training loop.

encode_nodes returns one embedding per molecule. Feature collation assigns one node to
each graph, making graph pooling an identity operation."""

from typing import Callable, Tuple

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class FeatureVectorEncoder(nn.Module):
    """An MLP from fixed-length molecular features to F_in embeddings. num_layers includes
    the output Linear layer."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512,
                num_layers: int = 3, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}; available: {list(_ACTIVATIONS)}")
        act = _ACTIVATIONS[activation]

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if num_layers == 1:
            layers = [nn.Linear(in_dim, out_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            for _ in range(num_layers - 2):
                layers += [act(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)]
            layers += [act(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def encode_nodes(self, batch) -> Tuple[torch.Tensor, None]:
        return self.mlp(batch["features"]), None   # [G, out_dim]


def build_baseline_encoder(cfg, in_dim: int, device
                           ) -> Tuple[Callable[[], nn.Module], int, dict]:
    """Return (encoder_factory, F_in, encoder_config) for the configured MLP family, using
    the derived feature width."""
    family = str(cfg.baseline.get("family", "mlp")).lower()
    kind = str(cfg.baseline.get("kind", "morgan_mlp"))
    if family == "mlp":
        mlp_cfg = cfg.baseline.mlp
        hidden_dim = int(mlp_cfg.get("hidden_dim", 512))
        num_layers = int(mlp_cfg.get("num_layers", 3))
        dropout = float(mlp_cfg.get("dropout", 0.1))
        activation = str(mlp_cfg.get("activation", "relu"))
        F_in = hidden_dim

        def encoder_factory() -> nn.Module:
            return FeatureVectorEncoder(
                in_dim=in_dim, out_dim=F_in, hidden_dim=hidden_dim,
                num_layers=num_layers, dropout=dropout, activation=activation,
            ).to(device)

        encoder_config = {
            "kind": kind, "family": family, "in_dim": in_dim, "out_dim": F_in,
            "hidden_dim": hidden_dim, "num_layers": num_layers,
            "dropout": dropout, "activation": activation,
        }
        return encoder_factory, F_in, encoder_config

    raise ValueError(
        f"build_baseline_encoder got baseline.family={family!r} (kind={kind!r}); only the "
        f"'mlp' family builds a torch encoder. The 'lgbm' family is handled by finetuning/admet/training/gbm.py "
        f"and should never reach here."
    )
