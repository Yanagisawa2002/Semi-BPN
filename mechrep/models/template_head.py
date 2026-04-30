"""Template prediction head for mechanism-template supervision."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - import-time environment guard
    raise RuntimeError("Template heads require PyTorch") from exc


class TemplateHead(nn.Module):
    def __init__(self, input_dim: int, num_templates: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_templates <= 0:
            raise ValueError("num_templates must be positive")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        if hidden_dim is None or hidden_dim <= 0:
            self.network = nn.Linear(input_dim, num_templates)
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_templates),
            )

    def forward(self, pair_representation: torch.Tensor) -> torch.Tensor:
        if pair_representation.ndim != 2:
            raise ValueError(
                f"pair_representation must have shape batch_size x input_dim, got {tuple(pair_representation.shape)}"
            )
        return self.network(pair_representation)
