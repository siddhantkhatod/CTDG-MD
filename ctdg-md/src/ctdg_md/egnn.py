from __future__ import annotations

import torch
from torch import Tensor, nn

from .graph import scatter_mean


class MLP(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )


class EGNNLayer(nn.Module):
    """E(n)-equivariant graph convolution with invariant scalar messages."""

    def __init__(
        self,
        hidden_dim: int,
        message_dim: int,
        edge_dim: int,
        time_dim: int,
        coordinate_scale: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        message_input = 2 * hidden_dim + edge_dim + time_dim + 1
        self.edge_mlp = MLP(message_input, message_dim, message_dim, dropout)
        self.edge_gate = nn.Sequential(nn.Linear(message_dim, 1), nn.Sigmoid())
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(message_dim, message_dim), nn.SiLU(), nn.Linear(message_dim, 1), nn.Tanh()
        )
        self.node_mlp = MLP(hidden_dim + message_dim, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.coordinate_scale = float(coordinate_scale)

    def forward(
        self,
        hidden: Tensor,
        positions: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        time_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source, destination = edge_index
        if source.numel() == 0:
            return hidden, positions
        relative = positions[destination] - positions[source]
        squared_distance = (relative * relative).sum(dim=-1, keepdim=True)
        if time_features.ndim == 1:
            time_features = time_features.unsqueeze(0).expand(source.numel(), -1)
        message_input = torch.cat(
            [hidden[source], hidden[destination], squared_distance, edge_features, time_features],
            dim=-1,
        )
        raw_messages = self.edge_mlp(message_input)
        messages = raw_messages * self.edge_gate(raw_messages)
        aggregated = scatter_mean(messages, destination, hidden.shape[0])
        next_hidden = self.norm(hidden + self.node_mlp(torch.cat([hidden, aggregated], dim=-1)))

        # A bounded scalar times a relative vector is E(n)-equivariant. Normalizing by
        # (1 + distance) prevents unstable updates without changing equivariance.
        coefficient = self.coordinate_mlp(messages) * self.coordinate_scale
        distance = torch.sqrt(squared_distance + 1e-8)
        displacement = relative / (1.0 + distance) * coefficient
        coordinate_update = scatter_mean(displacement, destination, hidden.shape[0])
        return next_hidden, positions + coordinate_update
