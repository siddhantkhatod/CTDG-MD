from __future__ import annotations

import torch
from torch import Tensor


def radius_graph(
    positions: Tensor,
    cutoff: float,
    max_neighbors: int,
    min_neighbors: int = 1,
) -> tuple[Tensor, Tensor]:
    """Build a directed distance graph for one unpadded molecular frame.

    Edges are ``source -> destination``. Neighbors within ``cutoff`` are retained,
    capped by nearest distance. If a node is isolated, its nearest non-self atoms are
    added up to ``min_neighbors``. The operation is deterministic and topology is not
    differentiated, as is standard for radius-graph construction.
    """
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [N, 3]")
    n = positions.shape[0]
    if n < 2:
        empty_edges = torch.empty((2, 0), dtype=torch.long, device=positions.device)
        return empty_edges, positions.new_empty((0,))
    if cutoff <= 0 or max_neighbors < 1 or min_neighbors < 0:
        raise ValueError("cutoff/max_neighbors/min_neighbors are invalid")

    distances = torch.cdist(positions, positions)
    inf = torch.tensor(float("inf"), device=positions.device, dtype=distances.dtype)
    distances = distances.masked_fill(torch.eye(n, dtype=torch.bool, device=positions.device), inf)
    sources: list[Tensor] = []
    destinations: list[Tensor] = []
    edge_distances: list[Tensor] = []
    for destination in range(n):
        row = distances[destination]
        candidates = torch.where(row < cutoff)[0]
        need = min(min_neighbors, n - 1)
        if candidates.numel() < need:
            candidates = torch.topk(row, k=need, largest=False).indices
        if candidates.numel() > max_neighbors:
            local = torch.topk(row[candidates], k=max_neighbors, largest=False).indices
            candidates = candidates[local]
        order = torch.argsort(row[candidates], stable=True)
        candidates = candidates[order]
        sources.append(candidates)
        destinations.append(torch.full_like(candidates, destination))
        edge_distances.append(row[candidates])
    edge_index = torch.stack([torch.cat(sources), torch.cat(destinations)], dim=0)
    return edge_index, torch.cat(edge_distances)


class RadialBasis(torch.nn.Module):
    """Gaussian distance expansion with a smooth cosine cutoff."""

    def __init__(self, count: int, cutoff: float) -> None:
        super().__init__()
        if count < 2 or cutoff <= 0:
            raise ValueError("RadialBasis requires count >= 2 and cutoff > 0")
        self.count = count
        self.cutoff = float(cutoff)
        centers = torch.linspace(0.0, self.cutoff, count)
        self.register_buffer("centers", centers)
        spacing = centers[1] - centers[0]
        self.gamma = float(1.0 / (spacing * spacing))

    def forward(self, distances: Tensor) -> Tensor:
        d = distances.unsqueeze(-1)
        rbf = torch.exp(-self.gamma * (d - self.centers) ** 2)
        x = torch.clamp(distances / self.cutoff, 0.0, 1.0)
        envelope = 0.5 * (torch.cos(torch.pi * x) + 1.0)
        envelope = torch.where(distances < self.cutoff, envelope, torch.zeros_like(envelope))
        return rbf * envelope.unsqueeze(-1)


def edge_categories(roles: Tensor, edge_index: Tensor, virtual_start: int | None = None) -> Tensor:
    """Return one-hot PP/PL/LP/LL/virtual categories for directed edges."""
    source, destination = edge_index
    result = torch.zeros((source.numel(), 5), device=roles.device, dtype=torch.float32)
    if source.numel() == 0:
        return result
    is_virtual = torch.zeros_like(source, dtype=torch.bool)
    if virtual_start is not None:
        is_virtual = (source >= virtual_start) | (destination >= virtual_start)
    physical = ~is_virtual
    if physical.any():
        category = roles[source[physical]].long() * 2 + roles[destination[physical]].long()
        result[physical, category] = 1.0
    result[is_virtual, 4] = 1.0
    return result


def scatter_mean(values: Tensor, index: Tensor, size: int) -> Tensor:
    out = values.new_zeros((size, values.shape[-1]))
    count = values.new_zeros((size, 1))
    if index.numel():
        out.index_add_(0, index, values)
        count.index_add_(0, index, values.new_ones((index.numel(), 1)))
    return out / count.clamp_min(1.0)
