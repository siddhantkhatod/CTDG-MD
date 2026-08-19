from __future__ import annotations

import torch
from torch import Tensor


def adjacency_matrix(edge_index: Tensor, num_nodes: int) -> Tensor:
    matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device, dtype=torch.float32)
    if edge_index.numel():
        source, destination = edge_index
        matrix[destination, source] = 1.0
    # CTDG-MD contact graphs are conceptually undirected; guard asymmetric inputs.
    return torch.maximum(matrix, matrix.transpose(0, 1))


def _kmeans_plus_plus(points: Tensor, k: int, seed: int) -> Tensor:
    n = points.shape[0]
    generator = torch.Generator(device=points.device).manual_seed(seed)
    first = torch.randint(0, n, (1,), generator=generator, device=points.device)
    centers = [points[first.item()].clone()]
    closest = torch.cdist(points, centers[0].unsqueeze(0)).square().squeeze(1)
    for _ in range(1, k):
        total = closest.sum()
        if total <= 1e-12:
            # All rows are identical. A deterministic fallback avoids multinomial failure.
            candidate = len(centers) % n
        else:
            candidate = int(torch.multinomial(closest / total, 1, generator=generator).item())
        centers.append(points[candidate].clone())
        next_distance = torch.cdist(points, centers[-1].unsqueeze(0)).square().squeeze(1)
        closest = torch.minimum(closest, next_distance)
    return torch.stack(centers)


def adjacency_kmeans(
    edge_index: Tensor,
    num_nodes: int,
    k: int,
    iterations: int = 10,
    tolerance: float = 1e-2,
    seed: int = 0,
) -> Tensor:
    """Algorithm 2 from *Virtual Nodes Go Temporal* on adjacency rows.

    The exact dense adjacency rows are used (rather than a feature projection). Empty
    clusters are repaired by moving the currently worst represented point.
    """
    if not 1 <= k <= num_nodes:
        raise ValueError(f"k must be in [1, {num_nodes}], received {k}")
    points = adjacency_matrix(edge_index, num_nodes)
    centers = _kmeans_plus_plus(points, k, seed)
    labels = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
    for _ in range(iterations):
        distances = torch.cdist(points, centers)
        new_labels = distances.argmin(dim=1)
        new_centers = centers.clone()
        for cluster in range(k):
            members = new_labels == cluster
            if members.any():
                new_centers[cluster] = points[members].mean(dim=0)
            else:
                residual = distances[torch.arange(num_nodes, device=points.device), new_labels]
                replacement = residual.argmax()
                new_labels[replacement] = cluster
                new_centers[cluster] = points[replacement]
        shift = torch.linalg.vector_norm(new_centers - centers, dim=1).max()
        unchanged = torch.equal(new_labels, labels)
        labels, centers = new_labels, new_centers
        if unchanged or shift <= tolerance:
            break
    return labels


def augment_with_temporal_virtual_nodes(
    hidden: Tensor,
    positions: Tensor,
    edge_index: Tensor,
    k: int,
    iterations: int = 10,
    tolerance: float = 1e-2,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Cluster, initialize and connect k temporal virtual nodes.

    Child-to-VN aggregation is degree weighted, VNs form a directed clique, and each
    child has bidirectional edges to its assigned VN. VN coordinates are weighted
    centroids, preserving E(3) equivariance. VNs are reconstructed at every frame.
    """
    n = hidden.shape[0]
    labels = adjacency_kmeans(edge_index, n, k, iterations, tolerance, seed)
    _source, destination = edge_index
    degree = hidden.new_zeros(n)
    if destination.numel():
        degree.index_add_(0, destination, hidden.new_ones(destination.numel()))
    weights = degree.clamp_min(1.0)

    vn_hidden = hidden.new_zeros((k, hidden.shape[-1]))
    vn_positions = positions.new_zeros((k, 3))
    denominators = hidden.new_zeros((k, 1))
    vn_hidden.index_add_(0, labels, hidden * weights.unsqueeze(-1))
    vn_positions.index_add_(0, labels, positions * weights.unsqueeze(-1))
    denominators.index_add_(0, labels, weights.unsqueeze(-1))
    vn_hidden = vn_hidden / denominators.clamp_min(1.0)
    vn_positions = vn_positions / denominators.clamp_min(1.0)

    nodes = torch.arange(n, device=hidden.device)
    assigned_vn = n + labels
    child_to_vn = torch.stack([nodes, assigned_vn])
    vn_to_child = torch.stack([assigned_vn, nodes])
    vn_ids = torch.arange(n, n + k, device=hidden.device)
    left, right = torch.meshgrid(vn_ids, vn_ids, indexing="ij")
    keep = left != right
    vn_clique = torch.stack([left[keep], right[keep]])
    augmented_edges = torch.cat([edge_index, child_to_vn, vn_to_child, vn_clique], dim=1)
    return (
        torch.cat([hidden, vn_hidden], dim=0),
        torch.cat([positions, vn_positions], dim=0),
        augmented_edges.long(),
        labels,
    )
