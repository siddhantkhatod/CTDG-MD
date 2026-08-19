import torch

from ctdg_md.virtual_nodes import adjacency_kmeans, augment_with_temporal_virtual_nodes


def disconnected_cliques():
    edges = []
    for group in ([0, 1, 2], [3, 4, 5]):
        for i in group:
            for j in group:
                if i != j:
                    edges.append((i, j))
    return torch.tensor(edges, dtype=torch.long).T


def test_adjacency_row_kmeans_finds_communities():
    labels = adjacency_kmeans(disconnected_cliques(), 6, 2, iterations=20, seed=7)
    assert len(torch.unique(labels[:3])) == 1
    assert len(torch.unique(labels[3:])) == 1
    assert labels[0] != labels[3]


def test_tvn_edges_and_centroids_are_equivariant():
    torch.manual_seed(4)
    hidden = torch.randn(6, 8)
    xyz = torch.randn(6, 3)
    edge = disconnected_cliques()
    augmented_h, augmented_x, augmented_e, labels = augment_with_temporal_virtual_nodes(
        hidden, xyz, edge, 2, seed=7
    )
    assert augmented_h.shape == (8, 8)
    assert augmented_x.shape == (8, 3)
    assert augmented_e.shape[1] == edge.shape[1] + 12 + 2
    translation = torch.tensor([3.0, -1.0, 2.0])
    _, translated_x, _, labels_2 = augment_with_temporal_virtual_nodes(
        hidden, xyz + translation, edge, 2, seed=7
    )
    assert torch.equal(labels, labels_2)
    assert torch.allclose(translated_x, augmented_x + translation, atol=1e-6)
