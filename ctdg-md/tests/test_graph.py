import torch

from ctdg_md.graph import RadialBasis, radius_graph


def test_radius_graph_is_directed_bounded_and_deterministic():
    xyz = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.2, 0, 0], [9.0, 0, 0]])
    edge, distance = radius_graph(xyz, cutoff=1.5, max_neighbors=2, min_neighbors=1)
    assert edge.shape[0] == 2
    assert edge.shape[1] == distance.numel()
    assert not torch.any(edge[0] == edge[1])
    # Even the isolated atom receives one nearest neighbor.
    assert set(edge[1].tolist()) == {0, 1, 2, 3}
    edge2, distance2 = radius_graph(xyz, cutoff=1.5, max_neighbors=2, min_neighbors=1)
    assert torch.equal(edge, edge2)
    assert torch.equal(distance, distance2)


def test_radial_basis_cutoff():
    rbf = RadialBasis(8, 5.0)
    value = rbf(torch.tensor([0.0, 2.0, 5.0, 7.0]))
    assert value.shape == (4, 8)
    assert torch.all(value[2:] == 0)
