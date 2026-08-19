import numpy as np
import torch

from ctdg_md.losses import conservation_penalties
from ctdg_md.metrics import compute_metrics, pairwise_ami_matrix


def test_conservation_penalties_for_pair_distance_energy():
    positions = torch.randn(2, 3, 5, 3, requires_grad=True)
    delta = positions[:, :, 0] - positions[:, :, 1]
    energy = delta.square().sum(dim=-1)
    mask = torch.ones((2, 5), dtype=torch.bool)
    force, torque = conservation_penalties(energy, positions, mask)
    assert force.item() < 1e-10
    assert torque.item() < 1e-9


def test_auc_accuracy_ami_and_matrix():
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.4, 0.7, 0.8, 0.9])
    embeddings = np.array([[0, 0], [0, 0.1], [0.2, 0], [3, 3], [3.1, 3], [3, 3.2]])
    result, assignment = compute_metrics(
        np.arange(6.0), np.arange(6.0) + 0.1, labels, probabilities, embeddings
    )
    assert result.auc == 1.0
    assert result.accuracy == 1.0
    assert result.ami > 0.99
    names, matrix = pairwise_ami_matrix({"a": assignment, "b": 1 - assignment})
    assert names == ["a", "b"]
    assert np.allclose(matrix, 1.0)
