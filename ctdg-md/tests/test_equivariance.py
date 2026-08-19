import pytest
import torch

from ctdg_md.config import ModelConfig
from ctdg_md.model import build_model


def tiny_config():
    return ModelConfig(
        hidden_dim=24,
        message_dim=24,
        num_layers=2,
        radial_basis=8,
        cutoff_angstrom=4.5,
        max_neighbors=8,
        time_dim=8,
        patch_size=2,
        num_virtual_nodes=2,
        dropout=0.0,
        fourier_dropout=0.0,
    )


def tiny_batch():
    torch.manual_seed(3)
    base = torch.tensor(
        [[0.0, 0, 0], [1.2, 0, 0], [0.0, 1.2, 0], [2.2, 0, 0], [2.4, 1, 0], [3.0, 0, 1]],
        dtype=torch.float32,
    )
    positions = torch.stack([base + 0.01 * torch.randn_like(base) for _ in range(4)]).unsqueeze(0)
    return {
        "positions": positions,
        "atom_numbers": torch.tensor([[6, 6, 8, 6, 7, 8]]),
        "roles": torch.tensor([[0, 0, 0, 1, 1, 1]]),
        "residue_ids": torch.tensor([[1, 1, 2, 30, 30, 30]]),
        "node_mask": torch.ones((1, 6), dtype=torch.bool),
        "times": torch.tensor([[0.0, 0.4, 1.1, 2.0]]),
    }


def transformed(batch):
    angle = torch.tensor(0.7)
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle), 0.0], [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([4.0, -2.0, 1.5])
    result = dict(batch)
    result["positions"] = batch["positions"] @ rotation.T + translation
    return result, rotation, translation


@pytest.mark.parametrize(
    "variant", ["constant", "schnet", "painn", "egnn", "egnn_tvn", "ctdg", "ctdg_tvn"]
)
def test_model_energy_invariance_and_coordinate_equivariance(variant):
    torch.manual_seed(9)
    model = build_model(tiny_config(), variant).eval()
    batch = tiny_batch()
    changed, rotation, translation = transformed(batch)
    with torch.no_grad():
        original = model(batch)
        moved = model(changed)
    assert torch.allclose(original.energy, moved.energy, atol=2e-4, rtol=2e-4)
    expected = original.coordinates @ rotation.T + translation
    assert torch.allclose(expected, moved.coordinates, atol=3e-4, rtol=3e-4)
