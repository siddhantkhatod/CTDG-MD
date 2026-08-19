import pytest
import torch

from ctdg_md.config import ModelConfig
from ctdg_md.model import build_model
from ctdg_md.utils import slice_time


def batch():
    torch.manual_seed(21)
    base = torch.tensor(
        [[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [2.0, 0, 0], [2.0, 1.0, 0]],
        dtype=torch.float32,
    )
    xyz = torch.stack([base + 0.02 * torch.randn_like(base) for _ in range(4)]).unsqueeze(0)
    return {
        "positions": xyz,
        "atom_numbers": torch.tensor([[6, 7, 8, 6, 8]]),
        "roles": torch.tensor([[0, 0, 0, 1, 1]]),
        "residue_ids": torch.tensor([[1, 1, 2, 9, 9]]),
        "node_mask": torch.ones((1, 5), dtype=torch.bool),
        "times": torch.tensor([[0.0, 0.3, 1.7, 4.2]]),
    }


def config():
    return ModelConfig(
        hidden_dim=16,
        message_dim=16,
        num_layers=1,
        radial_basis=6,
        max_neighbors=5,
        time_dim=8,
        patch_size=2,
        num_virtual_nodes=2,
        dropout=0,
        fourier_dropout=0,
    )


@pytest.mark.parametrize("variant", ["ctdg", "ctdg_tvn"])
def test_segmented_forward_matches_full_forward_values(variant):
    torch.manual_seed(8)
    model = build_model(config(), variant).eval()
    data = batch()
    with torch.no_grad():
        full = model(data)
        first = model(slice_time(data, 0, 2))
        second = model(slice_time(data, 2, 4), memory=first.memory)
    energy = torch.cat([first.energy, second.energy], dim=1)
    states = torch.cat([first.state_logits, second.state_logits], dim=1)
    coordinates = torch.cat([first.coordinates, second.coordinates], dim=1)
    assert torch.allclose(energy, full.energy, atol=2e-6, rtol=2e-6)
    assert torch.allclose(states, full.state_logits, atol=2e-6, rtol=2e-6)
    assert torch.allclose(coordinates, full.coordinates, atol=2e-6, rtol=2e-6)
    assert torch.allclose(second.memory, full.memory, atol=2e-6, rtol=2e-6)
