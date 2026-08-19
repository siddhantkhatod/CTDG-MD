import torch

from ctdg_md.config import ModelConfig
from ctdg_md.model import build_model
from ctdg_md.physics_eval import velocity_verlet_rollout


def test_velocity_verlet_zero_force_constant_model():
    model = build_model(ModelConfig(hidden_dim=8, message_dim=8, time_dim=4), "constant")
    positions = torch.tensor([[[[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]]])
    batch = {
        "positions": positions,
        "atom_numbers": torch.tensor([[6, 7, 8]]),
        "roles": torch.tensor([[0, 0, 1]]),
        "residue_ids": torch.tensor([[1, 1, 2]]),
        "node_mask": torch.ones((1, 3), dtype=torch.bool),
        "times": torch.tensor([[0.0]]),
    }
    trajectory, energy = velocity_verlet_rollout(
        model,
        batch,
        torch.zeros_like(positions),
        steps=2,
        dt_fs=1.0,
        energy_mean=0.0,
        energy_std=1.0,
    )
    assert trajectory.shape == (1, 3, 3, 3)
    assert energy.shape == (1, 2)
    assert torch.allclose(trajectory[:, 0], trajectory[:, -1])
