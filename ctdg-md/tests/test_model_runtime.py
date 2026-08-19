import pytest
import torch

from ctdg_md.config import ModelConfig, TrainingConfig
from ctdg_md.losses import multitask_loss
from ctdg_md.model import build_model


def batch():
    torch.manual_seed(2)
    return {
        "positions": torch.randn(1, 4, 7, 3),
        "atom_numbers": torch.tensor([[6, 6, 7, 8, 6, 7, 8]]),
        "roles": torch.tensor([[0, 0, 0, 0, 1, 1, 1]]),
        "residue_ids": torch.tensor([[1, 1, 2, 2, 10, 10, 10]]),
        "node_mask": torch.ones((1, 7), dtype=torch.bool),
        "times": torch.tensor([[0.0, 1.0, 2.5, 5.0]]),
        "energy": torch.randn(1, 4),
        "state": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
    }


def config():
    return ModelConfig(
        hidden_dim=16,
        message_dim=16,
        num_layers=1,
        radial_basis=6,
        max_neighbors=6,
        time_dim=8,
        patch_size=2,
        num_virtual_nodes=2,
        dropout=0,
        fourier_dropout=0,
    )


@pytest.mark.parametrize(
    "variant", ["constant", "schnet", "painn", "egnn", "egnn_tvn", "ctdg", "ctdg_tvn"]
)
def test_forward_backward_all_variants(variant):
    data = batch()
    model = build_model(config(), variant)
    output = model(data)
    terms = multitask_loss(
        output.energy,
        output.state_logits,
        data["energy"],
        data["state"],
        data["positions"],
        data["node_mask"],
        data["times"],
        TrainingConfig(force_balance_weight=0, torque_balance_weight=0),
        compute_physics=False,
    )
    terms.total.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    assert output.embeddings.shape == (1, 4, 48)
    if variant.startswith("ctdg"):
        assert output.memory is not None
