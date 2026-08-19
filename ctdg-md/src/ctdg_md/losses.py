from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import TrainingConfig


@dataclass
class LossTerms:
    total: Tensor
    energy: Tensor
    state: Tensor
    force_balance: Tensor
    torque_balance: Tensor
    temporal_smoothness: Tensor


def conservation_penalties(energy: Tensor, positions: Tensor, node_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Noether-style force and torque residuals from -dE/dx.

    Translation-invariant scalar energies have zero net force and rotation-invariant
    scalar energies have zero net torque. ``create_graph=True`` makes these residuals
    trainable. This is appropriate for MISATO's interaction energy; an NPT trajectory's
    total energy is not incorrectly forced to be constant.
    """
    gradients = torch.autograd.grad(
        energy.sum(),
        positions,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]
    forces = -gradients * node_mask[:, None, :, None]
    net_force = forces.sum(dim=2)
    count = node_mask.sum(dim=1).clamp_min(1).to(positions.dtype)
    center = (positions * node_mask[:, None, :, None]).sum(dim=2) / count[:, None, None]
    centered = positions - center[:, :, None, :]
    torque = torch.cross(centered, forces, dim=-1).sum(dim=2)
    return net_force.square().mean(), torque.square().mean()


def temporal_smoothness(energy: Tensor, times: Tensor) -> Tensor:
    if energy.shape[1] < 3:
        return energy.new_zeros(())
    dt_left = (times[:, 1:-1] - times[:, :-2]).clamp_min(1e-8)
    dt_right = (times[:, 2:] - times[:, 1:-1]).clamp_min(1e-8)
    left_velocity = (energy[:, 1:-1] - energy[:, :-2]) / dt_left
    right_velocity = (energy[:, 2:] - energy[:, 1:-1]) / dt_right
    return (right_velocity - left_velocity).square().mean()


def multitask_loss(
    predicted_energy: Tensor,
    state_logits: Tensor,
    target_energy: Tensor,
    target_state: Tensor,
    positions: Tensor,
    node_mask: Tensor,
    times: Tensor,
    config: TrainingConfig,
    compute_physics: bool,
) -> LossTerms:
    energy_loss = torch.nn.functional.smooth_l1_loss(predicted_energy, target_energy)
    state_loss = torch.nn.functional.binary_cross_entropy_with_logits(state_logits, target_state)
    smoothness = temporal_smoothness(predicted_energy, times)
    if compute_physics and (config.force_balance_weight > 0 or config.torque_balance_weight > 0):
        force_balance, torque_balance = conservation_penalties(
            predicted_energy, positions, node_mask
        )
    else:
        force_balance = predicted_energy.new_zeros(())
        torque_balance = predicted_energy.new_zeros(())
    total = (
        config.energy_loss_weight * energy_loss
        + config.state_loss_weight * state_loss
        + config.force_balance_weight * force_balance
        + config.torque_balance_weight * torque_balance
        + config.temporal_smoothness_weight * smoothness
    )
    return LossTerms(total, energy_loss, state_loss, force_balance, torque_balance, smoothness)
