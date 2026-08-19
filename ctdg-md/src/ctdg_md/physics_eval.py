from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


KCAL_MOL_ANGSTROM_AMU_TO_ANGSTROM_FS2 = 4.184e-4
COMMON_MASSES = {
    1: 1.008,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998,
    15: 30.974,
    16: 32.06,
    17: 35.45,
    35: 79.904,
    53: 126.904,
}


@dataclass(frozen=True)
class ForceMetrics:
    mae: float
    rmse: float
    cosine_similarity: float
    net_force_rmse: float
    samples: int


def force_metrics(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> ForceMetrics:
    """Supervised per-atom force metrics; raises rather than inventing missing labels."""
    reference = np.asarray(reference)
    prediction = np.asarray(prediction)
    mask = np.asarray(mask).astype(bool)
    if reference.shape != prediction.shape or reference.shape[-1] != 3:
        raise ValueError("Force arrays must match [..., atoms, 3]")
    if mask.shape != reference.shape[:-1]:
        raise ValueError("Force mask shape mismatch")
    if not np.isfinite(reference[mask]).all() or not np.isfinite(prediction[mask]).all():
        raise ValueError("Force arrays contain non-finite values")
    error = prediction[mask] - reference[mask]
    ref_flat = reference[mask]
    pred_flat = prediction[mask]
    denominator = np.linalg.norm(ref_flat, axis=-1) * np.linalg.norm(pred_flat, axis=-1)
    cosine = np.sum(ref_flat * pred_flat, axis=-1) / np.clip(denominator, 1e-12, None)
    net_force = (prediction * mask[..., None]).sum(axis=-2)
    return ForceMetrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        cosine_similarity=float(np.mean(cosine)),
        net_force_rmse=float(np.sqrt(np.mean(net_force**2))),
        samples=int(mask.sum()),
    )


def atomic_masses(atom_numbers: Tensor) -> Tensor:
    masses = []
    for number in atom_numbers.detach().cpu().tolist():
        if int(number) not in COMMON_MASSES:
            raise ValueError(
                f"No audited mass for atomic number {number}; provide an explicit mass tensor"
            )
        masses.append(COMMON_MASSES[int(number)])
    return atom_numbers.new_tensor(masses, dtype=torch.float32)


def _energy_force(
    model: nn.Module,
    batch: dict[str, Tensor],
    energy_mean: float,
    energy_std: float,
    memory: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    positions = batch["positions"].detach().requires_grad_(True)
    local = {**batch, "positions": positions}
    output = model(local, memory=memory)
    energy = output.energy * energy_std + energy_mean
    force = -torch.autograd.grad(energy.sum(), positions, create_graph=False)[0]
    return energy.detach(), force.detach(), None if output.memory is None else output.memory.detach()


def velocity_verlet_rollout(
    model: nn.Module,
    initial_batch: dict[str, Tensor],
    initial_velocity: Tensor,
    steps: int,
    dt_fs: float,
    energy_mean: float,
    energy_std: float,
    masses: Tensor | None = None,
    acceleration_factor: float = KCAL_MOL_ANGSTROM_AMU_TO_ANGSTROM_FS2,
) -> tuple[Tensor, Tensor]:
    """Energy-gradient velocity-Verlet rollout for fine-timestep compatible data.

    MISATO's 80 ps stored-frame interval is not a valid atomistic integration step;
    callers must provide a physically appropriate femtosecond `dt_fs` and velocities.
    """
    if steps < 1 or dt_fs <= 0:
        raise ValueError("steps and dt_fs must be positive")
    if initial_batch["positions"].shape[1] != 1:
        raise ValueError("Rollout requires exactly one initial frame")
    if initial_velocity.shape != initial_batch["positions"].shape:
        raise ValueError("initial_velocity must match [batch, 1, atoms, 3]")
    atom_numbers = initial_batch["atom_numbers"]
    if atom_numbers.shape[0] != 1:
        raise ValueError("Current rollout implementation accepts one graph")
    if not bool(initial_batch["node_mask"].all()):
        raise ValueError("Rollout requires an unpadded graph; remove masked atoms first")
    mass = atomic_masses(atom_numbers[0]) if masses is None else masses
    mass = mass.reshape(1, 1, -1, 1).to(initial_velocity)
    positions = initial_batch["positions"].detach()
    velocity = initial_velocity.detach()
    time = initial_batch["times"].detach()
    trajectories = [positions[:, 0]]
    energies = []
    memory: Tensor | None = None
    previous_positions: Tensor | None = None
    previous_time: Tensor | None = None

    for step in range(steps):
        batch = {**initial_batch, "positions": positions, "times": time}
        if memory is not None:
            assert previous_positions is not None and previous_time is not None
            batch["previous_positions"] = previous_positions[:, 0]
            batch["previous_time"] = previous_time[:, 0]
            batch["time_origin"] = initial_batch["times"][:, 0]
            batch["frame_offset"] = step
        energy, force, next_memory = _energy_force(
            model, batch, energy_mean, energy_std, memory
        )
        acceleration = force * acceleration_factor / mass
        half_velocity = velocity + 0.5 * dt_fs * acceleration
        next_positions = positions + dt_fs * half_velocity
        next_time = time + dt_fs / 1000.0  # model time convention is ps

        next_batch = {**initial_batch, "positions": next_positions, "times": next_time}
        if next_memory is not None:
            next_batch["previous_positions"] = positions[:, 0]
            next_batch["previous_time"] = time[:, 0]
            next_batch["time_origin"] = initial_batch["times"][:, 0]
            next_batch["frame_offset"] = step + 1
        _, next_force, _ = _energy_force(
            model, next_batch, energy_mean, energy_std, next_memory
        )
        next_acceleration = next_force * acceleration_factor / mass
        velocity = half_velocity + 0.5 * dt_fs * next_acceleration
        previous_positions, previous_time = positions, time
        positions, time, memory = next_positions.detach(), next_time.detach(), next_memory
        trajectories.append(positions[:, 0])
        energies.append(energy[:, 0])
    return torch.stack(trajectories, dim=1), torch.stack(energies, dim=1)
