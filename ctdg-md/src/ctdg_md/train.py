from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ProjectConfig
from .data import Normalization, build_loaders
from .losses import multitask_loss
from .metrics import MetricResult, compute_metrics
from .model import TrajectoryOutput, build_model
from .utils import move_batch, resolve_device, seed_everything, slice_time


@dataclass
class EvaluationOutput:
    metrics: MetricResult
    assignments: np.ndarray
    energy_true: np.ndarray
    energy_pred: np.ndarray
    state_true: np.ndarray
    state_probability: np.ndarray
    embeddings: np.ndarray
    groups: np.ndarray
    frame_index: np.ndarray
    mean_loss: float


@dataclass
class TrainingResult:
    variant: str
    seed: int
    best_epoch: int
    best_validation_loss: float
    checkpoint: str
    test_metrics: dict[str, float]
    seconds: float


def _segments(length: int, size: int) -> list[tuple[int, int]]:
    if size < 1:
        raise ValueError("segment_length must be positive")
    return [(start, min(start + size, length)) for start in range(0, length, size)]


def train_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    config: ProjectConfig,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running = 0.0
    batches = 0
    for batch_index, cpu_batch in enumerate(loader):
        batch = move_batch(cpu_batch, device)
        length = batch["positions"].shape[1]
        segments = _segments(length, config.training.segment_length)
        use_physics = (
            config.training.physics_every > 0
            and (batch_index + epoch * len(loader)) % config.training.physics_every == 0
            and (
                config.training.force_balance_weight > 0
                or config.training.torque_balance_weight > 0
            )
        )
        batch["positions"].requires_grad_(use_physics)
        optimizer.zero_grad(set_to_none=True)
        memory: torch.Tensor | None = None
        batch_loss = 0.0
        for start, stop in segments:
            segment = slice_time(batch, start, stop)
            output: TrajectoryOutput = model(segment, memory=memory)
            terms = multitask_loss(
                output.energy,
                output.state_logits,
                segment["energy"],
                segment["state"],
                segment["positions"],
                segment["node_mask"],
                segment["times"],
                config.training,
                compute_physics=use_physics,
            )
            if not torch.isfinite(terms.total):
                raise FloatingPointError(
                    f"Non-finite loss in epoch={epoch}, batch={batch_index}, segment={start}:{stop}"
                )
            scaled = terms.total / len(segments)
            scaled.backward()
            batch_loss += float(terms.total.detach())
            # Truncated BPTT is explicit: values cross boundaries, gradients do not.
            memory = None if output.memory is None else output.memory.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()
        running += batch_loss / len(segments)
        batches += 1
    return running / max(batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Any,
    normalization: Normalization,
    config: ProjectConfig,
    device: torch.device,
    seed: int,
) -> EvaluationOutput:
    model.eval()
    energy_true: list[np.ndarray] = []
    energy_pred: list[np.ndarray] = []
    state_true: list[np.ndarray] = []
    state_probability: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    groups: list[str] = []
    frame_indices: list[int] = []
    losses: list[float] = []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        output: TrajectoryOutput = model(batch)
        energy_component = torch.nn.functional.smooth_l1_loss(output.energy, batch["energy"])
        state_component = torch.nn.functional.binary_cross_entropy_with_logits(
            output.state_logits, batch["state"]
        )
        losses.append(float(energy_component + config.training.state_loss_weight * state_component))
        raw_prediction = output.energy * normalization.energy_std + normalization.energy_mean
        energy_pred.append(raw_prediction.cpu().numpy().reshape(-1))
        energy_true.append(batch["energy_raw"].cpu().numpy().reshape(-1))
        state_true.append(batch["state"].cpu().numpy().reshape(-1))
        state_probability.append(torch.sigmoid(output.state_logits).cpu().numpy().reshape(-1))
        embeddings.append(output.embeddings.cpu().numpy().reshape(-1, output.embeddings.shape[-1]))
        window_length = int(batch["positions"].shape[1])
        for pdb_id, start in zip(batch["pdb_id"], batch["start_frame"].cpu().tolist()):
            groups.extend([pdb_id] * window_length)
            frame_indices.extend(range(int(start), int(start) + window_length))
    arrays = {
        "energy_true": np.concatenate(energy_true),
        "energy_pred": np.concatenate(energy_pred),
        "state_true": np.concatenate(state_true).astype(np.int64),
        "state_probability": np.concatenate(state_probability),
        "embeddings": np.concatenate(embeddings),
    }
    group_array = np.asarray(groups, dtype="U16")
    frame_array = np.asarray(frame_indices, dtype=np.int64)
    metrics, assignments = compute_metrics(**arrays, seed=seed)
    return EvaluationOutput(
        metrics=metrics,
        assignments=assignments,
        groups=group_array,
        frame_index=frame_array,
        mean_loss=float(np.mean(losses)),
        **arrays,
    )


def _write_history(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one(
    config: ProjectConfig,
    variant: str | None = None,
    seed: int | None = None,
    loaders: dict[str, Any] | None = None,
    normalization: Normalization | None = None,
) -> tuple[TrainingResult, EvaluationOutput]:
    variant = (variant or config.model.name).lower()
    seed = config.seed if seed is None else seed
    if variant.startswith("ctdg") and config.training.segment_length < config.data.window_size:
        patch = config.model.patch_size
        if config.training.segment_length % patch or config.data.window_size % patch:
            raise ValueError(
                "For segmented CTDG training, segment_length and window_size must be "
                "multiples of patch_size so patch Fourier boundaries match evaluation"
            )
    seed_everything(seed)
    device = resolve_device(config.device)
    if loaders is None or normalization is None:
        loaders, normalization = build_loaders(config.data, config.training, seed=seed)
    model = build_model(config.model, variant).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    output_dir = Path(config.training.result_dir) / f"{variant}_seed{seed}"
    checkpoint_path = Path(config.training.checkpoint_dir) / f"{variant}_seed{seed}.pt"
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_epoch = -1
    stale = 0
    started = time.monotonic()
    for epoch in range(config.training.epochs):
        train_loss = train_epoch(model, loaders["train"], optimizer, config, device, epoch)
        validation = evaluate(model, loaders["val"], normalization, config, device, seed)
        if not math.isfinite(validation.mean_loss):
            raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": validation.mean_loss,
            **{f"val_{key}": value for key, value in validation.metrics.as_dict().items()},
        }
        history.append(row)
        print(
            f"[{variant} seed={seed}] epoch={epoch:03d} train={train_loss:.6f} "
            f"val={validation.mean_loss:.6f} AUC={validation.metrics.auc:.4f}"
        )
        if validation.mean_loss < best_loss - 1e-8:
            best_loss = validation.mean_loss
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config,
                normalization,
                variant,
                epoch,
                validation.mean_loss,
            )
        else:
            stale += 1
            if stale >= config.training.patience:
                break
    _write_history(history, output_dir / "history.csv")
    payload = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"], strict=True)
    test = evaluate(model, loaders["test"], normalization, config, device, seed)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        energy_true=test.energy_true,
        energy_pred=test.energy_pred,
        state_true=test.state_true,
        state_probability=test.state_probability,
        embeddings=test.embeddings,
        assignments=test.assignments,
        pdb_id=test.groups,
        frame_index=test.frame_index,
    )
    elapsed = time.monotonic() - started
    result = TrainingResult(
        variant=variant,
        seed=seed,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        checkpoint=str(checkpoint_path),
        test_metrics=test.metrics.as_dict(),
        seconds=elapsed,
    )
    (output_dir / "result.json").write_text(
        __import__("json").dumps(asdict(result), indent=2), encoding="utf-8"
    )
    return result, test
