from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import ProjectConfig
from .data import Normalization


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: ProjectConfig,
    normalization: Normalization,
    variant: str,
    epoch: int,
    validation_loss: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "config": config.to_dict(),
        "normalization": {
            "energy_mean": normalization.energy_mean,
            "energy_std": normalization.energy_std,
            "state_threshold": normalization.state_threshold,
            "energy_unit": normalization.energy_unit,
        },
        "variant": variant,
        "epoch": epoch,
        "validation_loss": validation_loss,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1 or "model_state" not in payload:
        raise ValueError(f"Unsupported or malformed checkpoint: {path}")
    return payload
