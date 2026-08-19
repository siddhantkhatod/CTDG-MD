from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def slice_time(batch: dict[str, Any], start: int, stop: int) -> dict[str, Any]:
    """Slice a trajectory while retaining exact left-boundary context for TBPTT."""
    if not 0 <= start < stop <= batch["positions"].shape[1]:
        raise ValueError(f"Invalid temporal slice [{start}:{stop}]")
    temporal = {"positions", "energy", "energy_raw", "state", "state_value", "times"}
    result: dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value[:, start:stop] if key in temporal and isinstance(value, Tensor) else value
    result["time_origin"] = batch.get("time_origin", batch["times"][:, 0])
    result["frame_offset"] = int(batch.get("frame_offset", 0)) + start
    if start > 0:
        result["previous_time"] = batch["times"][:, start - 1]
        result["previous_positions"] = batch["positions"][:, start - 1]
    return result


def atomic_json_dump(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
