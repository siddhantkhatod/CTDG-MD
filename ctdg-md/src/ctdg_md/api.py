from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

from .checkpoint import load_checkpoint
from .config import load_config, project_config_from_dict
from .model import build_model
from .utils import resolve_device


class PredictionRequest(BaseModel):
    positions: list[list[list[float]]] = Field(description="Coordinates [frames][atoms][3] in angstrom")
    atom_numbers: list[int]
    roles: list[int] = Field(description="0=protein, 1=ligand")
    residue_ids: list[int] | None = None
    times_ps: list[float] | None = None
    return_forces: bool = True

    @model_validator(mode="after")
    def validate_shapes(self) -> PredictionRequest:
        if not self.positions:
            raise ValueError("at least one frame is required")
        nodes = len(self.atom_numbers)
        if nodes < 2 or len(self.roles) != nodes:
            raise ValueError("atom_numbers and roles must have the same length >= 2")
        if any(number < 1 or number > 118 for number in self.atom_numbers):
            raise ValueError("atom_numbers must be valid atomic numbers in [1, 118]")
        if set(self.roles) - {0, 1} or 0 not in self.roles or 1 not in self.roles:
            raise ValueError("roles must contain both protein (0) and ligand (1)")
        if any(len(frame) != nodes or any(len(xyz) != 3 for xyz in frame) for frame in self.positions):
            raise ValueError("positions must have shape [frames][atoms][3]")
        if self.residue_ids is not None and len(self.residue_ids) != nodes:
            raise ValueError("residue_ids length differs from atom count")
        if self.times_ps is not None:
            if len(self.times_ps) != len(self.positions):
                raise ValueError("times_ps length differs from frame count")
            if not all(float("-inf") < value < float("inf") for value in self.times_ps):
                raise ValueError("times_ps must be finite")
            if any(b <= a for a, b in zip(self.times_ps, self.times_ps[1:])):
                raise ValueError("times_ps must be strictly increasing")
        return self


class Runtime:
    def __init__(self) -> None:
        checkpoint = os.getenv("CTDG_CHECKPOINT", "")
        self.unsafe_untrained = False
        if checkpoint:
            payload = load_checkpoint(checkpoint, map_location="cpu")
            self.config = project_config_from_dict(payload["config"])
            self.variant = str(payload["variant"])
            self.normalization = payload["normalization"]
            self.model = build_model(self.config.model, self.variant)
            self.model.load_state_dict(payload["model_state"], strict=True)
            self.checkpoint = str(Path(checkpoint).resolve())
        elif os.getenv("CTDG_ALLOW_UNTRAINED", "0") == "1":
            config_path = os.getenv("CTDG_CONFIG", "configs/default.yaml")
            self.config = load_config(config_path)
            self.variant = self.config.model.name
            self.model = build_model(self.config.model, self.variant)
            self.normalization = {"energy_mean": 0.0, "energy_std": 1.0, "energy_unit": "untrained"}
            self.checkpoint = None
            self.unsafe_untrained = True
        else:
            self.config = None
            self.variant = None
            self.model = None
            self.normalization = None
            self.checkpoint = None
        requested = os.getenv("CTDG_DEVICE", "auto")
        self.device = resolve_device(requested)
        if self.model is not None:
            self.model.to(self.device).eval()

    @property
    def ready(self) -> bool:
        return self.model is not None and (self.checkpoint is not None or self.unsafe_untrained)

    def predict(self, request: PredictionRequest) -> dict[str, Any]:
        if not self.ready or self.model is None or self.normalization is None or self.config is None:
            raise RuntimeError("No checkpoint loaded; set CTDG_CHECKPOINT to a trained .pt file")
        raw_positions = torch.tensor(request.positions, dtype=torch.float32, device=self.device)
        frames, nodes, _ = raw_positions.shape
        if nodes > self.config.data.max_nodes:
            raise ValueError(
                f"Request has {nodes} atoms but this checkpoint allows at most "
                f"{self.config.data.max_nodes}"
            )
        if not torch.isfinite(raw_positions).all():
            raise ValueError("positions contain NaN or infinity")
        atom_numbers = torch.tensor(request.atom_numbers, dtype=torch.long, device=self.device)
        roles = torch.tensor(request.roles, dtype=torch.long, device=self.device)
        residues = torch.tensor(
            request.residue_ids or [0] * nodes, dtype=torch.long, device=self.device
        )
        if request.times_ps is None:
            times = torch.arange(frames, dtype=torch.float32, device=self.device) * self.config.data.dt_ps
        else:
            times = torch.tensor(request.times_ps, dtype=torch.float32, device=self.device)
        if request.return_forces:
            raw_positions.requires_grad_(True)
        # Match training preprocessing: remove each frame's protein translation.
        protein_center = raw_positions[:, roles == 0].mean(dim=1, keepdim=True)
        centered_positions = raw_positions - protein_center
        batch = {
            "positions": centered_positions.unsqueeze(0),
            "atom_numbers": atom_numbers.unsqueeze(0),
            "roles": roles.unsqueeze(0),
            "residue_ids": residues.unsqueeze(0),
            "node_mask": torch.ones((1, nodes), dtype=torch.bool, device=self.device),
            "times": times.unsqueeze(0),
        }
        if request.return_forces:
            with torch.enable_grad():
                output = self.model(batch)
                native = output.energy[0] * float(self.normalization["energy_std"]) + float(
                    self.normalization["energy_mean"]
                )
                forces = -torch.autograd.grad(native.sum(), raw_positions, create_graph=False)[0]
        else:
            with torch.no_grad():
                output = self.model(batch)
                native = output.energy[0] * float(self.normalization["energy_std"]) + float(
                    self.normalization["energy_mean"]
                )
            forces = None
        response = {
            "energy": native.detach().cpu().tolist(),
            "energy_unit": self.normalization.get("energy_unit", "MISATO dataset-native"),
            "mobile_pose_probability": torch.sigmoid(output.state_logits[0]).detach().cpu().tolist(),
            "forces": None if forces is None else forces.detach().cpu().tolist(),
            "force_unit": (
                None
                if forces is None
                else f"{self.normalization.get('energy_unit', 'dataset-native energy')}/angstrom"
            ),
            "variant": self.variant,
            "frames": frames,
            "atoms": nodes,
        }
        if self.unsafe_untrained:
            response["warning"] = "UNTRAINED weights: smoke-test output is not scientifically meaningful"
        return response


app = FastAPI(title="CTDG-MD inference", version="2.0.0")
runtime = Runtime()


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    color = "#43d19e" if runtime.ready and not runtime.unsafe_untrained else "#ffb454"
    status = "READY" if runtime.ready and not runtime.unsafe_untrained else (
        "UNTRAINED SMOKE MODE" if runtime.unsafe_untrained else "CHECKPOINT REQUIRED"
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>CTDG-MD</title>
<style>body{{margin:0;background:#09111f;color:#e9f1ff;font:16px system-ui}}main{{max-width:900px;margin:64px auto;padding:36px;background:#111d31;border:1px solid #263b5e;border-radius:18px}}h1{{font-size:42px;margin:0}}.badge{{display:inline-block;color:#07121c;background:{color};padding:7px 12px;border-radius:999px;font-weight:800;margin:18px 0}}code{{background:#07101d;padding:4px 7px;border-radius:6px}}a{{color:#77bdfb}}li{{margin:10px 0}}</style></head><body><main><h1>CTDG-MD</h1><p>Protein–ligand continuous-time equivariant dynamics with k-temporal virtual nodes.</p><div class='badge'>{status}</div><ul><li>Variant: <code>{runtime.variant or 'not loaded'}</code></li><li>Device: <code>{runtime.device}</code></li><li><a href='/docs'>Interactive API documentation</a></li><li><a href='/health'>Health JSON</a></li></ul></main></body></html>"""


@app.get("/health")
def health() -> JSONResponse:
    payload = {
        "status": "ok" if runtime.ready else "not_ready",
        "checkpoint_loaded": runtime.checkpoint is not None,
        "unsafe_untrained": runtime.unsafe_untrained,
        "variant": runtime.variant,
        "device": str(runtime.device),
    }
    return JSONResponse(payload, status_code=200 if runtime.ready else 503)


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        return runtime.predict(request)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503 if not runtime.ready else 422, detail=str(error)) from error
