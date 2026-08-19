import numpy as np
from fastapi.testclient import TestClient

from ctdg_md.api import PredictionRequest, Runtime, app
from ctdg_md.config import ModelConfig, ProjectConfig, save_config


def test_api_health_and_fail_fast_without_checkpoint():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code in {200, 503}
    assert health.json()["status"] in {"not_ready", "ok"}
    response = client.post(
        "/predict",
        json={
            "positions": [[[0, 0, 0], [1, 0, 0]]],
            "atom_numbers": [6, 6],
            "roles": [0, 1],
        },
    )
    if health.json()["status"] == "not_ready":
        assert response.status_code == 503


def test_api_real_force_path_in_untrained_smoke_mode(tmp_path, monkeypatch):
    config = ProjectConfig()
    config.data.max_nodes = 8
    config.model = ModelConfig(
        hidden_dim=12,
        message_dim=12,
        num_layers=1,
        radial_basis=4,
        max_neighbors=4,
        time_dim=4,
        patch_size=2,
        dropout=0,
        fourier_dropout=0,
    )
    path = tmp_path / "config.yaml"
    save_config(config, path)
    monkeypatch.delenv("CTDG_CHECKPOINT", raising=False)
    monkeypatch.setenv("CTDG_ALLOW_UNTRAINED", "1")
    monkeypatch.setenv("CTDG_CONFIG", str(path))
    monkeypatch.setenv("CTDG_DEVICE", "cpu")
    runtime = Runtime()
    result = runtime.predict(
        PredictionRequest(
            positions=[
                [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                [[0, 0.1, 0], [1, 0.1, 0], [2, 0.1, 0]],
            ],
            atom_numbers=[6, 7, 8],
            roles=[0, 0, 1],
            times_ps=[0, 1],
            return_forces=True,
        )
    )
    forces = np.asarray(result["forces"])
    assert forces.shape == (2, 3, 3)
    assert np.isfinite(forces).all()
    assert np.linalg.norm(forces.sum(axis=1)) < 1e-4
    assert result["force_unit"].endswith("/angstrom")
