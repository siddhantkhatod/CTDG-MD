import csv

import numpy as np

from ctdg_md.external import evaluate_external_trajectory, import_external_energy_csv
from ctdg_md.physics_eval import force_metrics


def test_external_energy_alignment(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["split", "pdb_id", "frame", "energy_true", "extxyz_index"]
        )
        writer.writeheader()
        writer.writerow({"split": "test", "pdb_id": "A", "frame": 0, "energy_true": -1, "extxyz_index": 0})
        writer.writerow({"split": "test", "pdb_id": "A", "frame": 1, "energy_true": -2, "extxyz_index": 1})
    predictions = tmp_path / "pred.csv"
    with predictions.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdb_id", "frame", "energy_pred"])
        writer.writeheader()
        writer.writerow({"pdb_id": "A", "frame": 0, "energy_pred": -1.1})
        writer.writerow({"pdb_id": "A", "frame": 1, "energy_pred": -1.9})
    output = import_external_energy_csv(manifest, predictions, tmp_path / "out.npz")
    payload = np.load(output)
    assert np.allclose(payload["energy_pred"], [-1.1, -1.9])


def test_trajectory_and_force_metrics(tmp_path):
    truth = np.zeros((1, 2, 4, 3))
    truth[0, :, :, 0] = np.arange(4)
    predicted = truth + np.array([2.0, -1.0, 0.5])
    path = tmp_path / "trajectory.npz"
    np.savez(
        path,
        truth=truth,
        predicted=predicted,
        node_mask=np.ones((1, 4), dtype=bool),
        roles=np.array([[0, 0, 0, 1]]),
    )
    result = evaluate_external_trajectory(path, tmp_path / "metrics.json")
    assert result["ligand_rmsd"] < 1e-8
    reference_force = np.ones((2, 4, 3))
    predicted_force = reference_force + 0.1
    metrics = force_metrics(reference_force, predicted_force, np.ones((2, 4), dtype=bool))
    assert np.isclose(metrics.mae, 0.1)
