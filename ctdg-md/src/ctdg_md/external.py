from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import DataConfig
from .data import build_datasets


ELEMENTS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
    "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb",
    "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs",
    "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
    "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg",
    "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]


def export_same_target_extxyz(config: DataConfig, output_dir: str | Path) -> dict[str, Any]:
    """Export identical preprocessed frames for MACE/NequIP energy-only training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets, normalization = build_datasets(config)
    counts: dict[str, int] = {}
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_handle:
        manifest = csv.DictWriter(
            manifest_handle,
            fieldnames=["split", "pdb_id", "frame", "energy_true", "extxyz_index"],
        )
        manifest.writeheader()
        global_index = 0
        for split, dataset in datasets.items():
            count = 0
            path = output_dir / f"{split}.extxyz"
            with path.open("w", encoding="utf-8") as handle:
                for window_index in range(len(dataset)):
                    item = dataset[window_index]
                    atoms = item["atom_numbers"].numpy()
                    roles = item["roles"].numpy()
                    for local_frame, (positions, energy) in enumerate(
                        zip(item["positions"].numpy(), item["energy_raw"].numpy())
                    ):
                        frame = int(item["start_frame"] + local_frame)
                        handle.write(f"{len(atoms)}\n")
                        handle.write(
                            f"REF_energy={float(energy):.12g} pdb_id={item['pdb_id']} "
                            f"frame={frame} split={split} pbc=\"F F F\" "
                            "Properties=species:S:1:pos:R:3:role:I:1\n"
                        )
                        for atomic_number, xyz, role in zip(atoms, positions, roles):
                            if not 1 <= int(atomic_number) < len(ELEMENTS):
                                raise ValueError(f"Unsupported atomic number {atomic_number}")
                            handle.write(
                                f"{ELEMENTS[int(atomic_number)]} "
                                f"{xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g} {int(role)}\n"
                            )
                        manifest.writerow(
                            {
                                "split": split,
                                "pdb_id": item["pdb_id"],
                                "frame": frame,
                                "energy_true": float(energy),
                                "extxyz_index": global_index,
                            }
                        )
                        global_index += 1
                        count += 1
            counts[split] = count
    metadata = {
        "counts": counts,
        "energy_key": "REF_energy",
        "force_key": None,
        "normalization": normalization.__dict__,
        "warning": "MISATO interaction energy is not total potential energy; force labels are unavailable",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def import_external_energy_csv(
    manifest_path: str | Path,
    prediction_csv: str | Path,
    output_npz: str | Path,
) -> Path:
    """Validate and align external predictions into the publication NPZ contract."""
    with Path(manifest_path).open(encoding="utf-8") as handle:
        reference = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    with Path(prediction_csv).open(encoding="utf-8") as handle:
        predicted_rows = list(csv.DictReader(handle))
    required = {"pdb_id", "frame", "energy_pred"}
    if not predicted_rows or not required.issubset(predicted_rows[0]):
        raise ValueError(f"Prediction CSV requires columns {sorted(required)}")
    predictions: dict[tuple[str, int], float] = {}
    for row in predicted_rows:
        key = (row["pdb_id"], int(row["frame"]))
        if key in predictions:
            raise ValueError(f"Duplicate external prediction {key}")
        predictions[key] = float(row["energy_pred"])
    truth, estimate, groups, frames = [], [], [], []
    for row in reference:
        key = (row["pdb_id"], int(row["frame"]))
        if key not in predictions:
            raise ValueError(f"Missing external prediction {key}")
        truth.append(float(row["energy_true"]))
        estimate.append(predictions[key])
        groups.append(key[0])
        frames.append(key[1])
    if len(predictions) != len(reference):
        raise ValueError("External prediction CSV contains unexpected test rows")
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        energy_true=np.asarray(truth),
        energy_pred=np.asarray(estimate),
        pdb_id=np.asarray(groups, dtype="U16"),
        frame_index=np.asarray(frames, dtype=np.int64),
    )
    return output_npz


def _kabsch(reference: np.ndarray, mobile: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ref = reference[mask]
    mob = mobile[mask]
    ref_center, mob_center = ref.mean(0), mob.mean(0)
    covariance = (mob - mob_center).T @ (ref - ref_center)
    left, _, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = -1.0 if np.linalg.det(left @ right) < 0.0 else 1.0
    rotation = left @ correction @ right
    return (mobile - mob_center) @ rotation + ref_center


def evaluate_external_trajectory(npz_path: str | Path, output_json: str | Path) -> dict[str, float]:
    """Evaluate aligned all-atom trajectories from NeuralMD/BioMD-like models.

    Required arrays: truth/predicted [S,T,N,3], node_mask [S,N], roles [S,N].
    Protein atoms align every frame; ligand RMSD and all-atom MAE are then measured.
    """
    payload = np.load(npz_path)
    required = {"truth", "predicted", "node_mask", "roles"}
    if not required.issubset(payload.files):
        raise ValueError(f"Trajectory NPZ requires arrays {sorted(required)}")
    truth = payload["truth"]
    prediction = payload["predicted"]
    node_mask = payload["node_mask"].astype(bool)
    roles = payload["roles"]
    if truth.shape != prediction.shape or truth.ndim != 4 or truth.shape[-1] != 3:
        raise ValueError("truth and predicted must match [samples, frames, atoms, 3]")
    finite_atoms = np.isfinite(prediction).all(axis=-1)
    trajectory_mask = np.broadcast_to(node_mask[:, None, :], finite_atoms.shape)
    finite_fraction = float(finite_atoms[trajectory_mask].mean())
    ligand_rmsd, all_atom_mae, stable = [], [], []
    for sample in range(truth.shape[0]):
        valid = node_mask[sample]
        protein = valid & (roles[sample] == 0)
        ligand = valid & (roles[sample] == 1)
        if protein.sum() < 3 or ligand.sum() < 1:
            raise ValueError("Each trajectory needs >=3 protein and >=1 ligand atoms")
        aligned_frames = []
        for frame in range(truth.shape[1]):
            aligned = _kabsch(truth[sample, frame], prediction[sample, frame], protein)
            aligned_frames.append(aligned)
            ligand_rmsd.append(
                float(np.sqrt(np.mean(np.sum((aligned[ligand] - truth[sample, frame, ligand]) ** 2, axis=-1))))
            )
            all_atom_mae.append(float(np.mean(np.abs(aligned[valid] - truth[sample, frame, valid]))))
        displacement = np.linalg.norm(np.diff(np.asarray(aligned_frames)[:, ligand], axis=0), axis=-1)
        stable.append(float(np.mean(displacement < 10.0)))
    result = {
        "coordinate_mae": float(np.mean(all_atom_mae)),
        "ligand_rmsd": float(np.mean(ligand_rmsd)),
        "finite_fraction": finite_fraction,
        "step_stability_fraction": float(np.mean(stable)),
        "samples": int(truth.shape[0]),
        "frames_per_sample": int(truth.shape[1]),
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
