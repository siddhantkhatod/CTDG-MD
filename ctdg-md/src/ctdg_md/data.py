from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, TrainingConfig

REQUIRED_KEYS = {
    "atoms_number",
    "atoms_residue",
    "molecules_begin_atom_index",
    "trajectory_coordinates",
    "frames_interaction_energy",
    "frames_rmsd_ligand",
}


@dataclass(frozen=True)
class Normalization:
    energy_mean: float
    energy_std: float
    state_threshold: float
    energy_unit: str = "MISATO dataset-native interaction-energy unit"

    def normalize_energy(self, value: np.ndarray) -> np.ndarray:
        return (value - self.energy_mean) / self.energy_std

    def denormalize_energy(self, value: np.ndarray | Tensor) -> np.ndarray | Tensor:
        return value * self.energy_std + self.energy_mean


@dataclass(frozen=True)
class VerificationReport:
    source: str
    complexes: int
    frames: int
    atoms_min: int
    atoms_max: int
    energy_min: float
    energy_max: float
    finite: bool


def read_split_ids(split_dir: str | Path, split: str, limit: int) -> list[str]:
    path = Path(split_dir) / f"{split}_MD.txt"
    if not path.is_file():
        raise FileNotFoundError(f"MISATO split file is missing: {path}")
    ids = [line.strip().upper() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) < limit:
        raise ValueError(f"{path} has {len(ids)} IDs but {limit} are required")
    return ids[:limit]


def selected_ids(config: DataConfig) -> dict[str, list[str]]:
    return {
        "train": read_split_ids(config.split_dir, "train", config.train_complexes),
        "val": read_split_ids(config.split_dir, "val", config.val_complexes),
        "test": read_split_ids(config.split_dir, "test", config.test_complexes),
    }


def fit_normalization(config: DataConfig, train_ids: Iterable[str]) -> Normalization:
    path = Path(config.hdf5)
    if not path.is_file():
        raise FileNotFoundError(f"MISATO HDF5 file is missing: {path}")
    energies: list[np.ndarray] = []
    states: list[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        for pdb_id in train_ids:
            if pdb_id not in handle:
                raise KeyError(f"PDB ID {pdb_id} from train split is absent from {path}")
            group = handle[pdb_id]
            energies.append(np.asarray(group[config.energy_key][: config.frames_per_complex], dtype=np.float64))
            states.append(np.asarray(group[config.state_key][: config.frames_per_complex], dtype=np.float64))
    energy = np.concatenate(energies)
    state = np.concatenate(states)
    if not np.isfinite(energy).all() or not np.isfinite(state).all():
        raise ValueError("Training labels contain NaN or infinity")
    std = max(float(energy.std()), 1e-8)
    return Normalization(float(energy.mean()), std, float(np.median(state)))


class MISATOTrajectoryDataset(Dataset[dict[str, Any]]):
    """Windowed real protein-ligand trajectories from the MISATO HDF5 schema."""

    def __init__(
        self,
        config: DataConfig,
        ids: list[str],
        normalization: Normalization,
        split: str,
    ) -> None:
        if config.window_size > config.frames_per_complex:
            raise ValueError("window_size cannot exceed frames_per_complex")
        if config.stride < 1 or config.max_nodes < 2:
            raise ValueError("stride and max_nodes are invalid")
        self.config = config
        self.ids = ids
        self.normalization = normalization
        self.split = split
        self.windows = [
            (pdb_id, start)
            for pdb_id in ids
            for start in range(
                0,
                config.frames_per_complex - config.window_size + 1,
                config.stride,
            )
        ]
        self._handle: h5py.File | None = None
        self._selection_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _h5(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.config.hdf5, "r")
        return self._handle

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None and handle.id.valid:
            handle.close()

    def __len__(self) -> int:
        return len(self.windows)

    def _select_pocket(self, pdb_id: str, group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
        cached = self._selection_cache.get(pdb_id)
        if cached is not None:
            return cached
        atomic_numbers = np.asarray(group["atoms_number"][:], dtype=np.int64)
        ligand_start = int(np.asarray(group["molecules_begin_atom_index"][:])[-1])
        if not 0 < ligand_start < atomic_numbers.size:
            raise ValueError(f"{pdb_id}: invalid ligand start index {ligand_start}")
        heavy = atomic_numbers != 1 if self.config.remove_hydrogen else np.ones_like(atomic_numbers, bool)
        protein_indices = np.flatnonzero(heavy[:ligand_start])
        ligand_indices = ligand_start + np.flatnonzero(heavy[ligand_start:])
        if protein_indices.size == 0 or ligand_indices.size == 0:
            raise ValueError(f"{pdb_id}: no protein or ligand atoms after filtering")
        if ligand_indices.size >= self.config.max_nodes:
            raise ValueError(
                f"{pdb_id}: ligand has {ligand_indices.size} atoms, exceeding max_nodes={self.config.max_nodes}"
            )

        # Select the union of protein atoms entering the ligand pocket at any frame.
        coordinates = np.asarray(
            group["trajectory_coordinates"][: self.config.frames_per_complex], dtype=np.float32
        )
        minimum = np.full(protein_indices.size, np.inf, dtype=np.float32)
        for frame in coordinates:
            protein = frame[protein_indices]
            ligand = frame[ligand_indices]
            # Chunking prevents a pathological large temporary for very large ligands.
            frame_minimum = np.full(protein.shape[0], np.inf, dtype=np.float32)
            for left in range(0, ligand.shape[0], 64):
                delta = protein[:, None, :] - ligand[None, left : left + 64, :]
                distance2 = np.einsum("plc,plc->pl", delta, delta, optimize=True)
                frame_minimum = np.minimum(frame_minimum, np.sqrt(distance2.min(axis=1)))
            minimum = np.minimum(minimum, frame_minimum)
        pocket = np.flatnonzero(minimum <= self.config.pocket_cutoff_angstrom)
        available = self.config.max_nodes - ligand_indices.size
        if pocket.size == 0:
            pocket = np.argsort(minimum)[:1]
        pocket = pocket[np.argsort(minimum[pocket], kind="stable")][:available]
        selected_protein = protein_indices[pocket]
        indices = np.concatenate([selected_protein, ligand_indices]).astype(np.int64)
        roles = np.concatenate(
            [np.zeros(selected_protein.size, dtype=np.int64), np.ones(ligand_indices.size, dtype=np.int64)]
        )
        self._selection_cache[pdb_id] = (indices, roles)
        return indices, roles

    def __getitem__(self, item: int) -> dict[str, Any]:
        pdb_id, start = self.windows[item]
        group = self._h5()[pdb_id]
        indices, roles = self._select_pocket(pdb_id, group)
        stop = start + self.config.window_size
        positions = np.asarray(group["trajectory_coordinates"][start:stop], dtype=np.float32)[:, indices]
        # Translation removal uses only selected protein atoms and is frame local.
        protein_count = int((roles == 0).sum())
        centers = positions[:, :protein_count].mean(axis=1, keepdims=True)
        positions = positions - centers
        atomic_numbers = np.asarray(group["atoms_number"][:], dtype=np.int64)[indices]
        residue_ids = np.asarray(group["atoms_residue"][:], dtype=np.int64)[indices]
        raw_energy = np.asarray(group[self.config.energy_key][start:stop], dtype=np.float32)
        state_value = np.asarray(group[self.config.state_key][start:stop], dtype=np.float32)
        state = (state_value >= self.normalization.state_threshold).astype(np.float32)
        normalized = self.normalization.normalize_energy(raw_energy).astype(np.float32)
        times = np.arange(start, stop, dtype=np.float32) * self.config.dt_ps
        return {
            "positions": torch.from_numpy(positions.copy()),
            "atom_numbers": torch.from_numpy(atomic_numbers.copy()),
            "roles": torch.from_numpy(roles.copy()),
            "residue_ids": torch.from_numpy(residue_ids.copy()),
            "energy": torch.from_numpy(normalized.copy()),
            "energy_raw": torch.from_numpy(raw_energy.copy()),
            "state": torch.from_numpy(state.copy()),
            "state_value": torch.from_numpy(state_value.copy()),
            "times": torch.from_numpy(times),
            "pdb_id": pdb_id,
            "start_frame": start,
        }


def collate_trajectories(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(items)
    length = items[0]["positions"].shape[0]
    max_nodes = max(item["positions"].shape[1] for item in items)
    positions = torch.zeros((batch_size, length, max_nodes, 3), dtype=torch.float32)
    atom_numbers = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    roles = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    residues = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    for index, item in enumerate(items):
        nodes = item["positions"].shape[1]
        positions[index, :, :nodes] = item["positions"]
        atom_numbers[index, :nodes] = item["atom_numbers"]
        roles[index, :nodes] = item["roles"]
        residues[index, :nodes] = item["residue_ids"]
        mask[index, :nodes] = True
    return {
        "positions": positions,
        "atom_numbers": atom_numbers,
        "roles": roles,
        "residue_ids": residues,
        "node_mask": mask,
        "energy": torch.stack([item["energy"] for item in items]),
        "energy_raw": torch.stack([item["energy_raw"] for item in items]),
        "state": torch.stack([item["state"] for item in items]),
        "state_value": torch.stack([item["state_value"] for item in items]),
        "times": torch.stack([item["times"] for item in items]),
        "pdb_id": [item["pdb_id"] for item in items],
        "start_frame": torch.tensor([item["start_frame"] for item in items]),
    }


def build_datasets(config: DataConfig) -> tuple[dict[str, MISATOTrajectoryDataset], Normalization]:
    ids = selected_ids(config)
    normalization = fit_normalization(config, ids["train"])
    datasets = {
        split: MISATOTrajectoryDataset(config, values, normalization, split)
        for split, values in ids.items()
    }
    return datasets, normalization


def build_loaders(
    data_config: DataConfig,
    training_config: TrainingConfig,
    seed: int = 0,
) -> tuple[dict[str, DataLoader], Normalization]:
    datasets, normalization = build_datasets(data_config)
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=split == "train",
            num_workers=training_config.num_workers,
            collate_fn=collate_trajectories,
            generator=generator if split == "train" else None,
            persistent_workers=training_config.num_workers > 0,
        )
        for split, dataset in datasets.items()
    }
    return loaders, normalization


def verify_misato_10k(config: DataConfig, report_path: str | Path | None = None) -> VerificationReport:
    path = Path(config.hdf5)
    ids_by_split = selected_ids(config)
    ids = ids_by_split["train"] + ids_by_split["val"] + ids_by_split["test"]
    atom_counts: list[int] = []
    energies: list[np.ndarray] = []
    finite = True
    with h5py.File(path, "r") as handle:
        for pdb_id in ids:
            if pdb_id not in handle:
                raise KeyError(f"Selected PDB ID {pdb_id} is absent from {path}")
            group = handle[pdb_id]
            missing = REQUIRED_KEYS - set(group.keys())
            if missing:
                raise KeyError(f"{pdb_id}: required MISATO keys missing: {sorted(missing)}")
            coordinates = group["trajectory_coordinates"]
            if coordinates.ndim != 3 or coordinates.shape[0] < config.frames_per_complex or coordinates.shape[2] != 3:
                raise ValueError(f"{pdb_id}: invalid coordinate shape {coordinates.shape}")
            for frame_start in range(0, config.frames_per_complex, 10):
                coordinate_chunk = np.asarray(
                    coordinates[frame_start : min(frame_start + 10, config.frames_per_complex)]
                )
                finite = finite and bool(np.isfinite(coordinate_chunk).all())
            ligand_start = int(np.asarray(group["molecules_begin_atom_index"][:])[-1])
            if not 0 < ligand_start < coordinates.shape[1]:
                raise ValueError(f"{pdb_id}: invalid protein/ligand boundary {ligand_start}")
            energy = np.asarray(group[config.energy_key][: config.frames_per_complex])
            state = np.asarray(group[config.state_key][: config.frames_per_complex])
            if energy.shape != (config.frames_per_complex,) or state.shape != (config.frames_per_complex,):
                raise ValueError(f"{pdb_id}: frame-label shape mismatch")
            finite = finite and bool(np.isfinite(energy).all() and np.isfinite(state).all())
            atom_counts.append(int(coordinates.shape[1]))
            energies.append(energy)
    frame_count = len(ids) * config.frames_per_complex
    expected_complexes = config.train_complexes + config.val_complexes + config.test_complexes
    if len(set(ids)) != expected_complexes:
        raise ValueError("Complex leakage or duplicate PDB IDs detected across selected splits")
    if frame_count != 10_000:
        raise ValueError(
            f"Configuration selects {frame_count} frames; the benchmark contract requires exactly 10,000"
        )
    all_energy = np.concatenate(energies)
    report = VerificationReport(
        source=str(path),
        complexes=len(ids),
        frames=frame_count,
        atoms_min=min(atom_counts),
        atoms_max=max(atom_counts),
        energy_min=float(all_energy.min()),
        energy_max=float(all_energy.max()),
        finite=finite,
    )
    if not finite:
        raise ValueError("MISATO subset contains non-finite values")
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report
