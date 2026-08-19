from pathlib import Path

import h5py
import numpy as np

from ctdg_md.config import DataConfig
from ctdg_md.data import MISATOTrajectoryDataset, Normalization, collate_trajectories


def make_fixture(path: Path):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("TEST")
        # Four protein atoms and two ligand atoms; coordinates are deterministic test data.
        base = np.array(
            [[0, 0, 0], [1, 0, 0], [8, 8, 8], [0, 1, 0], [2, 0, 0], [2, 1, 0]], dtype=np.float32
        )
        trajectory = np.stack([base + np.array([frame, 0, 0]) * 0.01 for frame in range(4)])
        group.create_dataset("trajectory_coordinates", data=trajectory)
        group.create_dataset("atoms_number", data=np.array([6, 1, 6, 8, 6, 7]))
        group.create_dataset("atoms_residue", data=np.array([1, 1, 2, 2, 3, 3]))
        group.create_dataset("molecules_begin_atom_index", data=np.array([0, 4]))
        group.create_dataset("frames_interaction_energy", data=np.array([-2, -1, 0, 1], dtype=np.float32))
        group.create_dataset("frames_rmsd_ligand", data=np.array([0.1, 0.2, 0.8, 1.0], dtype=np.float32))


def test_misato_loader_filters_hydrogen_selects_pocket_and_collates(tmp_path):
    hdf = tmp_path / "fixture.h5"
    make_fixture(hdf)
    config = DataConfig(
        hdf5=str(hdf),
        frames_per_complex=4,
        window_size=2,
        stride=2,
        max_nodes=5,
        pocket_cutoff_angstrom=4.0,
        dt_ps=2.0,
    )
    normalization = Normalization(-0.5, 1.0, 0.5)
    dataset = MISATOTrajectoryDataset(config, ["TEST"], normalization, "train")
    assert len(dataset) == 2
    first, second = dataset[0], dataset[1]
    assert first["positions"].shape[0] == 2
    assert not (first["atom_numbers"] == 1).any()
    assert set(first["roles"].tolist()) == {0, 1}
    assert first["state"].tolist() == [0.0, 0.0]
    assert second["state"].tolist() == [1.0, 1.0]
    batch = collate_trajectories([first, second])
    assert batch["positions"].shape[:2] == (2, 2)
    assert batch["node_mask"].all()
