# Data card: MISATO-10k

## Source

- Upstream dataset: MISATO — Machine learning dataset of protein–ligand complexes for structure-based drug discovery.
- Publication: https://doi.org/10.1038/s43588-024-00627-2
- Full record: https://zenodo.org/records/7711953
- Derived 1,000-complex HDF5 used here: https://huggingface.co/datasets/chao1224/NeuralMD/tree/main/MISATO_1000/raw
- Downloaded size: 7,455,614,516 bytes for `MD.hdf5`.
- Data are not redistributed inside this project.

## Exact selection

The first 80 IDs in `train_MD.txt`, first 10 in `val_MD.txt`, and first 10 in `test_MD.txt`; 100 frames per ID. The verifier requires 100 distinct complexes and exactly 10,000 finite frames. These are 100 independent real MD trajectories, not an artificial concatenated trajectory.

## Fields used

- `trajectory_coordinates`: positions in each stored MD frame;
- `atoms_number`: atomic number;
- `atoms_residue`: residue index;
- `molecules_begin_atom_index[-1]`: protein/ligand boundary;
- `frames_interaction_energy`: regression target;
- `frames_rmsd_ligand`: physical state used for classification evaluation.

## Preprocessing

1. Remove hydrogen atoms by default.
2. Retain every ligand heavy atom.
3. Across all 100 frames, find protein atoms that ever enter the configured ligand-pocket radius.
4. Keep nearest pocket atoms up to `max_nodes`, with at least one protein atom.
5. Center each frame on selected protein atoms. This removes only translation.
6. Build dynamic graph edges in the model, not in the stored data.
7. Fit interaction-energy mean/std and ligand-RMSD state threshold on training complexes only.

## Leakage controls

- PDB IDs are disjoint across train/validation/test.
- Normalization and classification threshold use training only.
- Default windows are length 10, stride 10: no duplicated frames.
- Test is never used for early stopping.

## Scope and limitations

- 100 snapshots per complex are temporally coarse.
- The target is MISATO interaction energy, not experimental binding free energy and not isolated-system total potential energy.
- Pocket truncation trades global protein context for tractable atom graphs; change `max_nodes` and cutoff after a memory study.
- The downloader requires substantial disk and network bandwidth.
- Users remain responsible for reviewing and complying with upstream dataset terms and citing MISATO/NeuralMD.
