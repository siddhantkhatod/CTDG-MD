# CTDG-MD v2 publication workflow verification

Date: 2026-08-16

## Automated checks

- Ruff static analysis: passed.
- Compilation of every source, test and script: passed.
- Unit/integration tests: **31/31 passed**.
- All seven internal variants completed forward/backward and equivariance checks.
- Cluster bootstrap, sign-flip and Holm correction tests passed.
- External prediction alignment, trajectory Kabsch/RMSD and force metric tests passed.
- Energy-gradient velocity-Verlet runtime test passed.

## Real-data expanded ablation smoke run

All seven internal variants were executed end-to-end for one epoch on the official real MISATO `tiny_md.hdf5` sample (SHA-256 `554d20ea0822949e1a5b0dc1826f50b87e71de5a7ed83ac1fb5a2e68c64547de`). This run covered:

- constant baseline;
- SchNet;
- PaiNN;
- EGNN;
- EGNN + TVN;
- CTDG;
- CTDG + TVN;
- training, validation, checkpoint reload, test metrics and ablation aggregation.

The smoke run is a runtime verification only. Its one-epoch/single-seed numbers are intentionally not included as evidence of improvement. The temporary HDF5, dependencies, checkpoints and prediction arrays were deleted after verification.

## Publication claim behavior

The full command requires five seeds and exact MISATO-10k verification. It emits independent gates for:

- included same-target baselines;
- external MACE/NequIP/MMGBSA evidence;
- force evidence;
- NeuralMD/BioMD trajectory evidence.

Missing external results, missing force labels, negative R², confidence intervals crossing zero or Holm-adjusted p-values above the configured threshold keep the corresponding claim false.
