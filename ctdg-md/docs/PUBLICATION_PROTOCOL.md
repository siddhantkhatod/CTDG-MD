# Publication-scale comparison and claim protocol

## Claim scope

This project distinguishes four claims and never merges them:

1. **Same-target interaction-energy improvement** — CTDG+TVN versus constant, SchNet, PaiNN, EGNN, EGNN+TVN and CTDG, all trained on identical MISATO pockets/splits.
2. **Current external energy-model improvement** — additionally requires aligned MACE and NequIP predictions and the classical MM/GBSA comparator.
3. **Force improvement** — requires supervised per-atom force labels. Standard MISATO does not contain them, so this claim is blocked by default.
4. **Trajectory improvement** — requires aligned NeuralMD/BioMD trajectory outputs and rollout metrics. Energy MAE cannot establish this claim.

`publication_report.json` contains independent Boolean gates for each claim. Missing evidence produces `false`, never an inferred success.

## Mandatory internal study

Default protocol:

- exact 100-complex/10,000-frame MISATO selection;
- leakage-free 80/10/10 complex split;
- 50 epochs, validation early stopping patience 7;
- seeds 11, 22, 33, 44 and 55;
- identical pocket selection, atom cap, radius and test frames;
- constant, SchNet, PaiNN, EGNN, EGNN+TVN, CTDG and CTDG+TVN;
- energy MAE/RMSE/R² plus state AUC/accuracy/AMI;
- paired cluster bootstrap over independent test complexes;
- paired sign-flip tests with Holm family-wise correction;
- positive mean test R² required by default.

Run on one GPU:

```bash
ctdg-md publication --config configs/default.yaml
```

Run the 35 model/seed jobs as a SLURM array:

```bash
sbatch deployment/slurm_publication_array.sh
# after every array task succeeds
ctdg-md publication --config configs/default.yaml --aggregate-only
```

A same-target improvement passes only when CTDG+TVN has a paired MAE-difference confidence interval entirely below zero against every required baseline, every Holm-adjusted p-value is below 0.05, five seeds are present and mean R² is positive.

## MACE and NequIP

Export exactly the same centered pockets and labels:

```bash
ctdg-md export-external --config configs/default.yaml --output outputs/external/export
```

The exporter creates train/validation/test extended-XYZ files and a frame-level manifest. `REF_energy` is MISATO interaction energy—not total potential energy—and forces are intentionally absent.

Run official installations using the supplied energy-only configurations:

```bash
scripts/run_mace_energy_only.sh configs/mace_energy_only.yaml
scripts/run_nequip_energy_only.sh configs/nequip_energy_only.yaml
```

Convert each external model's test output to CSV columns `pdb_id,frame,energy_pred`, then align it strictly:

```bash
ctdg-md import-external-energy \
  --manifest outputs/external/export/manifest.csv \
  --predictions mace_seed11.csv \
  --output outputs/external/mace/seed11/test_predictions.npz
```

Repeat for every seed. Unexpected, duplicate or missing frames cause failure.

## Classical MM/GBSA

The coordinate-only MISATO HDF5 cannot reproduce classical MM/GBSA because atom charges, bonded parameters, solvent/topology and boundary conditions are absent. Obtain the separately distributed MISATO parameter/restart archive and audited Protein/Ligand index groups, then use:

```bash
scripts/run_gromacs_mmgbsa.sh system.tpr trajectory.xtc index.ndx run_prefix PDB_ID
```

Import its aligned test energies under `outputs/external/gromacs_mmgbsa/seed*/test_predictions.npz`. This evidence is mandatory for the broad external-energy claim.

## NeuralMD and BioMD

Run each official model on the identical test PDB IDs. Package outputs into:

- `truth`: `[samples, frames, atoms, 3]`
- `predicted`: same shape/order
- `node_mask`: `[samples, atoms]`
- `roles`: `[samples, atoms]`, with 0 protein and 1 ligand

Then run:

```bash
ctdg-md evaluate-trajectory \
  --predictions neuralmd_aligned.npz \
  --output outputs/external/neuralmd/trajectory_metrics.json
```

The evaluator performs protein Kabsch alignment and reports all-atom coordinate MAE, ligand RMSD, finite fraction and step stability. Merely importing this file does not automatically pass a superiority claim; a paired trajectory comparison must still be reviewed.

## Forces and rollouts

`physics_eval.force_metrics` computes force MAE/RMSE/cosine similarity when actual labels exist. With standard MISATO, the report explicitly says labels are unavailable.

`physics_eval.velocity_verlet_rollout` is a complete energy-gradient velocity-Verlet implementation for fine-timestep data with initial velocities. The 80 ps gap between stored MISATO frames is not a physically valid integration step, so rollout is blocked from being presented as a MISATO force-trajectory result.

## Reporting rule

Use only statements corresponding to a `true` gate in `outputs/publication/publication_report.json`. If external evidence or force labels are absent, the allowed wording is limited to improvement over the included same-target baselines.
