# Model card: CTDG-MD

## Intended use

Research comparison of geometric and temporal graph inductive biases for analysing real protein–ligand MD frames and predicting MISATO interaction energies. It is not a clinical, medicinal-chemistry, safety, or autonomous drug-design decision system.

## Variants

- `constant`: train-only intercept/majority baseline;
- `schnet`: continuous-filter convolution baseline;
- `painn`: scalar/vector equivariant baseline;
- `egnn`: static frame baseline;
- `egnn_tvn`: static frame baseline with temporal-style frame VNs;
- `ctdg`: continuous-time memory + patch Fourier encoder;
- `ctdg_tvn`: complete model.

External MACE, NequIP, classical MM/GBSA and NeuralMD/BioMD results are accepted only through strict same-frame adapters and are never silently treated as equivalent tasks.

## Inputs

A time-ordered position tensor, atomic numbers, residue indices, protein/ligand role, and timestamps. Dynamic edges are created from geometry. Coordinates are in angstrom; time is in ps in the default MISATO configuration.

## Outputs

- interaction-energy estimate in the training dataset's native unit;
- probability that ligand RMSD is above the train-only median;
- graph embedding used for AMI;
- equivariantly updated coordinates for implementation checking/analysis.

## Training objective

Smooth-L1 interaction-energy loss + binary cross entropy + optional force-balance, torque-balance and small temporal-curvature penalties. Force/torque residuals are second derivatives during optimization and increase cost.

## Evaluation

- Regression: MAE, RMSE, R².
- State: ROC-AUC and threshold-0.5 accuracy.
- Representation: adjusted mutual information versus RMSD state.
- Cross-model representation: pairwise AMI matrix.
- At least three random seeds are default; report mean and standard deviation.

## Known limitations

1. No supervised forces are available in MISATO.
2. The auxiliary two-state definition is operational, not a universal definition of bound/unbound conformations.
3. Fourier patches see all frames within a patch. This is suitable for trajectory analysis, not strictly causal online forecasting.
4. Radius-graph topology changes are nondifferentiable at cutoff boundaries.
5. TVN clustering is discrete and adds O(N²) dense adjacency-row work for the bounded pocket graph.
6. Energy invariance is architectural, but finite precision and changing cutoff topology can introduce numerical deviations.
7. Performance is unknown until the real-data ablation is actually run; the project does not ship fabricated scores.

## Reproducibility

Configuration, normalization, variant, epoch, optimizer state and model state are checkpointed. Deterministic algorithms are requested with warnings for unsupported kernels. Exact hardware/software versions should accompany published results.
