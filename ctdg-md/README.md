# CTDG-MD

A runnable research project for **protein–ligand molecular-dynamics analysis and interaction-energy prediction** using:

- an E(n)-equivariant graph neural network (EGNN),
- a continuous-time dynamic graph (CTDG) memory,
- non-stationary patch Fourier temporal encoding,
- segmented truncated backpropagation through time (TBPTT),
- k-temporal virtual nodes (k-TVNs) based on the attached *Virtual Nodes Go Temporal* paper,
- physics-informed force/torque balance regularization, and
- a four-way EGNN/CTDG/TVN ablation with energy MAE/RMSE/R², AUC, accuracy, AMI, and a pairwise AMI matrix.

The repository contains **no trajectory data or trained weights**.

## Important 10,000-frame assumption

MISATO is a benchmark of real protein–ligand MD simulations, but each published complex has **100 stored trajectory snapshots**, not one 10,000-snapshot continuous trajectory. It would be scientifically wrong to concatenate complexes and call them one continuous trajectory. This project therefore defines **MISATO-10k** as:

| Split | Independent complexes | Real frames/complex | Real frames |
|---|---:|---:|---:|
| train | 80 | 100 | 8,000 |
| validation | 10 | 100 | 1,000 |
| test | 10 | 100 | 1,000 |
| **total** | **100** | **100** | **10,000** |

The IDs come from the published MISATO_1000 train/validation/test files, so complexes do not leak across splits. Every frame is simulated data; **zero synthetic frames are used for training or evaluation**. The smaller 7.45 GB MISATO_1000 release is used.

MISATO combines curated experimental protein–ligand structures with explicit-solvent MD and provides `trajectory_coordinates`, ligand RMSD, and `frames_interaction_energy` labels. See [Siebenmorgen et al., 2024](https://doi.org/10.1038/s43588-024-00627-2), [Zenodo 7711953](https://zenodo.org/records/7711953), and the [MISATO repository](https://github.com/t7morgen/misato-dataset). The downloadable subset is the [NeuralMD MISATO_1000 release](https://huggingface.co/datasets/chao1224/NeuralMD/tree/main/MISATO_1000/raw).

## Architecture

```text
MD trajectory frames + timestamps
  -> frame-specific radius/contact graph (atoms are nodes)
  -> atomic/role/residue/speed embedding
  -> optional adjacency-row k-means communities
  -> optional k-TVNs: child<->VN and fully-connected VN clique
  -> E(n)-equivariant message and coordinate updates
  -> continuously decayed per-atom GRU memory
  -> patch rFFT, time-conditioned frequency gates, complex filter, irFFT
  -> protein / ligand / global pooling
  -> interaction-energy regression + ligand-pose-state classification
```

## Physics scope

`frames_interaction_energy` is the target. Forces are obtained as `-dE/dx`; net force and net torque residuals regularize translation/rotation invariance. MISATO does not provide force labels, so this project does **not** claim force-supervised accuracy. Because MISATO simulations use a thermostatted/barostatted ensemble and the label is interaction energy rather than isolated-system total energy, the code intentionally does not force energy to remain constant. A very small temporal curvature penalty is configurable.
