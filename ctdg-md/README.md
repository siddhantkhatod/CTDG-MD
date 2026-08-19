# CTDG-MD

A runnable research project for **protein–ligand molecular-dynamics analysis and interaction-energy prediction** using:

- an E(n)-equivariant graph neural network (EGNN),
- a continuous-time dynamic graph (CTDG) memory,
- non-stationary patch Fourier temporal encoding,
- segmented truncated backpropagation through time (TBPTT),
- k-temporal virtual nodes (k-TVNs) based on the attached *Virtual Nodes Go Temporal* paper,
- physics-informed force/torque balance regularization, and
- a four-way EGNN/CTDG/TVN ablation with energy MAE/RMSE/R², AUC, accuracy, AMI, and a pairwise AMI matrix.

The repository contains **no trajectory data or trained weights**. The `data/` directory in the release archive is empty by design.

## Important 10,000-frame assumption

MISATO is a benchmark of real protein–ligand MD simulations, but each published complex has **100 stored trajectory snapshots**, not one 10,000-snapshot continuous trajectory. It would be scientifically wrong to concatenate complexes and call them one continuous trajectory. This project therefore defines **MISATO-10k** as:

| Split | Independent complexes | Real frames/complex | Real frames |
|---|---:|---:|---:|
| train | 80 | 100 | 8,000 |
| validation | 10 | 100 | 1,000 |
| test | 10 | 100 | 1,000 |
| **total** | **100** | **100** | **10,000** |

The IDs come from the published MISATO_1000 train/validation/test files, so complexes do not leak across splits. Every frame is simulated data; **zero synthetic frames are used for training or evaluation**. Windows are non-overlapping by default. The smaller 7.45 GB MISATO_1000 release is used instead of forcing a 132.8 GB full-dataset download.

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

The publication ablation variants are:

1. `constant` — train-only intercept/majority baseline;
2. `schnet` — self-contained continuous-filter SchNet baseline;
3. `painn` — self-contained scalar/vector PaiNN baseline;
4. `egnn` — frame-independent EGNN;
5. `egnn_tvn` — EGNN + k-TVNs;
6. `ctdg` — EGNN + continuous-time memory + patch Fourier temporal encoder;
7. `ctdg_tvn` — complete model.

MACE, NequIP, classical MM/GBSA and NeuralMD/BioMD are supported through strict export/import adapters because they have separate official environments and, for classical/force evaluation, require topology or labels absent from the coordinate-only MISATO HDF5. See `docs/PUBLICATION_PROTOCOL.md`.

## Why AUC, accuracy, and AMI are valid here

Energy is a regression target, so AUC/accuracy do not apply directly to it. The project adds a real, non-synthetic auxiliary physical-state task:

- **state label:** frame ligand RMSD is above/below the median computed **only on training complexes**;
- **AUC/accuracy:** evaluate the mobile-pose probability head;
- **AMI:** K-means clusters of test embeddings versus the physical RMSD state;
- **AMI matrix:** pairwise AMI between test embedding clusters from the four model variants.

Energy quality is reported separately with MAE, RMSE, and R² in MISATO's dataset-native interaction-energy unit.

## Installation

Python 3.10+ is required. For a CPU environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e '.[dev]'
```

For CUDA, install the matching PyTorch wheel first, then `pip install -e '.[dev]'`.

## Download and validate the real benchmark

```bash
ctdg-md download --destination data/MISATO_1000/raw
ctdg-md verify --config configs/default.yaml
```

The downloader supports `.part` resume, checks the published 7,455,614,516-byte size, and by default verifies SHA-256 `b202e1c3...ddeb8b4c`. It then records provenance locally. Expect about 7.45 GB plus HDF5 access overhead.

`verify` fails unless all required keys exist, all 100 selected PDB IDs are unique across splits, every selected complex contains at least 100 finite frames, and the exact total is 10,000.

## Train

```bash
ctdg-md train --config configs/default.yaml --variant ctdg_tvn --seed 42
```

Checkpoints go to `outputs/checkpoints/`. The trainer uses AdamW, gradient clipping, early stopping, and explicit segmented TBPTT. Model memory values cross segment boundaries but are detached there, making the long-range memory cost bounded.

## Full ablation

```bash
ctdg-md ablate --config configs/default.yaml
```

Outputs:

- `outputs/ablation/ablation_runs.csv` — every seed/run;
- `outputs/ablation/ablation_summary.csv` — mean and standard deviation;
- `outputs/ablation/ami_matrix.csv` — pairwise model-cluster agreement;
- `outputs/ablation/best_models.json` — winner for AUC, accuracy, AMI and aggregate mean-rank;
- per-run histories, checkpoints, and test predictions.

The publication default is 7 variants × 5 seeds × up to 50 epochs. Use GPUs or the supplied SLURM array. For a runtime smoke test, make a copied config with `epochs: 1`, smaller hidden dimensions, and fewer selected complexes; do **not** report that smoke run as a benchmark improvement.

To run the claim-gated study:

```bash
ctdg-md publication --config configs/default.yaml
```

The generated report uses paired complex-level bootstrap confidence intervals, paired sign-flip tests, Holm multiple-comparison correction, a positive-R² gate, and explicit blocks for missing force, rollout or external-model evidence.

## Deploy inference

A trained checkpoint is mandatory by default:

```bash
export CTDG_CHECKPOINT=outputs/checkpoints/ctdg_tvn_seed42.pt
ctdg-md serve --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` or `/docs`. `POST /predict` accepts frame coordinates, atom numbers, protein/ligand roles, optional residue IDs, and optional irregular timestamps. It returns interaction energies, conformational-state probabilities, and—by default—forces computed as `-dE/dx`. Set `return_forces: false` for lower-cost energy/state-only inference.

Docker:

```bash
docker build -t ctdg-md .
docker run --rm -p 8000:8000 \
  -e CTDG_CHECKPOINT=/models/ctdg_tvn_seed42.pt \
  -v "$PWD/outputs/checkpoints:/models:ro" ctdg-md
```

`CTDG_ALLOW_UNTRAINED=1` exists only for API plumbing smoke tests; responses are explicitly marked unsafe and are not scientifically meaningful.

## Verification

```bash
python scripts/verify_implementation.py --skip-data   # compile + all unit/integration tests
python scripts/verify_implementation.py               # also validate downloaded real data
```

Tests cover graph construction, exact adjacency-row clustering, TVN connectivity, E(n) invariance/equivariance for all seven internal variants, irregular-time memory, patch Fourier gradients, conservation residuals, data loading, paired bootstrap/Holm statistics, external-result alignment, force/trajectory metrics, velocity-Verlet rollout, every model's forward/backward pass, and API fail-fast behavior.

## Physics scope

`frames_interaction_energy` is the target. Forces are obtained as `-dE/dx`; net force and net torque residuals regularize translation/rotation invariance. MISATO does not provide force labels, so this project does **not** claim force-supervised accuracy. Because MISATO simulations use a thermostatted/barostatted ensemble and the label is interaction energy rather than isolated-system total energy, the code intentionally does not force energy to remain constant. A very small temporal curvature penalty is configurable.

## References

- Satorras, Hoogeboom & Welling, [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844), 2021.
- Rossi et al., [Temporal Graph Networks](https://arxiv.org/abs/2006.10637), 2020.
- Ennadir et al., [Virtual Nodes Go Temporal](https://openreview.net/forum?id=jdKtkiH9ze), LoG 2025; [official code](https://github.com/king/TVNs).
- Siebenmorgen et al., [MISATO](https://doi.org/10.1038/s43588-024-00627-2), Nature Computational Science 2024.
- Liu et al., [NeuralMD](https://github.com/chao1224/NeuralMD), Nature Communications 2025.

See `docs/PUBLICATION_PROTOCOL.md`, `docs/V2_VERIFICATION.md`, `docs/FLOWCHART.md`, `docs/IMPLEMENTATION_AUDIT.md`, `docs/BUG_AUDIT_V1.1.md`, `docs/DATA_CARD.md`, and `docs/MODEL_CARD.md` for the claim gates, verification, implementation decisions and limitations.
