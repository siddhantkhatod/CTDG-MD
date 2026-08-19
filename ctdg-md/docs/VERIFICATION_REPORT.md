# Verification report

Date: 2026-08-15

## Pass 1 — component and mathematical checks

Historical v1.1 result: **21/21 tests passed**. The expanded v2 suite passes **31/31 tests**; see `V2_VERIFICATION.md`.

Covered radius graph behavior, radial cutoff, adjacency-row k-means, disconnected-community recovery, child/VN/clique edge counts, equivariant VN centroids, irregular continuous-time encoding, decayed memory gradients, patch Fourier forward/backward, force/torque conservation, AUC/accuracy/AMI, MISATO-schema loading, and API fail-fast behavior.

## Pass 2 — all-model implementation checks

Result: **passed for `egnn`, `egnn_tvn`, `ctdg`, and `ctdg_tvn`**.

For every variant:

- energy was invariant after a rigid rotation + translation;
- updated coordinates transformed equivariantly;
- forward and backward completed with finite gradients.

`compileall`, Ruff static checks, and the complete test suite then passed together.

## Pass 3 — trainer/checkpoint/deployment runtime

Result: **passed for all four variants** on a temporary, deterministic MISATO-schema integration fixture (not project training data):

- one complete epoch;
- segmented TBPTT;
- second-order force/torque regularization;
- validation and metric computation;
- early-stopping checkpoint write/read;
- test prediction NPZ write;
- trained-checkpoint FastAPI startup;
- `/health` ready response;
- valid `/predict` energy, force and state-probability response.

Additional v1.1 regression checks prove that segmented CTDG/CTDG+TVN values match full-window values with irregular timestamps and aligned Fourier patches. See `BUG_AUDIT_V1.1.md`.

## Real-data check status

The 7.45 GB HDF5 was intentionally **not bundled or downloaded into this project**, per the empty-data requirement. Therefore no fabricated benchmark score is reported. `ctdg-md verify` is a mandatory fail-fast real-data gate and must pass after the user runs `ctdg-md download`.

## Residual risks

No finite test process proves correctness for every dataset corruption, hardware/driver combination, or scientific use. Full-scale memory, throughput, hyperparameter sensitivity, and benchmark performance require the real-data run on the target hardware. These limitations are explicit rather than hidden behind claimed “perfect” validation.
