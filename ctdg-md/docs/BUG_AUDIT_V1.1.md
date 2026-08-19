# Full bug audit — v1.1.0

Date: 2026-08-15

The source, tests, configuration, downloader, checkpoint path, metric path, API, deployment files, archive layout, and file manifest were inspected. The audit found and fixed the following issues before this release.

| Finding | Consequence before fix | Fix and regression check |
|---|---|---|
| Forces existed only inside the physics loss | Flowchart output “forces” was not exposed to users | `/predict` now returns `-dE/dx` in dataset-native energy/angstrom; trained-checkpoint and API force tests pass |
| TBPTT boundary used an approximate first interval | Irregular timestamps could decay memory by the wrong amount | Every segment carries exact `previous_time` and `previous_positions` |
| First-frame speed reset at segment boundaries | Segmented CTDG values could differ from full-window values | Exact boundary speed is carried; speed is detached only from conservative coordinate derivatives |
| Patch Fourier time origin reset for each segment | Time-conditioned spectral gates differed between training segments and evaluation | Original window time origin is carried through every segment |
| TVN k-means seed reset per segment | A segmented TVN run could initialize clustering differently from full evaluation | Absolute frame offset is carried and used in the per-frame seed |
| EGNN baseline used speed and delta-time features | The intended static ablation was not fully static | EGNN now uses no speed, timestamp, memory, or Fourier context |
| One DataLoader generator was consumed across ablation runs | Later variants inherited training-order state from earlier runs | Loaders are rebuilt and seeded independently for every variant/seed |
| Undefined AUC could enter winner ranking | A single-class test split could produce an arbitrary winner | Non-finite metrics are explicitly excluded and reported undefined |
| `/health` always returned HTTP 200 | Docker could mark a missing-checkpoint service healthy | It now returns HTTP 503 until a model is ready |
| API preprocessing did not match training centering | Inference distribution differed from training | Each request frame is centered on its protein atoms before inference |
| API did not validate atom range or finite times/coordinates | Invalid scientific input could reach the network | Atomic numbers, times, positions, shape, roles and model atom limit are validated |
| Real-data verifier did not stream-check coordinates | NaN/Inf coordinates could pass schema verification | Every selected coordinate frame is now checked in bounded chunks |
| CTDG segment/PFT boundaries were not constrained | A non-aligned custom config could change patch composition | Training fails fast unless segmented CTDG window/segment boundaries align to patch size |
| Non-finite train/validation loss was not reported at origin | Failure could surface later as a missing checkpoint | Trainer raises a contextual `FloatingPointError` immediately |

## Repeat checks after fixes

1. Ruff static analysis: passed.
2. Python compilation of every source/script/test: passed.
3. Unit/integration suite: **21/21 passed**.
4. Full versus segmented CTDG values: passed for CTDG and CTDG+TVN with irregular timestamps.
5. Rotation/translation invariance and coordinate equivariance: passed for all four variants.
6. Forward/backward with finite gradients: passed for all four variants.
7. End-to-end one-epoch trainer, physics gradients, validation, checkpoint reload and test inference: passed for all four variants.
8. Trained-checkpoint API including force generation: passed.
9. Release ZIP extraction, empty `data/`, recompilation, tests and SHA-256 file manifest: passed.

The full 7.45 GB real dataset remains intentionally absent. Consequently, real benchmark scores are not fabricated; `ctdg-md verify` remains the mandatory post-download gate.
