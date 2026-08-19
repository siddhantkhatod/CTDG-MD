# Implementation audit

This audit maps every requested concept to executable code and a check. “Verified” means covered by the included test suite; it is not a claim that any nontrivial research system can be mathematically or operationally perfect on every future machine.

| Requested concept | Implementation | Verification |
|---|---|---|
| CTDG construction | `graph.radius_graph` rebuilds directed atom interactions at every timestamp. `CTDGModel` processes the ordered timestamp stream. | `test_graph.py`, all-model runtime tests |
| Nodes = atoms | MISATO atomic numbers, residues, protein/ligand role and invariant speed enter `NodeEncoder`. | data-loader and runtime tests |
| Dynamic interactions | Distance edges are recomputed per frame; cutoff and neighbor cap are configurable. | graph determinism/bounds test |
| E(n) message passing | Invariant messages depend on scalar features and squared distance; coordinates update by scalar-weighted relative vectors. | rotation/translation energy and coordinate tests for all variants |
| Continuous time | Learnable sine/cosine encoding of elapsed time plus exponential memory decay and GRU update. Irregular monotone timestamps are supported. | `test_temporal.py` |
| Non-stationary patch Fourier transform | Per-patch rFFT; patch power and center time produce frequency gates; learned complex filter; irFFT residual. | shape, finite-output and backward-gradient test |
| Segmented gradients | Trainer passes CTDG memory across segments and detaches it at each boundary (TBPTT). | training code and all-model backward smoke test |
| Energy prediction | Protein/global/ligand pooled embeddings feed an interaction-energy regression head; labels are train-normalized and report-time denormalized. | loader, model and metric tests |
| Force prediction | Inference returns `-dE/dx` in dataset-native energy/angstrom, differentiating through training-matched protein centering. | API force-path and net-force tests |
| Conformational state | Auxiliary mobile/stable ligand-pose head, labelled by train-only median ligand RMSD. | metric tests |
| Physics-informed loss | Net force and net torque from `-dE/dx`; small optional temporal curvature term. | exact pair-distance conservation test |
| Classical internal baselines | Trainable constant, continuous-filter SchNet and scalar/vector PaiNN use identical pockets/splits. | parametrized invariance + forward/backward tests |
| EGNN ablation | `egnn` and `egnn_tvn`. | parametrized equivariance + forward/backward tests |
| CTDG ablation | `ctdg` and `ctdg_tvn`. | parametrized equivariance + forward/backward tests |
| Publication statistics | Complex-cluster bootstrap, paired sign-flip tests and Holm correction. | controlled paired-improvement test |
| External comparators | Strict extxyz export, frame-keyed energy import, NeuralMD/BioMD trajectory evaluator and MM/GBSA workflow. | alignment and trajectory tests; missing evidence blocks claims |
| AUC / accuracy / AMI | AUC and accuracy on physical state; AMI between K-means embeddings and physical state. | controlled perfect-separation metric test |
| AMI matrix | Pairwise adjusted mutual information between variant embedding-cluster assignments. | label-permutation-invariance test |
| Real 10,000 frames | 80/10/10 complexes × 100 real MISATO frames. No synthetic train/eval fallback exists. | `verify_misato_10k` hard contract |
| Deployment | FastAPI checkpoint loader, strict input validation, health, predict, Docker. No checkpoint silently returns 503. | API test |

## TVN check against the attached paper

The paper's Algorithm 1/2 and Appendix G.2 are implemented as follows:

1. **Assignment at every dynamic step:** adjacency-row k-means runs for every MD frame.
2. **Topology-aware rows:** the exact dense adjacency row is each atom's clustering point.
3. **K-means defaults:** k=2, 10 iterations and tolerance 1e-2.
4. **Child edges:** every physical atom has bidirectional edges to its assigned VN.
5. **Inter-VN edges:** all VNs form a directed clique.
6. **Aggregation:** degree-weighted child means initialize VN features.
7. **Geometric extension:** VN positions are degree-weighted child centroids. This is necessary for EGNN message passing and is E(3)-equivariant.
8. **No persistent cluster-ID assumption:** VNs are reconstructed each frame, so cluster-label permutations do not corrupt memory identity.

The molecular extension clusters frame contact topology, whereas the paper evaluates event streams such as TGB. That domain adaptation is explicit rather than presented as the paper's original experiment.

## EGNN formula check

For edge j→i:

- `m_ij = phi_e(h_j, h_i, ||x_i-x_j||², a_ij, tau(dt))`
- `h_i' = LN(h_i + phi_h(h_i, mean_j m_ij))`
- `x_i' = x_i + mean_j ((x_i-x_j)/(1+||x_i-x_j||)) phi_x(m_ij)`

Only invariant scalars multiply relative vectors; therefore translation, orthogonal transformation (rotation/reflection), and node-permutation equivariance are preserved away from nondifferentiable graph-boundary changes.

## Intentional corrections to the supplied flowchart

- “Energy conservation” is not applied as constant interaction energy: MISATO is NPT and the label varies physically.
- AUC/accuracy are not misapplied to regression; they evaluate a defined physical-state classification task.
- “Forces” are energy gradients without supervised force labels and are reported as such.
- MISATO does not contain a single 10,000-stored-frame complex. The project uses exactly 10,000 real frames across leakage-free complexes and never claims otherwise.
