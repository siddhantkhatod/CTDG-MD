# CTDG-MD implementation flow

The following is the corrected Mermaid source corresponding to the uploaded flowchart. The red warning in the screenshot is a Mermaid syntax/rendering problem, not a model-runtime error.

```mermaid
flowchart LR
    A[MD Trajectory Data] --> B[CTDG Construction<br/>Nodes: Atoms<br/>Edges: Dynamic Interactions]
    B --> C[Equivariant Message Passing<br/>Update coordinates and features<br/>E(n) symmetry]
    C --> D[Non-Stationary Temporal Encoding<br/>Patch Fourier Transform<br/>Time-varying frequency capture]
    D --> E[Truncated BPTT Training<br/>Segmented gradient calculation<br/>Memory-efficient long-range learning]
    E --> F[Physics-Informed Loss<br/>Force and torque balance<br/>Equivariance constraints]
    F --> G[MD Property Prediction<br/>Forces, interaction energies,<br/>conformational states]
```

## Source-code mapping

- **A:** `data.py`
- **B:** `graph.py`, frame graph construction in `model.py`
- **C:** `egnn.py`
- **D:** `temporal.py`
- **E:** segmentation in `train.py` and boundary context in `utils.py`
- **F:** `losses.py`
- **G:** model heads in `model.py` and force-returning inference in `api.py`

For MISATO's NPT interaction-energy target, “energy conservation” is deliberately not interpreted as constant energy over time. That would be physically incorrect. Translation/rotation symmetry is enforced through the equivariant architecture and net-force/net-torque residuals; force prediction is obtained from the learned energy gradient.
