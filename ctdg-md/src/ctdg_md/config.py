from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data"
    hdf5: str = "data/MISATO_1000/raw/MD.hdf5"
    split_dir: str = "data/MISATO_1000/raw"
    train_complexes: int = 80
    val_complexes: int = 10
    test_complexes: int = 10
    frames_per_complex: int = 100
    window_size: int = 10
    stride: int = 10
    dt_ps: float = 80.0
    pocket_cutoff_angstrom: float = 8.0
    max_nodes: int = 256
    remove_hydrogen: bool = True
    energy_key: str = "frames_interaction_energy"
    state_key: str = "frames_rmsd_ligand"


@dataclass
class ModelConfig:
    name: str = "ctdg_tvn"
    hidden_dim: int = 96
    num_layers: int = 3
    message_dim: int = 96
    radial_basis: int = 16
    cutoff_angstrom: float = 5.0
    max_neighbors: int = 32
    time_dim: int = 24
    patch_size: int = 5
    fourier_dropout: float = 0.05
    num_virtual_nodes: int = 2
    vn_kmeans_iterations: int = 10
    vn_tolerance: float = 1e-2
    coordinate_scale: float = 0.1
    dropout: float = 0.05


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    segment_length: int = 5
    grad_clip: float = 1.0
    patience: int = 7
    num_workers: int = 0
    energy_loss_weight: float = 1.0
    state_loss_weight: float = 0.25
    force_balance_weight: float = 1e-3
    torque_balance_weight: float = 1e-3
    temporal_smoothness_weight: float = 1e-4
    physics_every: int = 4
    checkpoint_dir: str = "outputs/checkpoints"
    result_dir: str = "outputs"


@dataclass
class AblationConfig:
    variants: list[str] = field(
        default_factory=lambda: [
            "constant",
            "schnet",
            "painn",
            "egnn",
            "egnn_tvn",
            "ctdg",
            "ctdg_tvn",
        ]
    )
    seeds: list[int] = field(default_factory=lambda: [11, 22, 33, 44, 55])


@dataclass
class PublicationConfig:
    candidate: str = "ctdg_tvn"
    required_baselines: list[str] = field(
        default_factory=lambda: ["constant", "schnet", "painn", "egnn", "egnn_tvn", "ctdg"]
    )
    external_required: list[str] = field(
        default_factory=lambda: ["mace", "nequip", "neuralmd", "gromacs_mmgbsa"]
    )
    external_result_dir: str = "outputs/external"
    min_seeds: int = 5
    bootstrap_replicates: int = 10_000
    confidence: float = 0.95
    familywise_alpha: float = 0.05
    require_positive_r2: bool = True
    force_label_key: str = "frames_forces"


@dataclass
class ProjectConfig:
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    publication: PublicationConfig = field(default_factory=PublicationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def project_config_from_dict(data: dict[str, Any]) -> ProjectConfig:
    known = {"seed", "device", "data", "model", "training", "ablation", "publication"}
    extra = set(data) - known
    if extra:
        raise ValueError(f"Unknown top-level config keys: {sorted(extra)}")
    return ProjectConfig(
        seed=int(data.get("seed", 42)),
        device=str(data.get("device", "auto")),
        data=DataConfig(**data.get("data", {})),
        model=ModelConfig(**data.get("model", {})),
        training=TrainingConfig(**data.get("training", {})),
        ablation=AblationConfig(**data.get("ablation", {})),
        publication=PublicationConfig(**data.get("publication", {})),
    )


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("Configuration root must be a mapping")
    return project_config_from_dict(raw)


def save_config(config: ProjectConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
