"""CTDG-MD: temporal equivariant graph models for protein-ligand trajectories."""

__version__ = "2.0.0"

from .config import ProjectConfig, load_config
from .model import CTDGModel, EGNNBaseline, build_model

__all__ = ["CTDGModel", "EGNNBaseline", "ProjectConfig", "build_model", "load_config"]
