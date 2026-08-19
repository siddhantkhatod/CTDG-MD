from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricResult:
    energy_mae: float
    energy_rmse: float
    energy_r2: float
    auc: float
    accuracy: float
    ami: float

    def as_dict(self) -> dict[str, float]:
        return {
            "energy_mae": self.energy_mae,
            "energy_rmse": self.energy_rmse,
            "energy_r2": self.energy_r2,
            "auc": self.auc,
            "accuracy": self.accuracy,
            "ami": self.ami,
        }


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan")


def cluster_assignments(embeddings: np.ndarray, clusters: int = 2, seed: int = 0) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [samples, features]")
    if embeddings.shape[0] < clusters:
        raise ValueError("fewer embeddings than clusters")
    if np.unique(embeddings, axis=0).shape[0] < clusters:
        return np.zeros(embeddings.shape[0], dtype=np.int64)
    return KMeans(n_clusters=clusters, n_init=20, random_state=seed).fit_predict(embeddings)


def compute_metrics(
    energy_true: np.ndarray,
    energy_pred: np.ndarray,
    state_true: np.ndarray,
    state_probability: np.ndarray,
    embeddings: np.ndarray,
    seed: int = 0,
) -> tuple[MetricResult, np.ndarray]:
    arrays = [energy_true, energy_pred, state_true, state_probability]
    arrays = [np.asarray(value).reshape(-1) for value in arrays]
    if not all(np.isfinite(value).all() for value in arrays) or not np.isfinite(embeddings).all():
        raise ValueError("Cannot evaluate non-finite predictions")
    predicted_state = (arrays[3] >= 0.5).astype(np.int64)
    assignments = cluster_assignments(np.asarray(embeddings), clusters=2, seed=seed)
    metrics = MetricResult(
        energy_mae=float(mean_absolute_error(arrays[0], arrays[1])),
        energy_rmse=float(mean_squared_error(arrays[0], arrays[1]) ** 0.5),
        energy_r2=float(r2_score(arrays[0], arrays[1])),
        auc=safe_auc(arrays[2], arrays[3]),
        accuracy=float(accuracy_score(arrays[2], predicted_state)),
        ami=float(adjusted_mutual_info_score(arrays[2], assignments)),
    )
    return metrics, assignments


def pairwise_ami_matrix(assignments: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = list(assignments)
    matrix = np.eye(len(names), dtype=np.float64)
    for left, left_name in enumerate(names):
        for right in range(left + 1, len(names)):
            right_name = names[right]
            score = adjusted_mutual_info_score(assignments[left_name], assignments[right_name])
            matrix[left, right] = matrix[right, left] = score
    return names, matrix
