from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedComparison:
    candidate: str
    baseline: str
    complexes: int
    candidate_mae: float
    baseline_mae: float
    mae_difference: float
    ci_low: float
    ci_high: float
    probability_improvement: float
    permutation_p: float
    holm_adjusted_p: float | None = None

    def as_dict(self) -> dict[str, float | int | str | None]:
        return self.__dict__.copy()


def _validate_alignment(
    truth: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
) -> None:
    size = truth.size
    if any(array.reshape(-1).size != size for array in (candidate, baseline, groups)):
        raise ValueError("Paired arrays are not aligned")
    if not all(np.isfinite(array).all() for array in (truth, candidate, baseline)):
        raise ValueError("Paired comparison contains non-finite values")


def complex_mae(truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(truth).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if truth.size != prediction.size or truth.size != groups.size:
        raise ValueError("truth, prediction and groups must have equal length")
    names = np.unique(groups)
    errors = np.asarray(
        [np.mean(np.abs(truth[groups == name] - prediction[groups == name])) for name in names]
    )
    return names, errors


def paired_complex_bootstrap(
    truth: np.ndarray,
    candidate_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    groups: np.ndarray,
    candidate_name: str,
    baseline_name: str,
    replicates: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Cluster bootstrap and paired sign-flip test at independent-complex level."""
    truth = np.asarray(truth).reshape(-1)
    candidate_prediction = np.asarray(candidate_prediction).reshape(-1)
    baseline_prediction = np.asarray(baseline_prediction).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    _validate_alignment(truth, candidate_prediction, baseline_prediction, groups)
    candidate_groups, candidate_error = complex_mae(truth, candidate_prediction, groups)
    baseline_groups, baseline_error = complex_mae(truth, baseline_prediction, groups)
    if not np.array_equal(candidate_groups, baseline_groups):
        raise ValueError("Candidate and baseline complex identities differ")
    differences = candidate_error - baseline_error
    count = differences.size
    if count < 2:
        raise ValueError("At least two independent complexes are required")
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, count, size=(replicates, count))
    bootstrap = differences[sampled].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])

    # Exact sign flips are exponential; Monte Carlo gives a reproducible paired test.
    signs = generator.choice(np.array([-1.0, 1.0]), size=(replicates, count))
    null_statistic = (signs * differences).mean(axis=1)
    observed = abs(float(differences.mean()))
    permutation_p = float((np.count_nonzero(np.abs(null_statistic) >= observed) + 1) / (replicates + 1))
    return PairedComparison(
        candidate=candidate_name,
        baseline=baseline_name,
        complexes=count,
        candidate_mae=float(candidate_error.mean()),
        baseline_mae=float(baseline_error.mean()),
        mae_difference=float(differences.mean()),
        ci_low=float(low),
        ci_high=float(high),
        probability_improvement=float(np.mean(bootstrap < 0.0)),
        permutation_p=permutation_p,
    )


def holm_bonferroni(comparisons: list[PairedComparison]) -> list[PairedComparison]:
    """Return Holm-adjusted p-values while preserving input order."""
    if not comparisons:
        return []
    raw = np.asarray([comparison.permutation_p for comparison in comparisons])
    order = np.argsort(raw)
    adjusted = np.empty_like(raw)
    running = 0.0
    total = len(raw)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * raw[index])
        running = max(running, value)
        adjusted[index] = running
    return [
        PairedComparison(**{**comparison.__dict__, "holm_adjusted_p": float(adjusted[index])})
        for index, comparison in enumerate(comparisons)
    ]


def mean_seed_prediction(paths: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load aligned NPZ files and average model predictions over random seeds."""
    if not paths:
        raise ValueError("At least one prediction file is required")
    predictions: list[np.ndarray] = []
    reference_truth: np.ndarray | None = None
    reference_groups: np.ndarray | None = None
    reference_frames: np.ndarray | None = None
    for path in paths:
        payload = np.load(path)
        truth = payload["energy_true"].reshape(-1)
        prediction = payload["energy_pred"].reshape(-1)
        groups = payload["pdb_id"].astype(str).reshape(-1)
        frames = payload["frame_index"].reshape(-1)
        if reference_truth is None:
            reference_truth, reference_groups, reference_frames = truth, groups, frames
        elif not (
            np.array_equal(reference_truth, truth)
            and np.array_equal(reference_groups, groups)
            and np.array_equal(reference_frames, frames)
        ):
            raise ValueError(f"Prediction file {path} is not aligned to earlier seeds")
        predictions.append(prediction)
    assert reference_truth is not None and reference_groups is not None
    return reference_truth, np.mean(predictions, axis=0), reference_groups
