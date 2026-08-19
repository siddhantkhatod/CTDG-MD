from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .ablation import run_ablation
from .config import ProjectConfig
from .data import selected_ids, verify_misato_10k
from .statistics import holm_bonferroni, mean_seed_prediction, paired_complex_bootstrap
from .utils import atomic_json_dump


def _prediction_paths(result_root: Path, variant: str, seeds: list[int]) -> list[str]:
    paths = [result_root / f"{variant}_seed{seed}" / "test_predictions.npz" for seed in seeds]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prediction files for {variant}: {missing}")
    return [str(path) for path in paths]


def _force_labels_available(config: ProjectConfig) -> bool:
    ids = selected_ids(config.data)
    first = ids["train"][0]
    with h5py.File(config.data.hdf5, "r") as handle:
        return config.publication.force_label_key in handle[first]


def _external_energy_paths(directory: Path, baseline: str) -> list[str]:
    folder = directory / baseline
    return [str(path) for path in sorted(folder.glob("seed*/test_predictions.npz"))]


def _load_existing_runs(config: ProjectConfig) -> dict[str, Any]:
    root = Path(config.training.result_dir)
    rows = []
    for variant in config.ablation.variants:
        for seed in config.ablation.seeds:
            path = root / f"{variant}_seed{seed}" / "result.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed run: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "variant": payload["variant"],
                    "seed": payload["seed"],
                    "best_epoch": payload["best_epoch"],
                    "val_loss": payload["best_validation_loss"],
                    "seconds": payload["seconds"],
                    **payload["test_metrics"],
                }
            )
    return {"runs": rows, "source": "existing completed runs"}


def run_publication_study(config: ProjectConfig, train: bool = True) -> dict[str, Any]:
    """Run/aggregate the full same-target study and emit machine-enforced claim gates.

    This function never turns missing force, rollout or external-baseline evidence
    into a positive claim. Every unsupported scope is reported explicitly.
    """
    publication = config.publication
    variants = config.ablation.variants
    seeds = config.ablation.seeds
    if publication.candidate not in variants:
        raise ValueError("Publication candidate is absent from ablation variants")
    missing_baselines = sorted(set(publication.required_baselines) - set(variants))
    if missing_baselines:
        raise ValueError(f"Required same-target baselines are absent: {missing_baselines}")
    if len(set(seeds)) < publication.min_seeds:
        raise ValueError(
            f"Publication study requires {publication.min_seeds} unique seeds, received {seeds}"
        )

    verification = verify_misato_10k(config.data)
    ablation = run_ablation(config) if train else _load_existing_runs(config)
    result_root = Path(config.training.result_dir)
    candidate_paths = _prediction_paths(result_root, publication.candidate, seeds)
    truth, candidate_prediction, groups = mean_seed_prediction(candidate_paths)

    comparisons = []
    for baseline in publication.required_baselines:
        baseline_paths = _prediction_paths(result_root, baseline, seeds)
        baseline_truth, baseline_prediction, baseline_groups = mean_seed_prediction(baseline_paths)
        if not np.array_equal(truth, baseline_truth) or not np.array_equal(groups, baseline_groups):
            raise ValueError(f"Baseline {baseline} is not aligned with the candidate test set")
        comparisons.append(
            paired_complex_bootstrap(
                truth,
                candidate_prediction,
                baseline_prediction,
                groups,
                publication.candidate,
                baseline,
                publication.bootstrap_replicates,
                publication.confidence,
                seed=config.seed,
            )
        )
    comparisons = holm_bonferroni(comparisons)

    candidate_runs = [
        row for row in ablation["runs"] if row["variant"] == publication.candidate
    ]
    candidate_r2 = float(np.mean([float(row["energy_r2"]) for row in candidate_runs]))
    internal_significance = all(
        comparison.ci_high < 0.0
        and comparison.holm_adjusted_p is not None
        and comparison.holm_adjusted_p < publication.familywise_alpha
        for comparison in comparisons
    )
    same_target_pass = internal_significance and (
        candidate_r2 > 0.0 if publication.require_positive_r2 else True
    )

    external_root = Path(publication.external_result_dir)
    external_evidence: dict[str, Any] = {}
    external_energy_comparisons = []
    for baseline in publication.external_required:
        paths = _external_energy_paths(external_root, baseline)
        if paths:
            ext_truth, ext_prediction, ext_groups = mean_seed_prediction(paths)
            if np.array_equal(truth, ext_truth) and np.array_equal(groups, ext_groups):
                comparison = paired_complex_bootstrap(
                    truth,
                    candidate_prediction,
                    ext_prediction,
                    groups,
                    publication.candidate,
                    baseline,
                    publication.bootstrap_replicates,
                    publication.confidence,
                    seed=config.seed,
                )
                external_energy_comparisons.append(comparison)
                external_evidence[baseline] = {"status": "energy_predictions_loaded"}
            else:
                external_evidence[baseline] = {"status": "rejected", "reason": "test alignment mismatch"}
        else:
            trajectory_report = external_root / baseline / "trajectory_metrics.json"
            external_evidence[baseline] = (
                {"status": "trajectory_metrics_loaded", "path": str(trajectory_report)}
                if trajectory_report.is_file()
                else {"status": "missing"}
            )
    external_energy_comparisons = holm_bonferroni(external_energy_comparisons)
    expected_external_energy = [
        name for name in publication.external_required if name not in {"neuralmd", "biomd"}
    ]
    external_energy_pass = len(external_energy_comparisons) == len(expected_external_energy) and all(
        comparison.ci_high < 0.0
        and comparison.holm_adjusted_p is not None
        and comparison.holm_adjusted_p < publication.familywise_alpha
        for comparison in external_energy_comparisons
    )

    force_available = _force_labels_available(config)
    neuralmd_report = external_root / "neuralmd" / "trajectory_metrics.json"
    trajectory_evidence = neuralmd_report.is_file()
    report: dict[str, Any] = {
        "protocol": {
            "data_verification": verification.__dict__,
            "epochs": config.training.epochs,
            "patience": config.training.patience,
            "seeds": seeds,
            "candidate": publication.candidate,
            "same_target_baselines": publication.required_baselines,
            "external_required": publication.external_required,
            "confidence": publication.confidence,
            "bootstrap_replicates": publication.bootstrap_replicates,
            "familywise_alpha": publication.familywise_alpha,
        },
        "energy": {
            "candidate_mean_r2": candidate_r2,
            "same_target_comparisons": [comparison.as_dict() for comparison in comparisons],
            "external_comparisons": [
                comparison.as_dict() for comparison in external_energy_comparisons
            ],
        },
        "evidence": {
            "external": external_evidence,
            "force_labels_available": force_available,
            "trajectory_report_available": trajectory_evidence,
        },
        "claim_gates": {
            "better_than_included_same_target_baselines": same_target_pass,
            "better_than_current_external_energy_models": same_target_pass and external_energy_pass,
            "better_force_prediction": False,
            "better_trajectory_rollout": False,
        },
        "claim_reasons": {
            "force": (
                "Force comparison must be run before a claim"
                if force_available
                else "MISATO does not provide supervised per-atom force labels"
            ),
            "trajectory": (
                "External trajectory metrics exist but require paired claim evaluation"
                if trajectory_evidence
                else "No aligned NeuralMD/BioMD rollout result was imported"
            ),
            "broad_current_ml": (
                "PASS" if same_target_pass and external_energy_pass else "External same-test evidence incomplete or not significant"
            ),
        },
        "ablation_summary": ablation,
    }
    output = result_root / "publication" / "publication_report.json"
    atomic_json_dump(report, output)
    return report
