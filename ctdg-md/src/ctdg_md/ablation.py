from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .config import ProjectConfig
from .metrics import pairwise_ami_matrix
from .train import train_one


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_ablation(config: ProjectConfig) -> dict[str, object]:
    """Train EGNN/CTDG x without/with temporal virtual nodes and compare metrics."""
    output_dir = Path(config.training.result_dir) / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, object]] = []
    first_seed_assignments: dict[str, np.ndarray] = {}
    for variant in config.ablation.variants:
        for index, seed in enumerate(config.ablation.seeds):
            # Rebuild loaders per run so this run's seed controls data ordering.
            result, evaluation = train_one(config, variant=variant, seed=seed)
            row: dict[str, object] = {
                "variant": variant,
                "seed": seed,
                "best_epoch": result.best_epoch,
                "val_loss": result.best_validation_loss,
                "seconds": result.seconds,
                **result.test_metrics,
            }
            raw_rows.append(row)
            if index == 0:
                first_seed_assignments[variant] = evaluation.assignments
    _write_csv(output_dir / "ablation_runs.csv", raw_rows)

    metric_names = ["energy_mae", "energy_rmse", "energy_r2", "auc", "accuracy", "ami", "seconds"]
    summary_rows: list[dict[str, object]] = []
    for variant in config.ablation.variants:
        rows = [row for row in raw_rows if row["variant"] == variant]
        summary: dict[str, object] = {"variant": variant, "runs": len(rows)}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            summary[f"{metric}_mean"] = float(np.nanmean(values))
            summary[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    _write_csv(output_dir / "ablation_summary.csv", summary_rows)

    names, matrix = pairwise_ami_matrix(first_seed_assignments)
    matrix_rows: list[dict[str, object]] = []
    for index, name in enumerate(names):
        matrix_rows.append({"variant": name, **{other: float(matrix[index, j]) for j, other in enumerate(names)}})
    _write_csv(output_dir / "ami_matrix.csv", matrix_rows)

    winners: dict[str, object] = {}
    for metric in ["auc", "accuracy", "ami"]:
        valid = [row for row in summary_rows if math.isfinite(float(row[f"{metric}_mean"]))]
        if valid:
            winner = max(valid, key=lambda row: float(row[f"{metric}_mean"]))
            winners[metric] = {"variant": winner["variant"], "value": winner[f"{metric}_mean"]}
        else:
            winners[metric] = {"variant": None, "value": None, "reason": "metric undefined"}
    # A transparent aggregate: average rank on the three requested maximization metrics.
    rank_totals = {str(row["variant"]): 0.0 for row in summary_rows}
    for metric in ["auc", "accuracy", "ami"]:
        ordered = sorted(
            summary_rows,
            key=lambda row: (
                math.isfinite(float(row[f"{metric}_mean"])), float(row[f"{metric}_mean"])
            ),
            reverse=True,
        )
        for rank, row in enumerate(ordered, start=1):
            rank_totals[str(row["variant"])] += rank
    aggregate_best = min(rank_totals, key=rank_totals.get)
    report: dict[str, object] = {
        "winners": winners,
        "aggregate_best_by_mean_rank": aggregate_best,
        "rank_totals": rank_totals,
        "ami_matrix_seed": config.ablation.seeds[0],
        "runs": raw_rows,
    }
    (output_dir / "best_models.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
