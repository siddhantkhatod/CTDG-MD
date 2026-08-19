#!/usr/bin/env bash
#SBATCH --job-name=ctdg-pub
#SBATCH --array=0-34
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%A_%a.log
set -euo pipefail

CONFIG=${CONFIG:-configs/default.yaml}
VARIANTS=(constant schnet painn egnn egnn_tvn ctdg ctdg_tvn)
SEEDS=(11 22 33 44 55)
variant_index=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
variant=${VARIANTS[$variant_index]}
seed=${SEEDS[$seed_index]}
mkdir -p outputs/slurm
srun ctdg-md train --config "$CONFIG" --variant "$variant" --seed "$seed"

# After every array task succeeds, run once on a CPU/login job:
# ctdg-md publication --config "$CONFIG" --aggregate-only
