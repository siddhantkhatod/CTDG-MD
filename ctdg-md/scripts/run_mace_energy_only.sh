#!/usr/bin/env bash
set -euo pipefail
# Requires the official MACE package and a GPU environment.
command -v mace_run_train >/dev/null || { echo 'mace_run_train is not installed' >&2; exit 2; }
CONFIG=${1:-configs/mace_energy_only.yaml}
mace_run_train --config "$CONFIG"
