#!/usr/bin/env bash
set -euo pipefail
# Requires the official NequIP package and a GPU environment.
command -v nequip-train >/dev/null || { echo 'nequip-train is not installed' >&2; exit 2; }
CONFIG=${1:-configs/nequip_energy_only.yaml}
nequip-train "$CONFIG"
