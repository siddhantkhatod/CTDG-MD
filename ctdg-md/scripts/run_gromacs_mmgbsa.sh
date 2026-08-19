#!/usr/bin/env bash
set -euo pipefail
# Recompute a classical MM/GBSA comparator when the separately distributed MISATO
# topology/restart data are available. MD.hdf5 alone is insufficient.
if [[ $# -lt 5 ]]; then
  echo "Usage: $0 topology.tpr trajectory.xtc index.ndx output_prefix pdb_id" >&2
  exit 2
fi
TPR=$1
XTC=$2
INDEX=$3
PREFIX=$4
PDB_ID=$5
command -v gmx_MMPBSA >/dev/null || {
  echo "gmx_MMPBSA is required; install its official release" >&2
  exit 2
}
[[ -f "$TPR" && -f "$XTC" && -f "$INDEX" ]] || {
  echo "TPR, XTC or index file is missing" >&2
  exit 2
}
cat > "${PREFIX}_mmpbsa.in" <<'EOF'
&general
  interval=1,
  verbose=1,
/
&gb
  igb=5,
  saltcon=0.150,
/
EOF
# Index groups 1 and 2 must be audited Protein and Ligand groups. The command
# fails rather than silently guessing molecular groups.
gmx_MMPBSA -O -i "${PREFIX}_mmpbsa.in" -cs "$TPR" -ct "$XTC" -ci "$INDEX" \
  -cg 1 2 -o "${PREFIX}_FINAL_RESULTS_MMPBSA.dat" -eo "${PREFIX}_energy.csv"
echo "Completed $PDB_ID -> ${PREFIX}_energy.csv"
