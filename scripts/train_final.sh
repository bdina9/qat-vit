# project/scripts/train_final.sh
#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/train_final.sh ./qat_search/best_params.yaml ./qat_search 4
#   args: <config_yaml> <output_dir> <nproc_per_node>

CONFIG="${1:-./qat_search/best_params.yaml}"
OUTDIR="${2:-./qat_search}"
NPROC="${3:-1}"

torchrun --standalone --nproc_per_node="${NPROC}" \
  -m src.training.qat_trainer \
  --config "${CONFIG}" \
  --output-dir "${OUTDIR}"
