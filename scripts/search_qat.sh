# project/scripts/search_qat.sh
#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/search_qat.sh --trials 30 --epochs 10 --output-dir ./qat_search
# Defaults match script args if omitted.

TRIALS="${1:-30}"
EPOCHS="${2:-10}"
OUTDIR="${3:-./qat_search}"

python -m src.training.optuna_search \
  --trials "${TRIALS}" \
  --epochs "${EPOCHS}" \
  --output-dir "${OUTDIR}"
