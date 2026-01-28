# project/scripts/evaluate.sh
#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/evaluate.sh ./qat_search/best_qat.pth ./qat_search/best_converted.pth

QAT_CKPT="${1:-./qat_search/best_qat.pth}"
QUANT_CKPT="${2:-./qat_search/best_converted.pth}"

python -m src.evaluation.comparator \
  --qat-ckpt "${QAT_CKPT}" \
  --quant-ckpt "${QUANT_CKPT}"
