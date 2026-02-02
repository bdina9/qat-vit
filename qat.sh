#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=""
CKPT_DIR=""
GPUS=4
MASTER_PORT=12345
MODEL_TYPE="ViT-B_16"
IMG_SIZE=384
BATCH_SIZE=8
EPOCHS=10
QAT_LR=1e-3
OUTPUT_DIR="qat_output"
NAME="vit"

usage() {
  cat <<EOF
Usage: bash qat.sh --data-dir <path> --ckpt-dir <path> [options]

Required:
  --data-dir   Dataset root containing train/ and val/
  --ckpt-dir   Directory containing ViT-B_16.npz (teacher)

Options:
  --gpus N
  --master-port PORT
  --model-type NAME
  --img-size N
  --batch-size N
  --epochs N
  --qat-lr LR
  --output-dir PATH
  --name NAME
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --ckpt-dir) CKPT_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --model-type) MODEL_TYPE="$2"; shift 2 ;;
    --img-size) IMG_SIZE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --qat-lr) QAT_LR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 2 ;;
  esac
done

[[ -n "$DATA_DIR" ]] || { echo "[ERROR] --data-dir is required"; usage; exit 2; }
[[ -n "$CKPT_DIR" ]] || { echo "[ERROR] --ckpt-dir is required"; usage; exit 2; }

TEACHER="${CKPT_DIR}/${MODEL_TYPE}.npz"
CALIB="calib-checkpoint/${MODEL_TYPE}_calib.pth"

[[ -f "$TEACHER" ]] || { echo "[ERROR] Missing teacher: $TEACHER"; exit 2; }
[[ -f "$CALIB" ]] || { echo "[ERROR] Missing calibrated ckpt: $CALIB (run calib.sh first)"; exit 2; }

echo "[+] DATA_DIR=$DATA_DIR"
echo "[+] CKPT_DIR=$CKPT_DIR"
echo "[+] Using teacher: $TEACHER"
echo "[+] Using calib:   $CALIB"

torchrun --nproc_per_node "$GPUS" --master_port "$MASTER_PORT" main.py \
  --train \
  --name "$NAME" \
  --pretrained_dir "$CALIB" \
  --data-path "$DATA_DIR" \
  --model_type "$MODEL_TYPE" \
  --quant-mode ft2 \
  --img_size "$IMG_SIZE" \
  --distill \
  --teacher "$TEACHER" \
  --output "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-epochs "$EPOCHS" \
  --qat-lr "$QAT_LR"
