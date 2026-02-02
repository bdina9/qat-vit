#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=""
CKPT_DIR=""
MASTER_PORT=12345
MODEL_TYPE="ViT-B_16"
IMG_SIZE=384
NUM_CALIB_BATCH=10
CALIB_BATCHSZ=8
CALIBRATOR="percentile"
PERCENTILE=99.99
OUT_DIR="calib-checkpoint"
NAME="vit"

usage() {
  cat <<EOF
Usage: bash calib.sh --data-dir <path> --ckpt-dir <path> [options]

Required:
  --data-dir   Dataset root containing train/ and val/
  --ckpt-dir   Directory containing ViT-B_16.npz

Options:
  --master-port PORT
  --model-type NAME
  --img-size N
  --num-calib-batch N
  --calib-batchsz N
  --calibrator NAME
  --percentile P
  --out-dir PATH
  --name NAME
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --ckpt-dir) CKPT_DIR="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --model-type) MODEL_TYPE="$2"; shift 2 ;;
    --img-size) IMG_SIZE="$2"; shift 2 ;;
    --num-calib-batch) NUM_CALIB_BATCH="$2"; shift 2 ;;
    --calib-batchsz) CALIB_BATCHSZ="$2"; shift 2 ;;
    --calibrator) CALIBRATOR="$2"; shift 2 ;;
    --percentile) PERCENTILE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 2 ;;
  esac
done

[[ -n "$DATA_DIR" ]] || { echo "[ERROR] --data-dir is required"; usage; exit 2; }
[[ -n "$CKPT_DIR" ]] || { echo "[ERROR] --ckpt-dir is required"; usage; exit 2; }

PRETRAINED="${CKPT_DIR}/${MODEL_TYPE}.npz"
[[ -f "$PRETRAINED" ]] || { echo "[ERROR] Missing pretrained: $PRETRAINED"; exit 2; }

mkdir -p "$OUT_DIR"

echo "[+] DATA_DIR=$DATA_DIR"
echo "[+] CKPT_DIR=$CKPT_DIR"
echo "[+] Using pretrained: $PRETRAINED"

torchrun --nproc_per_node 1 --master_port "$MASTER_PORT" main.py \
  --calib \
  --name "$NAME" \
  --pretrained_dir "$PRETRAINED" \
  --data-path "$DATA_DIR" \
  --model_type "$MODEL_TYPE" \
  --img_size "$IMG_SIZE" \
  --num-calib-batch "$NUM_CALIB_BATCH" \
  --calib-batchsz "$CALIB_BATCHSZ" \
  --quant-mode ft2 \
  --calibrator "$CALIBRATOR" \
  --percentile "$PERCENTILE" \
  --calib-output-path "$OUT_DIR"
