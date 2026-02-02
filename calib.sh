#!/usr/bin/env bash
# (clue-dev) dinab@dgx3:~/qat-vit$ ./calib.sh --data-dir /mnt/manatee/irad/clue/imagenet1k/ --ckpt-dir /mnt/manatee/irad/clue/checkpoints/
# [+] DATA_DIR=/mnt/manatee/irad/clue/imagenet1k/
# [+] CKPT_DIR=/mnt/manatee/irad/clue/checkpoints/
# [+] Using pretrained: /mnt/manatee/irad/clue/checkpoints//ViT-B_16.npz
# Traceback (most recent call last):
#   File "/home/dinab/qat-vit/main.py", line 29, in <module>
#     from apex import amp
#   File "/home/dinab/envs/clue-dev/lib/python3.10/site-packages/apex/__init__.py", line 13, in <module>
#     from pyramid.session import UnencryptedCookieSessionFactoryConfig
# ImportError: cannot import name 'UnencryptedCookieSessionFactoryConfig' from 'pyramid.session' (unknown location)
# E0202 14:00:27.983000 3957034 torch/distributed/elastic/multiprocessing/api.py:984] failed (exitcode: 1) local_rank: 0 (pid: 3957105) of binary: /home/dinab/envs/clue-dev/bin/python3.10
# Traceback (most recent call last):
#   File "/home/dinab/envs/clue-dev/bin/torchrun", line 8, in <module>
#     sys.exit(main())
#   File "/home/dinab/envs/clue-dev/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
#     return f(*args, **kwargs)
#   File "/home/dinab/envs/clue-dev/lib/python3.10/site-packages/torch/distributed/run.py", line 991, in main
#     run(args)
#   File "/home/dinab/envs/clue-dev/lib/python3.10/site-packages/torch/distributed/run.py", line 982, in run
#     elastic_launch(
#   File "/home/dinab/envs/clue-dev/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 170, in __call__
#     return launch_agent(self._config, self._entrypoint, list(args))
#   File "/home/dinab/envs/clue-dev/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 317, in launch_agent
#     raise ChildFailedError(
# torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
# ============================================================
# main.py FAILED
# ------------------------------------------------------------
# Failures:
#   <NO_OTHER_FAILURES>
# ------------------------------------------------------------
# Root Cause (first observed failure):
# [0]:
#   time      : 2026-02-02_14:00:27
#   host      : dgx3
#   rank      : 0 (local_rank: 0)
#   exitcode  : 1 (pid: 3957105)
#   error_file: <N/A>
#   traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
# ============================================================
# (clue-dev) dinab@dgx3:~/qat-vit$ 
# set -euo pipefail

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
