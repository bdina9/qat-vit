# ----------------------------------------------------------------------
# 1️⃣  Configuration (tweak any of these values as needed)
# ----------------------------------------------------------------------
# Distributed launch
NUM_NODES=1
GPUS_PER_NODE=4
MASTER_PORT=12355  # ✅ Changed to avoid common port conflicts
MASTER_ADDR="localhost"  # ✅ More reliable than 127.0.0.1 for NCCL

# Data / checkpoint locations
DATA_PATH="/mnt/manatee/irad/clue/imagenet1k/imagenet1k"
PRETRAINED_DIR="calib-checkpoint/ViT-B_16_calib.pth"
TEACHER_PATH="/mnt/manatee/irad/clue/checkpoints/ViT-B_16.npz"

# Model / training hyper‑parameters
MODEL_TYPE="ViT-B_16"
QUANT_MODE="ft2"
IMG_SIZE=384
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16
GRAD_ACC_STEPS=16
NUM_EPOCHS=10
QAT_LR=1e-3
MAX_GRAD_NORM=1.0
USE_CHECKPOINT=true
FP16=true

# 🎉 Warm‑up configuration
WARMUP_STEPS=500

DISTILL_SCALE=1.0

# Misc
BASE_NAME="vit_qat_384_checkpointed"
OUTPUT_DIR="qat_output"
DATASET="imagenet"  # ✅ Explicitly set dataset

# ----------------------------------------------------------------------
# 2️⃣  Build a timestamped experiment tag
# ----------------------------------------------------------------------
DATE_TAG=$(date +%Y%m%d_%H%M%S)
EXP_TAG="${BASE_NAME}_${DATE_TAG}"

# ----------------------------------------------------------------------
# 3️⃣  Export environment variables required by torchrun + NCCL
# ----------------------------------------------------------------------
export MASTER_ADDR
export MASTER_PORT
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ----------------------------------------------------------------------
# 4️⃣  Create output directory
# ----------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"

# ----------------------------------------------------------------------
# 5️⃣  Build the argument list (as a string for reliable subshell passing)
# ----------------------------------------------------------------------
# ✅ Using a string avoids array expansion issues in screen/tmux/nohup
COMMON_ARGS="--train \
  --name \"${BASE_NAME}\" \
  --tag \"${EXP_TAG}\" \
  --data-path \"${DATA_PATH}\" \
  --pretrained_dir \"${PRETRAINED_DIR}\" \
  --model_type \"${MODEL_TYPE}\" \
  --quant-mode \"${QUANT_MODE}\" \
  --img_size ${IMG_SIZE} \
  --dataset \"${DATASET}\" \
  --output_dir \"${OUTPUT_DIR}\" \
  --train_batch_size ${TRAIN_BATCH_SIZE} \
  --eval_batch_size ${EVAL_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACC_STEPS} \
  --num_epochs ${NUM_EPOCHS} \
  --qat_lr ${QAT_LR} \
  --max_grad_norm ${MAX_GRAD_NORM} \
  --warmup_steps ${WARMUP_STEPS}"

# Optional flags
[ "$USE_CHECKPOINT" = true ] && COMMON_ARGS="${COMMON_ARGS} --use-checkpoint"
[ "$FP16" = true ] && COMMON_ARGS="${COMMON_ARGS} --fp16"

# Distillation (only if teacher path exists)
if [ "$TEACHER_PATH" ] && [ -f "$TEACHER_PATH" ]; then
  COMMON_ARGS="${COMMON_ARGS} --distill --teacher \"${TEACHER_PATH}\" --distillation_loss_scale ${DISTILL_SCALE}"
  echo "✅ Distillation enabled with teacher: ${TEACHER_PATH}"
fi

# Full torchrun command as a single string
FULL_CMD="torchrun \
  --nproc_per_node=${GPUS_PER_NODE} \
  --master_port=${MASTER_PORT} \
  --master_addr=${MASTER_ADDR} \
  main.py ${COMMON_ARGS}"

# ----------------------------------------------------------------------
# 6️⃣  Launcher
# ----------------------------------------------------------------------
SELECT_LAUNCHER="${SELECT_LAUNCHER:-nohup}"

case "$SELECT_LAUNCHER" in
  nohup)
    OUT_LOG="${OUTPUT_DIR}/qat_run_${DATE_TAG}.log"
    echo "🚀 Launching via nohup (backgrounded)…"
    echo "📝 Command: $FULL_CMD"
    echo "📄 Log file: $OUT_LOG"
    
    # ✅ Use eval to properly expand the command string
    nohup bash -c "cd $(pwd) && eval $FULL_CMD" >"$OUT_LOG" 2>&1 &
    PID=$!
    
    echo "✅ Launched – PID $PID"
    echo "   Follow live output: tail -f $OUT_LOG"
    echo "   Monitor GPUs: watch -n 5 nvidia-smi"
    ;;

  screen)
    SESSION_NAME="qat_${DATE_TAG}"
    echo "🚀 Launching inside a detached screen session: $SESSION_NAME"
    echo "📝 Command: $FULL_CMD"
    
    # ✅ Use bash -c with eval for proper argument handling
    screen -S "$SESSION_NAME" -dm bash -c "cd $(pwd) && eval $FULL_CMD"
    
    echo "✅ To attach:   screen -r $SESSION_NAME"
    echo "✅ To detach:   Ctrl‑a d"
    echo "✅ To kill:     screen -S $SESSION_NAME -X quit"
    ;;

  tmux)
    SESSION_NAME="qat_${DATE_TAG}"
    echo "🚀 Launching inside a detached tmux session: $SESSION_NAME"
    echo "📝 Command: $FULL_CMD"
    
    # ✅ Use bash -c with eval for proper argument handling
    tmux new-session -d -s "$SESSION_NAME" "cd $(pwd) && eval $FULL_CMD"
    
    echo "✅ To attach:   tmux attach -t $SESSION_NAME"
    echo "✅ To detach:   Ctrl‑b d"
    echo "✅ To kill:     tmux kill-session -t $SESSION_NAME"
    ;;

  direct)
    # ✅ For debugging: run directly in foreground
    echo "🚀 Running directly in foreground (Ctrl-C to stop)…"
    echo "📝 Command: $FULL_CMD"
    eval $FULL_CMD
    ;;

  *)
    echo "❌ Unknown SELECT_LAUNCHER value: $SELECT_LAUNCHER"
    echo "    Use one of: nohup | screen | tmux | direct"
    exit 1
    ;;
esac

# ----------------------------------------------------------------------
# 7️⃣  Post-launch info
# ----------------------------------------------------------------------
echo ""
echo "📊 Experiment tag: ${EXP_TAG}"
echo "📁 Output dir: ${OUTPUT_DIR}"
echo "🔢 GPUs: ${GPUS_PER_NODE} | Batch per GPU: ${TRAIN_BATCH_SIZE} | Grad accum: ${GRAD_ACC_STEPS}"
echo "🎯 Effective batch size: $((TRAIN_BATCH_SIZE * GPUS_PER_NODE * GRAD_ACC_STEPS))"
echo "⏱️  NCCL timeout: 2 hours (adjust via NCCL_TIMEOUT env var)"
echo ""
echo "💡 Tips:"
echo "   - Monitor: tail -f ${OUTPUT_DIR}/qat_run_${DATE_TAG}.log"
echo "   - GPUs:    watch -n 2 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'"
echo "   - MLflow:  mlflow ui --port 5000  (then open http://localhost:5000)"
echo "   - Debug NCCL: export NCCL_DEBUG=INFO before running"