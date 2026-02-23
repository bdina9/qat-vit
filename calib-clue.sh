#!/bin/bash
# calib-clue.sh - Single GPU calibration

torchrun \
  --nproc_per_node=1 \
  --master_port=12345 \
  --master_addr=127.0.0.1 \
  main.py \
  --calib \
  --name "vit_calib_float2int8" \
  --data-path "/mnt/manatee/irad/clue/imagenet1k/imagenet1k" \
  --pretrained_dir "/mnt/manatee/irad/clue/checkpoints/ViT-B_16.npz" \
  --model_type "ViT-B_16" \
  --quant-mode "ft2" \
  --img_size 384 \
  --dataset "imagenet" \
  --output_dir "calib_output" \
  --train_batch_size 8 \
  --eval_batch_size 16 \
  --num-calib-batch 10 \
  --calib-batchsz 8 \
  --calib-output-path "calib-checkpoint" \
  --fp16 \
  --use-checkpoint