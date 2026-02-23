#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calibration module for INT8 quantization."""
from __future__ import absolute_import, division, print_function

import logging
import os
import torch
from tqdm import tqdm

from vit_qat.utils import load_pretrained_weights, cleanup_gpu_memory
from vit_qat.validate import valid
from vit_int8 import VisionTransformerINT8
from models.modeling import CONFIGS
import quant_utils
from data import build_loader

logger = logging.getLogger(__name__)


def calib(args, config, model=None):
    """
    Run calibration for INT8 quantization.
    """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    if model is None:
        num_classes = 1000
        model_cfg = CONFIGS[args.model_type]
        model = VisionTransformerINT8(
            model_cfg, args.img_size, zero_head=False, num_classes=num_classes
        )
        load_pretrained_weights(args, model)
        model.cuda(args.local_rank if args.local_rank != -1 else 0)

    _, _, train_loader, test_loader = build_loader(config, args)
    
    quant_utils.configure_model(model, args, calib=True)
    model.eval()
    quant_utils.enable_calibration(model)

    try:
        for step, (samples, _) in enumerate(
            tqdm(train_loader, desc="Calibration", total=args.num_calib_batch)
        ):
            if step >= args.num_calib_batch:
                break
            samples = samples.to(args.device)
            _ = model(samples)

        quant_utils.finish_calibration(model, args)
        quant_utils.configure_model(model, args, calib=False)

        if args.local_rank in [-1, 0]:
            val_acc1, val_acc5 = valid(
                args, config, model, test_loader, step=0, plot_confusion_matrix=False
            )
            logger.info(
                "Test Accuracy after calibration: Acc@1=%f, Acc@5=%f",
                val_acc1, val_acc5
            )

        # ✅ Save calibrated model to central location
        checkpoint_dir = "/mnt/manatee/irad/clue/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        out_path = os.path.join(checkpoint_dir, f"{args.model_type}_calib.pth")
        torch.save(model.state_dict(), out_path)
        logger.info("Calibrated model saved to %s", out_path)
        
        return val_acc1, val_acc5
        
    finally:
        # ✅ Cleanup
        del model
        cleanup_gpu_memory()