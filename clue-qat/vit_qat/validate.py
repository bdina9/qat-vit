#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validation module for model evaluation."""
from __future__ import absolute_import, division, print_function

import logging
import torch
import mlflow
from torchmetrics.classification import MulticlassConfusionMatrix

from vit_qat.utils import AverageMeter, accuracy, plot_confusion_matrix_mlflow, cleanup_gpu_memory

logger = logging.getLogger(__name__)


@torch.no_grad()
def valid(args, config, model, test_loader, step=0, plot_confusion_matrix=False):
    """
    Runs validation with TorchMetrics for Confusion Matrix.
    
    Returns:
        tuple: (acc1, acc5) top-1 and top-5 validation accuracies
    """
    loss_fct = torch.nn.CrossEntropyLoss()
    eval_losses = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    # ✅ TorchMetrics Confusion Matrix (on device, sync_on_compute=False)
    metric_cm = None
    if plot_confusion_matrix and args.local_rank in [-1, 0]:
        metric_cm = MulticlassConfusionMatrix(
            num_classes=1000, normalize=None, sync_on_compute=False
        ).to(args.device)

    model.eval()
    for step_batch, batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch

        logits, _ = model(x)
        loss = loss_fct(logits, y)

        acc1, acc5 = accuracy(logits, y, topk=(1, 5))
        
        # ✅ Update confusion matrix metric
        if metric_cm is not None:
            preds = torch.argmax(logits, dim=1)
            metric_cm.update(preds, y)

        eval_losses.update(loss.item(), y.size(0))
        acc1_meter.update(acc1.item(), y.size(0))
        acc5_meter.update(acc5.item(), y.size(0))

    logger.info(
        f" * Validation loss {eval_losses.avg:.4f} "
        f"Acc@1 {acc1_meter.avg:.3f}  Acc@5 {acc5_meter.avg:.3f}"
    )

    if args.local_rank in [-1, 0] and mlflow.active_run():
        mlflow.log_metric("val_loss", eval_losses.avg, step=int(step))
        mlflow.log_metric("val_acc1", acc1_meter.avg, step=int(step))
        mlflow.log_metric("val_acc5", acc5_meter.avg, step=int(step))
        
        # ✅ Plot confusion matrix if requested
        if plot_confusion_matrix and metric_cm is not None:
            cm_tensor = metric_cm.compute()
            # ✅ New call: full matrix + all metrics
            plot_confusion_matrix_mlflow(args, cm_tensor, step, num_classes=1000, prefix="val")
            del cm_tensor
            metric_cm.reset()

    # ✅ Clean up metric
    if metric_cm is not None:
        del metric_cm
    
    # ✅ GPU cleanup after validation
    cleanup_gpu_memory()

    return acc1_meter.avg, acc5_meter.avg