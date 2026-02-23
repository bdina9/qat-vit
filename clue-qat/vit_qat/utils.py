#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared utilities for training, validation, and calibration."""
from __future__ import absolute_import, division, print_function

import logging
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Knowledge_Distillation_Loss(torch.nn.Module):
    """Knowledge distillation loss using KL divergence."""
    def __init__(self, scale, T=3):
        super().__init__()
        self.kl = torch.nn.KLDivLoss(reduction="batchmean")
        self.T = T
        self.scale = scale
    
    def get_knowledge_distillation_loss(self, student_logits, teacher_logits):
        loss_kl = self.kl(
            torch.nn.functional.log_softmax(student_logits / self.T, dim=1),
            torch.nn.functional.softmax(teacher_logits / self.T, dim=1),
        )
        return self.scale * loss_kl


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]


def set_seed(args):
    """Set random seeds for reproducibility."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_model(args, model, checkpoint_dir="/mnt/manatee/irad/clue/checkpoints"):
    """Save model checkpoint to central location."""
    model_to_save = model.module if hasattr(model, "module") else model
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{args.name}_checkpoint.bin")
    torch.save(model_to_save.state_dict(), ckpt_path)
    logger.info("Saved checkpoint %s", ckpt_path)
    return ckpt_path


def load_pretrained_weights(args, model):
    """Load pretrained weights from .npz or .pth file."""
    if not os.path.isfile(args.pretrained_dir):
        raise FileNotFoundError(f"Pretrained file not found: {args.pretrained_dir}")
    
    if args.pretrained_dir.lower().endswith(".npz"):
        logger.info("Loading .npz weights from %s", args.pretrained_dir)
        model.load_from(np.load(args.pretrained_dir))
    else:
        logger.info("Loading torch checkpoint from %s", args.pretrained_dir)
        ckpt = torch.load(args.pretrained_dir, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)


def setup_logging(local_rank, log_file_path=None):
    """Configure logging for distributed training."""
    is_main_process = (local_rank in [-1, 0])
    log_level = logging.INFO if is_main_process else logging.WARNING
    
    logger = logging.getLogger()
    logger.handlers = []
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=log_level,
        force=True
    )
    
    if log_file_path and is_main_process:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
    
    return logger


def cleanup_gpu_memory():
    """Clean up GPU memory - call between epochs and at end of training."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        logger.debug(f"GPU Memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

def plot_confusion_matrix_mlflow(args, cm_tensor, step, num_classes=200, prefix="val"):
    """
    Plot and log confusion matrix + per-class metrics to MLflow.
    Generates:
    1. Full confusion matrix heatmap (1000x1000)
    2. Per-class Accuracy, Precision, Recall, F1-score distributions
    3. Top/bottom performing classes bar charts
    """
    import mlflow
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Convert TorchMetrics tensor to numpy
    cm = cm_tensor.cpu().numpy().astype(np.float64)
    
    # ===== Compute per-class metrics =====
    # TP = diagonal, FN = row sum - TP, FP = col sum - TP
    TP = np.diag(cm)
    FN = cm.sum(axis=1) - TP
    FP = cm.sum(axis=0) - TP
    
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)  # Same as per-class accuracy
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        accuracy = TP / (TP + FN + FP)  # Optional: overall accuracy per class
        
    # Replace NaN/inf with 0
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)
    accuracy = np.nan_to_num(accuracy)
    
    cm_dir = os.path.join(args.output_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    
    # ===== Plot 1: Full Confusion Matrix (1000x1000) =====
    # Use low-res interpolation for readability
    fig1, ax1 = plt.subplots(figsize=(20, 16))
    
    # Normalize rows for better visualization (each true class sums to 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    
    # Use imshow with interpolation to make 1000x1000 viewable
    im = ax1.imshow(cm_norm, cmap='viridis', interpolation='bilinear', aspect='auto')
    ax1.set_title(f'{prefix.upper()} Full Confusion Matrix (1000 Classes) - Epoch {step}', 
                  fontsize=16, pad=20)
    ax1.set_xlabel('Predicted Class', fontsize=12)
    ax1.set_ylabel('True Class', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Count', rotation=270, labelpad=20)
    
    # Add diagonal line to highlight correct predictions
    ax1.plot(np.arange(num_classes), np.arange(num_classes), 
             color='red', linestyle='--', linewidth=0.5, alpha=0.3, label='Correct')
    
    plot1_path = os.path.join(cm_dir, f"{prefix}_full_cm_epoch_{step}.png")
    plt.savefig(plot1_path, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig1)
    
    if mlflow.active_run():
        mlflow.log_artifact(plot1_path, artifact_path=f"confusion_matrices/{prefix}")
    
    # ===== Plot 2: Per-Class Metrics Distribution Histograms =====
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [('Accuracy', accuracy, 'green'), 
               ('Precision', precision, 'blue'),
               ('Recall', recall, 'orange'), 
               ('F1-Score', f1, 'red')]
    
    for idx, (name, values, color) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        ax.hist(values, bins=50, edgecolor='black', alpha=0.7, color=color)
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel('Number of Classes', fontsize=10)
        ax.set_title(f'{name} Distribution (Mean: {np.mean(values):.3f})', fontsize=11)
        ax.axvline(np.mean(values), color='black', linestyle='--', linewidth=1.5, 
                   label=f'Mean: {np.mean(values):.3f}')
        ax.axvline(np.median(values), color='gray', linestyle=':', linewidth=1, 
                   label=f'Median: {np.median(values):.3f}')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    fig2.suptitle(f'{prefix.upper()} Per-Class Metrics - Epoch {step}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plot2_path = os.path.join(cm_dir, f"{prefix}_metrics_dist_epoch_{step}.png")
    plt.savefig(plot2_path, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    if mlflow.active_run():
        mlflow.log_artifact(plot2_path, artifact_path=f"confusion_matrices/{prefix}")
    
    # ===== Plot 3: Top 20 Best & Worst F1-Score Classes =====
    fig3, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Worst 20 by F1
    worst_idx = np.argsort(f1)[:20]
    axes[0].barh(range(20), f1[worst_idx], color='salmon', edgecolor='black')
    axes[0].set_yticks(range(20))
    axes[0].set_yticklabels(worst_idx, fontsize=8)
    axes[0].set_xlabel('F1-Score', fontsize=10)
    axes[0].set_title('20 Worst Performing Classes (by F1)', fontsize=11)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Best 20 by F1
    best_idx = np.argsort(f1)[-20:][::-1]
    axes[1].barh(range(20), f1[best_idx], color='lightgreen', edgecolor='black')
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels(best_idx, fontsize=8)
    axes[1].set_xlabel('F1-Score', fontsize=10)
    axes[1].set_title('20 Best Performing Classes (by F1)', fontsize=11)
    axes[1].grid(axis='x', alpha=0.3)
    
    fig3.suptitle(f'{prefix.upper()} F1-Score: Best vs Worst Classes - Epoch {step}', 
                  fontsize=14, y=1.05)
    plt.tight_layout()
    
    plot3_path = os.path.join(cm_dir, f"{prefix}_f1_extremes_epoch_{step}.png")
    plt.savefig(plot3_path, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    if mlflow.active_run():
        mlflow.log_artifact(plot3_path, artifact_path=f"confusion_matrices/{prefix}")
    
    # ===== Log summary metrics to MLflow =====
    if mlflow.active_run():
        mlflow.log_metrics({
            f"{prefix}/mean_accuracy": float(np.mean(accuracy)),
            f"{prefix}/mean_precision": float(np.mean(precision)),
            f"{prefix}/mean_recall": float(np.mean(recall)),
            f"{prefix}/mean_f1": float(np.mean(f1)),
            f"{prefix}/std_f1": float(np.std(f1)),
            f"{prefix}/min_f1": float(np.min(f1)),
            f"{prefix}/max_f1": float(np.max(f1)),
        }, step=int(step))
    
    logger.info(f"Logged {prefix} full CM + metrics plots for epoch {step} to MLflow")
    
    # ✅ Clean up large arrays
    del cm, cm_norm, TP, FN, FP, precision, recall, f1, accuracy