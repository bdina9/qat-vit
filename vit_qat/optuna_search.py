#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optuna hyperparameter search wrapper for ViT-QAT training."""
import optuna
from optuna.integration import PyTorchLightningPruningCallback  # Optional
from vit_qat.train_kd import train
from vit_qat.validate import valid
from vit_qat.utils import cleanup_gpu_memory
import torch
import gc

def objective(trial, args_template, config):
    """
    Optuna objective function: returns validation accuracy to maximize.
    """
    # 🔹 Sample hyperparameters
    lr = trial.suggest_float("qat_lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    distill_scale = trial.suggest_float("distillation_loss_scale", 0.1, 10.0, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.3)
    
    # Optional: quantization-aware params if exposed
    # quant_bits = trial.suggest_categorical("quant_bits", [8])  # Future: try 4/8
    
    # 🔹 Create trial-specific args (copy template to avoid mutation)
    import copy
    args = copy.deepcopy(args_template)
    args.qat_lr = lr
    args.weight_decay = weight_decay
    args.distillation_loss_scale = distill_scale
    args.warmup_steps = int(args.num_epochs * len(train_loader) * warmup_ratio)  # Approximate
    
    # 🔹 Unique run tag for this trial
    args.tag = f"{args.tag}_trial_{trial.number}"
    args.name = f"{args.name}_t{trial.number}"
    
    # 🔹 Setup MLflow run for this trial (optional but recommended)
    import mlflow
    if args.local_rank in [-1, 0]:
        mlflow.set_tag("optuna_trial", trial.number)
        mlflow.log_params({
            "qat_lr": lr,
            "weight_decay": weight_decay,
            "distillation_loss_scale": distill_scale,
            "warmup_ratio": warmup_ratio,
        })
    
    try:
        # 🔹 Run training (modified to return epoch metrics for pruning)
        best_val_acc = train_with_pruning_hook(args, config, trial)
        
        # 🔹 Report final result
        trial.set_user_attr("best_val_acc", best_val_acc)
        return best_val_acc
        
    except Exception as e:
        trial.set_user_attr("error", str(e))
        raise optuna.TrialPruned() if "prune" in str(e).lower() else e
        
    finally:
        # 🔹 Aggressive cleanup between trials
        cleanup_gpu_memory()
        gc.collect()
        torch.cuda.empty_cache()