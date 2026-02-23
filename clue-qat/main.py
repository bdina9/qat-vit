# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """Main entry point for ViT QAT pipeline."""
# from __future__ import absolute_import, division, print_function

# import argparse
# import logging
# import os
# import sys
# from datetime import timedelta

# import torch
# import torch.distributed as dist
# import mlflow

# sys.path.insert(0, "./ViT-pytorch")

# from models.modeling import CONFIGS
# from config import get_config
# import quant_utils
# from data import build_loader
# from vit_int8 import VisionTransformerINT8

# from vit_qat.utils import set_seed, setup_logging, cleanup_gpu_memory
# from vit_qat.train_kd import train
# from vit_qat.calibrate import calib

# logger = logging.getLogger(__name__)


# def parse_option():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--data-path', type=str, required=True, help='path to dataset')
#     parser.add_argument('--pretrained_dir', type=str, required=True)
#     parser.add_argument("--output_dir", default="output", type=str)
    
#     parser.add_argument('--train', action='store_true')
#     parser.add_argument('--calib', action='store_true')
#     parser.add_argument('--eval', action='store_true')
#     parser.add_argument('--throughput', action='store_true')
    
#     parser.add_argument("--model_type", choices=[
#         "ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"
#     ], default="ViT-B_16")
#     parser.add_argument("--img_size", default=384, type=int)
#     parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="imagenet")
    
#     parser.add_argument("--num_epochs", type=int, default=90)
#     parser.add_argument("--qat_lr", type=float, default=1e-3)
#     parser.add_argument("--train_batch_size", default=16, type=int)
#     parser.add_argument("--eval_batch_size", default=16, type=int)
#     parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
#     parser.add_argument("--max_grad_norm", default=1.0, type=float)
#     parser.add_argument("--weight_decay", default=0, type=float)
#     parser.add_argument("--warmup_steps", default=500, type=int)
#     parser.add_argument("--use-checkpoint", action='store_true')
    
#     parser.add_argument('--fp16', action='store_true')
#     parser.add_argument('--fp16_opt_level', type=str, default='O2')
#     parser.add_argument('--loss_scale', type=float, default=0)
#     parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'])
    
#     parser.add_argument("--distill", action='store_true')
#     parser.add_argument("--teacher", type=str)
#     parser.add_argument('--distillation_loss_scale', type=float, default=1.0)
    
#     parser.add_argument('--num-calib-batch', type=int, default=10)
#     parser.add_argument('--calib-batchsz', type=int, default=8)
#     parser.add_argument('--calib-output-path', type=str, default='calib-checkpoint')
    
#     parser.add_argument("--local_rank", type=int, default=-1)
#     parser.add_argument('--seed', type=int, default=42)
    
#     parser.add_argument("--name", required=True)
#     parser.add_argument('--tag', help='tag of experiment')
#     parser.add_argument('--output', default='output', type=str, metavar='PATH')
    
#     quant_utils.add_arguments(parser)
    
#     args, _ = parser.parse_known_args()
#     args.batch_size = args.train_batch_size
    
#     if args.quant_mode is not None:
#         args = quant_utils.set_args(args)
#     quant_utils.set_default_quantizers(args)
    
#     config = get_config(args)
#     return args, config


# def setup_distributed(args):
#     """Initialize distributed training environment."""
#     if "LOCAL_RANK" in os.environ:
#         args.local_rank = int(os.environ["LOCAL_RANK"])

#     if args.local_rank != -1:
#         torch.cuda.set_device(args.local_rank)
#         args.device = torch.device("cuda", args.local_rank)
#         dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
#         args.n_gpu = 1
#         args.world_size = dist.get_world_size()
#         args.rank = dist.get_rank()
#         is_main_process = (args.rank == 0)
#     else:
#         args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         args.n_gpu = torch.cuda.device_count()
#         args.world_size = 1
#         args.rank = 0
#         is_main_process = True
    
#     return is_main_process


# def main():
#     """Main entry point."""
#     args, config = parse_option()
#     is_main_process = setup_distributed(args)
    
#     log_file_path = os.path.join(os.getcwd(), "training.log")
#     setup_logging(args.local_rank, log_file_path)
    
#     logger.info(
#         f"Rank={args.rank}/{args.world_size} | Local Rank={args.local_rank} | "
#         f"Device={args.device} | n_gpu={args.n_gpu} | img_size={args.img_size} | "
#         f"use_checkpoint={args.use_checkpoint} | fp16={args.fp16}"
#     )

#     set_seed(args)
#     os.makedirs(args.output_dir, exist_ok=True)

#     if is_main_process:
#         mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")
#         mlflow.set_experiment("ViT_QAT_Clue")

#     try:
#         if args.calib:
#             calib(args, config)
        
#         elif args.train:
#             if is_main_process:
#                 mlflow.start_run(run_name=args.name)
#                 mlflow.log_params(vars(args))
#                 logger.info("MLflow run started (rank 0 only)")
            
#             train(args, config)
            
#         elif args.eval:
#             logger.info("Evaluation mode not yet implemented")
            
#         elif args.throughput:
#             logger.info("Throughput mode not yet implemented")
            
#         else:
#             logger.error("No task specified. Use --train, --calib, --eval, or --throughput")
#             sys.exit(1)
            
#     finally:
#         # ✅ Final cleanup
#         if is_main_process and mlflow.active_run():
#             mlflow.log_artifact(log_file_path, artifact_path="logs")
#             mlflow.end_run()
#             logger.info("MLflow run closed (rank 0)")

#         # ✅ GPU cleanup
#         cleanup_gpu_memory()

#         # ✅ Destroy process group
#         if dist.is_initialized():
#             dist.destroy_process_group()


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main entry point for ViT QAT pipeline with Optuna HPO support."""
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import copy
from datetime import timedelta

import torch
import torch.distributed as dist
import mlflow

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

sys.path.insert(0, "./ViT-pytorch")

from models.modeling import CONFIGS
from config import get_config
import quant_utils
from data import build_loader
from vit_int8 import VisionTransformerINT8
from vit_qat.utils import set_seed, setup_logging, cleanup_gpu_memory
from vit_qat.train_kd import train
from vit_qat.calibrate import calib

logger = logging.getLogger(__name__)


def parse_option():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-path', type=str, required=True, help='path to dataset')
    parser.add_argument('--pretrained_dir', type=str, required=True)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--calib', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--throughput', action='store_true')
    
    parser.add_argument("--model_type", choices=[
        "ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"
    ], default="ViT-B_16")
    parser.add_argument("--img_size", default=384, type=int)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="imagenet")
    
    parser.add_argument("--num_epochs", type=int, default=90)
    parser.add_argument("--qat_lr", type=float, default=1e-3)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--warmup_steps", default=500, type=int)
    
    parser.add_argument("--use-checkpoint", action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O2')
    parser.add_argument("--loss_scale", type=float, default=0)
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'])
    
    parser.add_argument("--distill", action='store_true')
    parser.add_argument("--teacher", type=str)
    parser.add_argument('--distillation_loss_scale', type=float, default=1.0)
    
    parser.add_argument('--num-calib-batch', type=int, default=10)
    parser.add_argument('--calib-batchsz', type=int, default=8)
    parser.add_argument('--calib-output-path', type=str, default='calib-checkpoint')
    
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument("--name", required=True)
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--output', default='output', type=str, metavar='PATH')
    
    # Optuna HPO arguments
    parser.add_argument('--hpo', action='store_true', help='Enable Optuna hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--hpo_storage', type=str, default='sqlite:///optuna_vit_qat.db')
    parser.add_argument('--hpo_study_name', type=str, default='vit_qat_hpo')
    parser.add_argument('--hpo_sampler', type=str, default='TPESampler',
                       choices=['TPESampler', 'RandomSampler'])
    parser.add_argument('--hpo_direction', type=str, default='maximize',
                       choices=['maximize', 'minimize'])
    parser.add_argument('--hpo_timeout', type=int, default=None)
    parser.add_argument('--hpo_epochs', type=int, default=None,
                       help='Override num_epochs for HPO trials')
    
    quant_utils.add_arguments(parser)
    
    args, _ = parser.parse_known_args()
    args.batch_size = args.train_batch_size
    
    if args.quant_mode is not None:
        args = quant_utils.set_args(args)
    quant_utils.set_default_quantizers(args)
    
    config = get_config(args)
    return args, config


def setup_distributed(args):
    """Initialize distributed training environment."""
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        args.n_gpu = 1
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        is_main_process = (args.rank == 0)
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.world_size = 1
        args.rank = 0
        is_main_process = True
    
    return is_main_process


def define_hpo_search_space(trial, args_base):
    """Define the hyperparameter search space for Optuna."""
    search_params = {}
    
    search_params['qat_lr'] = trial.suggest_float("qat_lr", 1e-5, 1e-2, log=True)
    search_params['weight_decay'] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    if args_base.distill:
        search_params['distillation_loss_scale'] = trial.suggest_float(
            "distillation_loss_scale", 0.1, 10.0, log=True
        )
    
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.3)
    search_params['max_grad_norm'] = trial.suggest_float("max_grad_norm", 0.5, 5.0)
    
    return search_params, warmup_ratio


def hpo_objective(trial, args_template, config, is_main_process):
    """Optuna objective function: runs training and returns validation accuracy."""
    search_params, warmup_ratio = define_hpo_search_space(trial, args_template)
    
    args = copy.deepcopy(args_template)
    
    for key, value in search_params.items():
        setattr(args, key, value)
    
    if args.hpo_epochs is not None:
        args.num_epochs = args.hpo_epochs
    args.warmup_steps = int(args.num_epochs * 1000 * warmup_ratio)
    
    args.tag = f"{args.tag or args.name}_trial{trial.number}"
    args.name = f"{args.name}_t{trial.number}"
    args.output_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 🔹 KEY FIX: Use nested MLflow run for each trial
    trial_run_name = f"trial_{trial.number}"
    if is_main_process and mlflow.active_run():
        with mlflow.start_run(run_name=trial_run_name, nested=True):
            mlflow.log_params({
                **search_params,
                "warmup_ratio": warmup_ratio,
                "trial_number": trial.number,
            })
            logger.info(f"🔍 Trial {trial.number} params: {search_params}")
            best_val_acc = _run_training_with_trial(args, config, trial)
    else:
        # Non-main process or no active run - just train
        best_val_acc = _run_training_with_trial(args, config, trial)
    
    return best_val_acc


def _run_training_with_trial(args, config, trial):
    """Helper to run training and handle Optuna pruning."""
    try:
        best_val_acc = train(args, config, trial=trial)
        if trial is not None:
            trial.set_user_attr("best_val_acc", best_val_acc)
            trial.set_user_attr("completed", True)
        return best_val_acc
    except optuna.TrialPruned:
        if trial is not None:
            trial.set_user_attr("pruned", True)
        raise
    except Exception as e:
        logger.error(f"❌ Trial {trial.number if trial else 'N/A'} failed: {e}")
        if trial is not None:
            trial.set_user_attr("error", str(e))
            trial.set_user_attr("completed", False)
        raise
    finally:
        cleanup_gpu_memory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_hpo_study(args, config, is_main_process):
    """Run Optuna hyperparameter optimization study."""
    if not HAS_OPTUNA:
        logger.error("❌ Optuna not installed. Run: pip install optuna optuna-integration")
        sys.exit(1)
    
    if args.hpo_sampler == "TPESampler":
        sampler = optuna.samplers.TPESampler(
            seed=args.seed,
            multivariate=True,
            group=True,
        )
    else:
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    
    study = optuna.create_study(
        study_name=args.hpo_study_name,
        storage=args.hpo_storage,
        direction=args.hpo_direction,
        sampler=sampler,
        load_if_exists=True,
    )
    
    logger.info(f"🔍 Starting Optuna study: {args.hpo_study_name}")
    logger.info(f"   Trials: {args.n_trials} | Sampler: {args.hpo_sampler} | Direction: {args.hpo_direction}")
    
    study.optimize(
        lambda trial: hpo_objective(trial, args, config, is_main_process),
        n_trials=args.n_trials,
        timeout=args.hpo_timeout,
        gc_after_trial=True,
        show_progress_bar=(is_main_process and args.local_rank in [-1, 0]),
    )
    
    if is_main_process:
        logger.info(f"✅ Study complete!")
        logger.info(f"🏆 Best value: {study.best_value:.4f}")
        logger.info(f"⚙️  Best params: {study.best_params}")
        
        best_cfg_path = os.path.join(args.output_dir, "best_hpo_config.json")
        import json
        with open(best_cfg_path, "w") as f:
            json.dump({
                "best_value": study.best_value,
                "best_params": study.best_params,
                "study_name": args.hpo_study_name,
                "n_trials": args.n_trials,
            }, f, indent=2)
        logger.info(f"💾 Best config saved to {best_cfg_path}")
        
        logger.info("\n📊 Top 5 trials:")
        for i, trial in enumerate(study.best_trials[:5], 1):
            logger.info(f"  {i}. Value: {trial.value:.4f} | Params: {trial.params}")
    
    return study.best_value


def main():
    """Main entry point."""
    args, config = parse_option()
    is_main_process = setup_distributed(args)
    
    log_file_path = os.path.join(os.getcwd(), "training.log")
    setup_logging(args.local_rank, log_file_path)
    
    logger.info(
        f"Rank={args.rank}/{args.world_size} | Local Rank={args.local_rank} | "
        f"Device={args.device} | n_gpu={args.n_gpu} | img_size={args.img_size} | "
        f"use_checkpoint={args.use_checkpoint} | fp16={args.fp16} | hpo={args.hpo}"
    )
    
    set_seed(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if is_main_process:
        mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")
        mlflow.set_experiment("ViT_QAT_Clue")
    
    try:
        if args.calib:
            calib(args, config)
            
        elif args.train or args.hpo:
            if args.hpo:
                if not HAS_OPTUNA:
                    logger.error("❌ --hpo requires Optuna. Install with: pip install optuna")
                    sys.exit(1)
                # 🔹 Start parent run for HPO study (not individual trials)
                if is_main_process:
                    mlflow.start_run(run_name=f"{args.name}_hpo_study")
                    # Log study-level params only (not trial-specific)
                    mlflow.log_params({
                        "n_trials": args.n_trials,
                        "hpo_sampler": args.hpo_sampler,
                        "hpo_direction": args.hpo_direction,
                        "base_qat_lr": args.qat_lr,  # Log base, not sampled
                        "base_weight_decay": args.weight_decay,
                    })
                run_hpo_study(args, config, is_main_process)
            else:
                # Normal training mode
                if is_main_process:
                    mlflow.start_run(run_name=args.name)
                    mlflow.log_params(vars(args))
                    logger.info("MLflow run started (rank 0 only)")
                train(args, config)
                
        elif args.eval:
            logger.info("Evaluation mode not yet implemented")
        elif args.throughput:
            logger.info("Throughput mode not yet implemented")
        else:
            logger.error("No task specified. Use --train, --calib, --eval, --throughput, or --hpo")
            sys.exit(1)
            
    finally:
        if is_main_process and mlflow.active_run():
            if os.path.exists(log_file_path):
                mlflow.log_artifact(log_file_path, artifact_path="logs")
            mlflow.end_run()
            logger.info("MLflow run closed (rank 0)")
        
        cleanup_gpu_memory()
        
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()