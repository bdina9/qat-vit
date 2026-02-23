# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """Training module with knowledge distillation support."""
# from __future__ import absolute_import, division, print_function

# import logging
# import numpy as np
# import torch
# import torch.distributed as dist
# from torch.cuda.amp import autocast, GradScaler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from tqdm import tqdm

# import mlflow

# from vit_qat.utils import (
#     AverageMeter, accuracy, save_model, load_pretrained_weights,
#     Knowledge_Distillation_Loss, plot_confusion_matrix_mlflow, cleanup_gpu_memory
# )
# from vit_qat.validate import valid
# from vit_int8 import VisionTransformerINT8
# from models.modeling import CONFIGS
# import quant_utils
# from data import build_loader

# logger = logging.getLogger(__name__)


# def train(args, config):
#     """Main training loop with knowledge distillation."""
#     torch.cuda.empty_cache()
#     if args.local_rank in [-1, 0]:
#         logger.info(f"GPU memory before model init: {torch.cuda.memory_allocated()/1024**3:.2f}GB used")

#     num_classes = 1000
#     logger.info(f"Training with num_classes={num_classes} for ImageNet-1K")
    
#     model_cfg = CONFIGS[args.model_type]
#     model = VisionTransformerINT8(
#         model_cfg, args.img_size, zero_head=False, num_classes=num_classes
#     )

#     if args.use_checkpoint:
#         logger.info("Enabling gradient checkpointing (essential for 384x384)")
#         if hasattr(model.transformer.encoder, 'set_grad_checkpointing'):
#             model.transformer.encoder.set_grad_checkpointing(True)
#         elif hasattr(model.transformer.encoder, 'gradient_checkpointing'):
#             model.transformer.encoder.gradient_checkpointing = True
#         else:
#             logger.warning("Model doesn't support gradient checkpointing - expect OOM at 384x384!")

#     load_pretrained_weights(args, model)
#     model.cuda(args.local_rank)
#     model.train()

#     teacher = None
#     kd_loss = None
#     if args.distill:
#         teacher = VisionTransformerINT8(
#             model_cfg, args.img_size, zero_head=False, num_classes=num_classes
#         )
#         teacher.load_from(np.load(args.teacher))
#         teacher.cuda(args.local_rank)
#         teacher.eval()
#         kd_loss = Knowledge_Distillation_Loss(
#             scale=args.distillation_loss_scale
#         ).cuda(args.local_rank)
#         quant_utils.set_quantizer_by_name(teacher, [""])

#     if args.local_rank != -1:
#         model = DDP(
#             model,
#             device_ids=[args.local_rank],
#             output_device=args.local_rank,
#             find_unused_parameters=False,
#         )

#     optimizer = torch.optim.SGD(
#         model.parameters(),
#         lr=args.qat_lr,
#         momentum=0.9,
#         weight_decay=args.weight_decay,
#     )
#     scaler = GradScaler(enabled=args.fp16)
#     quant_utils.configure_model(model, args, calib=False)

#     _, _, train_loader, test_loader = build_loader(config, args)
    
#     if args.local_rank in [-1, 0]:
#         for batch in test_loader:
#             x, y = batch
#             logger.info(f"🔍 Label range: min={y.min()}, max={y.max()}, unique={len(y.unique())}")
#             logger.info(f"🔍 Expected num_classes: {num_classes}")
#             if y.max() >= num_classes:
#                 logger.error(f"❌ ERROR: Label max ({y.max()}) >= num_classes ({num_classes})!")
#             break
    
#     losses = AverageMeter()
#     # ✅ Training accuracy meters
#     train_acc1_meter = AverageMeter()
#     train_acc5_meter = AverageMeter()
    
#     best_acc = 0
#     global_step = 0

#     try:
#         for epoch_i in range(args.num_epochs):
#             model.train()
#             losses.reset()
#             train_acc1_meter.reset()
#             train_acc5_meter.reset()
            
#             # ✅ GPU cleanup at epoch start
#             cleanup_gpu_memory()
            
#             epoch_pbar = tqdm(
#                 train_loader,
#                 desc=f"Epoch {epoch_i+1}/{args.num_epochs}",
#                 disable=args.local_rank not in [-1, 0],
#             )
#             for step, batch in enumerate(epoch_pbar):
#                 batch = tuple(t.to(args.device) for t in batch)
#                 x, y = batch

#                 with autocast(enabled=args.fp16):
#                     logits, loss = model(x, y)
#                     if teacher:
#                         with torch.no_grad():
#                             teacher_logits, _ = teacher(x.float())
#                         loss = loss + kd_loss.get_knowledge_distillation_loss(
#                             logits, teacher_logits
#                         )

#                 if args.gradient_accumulation_steps > 1:
#                     loss = loss / args.gradient_accumulation_steps

#                 scaler.scale(loss).backward()

#                 if (step + 1) % args.gradient_accumulation_steps == 0:
#                     scaler.unscale_(optimizer)
#                     torch.nn.utils.clip_grad_norm_(
#                         model.parameters(), args.max_grad_norm
#                     )
#                     scaler.step(optimizer)
#                     scaler.update()
#                     optimizer.zero_grad()
                    
#                     losses.update(loss.item(), y.size(0))
                    
#                     # ✅ Compute and track training accuracy
#                     with torch.no_grad():
#                         acc1, acc5 = accuracy(logits, y, topk=(1, 5))
#                         train_acc1_meter.update(acc1.item(), y.size(0))
#                         train_acc5_meter.update(acc5.item(), y.size(0))

#                     epoch_pbar.set_description(
#                         f"EPOCH [{epoch_i+1}/{args.num_epochs}] "
#                         f"loss={losses.val:.4f} "
#                         f"Acc@1={train_acc1_meter.avg:.2f}"
#                     )

#                     if args.local_rank in [-1, 0] and mlflow.active_run():
#                         mlflow.log_metric("train_loss", losses.val, step=int(global_step))
#                         mlflow.log_metric("train_acc1", train_acc1_meter.avg, step=int(global_step))
#                         mlflow.log_metric("train_acc5", train_acc5_meter.avg, step=int(global_step))
#                         # ❌ Removed LR logging
                    
#                     global_step += 1

#             if args.local_rank in [-1, 0]:
#                 mlflow.log_metric("train_loss_epoch_avg", losses.avg, step=int(epoch_i))
#                 mlflow.log_metric("train_acc1_epoch_avg", train_acc1_meter.avg, step=int(epoch_i))
#                 mlflow.log_metric("train_acc5_epoch_avg", train_acc5_meter.avg, step=int(epoch_i))
                
#                 # ✅ Validation with confusion matrix EVERY epoch
#                 val_acc1, val_acc5 = valid(
#                     args, config, model, test_loader, step=epoch_i, 
#                     plot_confusion_matrix=True
#                 )

#                 logger.info(
#                     f"Epoch {epoch_i+1} | "
#                     f"Train Loss: {losses.avg:.4f} | "
#                     f"Train Acc@1: {train_acc1_meter.avg:.2f}% | "
#                     f"Val Acc@1: {val_acc1:.2f}% | "
#                     f"Val Acc@5: {val_acc5:.2f}%"
#                 )
                
#                 if val_acc1 > best_acc:
#                     best_acc = val_acc1
#                     # ✅ Save checkpoint to /mnt/manatee/irad/clue/checkpoints/
#                     save_model(args, model, checkpoint_dir="/mnt/manatee/irad/clue/checkpoints")

#                 # ✅ GPU cleanup at epoch end
#                 cleanup_gpu_memory()

#     finally:
#         # ✅ Cleanup at end of training
#         if args.local_rank in [-1, 0]:
#             logger.info(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
        
#         # ✅ Free model memory
#         del model, optimizer, scaler
#         if teacher is not None:
#             del teacher
#         if kd_loss is not None:
#             del kd_loss
        
#         # ✅ Final GPU cleanup
#         cleanup_gpu_memory()
        
#         # ✅ Destroy DDP if needed
#         if dist.is_initialized():
#             dist.destroy_process_group()
    
#     return best_acc

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training module with knowledge distillation support."""
from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import mlflow

from vit_qat.utils import (
    AverageMeter, accuracy, save_model, load_pretrained_weights,
    Knowledge_Distillation_Loss, plot_confusion_matrix_mlflow, cleanup_gpu_memory
)
from vit_qat.validate import valid
from vit_int8 import VisionTransformerINT8
from models.modeling import CONFIGS
import quant_utils
from data import build_loader

# 🔹 Optuna optional import for hyperparameter search
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

logger = logging.getLogger(__name__)


def train(args, config, trial=None):
    """
    Main training loop with knowledge distillation.
    
    Args:
        args: Training arguments
        config: Configuration object
        trial: Optional Optuna trial object for hyperparameter optimization
    """
    torch.cuda.empty_cache()
    if args.local_rank in [-1, 0]:
        logger.info(f"GPU memory before model init: {torch.cuda.memory_allocated()/1024**3:.2f}GB used")

    num_classes = 200
    logger.info(f"Training with num_classes={num_classes} for ImageNet-1K")
    
    model_cfg = CONFIGS[args.model_type]
    model = VisionTransformerINT8(
        model_cfg, args.img_size, zero_head=False, num_classes=num_classes
    )

    if args.use_checkpoint:
        logger.info("Enabling gradient checkpointing (essential for 384x384)")
        if hasattr(model.transformer.encoder, 'set_grad_checkpointing'):
            model.transformer.encoder.set_grad_checkpointing(True)
        elif hasattr(model.transformer.encoder, 'gradient_checkpointing'):
            model.transformer.encoder.gradient_checkpointing = True
        else:
            logger.warning("Model doesn't support gradient checkpointing - expect OOM at 384x384!")

    load_pretrained_weights(args, model)
    model.cuda(args.local_rank)
    model.train()

    teacher = None
    kd_loss = None
    if args.distill:
        teacher = VisionTransformerINT8(
            model_cfg, args.img_size, zero_head=False, num_classes=num_classes
        )
        teacher.load_from(np.load(args.teacher))
        teacher.cuda(args.local_rank)
        teacher.eval()
        kd_loss = Knowledge_Distillation_Loss(
            scale=args.distillation_loss_scale
        ).cuda(args.local_rank)
        quant_utils.set_quantizer_by_name(teacher, [""])

    if args.local_rank != -1:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.qat_lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=args.fp16)
    quant_utils.configure_model(model, args, calib=False)

    _, _, train_loader, test_loader = build_loader(config, args)
    
    if args.local_rank in [-1, 0]:
        for batch in test_loader:
            x, y = batch
            logger.info(f"🔍 Label range: min={y.min()}, max={y.max()}, unique={len(y.unique())}")
            logger.info(f"🔍 Expected num_classes: {num_classes}")
            if y.max() >= num_classes:
                logger.error(f"❌ ERROR: Label max ({y.max()}) >= num_classes ({num_classes})!")
            break
    
    losses = AverageMeter()
    train_acc1_meter = AverageMeter()
    train_acc5_meter = AverageMeter()
    
    best_acc = 0
    global_step = 0

    try:
        for epoch_i in range(args.num_epochs):
            model.train()
            losses.reset()
            train_acc1_meter.reset()
            train_acc5_meter.reset()
            
            cleanup_gpu_memory()
            
            epoch_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch_i+1}/{args.num_epochs}",
                disable=args.local_rank not in [-1, 0],
            )
            for step, batch in enumerate(epoch_pbar):
                batch = tuple(t.to(args.device) for t in batch)
                x, y = batch

                with autocast(enabled=args.fp16):
                    logits, loss = model(x, y)
                    if teacher:
                        with torch.no_grad():
                            teacher_logits, _ = teacher(x.float())
                        loss = loss + kd_loss.get_knowledge_distillation_loss(
                            logits, teacher_logits
                        )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    losses.update(loss.item(), y.size(0))
                    
                    with torch.no_grad():
                        acc1, acc5 = accuracy(logits, y, topk=(1, 5))
                        train_acc1_meter.update(acc1.item(), y.size(0))
                        train_acc5_meter.update(acc5.item(), y.size(0))

                    epoch_pbar.set_description(
                        f"EPOCH [{epoch_i+1}/{args.num_epochs}] "
                        f"loss={losses.val:.4f} "
                        f"Acc@1={train_acc1_meter.avg:.2f}"
                    )

                    if args.local_rank in [-1, 0] and mlflow.active_run():
                        mlflow.log_metric("train_loss", losses.val, step=int(global_step))
                        mlflow.log_metric("train_acc1", train_acc1_meter.avg, step=int(global_step))
                        mlflow.log_metric("train_acc5", train_acc5_meter.avg, step=int(global_step))
                    
                    global_step += 1

            if args.local_rank in [-1, 0]:
                mlflow.log_metric("train_loss_epoch_avg", losses.avg, step=int(epoch_i))
                mlflow.log_metric("train_acc1_epoch_avg", train_acc1_meter.avg, step=int(epoch_i))
                mlflow.log_metric("train_acc5_epoch_avg", train_acc5_meter.avg, step=int(epoch_i))
                
                # ✅ Validation with confusion matrix EVERY epoch
                val_acc1, val_acc5 = valid(
                    args, config, model, test_loader, step=epoch_i, 
                    plot_confusion_matrix=True
                )

                logger.info(
                    f"Epoch {epoch_i+1} | "
                    f"Train Loss: {losses.avg:.4f} | "
                    f"Train Acc@1: {train_acc1_meter.avg:.2f}% | "
                    f"Val Acc@1: {val_acc1:.2f}% | "
                    f"Val Acc@5: {val_acc5:.2f}%"
                )
                
                # 🔹 Optuna pruning hook: report intermediate value & check for pruning
                if HAS_OPTUNA and trial is not None:
                    trial.report(val_acc1, epoch_i)
                    if trial.should_prune():
                        logger.info(f"🪓 Trial {trial.number} pruned at epoch {epoch_i} (val_acc1={val_acc1:.2f})")
                        raise optuna.TrialPruned()
                
                if val_acc1 > best_acc:
                    best_acc = val_acc1
                    save_model(args, model, checkpoint_dir="/mnt/manatee/irad/clue/checkpoints")

                cleanup_gpu_memory()

    except optuna.TrialPruned:
        # 🔹 Handle Optuna pruning gracefully - cleanup and re-raise
        logger.info(f"⚡ Trial {trial.number} pruned - cleaning up resources")
        raise
        
    finally:
        if args.local_rank in [-1, 0]:
            logger.info(f"Training complete. Best validation accuracy: {best_acc:.2f}%")
        
        # ✅ Free model memory
        del model, optimizer, scaler
        if teacher is not None:
            del teacher
        if kd_loss is not None:
            del kd_loss
        
        cleanup_gpu_memory()
        
        if dist.is_initialized():
            dist.destroy_process_group()
    
    return best_acc