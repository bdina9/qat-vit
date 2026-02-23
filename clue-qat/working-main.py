#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import numpy as np
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import mlflow

sys.path.insert(0, "./ViT-pytorch")
from models.modeling import CONFIGS
from utils.dist_util import get_world_size
from data import build_loader
from config import get_config
import quant_utils
from vit_int8 import VisionTransformerINT8

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_file_path = os.path.join(os.getcwd(), "training.log")
file_handler = logging.FileHandler(log_file_path, mode="w")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

class AverageMeter(object):
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
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def save_model(args, model):
    model_to_save = model.module if hasattr(model, "module") else model
    ckpt_path = os.path.join(args.output_dir, f"{args.name}_checkpoint.bin")
    torch.save(model_to_save.state_dict(), ckpt_path)
    logger.info("Saved checkpoint %s", ckpt_path)

@torch.no_grad()
def valid(args, config, model, test_loader, step=0):
    """
    Runs validation, prints loss/Acc@1/Acc@5 for the epoch
    and records the same numbers as MLflow metrics.
    """
    loss_fct = torch.nn.CrossEntropyLoss()
    eval_losses = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    model.eval()
    for step_batch, batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch

        logits, _ = model(x)
        loss = loss_fct(logits, y)

        # top‑1 and top‑5 accuracies
        acc1, acc5 = accuracy(logits, y, topk=(1, 5))

        # keep running averages
        eval_losses.update(loss.item(), y.size(0))
        acc1_meter.update(acc1.item(), y.size(0))
        acc5_meter.update(acc5.item(), y.size(0))

    # ----- 1️⃣  Print the epoch‑averaged results ---------------------------------
    logger.info(
        f" * Validation loss {eval_losses.avg:.4f} "
        f"Acc@1 {acc1_meter.avg:.3f}  Acc@5 {acc5_meter.avg:.3f}"
    )

    # ----- 2️⃣  Log the same numbers to MLflow (only the main process) ----------
    if args.local_rank in [-1, 0] and mlflow.active_run():
        mlflow.log_metric("val_loss", eval_losses.avg, step=int(step))
        mlflow.log_metric("val_acc1", acc1_meter.avg, step=int(step))
        mlflow.log_metric("val_acc5", acc5_meter.avg, step=int(step))

    # ✅ FIXED: Return correct variable (was val_acc5_meter.avg which doesn't exist)
    return acc1_meter.avg, acc5_meter.avg

def load_pretrained_weights(args, model):
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

def train(args, config):
    torch.cuda.empty_cache()
    if args.local_rank in [-1, 0]:
        logger.info(f"GPU memory before model init: {torch.cuda.memory_allocated()/1024**3:.2f}GB used")

    # ✅ ImageNet-1K = 1000 classes (CORRECT)
    num_classes = 1000
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
        kd_loss = Knowledge_Distillation_Loss(scale=args.distillation_loss_scale).cuda(args.local_rank)
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
    
    # ✅ DEBUG: Verify label range in first batch
    if args.local_rank in [-1, 0]:
        for batch in test_loader:
            x, y = batch
            logger.info(f"🔍 Label range: min={y.min()}, max={y.max()}, unique={len(y.unique())}")
            logger.info(f"🔍 Expected num_classes: {num_classes}")
            if y.max() >= num_classes:
                logger.error(f"❌ ERROR: Label max ({y.max()}) >= num_classes ({num_classes})!")
            break
    
    losses = AverageMeter()
    best_acc = 0
    global_step = 0

    for epoch_i in range(args.num_epochs):
        model.train()
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
                    loss = loss + kd_loss.get_knowledge_distillation_loss(logits, teacher_logits)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # ✅ FIXED: Don't multiply loss by gradient_accumulation_steps
                losses.update(loss.item())

                epoch_pbar.set_description(
                    f"EPOCH [{epoch_i+1}/{args.num_epochs}] "
                    f"loss={losses.val:.4f} lr={optimizer.param_groups[0]['lr']:.7f}"
                )

                if args.local_rank in [-1, 0] and mlflow.active_run():
                    mlflow.log_metric("train_loss", losses.val, step=int(global_step))
                    mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=int(global_step))
                
                global_step += 1

        if args.local_rank in [-1, 0]:
            mlflow.log_metric("train_loss_epoch_avg", losses.avg, step=int(global_step))
            val_acc1, val_acc5 = valid(args, config, model, test_loader, step=global_step)

            logger.info(
                f"Epoch {epoch_i+1} Validation Acc@1: {val_acc1:.2f}%  "
                f"Acc@5: {val_acc5:.2f}%"
            )
            if val_acc1 > best_acc:
                best_acc = val_acc1
                save_model(args, model)

            losses.reset()

    if args.local_rank in [-1, 0]:
        logger.info(f"Training complete. Best validation accuracy: {best_acc:.2f}%")

def calib(args, config, model):
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    _, _, train_loader, test_loader = build_loader(config, args)
    quant_utils.configure_model(model, args, calib=True)
    model.eval()
    quant_utils.enable_calibration(model)

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
        val_acc1, val_acc5 = valid(args, config, model, test_loader)
        logger.info("Test Accuracy after calibration: Acc@1=%f, Acc@5=%f", val_acc1, val_acc5)

    os.makedirs(args.calib_output_path, exist_ok=True)
    out_path = os.path.join(
        args.calib_output_path, f"{args.model_type}_calib.pth"
    )
    torch.save(model.state_dict(), out_path)
    logger.info("Calibrated model saved to %s", out_path)

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, required=True, help='path to dataset')
    parser.add_argument('--pretrained', help='pretrained weight from checkpoint')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level')
    parser.add_argument('--output', default='output', type=str, metavar='PATH', help='root of output folder')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--calib', action='store_true', help='Perform calibration only')
    parser.add_argument('--train', action='store_true', help='Perform training only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--num-calib-batch', type=int, default=10, help='Number of batches for calibration')
    parser.add_argument('--calib-batchsz', type=int, default=8, help='Batch size when doing calibration')
    parser.add_argument('--calib-output-path', type=str, default='calib-checkpoint', help='Output directory to save calibrated model')
    parser.add_argument("--num-epochs", type=int, default=90, help="Number of epochs to run QAT fintuning")
    parser.add_argument("--qat-lr", type=float, default=1e-3, help="learning rate for QAT")
    parser.add_argument("--distill", action='store_true', help='Using distillation')
    parser.add_argument("--teacher", type=str, help='teacher model path')
    parser.add_argument('--distillation_loss_scale', type=float, default=1.0, help="scale applied to distillation component of loss")
    parser.add_argument("--name", required=True, help="Name of this run")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="imagenet", help="Which downstream task")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"], default="ViT-B_16", help="Which variant to use")
    parser.add_argument("--pretrained_dir", type=str, required=True, help="Where to search for pretrained ViT models")
    parser.add_argument("--output_dir", default="output", type=str, help="The output directory where checkpoints will be written")
    parser.add_argument("--img_size", default=384, type=int, help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training")
    parser.add_argument("--eval_batch_size", default=16, type=int, help="Total batch size for eval")
    parser.add_argument("--eval_every", default=2000, type=int, help="Run prediction on validation set every so many steps")
    parser.add_argument("--learning_rate", default=3e-2, type=float, help="The initial learning rate for SGD")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay if we apply some")
    parser.add_argument("--num_steps", default=10000, type=int, help="Total number of training epochs to perform")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine", help="How to decay the learning rate")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Step of training to perform learning rate warmup for")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help="Number of updates steps to accumulate")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2', help="For fp16: Apex AMP optimization level")
    parser.add_argument('--loss_scale', type=float, default=0, help="Loss scaling to improve fp16 numeric stability")
    
    quant_utils.add_arguments(parser)
    
    args, _ = parser.parse_known_args()
    
    args.batch_size = args.train_batch_size
    
    if args.quant_mode is not None:
        args = quant_utils.set_args(args)
    quant_utils.set_default_quantizers(args)
    
    config = get_config(args)
    return args, config

def main():
    args, config = parse_option()

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

    log_level = logging.INFO if is_main_process else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=log_level,
    )
    logger.info(
        f"Rank={args.rank}/{args.world_size} | Local Rank={args.local_rank} | "
        f"Device={args.device} | n_gpu={args.n_gpu} | img_size={args.img_size} | "
        f"use_checkpoint={args.use_checkpoint} | fp16={args.fp16}"
    )

    set_seed(args)
    os.makedirs(args.output_dir, exist_ok=True)

    if is_main_process:
        mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")
        mlflow.set_experiment("ViT_QAT_Clue")

    if args.calib:
        num_classes = 1000  # ImageNet
        model_cfg = CONFIGS[args.model_type]
        model = VisionTransformerINT8(
            model_cfg, args.img_size, zero_head=False, num_classes=num_classes
        )
        load_pretrained_weights(args, model)
        model.cuda(args.local_rank if args.local_rank != -1 else 0)
        calib(args, config, model)

    if args.train:
        if is_main_process:
            mlflow.start_run(run_name=args.name)
            mlflow.log_params(vars(args))
            logger.info("MLflow run started (rank 0 only)")
        try:
            train(args, config)
        finally:
            if is_main_process:
                mlflow.log_artifact(log_file_path, artifact_path="logs")
                mlflow.end_run()
                logger.info("MLflow run closed (rank 0)")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()