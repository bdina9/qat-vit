#!/usr/bin/env python
"""
Final QAT training with DDP using best hyperparameters from Optuna search.
Produces deployable INT8 quantized model.

Notes:
- Eager QAT/convert are CPU-oriented for true int8 execution; we evaluate the converted model on CPU.
- prepare_qat requires the model to be in train() mode.
- AMP is supported pre-QAT only; AMP is disabled once QAT starts (fixes dtype issues with fake-quant).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

# Torch quantization helpers (deprecated long-term, but functional today)
from torch.ao.quantization import convert, get_default_qat_qconfig, prepare_qat

# Fix import path (repo-root style)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models import model_registry  # noqa: E402


@torch.no_grad()
def evaluate_fp32(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Evaluate (fake-quant or fp32) on the given device."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / max(1, total)


@torch.no_grad()
def evaluate_quantized_cpu(quant_model: nn.Module, dataloader: DataLoader) -> float:
    """
    Evaluate a REAL quantized model.
    Eager-mode quantized ops are CPU-oriented; evaluate on CPU.
    """
    quant_model.eval()
    quant_model.to("cpu")
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to("cpu")
        labels = labels.to("cpu")
        outputs = quant_model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / max(1, total)


def _unwrap_ddp(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Final QAT training with DDP")
    parser.add_argument("--config", type=str, required=True, help="Path to best_params.yaml")
    parser.add_argument("--output-dir", type=str, default="./qat_search", help="Output directory")

    # Optional AMP for speed (pre-QAT only)
    parser.add_argument("--amp", action="store_true", help="Enable AMP pre-QAT (disabled once QAT starts).")

    # MLflow backend (recommend sqlite)
    parser.add_argument("--mlflow-uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--mlflow-exp", type=str, default="clue-vit-qat-final")

    args = parser.parse_args()

    # ---- DDP init (use LOCAL_RANK for GPU assignment) ----
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    is_distributed = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    # ---- Load hyperparameters ----
    with open(args.config, "r") as f:
        hparams = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- MLflow (rank 0 only) ----
    if rank == 0:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_exp)
        mlflow.start_run(run_name="final_training")
        mlflow.log_params(hparams)
        mlflow.log_param("amp_pre_qat", int(args.amp))
        mlflow.enable_system_metrics_logging()

        print("=" * 70)
        print("FINAL QAT TRAINING (DDP)")
        print("=" * 70)
        print(f"Hyperparameters: {hparams}")
        print("=" * 70)

    # ---- Data ----
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    trainset = datasets.CIFAR10(root="./data", train=True, download=(rank == 0), transform=transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=(rank == 0), transform=transform)

    if is_distributed:
        dist.barrier()

    train_sampler = (
        DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    )

    trainloader = DataLoader(
        trainset,
        batch_size=int(hparams["batch_size"]),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    testloader_gpu = DataLoader(
        testset,
        batch_size=int(hparams["batch_size"]),
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    # For real INT8 eager quant eval (CPU)
    testloader_cpu = DataLoader(
        testset,
        batch_size=int(hparams["batch_size"]),
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
    )

    # ---- Models ----
    teacher = model_registry.create_teacher("vit", num_classes=10).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    model: nn.Module = model_registry.create_student("vit", num_classes=10, qat_wrapper=True).to(device)

    # ---- Losses ----
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=float(hparams["label_smoothing"]))
    criterion_kd = nn.KLDivLoss(reduction="batchmean")
    kd_temp = float(hparams["kd_temp"])
    kd_alpha = float(hparams["kd_alpha"])

    # ---- Optimizer ----
    def make_optimizer(params, lr_scale: float = 1.0):
        return torch.optim.AdamW(
            params,
            lr=float(hparams["lr"]) * lr_scale,
            weight_decay=float(hparams["weight_decay"]),
        )

    optimizer = make_optimizer(model.parameters(), lr_scale=1.0)

    # ---- DDP wrap (initial) ----
    if is_distributed:
        ddp_model: nn.Module = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    else:
        ddp_model = model

    # AMP scaler (new API). We will disable it once QAT starts.
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    qat_enabled = False
    best_quant_acc = 0.0

    epochs = int(hparams["epochs"])
    qat_start_epoch = int(hparams["qat_start_epoch"])
    qbackend = str(hparams.get("qat_backend", "qnnpack"))

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ---- Enable QAT once ----
        if (not qat_enabled) and (epoch >= qat_start_epoch):
            if rank == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Enabling QAT at epoch {epoch}")

            base = _unwrap_ddp(ddp_model)
            base.train()
            base.qconfig = get_default_qat_qconfig(qbackend)
            prepared = prepare_qat(base, inplace=False).to(device)
            prepared.train()

            if is_distributed:
                ddp_model = DDP(prepared, device_ids=[local_rank] if device.type == "cuda" else None)
            else:
                ddp_model = prepared

            optimizer = make_optimizer(ddp_model.parameters(), lr_scale=0.5)
            qat_enabled = True

        ddp_model.train()

        # AMP only before QAT starts
        amp_enabled_this_epoch = bool(args.amp) and (not qat_enabled) and (device.type == "cuda")

        it = trainloader
        if rank == 0:
            it = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} (amp={int(amp_enabled_this_epoch)} qat={int(qat_enabled)})", leave=False)

        running_loss = 0.0
        n_batches = 0

        for images, labels in it:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                teacher_out = teacher(images)

            # autocast only pre-QAT
            with torch.amp.autocast("cuda", enabled=amp_enabled_this_epoch):
                student_out = ddp_model(images)

                loss_ce = criterion_ce(student_out, labels)
                loss_kd = criterion_kd(
                    torch.log_softmax(student_out / kd_temp, dim=1),
                    torch.softmax(teacher_out / kd_temp, dim=1),
                ) * (kd_temp**2)

                loss = kd_alpha * loss_kd + (1.0 - kd_alpha) * loss_ce

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled_this_epoch:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                optimizer.step()

            running_loss += float(loss.detach().item())
            n_batches += 1

        if is_distributed:
            dist.barrier()

        # ---- Validation + checkpoint (rank 0 only) ----
        if rank == 0:
            qat_acc = evaluate_fp32(ddp_model, testloader_gpu, device)

            quant_acc = qat_acc
            is_last = epoch == (epochs - 1)

            if is_last:
                base = _unwrap_ddp(ddp_model)
                base.eval()
                quant_model = convert(base, inplace=False)
                quant_acc = evaluate_quantized_cpu(quant_model, testloader_cpu)

            if quant_acc > best_quant_acc:
                best_quant_acc = quant_acc
                base = _unwrap_ddp(ddp_model)
                torch.save(base.state_dict(), output_dir / "best_qat.pth")
                if is_last:
                    base.eval()
                    torch.save(convert(base, inplace=False).state_dict(), output_dir / "best_converted.pth")

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Epoch {epoch+1:2d}/{epochs} | "
                f"Loss: {running_loss/max(1,n_batches):.4f} | "
                f"QAT Acc: {qat_acc:5.2f}% | Quant Acc: {quant_acc:5.2f}%"
            )

            mlflow.log_metric("train_loss", running_loss / max(1, n_batches), step=epoch)
            mlflow.log_metric("qat_acc", qat_acc, step=epoch)
            if is_last:
                mlflow.log_metric("quant_acc", quant_acc, step=epoch)

        if is_distributed:
            dist.barrier()

    # ---- Final logging/cleanup ----
    if rank == 0:
        print("=" * 70)
        print(f"FINAL TRAINING COMPLETE | Best Quantized Accuracy: {best_quant_acc:.2f}%")
        print(f"Deployment model: {output_dir / 'best_converted.pth'}")
        print("=" * 70)

        mlflow.log_metric("final_quant_acc", best_quant_acc)

        converted = output_dir / "best_converted.pth"
        if converted.exists():
            mlflow.log_artifact(str(converted), "models")
        mlflow.log_artifact(str(args.config), "configs")
        mlflow.end_run()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
