# project/src/evaluation/evaluator.py
"""
Evaluation helpers for ViT teacher/student/QAT models.
- Handles QATWrapper checkpoints (best_qat.pth) and quantized checkpoints (best_converted.pth)
- Uses CIFAR-10 with ImageNet normalization (ViT default).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models import model_registry


def build_cifar10_loaders(
    data_root: str = "./data",
    batch_size: int = 256,
    num_workers: int = 4,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    return DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )


@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
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


def evaluate_checkpoint(
    model_name: str,
    checkpoint_path: Union[str, Path],
    qat_wrapper: bool = False,
    num_classes: int = 10,
    batch_size: int = 256,
    data_root: str = "./data",
    device: Optional[str] = None,
) -> float:
    """
    Load a model from the registry and evaluate a checkpoint.
    Args:
      model_name: registry name (e.g. vit_small_patch16_224_student)
      checkpoint_path: .pth file
      qat_wrapper: if True, create model with QATWrapper and load state_dict into wrapper
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    dev = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    model = model_registry.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        checkpoint_path=None,
        qat_wrapper=qat_wrapper,
    )

    state = torch.load(ckpt_path, map_location="cpu")

    # If loading into wrapper, try direct load; otherwise allow mismatch
    if qat_wrapper:
        model.load_state_dict(state, strict=False)
    else:
        # For non-wrapped model, the checkpoint may have wrapper keys stripped in create_student()
        model.load_state_dict(state, strict=False)

    model = model.to(dev)

    loader = build_cifar10_loaders(data_root=data_root, batch_size=batch_size)
    return evaluate_model(model, loader, dev)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Evaluate a checkpoint on CIFAR-10")
    p.add_argument("--model", required=True, help="Registry model name (e.g. vit_small_patch16_224_student)")
    p.add_argument("--ckpt", required=True, help="Pat_
