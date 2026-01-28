# project/src/training/optuna_search.py
"""
Optuna hyperparameter search for QAT + distillation (single-process by default).
Outputs a YAML file with best params (best_params.yaml).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import optuna
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.models import model_registry


@dataclass
class SearchConfig:
    n_trials: int = 30
    max_epochs: int = 10
    output_dir: str = "./qat_search"
    data_root: str = "./data"
    batch_size_choices = (64, 128, 256)
    device: Optional[str] = None


def _build_loaders(data_root: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    return trainloader, testloader


@torch.no_grad()
def _eval_acc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(1, total)


def run_optuna_search(cfg: SearchConfig) -> Dict:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device(cfg.device)
        if cfg.device
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )

    teacher = model_registry.create_teacher("vit", num_classes=10).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 5e-5, 3e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
        kd_temp = trial.suggest_float("kd_temp", 1.5, 6.0)
        kd_alpha = trial.suggest_float("kd_alpha", 0.2, 0.9)
        qat_start_epoch = trial.suggest_int("qat_start_epoch", 0, max(0, cfg.max_epochs - 2))
        batch_size = trial.suggest_categorical("batch_size", list(cfg.batch_size_choices))
        epochs = cfg.max_epochs

        trainloader, testloader = _build_loaders(cfg.data_root, batch_size)

        student = model_registry.create_student("vit", num_classes=10, qat_wrapper=True).to(device)

        ce = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
        kd = nn.KLDivLoss(reduction="batchmean")
        opt = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=float(weight_decay))

        qat_enabled = False
        model = student

        # lightweight training loop
        for epoch in range(epochs):
            if (not qat_enabled) and (epoch >= qat_start_epoch):
                model.qconfig = torch.ao.quantization.get_default_qat_qconfig("qnnpack")
                model = torch.ao.quantization.prepare_qat(model, inplace=False).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=float(lr) * 0.5, weight_decay=float(weight_decay))
                qat_enabled = True

            model.train()
            it = tqdm(trainloader, desc=f"trial {trial.number} ep {epoch+1}/{epochs}", leave=False)
            for x, y in it:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.no_grad():
                    t_out = teacher(x)

                s_out = model(x)

                loss_ce = ce(s_out, y)
                loss_kd = kd(
                    torch.log_softmax(s_out / kd_temp, dim=1),
                    torch.softmax(t_out / kd_temp, dim=1),
                ) * (kd_temp**2)

                loss = kd_alpha * loss_kd + (1.0 - kd_alpha) * loss_ce

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            acc = _eval_acc(model, testloader, device)
            trial.report(acc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg.n_trials)

    best = dict(study.best_params)
    best["epochs"] = cfg.max_epochs  # align downstream unless overridden
    best.setdefault("qat_backend", "qnnpack")

    best_path = out_dir / "best_params.yaml"
    with open(best_path, "w") as f:
        yaml.safe_dump(best, f, sort_keys=False)

    return {"best_params": best, "best_value": float(study.best_value), "best_params_path": str(best_path)}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Optuna QAT search (single-process)")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--output-dir", type=str, default="./qat_search")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    cfg = SearchConfig(
        n_trials=args.trials,
        max_epochs=args.epochs,
        output_dir=args.output_dir,
        data_root=args.data_root,
        device=args.device,
    )
    res = run_optuna_search(cfg)
    print(f"Best value: {res['best_value']:.2f}")
    print(f"Best params written to: {res['best_params_path']}")


if __name__ == "__main__":
    main()
