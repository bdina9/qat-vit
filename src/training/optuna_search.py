# project/src/training/optuna_search.py
"""
Optuna hyperparameter search for QAT + distillation (single-process, GPU-only).
Outputs a YAML file with best params (best_params.yaml).

Speed-focused changes:
- Fixed GPU device + fixed batch size (no search over those)
- Build DataLoaders once (reused across trials)
- Limit train/eval batches per epoch during search
- AMP enabled pre-QAT, disabled once QAT starts (fixes FP16 vs fake-quant dtype issues)
- Aggressive Optuna pruning + TPE sampler
- MLflow logging per trial + per epoch
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import mlflow
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

    # fixed (not searched)
    batch_size: int = 1024
    device: str = "cuda"

    # speed knobs (used only for optuna search)
    search_train_batches: int = 200
    search_eval_batches: int = 50
    num_workers: int = 8
    amp: bool = True                  # AMP allowed BEFORE QAT only
    cudnn_benchmark: bool = True

    # mlflow
    # Strongly recommend sqlite to avoid upcoming MLflow filestore deprecation warnings:
    #   sqlite:///mlflow.db
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "clue-vit-qat-optuna"


def _build_loaders(data_root: str, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    return trainloader, testloader


@torch.no_grad()
def _eval_acc_limited(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    correct = 0
    total = 0
    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
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

    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    if cfg.device != "cuda":
        raise ValueError("SearchConfig.device is forced to 'cuda' for this script.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for optuna search (GPU-only).")

    device = torch.device("cuda")

    # DataLoaders built once and reused across trials
    trainloader, testloader = _build_loaders(cfg.data_root, cfg.batch_size, cfg.num_workers)

    # Teacher built once
    teacher = model_registry.create_teacher("vit", num_classes=10).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ---- MLflow setup ----
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    # ---- Optuna study ----
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=0)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 5e-5, 3e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
        kd_temp = trial.suggest_float("kd_temp", 1.5, 6.0)
        kd_alpha = trial.suggest_float("kd_alpha", 0.2, 0.9)
        qat_start_epoch = trial.suggest_int("qat_start_epoch", 0, max(0, cfg.max_epochs - 2))

        epochs = int(cfg.max_epochs)
        max_train_batches = int(cfg.search_train_batches)
        max_eval_batches = int(cfg.search_eval_batches)

        student = model_registry.create_student("vit", num_classes=10, qat_wrapper=True).to(device)

        ce = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
        kd = nn.KLDivLoss(reduction="batchmean")

        opt = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=float(weight_decay))

        # AMP scaler (new API)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp))

        qat_enabled = False
        model: nn.Module = student

        run_name = f"trial_{trial.number:04d}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_param("batch_size", int(cfg.batch_size))
            mlflow.log_param("device", "cuda")
            mlflow.log_param("max_epochs", epochs)
            mlflow.log_param("search_train_batches", max_train_batches)
            mlflow.log_param("search_eval_batches", max_eval_batches)
            mlflow.log_param("num_workers", int(cfg.num_workers))
            mlflow.log_param("amp_pre_qat", int(cfg.amp))

            mlflow.log_param("lr", float(lr))
            mlflow.log_param("weight_decay", float(weight_decay))
            mlflow.log_param("label_smoothing", float(label_smoothing))
            mlflow.log_param("kd_temp", float(kd_temp))
            mlflow.log_param("kd_alpha", float(kd_alpha))
            mlflow.log_param("qat_start_epoch", int(qat_start_epoch))
            mlflow.log_param("qat_backend", "qnnpack")

            best_acc = 0.0
            last_acc = 0.0

            for epoch in range(epochs):
                # Enable QAT once
                if (not qat_enabled) and (epoch >= qat_start_epoch):
                    model.train()
                    backend = "qnnpack"
                    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
                    model = torch.ao.quantization.prepare_qat(model, inplace=False).to(device)
                    model.train()

                    # reduce LR once QAT starts
                    opt = torch.optim.AdamW(model.parameters(), lr=float(lr) * 0.5, weight_decay=float(weight_decay))
                    qat_enabled = True

                # IMPORTANT: disable AMP once QAT is enabled to avoid Half/Float fake-quant mismatch
                amp_enabled_this_epoch = bool(cfg.amp) and (not qat_enabled)

                model.train()

                running_loss = 0.0
                running_ce = 0.0
                running_kd = 0.0
                n_batches = 0

                it = tqdm(
                    trainloader,
                    desc=f"trial {trial.number} ep {epoch+1}/{epochs} (amp={int(amp_enabled_this_epoch)} qat={int(qat_enabled)})",
                    leave=False,
                    total=min(len(trainloader), max_train_batches),
                )

                for bi, (x, y) in enumerate(it):
                    if bi >= max_train_batches:
                        break

                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    with torch.no_grad():
                        t_out = teacher(x)

                    # Use autocast only pre-QAT. Once QAT starts, force full-fp32 path.
                    with torch.amp.autocast("cuda", enabled=amp_enabled_this_epoch):
                        s_out = model(x)

                        loss_ce = ce(s_out, y)
                        loss_kd = kd(
                            torch.log_softmax(s_out / kd_temp, dim=1),
                            torch.softmax(t_out / kd_temp, dim=1),
                        ) * (kd_temp**2)

                        loss = kd_alpha * loss_kd + (1.0 - kd_alpha) * loss_ce

                    opt.zero_grad(set_to_none=True)

                    if amp_enabled_this_epoch:
                        scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()

                    running_loss += float(loss.detach().item())
                    running_ce += float(loss_ce.detach().item())
                    running_kd += float(loss_kd.detach().item())
                    n_batches += 1

                last_acc = _eval_acc_limited(model, testloader, device, max_batches=max_eval_batches)
                best_acc = max(best_acc, last_acc)

                trial.report(last_acc, step=epoch)

                denom = max(1, n_batches)
                mlflow.log_metric("train_loss", running_loss / denom, step=epoch)
                mlflow.log_metric("train_loss_ce", running_ce / denom, step=epoch)
                mlflow.log_metric("train_loss_kd", running_kd / denom, step=epoch)
                mlflow.log_metric("val_acc_limited", float(last_acc), step=epoch)
                mlflow.log_metric("best_val_acc_limited", float(best_acc), step=epoch)
                mlflow.log_metric("qat_enabled", int(qat_enabled), step=epoch)
                mlflow.log_metric("amp_enabled", int(amp_enabled_this_epoch), step=epoch)

                if trial.should_prune():
                    mlflow.set_tag("optuna_state", "PRUNED")
                    raise optuna.TrialPruned()

            mlflow.set_tag("optuna_state", "COMPLETE")
            mlflow.log_metric("final_val_acc_limited", float(last_acc))
            mlflow.log_metric("best_val_acc_limited_final", float(best_acc))

        return float(last_acc)

    study.optimize(objective, n_trials=cfg.n_trials)

    best = dict(study.best_params)
    best["epochs"] = int(cfg.max_epochs)
    best["batch_size"] = int(cfg.batch_size)
    best["qat_backend"] = "qnnpack"

    best_path = out_dir / "best_params.yaml"
    with open(best_path, "w") as f:
        yaml.safe_dump(best, f, sort_keys=False)

    with mlflow.start_run(run_name="optuna_best_summary"):
        mlflow.log_params(best)
        mlflow.log_metric("best_value", float(study.best_value))
        mlflow.log_artifact(str(best_path), artifact_path="configs")

    return {"best_params": best, "best_value": float(study.best_value), "best_params_path": str(best_path)}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Optuna QAT search (single-process, GPU-only)")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--output-dir", type=str, default="./qat_search")
    p.add_argument("--data-root", type=str, default="./data")

    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-amp", action="store_true")

    p.add_argument("--search-train-batches", type=int, default=200)
    p.add_argument("--search-eval-batches", type=int, default=50)

    p.add_argument("--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db")
    p.add_argument("--mlflow-experiment", type=str, default="clue-vit-qat-optuna")

    args = p.parse_args()

    cfg = SearchConfig(
        n_trials=args.trials,
        max_epochs=args.epochs,
        output_dir=args.output_dir,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=(not args.no_amp),
        search_train_batches=args.search_train_batches,
        search_eval_batches=args.search_eval_batches,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
    )

    res = run_optuna_search(cfg)
    print(f"Best value: {res['best_value']:.2f}")
    print(f"Best params written to: {res['best_params_path']}")


if __name__ == "__main__":
    main()
