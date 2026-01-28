# project/src/evaluation/comparator.py
"""
Compare multiple checkpoints and print a compact report.
Typical use:
- Compare teacher baseline vs student QAT vs quantized deployable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

from src.evaluation.evaluator import evaluate_checkpoint


@dataclass(frozen=True)
class CompareItem:
    name: str
    model: str
    ckpt: Union[str, Path]
    qat_wrapper: bool = False


def compare_checkpoints(
    items: Sequence[CompareItem],
    batch_size: int = 256,
    data_root: str = "./data",
    device: Optional[str] = None,
) -> List[tuple]:
    results = []
    for it in items:
        acc = evaluate_checkpoint(
            model_name=it.model,
            checkpoint_path=it.ckpt,
            qat_wrapper=it.qat_wrapper,
            batch_size=batch_size,
            data_root=data_root,
            device=device,
        )
        results.append((it.name, float(acc)))
    return results


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Compare checkpoints")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--device", type=str, default=None)

    # simple fixed conventions
    p.add_argument("--teacher-ckpt", type=str, default="", help="Optional: teacher checkpoint path")
    p.add_argument("--qat-ckpt", type=str, default="./qat_search/best_qat.pth")
    p.add_argument("--quant-ckpt", type=str, default="./qat_search/best_converted.pth")
    args = p.parse_args()

    items: List[CompareItem] = []
    if args.teacher_ckpt:
        items.append(CompareItem("teacher", "vit_base_patch16_224_teacher", args.teacher_ckpt, False))

    items.append(CompareItem("student_qat", "vit_small_patch16_224_student", args.qat_ckpt, True))
    items.append(CompareItem("student_quant", "vit_small_patch16_224_student", args.quant_ckpt, False))

    results = compare_checkpoints(
        items,
        batch_size=args.batch_size,
        data_root=args.data_root,
        device=args.device,
    )

    print("\nRESULTS")
    print("-" * 50)
    for name, acc in results:
        print(f"{name:14s} : {acc:6.2f}%")
    print("-" * 50)


if __name__ == "__main__":
    main()
