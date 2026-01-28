# project/src/evaluation/__init__.py
"""Evaluation package exports."""

from .evaluator import evaluate_checkpoint, evaluate_model
from .comparator import compare_checkpoints

__all__ = [
    "evaluate_model",
    "evaluate_checkpoint",
    "compare_checkpoints",
]
