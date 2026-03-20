"""Evaluation metrics and analysis for Ogham OCR."""

from .metrics import compute_cer, compute_wer, compute_exact_match
from .analysis import EvaluationStrategy, RiskMonitor
from .logger import ExperimentLogger

__all__ = [
    "compute_cer",
    "compute_wer",
    "compute_exact_match",
    "EvaluationStrategy",
    "RiskMonitor",
    "ExperimentLogger",
]
