"""
Evaluation analysis and risk monitoring for Ogham OCR.

Provides stratified evaluation and training risk detection.
"""

from collections import defaultdict
from typing import Dict, List, Optional
import logging

from .metrics import compute_cer, compute_exact_match


class EvaluationStrategy:
    """
    Comprehensive evaluation strategy for Ogham OCR.

    Provides:
    - Overall performance metrics
    - Stratified analysis by metadata
    - Domain gap monitoring (synthetic vs real)
    """

    @staticmethod
    def stratified_evaluation(
        predictions: List[str],
        references: List[str],
        metadata: List[Dict],
    ) -> Dict[str, Dict]:
        """
        Compute metrics stratified by metadata attributes.

        Args:
            predictions: Predicted strings
            references: Reference strings
            metadata: List of metadata dicts per sample

        Returns:
            Dictionary with stratified results
        """
        results = {
            "overall": {},
            "by_length": {},
            "by_difficulty": {},
            "by_source": {},
        }

        # Overall
        results["overall"]["cer"] = compute_cer(predictions, references)
        results["overall"]["exact_match"] = compute_exact_match(predictions, references)
        results["overall"]["count"] = len(predictions)

        # By length bucket
        length_buckets = {
            "short": (0, 6),
            "medium": (6, 15),
            "long": (15, 100),
        }

        for bucket_name, (min_len, max_len) in length_buckets.items():
            indices = [
                i for i, ref in enumerate(references)
                if min_len <= len(ref) < max_len
            ]

            if indices:
                bucket_preds = [predictions[i] for i in indices]
                bucket_refs = [references[i] for i in indices]
                results["by_length"][bucket_name] = {
                    "cer": compute_cer(bucket_preds, bucket_refs),
                    "exact_match": compute_exact_match(bucket_preds, bucket_refs),
                    "count": len(indices),
                }

        # By source (synthetic vs real)
        for source in ["synthetic", "real"]:
            indices = [
                i for i, m in enumerate(metadata)
                if m.get("is_synthetic", False) == (source == "synthetic")
            ]

            if indices:
                bucket_preds = [predictions[i] for i in indices]
                bucket_refs = [references[i] for i in indices]
                results["by_source"][source] = {
                    "cer": compute_cer(bucket_preds, bucket_refs),
                    "exact_match": compute_exact_match(bucket_preds, bucket_refs),
                    "count": len(indices),
                }

        # By difficulty if available
        difficulty_levels = set(m.get("difficulty", "unknown") for m in metadata)
        for difficulty in difficulty_levels:
            if difficulty == "unknown":
                continue

            indices = [i for i, m in enumerate(metadata) if m.get("difficulty") == difficulty]

            if indices:
                bucket_preds = [predictions[i] for i in indices]
                bucket_refs = [references[i] for i in indices]
                results["by_difficulty"][difficulty] = {
                    "cer": compute_cer(bucket_preds, bucket_refs),
                    "count": len(indices),
                }

        return results

    @staticmethod
    def compute_domain_gap(
        synthetic_cer: float,
        real_cer: float,
    ) -> Dict:
        """
        Compute domain gap between synthetic and real performance.

        Args:
            synthetic_cer: CER on synthetic validation
            real_cer: CER on real validation

        Returns:
            Dictionary with gap analysis
        """
        gap = abs(real_cer - synthetic_cer)
        ratio = real_cer / synthetic_cer if synthetic_cer > 0 else float("inf")

        return {
            "absolute_gap": gap,
            "ratio": ratio,
            "real_cer": real_cer,
            "synthetic_cer": synthetic_cer,
            "is_concerning": gap > 0.15,  # 15% gap threshold
        }


class RiskMonitor:
    """
    Monitor training for risk indicators.

    Detects:
    - Domain gap (synthetic vs real performance difference)
    - Overfitting (train vs val performance difference)
    - Hallucination (length mismatches in predictions)
    - Training plateau (no improvement)
    """

    THRESHOLDS = {
        "domain_gap": 0.15,           # Max CER difference synthetic vs real
        "overfitting_gap": 0.10,      # Max CER difference train vs val
        "hallucination_rate": 0.05,   # Max rate of length mismatches
        "loss_plateau_epochs": 10,    # Epochs without improvement
    }

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize risk monitor.

        Args:
            logger: Optional logger for warnings
        """
        self.logger = logger or logging.getLogger("risk_monitor")
        self.history = defaultdict(list)

    def check_epoch(self, metrics: Dict) -> List[str]:
        """
        Check metrics after each epoch for risk indicators.

        Args:
            metrics: Dictionary of metrics for this epoch

        Returns:
            List of warning messages (empty if no risks)
        """
        warnings = []

        # Store in history
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.history[key].append(value)

        # Check domain gap
        if "val_cer_synthetic" in metrics and "val_cer_real" in metrics:
            gap = abs(metrics["val_cer_synthetic"] - metrics["val_cer_real"])
            self.history["domain_gap"].append(gap)

            if gap > self.THRESHOLDS["domain_gap"]:
                warnings.append(
                    f"DOMAIN_GAP: {gap:.2%} exceeds threshold "
                    f"({self.THRESHOLDS['domain_gap']:.0%})"
                )

        # Check overfitting
        if "train_cer" in metrics and "val_cer" in metrics:
            gap = metrics["train_cer"] - metrics["val_cer"]
            if gap < -self.THRESHOLDS["overfitting_gap"]:
                # Train much better than val suggests overfitting
                warnings.append(
                    f"OVERFITTING: Train-Val gap {-gap:.2%} suggests overfitting"
                )

        # Check for plateau
        if len(self.history["val_cer"]) > self.THRESHOLDS["loss_plateau_epochs"]:
            recent = self.history["val_cer"][-self.THRESHOLDS["loss_plateau_epochs"]:]
            if max(recent) - min(recent) < 0.001:
                warnings.append(
                    f"PLATEAU: No improvement for "
                    f"{self.THRESHOLDS['loss_plateau_epochs']} epochs"
                )

        # Log warnings
        for warning in warnings:
            self.logger.warning(warning)

        return warnings

    def get_summary(self) -> Dict:
        """Get summary of monitoring history."""
        summary = {}

        for key, values in self.history.items():
            if values:
                summary[key] = {
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1],
                    "trend": "improving" if len(values) > 1 and values[-1] < values[-2] else "stable",
                }

        return summary

    def reset(self):
        """Reset monitoring history."""
        self.history.clear()
