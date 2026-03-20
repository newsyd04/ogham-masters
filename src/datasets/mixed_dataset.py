"""
Mixed dataset combining real and synthetic Ogham data.

Supports dynamic ratio adjustment for curriculum learning.

★ Insight ─────────────────────────────────────
Curriculum learning strategy:
1. Start with mostly synthetic (easy, clean)
2. Gradually increase real data ratio
3. Progress from easy to hard synthetic
4. Final phase: balanced real/synthetic mix
─────────────────────────────────────────────────
"""

from typing import Any, Dict, List, Optional

try:
    import torch
    from torch.utils.data import Dataset, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


class MixedOghamDataset(Dataset):
    """
    Dataset mixing real and synthetic Ogham data.

    Supports:
    - Controllable real/synthetic ratio
    - Dynamic ratio adjustment for curriculum learning
    - Weighted sampling based on ratio
    """

    def __init__(
        self,
        real_dataset: Dataset,
        synthetic_dataset: Dataset,
        synthetic_ratio: float = 0.8,
        curriculum_schedule: Optional[Dict[int, Dict]] = None,
    ):
        """
        Initialize mixed dataset.

        Args:
            real_dataset: Dataset of real Ogham images
            synthetic_dataset: Dataset of synthetic images
            synthetic_ratio: Proportion of synthetic samples (0-1)
            curriculum_schedule: Optional schedule mapping epoch -> config
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.synthetic_ratio = synthetic_ratio
        self.curriculum_schedule = curriculum_schedule or {}

        self._current_epoch = 0
        self._update_sampling()

    def _update_sampling(self):
        """Update sampling weights based on current ratio."""
        n_real = len(self.real_dataset)
        n_synthetic = len(self.synthetic_dataset)

        # Calculate per-sample weights to achieve desired ratio
        if n_real > 0 and self.synthetic_ratio < 1.0:
            real_weight = (1 - self.synthetic_ratio) / n_real
        else:
            real_weight = 0

        if n_synthetic > 0 and self.synthetic_ratio > 0:
            synthetic_weight = self.synthetic_ratio / n_synthetic
        else:
            synthetic_weight = 0

        self.weights = (
            [real_weight] * n_real +
            [synthetic_weight] * n_synthetic
        )

        self.total_samples = n_real + n_synthetic
        self._n_real = n_real

    def set_epoch(self, epoch: int):
        """
        Update configuration for curriculum learning.

        Args:
            epoch: Current training epoch
        """
        self._current_epoch = epoch

        if epoch in self.curriculum_schedule:
            schedule = self.curriculum_schedule[epoch]

            # Update ratio
            if "synthetic_ratio" in schedule:
                self.synthetic_ratio = schedule["synthetic_ratio"]
                self._update_sampling()

            # Update difficulty
            if "difficulty" in schedule:
                difficulty = schedule["difficulty"]
                if hasattr(self.synthetic_dataset, "set_difficulty"):
                    self.synthetic_dataset.set_difficulty(difficulty)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index."""
        if idx < self._n_real:
            return self.real_dataset[idx]
        else:
            return self.synthetic_dataset[idx - self._n_real]

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Get sampler that respects synthetic/real ratio.

        Returns:
            WeightedRandomSampler configured for current ratio
        """
        return WeightedRandomSampler(
            weights=self.weights,
            num_samples=self.total_samples,
            replacement=True,
        )

    def get_current_config(self) -> Dict:
        """Get current configuration."""
        return {
            "epoch": self._current_epoch,
            "synthetic_ratio": self.synthetic_ratio,
            "n_real": self._n_real,
            "n_synthetic": len(self.synthetic_dataset),
            "total": self.total_samples,
        }


class CurriculumScheduler:
    """
    Curriculum learning scheduler for Ogham OCR training.

    Provides predefined schedules that progressively:
    1. Increase difficulty of synthetic data
    2. Increase proportion of real data
    """

    @staticmethod
    def get_default_schedule(total_epochs: int = 50) -> Dict[int, Dict]:
        """
        Get default curriculum schedule.

        Phases:
        - 0-10: Easy synthetic, minimal real
        - 10-20: Medium synthetic, introduce real
        - 20-35: Medium synthetic, more real
        - 35-50: Hard synthetic, balanced
        """
        return {
            0: {
                "synthetic_ratio": 0.95,
                "difficulty": "easy",
                "description": "Bootstrap with easy synthetic",
            },
            10: {
                "synthetic_ratio": 0.85,
                "difficulty": "medium",
                "description": "Medium difficulty, 15% real",
            },
            20: {
                "synthetic_ratio": 0.70,
                "difficulty": "medium",
                "description": "70% synthetic, 30% real",
            },
            35: {
                "synthetic_ratio": 0.50,
                "difficulty": "hard",
                "description": "Hard synthetic, balanced",
            },
        }

    @staticmethod
    def get_aggressive_real_schedule(total_epochs: int = 50) -> Dict[int, Dict]:
        """
        Schedule that prioritizes real data earlier.

        Use when real data quality is high.
        """
        return {
            0: {"synthetic_ratio": 0.90, "difficulty": "easy"},
            5: {"synthetic_ratio": 0.70, "difficulty": "medium"},
            15: {"synthetic_ratio": 0.50, "difficulty": "medium"},
            25: {"synthetic_ratio": 0.30, "difficulty": "hard"},
            40: {"synthetic_ratio": 0.20, "difficulty": "hard"},
        }

    @staticmethod
    def get_synthetic_only_schedule(total_epochs: int = 50) -> Dict[int, Dict]:
        """
        Schedule using only synthetic data.

        Use for initial experiments or when real data unavailable.
        """
        return {
            0: {"synthetic_ratio": 1.0, "difficulty": "easy"},
            15: {"synthetic_ratio": 1.0, "difficulty": "medium"},
            30: {"synthetic_ratio": 1.0, "difficulty": "hard"},
        }

    @staticmethod
    def get_custom_schedule(
        phases: List[Dict],
    ) -> Dict[int, Dict]:
        """
        Create custom schedule from phase definitions.

        Args:
            phases: List of {"epoch": int, "synthetic_ratio": float, "difficulty": str}

        Returns:
            Schedule dictionary
        """
        return {p["epoch"]: {k: v for k, v in p.items() if k != "epoch"} for p in phases}


def create_mixed_dataset(
    real_dataset: Dataset,
    synthetic_dataset: Dataset,
    schedule_type: str = "default",
    total_epochs: int = 50,
) -> MixedOghamDataset:
    """
    Create mixed dataset with curriculum schedule.

    Args:
        real_dataset: Real Ogham dataset
        synthetic_dataset: Synthetic dataset
        schedule_type: "default", "aggressive_real", or "synthetic_only"
        total_epochs: Total training epochs

    Returns:
        Configured MixedOghamDataset
    """
    schedules = {
        "default": CurriculumScheduler.get_default_schedule,
        "aggressive_real": CurriculumScheduler.get_aggressive_real_schedule,
        "synthetic_only": CurriculumScheduler.get_synthetic_only_schedule,
    }

    schedule_fn = schedules.get(schedule_type, schedules["default"])
    schedule = schedule_fn(total_epochs)

    # Get initial ratio from schedule
    initial_ratio = schedule.get(0, {}).get("synthetic_ratio", 0.8)

    return MixedOghamDataset(
        real_dataset=real_dataset,
        synthetic_dataset=synthetic_dataset,
        synthetic_ratio=initial_ratio,
        curriculum_schedule=schedule,
    )
