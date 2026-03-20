"""Tests for curriculum learning and mixed dataset."""

import pytest
from unittest.mock import MagicMock

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _make_dummy_dataset(size, is_synthetic=False):
    """Create a minimal mock dataset with proper __len__ and __getitem__."""
    sample = {
        "pixel_values": torch.randn(3, 384, 384) if TORCH_AVAILABLE else None,
        "labels": torch.tensor([1, 2, 0]) if TORCH_AVAILABLE else None,
        "is_synthetic": is_synthetic,
    }

    class _DummyDataset(Dataset):
        def __init__(self):
            self.getitem_calls = []

        def __len__(self):
            return size

        def __getitem__(self, idx):
            self.getitem_calls.append(idx)
            return sample

    return _DummyDataset()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
class TestCurriculumScheduler:
    """Tests for CurriculumScheduler schedule generation."""

    def test_default_schedule_has_expected_epochs(self):
        """Default schedule should have entries at epochs 0, 10, 20, 35."""
        from src.datasets.mixed_dataset import CurriculumScheduler

        schedule = CurriculumScheduler.get_default_schedule()
        assert 0 in schedule
        assert 10 in schedule
        assert 20 in schedule
        assert 35 in schedule

    def test_default_schedule_decreasing_synthetic_ratio(self):
        """Synthetic ratio should decrease over time in default schedule."""
        from src.datasets.mixed_dataset import CurriculumScheduler

        schedule = CurriculumScheduler.get_default_schedule()
        ratios = [schedule[e]["synthetic_ratio"] for e in sorted(schedule.keys())]
        # Each ratio should be <= previous
        for i in range(1, len(ratios)):
            assert ratios[i] <= ratios[i - 1], \
                f"Ratio at index {i} ({ratios[i]}) > ratio at index {i-1} ({ratios[i-1]})"

    def test_synthetic_only_schedule_all_1(self):
        """Synthetic-only schedule should have ratio 1.0 throughout."""
        from src.datasets.mixed_dataset import CurriculumScheduler

        schedule = CurriculumScheduler.get_synthetic_only_schedule()
        for epoch, config in schedule.items():
            assert config["synthetic_ratio"] == 1.0

    def test_custom_schedule(self):
        """Custom schedule should accept arbitrary phases."""
        from src.datasets.mixed_dataset import CurriculumScheduler

        phases = [
            {"epoch": 0, "synthetic_ratio": 0.9, "difficulty": "easy"},
            {"epoch": 5, "synthetic_ratio": 0.5, "difficulty": "hard"},
        ]
        schedule = CurriculumScheduler.get_custom_schedule(phases)
        assert schedule[0]["synthetic_ratio"] == 0.9
        assert schedule[5]["difficulty"] == "hard"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
class TestMixedOghamDataset:
    """Tests for MixedOghamDataset sampling logic."""

    def test_length_is_sum(self):
        """Total length should be sum of real + synthetic."""
        from src.datasets.mixed_dataset import MixedOghamDataset

        real = _make_dummy_dataset(10)
        synth = _make_dummy_dataset(90)
        mixed = MixedOghamDataset(real, synth, synthetic_ratio=0.8)

        assert len(mixed) == 100

    def test_set_epoch_updates_ratio(self):
        """set_epoch should update synthetic_ratio when schedule triggers."""
        from src.datasets.mixed_dataset import MixedOghamDataset

        schedule = {
            0: {"synthetic_ratio": 0.9},
            5: {"synthetic_ratio": 0.5},
        }
        real = _make_dummy_dataset(10)
        synth = _make_dummy_dataset(90)
        mixed = MixedOghamDataset(real, synth, synthetic_ratio=0.9,
                                   curriculum_schedule=schedule)

        assert mixed.synthetic_ratio == 0.9
        mixed.set_epoch(5)
        assert mixed.synthetic_ratio == 0.5

    def test_weighted_sampler_returns_correct_type(self):
        """get_weighted_sampler should return a WeightedRandomSampler."""
        from src.datasets.mixed_dataset import MixedOghamDataset
        from torch.utils.data import WeightedRandomSampler

        real = _make_dummy_dataset(10)
        synth = _make_dummy_dataset(90)
        mixed = MixedOghamDataset(real, synth, synthetic_ratio=0.8)

        sampler = mixed.get_weighted_sampler()
        assert isinstance(sampler, WeightedRandomSampler)

    def test_getitem_routes_correctly(self):
        """Indices < n_real should go to real, rest to synthetic."""
        from src.datasets.mixed_dataset import MixedOghamDataset

        real = _make_dummy_dataset(10, is_synthetic=False)
        synth = _make_dummy_dataset(20, is_synthetic=True)
        mixed = MixedOghamDataset(real, synth)

        # Access real range
        mixed[0]
        assert 0 in real.getitem_calls

        # Access synthetic range (offset by n_real)
        mixed[15]
        assert 5 in synth.getitem_calls  # 15 - 10 = 5


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_defaults(self):
        """Config should have sensible defaults."""
        from src.training.trainer import TrainingConfig

        config = TrainingConfig()
        assert config.freeze_encoder_epochs == 5
        assert config.use_amp is True
        assert config.batch_size == 16

    def test_to_dict_includes_freeze_epochs(self):
        """to_dict should include freeze_encoder_epochs."""
        from src.training.trainer import TrainingConfig

        config = TrainingConfig(freeze_encoder_epochs=10)
        d = config.to_dict()
        assert d["freeze_encoder_epochs"] == 10
