"""Tests for data collators."""

import pytest
from unittest.mock import MagicMock

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
class TestOghamDataCollator:
    """Tests for OghamDataCollator."""

    def _make_processor_mock(self, pad_token_id=0):
        processor = MagicMock()
        processor.tokenizer.pad_token_id = pad_token_id
        return processor

    def _make_batch(self, n=2, seq_len=8, pad_token_id=0):
        """Create a synthetic batch of samples."""
        batch = []
        for i in range(n):
            labels = torch.tensor([1, 2, 3, pad_token_id, pad_token_id, pad_token_id, pad_token_id, pad_token_id])
            batch.append({
                "pixel_values": torch.randn(3, 384, 384),
                "labels": labels,
            })
        return batch

    def test_stacks_pixel_values(self):
        """Collator should stack pixel_values into a batch tensor."""
        from src.datasets.collator import OghamDataCollator

        proc = self._make_processor_mock()
        collator = OghamDataCollator(proc)
        batch = self._make_batch(n=3)
        result = collator(batch)

        assert result["pixel_values"].shape == (3, 3, 384, 384)

    def test_replaces_pad_with_minus_100(self):
        """Padding tokens in labels should become -100 for loss masking."""
        from src.datasets.collator import OghamDataCollator

        proc = self._make_processor_mock(pad_token_id=0)
        collator = OghamDataCollator(proc)
        batch = self._make_batch(n=1)
        result = collator(batch)

        labels = result["labels"][0]
        # Pad positions (value 0) should now be -100
        assert (labels[3:] == -100).all()
        # Non-pad positions should be unchanged
        assert labels[0].item() == 1
        assert labels[1].item() == 2
        assert labels[2].item() == 3

    def test_drops_metadata_keys(self):
        """OghamDataCollator should NOT include metadata."""
        from src.datasets.collator import OghamDataCollator

        proc = self._make_processor_mock()
        collator = OghamDataCollator(proc)
        batch = self._make_batch(n=1)
        batch[0]["text"] = "ᚋᚐᚊᚔ"
        batch[0]["is_synthetic"] = True
        result = collator(batch)

        assert "text" not in result
        assert "texts" not in result
        assert "is_synthetic" not in result


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
class TestMetadataCollator:
    """Tests for MetadataCollator."""

    def _make_processor_mock(self, pad_token_id=0):
        processor = MagicMock()
        processor.tokenizer.pad_token_id = pad_token_id
        return processor

    def test_preserves_is_synthetic(self):
        """MetadataCollator should preserve is_synthetic metadata."""
        from src.datasets.collator import MetadataCollator

        proc = self._make_processor_mock()
        collator = MetadataCollator(proc)
        batch = [
            {
                "pixel_values": torch.randn(3, 384, 384),
                "labels": torch.tensor([1, 2, 0, 0]),
                "is_synthetic": True,
                "text": "ᚋᚐ",
            },
            {
                "pixel_values": torch.randn(3, 384, 384),
                "labels": torch.tensor([3, 4, 0, 0]),
                "is_synthetic": False,
                "text": "ᚊᚔ",
            },
        ]
        result = collator(batch)

        assert "is_synthetic" in result
        assert result["is_synthetic"] == [True, False]
        assert result["texts"] == ["ᚋᚐ", "ᚊᚔ"]

    def test_preserves_stone_ids(self):
        """MetadataCollator should preserve stone_id metadata."""
        from src.datasets.collator import MetadataCollator

        proc = self._make_processor_mock()
        collator = MetadataCollator(proc)
        batch = [
            {
                "pixel_values": torch.randn(3, 384, 384),
                "labels": torch.tensor([1, 2, 0, 0]),
                "stone_id": "I-COR-001",
            },
        ]
        result = collator(batch)

        assert result["stone_ids"] == ["I-COR-001"]
