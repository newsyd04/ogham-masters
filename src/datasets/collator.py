"""
Data collator for Ogham OCR batches.

Handles batching and padding for TrOCR training.
"""

from typing import Any, Dict, List

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class OghamDataCollator:
    """
    Custom collator for Ogham OCR batches.

    Handles:
    - Stacking pixel values
    - Padding labels
    - Masking padding tokens for loss computation
    """

    def __init__(
        self,
        processor: Any,
        max_length: int = 32,
        pad_to_multiple_of: int = None,
    ):
        """
        Initialize collator.

        Args:
            processor: TrOCR processor
            max_length: Maximum sequence length
            pad_to_multiple_of: Pad to multiple of this value
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        self.processor = processor
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            Batched tensors ready for model
        """
        # Stack pixel values
        pixel_values = torch.stack([item["pixel_values"] for item in batch])

        # Stack labels
        labels = torch.stack([item["labels"] for item in batch])

        # Replace padding token id with -100 for loss computation
        # This tells CrossEntropyLoss to ignore these positions
        labels = labels.clone()
        labels[labels == self.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


class MetadataCollator(OghamDataCollator):
    """
    Collator that also preserves metadata for debugging/analysis.
    """

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate with metadata preservation."""
        result = super().__call__(batch)

        # Collect metadata if present
        if "text" in batch[0]:
            result["texts"] = [item["text"] for item in batch]

        if "stone_id" in batch[0]:
            result["stone_ids"] = [item["stone_id"] for item in batch]

        if "is_synthetic" in batch[0]:
            result["is_synthetic"] = [item["is_synthetic"] for item in batch]

        return result


def create_collator(processor: Any, with_metadata: bool = False) -> OghamDataCollator:
    """
    Create data collator.

    Args:
        processor: TrOCR processor
        with_metadata: Include metadata in batches

    Returns:
        Configured collator
    """
    if with_metadata:
        return MetadataCollator(processor)
    return OghamDataCollator(processor)
