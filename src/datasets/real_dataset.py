"""
PyTorch dataset for real Ogham inscription images.

Loads preprocessed images and their transcriptions for training TrOCR.

★ Insight ─────────────────────────────────────
Key design decisions:
1. Stone-level organization prevents data leakage
2. Confidence filtering controls training data quality
3. Multiple images per stone are treated as separate samples
4. Preprocessing is cached for efficiency
─────────────────────────────────────────────────
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class RealOghamDataset(Dataset):
    """
    Dataset for real Ogham inscription images.

    Loads images from the processed directory and pairs them with
    transcription annotations. Supports filtering by confidence level
    and split assignment.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        processor: Any,
        tokenizer: Optional[Any] = None,
        transform: Optional[Any] = None,
        max_length: int = 64,
        confidence_filter: Optional[List[str]] = None,
        return_metadata: bool = False,
        mode: str = "ogham",
        curated_dir: Optional[str] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory of the dataset
            split: Data split ("train", "val", or "test")
            processor: TrOCR processor for tokenization
            tokenizer: Optional tokenizer override (use for extended Ogham tokenizer)
            transform: Optional albumentations transforms
            max_length: Maximum sequence length for tokenization
            confidence_filter: List of confidence levels to include
            return_metadata: Whether to return sample metadata
            mode: Output mode - "ogham" for Unicode, "latin" for transliteration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.data_dir = Path(data_dir)
        self.split = split
        self.processor = processor
        self.tokenizer = tokenizer or processor.tokenizer
        self.transform = transform
        self.max_length = max_length
        self.confidence_filter = confidence_filter or ["verified", "probable"]
        self.return_metadata = return_metadata
        self.mode = mode
        self.curated_dir = Path(curated_dir) if curated_dir else None

        # Load curation data if curated_dir is provided
        self.curation = {}
        curation_file = self.data_dir / "processed" / "curation.json"
        if curation_file.exists():
            with open(curation_file) as f:
                self.curation = json.load(f)

        # Load samples
        self.samples = self._load_split()

    def _load_split(self) -> List[Dict]:
        """Load samples for this split."""
        # Load split file
        split_file = self.data_dir / "splits" / f"{self.split}_stones.txt"

        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                f"Run create_splits() first."
            )

        with open(split_file) as f:
            stone_ids = set(line.strip() for line in f if line.strip())

        # Load transcription annotations
        annotations_file = self.data_dir / "processed" / "annotations" / "transcriptions.json"

        if not annotations_file.exists():
            raise FileNotFoundError(
                f"Annotations file not found: {annotations_file}. "
                f"Complete annotations first."
            )

        with open(annotations_file) as f:
            all_annotations = json.load(f)

        # Build sample list
        samples = []

        for stone_id in stone_ids:
            if stone_id not in all_annotations:
                continue

            annotation = all_annotations[stone_id]

            # Filter by confidence
            confidence = annotation.get("confidence", "uncertain")
            if confidence not in self.confidence_filter:
                continue

            transcription = annotation.get("transcription", "")
            if not transcription:
                continue

            # Find images: prefer curated dir, then processed/cropped, then raw
            image_paths = []
            if self.curated_dir and (self.curated_dir / stone_id).exists():
                curated_stone_dir = self.curated_dir / stone_id
                image_paths = list(curated_stone_dir.glob("*.png")) + list(curated_stone_dir.glob("*.jpg"))
            elif (self.data_dir / "processed" / "cropped" / stone_id).exists():
                crop_dir = self.data_dir / "processed" / "cropped" / stone_id
                image_paths = list(crop_dir.glob("*.png")) + list(crop_dir.glob("*.jpg"))
            elif (self.data_dir / "raw" / "images" / stone_id).exists():
                raw_dir = self.data_dir / "raw" / "images" / stone_id
                image_paths = list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpg"))

            if not image_paths:
                continue

            # Create a sample for each image
            for image_path in image_paths:
                # Use curated transcription if available
                curation_key = f"{stone_id}/{image_path.name}"
                curation_entry = self.curation.get(curation_key, {})

                # Skip images marked as 'drop' in curation
                if curation_entry.get("status") == "drop":
                    continue

                # Use edited transcription if available
                sample_transcription = curation_entry.get("transcription", transcription)
                if not sample_transcription:
                    continue

                samples.append({
                    "image_path": str(image_path),
                    "stone_id": stone_id,
                    "transcription": sample_transcription,
                    "confidence": confidence,
                    "is_synthetic": False,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = self.samples[idx]

        # Load image
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required. Install with: pip install opencv-python")

        image = cv2.imread(sample["image_path"])
        if image is None:
            raise ValueError(f"Failed to load image: {sample['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Process for TrOCR
        # If transform already produced tensor, use it directly
        if isinstance(image, np.ndarray):
            pixel_values = self.processor(
                images=image,
                return_tensors="pt",
            ).pixel_values.squeeze(0)
        else:
            # Already a tensor from albumentations
            pixel_values = image

        # Choose target text based on mode
        text = sample["transcription"]
        if self.mode == "latin":
            from ..utils.ogham import OGHAM_TO_LATIN
            text = "".join(OGHAM_TO_LATIN.get(ch, ch) for ch in text)

        # Tokenize using the (possibly extended) tokenizer
        labels = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        result = {
            "pixel_values": pixel_values,
            "labels": labels,
        }

        if self.return_metadata:
            result["text"] = text
            result["stone_id"] = sample["stone_id"]
            result["confidence"] = sample["confidence"]
            result["is_synthetic"] = False

        return result

    def get_sample_weights(self) -> List[float]:
        """
        Get sample weights based on confidence.

        Higher weights for verified transcriptions.
        """
        weights = []
        weight_map = {"verified": 1.0, "probable": 0.8, "uncertain": 0.5}

        for sample in self.samples:
            confidence = sample.get("confidence", "uncertain")
            weights.append(weight_map.get(confidence, 0.5))

        return weights

    def get_stone_ids(self) -> List[str]:
        """Get unique stone IDs in this dataset."""
        return list(set(s["stone_id"] for s in self.samples))


def create_real_dataset(
    data_dir: str,
    split: str,
    processor: Any,
    with_augmentation: bool = True,
    augmentation_severity: str = "medium",
) -> RealOghamDataset:
    """
    Create a real Ogham dataset with standard configuration.

    Args:
        data_dir: Dataset directory
        split: "train", "val", or "test"
        processor: TrOCR processor
        with_augmentation: Apply augmentation (train only)
        augmentation_severity: "light", "medium", or "heavy"

    Returns:
        Configured RealOghamDataset
    """
    transform = None

    if with_augmentation and split == "train":
        try:
            from ..generation.augmentation import get_train_transforms
            transform = get_train_transforms(severity=augmentation_severity)
        except ImportError:
            pass  # No augmentation without albumentations

    return RealOghamDataset(
        data_dir=data_dir,
        split=split,
        processor=processor,
        transform=transform,
    )
