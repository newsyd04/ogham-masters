"""
PyTorch dataset for on-the-fly synthetic Ogham generation.

Generates training samples procedurally without requiring disk storage.

★ Insight ─────────────────────────────────────
On-the-fly generation advantages:
1. No disk storage for millions of images
2. Infinite effective variation
3. Deterministic with seeds for reproducibility
4. Adjustable difficulty for curriculum learning
─────────────────────────────────────────────────
"""

from typing import Any, Dict, List, Optional
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


class SyntheticOghamDataset(Dataset):
    """
    On-the-fly synthetic Ogham dataset for TrOCR training.

    Generates images procedurally using the sequence sampler and renderer.
    Each sample is deterministically generated from its index, ensuring
    reproducibility while avoiding disk storage.
    """

    def __init__(
        self,
        size: int,
        font_paths: List[str],
        tokenizer: Any,
        processor: Optional[Any] = None,
        image_height: int = 384,
        difficulty: str = "medium",
        seed: int = 42,
        transform: Optional[Any] = None,
        max_length: int = 32,
        return_metadata: bool = False,
    ):
        """
        Initialize synthetic dataset.

        Args:
            size: Number of samples in the dataset
            font_paths: Paths to Ogham-compatible fonts
            tokenizer: TrOCR tokenizer
            processor: TrOCR processor for image normalization (recommended)
            image_height: Height of generated images
            difficulty: "easy", "medium", or "hard"
            seed: Base random seed
            transform: Optional augmentation transforms
            max_length: Maximum sequence length
            return_metadata: Whether to return sample metadata
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        self.size = size
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_height = image_height
        self.difficulty = difficulty
        self.base_seed = seed
        self.transform = transform
        self.max_length = max_length
        self.return_metadata = return_metadata

        # Initialize generators
        from ..generation.sequence_sampler import DifficultyAwareSequenceSampler
        from ..generation.renderer import OghamRenderer

        self.sequence_sampler = DifficultyAwareSequenceSampler(
            difficulty=difficulty,
            seed=seed,
        )

        self.renderer = OghamRenderer(
            font_paths=font_paths,
            image_height=image_height,
            seed=seed,
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Generate a single sample on-the-fly.

        Deterministic: same idx always produces same sample.
        """
        # Seed based on index for reproducibility
        sample_seed = self.base_seed + idx
        self.sequence_sampler.set_seed(sample_seed)
        self.renderer.set_seed(sample_seed)

        # Generate text
        text = self.sequence_sampler.sample()

        # Render image
        image, render_info = self.renderer.render(text)

        # Apply augmentation if provided
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Convert to tensor using TrOCR processor (matches real dataset pipeline)
        if isinstance(image, np.ndarray) and self.processor is not None:
            pixel_values = self.processor(
                images=image,
                return_tensors="pt",
            ).pixel_values.squeeze(0)
        elif isinstance(image, np.ndarray):
            # Fallback: manual normalization (no processor available)
            image = image.astype(np.float32) / 255.0
            pixel_values = torch.from_numpy(image).permute(2, 0, 1)
        else:
            pixel_values = image

        # Tokenize text
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
            result["stone_id"] = f"synthetic_{idx}"
            result["is_synthetic"] = True
            result["render_info"] = render_info

        return result

    def set_difficulty(self, difficulty: str):
        """
        Change difficulty level (for curriculum learning).

        Args:
            difficulty: "easy", "medium", or "hard"
        """
        self.difficulty = difficulty
        self.sequence_sampler.set_difficulty(difficulty)

        # Also adjust transform severity if using augmentation
        if self.transform is not None:
            try:
                from ..generation.augmentation import get_train_transforms
                self.transform = get_train_transforms(severity=difficulty)
            except ImportError:
                pass


class LazySyntheticDataset(Dataset):
    """
    Memory-efficient synthetic dataset that doesn't store font objects.

    Initializes renderer lazily on first access, making it safe to
    use with multiprocess DataLoader.
    """

    def __init__(
        self,
        size: int,
        font_dir: str,
        tokenizer: Any,
        processor: Optional[Any] = None,
        image_height: int = 384,
        difficulty: str = "medium",
        seed: int = 42,
        max_length: int = 64,
        mode: str = "ogham",
    ):
        """
        Initialize lazy synthetic dataset.

        Args:
            size: Number of samples
            font_dir: Directory containing Ogham fonts
            tokenizer: TrOCR tokenizer
            processor: TrOCR processor for image normalization (recommended)
            image_height: Image height
            difficulty: Difficulty level
            seed: Random seed
            max_length: Max sequence length
            mode: Output mode - "ogham" for Unicode, "latin" for transliteration
        """
        self.size = size
        self.font_dir = font_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_height = image_height
        self.difficulty = difficulty
        self.base_seed = seed
        self.max_length = max_length
        self.mode = mode

        # Lazy initialization
        self._renderer = None
        self._sampler = None

    def _init_generators(self):
        """Initialize generators on first use."""
        if self._sampler is None:
            from ..generation.sequence_sampler import DifficultyAwareSequenceSampler
            self._sampler = DifficultyAwareSequenceSampler(
                difficulty=self.difficulty,
                seed=self.base_seed,
            )

        if self._renderer is None:
            from ..generation.renderer import create_renderer
            self._renderer = create_renderer(
                font_dir=self.font_dir,
                image_height=self.image_height,
                seed=self.base_seed,
            )

    def __len__(self) -> int:
        return self.size

    def set_difficulty(self, difficulty: str):
        """
        Change difficulty level (for curriculum learning).

        Args:
            difficulty: "easy", "medium", or "hard"
        """
        self.difficulty = difficulty
        if self._sampler is not None:
            self._sampler.set_difficulty(difficulty)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate sample lazily."""
        self._init_generators()

        # Set deterministic seed
        sample_seed = self.base_seed + idx
        self._sampler.set_seed(sample_seed)
        self._renderer.set_seed(sample_seed)

        # Generate
        text = self._sampler.sample()
        image, _ = self._renderer.render(text)

        # Choose target text based on mode
        if self.mode == "latin":
            from ..utils.ogham import OGHAM_TO_LATIN
            target_text = "".join(OGHAM_TO_LATIN.get(ch, ch) for ch in text)
        else:
            target_text = text

        # Convert to tensor using TrOCR processor (matches real dataset pipeline)
        if self.processor is not None:
            pixel_values = self.processor(
                images=image,
                return_tensors="pt",
            ).pixel_values.squeeze(0)
        else:
            # Fallback: manual normalization (no processor available)
            image = image.astype(np.float32) / 255.0
            pixel_values = torch.from_numpy(image).permute(2, 0, 1)

        labels = self.tokenizer(
            target_text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": target_text,
            "is_synthetic": True,
        }


def create_synthetic_dataset(
    size: int,
    font_dir: str,
    tokenizer: Any,
    processor: Optional[Any] = None,
    difficulty: str = "medium",
    seed: int = 42,
    lazy: bool = True,
    mode: str = "ogham",
) -> Dataset:
    """
    Create synthetic Ogham dataset.

    Args:
        size: Number of samples
        font_dir: Directory with Ogham fonts
        tokenizer: TrOCR tokenizer
        processor: TrOCR processor for image normalization (recommended)
        difficulty: "easy", "medium", or "hard"
        seed: Random seed
        lazy: Use lazy initialization (recommended for DataLoader)
        mode: Output mode - "ogham" for Unicode, "latin" for transliteration

    Returns:
        Synthetic dataset
    """
    if lazy:
        return LazySyntheticDataset(
            size=size,
            font_dir=font_dir,
            tokenizer=tokenizer,
            processor=processor,
            difficulty=difficulty,
            seed=seed,
            mode=mode,
        )

    from pathlib import Path
    font_paths = [
        str(p) for p in Path(font_dir).glob("*.ttf")
    ] + [
        str(p) for p in Path(font_dir).glob("*.otf")
    ]

    return SyntheticOghamDataset(
        size=size,
        font_paths=font_paths,
        tokenizer=tokenizer,
        processor=processor,
        difficulty=difficulty,
        seed=seed,
    )
