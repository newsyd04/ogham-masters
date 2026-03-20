"""
Stone-level data splitting for Ogham OCR.

Ensures all images from the same stone are in the same split
to prevent data leakage.

★ Insight ─────────────────────────────────────
Critical principle: Stone-level splitting
- Multiple images per stone share the same inscription
- Having same inscription in train and test = data leakage
- Stratification ensures balanced splits
─────────────────────────────────────────────────
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class StoneLevelSplitter:
    """
    Split dataset at stone level to prevent data leakage.

    All images from the same stone go to the same split.
    Supports stratification by region, inscription length, etc.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize splitter.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def split(
        self,
        stone_ids: List[str],
        stone_metadata: Optional[Dict[str, Dict]] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by: Optional[str] = "region",
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split stones into train/val/test sets.

        Args:
            stone_ids: List of stone identifiers
            stone_metadata: Optional metadata for stratification
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for test
            stratify_by: Metadata key for stratification (None for random)

        Returns:
            Tuple of (train_stones, val_stones, test_stones)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
            "Ratios must sum to 1"

        if stratify_by and stone_metadata:
            return self._stratified_split(
                stone_ids, stone_metadata,
                train_ratio, val_ratio, stratify_by
            )
        else:
            return self._random_split(
                stone_ids, train_ratio, val_ratio
            )

    def _random_split(
        self,
        stone_ids: List[str],
        train_ratio: float,
        val_ratio: float,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Simple random split."""
        stones = list(stone_ids)
        self.rng.shuffle(stones)

        n = len(stones)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = stones[:n_train]
        val = stones[n_train:n_train + n_val]
        test = stones[n_train + n_val:]

        return train, val, test

    def _stratified_split(
        self,
        stone_ids: List[str],
        stone_metadata: Dict[str, Dict],
        train_ratio: float,
        val_ratio: float,
        stratify_by: str,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Split with stratification by metadata attribute."""
        # Group stones by stratification key
        groups = defaultdict(list)

        for stone_id in stone_ids:
            meta = stone_metadata.get(stone_id, {})
            key = meta.get(stratify_by, "unknown")
            groups[key].append(stone_id)

        train, val, test = [], [], []

        # Split each group proportionally
        for group_name, group_stones in groups.items():
            g_stones = list(group_stones)
            self.rng.shuffle(g_stones)

            n = len(g_stones)
            n_train = max(1, int(n * train_ratio)) if n >= 3 else n
            n_val = max(1, int(n * val_ratio)) if n >= 3 else 0

            train.extend(g_stones[:n_train])
            val.extend(g_stones[n_train:n_train + n_val])
            test.extend(g_stones[n_train + n_val:])

        return train, val, test


def create_splits(
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by: Optional[str] = "region",
    seed: int = 42,
    overwrite: bool = False,
) -> Dict:
    """
    Create and save train/val/test splits for a dataset.

    Args:
        data_dir: Root dataset directory
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for test
        stratify_by: Metadata key for stratification
        seed: Random seed
        overwrite: Overwrite existing splits

    Returns:
        Split statistics
    """
    data_dir = Path(data_dir)
    splits_dir = data_dir / "splits"

    # Check for existing splits
    if splits_dir.exists() and not overwrite:
        existing = list(splits_dir.glob("*_stones.txt"))
        if existing:
            raise FileExistsError(
                f"Splits already exist in {splits_dir}. "
                f"Use overwrite=True to replace."
            )

    splits_dir.mkdir(parents=True, exist_ok=True)

    # Get stone IDs from images directory
    images_dir = data_dir / "raw" / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    stone_ids = [d.name for d in images_dir.iterdir() if d.is_dir()]

    if not stone_ids:
        raise ValueError("No stones found in images directory")

    # Load stone metadata if available
    stone_metadata = {}
    metadata_file = data_dir / "raw" / "metadata" / "stone_metadata.jsonl"

    if metadata_file.exists():
        with open(metadata_file) as f:
            for line in f:
                meta = json.loads(line)
                stone_metadata[meta["stone_id"]] = meta

    # Create splits
    splitter = StoneLevelSplitter(seed=seed)
    train_stones, val_stones, test_stones = splitter.split(
        stone_ids=stone_ids,
        stone_metadata=stone_metadata if stone_metadata else None,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify_by=stratify_by if stone_metadata else None,
    )

    # Save split files
    for split_name, split_stones in [
        ("train", train_stones),
        ("val", val_stones),
        ("test", test_stones),
    ]:
        split_file = splits_dir / f"{split_name}_stones.txt"
        with open(split_file, "w") as f:
            for stone_id in sorted(split_stones):
                f.write(f"{stone_id}\n")

    # Save split metadata
    metadata = {
        "seed": seed,
        "stratify_by": stratify_by,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "stone_counts": {
            "train": len(train_stones),
            "val": len(val_stones),
            "test": len(test_stones),
            "total": len(stone_ids),
        },
    }

    with open(splits_dir / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_splits(data_dir: str) -> Dict[str, List[str]]:
    """
    Load existing splits from disk.

    Args:
        data_dir: Dataset directory

    Returns:
        Dictionary mapping split name to stone IDs
    """
    splits_dir = Path(data_dir) / "splits"

    if not splits_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")

    splits = {}
    for split_name in ["train", "val", "test"]:
        split_file = splits_dir / f"{split_name}_stones.txt"
        if split_file.exists():
            with open(split_file) as f:
                splits[split_name] = [line.strip() for line in f if line.strip()]

    return splits
