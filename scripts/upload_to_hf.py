#!/usr/bin/env python3
"""
Upload sharded Ogham dataset to Hugging Face Hub.

Reads the shard_XX/labels.csv + images layout and pushes as an HF Dataset
with train/validation splits. Images are embedded in parquet files for
fast streaming — no more Drive FUSE bottlenecks.

Usage:
    # From Colab (with Drive mounted):
    python scripts/upload_to_hf.py \
        --train-dir /content/drive/MyDrive/ogham_ocr/datasets/synthetic_200k \
        --val-dir /content/drive/MyDrive/ogham_ocr/datasets/synthetic_val \
        --repo-name yourusername/ogham-synthetic-200k \
        --private

    # Dry run (count samples without uploading):
    python scripts/upload_to_hf.py \
        --train-dir /path/to/synthetic_200k \
        --dry-run
"""

import argparse
import csv
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def collect_samples(data_dir: Path) -> dict:
    """Scan sharded or single-directory layout, return column dict."""
    records = {
        "image": [],
        "ogham_text": [],
        "latin_transliteration": [],
        "difficulty": [],
    }
    skipped = 0

    shard_dirs = sorted(data_dir.glob("shard_*"))
    dirs_to_scan = shard_dirs if shard_dirs else [data_dir]

    for d in dirs_to_scan:
        csv_path = d / "labels.csv"
        if not csv_path.exists():
            log.warning(f"No labels.csv in {d.name}, skipping")
            continue

        shard_count = 0
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Resolve image path
                image_path = d / "images" / row["image_file"]
                if not image_path.exists():
                    image_path = d / row["image_file"]
                if not image_path.exists():
                    skipped += 1
                    continue

                # Filter corrupt/truncated files
                try:
                    if image_path.stat().st_size < 500:
                        skipped += 1
                        continue
                except OSError:
                    skipped += 1
                    continue

                records["image"].append(str(image_path))
                records["ogham_text"].append(row["ogham_text"])
                records["latin_transliteration"].append(row["latin_transliteration"])
                records["difficulty"].append(row.get("difficulty", "unknown"))
                shard_count += 1

        log.info(f"  {d.name}: {shard_count} samples")

    if skipped:
        log.warning(f"Skipped {skipped} missing/corrupt files total")

    return records


def main():
    parser = argparse.ArgumentParser(description="Upload Ogham dataset to HF Hub")
    parser.add_argument("--train-dir", type=str, required=True,
                        help="Path to training data (sharded or single dir)")
    parser.add_argument("--val-dir", type=str, default=None,
                        help="Path to validation data")
    parser.add_argument("--repo-name", type=str, default=None,
                        help="HF Hub repo name (e.g. username/ogham-synthetic-200k)")
    parser.add_argument("--private", action="store_true",
                        help="Make the dataset private")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just count samples, don't upload")
    args = parser.parse_args()

    # Collect training samples
    train_dir = Path(args.train_dir)
    log.info(f"Scanning training data: {train_dir}")
    train_records = collect_samples(train_dir)
    log.info(f"Training: {len(train_records['image'])} samples")

    # Collect validation samples
    val_records = None
    if args.val_dir:
        val_dir = Path(args.val_dir)
        log.info(f"Scanning validation data: {val_dir}")
        val_records = collect_samples(val_dir)
        log.info(f"Validation: {len(val_records['image'])} samples")

    if args.dry_run:
        log.info("Dry run complete — no upload.")
        return

    if not args.repo_name:
        parser.error("--repo-name is required (unless --dry-run)")

    # Build HF datasets
    from datasets import Dataset, DatasetDict, Features, Image, Value

    features = Features({
        "image": Image(),
        "ogham_text": Value("string"),
        "latin_transliteration": Value("string"),
        "difficulty": Value("string"),
    })

    log.info("Building HF Dataset objects...")
    splits = {}
    splits["train"] = Dataset.from_dict(train_records, features=features)

    if val_records:
        splits["validation"] = Dataset.from_dict(val_records, features=features)

    ds = DatasetDict(splits)

    log.info(f"Uploading to {args.repo_name} (private={args.private})...")
    log.info("This may take 30-60 minutes for 185k images. Progress will show below.")
    ds.push_to_hub(args.repo_name, private=args.private)

    log.info(f"Done! Dataset available at: https://huggingface.co/datasets/{args.repo_name}")
    log.info(f"\nTo use in training:")
    log.info(f"  python scripts/train_colab.py --hf-dataset {args.repo_name} ...")


if __name__ == "__main__":
    main()
