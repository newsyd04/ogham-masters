"""Retry downloading images for DIAS stones that have metadata but no images.

Of 326 stones discovered, only 134 have downloaded images (198 total).
This script identifies the 192 stones without images and retries the download.

Usage:
    python scripts/retry_dias_images.py [--data-dir ./ogham_dataset]
    python scripts/retry_dias_images.py --dry-run  # list missing stones only
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scrapers.dias_scraper import create_dias_scraper


def find_missing_stones(data_dir: Path) -> list:
    """Find stones in metadata that have no downloaded images."""
    metadata_path = data_dir / "raw" / "metadata" / "stone_metadata.jsonl"
    images_dir = data_dir / "raw" / "images"

    if not metadata_path.exists():
        print(f"Error: No metadata file at {metadata_path}")
        sys.exit(1)

    # Read all stone IDs from metadata
    all_stone_ids = []
    with open(metadata_path) as f:
        for line in f:
            stone = json.loads(line.strip())
            sid = stone.get("stone_id")
            if sid and stone.get("region") == "Ireland":
                all_stone_ids.append(sid)

    # Check which have image directories with actual files
    stones_with_images = set()
    if images_dir.exists():
        for stone_dir in images_dir.iterdir():
            if stone_dir.is_dir():
                image_files = (
                    list(stone_dir.glob("*.jpg"))
                    + list(stone_dir.glob("*.jpeg"))
                    + list(stone_dir.glob("*.png"))
                )
                if image_files:
                    stones_with_images.add(stone_dir.name)

    missing = [sid for sid in all_stone_ids if sid not in stones_with_images]
    return sorted(missing)


def main():
    parser = argparse.ArgumentParser(description="Retry DIAS image downloads")
    parser.add_argument("--data-dir", type=str, default="./ogham_dataset",
                        help="Root dataset directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="List missing stones without downloading")
    parser.add_argument("--rate-limit", type=float, default=3.0,
                        help="Seconds between requests")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    missing = find_missing_stones(data_dir)

    print(f"Found {len(missing)} Irish stones without images")

    if args.dry_run:
        for sid in missing:
            print(f"  {sid}")
        return

    if not missing:
        print("All stones have images — nothing to do.")
        return

    # Create scraper
    scraper = create_dias_scraper(
        output_dir=str(data_dir),
        rate_limit=args.rate_limit,
        user_agent="OghamOCR-Research/1.0 (University research project)",
    )

    start = time.time()
    downloaded = 0
    failed = 0
    still_missing = 0

    for i, stone_id in enumerate(missing):
        print(f"[{i+1}/{len(missing)}] Trying {stone_id}...")
        try:
            downloads = scraper.download_stone_images(stone_id)
            successes = sum(1 for d in downloads if d.success)
            failures = sum(1 for d in downloads if not d.success)

            if successes > 0:
                downloaded += successes
                print(f"  Downloaded {successes} image(s)")
            elif not downloads:
                still_missing += 1
                print(f"  No images available on DIAS")
            else:
                failed += failures
                print(f"  Failed: {failures} image(s)")
        except Exception as e:
            failed += 1
            print(f"  Error: {e}")

    elapsed = time.time() - start

    print(f"\nRetry complete in {elapsed:.0f}s")
    print(f"  New images downloaded: {downloaded}")
    print(f"  Failed downloads: {failed}")
    print(f"  No images on DIAS: {still_missing}")

    # Update summary
    summary_path = data_dir / "raw" / "metadata" / "scrape_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        summary["retry_downloaded"] = downloaded
        summary["retry_failed"] = failed
        summary["retry_no_images"] = still_missing
        summary["images_downloaded"] += downloaded
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Updated {summary_path}")


if __name__ == "__main__":
    main()
