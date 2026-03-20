"""Full DIAS scraping run."""

import sys
import logging
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scrapers.dias_scraper import create_dias_scraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dias_run")

OUTPUT_DIR = "./ogham_dataset"

def main():
    start_time = time.time()

    logger.info("Creating DIAS scraper...")
    scraper = create_dias_scraper(
        output_dir=OUTPUT_DIR,
        rate_limit=3.0,
        user_agent="OghamOCR-Research/1.0 (University research project)",
    )

    # Phase 1: Discover and parse all stones
    logger.info("Phase 1: BFS discovery + metadata parsing...")
    stones = scraper.get_stone_listing()
    discovery_time = time.time() - start_time
    logger.info(f"Discovery complete: {len(stones)} stones in {discovery_time:.0f}s")

    # Save stone metadata
    metadata_dir = Path(OUTPUT_DIR) / "raw" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for stone in stones:
        scraper._save_stone_metadata(stone)
    logger.info(f"Saved stone metadata to {metadata_dir / 'stone_metadata.jsonl'}")

    # Phase 2: Download images for each stone
    logger.info("Phase 2: Downloading images...")
    total_images = 0
    total_failures = 0
    stones_with_images = 0

    for i, stone in enumerate(stones):
        logger.info(f"[{i+1}/{len(stones)}] Downloading images for {stone.stone_id}...")
        downloads = scraper.download_stone_images(stone.stone_id)

        successes = sum(1 for d in downloads if d.success)
        failures = sum(1 for d in downloads if not d.success)
        total_images += successes
        total_failures += failures
        if successes > 0:
            stones_with_images += 1

    elapsed = time.time() - start_time

    # Summary
    logger.info("=" * 60)
    logger.info("SCRAPING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Stones discovered:    {len(stones)}")
    logger.info(f"  Stones with images:   {stones_with_images}")
    logger.info(f"  Images downloaded:    {total_images}")
    logger.info(f"  Image failures:       {total_failures}")
    logger.info(f"  Total time:           {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Output directory:     {OUTPUT_DIR}")

    # Write summary JSON
    summary = {
        "stones_discovered": len(stones),
        "stones_with_images": stones_with_images,
        "images_downloaded": total_images,
        "image_failures": total_failures,
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": OUTPUT_DIR,
    }
    summary_path = metadata_dir / "scrape_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Summary saved to:     {summary_path}")


if __name__ == "__main__":
    main()
