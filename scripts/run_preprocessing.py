"""Run preprocessing pipeline on all downloaded Ogham images.

Processes raw stone photographs through:
1. Auto-orientation (aspect-ratio heuristic)
2. Multi-scale Sobel inscription cropping (adaptive thresholds)
3. Greyscale conversion
4. Bilateral denoising (stone texture suppression)
5. Retinex lighting normalisation (shadow removal)
6. Weathering-adaptive CLAHE contrast enhancement
7. Unsharp mask sharpening (faint stroke enhancement)
8. Resizing for TrOCR input

Usage:
    python scripts/run_preprocessing.py [--data-dir ./ogham_dataset]
    python scripts/run_preprocessing.py --force  # re-process all images
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.preprocessor import OghamPreprocessor, PreprocessConfig


def load_stone_metadata(data_dir: Path) -> dict:
    """Load stone metadata keyed by stone_id for weathering severity lookup."""
    metadata_path = data_dir / "raw" / "metadata" / "stone_metadata.jsonl"
    stone_meta = {}

    if not metadata_path.exists():
        print(f"Warning: No metadata file at {metadata_path}, skipping adaptive CLAHE")
        return stone_meta

    with open(metadata_path) as f:
        for line in f:
            stone = json.loads(line.strip())
            sid = stone.get("stone_id")
            if sid:
                stone_meta[sid] = stone

    print(f"Loaded metadata for {len(stone_meta)} stones")
    return stone_meta


def main():
    parser = argparse.ArgumentParser(description="Preprocess Ogham inscription images")
    parser.add_argument("--data-dir", type=str, default="./ogham_dataset",
                        help="Root dataset directory")
    parser.add_argument("--enhancement", type=str, default="clahe",
                        choices=["clahe", "bilateral", "none"],
                        help="Contrast enhancement method")
    parser.add_argument("--target-height", type=int, default=384,
                        help="Target image height in pixels")
    parser.add_argument("--force", action="store_true",
                        help="Re-process all images (ignore existing)")
    args = parser.parse_args()

    skip_existing = not args.force

    data_dir = Path(args.data_dir)
    images_dir = data_dir / "raw" / "images"
    output_dir = data_dir / "processed" / "cropped"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    # Load stone metadata for adaptive CLAHE
    stone_metadata = load_stone_metadata(data_dir)

    # Create preprocessor with v3 pipeline
    config = PreprocessConfig(
        # Orientation: auto-detect from aspect ratio
        fix_orientation=True,
        orientation_mode="auto",
        auto_orientation_threshold=1.2,
        # Cropping: multi-scale Sobel + adaptive thresholds
        crop_to_inscription=True,
        crop_padding_fraction=0.05,
        # Greyscale: remove colour noise
        convert_greyscale=True,
        # Denoising: bilateral filter for stone texture
        denoise=True,
        # Lighting: Retinex shadow normalisation
        normalize_lighting=True,
        # Enhancement: weathering-adaptive CLAHE
        enhance_contrast=True,
        enhancement_method=args.enhancement,
        adaptive_clahe=True,
        # Sharpening: unsharp mask for faint strokes
        sharpen=True,
        # Resize to target height
        resize=True,
        target_height=args.target_height,
    )
    preprocessor = OghamPreprocessor(config)

    print(f"Preprocessor v{preprocessor.VERSION}")
    print(f"  orientation: auto (threshold={config.auto_orientation_threshold})")
    print(f"  crop: multi-scale Sobel + adaptive thresholds")
    print(f"  greyscale: enabled")
    print(f"  denoise: bilateral (d={config.bilateral_d})")
    print(f"  lighting: Retinex (sigma={config.retinex_sigma})")
    print(f"  enhancement: {args.enhancement} (adaptive={config.adaptive_clahe})")
    print(f"  sharpen: unsharp mask (amount={config.unsharp_amount})")
    print(f"  target height: {args.target_height}px")

    # Find all stone directories with images
    stone_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(stone_dirs)} stone directories")

    start = time.time()
    processed_count = 0
    skipped_count = 0
    error_count = 0
    orientation_stats = {"rotated": 0, "kept": 0}
    crop_stats = {"cropped": 0, "skipped": 0}
    clahe_stats = {"adaptive_minimal": 0, "adaptive_moderate": 0,
                   "adaptive_severe": 0, "config_default": 0}

    for stone_dir in stone_dirs:
        stone_id = stone_dir.name
        stone_output_dir = output_dir / stone_id
        stone_output_dir.mkdir(parents=True, exist_ok=True)

        # Build per-stone metadata with weathering severity for adaptive CLAHE
        meta = {"stone_id": stone_id}
        if stone_id in stone_metadata:
            ws = stone_metadata[stone_id].get("weathering_severity")
            if ws:
                meta["weathering_severity"] = ws

        # Find all images for this stone
        image_files = sorted(
            list(stone_dir.glob("*.jpg")) +
            list(stone_dir.glob("*.jpeg")) +
            list(stone_dir.glob("*.png"))
        )

        for img_path in image_files:
            output_path = stone_output_dir / f"{img_path.stem}.png"

            # Skip if already processed
            if skip_existing and output_path.exists():
                skipped_count += 1
                continue

            try:
                log = preprocessor.process_file(
                    input_path=str(img_path),
                    output_path=str(output_path),
                    metadata=meta,
                )
                processed_count += 1

                # Collect stats from log
                for step in log.get("steps", []):
                    if step.get("step") == "orientation":
                        if step.get("applied_rotation", "none") != "none":
                            orientation_stats["rotated"] += 1
                        else:
                            orientation_stats["kept"] += 1
                    elif step.get("step") == "crop":
                        if step.get("skipped"):
                            crop_stats["skipped"] += 1
                        else:
                            crop_stats["cropped"] += 1
                    elif step.get("step") == "enhance":
                        src = step.get("clip_source", "config_default")
                        if src in clahe_stats:
                            clahe_stats[src] += 1
                        else:
                            clahe_stats["config_default"] += 1

                if processed_count % 25 == 0:
                    print(f"  [{processed_count}] Processed {stone_id}/{img_path.name}")

            except Exception as e:
                print(f"  ERROR processing {img_path}: {e}")
                error_count += 1

    elapsed = time.time() - start

    # Save processing summary
    summary = {
        "preprocessor_version": preprocessor.VERSION,
        "processed": processed_count,
        "skipped": skipped_count,
        "errors": error_count,
        "total_stones": len(stone_dirs),
        "enhancement": args.enhancement,
        "target_height": args.target_height,
        "elapsed_seconds": round(elapsed, 2),
        "orientation_stats": orientation_stats,
        "crop_stats": crop_stats,
        "clahe_stats": clahe_stats,
    }
    summary_path = output_dir / "preprocessing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPreprocessing complete in {elapsed:.1f}s")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped (existing): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Orientation: {orientation_stats['rotated']} rotated, {orientation_stats['kept']} kept horizontal")
    print(f"  Cropping: {crop_stats['cropped']} auto-cropped, {crop_stats['skipped']} skipped")
    print(f"  CLAHE: {clahe_stats}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
