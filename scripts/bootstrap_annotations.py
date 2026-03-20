"""Bootstrap annotations from DIAS scraped metadata.

Reads consensus transcriptions from stone_metadata.jsonl and converts
them into the annotation format expected by the training pipeline.

Cleans epigraphic notation (combining diacritics, brackets, lacunae markers)
to produce clean Ogham Unicode strings suitable for OCR training.

Usage:
    python scripts/bootstrap_annotations.py [--data-dir ./ogham_dataset]
"""

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.ogham import is_ogham_character, validate_ogham_string, OGHAM_SPACE


def clean_transcription(raw: str) -> str:
    """Clean epigraphic notation from a DIAS consensus transcription.

    Removes combining diacritical marks, editorial brackets, lacunae markers,
    and other non-Ogham characters to produce a clean Unicode Ogham string.

    Args:
        raw: Raw consensus transcription with epigraphic notation.

    Returns:
        Cleaned Ogham Unicode string (may be empty if too damaged).
    """
    if not raw:
        return ""

    text = raw

    # 1. Remove leading face/side numbers (e.g., "1ᚉᚑᚂᚑ...")
    text = re.sub(r"^\d+", "", text)

    # 2. Remove lacunae markers: [---], [.. ? ..], [... ? ...], etc.
    text = re.sub(r"\[\s*-+\s*\]", "", text)
    text = re.sub(r"\[\s*\.+\s*\?\s*\.+\s*\]", "", text)
    text = re.sub(r"\.+\s*\?\s*\.+", "", text)

    # 3. Extract content from editorial angle brackets: <X> → X
    text = re.sub(r"<([^>]*)>", r"\1", text)

    # 4. Extract content from square brackets: [X] → X (restored letters)
    text = re.sub(r"\[([^\]]*)\]", r"\1", text)

    # 5. Remove question marks and parenthetical uncertainty: (?)
    text = text.replace("(?)", "")
    text = text.replace("?", "")

    # 6. Remove combining diacritical marks (U+0300-U+036F)
    # These indicate uncertain readings in epigraphy (e.g., ᚌ̣ = uncertain G)
    cleaned_chars = []
    for char in text:
        if unicodedata.category(char).startswith("M"):
            continue  # Skip combining marks
        cleaned_chars.append(char)
    text = "".join(cleaned_chars)

    # 7. Keep only Ogham characters, Ogham space, and regular spaces
    result = []
    for char in text:
        if is_ogham_character(char):
            result.append(char)
        elif char == " ":
            # Convert regular spaces to Ogham space mark
            result.append(OGHAM_SPACE)
    text = "".join(result)

    # 8. Collapse multiple consecutive spaces
    text = re.sub(f"[{OGHAM_SPACE}]+", OGHAM_SPACE, text)

    # 9. Strip leading/trailing spaces
    text = text.strip(OGHAM_SPACE).strip()

    return text


def assess_confidence(stone: dict, cleaned: str, raw: str) -> str:
    """Determine annotation confidence based on reading quality.

    Args:
        stone: Full stone metadata dict.
        cleaned: Cleaned transcription string.
        raw: Original raw transcription.

    Returns:
        Confidence level: "verified", "probable", or "uncertain".
    """
    # Check how much was lost during cleaning
    raw_ogham_count = sum(1 for c in raw if is_ogham_character(c))
    clean_count = sum(1 for c in cleaned if is_ogham_character(c))

    if raw_ogham_count == 0:
        return "uncertain"

    # Ratio of preserved characters
    preservation_ratio = clean_count / raw_ogham_count

    # Check readings confidence from metadata
    readings = stone.get("transcription_readings", [])
    has_verified = any(r.get("confidence") == "verified" for r in readings)
    has_diplomatic = any(r.get("type") == "diplomatic" for r in readings)

    # High confidence: verified readings, diplomatic + interpretive agree, good preservation
    if has_verified and has_diplomatic and preservation_ratio > 0.8:
        return "verified"
    elif has_verified and preservation_ratio > 0.5:
        return "probable"
    else:
        return "uncertain"


def main():
    parser = argparse.ArgumentParser(description="Bootstrap annotations from DIAS metadata")
    parser.add_argument("--data-dir", type=str, default="./ogham_dataset",
                        help="Root dataset directory")
    parser.add_argument("--min-chars", type=int, default=2,
                        help="Minimum Ogham characters required after cleaning")
    parser.add_argument("--min-confidence", type=str, default="uncertain",
                        choices=["verified", "probable", "uncertain"],
                        help="Minimum confidence level to include")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    metadata_file = data_dir / "raw" / "metadata" / "stone_metadata.jsonl"
    images_dir = data_dir / "raw" / "images"
    output_dir = data_dir / "processed" / "annotations"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}")
        sys.exit(1)

    # Load all stone metadata
    stones = []
    with open(metadata_file) as f:
        for line in f:
            stones.append(json.loads(line.strip()))

    print(f"Loaded {len(stones)} stones from metadata")

    # Get stones with images on disk
    stones_with_images = set()
    if images_dir.exists():
        for d in images_dir.iterdir():
            if d.is_dir():
                # Check it actually has image files
                imgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
                if imgs:
                    stones_with_images.add(d.name)

    print(f"Found {len(stones_with_images)} stones with images on disk")

    # Process annotations
    annotations = {}
    stats = {
        "total_stones": len(stones),
        "with_consensus": 0,
        "with_images": 0,
        "bootstrapped": 0,
        "skipped_no_images": 0,
        "skipped_no_transcription": 0,
        "skipped_too_short": 0,
        "by_confidence": {"verified": 0, "probable": 0, "uncertain": 0},
    }

    confidence_levels = ["verified", "probable", "uncertain"]
    min_confidence_idx = confidence_levels.index(args.min_confidence)

    for stone in stones:
        stone_id = stone["stone_id"]
        raw_transcription = stone.get("consensus_transcription", "")

        if not raw_transcription:
            stats["skipped_no_transcription"] += 1
            continue

        stats["with_consensus"] += 1

        if stone_id not in stones_with_images:
            stats["skipped_no_images"] += 1
            continue

        stats["with_images"] += 1

        # Clean the transcription
        cleaned = clean_transcription(raw_transcription)

        # Check minimum character count
        ogham_count = sum(1 for c in cleaned if is_ogham_character(c))
        if ogham_count < args.min_chars:
            stats["skipped_too_short"] += 1
            continue

        # Assess confidence
        confidence = assess_confidence(stone, cleaned, raw_transcription)
        conf_idx = confidence_levels.index(confidence)

        if conf_idx > min_confidence_idx:
            continue

        # Create annotation entry
        annotations[stone_id] = {
            "stone_id": stone_id,
            "transcription": cleaned,
            "confidence": confidence,
            "source": "DIAS consensus (auto-bootstrapped)",
            "original_transcription": raw_transcription,
            "annotator": "auto-bootstrap",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "version": 1,
            "county": stone.get("county"),
            "region": stone.get("region"),
        }

        stats["bootstrapped"] += 1
        stats["by_confidence"][confidence] += 1

    # Save annotations
    output_file = output_dir / "transcriptions.json"
    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # Save bootstrap report
    report_file = output_dir / "bootstrap_report.json"
    stats["output_file"] = str(output_file)
    stats["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(report_file, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\nBootstrap Results:")
    print(f"  Stones with consensus transcription: {stats['with_consensus']}")
    print(f"  Stones with both consensus + images:  {stats['with_images']}")
    print(f"  Successfully bootstrapped:            {stats['bootstrapped']}")
    print(f"    Verified:   {stats['by_confidence']['verified']}")
    print(f"    Probable:   {stats['by_confidence']['probable']}")
    print(f"    Uncertain:  {stats['by_confidence']['uncertain']}")
    print(f"  Skipped (no images on disk):          {stats['skipped_no_images']}")
    print(f"  Skipped (too short after cleaning):   {stats['skipped_too_short']}")
    print(f"\nAnnotations saved to: {output_file}")

    # Show a few examples
    print(f"\nSample annotations:")
    for i, (stone_id, ann) in enumerate(list(annotations.items())[:5]):
        print(f"  {stone_id}: {ann['transcription']} [{ann['confidence']}]")
        print(f"    Original: {ann['original_transcription'][:80]}...")


if __name__ == "__main__":
    main()
