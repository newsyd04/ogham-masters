#!/usr/bin/env python3
"""
Evaluate TrOCR-small on real Ogham stone images (curated or raw).

Usage:
    # On Colab (with checkpoint on Drive):
    python scripts/eval_real_stones.py \
        --checkpoint /content/drive/MyDrive/ogham_ocr/checkpoints/best_ogham \
        --data-dir ogham_dataset/curated

    # Locally (download checkpoint first):
    python scripts/eval_real_stones.py \
        --checkpoint path/to/best_ogham \
        --data-dir ogham_dataset/curated

    # Use raw processed images instead of curated:
    python scripts/eval_real_stones.py \
        --checkpoint path/to/best_ogham \
        --data-dir ogham_dataset/processed/cropped
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_DIR = PROJECT_ROOT / "ogham_dataset"
ANNOTATIONS_FILE = DATASET_DIR / "processed" / "annotations" / "transcriptions.json"
SPLITS_DIR = DATASET_DIR / "splits"


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrOCR on real Ogham stones")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_ogham checkpoint directory")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR / "curated"),
                        help="Path to image directory (curated/ or processed/cropped/)")
    parser.add_argument("--split", type=str, default="test",
                        help="Which split to evaluate: test, val, or all")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")
    args = parser.parse_args()

    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
    from PIL import Image
    import editdistance

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    ckpt = Path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt}")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
    model = VisionEncoderDecoderModel.from_pretrained(str(ckpt))
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = model.to(device).eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Load annotations
    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)

    # Load split
    if args.split == "all":
        stone_ids = list(annotations.keys())
    else:
        split_file = SPLITS_DIR / f"{args.split}_stones.txt"
        with open(split_file) as f:
            stone_ids = [line.strip() for line in f if line.strip()]
    print(f"Split: {args.split} ({len(stone_ids)} stones)")

    # Load curated transcriptions if available
    curation_file = DATASET_DIR / "processed" / "curation.json"
    curation = {}
    if curation_file.exists():
        with open(curation_file) as f:
            curation = json.load(f)
        n_edited = sum(1 for v in curation.values() if "transcription" in v)
        if n_edited:
            print(f"Loaded {n_edited} curated transcriptions from curation.json")

    # Find images
    data_dir = Path(args.data_dir)
    samples = []
    for stone_id in stone_ids:
        ann = annotations.get(stone_id, {})
        transcription = ann.get("transcription", "")
        if not transcription:
            continue

        stone_dir = data_dir / stone_id
        if not stone_dir.exists():
            continue

        images = sorted(stone_dir.glob("*.png")) + sorted(stone_dir.glob("*.jpg")) + sorted(stone_dir.glob("*.JPG"))
        for img_path in images:
            # Skip metadata JSON files
            if img_path.suffix == ".json":
                continue

            # Use curated transcription if available
            curation_key = f"{stone_id}/{img_path.name}"
            curated_transcription = curation.get(curation_key, {}).get("transcription", None)
            ref = curated_transcription if curated_transcription else transcription

            samples.append({
                "stone_id": stone_id,
                "image_path": str(img_path),
                "reference": ref,
                "transcription_source": "curated" if curated_transcription else "original",
            })

    print(f"Found {len(samples)} images from {len(set(s['stone_id'] for s in samples))} stones")

    if not samples:
        print("No samples found! Check --data-dir path.")
        return

    # Run inference
    results = []
    total_cer_dist = 0
    total_chars = 0
    exact_matches = 0

    print(f"\n{'Stone':<15} {'CER':>6} {'Match':>6}  Reference -> Prediction")
    print("-" * 80)

    for i, sample in enumerate(samples):
        image = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=64,
                num_beams=4,
                decoder_start_token_id=tokenizer.cls_token_id,
                eos_token_id=tokenizer.sep_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        ref = sample["reference"]
        dist = editdistance.eval(pred, ref)
        cer = dist / max(len(ref), 1)
        is_exact = pred.strip() == ref.strip()

        total_cer_dist += dist
        total_chars += max(len(ref), 1)
        if is_exact:
            exact_matches += 1

        result = {
            "stone_id": sample["stone_id"],
            "image": Path(sample["image_path"]).name,
            "reference": ref,
            "prediction": pred,
            "cer": cer,
            "exact_match": is_exact,
        }
        results.append(result)

        match_str = "YES" if is_exact else ""
        print(f"  {sample['stone_id']:<13} {cer:>5.0%} {match_str:>6}  '{ref[:25]}' -> '{pred[:25]}'")

    # Summary
    mean_cer = total_cer_dist / max(total_chars, 1)
    exact_rate = exact_matches / max(len(results), 1)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {args.split} split on {data_dir.name}/ images")
    print(f"{'=' * 60}")
    print(f"  Stones: {len(set(s['stone_id'] for s in samples))}")
    print(f"  Images: {len(results)}")
    print(f"  Mean CER: {mean_cer:.2%}")
    print(f"  Exact Match: {exact_rate:.1%} ({exact_matches}/{len(results)})")
    print(f"{'=' * 60}")

    # Save results
    output = {
        "checkpoint": str(ckpt),
        "data_dir": str(data_dir),
        "split": args.split,
        "mean_cer": mean_cer,
        "exact_match": exact_rate,
        "n_stones": len(set(s["stone_id"] for s in samples)),
        "n_images": len(results),
        "per_image": results,
    }

    out_path = args.output or str(PROJECT_ROOT / "docs" / f"real_eval_{data_dir.name}_{args.split}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
