#!/usr/bin/env python3
"""
Generate synthetic freeform-style Ogham images.

Starts from the standard OghamRenderer (font-based, clean) and applies
hand-drawn-mimicking perturbations:
  - Elastic deformation (the main "wobble" effect)
  - Gaussian blur (softer edges)
  - Slight rotation
  - Mild brightness/contrast jitter
  - Optional stroke doubling via morphological dilation + re-threshold

Outputs to ``ogham_dataset/synthetic_freeform/``:
  - ``images/freeform_XXXX.png``
  - ``labels.csv`` with ``image_file,ogham_text,latin_transliteration,length``

Usage:
    python scripts/generate_freeform_synthetic.py --num 150 --out ogham_dataset/synthetic_freeform
"""

import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.renderer import OghamRenderer
from src.generation.sequence_sampler import OghamSequenceSampler
from src.utils.ogham import ogham_to_latin

FONT_DIR = PROJECT_ROOT / "data" / "fonts"

# Match the exact colours the curate_dataset.py render pipeline uses, so
# Phase 2 training on this data transfers cleanly to evaluation on real
# traced images. #b4b4b4 background + dark-grey strokes.
TRACE_BG_RGB = (180, 180, 180)
TRACE_FG_RGB = (50, 50, 50)


def build_freeform_augmenter(severity: str = "medium"):
    """Albumentations pipeline tuned to mimic hand-drawn Ogham traces.

    The key transform is ElasticTransform — it warps the image with a
    smooth random displacement field, which turns clean font-rendered
    strokes into naturally wobbly strokes. Tuned so wobble is visible but
    not destructive to character identity.
    """
    import albumentations as A
    import cv2

    if severity == "light":
        elastic_alpha, elastic_sigma = 40, 6
        blur_sigma = (0.5, 1.2)
        rotate_deg = 1.5
    elif severity == "heavy":
        elastic_alpha, elastic_sigma = 120, 10
        blur_sigma = (1.0, 2.2)
        rotate_deg = 3.0
    else:  # medium — closest to observed real freeform wobble
        elastic_alpha, elastic_sigma = 75, 8
        blur_sigma = (0.7, 1.8)
        rotate_deg = 2.0

    # Fill any border created by elastic warp / rotation with the trace grey
    # so background stays uniformly #b4b4b4 across the whole image.
    border_value = (180, 180, 180)

    return A.Compose([
        A.ElasticTransform(
            alpha=elastic_alpha,
            sigma=elastic_sigma,
            border_mode=cv2.BORDER_CONSTANT,
            value=border_value,
            p=1.0,
        ),
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=blur_sigma, p=0.8),
        A.Rotate(
            limit=rotate_deg,
            border_mode=cv2.BORDER_CONSTANT,
            value=border_value,
            p=0.7,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.08, contrast_limit=0.12, p=0.6
        ),
        # Mild thickness jitter: dilate or erode slightly
        A.OneOf([
            A.Morphological(scale=(1, 2), operation="dilation", p=1.0),
            A.Morphological(scale=(1, 2), operation="erosion", p=1.0),
        ], p=0.4),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=150, help="Number of images to generate")
    parser.add_argument("--out", type=str, default="ogham_dataset/synthetic_freeform",
                        help="Output directory (relative to repo root)")
    parser.add_argument("--severity", choices=["light", "medium", "heavy"], default="medium")
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Check dependencies
    try:
        import albumentations  # noqa: F401
        import cv2  # noqa: F401
        from PIL import Image
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install albumentations opencv-python-headless Pillow")
        sys.exit(1)

    out_dir = (PROJECT_ROOT / args.out).resolve()
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    font_paths = sorted(str(p) for p in FONT_DIR.glob("*.ttf"))
    if not font_paths:
        print(f"No fonts found in {FONT_DIR}")
        sys.exit(1)
    print(f"Using {len(font_paths)} fonts from {FONT_DIR}")

    random.seed(args.seed)
    sampler = OghamSequenceSampler(
        min_length=args.min_len,
        max_length=args.max_len,
        seed=args.seed,
    )
    renderer = OghamRenderer(font_paths=font_paths, seed=args.seed)
    aug = build_freeform_augmenter(args.severity)

    labels_path = out_dir / "labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as lf:
        writer = csv.writer(lf)
        writer.writerow(["image_file", "ogham_text", "latin_transliteration", "length", "severity"])

        # Force the bg/stroke colours and a thick stemline to match what the
        # curator tool saves (traced/ images all have bg=(180,180,180),
        # strokes=(50,50,50), and consistent thick strokes).
        style_override = {
            "bg_color": TRACE_BG_RGB,
            "fg_color": TRACE_FG_RGB,
            "stemline_thickness": 6,
        }

        # Pre-augmentation thickening: dilate strokes so font-rendered
        # character lines end up thicker, roughly matching real traced
        # stroke widths. Kernel size controls how much thicker.
        import cv2
        import numpy as np
        thicken_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for i in range(args.num):
            text = sampler.sample()
            clean_img, _ = renderer.render(text, style_override=style_override)

            # Thicken strokes (operate on inverse mask, then blend back)
            # Gray the image, find dark pixels (strokes), dilate them.
            gray = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
            stroke_mask = (gray < 120).astype(np.uint8) * 255
            thick_mask = cv2.dilate(stroke_mask, thicken_kernel, iterations=1)
            # Rebuild the image with thickened strokes
            thickened = np.full_like(clean_img, TRACE_BG_RGB)
            thickened[thick_mask > 0] = TRACE_FG_RGB
            clean_img = thickened

            # Apply freeform augmentations
            augmented = aug(image=clean_img)["image"]

            # Save as PNG
            out_name = f"freeform_{i:04d}.png"
            Image.fromarray(augmented).save(img_dir / out_name)
            writer.writerow([out_name, text, ogham_to_latin(text), len(text), args.severity])

            if (i + 1) % 25 == 0 or i == args.num - 1:
                print(f"  {i + 1:>4}/{args.num} — {out_name} — {text}")

    print(f"\nDone. {args.num} images written to {img_dir}")
    print(f"Labels: {labels_path}")
    print("\nNext: visually inspect a few samples to confirm the wobble level")
    print("matches your real freeform traces. If too clean, re-run with --severity heavy.")


if __name__ == "__main__":
    main()
