#!/usr/bin/env python3
"""
Transform real Ogham stone images to match the synthetic training image style.

Synthetic images have:
- Small, thin strokes on grey stone-textured background
- Low contrast (dark grey strokes on lighter grey)
- Horizontal orientation along a stem line
- Significant grey padding around the inscription
- Overall muted/soft appearance

This script applies transformations to close the domain gap:
1. Convert to greyscale
2. Reduce contrast (make strokes lighter, background darker)
3. Shrink inscription and add grey padding
4. Add stone-like texture/noise
5. Soften/blur to match synthetic softness

Usage:
    python scripts/match_synthetic_style.py \
        --input ogham_dataset/curated \
        --output ogham_dataset/style_matched \
        --preview   # show before/after for first 5 images

    python scripts/match_synthetic_style.py \
        --input ogham_dataset/curated \
        --output ogham_dataset/style_matched
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def match_synthetic_style(
    image: np.ndarray,
    target_height: int = 384,
    target_width: int = 0,  # 0 = auto based on aspect ratio
    inscription_scale: float = 0.3,
    bg_intensity: int = 180,
    stroke_intensity: int = 80,
    blur_sigma: float = 1.0,
    noise_sigma: float = 8.0,
    contrast_reduction: float = 0.4,
) -> np.ndarray:
    """
    Transform a real stone image to match synthetic training style.

    Args:
        image: Input BGR image
        target_height: Output image height
        target_width: Output image width
        inscription_scale: How much of the output the inscription fills (0-1)
        bg_intensity: Background grey level (0-255, higher = lighter)
        stroke_intensity: Stroke grey level (0-255, lower = darker)
        blur_sigma: Gaussian blur sigma for softening
        noise_sigma: Noise sigma for stone texture simulation
        contrast_reduction: How much to reduce contrast (0 = full contrast, 1 = no contrast)
    """
    # Rotate portrait images to landscape (synthetic images are always landscape)
    h_in, w_in = image.shape[:2]
    if h_in > w_in:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h_in, w_in = image.shape[:2]

    # Auto-calculate width based on input aspect ratio (matching synthetic style)
    if target_width <= 0:
        aspect = w_in / max(h_in, 1)
        # Synthetic images are 384 tall, width varies with text length
        target_width = max(134, min(1200, int(target_height * aspect * 1.2)))

    # Step 1: Convert to greyscale
    if len(image.shape) == 3:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grey = image.copy()

    # Step 2: Threshold to isolate strokes (adaptive for varying lighting)
    # Use Otsu's method to find the best threshold
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if strokes are white on dark (we want dark strokes)
    if np.mean(binary) < 128:
        binary = 255 - binary

    # Step 3: Resize inscription to be small within the target
    h, w = binary.shape
    max_inscription_h = int(target_height * inscription_scale)
    max_inscription_w = int(target_width * 0.9)  # leave some margin

    # Always resize to fit within target bounds
    scale = min(max_inscription_h / max(h, 1), max_inscription_w / max(w, 1))
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Clamp to target dimensions
    new_h = min(new_h, target_height)
    new_w = min(new_w, target_width)
    resized = resized[:new_h, :new_w]

    # Step 4: Create grey background and place inscription in center
    output = np.full((target_height, target_width), bg_intensity, dtype=np.uint8)

    # Center the inscription
    y_offset = max(0, (target_height - new_h) // 2)
    x_offset = max(0, (target_width - new_w) // 2)

    # Place inscription — where binary is dark (strokes), use stroke_intensity
    roi = output[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
    mask = resized < 128  # stroke pixels
    # Ensure mask matches roi shape exactly
    mask = mask[:roi.shape[0], :roi.shape[1]]
    roi[mask] = stroke_intensity

    # Step 5: Reduce contrast by blending towards mid-grey
    mid_grey = np.full_like(output, bg_intensity - 20)
    output = cv2.addWeighted(output, 1 - contrast_reduction, mid_grey, contrast_reduction, 0)

    # Step 6: Add Gaussian blur to soften
    if blur_sigma > 0:
        output = cv2.GaussianBlur(output, (0, 0), blur_sigma)

    # Step 7: Add stone-like noise texture
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, output.shape).astype(np.float32)
        output = np.clip(output.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Step 8: Apply slight CLAHE for subtle local contrast (like synthetic stone texture)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    output = clahe.apply(output)

    # Convert back to RGB (3-channel grey)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    return output_rgb


def process_directory(input_dir: Path, output_dir: Path, preview: bool = False, **kwargs):
    """Process all curated images in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_images = []
    count = 0

    for stone_dir in sorted(input_dir.iterdir()):
        if not stone_dir.is_dir():
            continue

        out_stone_dir = output_dir / stone_dir.name
        out_stone_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(stone_dir.glob("*.png")) + sorted(stone_dir.glob("*.jpg")):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            transformed = match_synthetic_style(image, **kwargs)
            out_path = out_stone_dir / (img_path.stem + ".png")
            cv2.imwrite(str(out_path), transformed)
            count += 1

            if preview and len(preview_images) < 5:
                preview_images.append((str(img_path), image, transformed))

            print(f"  {stone_dir.name}/{img_path.name} -> {out_path.name}")

    print(f"\nProcessed {count} images to {output_dir}")

    # Show preview
    if preview and preview_images:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            n = len(preview_images)
            fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
            if n == 1:
                axes = [axes]

            for i, (name, orig, transformed) in enumerate(preview_images):
                axes[i][0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
                axes[i][0].set_title(f"Original: {Path(name).parent.name}", fontsize=10)
                axes[i][0].axis("off")

                axes[i][1].imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
                axes[i][1].set_title("Style-matched", fontsize=10)
                axes[i][1].axis("off")

            plt.suptitle("Real → Synthetic Style Matching", fontsize=14, fontweight="bold")
            plt.tight_layout()
            preview_path = output_dir / "preview.png"
            plt.savefig(str(preview_path), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Preview saved to: {preview_path}")
        except ImportError:
            print("matplotlib not available for preview")


def main():
    parser = argparse.ArgumentParser(description="Match real Ogham images to synthetic style")
    parser.add_argument("--input", type=str, default="ogham_dataset/curated",
                        help="Input directory of curated images")
    parser.add_argument("--output", type=str, default="ogham_dataset/style_matched",
                        help="Output directory for style-matched images")
    parser.add_argument("--preview", action="store_true",
                        help="Save a before/after preview image")

    # Style parameters
    parser.add_argument("--target-height", type=int, default=384)
    parser.add_argument("--target-width", type=int, default=384)
    parser.add_argument("--inscription-scale", type=float, default=0.3,
                        help="How much of the output the inscription fills (0-1)")
    parser.add_argument("--bg-intensity", type=int, default=180,
                        help="Background grey level (higher = lighter)")
    parser.add_argument("--stroke-intensity", type=int, default=80,
                        help="Stroke grey level (lower = darker)")
    parser.add_argument("--blur", type=float, default=1.0,
                        help="Gaussian blur sigma")
    parser.add_argument("--noise", type=float, default=8.0,
                        help="Noise sigma for stone texture")
    parser.add_argument("--contrast-reduction", type=float, default=0.4,
                        help="Contrast reduction (0=full, 1=flat)")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Style:  scale={args.inscription_scale}, bg={args.bg_intensity}, "
          f"stroke={args.stroke_intensity}, blur={args.blur}, noise={args.noise}\n")

    process_directory(
        input_dir, output_dir,
        preview=args.preview,
        target_height=args.target_height,
        target_width=args.target_width,
        inscription_scale=args.inscription_scale,
        bg_intensity=args.bg_intensity,
        stroke_intensity=args.stroke_intensity,
        blur_sigma=args.blur,
        noise_sigma=args.noise,
        contrast_reduction=args.contrast_reduction,
    )


if __name__ == "__main__":
    main()
