#!/usr/bin/env python3
"""Render a fixed test string with each Ogham font so we can visually compare
how single-stroke characters (B-aicme, H-aicme, M-aicme, A-aicme) are drawn.

This surfaces font-specific rendering inconsistencies — e.g. a font that
renders ᚁ (B, 1-below) as a short tick instead of a clear stroke.

Output: one PNG per font in ``diag_fonts/``.

Usage:
    python scripts/diagnose_fonts.py
"""

import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.renderer import OghamRenderer

# Test string: one-stroke character from each aicme + a few multi-stroke ones
# for context. Shows how each font handles visually-confusable categories.
TEST_STRING = "ᚁᚅᚆᚊᚋᚏᚐᚔ"
# chars:      B  N  H  Q  M  R  A  I
# aicmes:     b1 b5 h1 h5 m1 m5 a1 a5

FONT_DIR = PROJECT_ROOT / "data" / "fonts"
OUT_DIR = PROJECT_ROOT / "diag_fonts"
OUT_DIR.mkdir(exist_ok=True)

TRACE_BG = (180, 180, 180)
TRACE_FG = (50, 50, 50)

for font_path in sorted(FONT_DIR.glob("*.ttf")):
    renderer = OghamRenderer(
        font_paths=[str(font_path)],
        char_height_range=(60, 61),         # min..max-1; (60,61) loads size 60 only
        stemline_thickness_range=(4, 4),
        char_spacing_range=(10, 10),
        seed=42,
    )
    img, meta = renderer.render(
        TEST_STRING,
        style_override={
            "bg_color": TRACE_BG,
            "fg_color": TRACE_FG,
            "stemline_thickness": 4,
        },
    )
    out_path = OUT_DIR / f"{font_path.stem}.png"
    Image.fromarray(img).save(out_path)
    print(f"  {font_path.name:<40} -> {out_path}  ({img.shape[1]}x{img.shape[0]})")

print(f"\nOpen each file and compare how the 'single-stroke' chars (ᚁ ᚆ ᚋ ᚐ) render:")
print(f"  open {OUT_DIR}/*.png")
print("\nThe chars in TEST_STRING:")
print("  ᚁ (B, 1 below)   ᚅ (N, 5 below)")
print("  ᚆ (H, 1 above)   ᚊ (Q, 5 above)")
print("  ᚋ (M, 1 across)  ᚏ (R, 5 across)")
print("  ᚐ (A, 1 notch)   ᚔ (I, 5 notches)")
