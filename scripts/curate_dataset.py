#!/usr/bin/env python3
"""
Interactive image viewer and annotation tool for curating the Ogham dataset.

Opens a local web UI where you can:
- Browse RAW stone images (original high-res photographs)
- Rotate, crop, enhance, invert images in-browser
- Mark images as: keep, enhance, or drop
- Add notes per image
- Export the curated dataset

Usage:
    python scripts/curate_dataset.py

Opens at http://localhost:8765
"""

import json
import os
import sys
import base64
import http.server
import socketserver
import io
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "ogham_dataset"
RAW_DIR = DATASET_DIR / "raw" / "images"
CROPPED_DIR = DATASET_DIR / "processed" / "cropped"
ANNOTATIONS_FILE = DATASET_DIR / "processed" / "annotations" / "transcriptions.json"
CURATION_FILE = DATASET_DIR / "processed" / "curation.json"
CURATED_DIR = DATASET_DIR / "curated"
TRACED_DIR = DATASET_DIR / "traced"         # synthetic-style (lines only on grey bg)
OVERLAY_DIR = DATASET_DIR / "overlay"       # Josh's option A (lines on stone photo)
TRACES_FILE = DATASET_DIR / "processed" / "traces.json"

# Synthetic-style rendering constants (match src/generation/renderer.py)
SYNTH_TARGET_HEIGHT = 384
SYNTH_PADDING = 20
SYNTH_BG_RGB = (180, 180, 180)
SYNTH_STROKE_RGB = (50, 50, 50)
# Cap aspect ratio to match TrOCR training distribution. Synthetic images in
# training peaked at roughly 4:1 — inscriptions wider than that get compressed
# horizontally on output so the TrOCR image processor's forced 384x384 resize
# doesn't squash them beyond recognition.
SYNTH_MAX_ASPECT = 4.0

PORT = 8765


def load_data():
    """Load all stone images — raw first, with processed as fallback."""
    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)

    stones = []

    # Collect all stone IDs from both raw and cropped
    stone_ids = set()
    if RAW_DIR.exists():
        for d in RAW_DIR.iterdir():
            if d.is_dir():
                stone_ids.add(d.name)
    for d in CROPPED_DIR.iterdir():
        if d.is_dir():
            stone_ids.add(d.name)

    for stone_id in sorted(stone_ids):
        ann = annotations.get(stone_id, {})

        # Get raw images
        raw_dir = RAW_DIR / stone_id
        raw_images = sorted(raw_dir.glob("*.jpg")) + sorted(raw_dir.glob("*.JPG")) if raw_dir.exists() else []

        # Get processed images
        crop_dir = CROPPED_DIR / stone_id
        crop_images = sorted(crop_dir.glob("*.png")) if crop_dir.exists() else []

        # Pair them: show raw with processed as reference
        for img_path in raw_images:
            # Find matching processed image
            base_name = img_path.stem  # e.g. I-COR-005_DIAS_000
            matching_processed = [p for p in crop_images if p.stem.startswith(base_name)]

            stones.append({
                "stone_id": stone_id,
                "image_name": img_path.name,
                "image_path": str(img_path),
                "processed_path": str(matching_processed[0]) if matching_processed else "",
                "source": "raw",
                "transcription": ann.get("transcription", ""),
                "original_transcription": ann.get("original_transcription", ""),
                "confidence": ann.get("confidence", ""),
                "county": ann.get("county", ""),
            })

        # Add processed-only images (no raw available)
        for img_path in crop_images:
            base_name = img_path.stem
            has_raw = any(r.stem == base_name for r in raw_images)
            if not has_raw:
                stones.append({
                    "stone_id": stone_id,
                    "image_name": img_path.name,
                    "image_path": str(img_path),
                    "processed_path": str(img_path),
                    "source": "processed",
                    "transcription": ann.get("transcription", ""),
                    "original_transcription": ann.get("original_transcription", ""),
                    "confidence": ann.get("confidence", ""),
                    "county": ann.get("county", ""),
                })

    return stones


def load_curation():
    """Load existing curation decisions."""
    if CURATION_FILE.exists():
        with open(CURATION_FILE) as f:
            return json.load(f)
    return {}


def save_curation(curation):
    """Save curation decisions."""
    with open(CURATION_FILE, "w") as f:
        json.dump(curation, f, indent=2, ensure_ascii=False)


def _load_traces():
    if TRACES_FILE.exists():
        with open(TRACES_FILE) as f:
            return json.load(f)
    return {}


def _save_traces(traces):
    TRACES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACES_FILE, "w") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)


_OGHAM_GEOMETRY = {
    "B": ("b", 1), "L": ("b", 2), "F": ("b", 3), "S": ("b", 4), "N": ("b", 5),
    "H": ("h", 1), "D": ("h", 2), "T": ("h", 3), "C": ("h", 4), "Q": ("h", 5),
    "M": ("m", 1), "G": ("m", 2), "NG": ("m", 3), "Z": ("m", 4), "R": ("m", 5),
    "A": ("a", 1), "O": ("a", 2), "U": ("a", 3), "E": ("a", 4), "I": ("a", 5),
}


def _render_strokes_as_synthetic(strokes, stem_p1, stem_p2):
    """Reconstruct a synthetic-style PNG from assisted-mode character strokes.

    Instead of scaling the user's wide canvas, we rebuild the inscription at
    synthetic-matching density — ~70px per character at 384 tall, with the
    stem horizontally centred and strokes sized to match the training
    distribution exactly. Returns None if insufficient assisted-mode data
    (caller should fall back to the alpha-mask render).
    """
    import cv2
    import numpy as np

    char_strokes = [s for s in strokes if s.get("kind") == "char"]
    if not char_strokes or not stem_p1 or not stem_p2:
        return None

    chars_sorted = sorted(char_strokes, key=lambda s: s.get("t", 0))
    n = len(chars_sorted)

    # Synthetic-matching parameters (tuned to training distribution)
    height = SYNTH_TARGET_HEIGHT                          # 384
    char_width = 70                                       # px per char
    stroke_len = 80                                       # perpendicular line length
    notch_len = 22                                        # half-length for vowel notches
    stroke_thickness = 4                                  # px
    stem_thickness = 4                                    # px
    stroke_spacing = 14                                   # between strokes of same char
    canvas_w = n * char_width + 2 * SYNTH_PADDING
    stem_y = height // 2

    canvas = np.full((height, canvas_w, 3), SYNTH_BG_RGB, dtype=np.uint8)
    color = tuple(int(c) for c in SYNTH_STROKE_RGB)

    cv2.line(canvas,
             (SYNTH_PADDING, stem_y),
             (canvas_w - SYNTH_PADDING, stem_y),
             color, stem_thickness, cv2.LINE_AA)

    for i, cs in enumerate(chars_sorted):
        key = cs.get("charKey", "")
        geom = _OGHAM_GEOMETRY.get(key)
        if not geom:
            continue
        aicme, count = geom
        cx = SYNTH_PADDING + (i + 0.5) * char_width

        for j in range(count):
            off = (j - (count - 1) / 2) * stroke_spacing
            sx = int(round(cx + off))
            if aicme == "b":
                cv2.line(canvas, (sx, stem_y), (sx, stem_y + stroke_len),
                         color, stroke_thickness, cv2.LINE_AA)
            elif aicme == "h":
                cv2.line(canvas, (sx, stem_y), (sx, stem_y - stroke_len),
                         color, stroke_thickness, cv2.LINE_AA)
            elif aicme == "m":
                # Diagonal top-left to bottom-right, matching font-rendered Ogham
                half = int(stroke_len * 0.5)
                cv2.line(canvas,
                         (sx - half, stem_y - half),
                         (sx + half, stem_y + half),
                         color, stroke_thickness, cv2.LINE_AA)
            elif aicme == "a":
                cv2.line(canvas, (sx, stem_y - notch_len), (sx, stem_y + notch_len),
                         color, stroke_thickness, cv2.LINE_AA)

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", canvas_bgr)
    return buf.tobytes()


def _render_traced_to_synthetic(image_data_url: str) -> bytes:
    """Convert a transparent black-strokes PNG into a synthetic-style
    grey-background PNG matching the TrOCR training distribution.

    Steps:
      1. Decode the input PNG with alpha channel
      2. Extract the alpha mask (where strokes exist)
      3. Crop to bounding box of strokes + a small padding
      4. Scale the inscription so the inscription content is 384px tall
      5. Composite onto grey background with dark-grey strokes
      6. Letterbox: if aspect ratio exceeds the training maximum (~4:1),
         pad the top and bottom of the image with grey so the aspect ratio
         drops to 4:1. This preserves stroke proportions when the TrOCR
         image processor later force-resizes to 384×384.
    """
    import cv2
    import numpy as np
    import base64 as b64

    # Decode base64 PNG
    png_bytes = b64.b64decode(image_data_url.split(",")[1])
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img_rgba = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if img_rgba is None or img_rgba.shape[-1] < 4:
        # No alpha channel — treat black pixels as the mask
        gray = cv2.cvtColor(img_rgba, cv2.COLOR_BGR2GRAY) if img_rgba.ndim == 3 else img_rgba
        alpha = 255 - gray
    else:
        alpha = img_rgba[..., 3]

    # Find bounding box of the strokes
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        # Empty trace — return a plain grey image
        blank = np.full((SYNTH_TARGET_HEIGHT, SYNTH_TARGET_HEIGHT, 3), SYNTH_BG_RGB, dtype=np.uint8)
        _, buf = cv2.imencode(".png", cv2.cvtColor(blank, cv2.COLOR_RGB2BGR))
        return buf.tobytes()

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    pad = 8
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(alpha.shape[1] - 1, x2 + pad)
    y2 = min(alpha.shape[0] - 1, y2 + pad)
    cropped_alpha = alpha[y1:y2 + 1, x1:x2 + 1]

    # Scale inscription content to SYNTH_TARGET_HEIGHT tall with aspect preserved.
    ch, cw = cropped_alpha.shape
    inner_h = SYNTH_TARGET_HEIGHT - 2 * SYNTH_PADDING
    scale = inner_h / ch
    inner_w = max(inner_h, int(cw * scale))
    mask_resized = cv2.resize(cropped_alpha, (inner_w, inner_h), interpolation=cv2.INTER_AREA)

    # Canvas is target_height tall, with symmetric horizontal padding
    canvas_h = SYNTH_TARGET_HEIGHT
    canvas_w = inner_w + 2 * SYNTH_PADDING
    out = np.full((canvas_h, canvas_w, 3), SYNTH_BG_RGB, dtype=np.uint8)
    stroke = np.array(SYNTH_STROKE_RGB, dtype=np.uint8)
    mask_norm = (mask_resized.astype(np.float32) / 255.0)[..., None]
    roi = out[SYNTH_PADDING:SYNTH_PADDING + inner_h,
              SYNTH_PADDING:SYNTH_PADDING + inner_w]
    roi[:] = (roi.astype(np.float32) * (1 - mask_norm) +
              stroke.astype(np.float32) * mask_norm).astype(np.uint8)

    # Letterbox vertically so the processor's 384x384 resize doesn't destroy
    # the strokes. If aspect ratio > SYNTH_MAX_ASPECT, grow the canvas height
    # by adding symmetric grey bars above and below the inscription.
    aspect = canvas_w / canvas_h
    if aspect > SYNTH_MAX_ASPECT:
        new_h = int(round(canvas_w / SYNTH_MAX_ASPECT))
        letterbox = np.full((new_h, canvas_w, 3), SYNTH_BG_RGB, dtype=np.uint8)
        y_off = (new_h - canvas_h) // 2
        letterbox[y_off:y_off + canvas_h] = out
        out = letterbox

    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", out_bgr)
    return buf.tobytes()


def build_html(stones, curation):
    """Build the single-page HTML app with image editing tools."""

    stones_json = json.dumps(stones, ensure_ascii=False)
    curation_json = json.dumps(curation, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Ogham Dataset Curator</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; }}
        .header {{ background: #16213e; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #0f3460; }}
        .header h1 {{ font-size: 18px; color: #e94560; }}
        .stats {{ font-size: 13px; color: #aaa; }}
        .stats span {{ color: #fff; font-weight: bold; margin: 0 4px; }}
        .container {{ display: flex; height: calc(100vh - 52px); }}

        /* Sidebar */
        .sidebar {{ width: 250px; background: #16213e; overflow-y: auto; border-right: 1px solid #0f3460; flex-shrink: 0; }}
        .sidebar-item {{ padding: 8px 12px; cursor: pointer; border-bottom: 1px solid #0f3460; font-size: 12px; }}
        .sidebar-item:hover {{ background: #1a1a40; }}
        .sidebar-item.active {{ background: #0f3460; border-left: 3px solid #e94560; }}
        .sidebar-item .stone-id {{ font-weight: bold; color: #fff; font-size: 13px; }}
        .sidebar-item .status {{ font-size: 11px; margin-top: 2px; }}
        .status-keep {{ color: #4caf50; }}
        .status-enhance {{ color: #ff9800; }}
        .status-drop {{ color: #f44336; }}
        .status-traced {{ color: #ba68c8; }}
        .status-pending {{ color: #666; }}
        .filter-row {{ padding: 8px 12px; background: #0f3460; display: flex; gap: 4px; flex-wrap: wrap; }}
        .filter-btn {{ padding: 3px 8px; border: 1px solid #333; border-radius: 3px; background: transparent; color: #aaa; cursor: pointer; font-size: 11px; }}
        .filter-btn.active {{ background: #1a4a80; color: #fff; border-color: #e94560; }}

        /* Main area */
        .main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}

        /* Image viewer */
        .image-area {{ flex: 1; display: flex; gap: 0; overflow: hidden; }}
        .image-panel {{ flex: 1; display: flex; flex-direction: column; background: #111; }}
        .image-panel-label {{ font-size: 11px; color: #666; padding: 4px 10px; background: #1a1a2e; text-align: center; }}
        .image-container {{ flex: 1; display: flex; justify-content: center; align-items: center; overflow: hidden; position: relative; cursor: crosshair; }}
        .image-container img {{ max-width: 100%; max-height: 100%; object-fit: contain; transition: transform 0.1s; }}
        .image-container canvas {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
        .divider {{ width: 3px; background: #0f3460; cursor: col-resize; }}

        /* Toolbar */
        .toolbar {{ background: #0f3460; padding: 8px 15px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; border-bottom: 1px solid #1a4a80; }}
        .tool-btn {{ padding: 6px 12px; border: 1px solid #333; border-radius: 4px; background: #16213e; color: #ccc; cursor: pointer; font-size: 12px; transition: all 0.15s; }}
        .tool-btn:hover {{ background: #1a4a80; color: #fff; }}
        .tool-btn.active {{ background: #e94560; color: #fff; border-color: #e94560; }}
        .tool-sep {{ width: 1px; height: 24px; background: #333; margin: 0 4px; }}
        .tool-slider {{ width: 80px; accent-color: #e94560; }}
        .tool-label {{ font-size: 11px; color: #888; }}

        /* Controls */
        .controls {{ background: #16213e; padding: 12px 20px; border-top: 2px solid #0f3460; }}
        .info-row {{ display: flex; gap: 15px; margin-bottom: 8px; font-size: 13px; align-items: center; }}
        .info-row label {{ color: #888; min-width: 80px; font-size: 12px; }}
        .info-row .value {{ color: #fff; font-family: monospace; }}
        .ogham-text {{ font-size: 22px; letter-spacing: 2px; }}
        .button-row {{ display: flex; gap: 8px; margin-bottom: 8px; align-items: center; }}
        .btn {{ padding: 8px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 13px; font-weight: bold; transition: all 0.15s; }}
        .btn-keep {{ background: #4caf50; color: white; }}
        .btn-keep:hover {{ background: #66bb6a; }}
        .btn-keep.selected {{ box-shadow: 0 0 0 3px #fff; }}
        .btn-enhance {{ background: #ff9800; color: white; }}
        .btn-enhance:hover {{ background: #ffa726; }}
        .btn-enhance.selected {{ box-shadow: 0 0 0 3px #fff; }}
        .btn-drop {{ background: #f44336; color: white; }}
        .btn-drop:hover {{ background: #ef5350; }}
        .btn-drop.selected {{ box-shadow: 0 0 0 3px #fff; }}
        .btn-nav {{ background: #0f3460; color: white; padding: 8px 16px; }}
        .btn-nav:hover {{ background: #1a4a80; }}
        .btn-export {{ background: #e94560; color: white; }}
        .notes-input {{ flex: 1; padding: 7px 10px; background: #1a1a2e; border: 1px solid #0f3460; color: #fff; border-radius: 4px; font-size: 12px; }}
        .nav-row {{ display: flex; justify-content: space-between; align-items: center; }}
        .progress {{ font-size: 12px; color: #888; }}
        .keyboard-hint {{ font-size: 10px; color: #444; margin-top: 4px; }}

        /* Crop overlay */
        .crop-overlay {{ position: absolute; border: 2px dashed #e94560; background: rgba(233,69,96,0.1); pointer-events: none; }}

        /* Trace overlay */
        #trace-canvas {{ position: absolute; top: 0; left: 0; max-width: 100%; max-height: 100%; object-fit: contain; pointer-events: auto; cursor: crosshair; display: none; }}
        #trace-canvas.active {{ display: block; }}

        /* Character keypad (assisted mode) — sits as its own column in the image-area */
        .keypad-column {{ width: 280px; min-width: 280px; max-width: 280px; background: #16213e; border-left: 2px solid #ba68c8; padding: 12px; overflow-y: auto; display: none; flex-shrink: 0; }}
        .keypad-column.active {{ display: block; }}
        .keypad {{ background: transparent; border: none; padding: 0; }}
        .keypad h4 {{ color: #ba68c8; font-size: 12px; margin-bottom: 4px; letter-spacing: 1px; }}
        .keypad-row {{ display: flex; gap: 4px; margin-bottom: 4px; flex-wrap: wrap; }}
        .keypad-btn {{ flex: 1; min-width: 42px; padding: 8px 4px; background: #0f3460; color: #fff; border: 1px solid #333; border-radius: 4px; cursor: pointer; font-family: monospace; font-size: 14px; }}
        .keypad-btn:hover {{ background: #1a4a80; border-color: #ba68c8; }}
        .keypad-btn .letter {{ display: block; font-size: 10px; color: #aaa; }}
        .keypad-btn .glyph {{ font-size: 22px; line-height: 1; letter-spacing: 1px; }}
        .keypad-hint {{ font-size: 10px; color: #aaa; margin-top: 8px; line-height: 1.4; }}
        .keypad-step {{ color: #ba68c8; font-weight: bold; font-size: 12px; margin-bottom: 6px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Ogham Dataset Curator</h1>
        <div class="stats">
            Total: <span id="total">0</span> |
            Keep: <span id="n-keep" style="color:#4caf50">0</span> |
            Traced: <span id="n-traced" style="color:#ba68c8">0</span> |
            Enhance: <span id="n-enhance" style="color:#ff9800">0</span> |
            Drop: <span id="n-drop" style="color:#f44336">0</span> |
            Pending: <span id="n-pending">0</span>
        </div>
    </div>
    <div class="container">
        <div class="sidebar">
            <div class="filter-row">
                <button class="filter-btn active" data-filter="all">All</button>
                <button class="filter-btn" data-filter="pending">Pending</button>
                <button class="filter-btn" data-filter="keep">Keep</button>
                <button class="filter-btn" data-filter="traced">Traced</button>
                <button class="filter-btn" data-filter="enhance">Enhance</button>
                <button class="filter-btn" data-filter="drop">Drop</button>
            </div>
            <div id="sidebar-list"></div>
        </div>
        <div class="main">
            <!-- Toolbar -->
            <div class="toolbar">
                <button class="tool-btn" onclick="rotateImage(-90)" title="Rotate left">&#8634; Rotate L</button>
                <button class="tool-btn" onclick="rotateImage(90)" title="Rotate right">&#8635; Rotate R</button>
                <button class="tool-btn" onclick="rotateImage(180)" title="Flip 180">&#8693; Flip</button>
                <div class="tool-sep"></div>
                <button class="tool-btn" onclick="toggleInvert()" id="btn-invert" title="Invert colors">&#9680; Invert</button>
                <button class="tool-btn" onclick="toggleGreyscale()" id="btn-grey" title="Greyscale">&#9641; Grey</button>
                <div class="tool-sep"></div>
                <span class="tool-label">Brightness</span>
                <input type="range" class="tool-slider" id="brightness" min="50" max="250" value="100" oninput="applyFilters()">
                <span class="tool-label">Contrast</span>
                <input type="range" class="tool-slider" id="contrast" min="50" max="300" value="100" oninput="applyFilters()">
                <span class="tool-label">Sharpen</span>
                <input type="range" class="tool-slider" id="sharpen" min="0" max="100" value="0" oninput="applyFilters()">
                <div class="tool-sep"></div>
                <button class="tool-btn" onclick="startCrop()" id="btn-crop" title="Crop tool">&#9986; Crop</button>
                <button class="tool-btn" onclick="applyCrop()" id="btn-apply-crop" style="display:none">Apply Crop</button>
                <button class="tool-btn" onclick="cancelCrop()" id="btn-cancel-crop" style="display:none">Cancel</button>
                <div class="tool-sep"></div>
                <button class="tool-btn" onclick="resetImage()" title="Reset all edits">&#8635; Reset</button>
                <button class="tool-btn" onclick="toggleView()" id="btn-view" title="Toggle processed view">&#9633; Show Processed</button>
                <div class="tool-sep"></div>
                <button class="tool-btn" onclick="zoomIn()">+</button>
                <button class="tool-btn" onclick="zoomOut()">-</button>
                <button class="tool-btn" onclick="zoomFit()">Fit</button>
                <div class="tool-sep"></div>
                <button class="tool-btn" onclick="toggleStyleMatch()" id="btn-style" title="Style match mode" style="background:#4CAF50;color:#fff">&#9881; Style Match</button>
                <button class="tool-btn" onclick="toggleTraceMode()" id="btn-trace" title="Trace mode (T)" style="background:#ba68c8;color:#fff">&#9997; Trace</button>
            </div>

            <!-- Style Match toolbar (hidden by default) -->
            <div class="toolbar" id="style-toolbar" style="display:none; background:#2e7d32;">
                <span class="tool-label" style="color:#fff;font-weight:bold">STYLE MATCH:</span>
                <span class="tool-label">Scale</span>
                <input type="range" class="tool-slider" id="sm-scale" min="10" max="80" value="30" oninput="updateStyleMatch()">
                <span class="tool-label" id="sm-scale-val">0.30</span>
                <span class="tool-label">BG</span>
                <input type="range" class="tool-slider" id="sm-bg" min="100" max="230" value="180" oninput="updateStyleMatch()">
                <span class="tool-label" id="sm-bg-val">180</span>
                <span class="tool-label">Stroke</span>
                <input type="range" class="tool-slider" id="sm-stroke" min="20" max="160" value="80" oninput="updateStyleMatch()">
                <span class="tool-label" id="sm-stroke-val">80</span>
                <span class="tool-label">Blur</span>
                <input type="range" class="tool-slider" id="sm-blur" min="0" max="50" value="10" oninput="updateStyleMatch()">
                <span class="tool-label" id="sm-blur-val">1.0</span>
                <span class="tool-label">Noise</span>
                <input type="range" class="tool-slider" id="sm-noise" min="0" max="30" value="8" oninput="updateStyleMatch()">
                <span class="tool-label" id="sm-noise-val">8</span>
                <span class="tool-label">Contrast Red.</span>
                <input type="range" class="tool-slider" id="sm-contrast" min="0" max="80" value="40" oninput="updateStyleMatch()">
                <span class="tool-label" id="sm-contrast-val">0.40</span>
                <div class="tool-sep"></div>
                <button class="tool-btn" onclick="applyStyleToCanvas()" style="background:#fff;color:#2e7d32;font-weight:bold">Apply to Canvas</button>
            </div>

            <!-- Trace toolbar (hidden by default) -->
            <div class="toolbar" id="trace-toolbar" style="display:none; background:#4a148c;">
                <span class="tool-label" style="color:#fff;font-weight:bold">TRACE:</span>
                <button class="tool-btn" onclick="setTraceMode('freeform')" id="btn-tm-freeform" style="background:#ba68c8;color:#fff">Freeform</button>
                <button class="tool-btn" onclick="setTraceMode('assisted')" id="btn-tm-assisted">Assisted</button>
                <div class="tool-sep"></div>
                <button class="tool-btn" onclick="setTraceTool('brush')" id="btn-tool-brush" style="background:#ba68c8;color:#fff">&#9997; Brush</button>
                <button class="tool-btn" onclick="setTraceTool('eraser')" id="btn-tool-eraser">&#9003; Eraser</button>
                <div class="tool-sep"></div>
                <span class="tool-label">Size</span>
                <input type="range" class="tool-slider" id="brush-size" min="2" max="14" value="6" oninput="updateBrushSizeLabel()">
                <span class="tool-label" id="brush-size-val">6</span>
                <div class="tool-sep"></div>
                <button class="tool-btn" onclick="undoTrace()" title="Undo last stroke">&#8634; Undo</button>
                <button class="tool-btn" onclick="clearTrace()" title="Clear all">&#10007; Clear</button>
                <div class="tool-sep"></div>
                <button class="tool-btn" onclick="renderTrace()" style="background:#fff;color:#4a148c;font-weight:bold" title="Render as synthetic">&#9654; Render</button>
                <button class="tool-btn" onclick="saveTraced()" style="background:#4caf50;color:#fff;font-weight:bold" title="Save as traced">Save Traced (0)</button>
            </div>

            <!-- Image area -->
            <div class="image-area">
                <div class="image-panel" id="main-panel">
                    <div class="image-panel-label" id="panel-label">Raw Image</div>
                    <div class="image-container" id="image-container">
                        <canvas id="main-canvas"></canvas>
                        <canvas id="trace-canvas"></canvas>
                        <div class="crop-overlay" id="crop-overlay" style="display:none"></div>
                    </div>
                </div>
                <div class="keypad-column" id="keypad-column">
                    <div class="keypad" id="keypad">
                        <div class="keypad-step" id="keypad-step">Step 1: click the stem line start point</div>
                        <h4>B-aicme (below stem)</h4>
                        <div class="keypad-row" id="kp-b"></div>
                        <h4>H-aicme (above stem)</h4>
                        <div class="keypad-row" id="kp-h"></div>
                        <h4>M-aicme (across stem)</h4>
                        <div class="keypad-row" id="kp-m"></div>
                        <h4>A-aicme (vowels on stem)</h4>
                        <div class="keypad-row" id="kp-a"></div>
                        <h4>Special</h4>
                        <div class="keypad-row">
                            <button class="keypad-btn" onclick="insertSpace()" style="flex:2"><span class="letter">SPACE</span><span class="glyph">&nbsp;</span></button>
                            <button class="keypad-btn" onclick="undoAssistedChar()"><span class="letter">UNDO</span><span class="glyph">&#8634;</span></button>
                        </div>
                        <div class="keypad-hint">Click position on stem, then pick character. Vowels are short notches on the stem.</div>
                    </div>
                </div>
                <div class="image-panel" id="traced-panel" style="display:none">
                    <div class="image-panel-label">Traced (synthetic-style)</div>
                    <div class="image-container">
                        <img id="traced-image" src="" style="max-width:100%;max-height:100%" />
                    </div>
                </div>
                <div class="divider" id="divider" style="display:none"></div>
                <div class="image-panel" id="processed-panel" style="display:none">
                    <div class="image-panel-label">Processed</div>
                    <div class="image-container">
                        <img id="processed-image" src="" />
                    </div>
                </div>
                <div class="image-panel" id="style-panel" style="display:none">
                    <div class="image-panel-label">Style Matched (server-side)</div>
                    <div class="image-container">
                        <img id="style-image" src="" style="max-width:100%;max-height:100%" />
                    </div>
                </div>
                <div class="image-panel" id="synth-panel" style="display:none">
                    <div class="image-panel-label">Synthetic Reference</div>
                    <div class="image-container">
                        <img id="synth-image" src="" style="max-width:100%;max-height:100%" />
                    </div>
                </div>
            </div>

            <!-- Controls -->
            <div class="controls">
                <div class="info-row">
                    <label>Stone:</label><span class="value" id="info-stone"></span>
                    <label>County:</label><span class="value" id="info-county"></span>
                    <label>Source:</label><span class="value" id="info-source"></span>
                    <label>File:</label><span class="value" id="info-image" style="font-size:11px"></span>
                </div>
                <div class="info-row">
                    <label>Transcription:</label>
                    <input class="notes-input ogham-text" id="info-transcription" style="font-size:20px;letter-spacing:2px;flex:1;max-width:500px" oninput="saveTranscription()" />
                    <span class="value" id="info-original-short" style="font-size:11px;color:#666;margin-left:10px"></span>
                    <button class="tool-btn" onclick="resetTranscription()" style="margin-left:5px" title="Reset to original">Reset</button>
                </div>
                <div class="button-row">
                    <button class="btn btn-keep" onclick="setStatus('keep')">Keep (1)</button>
                    <button class="btn btn-enhance" onclick="setStatus('enhance')">Enhance (2)</button>
                    <button class="btn btn-drop" onclick="setStatus('drop')">Drop (3)</button>
                    <input class="notes-input" id="notes" placeholder="Notes..." oninput="saveNotes()" />
                    <button class="btn btn-nav" onclick="navigate(-1)">&#9664; Prev</button>
                    <button class="btn btn-nav" onclick="navigate(1)">Next &#9654;</button>
                    <span class="progress" id="progress"></span>
                    <button class="btn btn-export" onclick="exportResults()">Export</button>
                </div>
                <div class="keyboard-hint">Keys: 1=Keep 2=Enhance 3=Drop T=Trace | A/D=Nav | R=Rotate | I=Invert | G=Grey | P=Toggle Processed | +/-=Zoom | 0=Reset | Ctrl+Z=Undo trace</div>
            </div>
        </div>
    </div>

    <script>
        const stones = {stones_json};
        let curation = {curation_json};
        let currentIdx = 0;
        let currentFilter = 'all';

        // Image state
        let rotation = 0;
        let inverted = false;
        let greyscale = false;
        let zoom = 1;
        let showProcessed = false;
        let cropping = false;
        let cropStart = null;
        let cropEnd = null;
        let originalImage = null;
        let currentImage = null;

        const canvas = document.getElementById('main-canvas');
        const ctx = canvas.getContext('2d');

        function getFilteredIndices() {{
            if (currentFilter === 'all') return stones.map((_, i) => i);
            return stones.map((s, i) => [s, i])
                .filter(([s, i]) => {{
                    const key = s.stone_id + '/' + s.image_name;
                    const status = curation[key]?.status || 'pending';
                    return status === currentFilter;
                }})
                .map(([_, i]) => i);
        }}

        function updateStats() {{
            let keep = 0, enhance = 0, drop = 0, traced = 0, pending = 0;
            stones.forEach(s => {{
                const key = s.stone_id + '/' + s.image_name;
                const status = curation[key]?.status || 'pending';
                if (status === 'keep') keep++;
                else if (status === 'enhance') enhance++;
                else if (status === 'drop') drop++;
                else if (status === 'traced') traced++;
                else pending++;
            }});
            document.getElementById('total').textContent = stones.length;
            document.getElementById('n-keep').textContent = keep;
            document.getElementById('n-enhance').textContent = enhance;
            document.getElementById('n-drop').textContent = drop;
            document.getElementById('n-traced').textContent = traced;
            document.getElementById('n-pending').textContent = pending;
        }}

        function renderSidebar() {{
            const list = document.getElementById('sidebar-list');
            const indices = getFilteredIndices();
            list.innerHTML = indices.map(i => {{
                const s = stones[i];
                const key = s.stone_id + '/' + s.image_name;
                const status = curation[key]?.status || 'pending';
                const active = i === currentIdx ? 'active' : '';
                const src = s.source === 'raw' ? '&#128247;' : '&#9881;';
                return `<div class="sidebar-item ${{active}}" onclick="goTo(${{i}})">
                    <div class="stone-id">${{src}} ${{s.stone_id}}</div>
                    <div class="status status-${{status}}">${{status.toUpperCase()}} - ${{s.image_name}}</div>
                </div>`;
            }}).join('');
        }}

        function loadImage(src) {{
            return new Promise((resolve) => {{
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = () => resolve(null);
                img.src = src;
            }});
        }}

        async function goTo(idx) {{
            currentIdx = idx;
            const s = stones[idx];
            const key = s.stone_id + '/' + s.image_name;

            // Reset image state
            rotation = 0;
            inverted = false;
            greyscale = false;
            zoom = 1;
            document.getElementById('brightness').value = 100;
            document.getElementById('contrast').value = 100;
            document.getElementById('sharpen').value = 0;
            document.getElementById('btn-invert').classList.remove('active');
            document.getElementById('btn-grey').classList.remove('active');

            // Load image: curated version if previously kept/enhanced, otherwise raw
            const curationEntry = curation[key] || {{}};
            const hasCurated = (curationEntry.status === 'keep' || curationEntry.status === 'enhance') && curationEntry.curated_path;
            const imgSrc = hasCurated
                ? '/image?path=' + encodeURIComponent(curationEntry.curated_path)
                : '/image?path=' + encodeURIComponent(s.image_path);
            originalImage = await loadImage(imgSrc);
            currentImage = originalImage;
            drawCanvas();

            // Show label indicating source
            const srcLabel = hasCurated ? 'Curated (previously edited)' : (s.source === 'raw' ? 'Raw Image' : 'Processed Image');
            document.getElementById('panel-label').textContent = srcLabel;

            // Load processed reference
            if (s.processed_path) {{
                document.getElementById('processed-image').src = '/image?path=' + encodeURIComponent(s.processed_path);
            }}

            // Update info
            document.getElementById('info-stone').textContent = s.stone_id;
            document.getElementById('info-county').textContent = s.county;
            document.getElementById('info-source').textContent = s.source.toUpperCase();
            document.getElementById('info-image').textContent = s.image_name;
            // Show edited transcription if exists, otherwise original
            const editedTranscription = curation[key]?.transcription || s.transcription;
            document.getElementById('info-transcription').value = editedTranscription;
            document.getElementById('info-original-short').textContent = editedTranscription !== s.transcription ? '(edited) orig: ' + s.transcription.slice(0, 20) + '...' : '';
            document.getElementById('notes').value = curation[key]?.notes || '';

            // Update button states
            const status = curation[key]?.status || '';
            document.querySelectorAll('.btn-keep,.btn-enhance,.btn-drop').forEach(b => b.classList.remove('selected'));
            if (status) document.querySelector('.btn-' + status)?.classList.add('selected');

            document.getElementById('progress').textContent = `${{idx + 1}} / ${{stones.length}}`;
            renderSidebar();

            // If in trace mode, reset + attempt to reload any saved strokes
            if (typeof traceMode !== 'undefined' && traceMode) {{
                resetTraceState();
                // Delay until the canvas has finished drawing
                setTimeout(() => {{
                    syncTraceCanvasSize();
                    fetch('/trace-data?key=' + encodeURIComponent(key))
                        .then(r => r.json())
                        .then(entry => {{
                            if (entry && entry.strokes) {{
                                traceStrokes = entry.strokes;
                                stemP1 = entry.stem_p1 || null;
                                stemP2 = entry.stem_p2 || null;
                                if (stemP1 && stemP2) assistedStep = 2;
                                updateAssistedHint();
                                updateTraceCounter();
                                drawTraceOverlay();
                            }}
                        }});
                }}, 50);
            }}
        }}

        function drawCanvas() {{
            if (!currentImage) return;

            const br = parseFloat(document.getElementById('brightness').value) / 100;
            const co = parseFloat(document.getElementById('contrast').value) / 100;
            const sh = parseFloat(document.getElementById('sharpen').value);

            // Handle rotation
            const rad = rotation * Math.PI / 180;
            const isRotated = rotation % 180 !== 0;
            const w = isRotated ? currentImage.height : currentImage.width;
            const h = isRotated ? currentImage.width : currentImage.height;

            canvas.width = w;
            canvas.height = h;

            ctx.save();
            ctx.translate(w / 2, h / 2);
            ctx.rotate(rad);
            ctx.drawImage(currentImage, -currentImage.width / 2, -currentImage.height / 2);
            ctx.restore();

            // Apply filters via pixel manipulation
            if (inverted || greyscale || br !== 1 || co !== 1 || sh > 0) {{
                const imageData = ctx.getImageData(0, 0, w, h);
                const d = imageData.data;

                // Sharpen: unsharp mask (need original copy)
                let orig = null;
                if (sh > 0) {{
                    orig = new Uint8ClampedArray(d);
                }}

                for (let i = 0; i < d.length; i += 4) {{
                    let r = d[i], g = d[i+1], b = d[i+2];

                    // Brightness
                    r *= br; g *= br; b *= br;

                    // Contrast
                    r = ((r / 255 - 0.5) * co + 0.5) * 255;
                    g = ((g / 255 - 0.5) * co + 0.5) * 255;
                    b = ((b / 255 - 0.5) * co + 0.5) * 255;

                    // Greyscale
                    if (greyscale) {{
                        const avg = 0.299 * r + 0.587 * g + 0.114 * b;
                        r = g = b = avg;
                    }}

                    // Invert
                    if (inverted) {{
                        r = 255 - r; g = 255 - g; b = 255 - b;
                    }}

                    d[i] = Math.max(0, Math.min(255, r));
                    d[i+1] = Math.max(0, Math.min(255, g));
                    d[i+2] = Math.max(0, Math.min(255, b));
                }}

                // Sharpen pass (unsharp mask using 3x3 neighbor average)
                if (sh > 0 && orig) {{
                    const amount = sh / 50;  // 0-2 range
                    const result = new Uint8ClampedArray(d);
                    for (let y = 1; y < h - 1; y++) {{
                        for (let x = 1; x < w - 1; x++) {{
                            const idx = (y * w + x) * 4;
                            for (let c = 0; c < 3; c++) {{
                                // Average of 3x3 neighbors from original
                                let blur = 0;
                                for (let dy = -1; dy <= 1; dy++) {{
                                    for (let dx = -1; dx <= 1; dx++) {{
                                        blur += d[((y+dy) * w + (x+dx)) * 4 + c];
                                    }}
                                }}
                                blur /= 9;
                                // Unsharp mask: original + amount * (original - blur)
                                const sharp = d[idx + c] + amount * (d[idx + c] - blur);
                                result[idx + c] = Math.max(0, Math.min(255, sharp));
                            }}
                        }}
                    }}
                    for (let i = 0; i < d.length; i++) d[i] = result[i];
                }}

                ctx.putImageData(imageData, 0, 0);
            }}

            // Zoom
            canvas.style.transform = `scale(${{zoom}})`;

            // Keep trace overlay aligned when the main canvas is redrawn
            if (typeof traceMode !== 'undefined' && traceMode) {{
                setTimeout(() => {{ syncTraceCanvasSize(); drawTraceOverlay(); }}, 0);
            }}
        }}

        function rotateImage(deg) {{
            rotation = (rotation + deg) % 360;
            drawCanvas();
        }}

        function toggleInvert() {{
            inverted = !inverted;
            document.getElementById('btn-invert').classList.toggle('active');
            drawCanvas();
        }}

        function toggleGreyscale() {{
            greyscale = !greyscale;
            document.getElementById('btn-grey').classList.toggle('active');
            drawCanvas();
        }}

        function applyFilters() {{ drawCanvas(); }}

        function zoomIn() {{ zoom = Math.min(zoom * 1.3, 5); drawCanvas(); }}
        function zoomOut() {{ zoom = Math.max(zoom / 1.3, 0.2); drawCanvas(); }}
        function zoomFit() {{ zoom = 1; drawCanvas(); }}

        function resetImage() {{
            rotation = 0;
            inverted = false;
            greyscale = false;
            zoom = 1;
            document.getElementById('brightness').value = 100;
            document.getElementById('contrast').value = 100;
            document.getElementById('sharpen').value = 0;
            document.getElementById('btn-invert').classList.remove('active');
            document.getElementById('btn-grey').classList.remove('active');
            currentImage = originalImage;
            drawCanvas();
        }}

        function toggleView() {{
            showProcessed = !showProcessed;
            document.getElementById('processed-panel').style.display = showProcessed ? 'flex' : 'none';
            document.getElementById('divider').style.display = showProcessed ? 'block' : 'none';
            document.getElementById('btn-view').classList.toggle('active');
            document.getElementById('btn-view').innerHTML = showProcessed ? '&#9633; Hide Processed' : '&#9633; Show Processed';
        }}

        // Crop functionality
        let cropRect = null;
        const container = document.getElementById('image-container');
        const overlay = document.getElementById('crop-overlay');

        function startCrop() {{
            cropping = !cropping;
            document.getElementById('btn-crop').classList.toggle('active');
            document.getElementById('btn-apply-crop').style.display = cropping ? 'inline-block' : 'none';
            document.getElementById('btn-cancel-crop').style.display = cropping ? 'inline-block' : 'none';
            if (!cropping) {{
                overlay.style.display = 'none';
                cropStart = cropEnd = null;
            }}
        }}

        function cancelCrop() {{
            cropping = false;
            document.getElementById('btn-crop').classList.remove('active');
            document.getElementById('btn-apply-crop').style.display = 'none';
            document.getElementById('btn-cancel-crop').style.display = 'none';
            overlay.style.display = 'none';
            cropStart = cropEnd = null;
        }}

        function applyCrop() {{
            if (!cropStart || !cropEnd) return;

            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;

            const x1 = Math.min(cropStart.x, cropEnd.x) * scaleX;
            const y1 = Math.min(cropStart.y, cropEnd.y) * scaleY;
            const x2 = Math.max(cropStart.x, cropEnd.x) * scaleX;
            const y2 = Math.max(cropStart.y, cropEnd.y) * scaleY;

            const cw = x2 - x1;
            const ch = y2 - y1;
            if (cw < 10 || ch < 10) return;

            const cropData = ctx.getImageData(x1, y1, cw, ch);
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = cw;
            tempCanvas.height = ch;
            tempCanvas.getContext('2d').putImageData(cropData, 0, 0);

            const img = new Image();
            img.onload = () => {{
                currentImage = img;
                rotation = 0;
                drawCanvas();
            }};
            img.src = tempCanvas.toDataURL();

            cancelCrop();
        }}

        let isDragging = false;

        container.addEventListener('mousedown', (e) => {{
            if (!cropping) return;
            isDragging = true;
            const rect = canvas.getBoundingClientRect();
            cropStart = {{ x: e.clientX - rect.left, y: e.clientY - rect.top }};
            cropEnd = null;
            overlay.style.display = 'block';
        }});

        container.addEventListener('mousemove', (e) => {{
            if (!cropping || !isDragging || !cropStart) return;
            const rect = canvas.getBoundingClientRect();
            cropEnd = {{ x: e.clientX - rect.left, y: e.clientY - rect.top }};

            const left = Math.min(cropStart.x, cropEnd.x) + rect.left - container.getBoundingClientRect().left;
            const top = Math.min(cropStart.y, cropEnd.y) + rect.top - container.getBoundingClientRect().top;
            const width = Math.abs(cropEnd.x - cropStart.x);
            const height = Math.abs(cropEnd.y - cropStart.y);

            overlay.style.left = left + 'px';
            overlay.style.top = top + 'px';
            overlay.style.width = width + 'px';
            overlay.style.height = height + 'px';
        }});

        container.addEventListener('mouseup', () => {{
            isDragging = false;
            if (!cropping) return;
            // Crop selection complete — click Apply Crop to confirm
        }});

        // Status & navigation
        function setStatus(status) {{
            const s = stones[currentIdx];
            const key = s.stone_id + '/' + s.image_name;
            if (!curation[key]) curation[key] = {{}};
            curation[key].status = status;
            curation[key].stone_id = s.stone_id;
            curation[key].image_name = s.image_name;

            // Save edited image to curated/ folder for keep and enhance
            if (status === 'keep' || status === 'enhance') {{
                const imageData = canvas.toDataURL('image/png');
                fetch('/save-image', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        stone_id: s.stone_id,
                        image_name: s.image_name,
                        image_data: imageData,
                    }}),
                }}).then(r => r.json()).then(d => {{
                    if (d.ok) {{
                        curation[key].curated_path = d.path;
                        saveCuration();
                    }}
                }});
            }}

            saveCuration();
            updateStats();
            goTo(currentIdx);
            setTimeout(() => navigate(1), 200);
        }}

        function saveTranscription() {{
            const s = stones[currentIdx];
            const key = s.stone_id + '/' + s.image_name;
            if (!curation[key]) curation[key] = {{}};
            const edited = document.getElementById('info-transcription').value;
            curation[key].transcription = edited;
            curation[key].original_transcription = s.transcription;
            curation[key].stone_id = s.stone_id;
            curation[key].image_name = s.image_name;
            // Show edit indicator
            document.getElementById('info-original-short').textContent =
                edited !== s.transcription ? '(edited) orig: ' + s.transcription.slice(0, 20) + '...' : '';
            saveCuration();
        }}

        function resetTranscription() {{
            const s = stones[currentIdx];
            document.getElementById('info-transcription').value = s.transcription;
            document.getElementById('info-original-short').textContent = '';
            const key = s.stone_id + '/' + s.image_name;
            if (curation[key]) {{
                delete curation[key].transcription;
                delete curation[key].original_transcription;
                saveCuration();
            }}
        }}

        function saveNotes() {{
            const s = stones[currentIdx];
            const key = s.stone_id + '/' + s.image_name;
            if (!curation[key]) curation[key] = {{}};
            curation[key].notes = document.getElementById('notes').value;
            curation[key].stone_id = s.stone_id;
            curation[key].image_name = s.image_name;
            saveCuration();
        }}

        function navigate(dir) {{
            const indices = getFilteredIndices();
            if (indices.length === 0) return;
            const currentFilterIdx = indices.indexOf(currentIdx);
            let next;
            if (currentFilterIdx === -1) {{
                next = dir > 0 ? 0 : indices.length - 1;
            }} else {{
                next = (currentFilterIdx + dir + indices.length) % indices.length;
            }}
            goTo(indices[next]);
        }}

        function saveCuration() {{
            fetch('/save', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(curation),
            }});
        }}

        function exportResults() {{
            const summary = {{ keep: [], enhance: [], drop: [] }};
            Object.entries(curation).forEach(([key, val]) => {{
                if (val.status && summary[val.status]) {{
                    summary[val.status].push({{ key, ...val }});
                }}
            }});
            const text = JSON.stringify(summary, null, 2);
            const blob = new Blob([text], {{ type: 'application/json' }});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'curation_export.json';
            a.click();
            alert(`Exported: ${{summary.keep.length}} keep, ${{summary.enhance.length}} enhance, ${{summary.drop.length}} drop`);
        }}

        // Filter buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentFilter = btn.dataset.filter;
                renderSidebar();
            }});
        }});

        // Style match functions
        let styleMatchMode = false;
        let styleMatchDebounce = null;

        function toggleStyleMatch() {{
            styleMatchMode = !styleMatchMode;
            document.getElementById('style-toolbar').style.display = styleMatchMode ? 'flex' : 'none';
            document.getElementById('style-panel').style.display = styleMatchMode ? 'flex' : 'none';
            document.getElementById('synth-panel').style.display = styleMatchMode ? 'flex' : 'none';
            document.getElementById('btn-style').classList.toggle('active');
            if (styleMatchMode) {{
                // Load synthetic reference
                document.getElementById('synth-image').src = '/synth-sample';
                updateStyleMatch();
            }}
        }}

        function updateStyleMatch() {{
            // Update slider value labels
            document.getElementById('sm-scale-val').textContent = (document.getElementById('sm-scale').value / 100).toFixed(2);
            document.getElementById('sm-bg-val').textContent = document.getElementById('sm-bg').value;
            document.getElementById('sm-stroke-val').textContent = document.getElementById('sm-stroke').value;
            document.getElementById('sm-blur-val').textContent = (document.getElementById('sm-blur').value / 10).toFixed(1);
            document.getElementById('sm-noise-val').textContent = document.getElementById('sm-noise').value;
            document.getElementById('sm-contrast-val').textContent = (document.getElementById('sm-contrast').value / 100).toFixed(2);

            // Debounce server call
            clearTimeout(styleMatchDebounce);
            styleMatchDebounce = setTimeout(() => {{
                const s = stones[currentIdx];
                const key = s.stone_id + '/' + s.image_name;
                const curationEntry = curation[key] || {{}};
                const imgPath = (curationEntry.curated_path) ? curationEntry.curated_path : s.image_path;
                const params = new URLSearchParams({{
                    path: imgPath,
                    scale: document.getElementById('sm-scale').value / 100,
                    bg: document.getElementById('sm-bg').value,
                    stroke: document.getElementById('sm-stroke').value,
                    blur: document.getElementById('sm-blur').value / 10,
                    noise: document.getElementById('sm-noise').value,
                    contrast: document.getElementById('sm-contrast').value / 100,
                }});
                document.getElementById('style-image').src = '/style-match?' + params.toString();
            }}, 200);
        }}

        function applyStyleToCanvas() {{
            // Load the style-matched image onto the main canvas
            const styleImg = document.getElementById('style-image');
            if (styleImg.complete && styleImg.naturalWidth > 0) {{
                const img = new Image();
                img.onload = () => {{
                    currentImage = img;
                    rotation = 0;
                    inverted = false;
                    greyscale = false;
                    document.getElementById('brightness').value = 100;
                    document.getElementById('contrast').value = 100;
                    document.getElementById('sharpen').value = 0;
                    drawCanvas();
                }};
                img.src = styleImg.src;
            }}
        }}

        // Keyboard shortcuts
        document.addEventListener('keydown', e => {{
            if (e.target.tagName === 'INPUT') return;
            if (e.key === '1') setStatus('keep');
            if (e.key === '2') setStatus('enhance');
            if (e.key === '3') setStatus('drop');
            if (e.key === 'a' || e.key === 'ArrowLeft') navigate(-1);
            if (e.key === 'd' || e.key === 'ArrowRight') navigate(1);
            if (e.key === 'r') rotateImage(90);
            if (e.key === 'i') toggleInvert();
            if (e.key === 'g') toggleGreyscale();
            if (e.key === 'p') toggleView();
            if (e.key === '=' || e.key === '+') zoomIn();
            if (e.key === '-') zoomOut();
            if (e.key === '0') resetImage();
            if (e.key === 't' || e.key === 'T') toggleTraceMode();
            if (traceMode && (e.key === 'z' || e.key === 'Z') && (e.ctrlKey || e.metaKey)) {{ e.preventDefault(); undoTrace(); }}
        }});

        // ============================================================
        // TRACE MODE
        // ============================================================
        const OGHAM_CHARS = {{
            B: {{glyph: 'ᚁ', aicme: 'b', count: 1}},
            L: {{glyph: 'ᚂ', aicme: 'b', count: 2}},
            F: {{glyph: 'ᚃ', aicme: 'b', count: 3}},
            S: {{glyph: 'ᚄ', aicme: 'b', count: 4}},
            N: {{glyph: 'ᚅ', aicme: 'b', count: 5}},
            H: {{glyph: 'ᚆ', aicme: 'h', count: 1}},
            D: {{glyph: 'ᚇ', aicme: 'h', count: 2}},
            T: {{glyph: 'ᚈ', aicme: 'h', count: 3}},
            C: {{glyph: 'ᚉ', aicme: 'h', count: 4}},
            Q: {{glyph: 'ᚊ', aicme: 'h', count: 5}},
            M: {{glyph: 'ᚋ', aicme: 'm', count: 1}},
            G: {{glyph: 'ᚌ', aicme: 'm', count: 2}},
            NG: {{glyph: 'ᚍ', aicme: 'm', count: 3}},
            Z: {{glyph: 'ᚎ', aicme: 'm', count: 4}},
            R: {{glyph: 'ᚏ', aicme: 'm', count: 5}},
            A: {{glyph: 'ᚐ', aicme: 'a', count: 1}},
            O: {{glyph: 'ᚑ', aicme: 'a', count: 2}},
            U: {{glyph: 'ᚒ', aicme: 'a', count: 3}},
            E: {{glyph: 'ᚓ', aicme: 'a', count: 4}},
            I: {{glyph: 'ᚔ', aicme: 'a', count: 5}},
        }};

        let traceMode = false;
        let traceSubMode = 'freeform';  // 'freeform' | 'assisted'
        let traceTool = 'brush';        // 'brush' | 'eraser'
        let traceDrawing = false;
        let traceStrokes = [];          // {{kind, ...}}; freeform={{kind:'free', points, width}}, char={{kind:'char', charKey, t}}
        let currentFreeStroke = null;

        // Assisted-mode state
        let assistedStep = 0;  // 0: stem p1, 1: stem p2, 2: picking position, 3: picking char
        let stemP1 = null, stemP2 = null;
        let pendingCharPos = null;  // projected anchor on the stem

        const traceCanvas = document.getElementById('trace-canvas');
        const traceCtx = traceCanvas.getContext('2d');

        function buildKeypad() {{
            const rows = {{b: 'kp-b', h: 'kp-h', m: 'kp-m', a: 'kp-a'}};
            const order = {{
                b: ['B', 'L', 'F', 'S', 'N'],
                h: ['H', 'D', 'T', 'C', 'Q'],
                m: ['M', 'G', 'NG', 'Z', 'R'],
                a: ['A', 'O', 'U', 'E', 'I'],
            }};
            Object.keys(order).forEach(aicme => {{
                const row = document.getElementById(rows[aicme]);
                row.innerHTML = order[aicme].map(k => {{
                    const c = OGHAM_CHARS[k];
                    return `<button class="keypad-btn" onclick="pickChar('${{k}}')"><span class="letter">${{k}}</span><span class="glyph">${{c.glyph}}</span></button>`;
                }}).join('');
            }});
        }}

        function toggleTraceMode() {{
            traceMode = !traceMode;
            document.getElementById('trace-toolbar').style.display = traceMode ? 'flex' : 'none';
            document.getElementById('btn-trace').classList.toggle('active', traceMode);
            traceCanvas.classList.toggle('active', traceMode);
            document.getElementById('keypad-column').classList.toggle('active', traceMode && traceSubMode === 'assisted');
            document.getElementById('traced-panel').style.display = traceMode ? 'flex' : 'none';
            if (traceMode) {{
                syncTraceCanvasSize();
                resetTraceState();
                drawTraceOverlay();
            }}
        }}

        function setTraceMode(mode) {{
            traceSubMode = mode;
            document.getElementById('btn-tm-freeform').style.background = mode === 'freeform' ? '#ba68c8' : '#16213e';
            document.getElementById('btn-tm-freeform').style.color = mode === 'freeform' ? '#fff' : '#ccc';
            document.getElementById('btn-tm-assisted').style.background = mode === 'assisted' ? '#ba68c8' : '#16213e';
            document.getElementById('btn-tm-assisted').style.color = mode === 'assisted' ? '#fff' : '#ccc';
            document.getElementById('keypad-column').classList.toggle('active', mode === 'assisted');
            resetAssistedFlow();
        }}

        function setTraceTool(tool) {{
            traceTool = tool;
            document.getElementById('btn-tool-brush').style.background = tool === 'brush' ? '#ba68c8' : '#16213e';
            document.getElementById('btn-tool-brush').style.color = tool === 'brush' ? '#fff' : '#ccc';
            document.getElementById('btn-tool-eraser').style.background = tool === 'eraser' ? '#ba68c8' : '#16213e';
            document.getElementById('btn-tool-eraser').style.color = tool === 'eraser' ? '#fff' : '#ccc';
        }}

        function updateBrushSizeLabel() {{
            document.getElementById('brush-size-val').textContent = document.getElementById('brush-size').value;
        }}

        function resetTraceState() {{
            traceStrokes = [];
            currentFreeStroke = null;
            resetAssistedFlow();
            updateTraceCounter();
        }}

        function resetAssistedFlow() {{
            assistedStep = 0;
            stemP1 = stemP2 = null;
            pendingCharPos = null;
            updateAssistedHint();
        }}

        function updateAssistedHint() {{
            const el = document.getElementById('keypad-step');
            if (assistedStep === 0) el.textContent = 'Step 1: click stem line START point';
            else if (assistedStep === 1) el.textContent = 'Step 2: click stem line END point';
            else if (assistedStep === 2) el.textContent = 'Click position on stem for next character';
            else if (assistedStep === 3) el.textContent = 'Now pick the character from keypad below';
        }}

        function syncTraceCanvasSize() {{
            traceCanvas.width = canvas.width;
            traceCanvas.height = canvas.height;
            // Match visual size to main canvas
            const r = canvas.getBoundingClientRect();
            const cr = canvas.parentElement.getBoundingClientRect();
            traceCanvas.style.width = r.width + 'px';
            traceCanvas.style.height = r.height + 'px';
            traceCanvas.style.left = (r.left - cr.left) + 'px';
            traceCanvas.style.top = (r.top - cr.top) + 'px';
        }}

        function canvasCoordsFromEvent(e) {{
            const rect = traceCanvas.getBoundingClientRect();
            const sx = traceCanvas.width / rect.width;
            const sy = traceCanvas.height / rect.height;
            return {{
                x: (e.clientX - rect.left) * sx,
                y: (e.clientY - rect.top) * sy,
            }};
        }}

        traceCanvas.addEventListener('mousedown', (e) => {{
            if (!traceMode) return;
            const p = canvasCoordsFromEvent(e);
            if (traceSubMode === 'freeform') {{
                traceDrawing = true;
                const width = parseInt(document.getElementById('brush-size').value);
                if (traceTool === 'brush') {{
                    currentFreeStroke = {{kind: 'free', points: [p], width, erase: false}};
                }} else {{
                    currentFreeStroke = {{kind: 'free', points: [p], width: width * 2, erase: true}};
                }}
            }} else {{
                // Assisted
                if (assistedStep === 0) {{
                    stemP1 = p;
                    assistedStep = 1;
                }} else if (assistedStep === 1) {{
                    // Snap to horizontal — Ogham stems in training are always
                    // flat, so force y2 == y1. Clicking by eye otherwise
                    // introduces a slight diagonal that the processor's resize
                    // exaggerates into a very visible tilt.
                    stemP2 = {{x: p.x, y: stemP1.y}};
                    assistedStep = 2;
                }} else if (assistedStep === 2) {{
                    // Project click onto stem line
                    pendingCharPos = projectOntoStem(p);
                    assistedStep = 3;
                }}
                updateAssistedHint();
                drawTraceOverlay();
            }}
        }});

        traceCanvas.addEventListener('mousemove', (e) => {{
            if (!traceMode || traceSubMode !== 'freeform' || !traceDrawing) return;
            const p = canvasCoordsFromEvent(e);
            currentFreeStroke.points.push(p);
            drawTraceOverlay();
        }});

        traceCanvas.addEventListener('mouseup', () => {{
            if (!traceMode || traceSubMode !== 'freeform') return;
            if (currentFreeStroke && currentFreeStroke.points.length > 1) {{
                traceStrokes.push(currentFreeStroke);
                updateTraceCounter();
            }}
            currentFreeStroke = null;
            traceDrawing = false;
            drawTraceOverlay();
        }});

        traceCanvas.addEventListener('mouseleave', () => {{
            if (currentFreeStroke && currentFreeStroke.points.length > 1) {{
                traceStrokes.push(currentFreeStroke);
                updateTraceCounter();
            }}
            currentFreeStroke = null;
            traceDrawing = false;
        }});

        function projectOntoStem(p) {{
            if (!stemP1 || !stemP2) return p;
            const dx = stemP2.x - stemP1.x;
            const dy = stemP2.y - stemP1.y;
            const l2 = dx * dx + dy * dy;
            if (l2 < 1) return stemP1;
            let t = ((p.x - stemP1.x) * dx + (p.y - stemP1.y) * dy) / l2;
            t = Math.max(0, Math.min(1, t));
            return {{x: stemP1.x + dx * t, y: stemP1.y + dy * t, t}};
        }}

        function pickChar(key) {{
            if (traceSubMode !== 'assisted') return;
            if (!pendingCharPos) {{ alert('Click a position on the stem first'); return; }}
            traceStrokes.push({{kind: 'char', charKey: key, t: pendingCharPos.t}});
            appendGlyphToTranscription(OGHAM_CHARS[key].glyph);
            pendingCharPos = null;
            assistedStep = 2;
            updateAssistedHint();
            updateTraceCounter();
            drawTraceOverlay();
        }}

        function insertSpace() {{
            appendGlyphToTranscription(' ');
        }}

        function appendGlyphToTranscription(glyph) {{
            const field = document.getElementById('info-transcription');
            field.value = (field.value || '') + glyph;
            saveTranscription();
        }}

        function undoAssistedChar() {{
            for (let i = traceStrokes.length - 1; i >= 0; i--) {{
                if (traceStrokes[i].kind === 'char') {{
                    traceStrokes.splice(i, 1);
                    // Also pop last glyph from transcription
                    const field = document.getElementById('info-transcription');
                    field.value = field.value.slice(0, -1);
                    saveTranscription();
                    updateTraceCounter();
                    drawTraceOverlay();
                    return;
                }}
            }}
        }}

        function undoTrace() {{
            if (traceStrokes.length === 0) return;
            const last = traceStrokes.pop();
            if (last.kind === 'char') {{
                const field = document.getElementById('info-transcription');
                field.value = field.value.slice(0, -1);
                saveTranscription();
            }}
            updateTraceCounter();
            drawTraceOverlay();
        }}

        function clearTrace() {{
            if (!confirm('Clear all traced strokes?')) return;
            traceStrokes = [];
            currentFreeStroke = null;
            resetAssistedFlow();
            updateTraceCounter();
            drawTraceOverlay();
        }}

        function updateTraceCounter() {{
            const count = traceStrokes.length;
            const btns = document.querySelectorAll('#trace-toolbar .tool-btn');
            btns.forEach(b => {{ if (b.textContent.startsWith('Save Traced')) b.textContent = `Save Traced (${{count}})`; }});
        }}

        function drawTraceOverlay() {{
            traceCtx.clearRect(0, 0, traceCanvas.width, traceCanvas.height);
            // Draw stem guide (assisted mode preview)
            if (traceSubMode === 'assisted') {{
                if (stemP1 && stemP2) {{
                    traceCtx.strokeStyle = 'rgba(186,104,200,0.9)';
                    traceCtx.lineWidth = parseInt(document.getElementById('brush-size').value);
                    traceCtx.beginPath();
                    traceCtx.moveTo(stemP1.x, stemP1.y);
                    traceCtx.lineTo(stemP2.x, stemP2.y);
                    traceCtx.stroke();
                }} else if (stemP1) {{
                    traceCtx.fillStyle = '#ba68c8';
                    traceCtx.beginPath();
                    traceCtx.arc(stemP1.x, stemP1.y, 6, 0, Math.PI * 2);
                    traceCtx.fill();
                }}
                if (pendingCharPos) {{
                    traceCtx.fillStyle = 'rgba(255,255,0,0.8)';
                    traceCtx.beginPath();
                    traceCtx.arc(pendingCharPos.x, pendingCharPos.y, 5, 0, Math.PI * 2);
                    traceCtx.fill();
                }}
            }}
            // Draw all strokes
            traceStrokes.forEach(st => {{
                if (st.kind === 'free') {{
                    drawFreeStroke(traceCtx, st);
                }} else if (st.kind === 'char') {{
                    const segs = charStrokeSegments(st.charKey, st.t);
                    traceCtx.strokeStyle = '#222';
                    traceCtx.lineWidth = parseInt(document.getElementById('brush-size').value);
                    traceCtx.lineCap = 'round';
                    segs.forEach(seg => {{
                        traceCtx.beginPath();
                        traceCtx.moveTo(seg.x1, seg.y1);
                        traceCtx.lineTo(seg.x2, seg.y2);
                        traceCtx.stroke();
                    }});
                }}
            }});
            // In-progress freeform stroke
            if (currentFreeStroke && currentFreeStroke.points.length > 1) {{
                drawFreeStroke(traceCtx, currentFreeStroke);
            }}
        }}

        function drawFreeStroke(ctxRef, st) {{
            if (st.erase) {{
                ctxRef.globalCompositeOperation = 'destination-out';
                ctxRef.strokeStyle = 'rgba(0,0,0,1)';
            }} else {{
                ctxRef.globalCompositeOperation = 'source-over';
                ctxRef.strokeStyle = '#222';
            }}
            ctxRef.lineWidth = st.width;
            ctxRef.lineCap = 'round';
            ctxRef.lineJoin = 'round';
            ctxRef.beginPath();
            ctxRef.moveTo(st.points[0].x, st.points[0].y);
            for (let i = 1; i < st.points.length; i++) {{
                ctxRef.lineTo(st.points[i].x, st.points[i].y);
            }}
            ctxRef.stroke();
            ctxRef.globalCompositeOperation = 'source-over';
        }}

        function charStrokeSegments(charKey, t) {{
            const char = OGHAM_CHARS[charKey];
            if (!char || !stemP1 || !stemP2) return [];
            const dx = stemP2.x - stemP1.x, dy = stemP2.y - stemP1.y;
            const len = Math.hypot(dx, dy) || 1;
            const ux = dx / len, uy = dy / len;       // unit along stem
            const px = -uy, py = ux;                  // unit perpendicular (screen "up"-ish)
            const ax = stemP1.x + ux * t * len;
            const ay = stemP1.y + uy * t * len;
            const spacing = 8;   // px between strokes of same char
            const strokeLen = 28;
            const notchLen = 6;
            const segs = [];
            for (let i = 0; i < char.count; i++) {{
                const off = (i - (char.count - 1) / 2) * spacing;
                const sx = ax + ux * off;
                const sy = ay + uy * off;
                if (char.aicme === 'b') {{
                    segs.push({{x1: sx, y1: sy, x2: sx + px * strokeLen, y2: sy + py * strokeLen}});
                }} else if (char.aicme === 'h') {{
                    segs.push({{x1: sx, y1: sy, x2: sx - px * strokeLen, y2: sy - py * strokeLen}});
                }} else if (char.aicme === 'm') {{
                    const hl = strokeLen * 0.6;
                    segs.push({{x1: sx - px * hl, y1: sy - py * hl, x2: sx + px * hl, y2: sy + py * hl}});
                }} else if (char.aicme === 'a') {{
                    segs.push({{x1: sx - px * notchLen, y1: sy - py * notchLen, x2: sx + px * notchLen, y2: sy + py * notchLen}});
                }}
            }}
            return segs;
        }}

        function flattenStrokesToCanvas() {{
            // Flatten the current trace (stem line + strokes + freehand) to a
            // black-on-transparent canvas matching the trace canvas resolution.
            const flat = document.createElement('canvas');
            flat.width = traceCanvas.width;
            flat.height = traceCanvas.height;
            const fctx = flat.getContext('2d');
            fctx.clearRect(0, 0, flat.width, flat.height);
            fctx.strokeStyle = '#000';
            fctx.fillStyle = '#000';
            fctx.lineCap = 'round';
            fctx.lineJoin = 'round';
            const brush = parseInt(document.getElementById('brush-size').value);

            // Draw the stem line first (assisted mode), so characters sit on top
            if (stemP1 && stemP2) {{
                fctx.lineWidth = brush;
                fctx.beginPath();
                fctx.moveTo(stemP1.x, stemP1.y);
                fctx.lineTo(stemP2.x, stemP2.y);
                fctx.stroke();
            }}

            traceStrokes.forEach(st => {{
                if (st.kind === 'free') {{
                    if (st.erase) return;
                    fctx.lineWidth = st.width;
                    fctx.beginPath();
                    fctx.moveTo(st.points[0].x, st.points[0].y);
                    for (let i = 1; i < st.points.length; i++) fctx.lineTo(st.points[i].x, st.points[i].y);
                    fctx.stroke();
                }} else if (st.kind === 'char') {{
                    const segs = charStrokeSegments(st.charKey, st.t);
                    fctx.lineWidth = brush;
                    segs.forEach(seg => {{
                        fctx.beginPath();
                        fctx.moveTo(seg.x1, seg.y1);
                        fctx.lineTo(seg.x2, seg.y2);
                        fctx.stroke();
                    }});
                }}
            }});
            return flat;
        }}

        function renderTrace() {{
            const hasStem = stemP1 && stemP2;
            if (traceStrokes.length === 0 && !hasStem) {{ alert('Nothing to render'); return; }}
            const flat = flattenStrokesToCanvas();
            const imageData = flat.toDataURL('image/png');
            fetch('/render-traced', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{
                    image_data: imageData,
                    strokes: traceStrokes,
                    stem_p1: stemP1, stem_p2: stemP2,
                }}),
            }}).then(r => r.json()).then(d => {{
                if (d.ok) {{
                    document.getElementById('traced-image').src = d.data_url;
                }}
            }});
        }}

        function buildOverlayComposite() {{
            // Composite the stone photo (main canvas, with any user filters applied)
            // with the trace strokes rendered on top. This is Josh's "option A":
            // the original image annotated by eye, which the OCR model will read.
            const flat = flattenStrokesToCanvas();
            const composite = document.createElement('canvas');
            composite.width = canvas.width;
            composite.height = canvas.height;
            const cctx = composite.getContext('2d');
            // Stone photo (with brightness/contrast/invert already baked in by drawCanvas)
            cctx.drawImage(canvas, 0, 0);
            // Trace strokes on top — the trace canvas is aligned to the main canvas
            // and shares resolution, so stamping flat at (0,0) matches pixel positions.
            cctx.drawImage(flat, 0, 0);
            return composite;
        }}

        function saveTraced() {{
            const hasStem = stemP1 && stemP2;
            if (traceStrokes.length === 0 && !hasStem) {{ alert('Trace something first'); return; }}
            const s = stones[currentIdx];
            const key = s.stone_id + '/' + s.image_name;

            const synthFlat = flattenStrokesToCanvas();
            const overlay = buildOverlayComposite();

            fetch('/save-traced', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{
                    stone_id: s.stone_id,
                    image_name: s.image_name,
                    image_data: synthFlat.toDataURL('image/png'),
                    overlay_data: overlay.toDataURL('image/png'),
                    strokes: traceStrokes,
                    sub_mode: traceSubMode,
                    stem_p1: stemP1, stem_p2: stemP2,
                }}),
            }}).then(r => r.json()).then(d => {{
                if (d.ok) {{
                    if (!curation[key]) curation[key] = {{}};
                    curation[key].status = 'traced';
                    curation[key].trace_path = d.trace_path;
                    curation[key].overlay_path = d.overlay_path;
                    curation[key].stone_id = s.stone_id;
                    curation[key].image_name = s.image_name;
                    saveCuration();
                    updateStats();
                    goTo(currentIdx);
                    setTimeout(() => {{ navigate(1); }}, 200);
                }}
            }});
        }}

        buildKeypad();

        // Init
        updateStats();
        renderSidebar();
        if (stones.length > 0) goTo(0);
    </script>
</body>
</html>"""


class CurationHandler(http.server.BaseHTTPRequestHandler):
    stones = []
    curation = {}

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "":
            html = build_html(self.stones, self.curation)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

        elif parsed.path == "/image":
            params = parse_qs(parsed.query)
            img_path = params.get("path", [""])[0]
            if os.path.exists(img_path):
                ext = Path(img_path).suffix.lower()
                mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext.lstrip("."), "image/png")
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.end_headers()
                with open(img_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()

        elif parsed.path == "/style-match":
            import cv2
            import numpy as np
            params = parse_qs(parsed.query)
            img_path = params.get("path", [""])[0]
            if os.path.exists(img_path):
                sys.path.insert(0, str(PROJECT_ROOT))
                from scripts.match_synthetic_style import match_synthetic_style
                image = cv2.imread(img_path)
                result = match_synthetic_style(
                    image,
                    inscription_scale=float(params.get("scale", [0.3])[0]),
                    bg_intensity=int(params.get("bg", [180])[0]),
                    stroke_intensity=int(params.get("stroke", [80])[0]),
                    blur_sigma=float(params.get("blur", [1.0])[0]),
                    noise_sigma=float(params.get("noise", [8.0])[0]),
                    contrast_reduction=float(params.get("contrast", [0.4])[0]),
                )
                _, png_data = cv2.imencode(".png", result)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(png_data.tobytes())
            else:
                self.send_response(404)
                self.end_headers()

        elif parsed.path == "/trace-data":
            params = parse_qs(parsed.query)
            key = params.get("key", [""])[0]
            traces = _load_traces()
            entry = traces.get(key, {})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(entry).encode())

        elif parsed.path == "/synth-sample":
            # Serve a random synthetic image for reference
            synth_dir = DATASET_DIR / "synthetic_demo"
            if not synth_dir.exists():
                synth_dir = DATASET_DIR / "synthetic_training"
            if synth_dir.exists():
                images = list(synth_dir.glob("**/*.png")) + list(synth_dir.glob("**/*.jpg"))
                if images:
                    import random
                    img_path = random.choice(images)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.end_headers()
                    with open(img_path, "rb") as f:
                        self.wfile.write(f.read())
                    return
            # Fallback: generate a simple grey placeholder
            import cv2
            import numpy as np
            placeholder = np.full((384, 384, 3), 180, dtype=np.uint8)
            cv2.putText(placeholder, "No synthetic samples found", (30, 192),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            _, png_data = cv2.imencode(".png", placeholder)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(png_data.tobytes())

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/save":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            self.curation.update(json.loads(body))
            save_curation(self.curation)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')

        elif self.path == "/save-image":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            stone_id = data["stone_id"]
            image_name = data["image_name"]
            image_data = data["image_data"]  # base64 PNG from canvas

            # Save to curated/<stone_id>/
            stone_dir = CURATED_DIR / stone_id
            stone_dir.mkdir(parents=True, exist_ok=True)

            # Convert to PNG filename
            out_name = Path(image_name).stem + ".png"
            out_path = stone_dir / out_name

            # Decode base64 and save
            import base64 as b64
            png_bytes = b64.b64decode(image_data.split(",")[1])
            with open(out_path, "wb") as f:
                f.write(png_bytes)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = json.dumps({"ok": True, "path": str(out_path)})
            self.wfile.write(resp.encode())
            print(f"  Saved curated image: {out_path}")

        elif self.path == "/render-traced":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            image_data = data["image_data"]

            synth_png = _render_strokes_as_synthetic(
                data.get("strokes", []),
                data.get("stem_p1"),
                data.get("stem_p2"),
            )
            if synth_png is None:
                synth_png = _render_traced_to_synthetic(image_data)
            import base64 as b64
            data_url = "data:image/png;base64," + b64.b64encode(synth_png).decode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = json.dumps({"ok": True, "data_url": data_url})
            self.wfile.write(resp.encode())

        elif self.path == "/save-traced":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            stone_id = data["stone_id"]
            image_name = data["image_name"]
            image_data = data["image_data"]         # transparent lines layer
            overlay_data = data.get("overlay_data")  # stone + lines composite
            strokes = data.get("strokes", [])
            import base64 as b64

            # (1) Synthetic-style render. If we have assisted-mode data (stem + char
            # positions), reconstruct the inscription at synthetic density — this gives
            # the model a much closer match to its training distribution than scaling
            # the user's wide canvas down. Fall back to the alpha-mask render for
            # pure freeform traces.
            synth_png = _render_strokes_as_synthetic(strokes,
                                                     data.get("stem_p1"),
                                                     data.get("stem_p2"))
            if synth_png is None:
                synth_png = _render_traced_to_synthetic(image_data)

            traced_dir = TRACED_DIR / stone_id
            traced_dir.mkdir(parents=True, exist_ok=True)
            traced_name = Path(image_name).stem + "_traced.png"
            traced_path = traced_dir / traced_name
            with open(traced_path, "wb") as f:
                f.write(synth_png)

            # (2) Overlay composite (stone + drawn lines) — Josh's Option A, primary
            overlay_path = None
            if overlay_data:
                overlay_dir = OVERLAY_DIR / stone_id
                overlay_dir.mkdir(parents=True, exist_ok=True)
                overlay_name = Path(image_name).stem + "_overlay.png"
                overlay_path = overlay_dir / overlay_name
                overlay_bytes = b64.b64decode(overlay_data.split(",")[1])
                with open(overlay_path, "wb") as f:
                    f.write(overlay_bytes)

            # Persist strokes for re-editing
            traces_all = _load_traces()
            traces_all[f"{stone_id}/{image_name}"] = {
                "strokes": strokes,
                "sub_mode": data.get("sub_mode"),
                "stem_p1": data.get("stem_p1"),
                "stem_p2": data.get("stem_p2"),
                "trace_path": str(traced_path),
                "overlay_path": str(overlay_path) if overlay_path else None,
            }
            _save_traces(traces_all)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = json.dumps({
                "ok": True,
                "trace_path": str(traced_path),
                "overlay_path": str(overlay_path) if overlay_path else None,
            })
            self.wfile.write(resp.encode())
            print(f"  Saved traced (synthetic-style): {traced_path}")
            if overlay_path:
                print(f"  Saved overlay (stone + lines): {overlay_path}")

        else:
            self.send_response(404)
            self.end_headers()


def main():
    stones = load_data()
    curation = load_curation()

    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    TRACED_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    raw_count = sum(1 for s in stones if s["source"] == "raw")
    proc_count = sum(1 for s in stones if s["source"] == "processed")
    print(f"Loaded {len(stones)} images ({raw_count} raw, {proc_count} processed-only)")
    print(f"  from {len(set(s['stone_id'] for s in stones))} stones")
    print(f"Existing curation: {len(curation)} decisions")
    print(f"Curated images will be saved to: {CURATED_DIR}")
    print(f"Traced (synthetic-style) will be saved to: {TRACED_DIR}")
    print(f"Overlay (stone + lines) will be saved to: {OVERLAY_DIR}")

    CurationHandler.stones = stones
    CurationHandler.curation = curation

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), CurationHandler) as httpd:
        print(f"\nCuration tool running at: http://localhost:{PORT}")
        print("Press Ctrl+C to stop\n")
        print("Keyboard shortcuts:")
        print("  1=Keep  2=Enhance  3=Drop  T=Trace  A/D=Navigate")
        print("  R=Rotate  I=Invert  G=Greyscale  P=Show Processed")
        print("  +/-=Zoom  0=Reset  C=Crop  Ctrl+Z=Undo last trace stroke")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nStopped. Curation saved to:", CURATION_FILE)


if __name__ == "__main__":
    main()
