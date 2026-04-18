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
    </style>
</head>
<body>
    <div class="header">
        <h1>Ogham Dataset Curator</h1>
        <div class="stats">
            Total: <span id="total">0</span> |
            Keep: <span id="n-keep" style="color:#4caf50">0</span> |
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

            <!-- Image area -->
            <div class="image-area">
                <div class="image-panel" id="main-panel">
                    <div class="image-panel-label" id="panel-label">Raw Image</div>
                    <div class="image-container" id="image-container">
                        <canvas id="main-canvas"></canvas>
                        <div class="crop-overlay" id="crop-overlay" style="display:none"></div>
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
                <div class="keyboard-hint">Keys: 1=Keep 2=Enhance 3=Drop | A/D=Nav | R=Rotate | I=Invert | G=Grey | P=Toggle Processed | +/-=Zoom | 0=Reset</div>
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
            let keep = 0, enhance = 0, drop = 0, pending = 0;
            stones.forEach(s => {{
                const key = s.stone_id + '/' + s.image_name;
                const status = curation[key]?.status || 'pending';
                if (status === 'keep') keep++;
                else if (status === 'enhance') enhance++;
                else if (status === 'drop') drop++;
                else pending++;
            }});
            document.getElementById('total').textContent = stones.length;
            document.getElementById('n-keep').textContent = keep;
            document.getElementById('n-enhance').textContent = enhance;
            document.getElementById('n-drop').textContent = drop;
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
                const params = new URLSearchParams({{
                    path: s.image_path,
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
        }});

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

        else:
            self.send_response(404)
            self.end_headers()


def main():
    stones = load_data()
    curation = load_curation()

    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    raw_count = sum(1 for s in stones if s["source"] == "raw")
    proc_count = sum(1 for s in stones if s["source"] == "processed")
    print(f"Loaded {len(stones)} images ({raw_count} raw, {proc_count} processed-only)")
    print(f"  from {len(set(s['stone_id'] for s in stones))} stones")
    print(f"Existing curation: {len(curation)} decisions")
    print(f"Curated images will be saved to: {CURATED_DIR}")

    CurationHandler.stones = stones
    CurationHandler.curation = curation

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), CurationHandler) as httpd:
        print(f"\nCuration tool running at: http://localhost:{PORT}")
        print("Press Ctrl+C to stop\n")
        print("Keyboard shortcuts:")
        print("  1=Keep  2=Enhance  3=Drop  A/D=Navigate")
        print("  R=Rotate  I=Invert  G=Greyscale  P=Show Processed")
        print("  +/-=Zoom  0=Reset  C=Crop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nStopped. Curation saved to:", CURATION_FILE)


if __name__ == "__main__":
    main()
