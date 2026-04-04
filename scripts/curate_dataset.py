#!/usr/bin/env python3
"""
Interactive image viewer and annotation tool for curating the Ogham dataset.

Opens a local web UI where you can:
- Browse all stone images
- See the transcription for each stone
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
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "ogham_dataset"
CROPPED_DIR = DATASET_DIR / "processed" / "cropped"
ANNOTATIONS_FILE = DATASET_DIR / "processed" / "annotations" / "transcriptions.json"
CURATION_FILE = DATASET_DIR / "processed" / "curation.json"

PORT = 8765


def load_data():
    """Load all stone images and annotations."""
    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)

    stones = []
    stone_dirs = sorted(CROPPED_DIR.iterdir())

    for stone_dir in stone_dirs:
        if not stone_dir.is_dir():
            continue
        stone_id = stone_dir.name
        ann = annotations.get(stone_id, {})

        images = sorted(stone_dir.glob("*.png"))
        for img_path in images:
            stones.append({
                "stone_id": stone_id,
                "image_name": img_path.name,
                "image_path": str(img_path),
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


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_html(stones, curation):
    """Build the single-page HTML app."""

    # Prepare data as JSON for JavaScript
    stones_json = json.dumps(stones, ensure_ascii=False)
    curation_json = json.dumps(curation, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Ogham Dataset Curator</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; }}
        .header {{ background: #16213e; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #0f3460; }}
        .header h1 {{ font-size: 20px; color: #e94560; }}
        .stats {{ font-size: 14px; color: #aaa; }}
        .stats span {{ color: #fff; font-weight: bold; margin: 0 5px; }}
        .container {{ display: flex; height: calc(100vh - 60px); }}
        .sidebar {{ width: 280px; background: #16213e; overflow-y: auto; border-right: 1px solid #0f3460; flex-shrink: 0; }}
        .sidebar-item {{ padding: 10px 15px; cursor: pointer; border-bottom: 1px solid #0f3460; font-size: 13px; }}
        .sidebar-item:hover {{ background: #1a1a40; }}
        .sidebar-item.active {{ background: #0f3460; border-left: 3px solid #e94560; }}
        .sidebar-item .stone-id {{ font-weight: bold; color: #fff; }}
        .sidebar-item .status {{ font-size: 11px; margin-top: 2px; }}
        .status-keep {{ color: #4caf50; }}
        .status-enhance {{ color: #ff9800; }}
        .status-drop {{ color: #f44336; }}
        .status-pending {{ color: #666; }}
        .main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
        .image-container {{ flex: 1; display: flex; justify-content: center; align-items: center; background: #111; overflow: auto; padding: 20px; }}
        .image-container img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
        .controls {{ background: #16213e; padding: 20px 30px; border-top: 2px solid #0f3460; }}
        .info-row {{ display: flex; gap: 20px; margin-bottom: 12px; font-size: 14px; }}
        .info-row label {{ color: #888; min-width: 100px; }}
        .info-row .value {{ color: #fff; font-family: monospace; font-size: 16px; }}
        .ogham-text {{ font-size: 24px; letter-spacing: 2px; }}
        .button-row {{ display: flex; gap: 10px; margin-bottom: 12px; }}
        .btn {{ padding: 10px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: bold; transition: all 0.2s; }}
        .btn-keep {{ background: #4caf50; color: white; }}
        .btn-keep:hover {{ background: #66bb6a; }}
        .btn-keep.selected {{ box-shadow: 0 0 0 3px #fff; }}
        .btn-enhance {{ background: #ff9800; color: white; }}
        .btn-enhance:hover {{ background: #ffa726; }}
        .btn-enhance.selected {{ box-shadow: 0 0 0 3px #fff; }}
        .btn-drop {{ background: #f44336; color: white; }}
        .btn-drop:hover {{ background: #ef5350; }}
        .btn-drop.selected {{ box-shadow: 0 0 0 3px #fff; }}
        .btn-nav {{ background: #0f3460; color: white; }}
        .btn-nav:hover {{ background: #1a4a80; }}
        .btn-export {{ background: #e94560; color: white; }}
        .notes-input {{ width: 100%; padding: 8px 12px; background: #1a1a2e; border: 1px solid #0f3460; color: #fff; border-radius: 4px; font-size: 13px; }}
        .nav-row {{ display: flex; justify-content: space-between; align-items: center; }}
        .progress {{ font-size: 13px; color: #888; }}
        .keyboard-hint {{ font-size: 11px; color: #555; margin-top: 8px; }}
        .filter-row {{ padding: 10px 15px; background: #0f3460; display: flex; gap: 5px; }}
        .filter-btn {{ padding: 4px 10px; border: 1px solid #333; border-radius: 3px; background: transparent; color: #aaa; cursor: pointer; font-size: 11px; }}
        .filter-btn.active {{ background: #1a4a80; color: #fff; border-color: #e94560; }}
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
            <div class="image-container">
                <img id="current-image" src="" alt="Select a stone" />
            </div>
            <div class="controls">
                <div class="info-row">
                    <label>Stone:</label><span class="value" id="info-stone"></span>
                    <label>County:</label><span class="value" id="info-county"></span>
                    <label>Image:</label><span class="value" id="info-image"></span>
                </div>
                <div class="info-row">
                    <label>Transcription:</label><span class="value ogham-text" id="info-transcription"></span>
                </div>
                <div class="info-row">
                    <label>Original:</label><span class="value" id="info-original" style="font-size:12px;color:#888"></span>
                </div>
                <div class="button-row">
                    <button class="btn btn-keep" onclick="setStatus('keep')">Keep (1)</button>
                    <button class="btn btn-enhance" onclick="setStatus('enhance')">Enhance (2)</button>
                    <button class="btn btn-drop" onclick="setStatus('drop')">Drop (3)</button>
                    <input class="notes-input" id="notes" placeholder="Notes (optional)..." oninput="saveNotes()" />
                </div>
                <div class="nav-row">
                    <div>
                        <button class="btn btn-nav" onclick="navigate(-1)">Prev (A)</button>
                        <button class="btn btn-nav" onclick="navigate(1)">Next (D)</button>
                    </div>
                    <span class="progress" id="progress"></span>
                    <button class="btn btn-export" onclick="exportResults()">Export Results</button>
                </div>
                <div class="keyboard-hint">Keyboard: 1=Keep, 2=Enhance, 3=Drop, A=Prev, D=Next, S=Save</div>
            </div>
        </div>
    </div>

    <script>
        const stones = {stones_json};
        let curation = {curation_json};
        let currentIdx = 0;
        let currentFilter = 'all';

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
                return `<div class="sidebar-item ${{active}}" onclick="goTo(${{i}})">
                    <div class="stone-id">${{s.stone_id}}</div>
                    <div class="status status-${{status}}">${{status.toUpperCase()}} - ${{s.image_name}}</div>
                </div>`;
            }}).join('');
        }}

        function goTo(idx) {{
            currentIdx = idx;
            const s = stones[idx];
            const key = s.stone_id + '/' + s.image_name;

            document.getElementById('current-image').src = '/image?path=' + encodeURIComponent(s.image_path);
            document.getElementById('info-stone').textContent = s.stone_id;
            document.getElementById('info-county').textContent = s.county;
            document.getElementById('info-image').textContent = s.image_name;
            document.getElementById('info-transcription').textContent = s.transcription;
            document.getElementById('info-original').textContent = s.original_transcription;
            document.getElementById('notes').value = curation[key]?.notes || '';

            // Update button states
            const status = curation[key]?.status || '';
            document.querySelectorAll('.btn-keep,.btn-enhance,.btn-drop').forEach(b => b.classList.remove('selected'));
            if (status) document.querySelector('.btn-' + status)?.classList.add('selected');

            document.getElementById('progress').textContent = `${{idx + 1}} / ${{stones.length}}`;
            renderSidebar();
        }}

        function setStatus(status) {{
            const s = stones[currentIdx];
            const key = s.stone_id + '/' + s.image_name;
            if (!curation[key]) curation[key] = {{}};
            curation[key].status = status;
            curation[key].stone_id = s.stone_id;
            curation[key].image_name = s.image_name;
            saveCuration();
            updateStats();
            goTo(currentIdx);
            // Auto-advance after marking
            setTimeout(() => navigate(1), 150);
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
            let nextFilterIdx;
            if (currentFilterIdx === -1) {{
                nextFilterIdx = dir > 0 ? 0 : indices.length - 1;
            }} else {{
                nextFilterIdx = (currentFilterIdx + dir + indices.length) % indices.length;
            }}
            goTo(indices[nextFilterIdx]);
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

        // Keyboard shortcuts
        document.addEventListener('keydown', e => {{
            if (e.target.tagName === 'INPUT') return;
            if (e.key === '1') setStatus('keep');
            if (e.key === '2') setStatus('enhance');
            if (e.key === '3') setStatus('drop');
            if (e.key === 'a' || e.key === 'ArrowLeft') navigate(-1);
            if (e.key === 'd' || e.key === 'ArrowRight') navigate(1);
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
        pass  # Suppress request logs

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
            if os.path.exists(img_path) and img_path.endswith(".png"):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(img_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
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
        else:
            self.send_response(404)
            self.end_headers()


def main():
    stones = load_data()
    curation = load_curation()

    print(f"Loaded {len(stones)} images from {len(set(s['stone_id'] for s in stones))} stones")
    print(f"Existing curation: {len(curation)} decisions")

    CurationHandler.stones = stones
    CurationHandler.curation = curation

    with socketserver.TCPServer(("", PORT), CurationHandler) as httpd:
        print(f"\nCuration tool running at: http://localhost:{PORT}")
        print("Press Ctrl+C to stop\n")
        print("Keyboard shortcuts:")
        print("  1 = Keep    2 = Enhance    3 = Drop")
        print("  A = Prev    D = Next")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nStopped. Curation saved to:", CURATION_FILE)


if __name__ == "__main__":
    main()
