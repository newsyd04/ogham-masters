"""
Streamlit annotation tool for Ogham inscriptions.

This tool provides a web interface for:
- Viewing and cropping Ogham inscription images
- Entering Unicode Ogham transcriptions with a virtual keyboard
- Managing annotation metadata and confidence levels
- Tracking annotation progress

Run with: streamlit run annotation_tool/app.py

★ Insight ─────────────────────────────────────
Design choices:
1. Ogham keyboard for easy Unicode input
2. Real-time validation of transcriptions
3. Stone-level organization (not image-level)
4. Version tracking for annotation changes
─────────────────────────────────────────────────
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.ogham import (
    ALL_CONSONANTS,
    ALL_VOWELS,
    ALL_FORFEDA,
    OGHAM_TO_LATIN,
    validate_ogham_string,
    estimate_difficulty,
    latin_to_ogham,
)
from src.schemas import TranscriptionConfidence, CropAnnotation


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Ogham Annotation Tool",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "ogham_buffer" not in st.session_state:
        st.session_state.ogham_buffer = ""
    if "current_stone_idx" not in st.session_state:
        st.session_state.current_stone_idx = 0
    if "annotations" not in st.session_state:
        st.session_state.annotations = {}
    if "crop_mode" not in st.session_state:
        st.session_state.crop_mode = False
    if "data_dir" not in st.session_state:
        st.session_state.data_dir = None


init_session_state()


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_stone_list(data_dir: str) -> List[str]:
    """Load list of stone IDs from data directory."""
    images_dir = Path(data_dir) / "raw" / "images"
    if not images_dir.exists():
        return []
    return sorted([d.name for d in images_dir.iterdir() if d.is_dir()])


@st.cache_data
def load_stone_images(data_dir: str, stone_id: str) -> List[str]:
    """Load image paths for a stone."""
    stone_dir = Path(data_dir) / "raw" / "images" / stone_id
    if not stone_dir.exists():
        return []

    extensions = {".jpg", ".jpeg", ".png", ".gif", ".tiff", ".tif"}
    return sorted([
        str(f) for f in stone_dir.iterdir()
        if f.suffix.lower() in extensions
    ])


def load_existing_annotations(data_dir: str) -> Dict:
    """Load existing annotations from file."""
    annotations_file = Path(data_dir) / "processed" / "annotations" / "transcriptions.json"
    if annotations_file.exists():
        with open(annotations_file) as f:
            return json.load(f)
    return {}


def save_annotations(data_dir: str, annotations: Dict):
    """Save annotations to file."""
    annotations_dir = Path(data_dir) / "processed" / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    annotations_file = annotations_dir / "transcriptions.json"
    with open(annotations_file, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)


# =============================================================================
# OGHAM KEYBOARD COMPONENT
# =============================================================================

def ogham_keyboard():
    """Render interactive Ogham keyboard."""
    st.subheader("Ogham Keyboard")

    # Character groups
    groups = [
        ("B Group (→)", ["ᚁ", "ᚂ", "ᚃ", "ᚄ", "ᚅ"], ["B", "L", "F", "S", "N"]),
        ("H Group (←)", ["ᚆ", "ᚇ", "ᚈ", "ᚉ", "ᚊ"], ["H", "D", "T", "C", "Q"]),
        ("M Group (↗)", ["ᚋ", "ᚌ", "ᚍ", "ᚎ", "ᚏ"], ["M", "G", "NG", "Z", "R"]),
        ("Vowels (⊥)", ["ᚐ", "ᚑ", "ᚒ", "ᚓ", "ᚔ"], ["A", "O", "U", "E", "I"]),
    ]

    for group_name, chars, labels in groups:
        st.caption(group_name)
        cols = st.columns(len(chars))
        for i, (char, label) in enumerate(zip(chars, labels)):
            with cols[i]:
                if st.button(f"{char}\n{label}", key=f"key_{char}", use_container_width=True):
                    st.session_state.ogham_buffer += char

    # Forfeda (rare characters)
    with st.expander("Forfeda (rare)"):
        forfeda = [("ᚕ", "EA"), ("ᚖ", "OI"), ("ᚗ", "UI"), ("ᚘ", "IA"), ("ᚙ", "AE")]
        cols = st.columns(len(forfeda))
        for i, (char, label) in enumerate(forfeda):
            with cols[i]:
                if st.button(f"{char}\n{label}", key=f"key_{char}"):
                    st.session_state.ogham_buffer += char

    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⌫ Backspace", use_container_width=True):
            st.session_state.ogham_buffer = st.session_state.ogham_buffer[:-1]
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.ogham_buffer = ""
    with col3:
        if st.button("Copy to Input", use_container_width=True):
            pass  # The buffer is automatically used

    # Show current buffer
    if st.session_state.ogham_buffer:
        st.info(f"**Buffer:** {st.session_state.ogham_buffer}")
        transliteration = "".join(OGHAM_TO_LATIN.get(c, "?") for c in st.session_state.ogham_buffer)
        st.caption(f"Transliteration: {transliteration}")


# =============================================================================
# IMAGE VIEWER COMPONENT
# =============================================================================

def image_viewer(image_path: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Display image with optional cropping interface.

    Returns:
        Tuple of (x1, y1, x2, y2) if crop selected, None otherwise
    """
    if not Path(image_path).exists():
        st.error(f"Image not found: {image_path}")
        return None

    image = Image.open(image_path)

    # Display image info
    st.caption(f"Size: {image.width} × {image.height} px | Path: {Path(image_path).name}")

    # Display image
    st.image(image, use_container_width=True)

    # Simple crop input (Streamlit doesn't have built-in drawing canvas)
    # For production, consider streamlit-drawable-canvas package
    if st.session_state.crop_mode:
        st.markdown("**Define crop region:**")
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.number_input("X1 (left)", 0, image.width, 0)
            y1 = st.number_input("Y1 (top)", 0, image.height, 0)
        with col2:
            x2 = st.number_input("X2 (right)", 0, image.width, image.width)
            y2 = st.number_input("Y2 (bottom)", 0, image.height, image.height)

        if x2 > x1 and y2 > y1:
            # Show crop preview
            cropped = image.crop((x1, y1, x2, y2))
            st.image(cropped, caption="Crop Preview", width=300)
            return (x1, y1, x2, y2)

    return None


# =============================================================================
# TRANSCRIPTION FORM
# =============================================================================

def transcription_form(stone_id: str, existing: Optional[Dict] = None) -> Optional[Dict]:
    """
    Render transcription input form.

    Args:
        stone_id: Stone identifier
        existing: Existing annotation data if any

    Returns:
        Dictionary with annotation data or None if not submitted
    """
    st.subheader("Transcription")

    # Pre-fill with existing data or keyboard buffer
    default_text = ""
    if existing and existing.get("transcription"):
        default_text = existing["transcription"]
    elif st.session_state.ogham_buffer:
        default_text = st.session_state.ogham_buffer

    # Unicode input
    transcription = st.text_input(
        "Ogham Unicode",
        value=default_text,
        help="Enter Ogham characters (Unicode range U+1680–U+169F) or use the keyboard above",
        key=f"transcription_{stone_id}",
    )

    # Latin transliteration input (alternative)
    latin_input = st.text_input(
        "Or enter Latin transliteration",
        placeholder="e.g., MAQI MUCOI",
        help="Will be converted to Ogham Unicode",
    )

    if latin_input and not transcription:
        try:
            transcription = latin_to_ogham(latin_input)
            st.info(f"Converted: {transcription}")
        except Exception as e:
            st.error(f"Conversion error: {e}")

    # Validation
    if transcription:
        is_valid, message = validate_ogham_string(transcription)
        if is_valid:
            st.success(f"✓ Valid: {len(transcription)} characters")

            # Show transliteration
            transliteration = "".join(OGHAM_TO_LATIN.get(c, "?") for c in transcription)
            st.caption(f"Reads as: {transliteration}")

            # Show difficulty
            difficulty = estimate_difficulty(transcription)
            st.progress(difficulty, text=f"Difficulty: {difficulty:.0%}")
        else:
            st.error(f"✗ {message}")

    # Confidence level
    confidence = st.radio(
        "Confidence Level",
        options=["verified", "probable", "uncertain"],
        index=1 if not existing else ["verified", "probable", "uncertain"].index(
            existing.get("confidence", "probable")
        ),
        help="verified = multiple expert sources agree, probable = single reputable source",
        horizontal=True,
    )

    # Source citation
    source = st.text_input(
        "Source",
        value=existing.get("source", "") if existing else "",
        placeholder="e.g., McManus 1991, p. 45",
    )

    # Notes
    notes = st.text_area(
        "Notes",
        value=existing.get("notes", "") if existing else "",
        placeholder="Any observations about the inscription...",
    )

    # Submit
    if st.button("💾 Save Annotation", type="primary", use_container_width=True):
        if not transcription:
            st.warning("Please enter a transcription")
            return None

        is_valid, _ = validate_ogham_string(transcription)
        if not is_valid:
            st.warning("Please fix validation errors before saving")
            return None

        return {
            "stone_id": stone_id,
            "transcription": transcription,
            "confidence": confidence,
            "source": source,
            "notes": notes,
            "annotator": st.session_state.get("annotator_name", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            "version": (existing.get("version", 0) + 1) if existing else 1,
        }

    return None


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render sidebar with navigation and settings."""
    with st.sidebar:
        st.title("🪨 Ogham Annotator")

        # Data directory selection
        st.subheader("Data Directory")
        data_dir = st.text_input(
            "Path to dataset",
            value=st.session_state.data_dir or "./ogham_dataset",
            help="Directory containing raw/images folder",
        )
        st.session_state.data_dir = data_dir

        if not Path(data_dir).exists():
            st.warning("Directory does not exist")
            return

        # Load stone list
        stones = load_stone_list(data_dir)
        if not stones:
            st.warning("No stones found in directory")
            return

        # Load existing annotations
        existing_annotations = load_existing_annotations(data_dir)
        st.session_state.annotations = existing_annotations

        # Progress metrics
        st.subheader("Progress")
        annotated = len(existing_annotations)
        total = len(stones)
        st.metric("Annotated", f"{annotated}/{total}")
        st.progress(annotated / total if total > 0 else 0)

        # Stone selection
        st.subheader("Select Stone")

        # Filter options
        filter_option = st.radio(
            "Show",
            ["All", "Unannotated", "Annotated"],
            horizontal=True,
        )

        if filter_option == "Unannotated":
            stones = [s for s in stones if s not in existing_annotations]
        elif filter_option == "Annotated":
            stones = [s for s in stones if s in existing_annotations]

        if stones:
            selected_stone = st.selectbox(
                "Stone ID",
                stones,
                index=min(st.session_state.current_stone_idx, len(stones) - 1),
            )
            st.session_state.current_stone_idx = stones.index(selected_stone)

            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("◀ Previous") and st.session_state.current_stone_idx > 0:
                    st.session_state.current_stone_idx -= 1
                    st.rerun()
            with col2:
                if st.button("Next ▶") and st.session_state.current_stone_idx < len(stones) - 1:
                    st.session_state.current_stone_idx += 1
                    st.rerun()

            return selected_stone
        else:
            st.info("No stones match filter")
            return None

        # Settings
        st.subheader("Settings")
        st.session_state.annotator_name = st.text_input(
            "Your name",
            value=st.session_state.get("annotator_name", ""),
        )
        st.session_state.crop_mode = st.checkbox("Enable crop mode")


# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    """Main application."""
    selected_stone = render_sidebar()

    if not selected_stone or not st.session_state.data_dir:
        st.title("🪨 Ogham Annotation Tool")
        st.markdown("""
        Welcome to the Ogham inscription annotation tool!

        **Getting Started:**
        1. Enter the path to your dataset directory in the sidebar
        2. Select a stone to annotate
        3. Use the Ogham keyboard to enter transcriptions
        4. Save your annotations

        **Directory Structure Expected:**
        ```
        dataset/
        ├── raw/
        │   └── images/
        │       ├── STONE_001/
        │       │   └── image.jpg
        │       └── STONE_002/
        │           └── image.jpg
        └── processed/
            └── annotations/
                └── transcriptions.json
        ```
        """)
        return

    # Main content area
    st.title(f"Stone: {selected_stone}")

    # Get existing annotation
    existing = st.session_state.annotations.get(selected_stone)
    if existing:
        st.info(f"📝 Has existing annotation (v{existing.get('version', 1)})")

    # Two-column layout
    col1, col2 = st.columns([3, 2])

    with col1:
        # Image viewer
        st.subheader("Images")
        images = load_stone_images(st.session_state.data_dir, selected_stone)

        if images:
            # Image selector tabs
            if len(images) > 1:
                tabs = st.tabs([f"Image {i+1}" for i in range(len(images))])
                for i, (tab, img_path) in enumerate(zip(tabs, images)):
                    with tab:
                        bbox = image_viewer(img_path)
            else:
                bbox = image_viewer(images[0])
        else:
            st.warning("No images found for this stone")

    with col2:
        # Ogham keyboard
        ogham_keyboard()

        st.divider()

        # Transcription form
        annotation = transcription_form(selected_stone, existing)

        if annotation:
            # Save annotation
            st.session_state.annotations[selected_stone] = annotation
            save_annotations(st.session_state.data_dir, st.session_state.annotations)
            st.success("✓ Annotation saved!")

            # Clear buffer
            st.session_state.ogham_buffer = ""

            # Auto-advance to next
            stones = load_stone_list(st.session_state.data_dir)
            if st.session_state.current_stone_idx < len(stones) - 1:
                if st.button("Continue to next stone →"):
                    st.session_state.current_stone_idx += 1
                    st.rerun()


if __name__ == "__main__":
    main()
