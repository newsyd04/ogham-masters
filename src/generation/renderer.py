"""
Ogham text renderer for synthetic image generation.

Renders Unicode Ogham text into images simulating stone inscriptions,
with configurable styles, colors, and stemline appearance.

★ Insight ─────────────────────────────────────
Rendering considerations:
1. Ogham traditionally has a central stemline
2. Characters are strokes relative to this line
3. After rotation for TrOCR, stemline becomes horizontal
4. Stone-like colors improve domain adaptation
─────────────────────────────────────────────────
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise ImportError("Pillow is required for rendering. Install with: pip install Pillow")


class OghamRenderer:
    """
    Render Ogham text as synthetic stone inscription images.

    Creates images with:
    - Horizontal stemline (text reads left-to-right after rotation)
    - Stone-like background colors
    - Variable character sizes and spacing
    - Configurable stemline thickness
    """

    # Default color palettes (RGB)
    STONE_BACKGROUNDS = [
        (180, 180, 180),  # Light gray (limestone)
        (150, 140, 130),  # Sandstone
        (100, 100, 100),  # Dark granite
        (200, 195, 185),  # Pale limestone
        (140, 135, 125),  # Brown-gray sandstone
        (170, 165, 160),  # Light weathered stone
        (120, 115, 110),  # Medium gray
    ]

    INSCRIPTION_COLORS = [
        (50, 50, 50),     # Dark carved grooves
        (30, 30, 30),     # Very dark
        (70, 60, 50),     # Brown-tinged dark
        (40, 40, 45),     # Blue-gray dark
        (60, 55, 50),     # Warm dark
    ]

    def __init__(
        self,
        font_paths: List[str],
        image_height: int = 384,
        char_height_range: Tuple[int, int] = (30, 60),
        stemline_thickness_range: Tuple[int, int] = (2, 6),
        background_colors: Optional[List[Tuple[int, int, int]]] = None,
        foreground_colors: Optional[List[Tuple[int, int, int]]] = None,
        char_spacing_range: Tuple[int, int] = (2, 8),
        padding: int = 20,
        seed: Optional[int] = None,
    ):
        """
        Initialize renderer.

        Args:
            font_paths: Paths to Ogham-compatible TTF/OTF fonts
            image_height: Height of generated images
            char_height_range: (min, max) character size range
            stemline_thickness_range: (min, max) stemline width range
            background_colors: Custom background colors (RGB tuples)
            foreground_colors: Custom foreground colors (RGB tuples)
            char_spacing_range: (min, max) pixel spacing between characters
            padding: Padding around text in pixels
            seed: Random seed for reproducibility
        """
        self.image_height = image_height
        self.char_height_range = char_height_range
        self.stemline_thickness_range = stemline_thickness_range
        self.char_spacing_range = char_spacing_range
        self.padding = padding

        self.background_colors = background_colors or self.STONE_BACKGROUNDS
        self.foreground_colors = foreground_colors or self.INSCRIPTION_COLORS

        self.rng = np.random.default_rng(seed)

        # Load fonts at multiple sizes
        self.fonts = self._load_fonts(font_paths)

        if not self.fonts:
            raise ValueError(
                "No Ogham fonts could be loaded! "
                "Please provide paths to TTF/OTF fonts that include Ogham characters. "
                "Recommended: BabelStone Ogham, Noto Sans Ogham"
            )

    # Reference character for stemline detection — B-group ᚁ has strokes
    # only below the stemline, so its bbox top edge IS the stemline position.
    _STEMLINE_REF_CHAR = "\u1681"  # ᚁ (Ogham letter Beith)

    def _load_fonts(self, font_paths: List[str]) -> List[ImageFont.FreeTypeFont]:
        """Load fonts at various sizes and compute stemline offsets."""
        fonts = []

        for path in font_paths:
            if not Path(path).exists():
                continue

            try:
                # Load at multiple sizes within range
                for size in range(
                    self.char_height_range[0],
                    self.char_height_range[1],
                    5,  # Step by 5
                ):
                    font = ImageFont.truetype(str(path), size)
                    fonts.append(font)
            except Exception as e:
                print(f"Warning: Could not load font {path}: {e}")

        # Pre-compute stemline offset for each font: the y-offset from the
        # draw origin to where the font's internal stemline sits.
        self._stemline_offsets = {}
        for font in fonts:
            ref_bbox = font.getbbox(self._STEMLINE_REF_CHAR)
            self._stemline_offsets[id(font)] = ref_bbox[1]

        return fonts

    def render(
        self,
        text: str,
        style_override: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Render Ogham text as an image.

        Args:
            text: Unicode Ogham string to render
            style_override: Optional dict to override random style choices

        Returns:
            Tuple of (image as numpy array, rendering metadata)
        """
        # Sample random style parameters
        style = self._sample_style(style_override)

        # Calculate text dimensions
        font = style["font"]
        total_width, max_char_height = self._calculate_text_dimensions(text, font, style["char_spacing"])

        # Create image with padding
        img_width = total_width + 2 * self.padding
        img_height = self.image_height

        # Ensure minimum width
        img_width = max(img_width, 128)

        # Create image
        image = Image.new("RGB", (img_width, img_height), style["bg_color"])
        draw = ImageDraw.Draw(image)

        # Draw stemline (horizontal through middle)
        stemline_y = img_height // 2
        draw.line(
            [(self.padding // 2, stemline_y), (img_width - self.padding // 2, stemline_y)],
            fill=style["fg_color"],
            width=style["stemline_thickness"],
        )

        # Draw characters — all at the same y so the font's built-in
        # stemline aligns with our drawn stemline
        stemline_offset = self._stemline_offsets[id(font)]
        text_y = stemline_y - stemline_offset

        x = self.padding
        for char in text:
            bbox = font.getbbox(char)
            char_width = bbox[2] - bbox[0]

            draw.text((x, text_y), char, font=font, fill=style["fg_color"])
            x += char_width + style["char_spacing"]

        # Convert to numpy array
        img_array = np.array(image)

        # Build metadata
        render_info = {
            "text": text,
            "text_length": len(text),
            "font_size": font.size,
            "bg_color": style["bg_color"],
            "fg_color": style["fg_color"],
            "stemline_thickness": style["stemline_thickness"],
            "char_spacing": style["char_spacing"],
            "image_size": (img_height, img_width),
        }

        return img_array, render_info

    def _sample_style(self, override: Optional[Dict] = None) -> Dict:
        """Sample random rendering style."""
        if override is None:
            override = {}

        style = {
            "font": override.get("font") or self.rng.choice(self.fonts),
            "bg_color": override.get("bg_color") or tuple(self.rng.choice(self.background_colors)),
            "fg_color": override.get("fg_color") or tuple(self.rng.choice(self.foreground_colors)),
            "stemline_thickness": override.get("stemline_thickness") or int(
                self.rng.integers(self.stemline_thickness_range[0], self.stemline_thickness_range[1] + 1)
            ),
            "char_spacing": override.get("char_spacing") or int(
                self.rng.integers(self.char_spacing_range[0], self.char_spacing_range[1] + 1)
            ),
        }
        return style

    def _calculate_text_dimensions(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        char_spacing: int,
    ) -> Tuple[int, int]:
        """Calculate total text width and max character height."""
        total_width = 0
        max_height = 0

        for char in text:
            bbox = font.getbbox(char)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]

            total_width += char_width + char_spacing
            max_height = max(max_height, char_height)

        # Remove extra spacing after last character
        if text:
            total_width -= char_spacing

        return total_width, max_height

    def render_batch(
        self,
        texts: List[str],
        consistent_style: bool = False,
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Render multiple texts.

        Args:
            texts: List of Ogham strings
            consistent_style: Use same style for all images

        Returns:
            List of (image, metadata) tuples
        """
        results = []
        style = self._sample_style() if consistent_style else None

        for text in texts:
            img, info = self.render(text, style_override=style)
            results.append((img, info))

        return results

    def set_seed(self, seed: int):
        """Reset random generator with new seed."""
        self.rng = np.random.default_rng(seed)


class StoneTextureRenderer(OghamRenderer):
    """
    Extended renderer that adds stone-like texture to images.

    Adds procedural noise to simulate stone surface variation.
    """

    def render(
        self,
        text: str,
        style_override: Optional[Dict] = None,
        texture_intensity: float = 0.15,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Render with stone texture.

        Args:
            text: Unicode Ogham string
            style_override: Optional style overrides
            texture_intensity: Strength of texture overlay (0-1)

        Returns:
            Tuple of (textured image, metadata)
        """
        # Base render
        img_array, render_info = super().render(text, style_override)

        # Add stone texture
        textured = self._add_stone_texture(img_array, texture_intensity)

        render_info["texture_applied"] = True
        render_info["texture_intensity"] = texture_intensity

        return textured, render_info

    def _add_stone_texture(
        self,
        image: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """Add procedural stone texture to image."""
        h, w = image.shape[:2]

        # Generate multi-scale Perlin-like noise
        noise = np.zeros((h, w), dtype=np.float32)

        for scale in [16, 32, 64]:
            # Create base noise at lower resolution
            noise_low = self.rng.random((h // scale + 1, w // scale + 1)).astype(np.float32)

            # Resize to full resolution (creates smooth variation)
            import cv2
            noise_scaled = cv2.resize(noise_low, (w, h), interpolation=cv2.INTER_CUBIC)

            noise += noise_scaled

        # Normalize to 0-1
        noise = (noise - noise.min()) / (noise.max() - noise.min())

        # Convert to 0-255 centered around 127
        noise = ((noise - 0.5) * 255 * intensity).astype(np.int32)

        # Apply to image
        textured = image.astype(np.int32)
        for c in range(3):
            textured[:, :, c] += noise

        # Clip to valid range
        textured = np.clip(textured, 0, 255).astype(np.uint8)

        return textured


def create_renderer(
    font_dir: str,
    image_height: int = 384,
    with_texture: bool = False,
    seed: Optional[int] = None,
) -> OghamRenderer:
    """
    Create renderer with fonts from directory.

    Args:
        font_dir: Directory containing Ogham fonts
        image_height: Target image height
        with_texture: Use StoneTextureRenderer
        seed: Random seed

    Returns:
        Configured renderer
    """
    font_dir = Path(font_dir)
    font_paths = list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf"))

    if not font_paths:
        raise ValueError(f"No fonts found in {font_dir}")

    renderer_class = StoneTextureRenderer if with_texture else OghamRenderer

    return renderer_class(
        font_paths=[str(p) for p in font_paths],
        image_height=image_height,
        seed=seed,
    )
