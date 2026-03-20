"""
Orientation handling for Ogham images.

Ogham inscriptions traditionally read bottom-to-top along a stone edge.
TrOCR and most OCR models expect left-to-right reading direction.
This module handles the conversion between orientations.
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np


class OrientationHandler:
    """
    Handle Ogham's vertical orientation for TrOCR.

    ★ Insight ─────────────────────────────────────
    Orientation modes:
    - rotate_90_cw: Bottom→Right (recommended for TrOCR)
    - rotate_90_ccw: Bottom→Left (would need RTL handling)
    - keep_vertical: No rotation (may not work well with TrOCR)
    ─────────────────────────────────────────────────
    """

    ORIENTATION_MODES = {
        "rotate_90_cw": "Rotate 90° clockwise (bottom→right)",
        "rotate_90_ccw": "Rotate 90° counter-clockwise (bottom→left)",
        "keep_vertical": "Keep vertical orientation",
        "auto": "Attempt automatic detection",
    }

    def __init__(self, mode: str = "rotate_90_cw"):
        """
        Initialize orientation handler.

        Args:
            mode: Orientation mode (default: rotate_90_cw)
        """
        if mode not in self.ORIENTATION_MODES and mode != "auto":
            raise ValueError(f"Unknown mode: {mode}. Options: {list(self.ORIENTATION_MODES.keys())}")
        self.mode = mode

    def fix_orientation(
        self,
        image: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fix image orientation for model input.

        Args:
            image: Input image (BGR numpy array)
            metadata: Optional metadata dict

        Returns:
            Tuple of (rotated_image, step_info)
        """
        if metadata is None:
            metadata = {}

        original_direction = metadata.get("reading_direction", "bottom_to_top")

        mode = self.mode
        if mode == "auto":
            mode = self._detect_orientation(image)

        if mode == "rotate_90_cw":
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            new_direction = "left_to_right"
        elif mode == "rotate_90_ccw":
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_direction = "right_to_left"
        else:
            rotated = image
            new_direction = original_direction

        step_info = {
            "step": "orientation",
            "mode": mode,
            "original_direction": original_direction,
            "new_direction": new_direction,
            "original_shape": image.shape,
            "new_shape": rotated.shape,
        }

        return rotated, step_info

    def _detect_orientation(self, image: np.ndarray) -> str:
        """
        Attempt to detect image orientation automatically.

        Uses aspect ratio heuristics:
        - Tall images (h > w) likely need rotation
        - Wide images (w > h) may already be oriented

        Returns:
            Recommended orientation mode
        """
        h, w = image.shape[:2]

        if h > w * 1.5:
            # Image is significantly taller than wide
            # Likely a vertical stone photo, rotate to horizontal
            return "rotate_90_cw"
        elif w > h * 1.5:
            # Image is significantly wider than tall
            # May already be rotated or a wide shot
            return "keep_vertical"
        else:
            # Roughly square, default to rotation
            return "rotate_90_cw"

    def convert_bbox_orientation(
        self,
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
        rotation: str,
    ) -> Tuple[int, int, int, int]:
        """
        Convert bounding box coordinates after rotation.

        Args:
            bbox: (x1, y1, x2, y2) in original image
            image_shape: (height, width) of original image
            rotation: Rotation applied ("rotate_90_cw", "rotate_90_ccw", etc.)

        Returns:
            Transformed (x1, y1, x2, y2) in rotated image
        """
        x1, y1, x2, y2 = bbox
        h, w = image_shape

        if rotation == "rotate_90_cw":
            # (x, y) -> (h - y, x)
            new_x1 = h - y2
            new_y1 = x1
            new_x2 = h - y1
            new_y2 = x2
        elif rotation == "rotate_90_ccw":
            # (x, y) -> (y, w - x)
            new_x1 = y1
            new_y1 = w - x2
            new_x2 = y2
            new_y2 = w - x1
        else:
            return bbox

        return (new_x1, new_y1, new_x2, new_y2)
