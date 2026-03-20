"""
Inscription cropping utilities for Ogham images.

Provides semi-automatic detection of stone edges and inscription regions,
designed to work with the annotation tool for manual verification.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class InscriptionCropper:
    """
    Extract inscription regions from full-stone photographs.

    Uses edge detection to find the stone's stemline (the edge where
    Ogham characters are carved) and suggests crop regions around it.
    """

    def __init__(
        self,
        min_line_length_ratio: float = 0.3,
        edge_margin_ratio: float = 0.25,
        canny_thresholds: Tuple[int, int] = (50, 150),
    ):
        """
        Initialize cropper.

        Args:
            min_line_length_ratio: Minimum line length as ratio of image height
            edge_margin_ratio: Crop margin around detected edge
            canny_thresholds: (low, high) thresholds for Canny edge detection
        """
        self.min_line_length_ratio = min_line_length_ratio
        self.edge_margin_ratio = edge_margin_ratio
        self.canny_low, self.canny_high = canny_thresholds

    def detect_stone_edge(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the stone edge (stemline) using Hough line detection.

        Args:
            image: Input image (BGR)

        Returns:
            Best line as [x1, y1, x2, y2] array, or None if not found
        """
        h, w = image.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better edge detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Edge detection
        edges = cv2.Canny(enhanced, self.canny_low, self.canny_high)

        # Morphological operations to clean up edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Hough line detection
        min_length = int(h * self.min_line_length_ratio)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=min_length,
            maxLineGap=20,
        )

        if lines is None or len(lines) == 0:
            return None

        # Find the best stemline candidate
        # Prefer near-vertical lines (stone edges are usually vertical)
        best_line = self._find_best_stemline(lines, h, w)
        return best_line

    def _find_best_stemline(
        self,
        lines: np.ndarray,
        image_height: int,
        image_width: int,
    ) -> Optional[np.ndarray]:
        """
        Find the best stemline candidate from detected lines.

        Criteria:
        1. Near-vertical orientation (±30°)
        2. Significant length
        3. Located towards edges of image (not center)
        """
        candidates = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle from vertical
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy == 0:
                continue  # Skip horizontal lines

            angle = np.arctan(dx / dy) * 180 / np.pi

            # Filter for near-vertical lines (±30°)
            if angle > 30:
                continue

            # Calculate length
            length = np.sqrt(dx ** 2 + dy ** 2)

            # Calculate center x position
            center_x = (x1 + x2) / 2

            # Score: prefer longer lines, prefer lines away from center
            edge_score = abs(center_x - image_width / 2) / (image_width / 2)
            length_score = length / image_height
            vertical_score = 1 - angle / 30

            total_score = length_score * 0.4 + edge_score * 0.3 + vertical_score * 0.3

            candidates.append((total_score, line[0]))

        if not candidates:
            return None

        # Return line with highest score
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def suggest_crop_region(
        self,
        image: np.ndarray,
        stemline: Optional[np.ndarray] = None,
    ) -> Tuple[int, int, int, int]:
        """
        Suggest a bounding box for the inscription region.

        Args:
            image: Input image
            stemline: Detected stemline [x1, y1, x2, y2], or None

        Returns:
            Bounding box as (x1, y1, x2, y2)
        """
        h, w = image.shape[:2]

        if stemline is not None:
            x1, y1, x2, y2 = stemline
            center_x = (x1 + x2) // 2

            # Crop region around detected stemline
            margin = int(w * self.edge_margin_ratio)
            crop_x1 = max(0, center_x - margin)
            crop_x2 = min(w, center_x + margin)

            return (crop_x1, 0, crop_x2, h)
        else:
            # Default: center strip
            margin = int(w * 0.1)
            return (margin, 0, w - margin, h)

    def interactive_crop(self, image_path: str) -> Dict:
        """
        Generate crop suggestion for manual verification.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with crop information for annotation tool
        """
        image = cv2.imread(image_path)
        if image is None:
            return {
                "image_path": image_path,
                "error": "Failed to load image",
                "needs_manual_review": True,
            }

        stemline = self.detect_stone_edge(image)
        suggested_bbox = self.suggest_crop_region(image, stemline)

        return {
            "image_path": image_path,
            "suggested_bbox": suggested_bbox,
            "stemline_detected": stemline is not None,
            "stemline": stemline.tolist() if stemline is not None else None,
            "confidence": "high" if stemline is not None else "low",
            "needs_manual_review": stemline is None,
            "image_size": (image.shape[0], image.shape[1]),
        }

    def apply_crop(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Apply a bounding box crop to an image.

        Args:
            image: Input image
            bbox: (x1, y1, x2, y2) bounding box

        Returns:
            Cropped image
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # Clamp to image bounds
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        return image[y1:y2, x1:x2]

    def batch_suggest_crops(self, image_paths: List[str]) -> List[Dict]:
        """
        Generate crop suggestions for multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of crop suggestions
        """
        suggestions = []
        for path in image_paths:
            suggestion = self.interactive_crop(path)
            suggestions.append(suggestion)
        return suggestions


def auto_crop_inscription(
    image: np.ndarray,
    min_confidence: str = "high",
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to auto-crop an inscription image.

    Args:
        image: Input image
        min_confidence: Minimum confidence required ("high" or "low")

    Returns:
        Tuple of (cropped_image, metadata)
    """
    cropper = InscriptionCropper()

    stemline = cropper.detect_stone_edge(image)
    bbox = cropper.suggest_crop_region(image, stemline)

    confidence = "high" if stemline is not None else "low"

    if min_confidence == "high" and confidence != "high":
        # Don't crop if not confident
        return image, {
            "cropped": False,
            "reason": "low_confidence",
            "confidence": confidence,
        }

    cropped = cropper.apply_crop(image, bbox)

    return cropped, {
        "cropped": True,
        "bbox": bbox,
        "confidence": confidence,
        "stemline_detected": stemline is not None,
    }
