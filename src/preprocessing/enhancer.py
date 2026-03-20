"""
Image enhancement utilities for Ogham inscription images.

Provides various enhancement methods to improve visibility of weathered
inscriptions, with support for A/B testing different approaches.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class ImageEnhancer:
    """
    Image enhancement for stone inscription images.

    Provides multiple enhancement methods:
    - CLAHE: Contrast Limited Adaptive Histogram Equalization
    - Bilateral: Edge-preserving denoising
    - Retinex: Illumination normalization
    - Combined: Multiple methods in sequence
    """

    # Enhancement experiment configurations
    ENHANCEMENT_CONFIGS = {
        "baseline": {
            "description": "No enhancement, just resize",
            "methods": [],
        },
        "clahe": {
            "description": "Contrast Limited Adaptive Histogram Equalization",
            "methods": [("clahe", {"clip_limit": 2.0, "tile_size": (8, 8)})],
        },
        "clahe_mild": {
            "description": "Mild CLAHE for subtle enhancement",
            "methods": [("clahe", {"clip_limit": 1.5, "tile_size": (8, 8)})],
        },
        "clahe_strong": {
            "description": "Strong CLAHE for weathered stones",
            "methods": [("clahe", {"clip_limit": 3.0, "tile_size": (4, 4)})],
        },
        "bilateral": {
            "description": "Edge-preserving bilateral filter",
            "methods": [("bilateral", {"d": 9, "sigma_color": 75, "sigma_space": 75})],
        },
        "retinex": {
            "description": "Multi-scale Retinex for shadow normalization",
            "methods": [("retinex", {"sigma_list": [15, 80, 250]})],
        },
        "combined": {
            "description": "Bilateral denoising + CLAHE",
            "methods": [
                ("bilateral", {"d": 5, "sigma_color": 50, "sigma_space": 50}),
                ("clahe", {"clip_limit": 1.5, "tile_size": (8, 8)}),
            ],
        },
        "weathered": {
            "description": "Aggressive enhancement for severely weathered stones",
            "methods": [
                ("bilateral", {"d": 9, "sigma_color": 100, "sigma_space": 100}),
                ("clahe", {"clip_limit": 3.0, "tile_size": (4, 4)}),
                ("unsharp_mask", {"amount": 1.0, "radius": 1.0}),
            ],
        },
    }

    def __init__(self, config_name: str = "clahe"):
        """
        Initialize enhancer with named configuration.

        Args:
            config_name: Name of enhancement configuration
        """
        if config_name not in self.ENHANCEMENT_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Options: {list(self.ENHANCEMENT_CONFIGS.keys())}")

        self.config_name = config_name
        self.config = self.ENHANCEMENT_CONFIGS[config_name]

    def enhance(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Apply enhancement pipeline to image.

        Args:
            image: Input image (BGR)

        Returns:
            Tuple of (enhanced_image, enhancement_info)
        """
        result = image.copy()
        steps_applied = []

        for method_name, params in self.config["methods"]:
            if method_name == "clahe":
                result = self._apply_clahe(result, **params)
            elif method_name == "bilateral":
                result = self._apply_bilateral(result, **params)
            elif method_name == "retinex":
                result = self._apply_retinex(result, **params)
            elif method_name == "unsharp_mask":
                result = self._apply_unsharp_mask(result, **params)

            steps_applied.append({"method": method_name, "params": params})

        return result, {
            "config": self.config_name,
            "description": self.config["description"],
            "steps": steps_applied,
        }

    def _apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        Apply CLAHE to luminance channel.

        CLAHE prevents over-amplification of noise in homogeneous regions
        while enhancing local contrast.
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_enhanced = clahe.apply(l_channel)

        # Merge and convert back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def _apply_bilateral(
        self,
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75,
    ) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving smoothing.

        Reduces noise while preserving the sharp edges of carved characters.
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def _apply_retinex(
        self,
        image: np.ndarray,
        sigma_list: List[int] = [15, 80, 250],
    ) -> np.ndarray:
        """
        Apply Multi-Scale Retinex with Color Restoration (MSRCR).

        Good for normalizing illumination variations across the image.
        """
        # Convert to float
        img_float = image.astype(np.float32) + 1.0

        # Apply Retinex to each channel
        result = np.zeros_like(img_float)

        for c in range(3):
            channel = img_float[:, :, c]
            log_channel = np.log(channel)

            # Multi-scale processing
            retinex = np.zeros_like(channel)
            for sigma in sigma_list:
                blur = cv2.GaussianBlur(channel, (0, 0), sigma)
                retinex += log_channel - np.log(blur + 1.0)

            retinex /= len(sigma_list)

            # Normalize
            result[:, :, c] = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)

        return result.astype(np.uint8)

    def _apply_unsharp_mask(
        self,
        image: np.ndarray,
        amount: float = 1.0,
        radius: float = 1.0,
    ) -> np.ndarray:
        """
        Apply unsharp masking for edge enhancement.

        Sharpens character edges for better recognition.
        """
        # Create blurred version
        blurred = cv2.GaussianBlur(image, (0, 0), radius)

        # Unsharp mask: original + amount * (original - blurred)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

        return sharpened

    @classmethod
    def list_configs(cls) -> Dict[str, str]:
        """Return available configuration names and descriptions."""
        return {name: config["description"] for name, config in cls.ENHANCEMENT_CONFIGS.items()}


def compare_enhancements(
    image: np.ndarray,
    configs: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Apply multiple enhancement configurations for comparison.

    Args:
        image: Input image
        configs: List of config names (default: all)

    Returns:
        Dictionary mapping config names to enhanced images
    """
    if configs is None:
        configs = list(ImageEnhancer.ENHANCEMENT_CONFIGS.keys())

    results = {"original": image.copy()}

    for config_name in configs:
        enhancer = ImageEnhancer(config_name)
        enhanced, _ = enhancer.enhance(image)
        results[config_name] = enhanced

    return results


def create_comparison_grid(
    images: Dict[str, np.ndarray],
    cols: int = 3,
    tile_size: Tuple[int, int] = (400, 400),
) -> np.ndarray:
    """
    Create a comparison grid of enhanced images.

    Args:
        images: Dictionary of name -> image
        cols: Number of columns in grid
        tile_size: Size of each tile (width, height)

    Returns:
        Grid image
    """
    names = list(images.keys())
    n = len(names)
    rows = (n + cols - 1) // cols

    tile_w, tile_h = tile_size
    grid = np.ones((rows * tile_h, cols * tile_w, 3), dtype=np.uint8) * 255

    for i, name in enumerate(names):
        row = i // cols
        col = i % cols

        img = images[name]
        # Resize to tile size
        resized = cv2.resize(img, tile_size)

        # Add label
        cv2.putText(
            resized, name, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
        )

        # Place in grid
        y1, y2 = row * tile_h, (row + 1) * tile_h
        x1, x2 = col * tile_w, (col + 1) * tile_w
        grid[y1:y2, x1:x2] = resized

    return grid
