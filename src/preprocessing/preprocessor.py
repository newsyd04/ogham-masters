"""
Main preprocessing pipeline for Ogham inscription images.

This module orchestrates the preprocessing steps:
1. Orientation correction (vertical to horizontal for TrOCR)
2. Optional cropping to inscription region
3. Contrast enhancement
4. Resizing for model input

★ Insight ─────────────────────────────────────
Key preprocessing considerations for Ogham:
1. TrOCR expects left-to-right reading; Ogham reads bottom-to-top
2. Stone inscriptions have variable aspect ratios
3. Weathering creates challenging low-contrast regions
─────────────────────────────────────────────────
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import json
import logging

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""

    # Orientation
    fix_orientation: bool = True
    orientation_mode: str = "auto"  # "auto", "rotate_90_cw", "rotate_90_ccw", "keep_vertical"
    auto_orientation_threshold: float = 1.2  # h/w ratio above which image is considered vertical

    # Cropping
    crop_to_inscription: bool = True  # Use edge-based auto-crop when no bbox provided
    crop_padding_fraction: float = 0.05  # Padding around detected inscription region

    # Greyscale
    convert_greyscale: bool = True  # Remove colour to reduce domain gap

    # Denoising (reduces stone grain texture, improves domain gap)
    denoise: bool = True
    bilateral_d: int = 7           # Filter neighbourhood diameter
    bilateral_sigma_color: float = 50.0  # Color similarity threshold
    bilateral_sigma_space: float = 50.0  # Spatial extent

    # Shadow/lighting normalisation (removes directional lighting)
    normalize_lighting: bool = True
    retinex_sigma: float = 80.0  # Gaussian blur sigma for illumination estimate

    # Enhancement
    enhance_contrast: bool = True
    enhancement_method: str = "clahe"  # "clahe", "bilateral", "none"
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    adaptive_clahe: bool = True  # Scale clip_limit based on weathering_severity metadata

    # Sharpening (makes faint inscription strokes crisper)
    sharpen: bool = True
    unsharp_amount: float = 1.0   # Sharpening strength
    unsharp_sigma: float = 1.5    # Edge detection scale

    # Resizing
    resize: bool = True
    target_height: int = 384
    max_width: int = 2048
    min_width: int = 128
    padding_value: int = 255  # White padding

    # Normalization
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Output
    output_format: str = "png"
    preserve_originals: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "fix_orientation": self.fix_orientation,
            "orientation_mode": self.orientation_mode,
            "auto_orientation_threshold": self.auto_orientation_threshold,
            "crop_to_inscription": self.crop_to_inscription,
            "crop_padding_fraction": self.crop_padding_fraction,
            "convert_greyscale": self.convert_greyscale,
            "denoise": self.denoise,
            "bilateral_d": self.bilateral_d,
            "bilateral_sigma_color": self.bilateral_sigma_color,
            "bilateral_sigma_space": self.bilateral_sigma_space,
            "normalize_lighting": self.normalize_lighting,
            "retinex_sigma": self.retinex_sigma,
            "enhance_contrast": self.enhance_contrast,
            "enhancement_method": self.enhancement_method,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_size": self.clahe_tile_size,
            "adaptive_clahe": self.adaptive_clahe,
            "sharpen": self.sharpen,
            "unsharp_amount": self.unsharp_amount,
            "unsharp_sigma": self.unsharp_sigma,
            "resize": self.resize,
            "target_height": self.target_height,
            "max_width": self.max_width,
            "min_width": self.min_width,
            "padding_value": self.padding_value,
            "normalize": self.normalize,
            "mean": self.mean,
            "std": self.std,
            "output_format": self.output_format,
        }


class OghamPreprocessor:
    """
    Preprocessing pipeline for Ogham inscription images.

    Transforms raw stone photographs into model-ready inputs.
    """

    VERSION = "3.0.0"

    def __init__(self, config: PreprocessConfig):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = logging.getLogger("ogham_preprocessor")

        # Build transform pipeline
        self.transforms = self._build_transform_pipeline()

    def _build_transform_pipeline(self) -> List[Callable]:
        """Build list of transform functions from config.

        Pipeline order:
        1. Orient (rotate vertical stones to horizontal)
        2. Crop (isolate inscription region)
        3. Greyscale (remove colour noise)
        4. Denoise (bilateral filter to smooth stone texture)
        5. Lighting normalisation (Retinex to remove shadows)
        6. Contrast enhancement (CLAHE)
        7. Sharpen (unsharp mask for faint strokes)
        8. Resize (scale to model input dimensions)
        """
        pipeline = []

        if self.config.fix_orientation:
            pipeline.append(self._fix_orientation)

        if self.config.crop_to_inscription:
            pipeline.append(self._crop_to_inscription)

        if self.config.convert_greyscale:
            pipeline.append(self._convert_greyscale)

        if self.config.denoise:
            pipeline.append(self._denoise)

        if self.config.normalize_lighting:
            pipeline.append(self._normalize_lighting)

        if self.config.enhance_contrast:
            pipeline.append(self._enhance_contrast)

        if self.config.sharpen:
            pipeline.append(self._sharpen)

        if self.config.resize:
            pipeline.append(self._resize_for_model)

        return pipeline

    def process(
        self,
        image: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply full preprocessing pipeline to an image.

        Args:
            image: Input image as numpy array (BGR format from cv2)
            metadata: Optional metadata dict (updated with processing info)

        Returns:
            Tuple of (processed_image, processing_log)
        """
        if metadata is None:
            metadata = {}

        processed = image.copy()
        processing_log = {
            "version": self.VERSION,
            "config": self.config.to_dict(),
            "steps": [],
            "input_shape": image.shape,
        }

        # Apply each transform
        for transform in self.transforms:
            processed, step_info = transform(processed, metadata)
            processing_log["steps"].append(step_info)

        processing_log["output_shape"] = processed.shape

        return processed, processing_log

    def process_file(
        self,
        input_path: str,
        output_path: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Process an image file and save result.

        Args:
            input_path: Path to input image
            output_path: Path for output image
            metadata: Optional metadata

        Returns:
            Processing log dictionary
        """
        # Load image
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Failed to load image: {input_path}")

        # Process
        processed, log = self.process(image, metadata)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save processed image
        cv2.imwrite(str(output_path), processed)

        # Save processing log
        log_path = str(output_path) + ".json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        log["output_path"] = str(output_path)
        return log

    def _fix_orientation(
        self,
        image: np.ndarray,
        metadata: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fix image orientation for TrOCR processing.

        Ogham reads bottom-to-top, but TrOCR expects left-to-right.

        In "auto" mode, uses aspect ratio to decide:
        - Tall images (h/w > threshold): likely vertical stone, rotate 90° CW
        - Wide images: likely already photographed horizontally, keep as-is
        """
        h, w = image.shape[:2]
        mode = self.config.orientation_mode

        if mode == "auto":
            aspect_ratio = h / w
            if aspect_ratio > self.config.auto_orientation_threshold:
                # Tall image — vertical stone, needs rotation
                rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                applied_rotation = "rotate_90_cw"
                new_direction = "left_to_right"
            else:
                # Wide or square image — already horizontal
                rotated = image
                applied_rotation = "none"
                new_direction = "left_to_right"
        elif mode == "rotate_90_cw":
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            applied_rotation = "rotate_90_cw"
            new_direction = "left_to_right"
        elif mode == "rotate_90_ccw":
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            applied_rotation = "rotate_90_ccw"
            new_direction = "right_to_left"
        else:
            rotated = image
            applied_rotation = "none"
            new_direction = metadata.get("reading_direction", "unknown")

        step_info = {
            "step": "orientation",
            "mode": mode,
            "applied_rotation": applied_rotation,
            "aspect_ratio": round(h / w, 3),
            "new_direction": new_direction,
            "original_shape": image.shape,
            "new_shape": rotated.shape,
        }

        return rotated, step_info

    def _crop_to_inscription(
        self,
        image: np.ndarray,
        metadata: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Crop image to inscription region.

        Priority:
        1. Use pre-computed bbox from metadata if available
        2. Otherwise, auto-detect inscription via edge density
        """
        bbox = metadata.get("bbox")

        if bbox is not None:
            return self._crop_from_bbox(image, bbox)

        # Auto-detect inscription region via edge concentration
        return self._auto_crop_inscription(image)

    def _crop_from_bbox(
        self, image: np.ndarray, bbox: Tuple
    ) -> Tuple[np.ndarray, Dict]:
        """Crop using a pre-computed bounding box."""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return image, {"step": "crop", "skipped": True, "reason": "invalid_bbox"}

        cropped = image[y1:y2, x1:x2]
        return cropped, {
            "step": "crop",
            "method": "bbox",
            "bbox": [x1, y1, x2, y2],
            "original_shape": image.shape,
            "cropped_shape": cropped.shape,
        }

    def _auto_crop_inscription(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Auto-detect inscription region using multi-scale vertical edge projection.

        Uses multi-scale Sobel (kernel sizes 3, 5, 7) to detect vertical edges
        across a range of stroke widths, with adaptive thresholds based on
        image noise floor rather than fixed percentages.

        Steps:
        1. Bilateral pre-filter to suppress stone texture
        2. Multi-scale Sobel x-gradient for vertical edges
        3. Otsu binarisation of combined edge map
        4. Project onto y-axis → find stroke band (adaptive threshold)
        5. Project onto x-axis → find inscription span (adaptive threshold)
        6. Crop with padding
        """
        h, w = image.shape[:2]
        pad_frac = self.config.crop_padding_fraction

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Bilateral filter: smooth stone grain, preserve carved edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Multi-scale Sobel: combine x-gradients at kernel sizes 3, 5, 7
        # to detect strokes across 5-15px width range
        vert_edge_mag = np.zeros_like(filtered, dtype=np.float64)
        for ksize in (3, 5, 7):
            sobel_x = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=ksize)
            vert_edge_mag += np.abs(sobel_x)
        vert_edge_mag /= 3.0

        # Normalize to 0-255 range for consistent thresholding
        if vert_edge_mag.max() > 0:
            vert_edges = (vert_edge_mag / vert_edge_mag.max() * 255).astype(np.uint8)
        else:
            return image, {"step": "crop", "method": "vertical_projection",
                           "skipped": True, "reason": "no_vertical_edges"}

        # Otsu threshold: automatically splits stroke edges from noise
        _, binary = cv2.threshold(vert_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Check there are enough edges to work with
        edge_density = np.sum(binary > 0) / (h * w)
        if edge_density < 0.002:
            return image, {"step": "crop", "method": "vertical_projection",
                           "skipped": True, "reason": "insufficient_vertical_edges",
                           "edge_density": round(edge_density, 5)}

        # Project vertical edges onto y-axis (sum each row)
        y_profile = np.sum(binary > 0, axis=1).astype(np.float64)

        # Smooth the profile to avoid noise spikes
        kernel_size = max(3, h // 20) | 1  # odd number, ~5% of height
        y_profile_smooth = cv2.GaussianBlur(
            y_profile.reshape(-1, 1), (1, kernel_size), 0
        ).flatten()

        # Adaptive threshold: midpoint between noise floor and signal peak
        y_noise_floor = float(np.percentile(y_profile_smooth, 25))
        y_peak = float(np.max(y_profile_smooth))
        y_thresh = y_noise_floor + 0.4 * (y_peak - y_noise_floor)
        y_active = np.where(y_profile_smooth > y_thresh)[0]

        if len(y_active) == 0:
            return image, {"step": "crop", "method": "vertical_projection",
                           "skipped": True, "reason": "no_vertical_band_found"}

        y1 = int(y_active[0])
        y2 = int(y_active[-1])

        # Project vertical edges onto x-axis (sum each column)
        x_profile = np.sum(binary > 0, axis=0).astype(np.float64)

        kernel_size_x = max(3, w // 20) | 1
        x_profile_smooth = cv2.GaussianBlur(
            x_profile.reshape(-1, 1), (1, kernel_size_x), 0
        ).flatten()

        # Adaptive threshold for x-axis
        x_noise_floor = float(np.percentile(x_profile_smooth, 25))
        x_peak = float(np.max(x_profile_smooth))
        x_thresh = x_noise_floor + 0.3 * (x_peak - x_noise_floor)
        x_active = np.where(x_profile_smooth > x_thresh)[0]

        if len(x_active) == 0:
            return image, {"step": "crop", "method": "vertical_projection",
                           "skipped": True, "reason": "no_horizontal_span_found"}

        x1 = int(x_active[0])
        x2 = int(x_active[-1])

        # Add padding
        pad_y = int((y2 - y1) * pad_frac)
        pad_x = int((x2 - x1) * pad_frac)
        y1 = max(0, y1 - pad_y)
        y2 = min(h, y2 + pad_y)
        x1 = max(0, x1 - pad_x)
        x2 = min(w, x2 + pad_x)

        # Safety checks
        crop_area = (x2 - x1) * (y2 - y1)
        image_area = h * w
        crop_ratio = crop_area / image_area

        if crop_ratio < 0.05:
            return image, {"step": "crop", "method": "vertical_projection",
                           "skipped": True, "reason": "detected_region_too_small",
                           "crop_ratio": round(crop_ratio, 3)}

        cropped = image[y1:y2, x1:x2]

        return cropped, {
            "step": "crop",
            "method": "vertical_projection",
            "scales": [3, 5, 7],
            "bbox": [x1, y1, x2, y2],
            "crop_ratio": round(crop_ratio, 3),
            "edge_density": round(edge_density, 5),
            "original_shape": image.shape,
            "cropped_shape": cropped.shape,
        }

    def _denoise(
        self,
        image: np.ndarray,
        metadata: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Bilateral denoising to smooth stone texture while preserving edges.

        Bilateral filters preserve sharp edges (carved strokes) while
        smoothing gradual changes (stone grain, weathering, moss). This
        reduces the domain gap between textured real images and clean
        synthetic images.
        """
        denoised = cv2.bilateralFilter(
            image,
            self.config.bilateral_d,
            self.config.bilateral_sigma_color,
            self.config.bilateral_sigma_space,
        )

        return denoised, {
            "step": "denoise",
            "method": "bilateral",
            "diameter": self.config.bilateral_d,
            "sigma_color": self.config.bilateral_sigma_color,
            "sigma_space": self.config.bilateral_sigma_space,
        }

    def _normalize_lighting(
        self,
        image: np.ndarray,
        metadata: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Single-scale Retinex to normalise illumination.

        Removes directional lighting and shadows from outdoor stone
        photographs. Works by estimating the illumination component
        (low-frequency) and dividing it out, leaving only reflectance
        (the stone surface detail).
        """
        # Work in LAB space — only normalise the L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32) + 1.0

        # Estimate illumination via Gaussian blur
        l_blur = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=self.config.retinex_sigma)

        # Retinex: log(image) - log(illumination) = log(reflectance)
        l_retinex = np.log(l_channel) - np.log(l_blur + 1.0)

        # Normalise back to 0-255
        l_norm = cv2.normalize(l_retinex, None, 0, 255, cv2.NORM_MINMAX)
        lab[:, :, 0] = l_norm.astype(np.uint8)

        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return result, {
            "step": "normalize_lighting",
            "method": "retinex",
            "sigma": self.config.retinex_sigma,
        }

    def _sharpen(
        self,
        image: np.ndarray,
        metadata: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Unsharp masking to sharpen faint inscription strokes.

        Subtracts a blurred version of the image to emphasise
        high-frequency detail (carved edges). Particularly helpful
        for severely weathered stones where CLAHE alone leaves
        strokes looking soft.
        """
        # Create blurred version
        blurred = cv2.GaussianBlur(
            image.astype(np.float32),
            (0, 0),
            sigmaX=self.config.unsharp_sigma,
        )

        # Unsharp mask: original + amount * (original - blurred)
        amount = self.config.unsharp_amount
        sharpened = image.astype(np.float32) + amount * (image.astype(np.float32) - blurred)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        return sharpened, {
            "step": "sharpen",
            "method": "unsharp_mask",
            "amount": amount,
            "sigma": self.config.unsharp_sigma,
        }

    # Weathering severity → CLAHE clip_limit mapping
    WEATHERING_CLIP_LIMITS = {
        "minimal": 1.5,
        "moderate": 2.5,
        "severe": 3.5,
    }

    def _convert_greyscale(
        self,
        image: np.ndarray,
        metadata: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Convert to greyscale and replicate to 3 channels.

        Ogham strokes are purely structural — colour adds no useful
        signal and introduces domain gap (moss, lichen, lighting).
        The 3-channel output maintains compatibility with the ViT encoder.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return image_3ch, {
            "step": "greyscale",
            "original_channels": image.shape[2] if len(image.shape) == 3 else 1,
        }

    def _get_effective_clip_limit(self, metadata: Dict) -> Tuple[float, str]:
        """
        Determine CLAHE clip limit, optionally adapting to weathering severity.

        Returns:
            Tuple of (clip_limit, source) where source describes how it was chosen.
        """
        if not self.config.adaptive_clahe:
            return self.config.clahe_clip_limit, "config_default"

        severity = metadata.get("weathering_severity")
        if severity and severity in self.WEATHERING_CLIP_LIMITS:
            return self.WEATHERING_CLIP_LIMITS[severity], f"adaptive_{severity}"

        return self.config.clahe_clip_limit, "config_default"

    def _enhance_contrast(
        self,
        image: np.ndarray,
        metadata: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Enhance image contrast via CLAHE.

        When adaptive_clahe is enabled, clip_limit scales with weathering severity.
        Retinex is now a separate pipeline step (normalize_lighting).
        """
        method = self.config.enhancement_method

        if method == "none":
            return image, {"step": "enhance", "method": "none", "skipped": True}

        # Convert to LAB color space for luminance processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clip_limit, clip_source = self._get_effective_clip_limit(metadata)

        if method == "clahe":
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=self.config.clahe_tile_size,
            )
            l_enhanced = clahe.apply(l_channel)

        elif method == "bilateral":
            # Bilateral filter for edge-preserving smoothing + CLAHE
            l_enhanced = cv2.bilateralFilter(l_channel, 9, 75, 75)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_enhanced)

        else:
            return image, {"step": "enhance", "method": method, "error": "unknown_method"}

        # Merge channels and convert back
        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        step_info = {
            "step": "enhance",
            "method": method,
            "clip_limit": clip_limit,
            "clip_source": clip_source,
            "tile_size": list(self.config.clahe_tile_size),
        }

        return enhanced, step_info

    def _resize_for_model(
        self,
        image: np.ndarray,
        metadata: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resize image for model input.

        Maintains aspect ratio, scales to target height,
        pads width to multiple of 16 for ViT patches.
        """
        h, w = image.shape[:2]

        # Scale to target height
        scale = self.config.target_height / h
        new_w = int(w * scale)

        # Clamp width
        new_w = max(self.config.min_width, min(new_w, self.config.max_width))

        # Resize
        resized = cv2.resize(
            image,
            (new_w, self.config.target_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )

        # Pad width to multiple of 16
        pad_w = (16 - new_w % 16) % 16
        if pad_w > 0:
            padding = np.full(
                (self.config.target_height, pad_w, 3),
                self.config.padding_value,
                dtype=np.uint8,
            )
            resized = np.concatenate([resized, padding], axis=1)

        step_info = {
            "step": "resize",
            "original_size": (h, w),
            "scale_factor": scale,
            "new_width_before_padding": new_w,
            "padding_added": pad_w,
            "final_size": resized.shape[:2],
        }

        return resized, step_info

    def normalize_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        Apply ImageNet normalization for model input.

        Args:
            image: Image in BGR uint8 format

        Returns:
            Normalized image as float32
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Scale to 0-1
        normalized = rgb.astype(np.float32) / 255.0

        # Apply mean/std normalization
        mean = np.array(self.config.mean)
        std = np.array(self.config.std)
        normalized = (normalized - mean) / std

        return normalized


def create_preprocessor(
    enhancement: str = "clahe",
    target_height: int = 384,
    **kwargs,
) -> OghamPreprocessor:
    """
    Create preprocessor with common configurations.

    Args:
        enhancement: Enhancement method ("clahe", "bilateral", "none")
        target_height: Target image height
        **kwargs: Additional config options

    Returns:
        Configured OghamPreprocessor
    """
    config = PreprocessConfig(
        enhancement_method=enhancement,
        target_height=target_height,
        **kwargs,
    )
    return OghamPreprocessor(config)
