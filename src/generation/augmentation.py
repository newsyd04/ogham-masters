"""
Augmentation pipeline for Ogham OCR training.

Provides realistic degradation transforms that simulate:
- Weathering (noise, blur, fading)
- Lighting variations (shadows, brightness)
- Physical damage (moss, cracks, occlusion)

★ Insight ─────────────────────────────────────
Augmentation strategy:
1. Light augmentation for curriculum learning start
2. Progressive degradation matching real weathering
3. Domain-specific transforms (moss patches, shadows)
4. Geometric variations simulate different camera angles
─────────────────────────────────────────────────
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None
    ToTensorV2 = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


class OghamAugmentation:
    """
    Augmentation pipeline for Ogham inscription images.

    Provides transforms at different severity levels for
    curriculum learning.
    """

    @staticmethod
    def get_train_transforms(
        severity: str = "medium",
        image_height: int = 384,
        normalize: bool = True,
    ) -> Any:
        """
        Get training augmentation transforms.

        Args:
            severity: "light", "medium", or "heavy"
            image_height: Target image height
            normalize: Whether to apply ImageNet normalization

        Returns:
            Albumentations Compose object
        """
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError(
                "albumentations is required for augmentation. "
                "Install with: pip install albumentations"
            )

        # Base geometric transforms (always applied)
        base_transforms = [
            # Slight rotation and shift
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=3,  # Slight rotation only
                border_mode=cv2.BORDER_CONSTANT if CV2_AVAILABLE else 0,
                value=(200, 200, 200),  # Light gray border
                p=0.5,
            ),
            # Perspective (simulate camera angle)
            A.Perspective(scale=(0.02, 0.05), p=0.3),
        ]

        # Severity-specific transforms
        severity_transforms = {
            "light": [
                A.GaussNoise(var_limit=(5, 15), p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.5,
                ),
            ],
            "medium": [
                # Stone grain texture simulation
                A.GaussNoise(var_limit=(10, 30), p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), p=0.3),
                # Weathering: faded contrast (inscriptions lose sharpness)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.7,
                ),
                # Stroke fading: reduce contrast in carved grooves
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                # Shadow simulation (outdoor photography)
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=4,
                    p=0.4,
                ),
                # Moss/lichen patches (greenish occlusion)
                A.CoarseDropout(
                    max_holes=5,
                    max_height=20,
                    max_width=20,
                    min_height=5,
                    min_width=5,
                    fill_value=(100, 120, 90),  # Greenish
                    p=0.3,
                ),
                # Greyscale (match preprocessed real images)
                A.ToGray(p=0.5),
            ],
            "heavy": [
                # Severe stone grain
                A.GaussNoise(var_limit=(20, 50), p=0.7),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=5, p=0.3),
                # Heavy weathering: significant contrast/brightness loss
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.8,
                ),
                # Severe stroke fading (inscriptions barely visible)
                A.RandomGamma(gamma_limit=(60, 140), p=0.6),
                # Heavy shadows from variable outdoor lighting
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_lower=2,
                    num_shadows_upper=4,
                    shadow_dimension=5,
                    p=0.6,
                ),
                # Lichen/moss patches covering inscription
                A.CoarseDropout(
                    max_holes=8,
                    max_height=25,
                    max_width=25,
                    min_height=8,
                    min_width=8,
                    fill_value=(100, 120, 90),  # Green lichen
                    p=0.4,
                ),
                # Deep cracks in stone
                A.CoarseDropout(
                    max_holes=5,
                    max_height=40,
                    max_width=4,
                    min_height=15,
                    min_width=1,
                    fill_value=0,  # Black (deep cracks)
                    p=0.3,
                ),
                # Partial erosion / damage
                A.GridDropout(ratio=0.15, random_offset=True, p=0.25),
                # Greyscale (match preprocessed real images)
                A.ToGray(p=0.6),
                A.ChannelDropout(channel_drop_range=(1, 1), p=0.15),
            ],
        }

        transforms = base_transforms + severity_transforms.get(severity, severity_transforms["medium"])

        # Normalization
        if normalize:
            transforms.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )

        # Convert to tensor
        transforms.append(ToTensorV2())

        return A.Compose(transforms)

    @staticmethod
    def get_val_transforms(normalize: bool = True) -> Any:
        """
        Get validation transforms (minimal, just normalization).

        Args:
            normalize: Whether to apply ImageNet normalization

        Returns:
            Albumentations Compose object
        """
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations is required")

        transforms = []

        if normalize:
            transforms.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )

        transforms.append(ToTensorV2())

        return A.Compose(transforms)

    @staticmethod
    def get_test_time_augmentation() -> List[Any]:
        """
        Get test-time augmentation transforms.

        Returns multiple transform variants for TTA.
        """
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations is required")

        tta_transforms = [
            # Original (no augmentation)
            A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Slight brightness increase
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=0, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Slight brightness decrease
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.1), contrast_limit=0, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            # Horizontal flip (valid for some Ogham orientations)
            A.Compose([
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
        ]

        return tta_transforms


# Convenience functions for common use cases
def get_train_transforms(severity: str = "medium", normalize: bool = True) -> Any:
    """Get training transforms."""
    return OghamAugmentation.get_train_transforms(severity, normalize=normalize)


def get_val_transforms(normalize: bool = True) -> Any:
    """Get validation transforms."""
    return OghamAugmentation.get_val_transforms(normalize)


class SimpleAugmentation:
    """
    Simple augmentation without albumentations dependency.

    Provides basic transforms using only numpy and cv2.
    """

    def __init__(
        self,
        noise_std: float = 15.0,
        brightness_delta: float = 0.1,
        blur_kernel: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Initialize simple augmentation.

        Args:
            noise_std: Standard deviation of Gaussian noise
            brightness_delta: Max brightness change (-delta to +delta)
            blur_kernel: Gaussian blur kernel size
            seed: Random seed
        """
        self.noise_std = noise_std
        self.brightness_delta = brightness_delta
        self.blur_kernel = blur_kernel
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        if not CV2_AVAILABLE:
            return image

        result = image.copy().astype(np.float32)

        # Random Gaussian noise
        if self.rng.random() > 0.5:
            noise = self.rng.normal(0, self.noise_std, result.shape)
            result += noise

        # Random brightness
        if self.rng.random() > 0.5:
            delta = self.rng.uniform(-self.brightness_delta, self.brightness_delta)
            result = result * (1 + delta)

        # Random blur
        if self.rng.random() > 0.7:
            result = cv2.GaussianBlur(
                result.astype(np.uint8),
                (self.blur_kernel, self.blur_kernel),
                0,
            ).astype(np.float32)

        # Clip and convert
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result
