"""Tests for the preprocessing pipeline."""

import pytest
import numpy as np
from src.preprocessing.preprocessor import OghamPreprocessor, PreprocessConfig


class TestAutoOrientation:
    """Tests for aspect-ratio-based orientation detection."""

    def _make_image(self, h, w):
        """Create a dummy BGR image."""
        return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    def test_tall_image_gets_rotated(self):
        """Images taller than threshold should be rotated 90° CW."""
        config = PreprocessConfig(
            fix_orientation=True,
            orientation_mode="auto",
            auto_orientation_threshold=1.2,
            crop_to_inscription=False,
            convert_greyscale=False,
            denoise=False,
            normalize_lighting=False,
            enhance_contrast=False,
            sharpen=False,
            resize=False,
        )
        proc = OghamPreprocessor(config)
        img = self._make_image(500, 200)  # h/w = 2.5, well above threshold

        result, log = proc.process(img)
        step = log["steps"][0]

        assert step["applied_rotation"] == "rotate_90_cw"
        assert result.shape == (200, 500, 3)  # width/height swapped

    def test_wide_image_kept_as_is(self):
        """Images wider than tall should not be rotated."""
        config = PreprocessConfig(
            fix_orientation=True,
            orientation_mode="auto",
            auto_orientation_threshold=1.2,
            crop_to_inscription=False,
            convert_greyscale=False,
            denoise=False,
            normalize_lighting=False,
            enhance_contrast=False,
            sharpen=False,
            resize=False,
        )
        proc = OghamPreprocessor(config)
        img = self._make_image(200, 500)  # h/w = 0.4, below threshold

        result, log = proc.process(img)
        step = log["steps"][0]

        assert step["applied_rotation"] == "none"
        assert result.shape == (200, 500, 3)

    def test_square_image_kept_as_is(self):
        """Nearly-square images (below threshold) should not be rotated."""
        config = PreprocessConfig(
            fix_orientation=True,
            orientation_mode="auto",
            auto_orientation_threshold=1.2,
            crop_to_inscription=False,
            convert_greyscale=False,
            denoise=False,
            normalize_lighting=False,
            enhance_contrast=False,
            sharpen=False,
            resize=False,
        )
        proc = OghamPreprocessor(config)
        img = self._make_image(300, 280)  # h/w = 1.07, below 1.2

        result, log = proc.process(img)
        assert log["steps"][0]["applied_rotation"] == "none"

    def test_forced_rotate_cw_ignores_aspect_ratio(self):
        """Explicit rotate_90_cw should always rotate regardless of shape."""
        config = PreprocessConfig(
            fix_orientation=True,
            orientation_mode="rotate_90_cw",
            crop_to_inscription=False,
            convert_greyscale=False,
            denoise=False,
            normalize_lighting=False,
            enhance_contrast=False,
            sharpen=False,
            resize=False,
        )
        proc = OghamPreprocessor(config)
        img = self._make_image(200, 500)  # wide image

        result, log = proc.process(img)
        assert log["steps"][0]["applied_rotation"] == "rotate_90_cw"
        assert result.shape == (500, 200, 3)


class TestGreyscaleConversion:
    """Tests for greyscale conversion."""

    def test_converts_to_greyscale_3ch(self):
        """Should produce 3-channel greyscale (all channels equal)."""
        config = PreprocessConfig(
            fix_orientation=False,
            crop_to_inscription=False,
            convert_greyscale=True,
            denoise=False,
            normalize_lighting=False,
            enhance_contrast=False,
            sharpen=False,
            resize=False,
        )
        proc = OghamPreprocessor(config)
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)

        result, log = proc.process(img)

        assert result.shape == (100, 200, 3)
        # All three channels should be identical after greyscale conversion
        assert np.array_equal(result[:, :, 0], result[:, :, 1])
        assert np.array_equal(result[:, :, 1], result[:, :, 2])


class TestAdaptiveCLAHE:
    """Tests for weathering-adaptive CLAHE."""

    def test_severe_gets_higher_clip(self):
        """Severe weathering should produce a higher clip limit."""
        config = PreprocessConfig(
            fix_orientation=False,
            crop_to_inscription=False,
            convert_greyscale=False,
            denoise=False,
            normalize_lighting=False,
            enhance_contrast=True,
            enhancement_method="clahe",
            adaptive_clahe=True,
            sharpen=False,
            resize=False,
        )
        proc = OghamPreprocessor(config)

        clip, source = proc._get_effective_clip_limit({"weathering_severity": "severe"})
        assert clip == 3.5
        assert source == "adaptive_severe"

    def test_minimal_gets_lower_clip(self):
        """Minimal weathering should produce a lower clip limit."""
        config = PreprocessConfig(
            fix_orientation=False,
            crop_to_inscription=False,
            convert_greyscale=False,
            denoise=False,
            normalize_lighting=False,
            enhance_contrast=True,
            adaptive_clahe=True,
            sharpen=False,
            resize=False,
        )
        proc = OghamPreprocessor(config)

        clip, source = proc._get_effective_clip_limit({"weathering_severity": "minimal"})
        assert clip == 1.5
        assert source == "adaptive_minimal"

    def test_missing_severity_uses_default(self):
        """Missing weathering_severity should fall back to config default."""
        config = PreprocessConfig(
            fix_orientation=False,
            crop_to_inscription=False,
            convert_greyscale=False,
            denoise=False,
            normalize_lighting=False,
            enhance_contrast=True,
            adaptive_clahe=True,
            clahe_clip_limit=2.0,
            sharpen=False,
            resize=False,
        )
        proc = OghamPreprocessor(config)

        clip, source = proc._get_effective_clip_limit({})
        assert clip == 2.0
        assert source == "config_default"

    def test_adaptive_disabled_uses_config(self):
        """When adaptive_clahe=False, always use config clip_limit."""
        config = PreprocessConfig(
            fix_orientation=False,
            crop_to_inscription=False,
            convert_greyscale=False,
            denoise=False,
            normalize_lighting=False,
            enhance_contrast=True,
            adaptive_clahe=False,
            clahe_clip_limit=2.0,
            sharpen=False,
            resize=False,
        )
        proc = OghamPreprocessor(config)

        clip, source = proc._get_effective_clip_limit({"weathering_severity": "severe"})
        assert clip == 2.0
        assert source == "config_default"


class TestAutoCrop:
    """Tests for vertical projection profile auto-cropping."""

    def test_vertical_strokes_get_cropped(self):
        """Image with vertical strokes in the centre should be cropped tightly."""
        config = PreprocessConfig(
            fix_orientation=False,
            crop_to_inscription=True,
            convert_greyscale=False,
            denoise=False,
            normalize_lighting=False,
            enhance_contrast=False,
            sharpen=False,
            resize=False,
        )
        proc = OghamPreprocessor(config)

        # Simulate a horizontal Ogham image: grey stone with vertical dark strokes
        # in a narrow band (columns 200-350, rows 120-280)
        img = np.full((400, 600, 3), 180, dtype=np.uint8)
        for col in range(200, 350, 12):
            img[120:280, col:col+3] = 30  # strong vertical dark lines

        result, log = proc.process(img)
        crop_step = [s for s in log["steps"] if s["step"] == "crop"][0]

        assert not crop_step.get("skipped", False), f"Crop was skipped: {crop_step}"
        assert crop_step["method"] == "vertical_projection"
        # Should have reduced the image size
        assert result.shape[1] < img.shape[1] or result.shape[0] < img.shape[0]

    def test_crop_tighter_than_original(self):
        """Cropped region should exclude large empty margins."""
        config = PreprocessConfig(
            fix_orientation=False,
            crop_to_inscription=True,
            crop_padding_fraction=0.05,
            convert_greyscale=False,
            denoise=False,
            normalize_lighting=False,
            enhance_contrast=False,
            sharpen=False,
            resize=False,
        )
        proc = OghamPreprocessor(config)

        # 800px wide image with strokes only in the middle 200px
        img = np.full((300, 800, 3), 170, dtype=np.uint8)
        for col in range(300, 500, 10):
            img[80:220, col:col+3] = 25

        result, log = proc.process(img)
        crop_step = [s for s in log["steps"] if s["step"] == "crop"][0]

        if not crop_step.get("skipped", False):
            # Width should be well under 800 (the strokes only span ~200px + padding)
            assert result.shape[1] < 500, f"Crop too wide: {result.shape[1]}"
            assert crop_step["crop_ratio"] < 0.7

    def test_uniform_image_not_cropped(self):
        """Uniform image with no edges should not be cropped."""
        config = PreprocessConfig(
            fix_orientation=False,
            crop_to_inscription=True,
            convert_greyscale=False,
            denoise=False,
            normalize_lighting=False,
            enhance_contrast=False,
            sharpen=False,
            resize=False,
        )
        proc = OghamPreprocessor(config)

        img = np.full((400, 600, 3), 128, dtype=np.uint8)

        result, log = proc.process(img)
        crop_step = [s for s in log["steps"] if s["step"] == "crop"][0]

        assert crop_step.get("skipped", False)

    def test_horizontal_edges_not_triggering_crop(self):
        """Only horizontal lines (no vertical edges) should not trigger crop."""
        config = PreprocessConfig(
            fix_orientation=False,
            crop_to_inscription=True,
            convert_greyscale=False,
            denoise=False,
            normalize_lighting=False,
            enhance_contrast=False,
            sharpen=False,
            resize=False,
        )
        proc = OghamPreprocessor(config)

        # Only horizontal lines — Sobel x-gradient won't respond to these
        img = np.full((400, 600, 3), 170, dtype=np.uint8)
        for row in range(100, 300, 15):
            img[row:row+2, 50:550] = 30  # horizontal dark lines

        result, log = proc.process(img)
        crop_step = [s for s in log["steps"] if s["step"] == "crop"][0]

        # Should skip or produce minimal crop — horizontal edges don't
        # indicate Ogham strokes
        # (may still detect some edges at line endpoints, but should be sparse)
        if not crop_step.get("skipped", False):
            # If it did crop, it should not have cropped much
            assert crop_step.get("crop_ratio", 1.0) > 0.6


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_doesnt_crash(self):
        """Full v3 pipeline should run without errors on a synthetic image."""
        config = PreprocessConfig(
            fix_orientation=True,
            orientation_mode="auto",
            crop_to_inscription=True,
            convert_greyscale=True,
            denoise=True,
            normalize_lighting=True,
            enhance_contrast=True,
            enhancement_method="clahe",
            adaptive_clahe=True,
            sharpen=True,
            resize=True,
            target_height=384,
        )
        proc = OghamPreprocessor(config)

        # Tall image with some edges
        img = np.random.randint(50, 200, (800, 300, 3), dtype=np.uint8)
        # Add some strokes
        for i in range(100, 200, 15):
            img[100:700, i:i+4] = 20

        result, log = proc.process(img, metadata={"weathering_severity": "moderate"})

        assert result.shape[0] == 384  # target height
        # 8 steps: orientation, crop, greyscale, denoise, lighting, enhance, sharpen, resize
        assert len(log["steps"]) == 8
        assert log["version"] == "3.0.0"
        # Verify all step types present
        step_names = [s["step"] for s in log["steps"]]
        assert "denoise" in step_names
        assert "normalize_lighting" in step_names
        assert "sharpen" in step_names
