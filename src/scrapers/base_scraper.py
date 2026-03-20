"""
Base scraper class for Ogham image sources.

This module provides the abstract base class that all source-specific scrapers
inherit from, implementing common functionality like rate limiting, logging,
and file management.

★ Insight ─────────────────────────────────────
Key design decisions:
1. Rate limiting is built into the base class to respect robots.txt
2. All downloads include provenance metadata for reproducibility
3. Abstract methods force consistent interfaces across scrapers
─────────────────────────────────────────────────
"""

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse
import requests

from ..schemas import (
    ImageMetadata,
    StoneMetadata,
    LicenseType,
    TranscriptionConfidence,
    ImageType,
    QualityLevel,
)


@dataclass
class ImageDownload:
    """Result of downloading an image."""

    success: bool
    image_path: Optional[Path] = None
    metadata: Optional[ImageMetadata] = None
    error: Optional[str] = None


@dataclass
class ScraperConfig:
    """Configuration for a scraper instance."""

    output_dir: Path
    rate_limit: float = 2.0  # Seconds between requests
    max_retries: int = 3
    timeout: int = 30
    user_agent: str = "OghamOCR-Research/1.0 (University research project; dissertation research)"
    respect_robots: bool = True
    save_metadata: bool = True
    log_level: int = logging.INFO


class OghamScraperBase(ABC):
    """
    Abstract base class for Ogham image scrapers.

    Provides common functionality:
    - Rate-limited HTTP requests
    - Structured logging
    - Consistent file organization
    - Metadata management

    Subclasses must implement:
    - get_stone_listing(): Return list of available stones
    - download_stone_images(): Download images for a specific stone
    - parse_transcription(): Extract transcription from page
    """

    # Class-level source identification
    SOURCE_NAME: str = "unknown"
    SOURCE_URL: str = ""
    DEFAULT_LICENSE: LicenseType = LicenseType.UNKNOWN

    def __init__(self, config: ScraperConfig):
        """
        Initialize scraper with configuration.

        Args:
            config: Scraper configuration options
        """
        self.config = config
        self.output_dir = Path(config.output_dir)

        # Setup directories
        self.images_dir = self.output_dir / "raw" / "images"
        self.metadata_dir = self.output_dir / "raw" / "metadata"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.user_agent,
        })

        # Rate limiting
        self._last_request_time = 0.0

        # Logging
        self.logger = logging.getLogger(f"ogham_scraper.{self.SOURCE_NAME}")
        self.logger.setLevel(config.log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            self.logger.addHandler(handler)

        # Tracking
        self._downloaded_count = 0
        self._error_count = 0

    # =========================================================================
    # ABSTRACT METHODS (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def get_stone_listing(self) -> List[StoneMetadata]:
        """
        Return list of all stones available from this source.

        Returns:
            List of StoneMetadata objects representing available stones
        """
        pass

    @abstractmethod
    def download_stone_images(self, stone_id: str) -> List[ImageDownload]:
        """
        Download all images for a specific stone.

        Args:
            stone_id: Canonical stone identifier

        Returns:
            List of ImageDownload results
        """
        pass

    @abstractmethod
    def parse_transcription(self, page_content: str) -> Optional[str]:
        """
        Extract Ogham transcription from page content.

        Args:
            page_content: HTML content of the stone's page

        Returns:
            Unicode Ogham transcription or None if not found
        """
        pass

    # =========================================================================
    # COMMON METHODS
    # =========================================================================

    def _rate_limited_get(self, url: str, **kwargs) -> requests.Response:
        """
        Make a rate-limited GET request.

        Enforces minimum delay between requests to respect server resources.

        Args:
            url: URL to fetch
            **kwargs: Additional arguments passed to requests.get()

        Returns:
            Response object
        """
        # Enforce rate limit
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit:
            sleep_time = self.config.rate_limit - elapsed
            self.logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        # Make request with retries
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(
                    url,
                    timeout=self.config.timeout,
                    **kwargs
                )
                self._last_request_time = time.time()
                response.raise_for_status()
                return response

            except requests.RequestException as e:
                last_error = e
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )
                time.sleep(self.config.rate_limit * (attempt + 1))  # Exponential backoff

        raise last_error

    def _download_image(
        self,
        url: str,
        stone_id: str,
        source_id: str,
        index: int = 0,
        transcription: Optional[str] = None,
        transcription_confidence: TranscriptionConfidence = TranscriptionConfidence.MISSING,
        image_type: ImageType = ImageType.PHOTOGRAPH,
        **extra_metadata
    ) -> ImageDownload:
        """
        Download a single image and save with metadata.

        Args:
            url: Image URL
            stone_id: Canonical stone identifier
            source_id: Source-specific identifier
            index: Image index for this stone
            transcription: Ogham transcription if known
            transcription_confidence: Confidence level of transcription
            image_type: Type of image
            **extra_metadata: Additional metadata fields

        Returns:
            ImageDownload result
        """
        try:
            # Download image
            response = self._rate_limited_get(url)

            # Determine file extension
            content_type = response.headers.get("content-type", "")
            if "jpeg" in content_type or "jpg" in content_type:
                ext = ".jpg"
            elif "png" in content_type:
                ext = ".png"
            elif "gif" in content_type:
                ext = ".gif"
            else:
                # Try to get from URL
                ext = Path(urlparse(url).path).suffix or ".jpg"

            # Generate image ID
            image_id = hashlib.md5(f"{stone_id}_{self.SOURCE_NAME}_{index}".encode()).hexdigest()[:12]

            # Create stone directory
            stone_dir = self.images_dir / stone_id
            stone_dir.mkdir(exist_ok=True)

            # Save image
            filename = f"{stone_id}_{self.SOURCE_NAME}_{index:03d}{ext}"
            image_path = stone_dir / filename

            with open(image_path, "wb") as f:
                f.write(response.content)

            # Create metadata
            metadata = ImageMetadata(
                image_id=image_id,
                stone_id=stone_id,
                source_id=source_id,
                source_name=self.SOURCE_NAME,
                source_url=url,
                download_date=datetime.utcnow().isoformat(),
                license=self.DEFAULT_LICENSE,
                transcription=transcription,
                transcription_confidence=transcription_confidence,
                image_type=image_type,
                **{k: v for k, v in extra_metadata.items() if hasattr(ImageMetadata, k)}
            )

            # Save metadata if configured
            if self.config.save_metadata:
                self._save_image_metadata(metadata)

            self._downloaded_count += 1
            self.logger.info(f"Downloaded: {filename}")

            return ImageDownload(
                success=True,
                image_path=image_path,
                metadata=metadata
            )

        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Failed to download {url}: {e}")
            return ImageDownload(
                success=False,
                error=str(e)
            )

    def _save_image_metadata(self, metadata: ImageMetadata):
        """Append image metadata to JSONL file."""
        metadata_file = self.metadata_dir / "image_metadata.jsonl"
        with open(metadata_file, "a") as f:
            f.write(json.dumps(metadata.to_dict()) + "\n")

    def _save_stone_metadata(self, metadata: StoneMetadata):
        """Append stone metadata to JSONL file."""
        metadata_file = self.metadata_dir / "stone_metadata.jsonl"
        with open(metadata_file, "a") as f:
            f.write(json.dumps(metadata.to_dict()) + "\n")

    def download_all(self, stone_ids: Optional[List[str]] = None, max_stones: Optional[int] = None) -> Dict:
        """
        Download images for all (or specified) stones.

        Args:
            stone_ids: Optional list of specific stone IDs to download
            max_stones: Optional maximum number of stones to process

        Returns:
            Dictionary with download statistics
        """
        self.logger.info(f"Starting download from {self.SOURCE_NAME}")

        # Get stone listing
        stones = self.get_stone_listing()
        self.logger.info(f"Found {len(stones)} stones")

        # Filter if specific IDs requested
        if stone_ids:
            stones = [s for s in stones if s.stone_id in stone_ids]
            self.logger.info(f"Filtered to {len(stones)} requested stones")

        # Limit if max_stones specified
        if max_stones:
            stones = stones[:max_stones]
            self.logger.info(f"Limited to {max_stones} stones")

        # Download each stone
        results = {
            "total_stones": len(stones),
            "successful_downloads": 0,
            "failed_downloads": 0,
            "stones_processed": [],
        }

        for i, stone in enumerate(stones):
            self.logger.info(f"Processing stone {i + 1}/{len(stones)}: {stone.stone_id}")

            # Save stone metadata
            self._save_stone_metadata(stone)

            # Download images
            downloads = self.download_stone_images(stone.stone_id)

            successful = sum(1 for d in downloads if d.success)
            results["successful_downloads"] += successful
            results["failed_downloads"] += len(downloads) - successful
            results["stones_processed"].append({
                "stone_id": stone.stone_id,
                "images_downloaded": successful,
                "images_failed": len(downloads) - successful,
            })

        self.logger.info(
            f"Download complete. "
            f"Total: {results['successful_downloads']} successful, "
            f"{results['failed_downloads']} failed"
        )

        return results

    def get_stats(self) -> Dict:
        """Get current scraper statistics."""
        return {
            "source": self.SOURCE_NAME,
            "downloaded_count": self._downloaded_count,
            "error_count": self._error_count,
            "output_dir": str(self.output_dir),
        }
