"""
Wikimedia Commons scraper for Ogham images.

Wikimedia Commons provides CC-licensed photographs of Ogham stones contributed
by various photographers. This scraper uses the MediaWiki API to find and
download relevant images.

Source: https://commons.wikimedia.org/

★ Insight ─────────────────────────────────────
Key advantages of Wikimedia:
1. Clear CC licenses (CC-BY, CC-BY-SA, CC0)
2. High-resolution images available
3. Well-structured API for programmatic access
4. Multiple images per stone from different contributors
─────────────────────────────────────────────────
"""

import re
from typing import Dict, List, Optional
from urllib.parse import quote, urlencode

from ..schemas import (
    StoneMetadata,
    LicenseType,
    TranscriptionConfidence,
    ImageType,
)
from .base_scraper import OghamScraperBase, ImageDownload, ScraperConfig


class WikimediaScraper(OghamScraperBase):
    """
    Scraper for Wikimedia Commons Ogham images.

    Uses the MediaWiki API to:
    - Search for Ogham-related categories and images
    - Extract license information
    - Download full-resolution images
    """

    SOURCE_NAME = "Wikimedia"
    SOURCE_URL = "https://commons.wikimedia.org/"
    DEFAULT_LICENSE = LicenseType.CC_BY_SA

    # API endpoint
    API_URL = "https://commons.wikimedia.org/w/api.php"

    # Categories to search
    OGHAM_CATEGORIES = [
        "Category:Ogham inscriptions",
        "Category:Ogham stones",
        "Category:Ogham inscriptions in Ireland",
        "Category:Ogham inscriptions in Scotland",
        "Category:Ogham inscriptions in Wales",
    ]

    # License mapping
    LICENSE_MAP = {
        "cc-zero": LicenseType.CC0,
        "cc0": LicenseType.CC0,
        "pd": LicenseType.CC0,
        "public domain": LicenseType.CC0,
        "cc-by-4.0": LicenseType.CC_BY,
        "cc-by-3.0": LicenseType.CC_BY,
        "cc-by-2.5": LicenseType.CC_BY,
        "cc-by-sa-4.0": LicenseType.CC_BY_SA,
        "cc-by-sa-3.0": LicenseType.CC_BY_SA,
        "cc-by-sa-2.5": LicenseType.CC_BY_SA,
    }

    def __init__(self, config: ScraperConfig):
        """Initialize Wikimedia scraper."""
        super().__init__(config)

        # Cache for category members
        self._category_cache: Dict[str, List] = {}

    def get_stone_listing(self) -> List[StoneMetadata]:
        """
        Get list of Ogham stones from Wikimedia categories.

        Note: Wikimedia doesn't have structured stone data like CISP,
        so we extract what we can from image titles and descriptions.

        Returns:
            List of StoneMetadata
        """
        self.logger.info("Fetching Ogham images from Wikimedia Commons...")

        stones_map = {}  # stone_id -> StoneMetadata

        for category in self.OGHAM_CATEGORIES:
            images = self._get_category_images(category)

            for img_info in images:
                # Try to extract stone identifier from image title
                stone_id = self._extract_stone_id(img_info.get("title", ""))

                if stone_id and stone_id not in stones_map:
                    # Create basic metadata
                    stones_map[stone_id] = StoneMetadata(
                        stone_id=stone_id,
                        name=stone_id.replace("_", " "),
                        region=self._extract_region_from_category(category),
                    )

        stones = list(stones_map.values())
        self.logger.info(f"Found {len(stones)} unique stones")
        return stones

    def _get_category_images(self, category: str) -> List[Dict]:
        """
        Get all images in a Wikimedia category.

        Args:
            category: Category name (e.g., "Category:Ogham inscriptions")

        Returns:
            List of image info dictionaries
        """
        if category in self._category_cache:
            return self._category_cache[category]

        images = []
        continue_token = None

        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category,
                "cmtype": "file",
                "cmlimit": "500",
                "format": "json",
            }

            if continue_token:
                params["cmcontinue"] = continue_token

            try:
                response = self._rate_limited_get(self.API_URL, params=params)
                data = response.json()

                members = data.get("query", {}).get("categorymembers", [])
                images.extend(members)

                # Check for continuation
                if "continue" in data:
                    continue_token = data["continue"].get("cmcontinue")
                else:
                    break

            except Exception as e:
                self.logger.warning(f"Failed to fetch category {category}: {e}")
                break

        self._category_cache[category] = images
        self.logger.debug(f"Found {len(images)} images in {category}")
        return images

    def download_stone_images(self, stone_id: str) -> List[ImageDownload]:
        """
        Download all Wikimedia images for a specific stone.

        Args:
            stone_id: Stone identifier

        Returns:
            List of ImageDownload results
        """
        downloads = []

        # Search for images related to this stone
        search_results = self._search_images(stone_id)

        for i, result in enumerate(search_results):
            page_id = result.get("pageid")
            if not page_id:
                continue

            # Get image info including URL and license
            img_info = self._get_image_info(page_id)
            if not img_info:
                continue

            img_url = img_info.get("url")
            if not img_url:
                continue

            # Determine license
            license_type = self._parse_license(img_info.get("extmetadata", {}))

            # Download
            download = self._download_image(
                url=img_url,
                stone_id=stone_id,
                source_id=f"wikimedia_{page_id}",
                index=i,
                image_type=ImageType.PHOTOGRAPH,
                license_url=f"https://commons.wikimedia.org/wiki/File:{quote(result.get('title', '').replace('File:', ''))}",
            )

            # Update license in metadata if download succeeded
            if download.success and download.metadata:
                download.metadata.license = license_type

            downloads.append(download)

        return downloads

    def _search_images(self, query: str) -> List[Dict]:
        """
        Search for images matching a query.

        Args:
            query: Search query (stone name/ID)

        Returns:
            List of search results
        """
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f"Ogham {query}",
            "srnamespace": "6",  # File namespace
            "srlimit": "50",
            "format": "json",
        }

        try:
            response = self._rate_limited_get(self.API_URL, params=params)
            data = response.json()
            return data.get("query", {}).get("search", [])
        except Exception as e:
            self.logger.warning(f"Search failed for {query}: {e}")
            return []

    def _get_image_info(self, page_id: int) -> Optional[Dict]:
        """
        Get detailed information about an image.

        Args:
            page_id: Wikimedia page ID

        Returns:
            Image info dictionary or None
        """
        params = {
            "action": "query",
            "pageids": str(page_id),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": "2048",  # Get reasonably sized version
            "format": "json",
        }

        try:
            response = self._rate_limited_get(self.API_URL, params=params)
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            page = pages.get(str(page_id), {})
            imageinfo = page.get("imageinfo", [])

            if imageinfo:
                return imageinfo[0]

        except Exception as e:
            self.logger.warning(f"Failed to get image info for page {page_id}: {e}")

        return None

    def _parse_license(self, extmetadata: Dict) -> LicenseType:
        """
        Parse license from Wikimedia extended metadata.

        Args:
            extmetadata: Extended metadata dictionary

        Returns:
            License type
        """
        license_short = extmetadata.get("LicenseShortName", {}).get("value", "").lower()
        license_url = extmetadata.get("LicenseUrl", {}).get("value", "").lower()

        # Check against known licenses
        for key, license_type in self.LICENSE_MAP.items():
            if key in license_short or key in license_url:
                return license_type

        # Default to CC-BY-SA (most common on Wikimedia)
        return LicenseType.CC_BY_SA

    def _extract_stone_id(self, title: str) -> Optional[str]:
        """
        Extract stone identifier from image title.

        Args:
            title: Image title (e.g., "File:Ogham Stone Ballycrovane.jpg")

        Returns:
            Normalized stone ID or None
        """
        # Remove "File:" prefix
        title = re.sub(r"^File:", "", title, flags=re.IGNORECASE)

        # Remove file extension
        title = re.sub(r"\.(jpg|jpeg|png|gif|tiff?)$", "", title, flags=re.IGNORECASE)

        # Extract potential stone name
        # Common patterns: "Ogham Stone X", "X Ogham", "Ogham inscription at X"
        patterns = [
            r"Ogham\s+(?:Stone\s+)?(?:at\s+)?(.+)",
            r"(.+?)\s+Ogham",
            r"CIIC\s*(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Normalize
                return re.sub(r"[^A-Za-z0-9]+", "_", name).upper()

        return None

    def _extract_region_from_category(self, category: str) -> str:
        """Extract region from category name."""
        if "Ireland" in category:
            return "Ireland"
        elif "Scotland" in category:
            return "Scotland"
        elif "Wales" in category:
            return "Wales"
        return "unknown"

    def parse_transcription(self, page_content: str) -> Optional[str]:
        """
        Wikimedia doesn't typically have structured transcriptions.

        Returns:
            None (transcriptions must come from other sources)
        """
        return None


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_wikimedia_scraper(output_dir: str, **kwargs) -> WikimediaScraper:
    """
    Create a Wikimedia scraper with default configuration.

    Args:
        output_dir: Directory to save downloaded images
        **kwargs: Additional configuration options

    Returns:
        Configured WikimediaScraper instance
    """
    config = ScraperConfig(
        output_dir=output_dir,
        rate_limit=kwargs.get("rate_limit", 1.0),  # Wikimedia allows faster
        user_agent=kwargs.get("user_agent", "OghamOCR-Research/1.0"),
    )
    return WikimediaScraper(config)
