"""
CISP (Celtic Inscribed Stones Project) scraper.

The Celtic Inscribed Stones Project at UCL is the primary academic source
for Ogham inscription data. This scraper extracts photographs and transcriptions
from the CISP database.

Source: https://www.ucl.ac.uk/archaeology/cisp/

★ Insight ─────────────────────────────────────
CISP provides structured data including:
- Multiple transcription readings (Macalister, McManus, etc.)
- Geographic location data
- Bibliography references
- Multiple photographs per stone when available
─────────────────────────────────────────────────
"""

import re
from typing import Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ..schemas import (
    StoneMetadata,
    LicenseType,
    TranscriptionConfidence,
    ImageType,
    WeatheringSeverity,
)
from ..utils.ogham import latin_to_ogham, validate_ogham_string
from .base_scraper import OghamScraperBase, ImageDownload, ScraperConfig


class CISPScraper(OghamScraperBase):
    """
    Scraper for the Celtic Inscribed Stones Project database.

    CISP organizes stones by site, with each stone having:
    - A unique identifier (e.g., "BALIS/1" for Ballisnahaha stone 1)
    - One or more photographs
    - Transcription readings from various scholars
    - Location and bibliography information
    """

    SOURCE_NAME = "CISP"
    SOURCE_URL = "https://www.ucl.ac.uk/archaeology/cisp/"
    DEFAULT_LICENSE = LicenseType.ACADEMIC

    # Base URLs for CISP
    SITE_INDEX_URL = "https://www.ucl.ac.uk/archaeology/cisp/database/site_index.html"
    STONE_BASE_URL = "https://www.ucl.ac.uk/archaeology/cisp/database/"

    def __init__(self, config: ScraperConfig):
        """Initialize CISP scraper."""
        super().__init__(config)

        # Cache for parsed site index
        self._site_cache: Optional[Dict] = None

    def get_stone_listing(self) -> List[StoneMetadata]:
        """
        Fetch and parse the CISP site index to get all stone listings.

        Returns:
            List of StoneMetadata for all Ogham stones in CISP
        """
        self.logger.info("Fetching CISP site index...")

        try:
            response = self._rate_limited_get(self.SITE_INDEX_URL)
            soup = BeautifulSoup(response.text, "html.parser")

            stones = []

            # CISP organizes by site, then stone within site
            # Parse the alphabetical site listing
            site_links = soup.select("a[href*='/database/site/']")

            for link in site_links:
                site_url = urljoin(self.SITE_INDEX_URL, link.get("href", ""))
                site_name = link.get_text(strip=True)

                # Get stones for this site
                site_stones = self._parse_site_page(site_url, site_name)
                stones.extend(site_stones)

                self.logger.debug(f"Found {len(site_stones)} stones at {site_name}")

            self.logger.info(f"Total stones found: {len(stones)}")
            return stones

        except Exception as e:
            self.logger.error(f"Failed to fetch site index: {e}")
            return []

    def _parse_site_page(self, site_url: str, site_name: str) -> List[StoneMetadata]:
        """
        Parse a CISP site page to extract stone information.

        Args:
            site_url: URL of the site page
            site_name: Name of the site

        Returns:
            List of StoneMetadata for stones at this site
        """
        stones = []

        try:
            response = self._rate_limited_get(site_url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract region from page content
            region = self._extract_region(soup)

            # Find stone links on the page
            stone_links = soup.select("a[href*='/stone/']")

            for link in stone_links:
                stone_url = urljoin(site_url, link.get("href", ""))
                stone_ref = link.get_text(strip=True)

                # Create stone ID from site name and reference
                # e.g., "BALIS/1" from Ballisnahaha site, stone 1
                stone_id = self._normalize_stone_id(stone_ref)

                # Get detailed stone info
                stone_meta = self._parse_stone_page(stone_url, stone_id, site_name, region)
                if stone_meta:
                    stones.append(stone_meta)

        except Exception as e:
            self.logger.warning(f"Failed to parse site {site_name}: {e}")

        return stones

    def _parse_stone_page(
        self,
        stone_url: str,
        stone_id: str,
        site_name: str,
        region: str
    ) -> Optional[StoneMetadata]:
        """
        Parse a CISP stone page to extract detailed metadata.

        Args:
            stone_url: URL of the stone's page
            stone_id: Normalized stone identifier
            site_name: Name of the site where stone is located
            region: Geographic region

        Returns:
            StoneMetadata or None if parsing fails
        """
        try:
            response = self._rate_limited_get(stone_url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract transcription readings
            readings = self._extract_readings(soup)

            # Get consensus/best reading
            consensus = None
            if readings:
                # Prefer readings from McManus, then Macalister, then others
                for preferred_source in ["McManus", "Macalister", "CISP"]:
                    for reading in readings:
                        if preferred_source.lower() in reading.get("source", "").lower():
                            consensus = reading.get("text")
                            break
                    if consensus:
                        break
                if not consensus and readings:
                    consensus = readings[0].get("text")

            # Extract CIIC number if present
            ciic_number = self._extract_ciic_number(soup)

            # Extract location details
            county = self._extract_field(soup, "County")
            townland = self._extract_field(soup, "Townland")
            current_location = self._extract_field(soup, "Present location")

            # Extract physical properties
            stone_type = self._extract_field(soup, "Stone type")
            material = self._extract_field(soup, "Material")

            # Extract bibliography
            bibliography = self._extract_bibliography(soup)

            return StoneMetadata(
                stone_id=stone_id,
                ciic_number=ciic_number,
                name=site_name,
                region=region,
                county=county,
                townland=townland,
                findspot_name=site_name,
                current_location=current_location,
                stone_type=stone_type,
                material=material,
                transcription_readings=readings,
                consensus_transcription=consensus,
                bibliography=bibliography,
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse stone {stone_id}: {e}")
            return None

    def download_stone_images(self, stone_id: str) -> List[ImageDownload]:
        """
        Download all images for a specific stone from CISP.

        Args:
            stone_id: Stone identifier (e.g., "BALIS_1")

        Returns:
            List of ImageDownload results
        """
        downloads = []

        # Construct stone page URL
        stone_url = f"{self.STONE_BASE_URL}stone/{stone_id.replace('_', '/')}.html"

        try:
            response = self._rate_limited_get(stone_url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract transcription for metadata
            readings = self._extract_readings(soup)
            transcription = None
            confidence = TranscriptionConfidence.MISSING

            if readings:
                best_reading = readings[0]
                transcription = best_reading.get("text")
                conf_str = best_reading.get("confidence", "uncertain")
                confidence = TranscriptionConfidence(conf_str)

            # Find image links
            image_links = soup.select("img[src*='images/'], img[src*='photos/']")

            for i, img in enumerate(image_links):
                img_src = img.get("src", "")
                if not img_src:
                    continue

                img_url = urljoin(stone_url, img_src)

                # Determine image type from context
                img_type = ImageType.PHOTOGRAPH
                alt_text = img.get("alt", "").lower()
                if "drawing" in alt_text or "sketch" in alt_text:
                    img_type = ImageType.DRAWING
                elif "rubbing" in alt_text:
                    img_type = ImageType.RUBBING

                # Download with metadata
                download = self._download_image(
                    url=img_url,
                    stone_id=stone_id,
                    source_id=f"CISP_{stone_id}_{i}",
                    index=i,
                    transcription=transcription,
                    transcription_confidence=confidence,
                    image_type=img_type,
                )
                downloads.append(download)

        except Exception as e:
            self.logger.error(f"Failed to download images for {stone_id}: {e}")
            downloads.append(ImageDownload(success=False, error=str(e)))

        return downloads

    def parse_transcription(self, page_content: str) -> Optional[str]:
        """
        Extract Ogham transcription from CISP page content.

        CISP provides transcriptions in Latin alphabet which we convert to Unicode.

        Args:
            page_content: HTML content of the stone's page

        Returns:
            Unicode Ogham transcription or None
        """
        soup = BeautifulSoup(page_content, "html.parser")
        readings = self._extract_readings(soup)

        if readings:
            # Get the first valid reading
            for reading in readings:
                latin_text = reading.get("text", "")
                if latin_text:
                    try:
                        ogham = latin_to_ogham(latin_text)
                        if validate_ogham_string(ogham)[0]:
                            return ogham
                    except Exception:
                        continue

        return None

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _extract_readings(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract transcription readings from stone page."""
        readings = []

        # CISP presents readings in a structured format
        # Look for reading sections
        reading_sections = soup.select(".reading, .transcription, [class*='reading']")

        for section in reading_sections:
            text = section.get_text(strip=True)
            source = "Unknown"

            # Try to extract source attribution
            source_elem = section.find_previous(["h3", "h4", "strong"])
            if source_elem:
                source = source_elem.get_text(strip=True)

            # Clean up the reading text
            # Remove common prefixes/suffixes
            text = re.sub(r"^(Reading|Transcription):\s*", "", text, flags=re.IGNORECASE)
            text = text.strip()

            if text:
                readings.append({
                    "text": text,
                    "source": source,
                    "confidence": self._estimate_confidence(text),
                })

        # Also check for readings in tables
        for table in soup.select("table"):
            rows = table.select("tr")
            for row in rows:
                cells = row.select("td")
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True).lower()
                    if "reading" in label or "transcription" in label:
                        text = cells[1].get_text(strip=True)
                        if text:
                            readings.append({
                                "text": text,
                                "source": label.title(),
                                "confidence": self._estimate_confidence(text),
                            })

        return readings

    def _extract_region(self, soup: BeautifulSoup) -> str:
        """Extract geographic region from page."""
        # Check common patterns
        for pattern in ["Ireland", "Scotland", "Wales", "Isle of Man", "Cornwall"]:
            if pattern.lower() in soup.get_text().lower():
                return pattern
        return "unknown"

    def _extract_ciic_number(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract CIIC (Corpus Inscriptionum Insularum Celticarum) number."""
        text = soup.get_text()

        # Look for patterns like "CIIC 1", "CIIC no. 123", etc.
        match = re.search(r"CIIC\s*(?:no\.?\s*)?(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None

    def _extract_field(self, soup: BeautifulSoup, field_name: str) -> Optional[str]:
        """Extract a named field from the page."""
        # Look for label:value patterns
        pattern = rf"{field_name}\s*[:\-]\s*([^<\n]+)"
        match = re.search(pattern, soup.get_text(), re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Also check table cells
        for td in soup.select("td"):
            if field_name.lower() in td.get_text().lower():
                next_td = td.find_next_sibling("td")
                if next_td:
                    return next_td.get_text(strip=True)

        return None

    def _extract_bibliography(self, soup: BeautifulSoup) -> List[str]:
        """Extract bibliography references."""
        bibliography = []

        # Look for bibliography section
        bib_section = soup.find(["h2", "h3", "h4"], string=re.compile(r"bibliography", re.I))
        if bib_section:
            # Get following list or paragraphs
            for elem in bib_section.find_next_siblings(["ul", "ol", "p"]):
                if elem.name in ["ul", "ol"]:
                    for li in elem.select("li"):
                        bibliography.append(li.get_text(strip=True))
                else:
                    text = elem.get_text(strip=True)
                    if text:
                        bibliography.append(text)
                # Stop at next section
                if elem.name in ["h2", "h3", "h4"]:
                    break

        return bibliography

    def _estimate_confidence(self, reading: str) -> str:
        """Estimate confidence level based on reading characteristics."""
        # Check for uncertainty markers
        if "?" in reading or "[" in reading or "uncertain" in reading.lower():
            return "uncertain"
        if "probable" in reading.lower():
            return "probable"
        return "probable"  # Default to probable for CISP readings

    def _normalize_stone_id(self, ref: str) -> str:
        """Normalize stone reference to consistent ID format."""
        # Replace slashes and spaces with underscores
        normalized = re.sub(r"[/\s]+", "_", ref)
        # Remove any special characters except underscores
        normalized = re.sub(r"[^A-Za-z0-9_]", "", normalized)
        return normalized.upper()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_cisp_scraper(output_dir: str, **kwargs) -> CISPScraper:
    """
    Create a CISP scraper with default configuration.

    Args:
        output_dir: Directory to save downloaded images
        **kwargs: Additional configuration options

    Returns:
        Configured CISPScraper instance
    """
    config = ScraperConfig(
        output_dir=output_dir,
        rate_limit=kwargs.get("rate_limit", 2.0),
        user_agent=kwargs.get("user_agent", "OghamOCR-Research/1.0"),
    )
    return CISPScraper(config)
