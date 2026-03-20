"""
DIAS (Dublin Institute for Advanced Studies) "Ogham in 3D" scraper.

The Ogham in 3D project (ogham.celt.dias.ie) provides the most comprehensive
digital documentation of Ogham stones, including photographs, 3D models,
diplomatic and interpretive readings, translations, and detailed provenance.

Source: https://ogham.celt.dias.ie/
License: CC-BY-NC-SA 3.0 Ireland
"""

import re
from typing import Dict, List, Optional, Set, Tuple
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


class DIASScraper(OghamScraperBase):
    """
    Scraper for the DIAS 'Ogham in 3D' database.

    DIAS organizes stones by a project ID system:
    - Format: {country_code}-{county_code}-{number}
    - Example: I-COR-001 (Ireland, Cork, stone 001)
    - Special prefixes: L (lost), X (dubious)

    Each stone page provides:
    - Provenance (discovery, findspot, coordinates)
    - Support (object type, material, dimensions)
    - Inscription (diplomatic + interpretive readings)
    - Translation and commentary
    - Bibliography (with Zotero links)
    - Photographs and 3D model viewer
    """

    SOURCE_NAME = "DIAS"
    SOURCE_URL = "https://ogham.celt.dias.ie/"
    DEFAULT_LICENSE = LicenseType.CC_BY_NC_SA

    BASE_URL = "https://ogham.celt.dias.ie/"
    OLD_SITE_OVERVIEW_URL = (
        "https://ogham.celt.dias.ie/version2013/menu.php?lang=en&menuitem=81"
    )

    # Known Irish county codes used in DIAS project IDs
    COUNTY_CODES = [
        "COR", "KER", "WAT", "KIL", "MAY", "WIC", "CLA", "GAL",
        "TIP", "FER", "DER", "ANT", "DOW", "CAR", "WEX", "LIM",
        "OFF", "LOU", "MEA", "WES",
    ]

    MAX_PER_COUNTY = 50

    def __init__(
        self,
        config: ScraperConfig,
        use_old_site: bool = False,
        enumerate_counties: bool = False,
    ):
        """
        Initialize DIAS scraper.

        Args:
            config: Scraper configuration
            use_old_site: Also cross-reference old site overview for completeness
            enumerate_counties: Also probe county code sequences for missed stones
        """
        super().__init__(config)

        self.use_old_site = use_old_site
        self.enumerate_counties = enumerate_counties

        self._discovered_ids: Set[str] = set()
        self._stone_page_cache: Dict[str, BeautifulSoup] = {}

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================

    def get_stone_listing(self) -> List[StoneMetadata]:
        """
        Discover and parse all available stones from the DIAS site.

        Uses next/prev navigation link crawling as the primary enumeration
        strategy. Optional tiers (old site cross-reference, county code
        enumeration) can be enabled via constructor flags.

        Returns:
            List of StoneMetadata for all discovered stones
        """
        self.logger.info("Discovering DIAS stone listings...")

        # Tier 1: Crawl via next/prev navigation links (always runs)
        self._crawl_navigation_links(seed_id="I-COR-001")
        self.logger.info(
            f"Navigation crawl discovered {len(self._discovered_ids)} stones"
        )

        # Tier 2: Cross-reference with old site overview (opt-in)
        if self.use_old_site:
            old_site_ids = self._scrape_old_site_overview()
            new_from_old = old_site_ids - self._discovered_ids
            if new_from_old:
                self.logger.info(
                    f"Old site overview found {len(new_from_old)} additional IDs"
                )
                self._discovered_ids.update(new_from_old)

        # Tier 3: County code enumeration (opt-in)
        if self.enumerate_counties:
            probed_ids = self._enumerate_county_codes()
            new_from_probe = probed_ids - self._discovered_ids
            if new_from_probe:
                self.logger.info(
                    f"County enumeration found {len(new_from_probe)} additional IDs"
                )
                self._discovered_ids.update(new_from_probe)

        self.logger.info(f"Total unique stones discovered: {len(self._discovered_ids)}")

        # Parse each stone page for full metadata
        stones = []
        for stone_id in sorted(self._discovered_ids):
            stone_meta = self._parse_stone_page(stone_id)
            if stone_meta:
                stones.append(stone_meta)

        self.logger.info(f"Successfully parsed {len(stones)} stone records")
        return stones

    def download_stone_images(self, stone_id: str) -> List[ImageDownload]:
        """
        Download all photographs for a specific stone from DIAS.

        Image URLs follow the pattern:
        /images/IRE/I-COR/I-COR-001-Coomleagh-East-a.jpg

        Args:
            stone_id: Project ID (e.g., "I-COR-001")

        Returns:
            List of ImageDownload results
        """
        downloads = []

        try:
            soup = self._get_stone_soup(stone_id)

            # Extract transcription for image metadata
            readings = self._extract_readings(soup)
            consensus = self._determine_consensus(readings)
            confidence = TranscriptionConfidence.VERIFIED if consensus else TranscriptionConfidence.MISSING
            translation = self._extract_translation(soup)
            coordinates = self._extract_coordinates(soup)
            findspot = self._extract_findspot(soup)

            # Find stone photographs
            image_index = 0
            for img in soup.find_all("img"):
                src = img.get("src", "")
                if not self._is_stone_image(src):
                    continue

                img_url = urljoin(self.BASE_URL, src)
                alt_text = img.get("alt", "")

                download = self._download_image(
                    url=img_url,
                    stone_id=stone_id,
                    source_id=f"DIAS_{stone_id}_{image_index}",
                    index=image_index,
                    transcription=consensus,
                    transcription_confidence=confidence,
                    image_type=self._classify_image_type(src, alt_text),
                    transliteration=self._get_diplomatic_reading(readings),
                    view_angle=self._extract_view_angle(src),
                    findspot_name=findspot,
                    coordinates=coordinates,
                    license_url="https://creativecommons.org/licenses/by-nc-sa/3.0/ie/",
                )
                downloads.append(download)
                image_index += 1

        except Exception as e:
            self.logger.error(f"Failed to download images for {stone_id}: {e}")
            downloads.append(ImageDownload(success=False, error=str(e)))

        return downloads

    def parse_transcription(self, page_content: str) -> Optional[str]:
        """
        Extract Ogham transcription from DIAS page content.

        Prefers the interpretive reading (Ogham Unicode) if it validates.
        Falls back to converting the diplomatic (Latin) reading.

        Args:
            page_content: HTML content of the stone's page

        Returns:
            Unicode Ogham transcription or None
        """
        soup = BeautifulSoup(page_content, "html.parser")
        readings = self._extract_readings(soup)

        # Try interpretive reading first (already Ogham Unicode)
        for reading in readings:
            if reading.get("type") == "interpretive":
                # Strip editorial combining marks (dots below, etc.)
                cleaned = re.sub(r"[\u0323\u0331.?\u2026\s]+", "", reading["text"])
                if cleaned and validate_ogham_string(cleaned)[0]:
                    return cleaned

        # Fall back to converting diplomatic reading
        for reading in readings:
            if reading.get("type") == "diplomatic":
                try:
                    ogham = latin_to_ogham(reading["text"])
                    if validate_ogham_string(ogham)[0]:
                        return ogham
                except Exception:
                    continue

        return None

    # =========================================================================
    # ENUMERATION: TIER 1 — NAVIGATION LINK CRAWLING
    # =========================================================================

    def _crawl_navigation_links(self, seed_id: str = "I-COR-001"):
        """
        Discover stones by following next/prev links on stone pages.

        Each stone page has Previous/Next navigation links forming a chain
        through all published stones.

        Args:
            seed_id: A known stone ID to start crawling from
        """
        to_visit = {seed_id}
        visited: Set[str] = set()

        while to_visit:
            current_id = to_visit.pop()
            if current_id in visited:
                continue
            visited.add(current_id)

            stone_url = urljoin(self.BASE_URL, current_id)
            try:
                response = self._rate_limited_get(stone_url)
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.text, "html.parser")

                if not self._is_stone_page(response.text):
                    continue

                self._discovered_ids.add(current_id)
                self._stone_page_cache[current_id] = soup

                if len(self._discovered_ids) % 10 == 0 or len(self._discovered_ids) <= 3:
                    self.logger.info(
                        f"Crawl progress: {len(self._discovered_ids)} stones "
                        f"(visited {len(visited)}, queue {len(to_visit)})"
                    )

                # Extract next/prev links
                for link in soup.find_all("a"):
                    href = link.get("href", "").strip().strip("/")
                    link_text = link.get_text(strip=True).lower()

                    if link_text in ("previous", "next", "prev", "< previous", "next >"):
                        nav_id = href.split("/")[-1]
                        if self._is_valid_project_id(nav_id) and nav_id not in visited:
                            to_visit.add(nav_id)

                    # Also follow any link that looks like a project ID
                    elif self._is_valid_project_id(href) and href not in visited:
                        to_visit.add(href)

            except Exception as e:
                self.logger.debug(f"Failed to fetch {current_id}: {e}")

    # =========================================================================
    # ENUMERATION: TIER 2 — OLD SITE CROSS-REFERENCE
    # =========================================================================

    def _scrape_old_site_overview(self) -> Set[str]:
        """
        Parse the old site overview page to discover stone IDs.

        The old site at version2013/ lists stones with CIIC numbers and
        site names. We attempt to resolve these to new-format project IDs.

        Returns:
            Set of project IDs discovered from the old site
        """
        discovered: Set[str] = set()

        try:
            response = self._rate_limited_get(self.OLD_SITE_OVERVIEW_URL)
            soup = BeautifulSoup(response.text, "html.parser")

            stone_links = soup.find_all("a", href=re.compile(r"stone\.php.*stone="))

            for link in stone_links:
                href = link.get("href", "")
                link_text = link.get_text(strip=True)

                ciic_match = re.match(r"(\d+[a-z]?)\.\s*(.+)", link_text)
                if not ciic_match:
                    continue

                old_stone_url = urljoin(self.OLD_SITE_OVERVIEW_URL, href)
                new_id = self._resolve_old_to_new_id(old_stone_url)
                if new_id:
                    discovered.add(new_id)

        except Exception as e:
            self.logger.warning(f"Failed to parse old site overview: {e}")

        return discovered

    def _resolve_old_to_new_id(self, old_stone_url: str) -> Optional[str]:
        """
        Resolve an old-format stone URL to a new project ID.

        Searches the old stone page for references to the new-format ID.

        Args:
            old_stone_url: URL from the old site

        Returns:
            New-format project ID or None
        """
        try:
            response = self._rate_limited_get(old_stone_url)
            text = response.text

            # Look for new-format ID in page content
            id_match = re.search(
                r"ogham\.celt\.dias\.ie/(I-[A-Z]{2,4}-[A-Z0-9]{2,4})", text
            )
            if id_match:
                return id_match.group(1)

            # Standalone ID pattern
            id_match = re.search(r"\b(I-[A-Z]{2,4}-[A-Z]?\d{2,3})\b", text)
            if id_match:
                return id_match.group(1)

        except Exception:
            pass

        return None

    # =========================================================================
    # ENUMERATION: TIER 3 — COUNTY CODE ENUMERATION
    # =========================================================================

    def _enumerate_county_codes(self) -> Set[str]:
        """
        Probe sequential IDs for each known county code.

        Stops probing a county after 10 consecutive misses.

        Returns:
            Set of valid project IDs discovered
        """
        discovered: Set[str] = set()

        for county in self.COUNTY_CODES:
            consecutive_misses = 0

            for num in range(1, self.MAX_PER_COUNTY + 1):
                stone_id = f"I-{county}-{num:03d}"

                if stone_id in self._discovered_ids:
                    consecutive_misses = 0
                    continue

                stone_url = urljoin(self.BASE_URL, stone_id)
                try:
                    response = self._rate_limited_get(stone_url)
                    if response.status_code == 200 and self._is_stone_page(response.text):
                        discovered.add(stone_id)
                        consecutive_misses = 0
                        self.logger.debug(f"Enumeration found: {stone_id}")
                    else:
                        consecutive_misses += 1
                except Exception:
                    consecutive_misses += 1

                if consecutive_misses >= 10:
                    break

        return discovered

    # =========================================================================
    # STONE PAGE PARSING
    # =========================================================================

    def _parse_stone_page(self, stone_id: str) -> Optional[StoneMetadata]:
        """
        Parse a DIAS stone page into StoneMetadata.

        Args:
            stone_id: Project ID (e.g., "I-COR-001")

        Returns:
            StoneMetadata or None if parsing fails
        """
        try:
            soup = self._get_stone_soup(stone_id)

            readings = self._extract_readings(soup)
            consensus = self._determine_consensus(readings)
            preservation = self._extract_condition(soup)

            return StoneMetadata(
                stone_id=stone_id,
                ciic_number=self._extract_ciic_number(soup),
                name=self._extract_name(soup),
                region="Ireland",
                county=self._extract_county(soup),
                townland=self._extract_townland(soup),
                findspot_name=self._extract_findspot(soup),
                current_location=self._extract_current_location(soup),
                coordinates=self._extract_coordinates(soup),
                stone_type=self._extract_object_type(soup),
                material=self._extract_material(soup),
                height_cm=self._extract_height_cm(soup),
                transcription_readings=readings,
                consensus_transcription=consensus,
                inscription_language="Primitive Irish",
                estimated_date_range=self._extract_date_range(soup),
                preservation_state=preservation,
                weathering_severity=self._estimate_weathering(preservation),
                bibliography=self._extract_bibliography(soup),
            )

        except Exception as e:
            self.logger.warning(f"Failed to parse stone {stone_id}: {e}")
            return None

    # =========================================================================
    # DATA EXTRACTION HELPERS
    # =========================================================================

    def _get_stone_soup(self, stone_id: str) -> BeautifulSoup:
        """Get parsed HTML for a stone page, using cache if available."""
        if stone_id in self._stone_page_cache:
            return self._stone_page_cache[stone_id]

        response = self._rate_limited_get(urljoin(self.BASE_URL, stone_id))
        soup = BeautifulSoup(response.text, "html.parser")
        self._stone_page_cache[stone_id] = soup
        return soup

    def _extract_name(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract stone name from the page title/heading.

        Format observed: "Irish Name | English Name"
        """
        h1 = soup.find("h1")
        if not h1:
            return None
        text = h1.get_text(strip=True)
        if "|" in text:
            after_pipe = text.split("|", 1)[1]
            return re.sub(r"\s*\([^)]+\)\s*$", "", after_pipe).strip()
        return re.sub(r"\s*\([^)]+\)\s*$", "", text).strip()

    def _extract_ciic_number(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract CIIC number from headings or text."""
        text = soup.get_text()
        match = re.search(r"CIIC\s+(\d+)", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _extract_county(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract county from 'Co. Cork' patterns in the text."""
        text = soup.get_text()
        # Match "Co. Cork," or "Co. Cork " — stop at comma, period, or lowercase word
        match = re.search(r"Co\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_townland(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract townland name from the Findspot section."""
        text = soup.get_text()
        match = re.search(
            r"Findspot[:\s]*([^(,\n]+?)(?:\s*\([^)]+\))?,\s*Co\.", text
        )
        if match:
            return match.group(1).strip()
        return None

    def _extract_coordinates(self, soup: BeautifulSoup) -> Optional[Tuple[float, float]]:
        """
        Extract WGS84 coordinates.

        DIAS pages embed coordinates in elements with id="lat"/"long"
        or as inline text like "52.110277, -10.473175".
        """
        # Try element IDs first
        lat_elem = soup.find(id="lat")
        lon_elem = soup.find(id="long")
        if lat_elem and lon_elem:
            try:
                return (
                    float(lat_elem.get_text(strip=True)),
                    float(lon_elem.get_text(strip=True)),
                )
            except ValueError:
                pass

        # Try data attributes
        for elem in soup.find_all(attrs={"data-lat": True, "data-lng": True}):
            try:
                return (float(elem["data-lat"]), float(elem["data-lng"]))
            except (ValueError, KeyError):
                pass

        # Fallback: search for WGS84 coordinate pattern in text
        text = soup.get_text()
        match = re.search(r"WGS84[:\s]*(-?\d+\.\d+)[,\s]+(-?\d+\.\d+)", text)
        if match:
            try:
                return (float(match.group(1)), float(match.group(2)))
            except ValueError:
                pass

        return None

    def _extract_findspot(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract findspot name."""
        return self._extract_field_value(soup, "Findspot")

    def _extract_current_location(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract current location / last recorded location."""
        return (
            self._extract_field_value(soup, "Last recorded location")
            or self._extract_field_value(soup, "Current repository")
            or self._extract_field_value(soup, "Present location")
            or self._extract_field_value(soup, "Current location")
        )

    def _extract_object_type(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract object type (e.g., 'Pillar', 'Cross-slab')."""
        return self._extract_field_value(soup, "Object type")

    def _extract_material(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract material type (e.g., 'Grit', 'Sandstone')."""
        return self._extract_field_value(soup, "Material")

    def _extract_height_cm(self, soup: BeautifulSoup) -> Optional[float]:
        """
        Extract height from dimensions and convert to cm.

        DIAS format: "H 2.00 x W 1.10 x D 0.60 m" (meters).
        """
        text = soup.get_text()
        match = re.search(
            r"H\s+([\d.]+)\s*[x\u00d7]\s*W\s+([\d.]+)\s*[x\u00d7]\s*D\s+([\d.]+)\s*m",
            text,
            re.IGNORECASE,
        )
        if match:
            try:
                return float(match.group(1)) * 100
            except ValueError:
                pass
        return None

    def _extract_readings(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract inscription readings from the page.

        DIAS uses Bootstrap tabs for readings:
        - #nav-home: Interpretive tab (Ogham + Latin transliteration)
        - #nav-profile: Diplomatic tab (raw stroke reading)
        - #apparatus: Critical apparatus (Macalister etc.)
        """
        readings = []

        # Interpretive reading from the first tab pane
        interp_pane = soup.find("div", id="nav-home")
        if interp_pane:
            edition_divs = interp_pane.find_all("div", id="editionF")
            if edition_divs:
                # First editionF div = Ogham Unicode reading
                ogham_text = edition_divs[0].get_text(strip=True)
                if ogham_text and len(ogham_text) > 1:
                    readings.append({
                        "text": ogham_text,
                        "source": "DIAS O3D (Interpretive)",
                        "confidence": "verified",
                        "type": "interpretive",
                    })
                # Second editionF div = Latin transliteration
                if len(edition_divs) > 1:
                    latin_text = edition_divs[1].get_text(strip=True)
                    if latin_text and len(latin_text) > 1:
                        readings.append({
                            "text": latin_text,
                            "source": "DIAS O3D (Interpretive Latin)",
                            "confidence": "verified",
                            "type": "interpretive_latin",
                        })

        # Diplomatic reading from the second tab pane
        diplo_pane = soup.find("div", id="nav-profile")
        if diplo_pane:
            edition_divs = diplo_pane.find_all("div", id="editionF")
            if edition_divs:
                diplo_text = edition_divs[0].get_text(strip=True)
                if diplo_text and len(diplo_text) > 1:
                    readings.append({
                        "text": diplo_text,
                        "source": "DIAS O3D (Diplomatic)",
                        "confidence": "verified",
                        "type": "diplomatic",
                    })

        # Critical apparatus (Macalister's reading etc.)
        apparatus = soup.find("div", id="apparatus")
        if apparatus:
            apparatus_text = apparatus.get_text()
            mac_match = re.search(
                r"Macalister\s*\([^)]+\)\s*read[^:]*:\s*(.+)",
                apparatus_text,
            )
            if mac_match:
                mac_text = mac_match.group(1).strip()
                if mac_text and len(mac_text) > 1:
                    readings.append({
                        "text": mac_text,
                        "source": "Macalister (1945)",
                        "confidence": "probable",
                        "type": "historic",
                    })

        return readings

    def _determine_consensus(self, readings: List[Dict]) -> Optional[str]:
        """
        Get the best consensus reading.

        Priority: interpretive > diplomatic > historic.
        """
        for reading_type in ("interpretive", "diplomatic", "historic"):
            for reading in readings:
                if reading.get("type") == reading_type:
                    return reading["text"]
        return None

    def _extract_translation(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract English translation from the TRANSLATION section."""
        trans_heading = soup.find(
            "h3", string=re.compile(r"TRANSLATION", re.IGNORECASE)
        )
        if trans_heading:
            parts = []
            for sib in trans_heading.find_next_siblings():
                if sib.name in ("h1", "h2", "h3"):
                    break
                text = sib.get_text(strip=True)
                if text:
                    parts.append(text)
            if parts:
                return " ".join(parts).strip("\"'\u2018\u2019\u201c\u201d ")
        return None

    def _extract_condition(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract condition/preservation description."""
        return self._extract_field_value(soup, "Condition")

    def _estimate_weathering(
        self, condition_text: Optional[str]
    ) -> Optional[WeatheringSeverity]:
        """Map condition description to WeatheringSeverity."""
        if not condition_text:
            return None
        text_lower = condition_text.lower()
        if any(w in text_lower for w in ("badly damaged", "severe", "illegible", "destroyed")):
            return WeatheringSeverity.SEVERE
        if any(w in text_lower for w in ("weathered", "damaged", "worn", "eroded", "flaking")):
            return WeatheringSeverity.MODERATE
        if any(w in text_lower for w in ("good", "well preserved", "clear", "intact")):
            return WeatheringSeverity.MINIMAL
        return WeatheringSeverity.MODERATE

    def _extract_date_range(self, soup: BeautifulSoup) -> Optional[Tuple[int, int]]:
        """
        Extract estimated date range from the Date field or commentary.

        Checks the <b>Date:</b> field first, then falls back to
        searching commentary text for century references.
        """
        # Try the explicit Date field first (e.g., "Mid sixth century (linguistic)")
        date_text = self._extract_field_value(soup, "Date")
        if date_text:
            result = self._parse_century_text(date_text)
            if result:
                return result

        # Fall back to searching full page text
        text = soup.get_text()
        return self._parse_century_text(text)

    def _parse_century_text(self, text: str) -> Optional[Tuple[int, int]]:
        """Parse century references from text into a date range."""
        # "first half of the sixth century" -> (500, 550)
        half_match = re.search(
            r"(first|second|early|late|mid)\s+(?:half\s+(?:of\s+)?)?(?:the\s+)?(\w+)\s+centur",
            text,
            re.IGNORECASE,
        )
        if half_match:
            century = self._word_to_century(half_match.group(2))
            if century:
                qualifier = half_match.group(1).lower()
                if qualifier in ("first", "early"):
                    return (century - 100, century - 50)
                elif qualifier == "mid":
                    return (century - 75, century - 25)
                else:
                    return (century - 50, century)

        # "5th-6th century" or "fifth to sixth century"
        range_match = re.search(
            r"(\d+)(?:st|nd|rd|th)[\s\-]+(?:to\s+)?(\d+)(?:st|nd|rd|th)?\s*centur",
            text,
            re.IGNORECASE,
        )
        if range_match:
            return (
                int(range_match.group(1)) * 100 - 100,
                int(range_match.group(2)) * 100,
            )

        # "sixth century" (single century)
        single_match = re.search(
            r"(\w+)\s+centur", text, re.IGNORECASE
        )
        if single_match:
            century = self._word_to_century(single_match.group(1))
            if century:
                return (century - 100, century)

        return None

    def _extract_bibliography(self, soup: BeautifulSoup) -> List[str]:
        """Extract bibliography entries from the REFERENCES section."""
        bibliography = []

        ref_heading = soup.find(
            ["h2", "h3", "h4"], string=re.compile(r"REFERENCES?|BIBLIOGRAPHY", re.IGNORECASE)
        )
        if ref_heading:
            for elem in ref_heading.find_next_siblings():
                if elem.name in ("h1", "h2"):
                    break
                text = elem.get_text(strip=True)
                if text and len(text) > 5:
                    bibliography.append(text)

        return bibliography

    def _extract_field_value(self, soup: BeautifulSoup, field_name: str) -> Optional[str]:
        """
        Extract a labelled field value from the page.

        DIAS uses <b>Field:</b> Value pattern for most fields.
        Also checks <strong> and raw text as fallbacks.
        """
        # <b>Field:</b> value pattern (primary DIAS format)
        for tag in soup.find_all(["b", "strong"]):
            tag_text = tag.get_text(strip=True)
            if field_name.lower() in tag_text.lower():
                next_sib = tag.next_sibling
                if next_sib:
                    value = str(next_sib).strip().lstrip(":").strip()
                    if value:
                        return value
                # Try parent text minus the label
                parent = tag.parent
                if parent:
                    full_text = parent.get_text(strip=True)
                    match = re.search(
                        rf"{re.escape(field_name)}[^:]*[:\s]+(.+)",
                        full_text,
                        re.IGNORECASE,
                    )
                    if match:
                        return match.group(1).strip()

        return None

    # =========================================================================
    # IMAGE HELPERS
    # =========================================================================

    def _is_stone_image(self, src: str) -> bool:
        """Check if an image src is a stone photograph (not a UI element)."""
        if not src:
            return False
        src_lower = src.lower()
        if "/images/ire/" in src_lower:
            return True
        if re.search(r"I-[A-Z]{2,4}-\d{2,3}", src):
            return True
        return False

    def _classify_image_type(self, src: str, alt_text: str) -> ImageType:
        """Classify image type from filename and alt text."""
        combined = (src + " " + alt_text).lower()
        if "drawing" in combined or "sketch" in combined:
            return ImageType.DRAWING
        if "rubbing" in combined:
            return ImageType.RUBBING
        if "3d" in combined or "render" in combined:
            return ImageType.RENDER_3D
        return ImageType.PHOTOGRAPH

    def _extract_view_angle(self, src: str) -> Optional[str]:
        """
        Infer view angle from DIAS image filename convention.

        DIAS filenames end with -a.jpg, -b.jpg etc. for different views.
        """
        filename = src.split("/")[-1].lower()
        if "-a." in filename:
            return "front"
        if "-b." in filename:
            return "back"
        if "-c." in filename:
            return "side"
        if "-d." in filename:
            return "detail"
        return None

    def _get_diplomatic_reading(self, readings: List[Dict]) -> Optional[str]:
        """Get the Latin alphabet reading as transliteration."""
        for reading_type in ("interpretive_latin", "diplomatic"):
            for reading in readings:
                if reading.get("type") == reading_type:
                    return reading["text"]
        return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _is_valid_project_id(self, text: str) -> bool:
        """Check if a string looks like a valid DIAS project ID."""
        return bool(re.match(r"^I-[A-Z]{2,4}-[A-Z]?\d{2,3}$", text))

    def _is_stone_page(self, html_content: str) -> bool:
        """Check if HTML is a valid stone detail page (not a 404/generic page)."""
        return "CIIC" in html_content or "Findspot" in html_content or "INSCRIPTION" in html_content

    @staticmethod
    def _word_to_century(word: str) -> Optional[int]:
        """Convert a century word or number to its end year."""
        word_map = {
            "first": 100, "second": 200, "third": 300, "fourth": 400,
            "fifth": 500, "sixth": 600, "seventh": 700, "eighth": 800,
            "1st": 100, "2nd": 200, "3rd": 300, "4th": 400,
            "5th": 500, "6th": 600, "7th": 700, "8th": 800,
        }
        result = word_map.get(word.lower())
        if result:
            return result
        # Try numeric
        try:
            return int(word) * 100
        except ValueError:
            return None


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_dias_scraper(output_dir: str, **kwargs) -> DIASScraper:
    """
    Create a DIAS scraper with default configuration.

    Args:
        output_dir: Directory to save downloaded images
        **kwargs: Additional configuration options

    Returns:
        Configured DIASScraper instance
    """
    config = ScraperConfig(
        output_dir=output_dir,
        rate_limit=kwargs.get("rate_limit", 3.0),
        user_agent=kwargs.get("user_agent", "OghamOCR-Research/1.0"),
    )
    return DIASScraper(
        config,
        use_old_site=kwargs.get("use_old_site", False),
        enumerate_counties=kwargs.get("enumerate_counties", False),
    )
