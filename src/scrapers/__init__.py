"""Web scrapers for Ogham inscription images."""

from .base_scraper import OghamScraperBase, ImageDownload, ScraperConfig
from .cisp_scraper import CISPScraper
from .dias_scraper import DIASScraper
from .wikimedia_scraper import WikimediaScraper

__all__ = [
    "OghamScraperBase",
    "ImageDownload",
    "ScraperConfig",
    "CISPScraper",
    "DIASScraper",
    "WikimediaScraper",
]
