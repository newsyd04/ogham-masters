"""
Core data schemas for the Ogham OCR pipeline.

This module defines the data structures used throughout the pipeline for
consistent data representation and type safety.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from pathlib import Path


class LicenseType(Enum):
    """License categories for downloaded images."""
    CC0 = "CC0"
    CC_BY = "CC-BY"
    CC_BY_SA = "CC-BY-SA"
    CC_BY_NC = "CC-BY-NC"
    CC_BY_NC_SA = "CC-BY-NC-SA"
    OGL = "OGL"  # UK Open Government License
    ACADEMIC = "ACADEMIC"
    UNKNOWN = "UNKNOWN"


LICENSE_PERMISSIONS = {
    LicenseType.CC0: {"commercial": True, "attribution": False, "derivatives": True},
    LicenseType.CC_BY: {"commercial": True, "attribution": True, "derivatives": True},
    LicenseType.CC_BY_SA: {"commercial": True, "attribution": True, "derivatives": "sharealike"},
    LicenseType.CC_BY_NC: {"commercial": False, "attribution": True, "derivatives": True},
    LicenseType.CC_BY_NC_SA: {"commercial": False, "attribution": True, "derivatives": "sharealike"},
    LicenseType.OGL: {"commercial": True, "attribution": True, "derivatives": True},
    LicenseType.ACADEMIC: {"commercial": False, "attribution": True, "derivatives": "research_only"},
    LicenseType.UNKNOWN: {"commercial": False, "attribution": True, "derivatives": False},
}


class TranscriptionConfidence(Enum):
    """Confidence level for transcription readings."""
    VERIFIED = "verified"      # Expert consensus, multiple sources agree
    PROBABLE = "probable"      # Single reputable source, no contradictions
    UNCERTAIN = "uncertain"    # Disputed readings or partial only
    MISSING = "missing"        # No transcription available


class ImageType(Enum):
    """Type of image source."""
    PHOTOGRAPH = "photograph"
    RUBBING = "rubbing"
    DRAWING = "drawing"
    RENDER_3D = "3d_render"


class QualityLevel(Enum):
    """Estimated image quality level."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNUSABLE = "unusable"


class WeatheringSeverity(Enum):
    """Weathering severity of the inscription."""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class ImageMetadata:
    """Metadata for a single downloaded image."""

    # Identifiers
    image_id: str              # UUID for this specific image
    stone_id: str              # Canonical stone identifier (e.g., "CIIC_001")
    source_id: str             # Source-specific ID

    # Source information
    source_name: str           # "CISP", "Canmore", etc.
    source_url: str            # Original URL
    download_date: str         # ISO 8601
    license: LicenseType       # License category
    license_url: Optional[str] = None  # Link to license terms

    # Content metadata
    transcription: Optional[str] = None         # Unicode Ogham if available
    transcription_source: Optional[str] = None  # Who made the transcription
    transcription_confidence: TranscriptionConfidence = TranscriptionConfidence.MISSING
    transliteration: Optional[str] = None       # Latin alphabet version
    translation: Optional[str] = None           # English meaning if known

    # Image metadata
    image_type: ImageType = ImageType.PHOTOGRAPH
    view_angle: Optional[str] = None   # "front", "side", "detail", "overview"
    is_cropped: bool = False           # Whether image is pre-cropped to inscription

    # Location
    findspot_name: Optional[str] = None
    current_location: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None  # lat, lon

    # Quality indicators
    estimated_quality: QualityLevel = QualityLevel.MEDIUM
    lighting_conditions: Optional[str] = None
    weathering_severity: Optional[WeatheringSeverity] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "image_id": self.image_id,
            "stone_id": self.stone_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "download_date": self.download_date,
            "license": self.license.value,
            "license_url": self.license_url,
            "transcription": self.transcription,
            "transcription_source": self.transcription_source,
            "transcription_confidence": self.transcription_confidence.value,
            "transliteration": self.transliteration,
            "translation": self.translation,
            "image_type": self.image_type.value,
            "view_angle": self.view_angle,
            "is_cropped": self.is_cropped,
            "findspot_name": self.findspot_name,
            "current_location": self.current_location,
            "coordinates": self.coordinates,
            "estimated_quality": self.estimated_quality.value,
            "lighting_conditions": self.lighting_conditions,
            "weathering_severity": self.weathering_severity.value if self.weathering_severity else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ImageMetadata":
        """Create from dictionary."""
        return cls(
            image_id=data["image_id"],
            stone_id=data["stone_id"],
            source_id=data["source_id"],
            source_name=data["source_name"],
            source_url=data["source_url"],
            download_date=data["download_date"],
            license=LicenseType(data["license"]),
            license_url=data.get("license_url"),
            transcription=data.get("transcription"),
            transcription_source=data.get("transcription_source"),
            transcription_confidence=TranscriptionConfidence(data.get("transcription_confidence", "missing")),
            transliteration=data.get("transliteration"),
            translation=data.get("translation"),
            image_type=ImageType(data.get("image_type", "photograph")),
            view_angle=data.get("view_angle"),
            is_cropped=data.get("is_cropped", False),
            findspot_name=data.get("findspot_name"),
            current_location=data.get("current_location"),
            coordinates=tuple(data["coordinates"]) if data.get("coordinates") else None,
            estimated_quality=QualityLevel(data.get("estimated_quality", "medium")),
            lighting_conditions=data.get("lighting_conditions"),
            weathering_severity=WeatheringSeverity(data["weathering_severity"]) if data.get("weathering_severity") else None,
        )


@dataclass
class StoneMetadata:
    """Metadata for a single Ogham stone."""

    stone_id: str              # Canonical identifier (e.g., "CIIC_001")
    ciic_number: Optional[int] = None  # CIIC catalog number
    name: Optional[str] = None         # Common name if any

    # Location
    region: str = "unknown"    # "Ireland", "Scotland", "Wales", etc.
    county: Optional[str] = None
    townland: Optional[str] = None
    findspot_name: Optional[str] = None
    current_location: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None

    # Physical properties
    stone_type: Optional[str] = None   # "pillar", "cross-slab", etc.
    material: Optional[str] = None     # "sandstone", "granite", etc.
    height_cm: Optional[float] = None

    # Inscription
    transcription_readings: List[Dict] = field(default_factory=list)  # Multiple interpretations
    consensus_transcription: Optional[str] = None
    inscription_language: str = "Primitive Irish"
    estimated_date_range: Optional[Tuple[int, int]] = None  # (start_year, end_year)

    # Condition
    preservation_state: Optional[str] = None
    weathering_severity: Optional[WeatheringSeverity] = None

    # References
    bibliography: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "stone_id": self.stone_id,
            "ciic_number": self.ciic_number,
            "name": self.name,
            "region": self.region,
            "county": self.county,
            "townland": self.townland,
            "findspot_name": self.findspot_name,
            "current_location": self.current_location,
            "coordinates": self.coordinates,
            "stone_type": self.stone_type,
            "material": self.material,
            "height_cm": self.height_cm,
            "transcription_readings": self.transcription_readings,
            "consensus_transcription": self.consensus_transcription,
            "inscription_language": self.inscription_language,
            "estimated_date_range": self.estimated_date_range,
            "preservation_state": self.preservation_state,
            "weathering_severity": self.weathering_severity.value if self.weathering_severity else None,
            "bibliography": self.bibliography,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StoneMetadata":
        """Create from dictionary."""
        return cls(
            stone_id=data["stone_id"],
            ciic_number=data.get("ciic_number"),
            name=data.get("name"),
            region=data.get("region", "unknown"),
            county=data.get("county"),
            townland=data.get("townland"),
            findspot_name=data.get("findspot_name"),
            current_location=data.get("current_location"),
            coordinates=tuple(data["coordinates"]) if data.get("coordinates") else None,
            stone_type=data.get("stone_type"),
            material=data.get("material"),
            height_cm=data.get("height_cm"),
            transcription_readings=data.get("transcription_readings", []),
            consensus_transcription=data.get("consensus_transcription"),
            inscription_language=data.get("inscription_language", "Primitive Irish"),
            estimated_date_range=tuple(data["estimated_date_range"]) if data.get("estimated_date_range") else None,
            preservation_state=data.get("preservation_state"),
            weathering_severity=WeatheringSeverity(data["weathering_severity"]) if data.get("weathering_severity") else None,
            bibliography=data.get("bibliography", []),
        )


@dataclass
class OghamSample:
    """Single training sample for the OCR model."""

    sample_id: str              # Unique identifier
    stone_id: str               # For split integrity
    image_path: str             # Relative path from dataset root

    # Ground truth
    transcription: str          # Unicode Ogham sequence
    transcription_length: int   # Character count
    confidence: TranscriptionConfidence

    # Preprocessing applied
    preprocessing_version: str  # Track preprocessing changes
    is_synthetic: bool          # False for real images

    # For curriculum learning
    difficulty_score: float = 0.5     # 0.0 (easy) to 1.0 (hard)
    weathering_level: int = 1         # 0-3 scale

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "sample_id": self.sample_id,
            "stone_id": self.stone_id,
            "image_path": self.image_path,
            "transcription": self.transcription,
            "transcription_length": self.transcription_length,
            "confidence": self.confidence.value,
            "preprocessing_version": self.preprocessing_version,
            "is_synthetic": self.is_synthetic,
            "difficulty_score": self.difficulty_score,
            "weathering_level": self.weathering_level,
        }


@dataclass
class TranscriptionRecord:
    """Record of transcription with multiple interpretations."""

    stone_id: str
    readings: List[Dict] = field(default_factory=list)
    # Example reading: {"text": "ᚋᚐᚊᚔ", "source": "Macalister 1945", "confidence": "probable"}
    consensus_reading: Optional[str] = None
    notes: str = ""

    def add_reading(self, text: str, source: str, confidence: str):
        """Add a new reading interpretation."""
        self.readings.append({
            "text": text,
            "source": source,
            "confidence": confidence,
        })

    def get_best_reading(self) -> Optional[str]:
        """Get the most confident reading."""
        if self.consensus_reading:
            return self.consensus_reading

        confidence_order = {"verified": 3, "probable": 2, "uncertain": 1}
        if self.readings:
            best = max(self.readings, key=lambda x: confidence_order.get(x.get("confidence", "uncertain"), 0))
            return best["text"]
        return None


@dataclass
class CropAnnotation:
    """Annotation for a cropped region of an image."""

    image_id: str
    crop_id: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    transcription: str
    confidence: TranscriptionConfidence
    annotator: str
    timestamp: str
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "image_id": self.image_id,
            "crop_id": self.crop_id,
            "bbox": self.bbox,
            "transcription": self.transcription,
            "confidence": self.confidence.value,
            "annotator": self.annotator,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }
