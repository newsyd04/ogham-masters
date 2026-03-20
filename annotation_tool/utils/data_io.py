"""
Data I/O utilities for the annotation tool.

Handles loading and saving annotations with versioning support.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class AnnotationManager:
    """Manages annotation data with versioning."""

    def __init__(self, data_dir: str):
        """
        Initialize annotation manager.

        Args:
            data_dir: Root directory of the dataset
        """
        self.data_dir = Path(data_dir)
        self.annotations_dir = self.data_dir / "processed" / "annotations"
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

        self.transcriptions_file = self.annotations_dir / "transcriptions.json"
        self.crops_file = self.annotations_dir / "crop_annotations.json"
        self.history_file = self.annotations_dir / "annotation_history.jsonl"

    def load_transcriptions(self) -> Dict[str, Dict]:
        """Load all transcription annotations."""
        if self.transcriptions_file.exists():
            with open(self.transcriptions_file) as f:
                return json.load(f)
        return {}

    def save_transcriptions(self, annotations: Dict[str, Dict]):
        """Save transcription annotations."""
        with open(self.transcriptions_file, "w") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

    def save_single_annotation(self, stone_id: str, annotation: Dict):
        """
        Save a single annotation with history tracking.

        Args:
            stone_id: Stone identifier
            annotation: Annotation data
        """
        # Load current annotations
        annotations = self.load_transcriptions()

        # Get previous version
        previous = annotations.get(stone_id)

        # Update annotation
        annotation["stone_id"] = stone_id
        annotation["updated_at"] = datetime.utcnow().isoformat()
        annotation["version"] = (previous.get("version", 0) + 1) if previous else 1

        annotations[stone_id] = annotation

        # Save annotations
        self.save_transcriptions(annotations)

        # Log to history
        self._log_history(stone_id, annotation, previous)

    def _log_history(self, stone_id: str, new: Dict, previous: Optional[Dict]):
        """Log annotation change to history file."""
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "stone_id": stone_id,
            "version": new.get("version", 1),
            "transcription": new.get("transcription"),
            "previous_transcription": previous.get("transcription") if previous else None,
            "annotator": new.get("annotator"),
            "confidence": new.get("confidence"),
        }

        with open(self.history_file, "a") as f:
            f.write(json.dumps(history_entry) + "\n")

    def load_crop_annotations(self) -> Dict[str, List[Dict]]:
        """Load crop bounding box annotations."""
        if self.crops_file.exists():
            with open(self.crops_file) as f:
                return json.load(f)
        return {}

    def save_crop_annotation(self, image_id: str, crop: Dict):
        """Save a crop annotation."""
        crops = self.load_crop_annotations()

        if image_id not in crops:
            crops[image_id] = []

        crop["created_at"] = datetime.utcnow().isoformat()
        crops[image_id].append(crop)

        with open(self.crops_file, "w") as f:
            json.dump(crops, f, indent=2)

    def get_progress_stats(self) -> Dict:
        """Get annotation progress statistics."""
        # Count stones with images
        images_dir = self.data_dir / "raw" / "images"
        total_stones = 0
        if images_dir.exists():
            total_stones = len([d for d in images_dir.iterdir() if d.is_dir()])

        # Count annotated
        annotations = self.load_transcriptions()
        annotated = len(annotations)

        # Count by confidence
        by_confidence = {"verified": 0, "probable": 0, "uncertain": 0}
        for ann in annotations.values():
            conf = ann.get("confidence", "uncertain")
            by_confidence[conf] = by_confidence.get(conf, 0) + 1

        return {
            "total_stones": total_stones,
            "annotated": annotated,
            "remaining": total_stones - annotated,
            "progress_percent": (annotated / total_stones * 100) if total_stones > 0 else 0,
            "by_confidence": by_confidence,
        }

    def export_for_training(self, output_file: str, min_confidence: str = "probable"):
        """
        Export annotations in training format.

        Args:
            output_file: Output JSON file path
            min_confidence: Minimum confidence level to include
        """
        annotations = self.load_transcriptions()

        confidence_levels = ["verified", "probable", "uncertain"]
        min_level = confidence_levels.index(min_confidence)

        training_data = []
        for stone_id, ann in annotations.items():
            conf = ann.get("confidence", "uncertain")
            if confidence_levels.index(conf) <= min_level:
                training_data.append({
                    "stone_id": stone_id,
                    "transcription": ann.get("transcription"),
                    "confidence": conf,
                })

        with open(output_file, "w") as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)

        return len(training_data)
