"""
Google Colab storage management.

Handles persistent storage on Google Drive and local caching
for efficient training in Colab environments.

★ Insight ─────────────────────────────────────
Colab storage strategy:
1. Keep raw data and checkpoints on Drive (persistent)
2. Cache preprocessed data locally (fast but volatile)
3. Sync important outputs back to Drive
4. Handle session restarts gracefully
─────────────────────────────────────────────────
"""

import shutil
from pathlib import Path
from typing import Dict, Optional
import json
import logging


class ColabStorageManager:
    """
    Manage storage between Google Drive (persistent) and local (fast).

    Directory structure:
    - /content/drive/MyDrive/ogham_ocr/  (Drive - persistent)
        - datasets/
        - checkpoints/
        - logs/
    - /content/cache/  (Local - fast but volatile)
        - processed_images/
        - tokenizer_cache/
    """

    DRIVE_ROOT = Path("/content/drive/MyDrive/ogham_ocr")
    LOCAL_CACHE = Path("/content/cache")

    def __init__(self, auto_mount: bool = True):
        """
        Initialize storage manager.

        Args:
            auto_mount: Automatically mount Google Drive
        """
        self.log = logging.getLogger("colab_storage")
        self._drive_mounted = False

        if auto_mount:
            self.mount_drive()

        # Ensure local cache exists
        self.LOCAL_CACHE.mkdir(parents=True, exist_ok=True)

    def mount_drive(self):
        """Mount Google Drive if in Colab environment."""
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            self._drive_mounted = True
            self.log.info("Google Drive mounted successfully")

            # Create directory structure
            self._ensure_directories()

        except ImportError:
            self.log.warning("Not in Colab environment, using local paths")
            # Use local directories instead
            self.DRIVE_ROOT = Path("./ogham_data")
            self._ensure_directories()

        except Exception as e:
            self.log.error(f"Failed to mount Drive: {e}")
            raise

    def _ensure_directories(self):
        """Create required directory structure."""
        directories = [
            self.DRIVE_ROOT / "datasets" / "real",
            self.DRIVE_ROOT / "datasets" / "fonts",
            self.DRIVE_ROOT / "checkpoints",
            self.DRIVE_ROOT / "logs",
            self.DRIVE_ROOT / "experiments",
            self.LOCAL_CACHE / "processed_images",
            self.LOCAL_CACHE / "tokenizer_cache",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, experiment_name: str) -> Path:
        """Get path for model checkpoints (persistent on Drive)."""
        path = self.DRIVE_ROOT / "checkpoints" / experiment_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_dataset_path(self, dataset_type: str = "real") -> Path:
        """Get path for datasets."""
        return self.DRIVE_ROOT / "datasets" / dataset_type

    def get_font_path(self) -> Path:
        """Get path for Ogham fonts."""
        return self.DRIVE_ROOT / "datasets" / "fonts"

    def get_log_path(self, experiment_name: str) -> Path:
        """Get path for experiment logs."""
        path = self.DRIVE_ROOT / "logs" / experiment_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cache_locally(self, src_path: Path, cache_name: str) -> Path:
        """
        Copy from Drive to local cache for faster access.

        Args:
            src_path: Source path on Drive
            cache_name: Name for cached directory

        Returns:
            Path to local cache
        """
        local_path = self.LOCAL_CACHE / cache_name

        if not local_path.exists():
            self.log.info(f"Caching {src_path} to {local_path}")
            shutil.copytree(src_path, local_path)
        else:
            self.log.info(f"Using existing cache: {local_path}")

        return local_path

    def sync_to_drive(self, local_path: Path, drive_subpath: str):
        """
        Sync local files back to Drive.

        Args:
            local_path: Local path to sync
            drive_subpath: Destination path relative to DRIVE_ROOT
        """
        drive_path = self.DRIVE_ROOT / drive_subpath
        drive_path.parent.mkdir(parents=True, exist_ok=True)

        self.log.info(f"Syncing {local_path} to {drive_path}")
        shutil.copytree(local_path, drive_path, dirs_exist_ok=True)

    def clear_cache(self):
        """Clear local cache to free up space."""
        if self.LOCAL_CACHE.exists():
            shutil.rmtree(self.LOCAL_CACHE)
            self.LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
            self.log.info("Local cache cleared")

    def get_storage_stats(self) -> Dict:
        """Get storage usage statistics."""
        import os

        def get_dir_size(path: Path) -> int:
            if not path.exists():
                return 0
            total = 0
            for entry in os.scandir(path):
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += get_dir_size(Path(entry.path))
            return total

        return {
            "drive_total_mb": get_dir_size(self.DRIVE_ROOT) / (1024 * 1024),
            "cache_total_mb": get_dir_size(self.LOCAL_CACHE) / (1024 * 1024),
            "checkpoints_mb": get_dir_size(self.DRIVE_ROOT / "checkpoints") / (1024 * 1024),
            "datasets_mb": get_dir_size(self.DRIVE_ROOT / "datasets") / (1024 * 1024),
        }

    def save_experiment_config(self, experiment_name: str, config: Dict):
        """Save experiment configuration."""
        config_path = self.DRIVE_ROOT / "experiments" / f"{experiment_name}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def load_experiment_config(self, experiment_name: str) -> Optional[Dict]:
        """Load experiment configuration."""
        config_path = self.DRIVE_ROOT / "experiments" / f"{experiment_name}.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return None


def setup_colab_environment() -> ColabStorageManager:
    """
    Setup Colab environment for training.

    Returns:
        Configured ColabStorageManager
    """
    storage = ColabStorageManager()

    # Print storage stats
    stats = storage.get_storage_stats()
    print(f"Storage usage:")
    print(f"  Drive total: {stats['drive_total_mb']:.1f} MB")
    print(f"  Cache total: {stats['cache_total_mb']:.1f} MB")
    print(f"  Checkpoints: {stats['checkpoints_mb']:.1f} MB")
    print(f"  Datasets: {stats['datasets_mb']:.1f} MB")

    return storage
