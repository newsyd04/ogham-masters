"""
Checkpoint management for training.

Handles saving and loading model checkpoints with proper versioning.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CheckpointManager:
    """
    Manage model checkpoints during training.

    Features:
    - Save latest and best checkpoints
    - Periodic numbered checkpoints
    - Automatic cleanup of old checkpoints
    - Resume training support
    """

    def __init__(
        self,
        experiment_name: str,
        checkpoint_dir: str,
        keep_last_n: int = 3,
    ):
        """
        Initialize checkpoint manager.

        Args:
            experiment_name: Name of the experiment
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of periodic checkpoints to keep
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")

        self.experiment_name = experiment_name
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

        self.log = logging.getLogger("checkpoint_manager")

    def save_checkpoint(
        self,
        model: Any,
        optimizer: Any,
        epoch: int,
        metrics: Dict,
        is_best: bool = False,
    ):
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Metrics to store
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        # Always save latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        self.log.info(f"Saved checkpoint: {latest_path}")

        # Save best if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            self.log.info(f"Saved best checkpoint: {best_path}")

            # Save metrics separately for easy inspection
            metrics_path = self.checkpoint_dir / "best_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump({"epoch": epoch, **metrics}, f, indent=2)

        # Save numbered checkpoint for periodic saves
        numbered_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(checkpoint, numbered_path)

        # Cleanup old numbered checkpoints
        self._cleanup_old_checkpoints()

    def load_checkpoint(
        self,
        checkpoint_type: str = "latest",
        map_location: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Load a checkpoint.

        Args:
            checkpoint_type: "latest", "best", or epoch number
            map_location: Device to map tensors to

        Returns:
            Checkpoint dictionary or None if not found
        """
        if checkpoint_type == "latest":
            checkpoint_path = self.checkpoint_dir / "latest.pt"
        elif checkpoint_type == "best":
            checkpoint_path = self.checkpoint_dir / "best.pt"
        elif checkpoint_type.isdigit():
            checkpoint_path = self.checkpoint_dir / f"epoch_{int(checkpoint_type):04d}.pt"
        else:
            checkpoint_path = self.checkpoint_dir / checkpoint_type

        if not checkpoint_path.exists():
            self.log.warning(f"Checkpoint not found: {checkpoint_path}")
            return None

        self.log.info(f"Loading checkpoint: {checkpoint_path}")

        if map_location:
            return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        return torch.load(checkpoint_path, weights_only=True)

    def _cleanup_old_checkpoints(self):
        """Remove old numbered checkpoints, keeping last N."""
        numbered = sorted(self.checkpoint_dir.glob("epoch_*.pt"))

        if len(numbered) > self.keep_last_n:
            to_remove = numbered[:-self.keep_last_n]
            for path in to_remove:
                path.unlink()
                self.log.debug(f"Removed old checkpoint: {path}")

    def get_available_checkpoints(self) -> Dict[str, Path]:
        """Get dictionary of available checkpoints."""
        checkpoints = {}

        for name in ["latest", "best"]:
            path = self.checkpoint_dir / f"{name}.pt"
            if path.exists():
                checkpoints[name] = path

        for path in sorted(self.checkpoint_dir.glob("epoch_*.pt")):
            epoch = int(path.stem.split("_")[1])
            checkpoints[f"epoch_{epoch}"] = path

        return checkpoints

    def export_for_inference(
        self,
        output_path: str,
        checkpoint_type: str = "best",
    ):
        """
        Export checkpoint for inference (model weights only).

        Args:
            output_path: Path to save exported model
            checkpoint_type: Which checkpoint to export
        """
        checkpoint = self.load_checkpoint(checkpoint_type)
        if checkpoint is None:
            raise FileNotFoundError(f"Checkpoint {checkpoint_type} not found")

        export = {
            "model_state_dict": checkpoint["model_state_dict"],
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint.get("metrics", {}),
        }

        torch.save(export, output_path)
        self.log.info(f"Exported model to: {output_path}")
