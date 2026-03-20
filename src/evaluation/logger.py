"""
Experiment logging for Ogham OCR training.

Provides unified logging to console, files, and optionally W&B.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging


class ExperimentLogger:
    """
    Comprehensive experiment logging.

    Supports:
    - Console logging
    - File logging (JSON lines)
    - Weights & Biases integration (optional)
    - Artifact tracking
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str,
        use_wandb: bool = True,
        wandb_project: str = "ogham-ocr",
        config: Optional[Dict] = None,
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save logs
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            config: Experiment configuration to log
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.events_file = self.log_dir / "events.jsonl"

        # Setup console logging
        self.console_logger = logging.getLogger(f"experiment.{experiment_name}")
        self.console_logger.setLevel(logging.INFO)

        if not self.console_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            self.console_logger.addHandler(handler)

        # Setup W&B if available and requested
        self.wandb = None
        self.use_wandb = use_wandb

        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    config=config or {},
                )
                self.wandb = wandb
                self.console_logger.info(f"W&B initialized: {wandb.run.url}")
            except ImportError:
                self.console_logger.warning("wandb not available, logging locally only")
                self.use_wandb = False
            except Exception as e:
                self.console_logger.warning(f"W&B init failed: {e}")
                self.use_wandb = False

        # Log initial config
        if config:
            self.log_config(config)

    def log_config(self, config: Dict):
        """Log experiment configuration."""
        # Save to file
        config_file = self.log_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        # Update W&B
        if self.wandb:
            self.wandb.config.update(config)

    def log(self, metrics: Dict, step: Optional[int] = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric values
            step: Optional step number
        """
        # Add timestamp
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "step": step,
            **metrics,
        }

        # Write to file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Log to W&B
        if self.wandb:
            self.wandb.log(metrics, step=step)

        # Console summary
        self.console_logger.info(
            f"Step {step}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in metrics.items() if k != "timestamp")
        )

    def log_event(self, event_type: str, data: Dict):
        """
        Log an event (e.g., checkpoint saved, error occurred).

        Args:
            event_type: Type of event
            data: Event data
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            **data,
        }

        with open(self.events_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self.console_logger.info(f"Event [{event_type}]: {data}")

    def log_predictions(
        self,
        predictions: List[str],
        references: List[str],
        metadata: Optional[List[Dict]] = None,
        split: str = "val",
        max_samples: int = 50,
    ):
        """
        Log sample predictions for qualitative analysis.

        Args:
            predictions: Predicted strings
            references: Reference strings
            metadata: Optional metadata per sample
            split: Data split name
            max_samples: Maximum samples to log
        """
        samples = []
        for i in range(min(len(predictions), max_samples)):
            sample = {
                "prediction": predictions[i],
                "reference": references[i],
                "correct": predictions[i] == references[i],
            }
            if metadata and i < len(metadata):
                sample["metadata"] = metadata[i]
            samples.append(sample)

        # Save to file
        predictions_file = self.log_dir / f"{split}_predictions.json"
        with open(predictions_file, "w") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        # Log to W&B as table
        if self.wandb:
            columns = ["prediction", "reference", "correct"]
            table_data = [[s["prediction"], s["reference"], s["correct"]] for s in samples]
            table = self.wandb.Table(columns=columns, data=table_data)
            self.wandb.log({f"{split}_predictions": table})

    def log_artifact(self, name: str, artifact_type: str, path: str):
        """
        Log an artifact (model, dataset, etc.).

        Args:
            name: Artifact name
            artifact_type: Type (model, dataset, etc.)
            path: Path to artifact
        """
        if self.wandb:
            artifact = self.wandb.Artifact(name, type=artifact_type)
            artifact.add_file(path)
            self.wandb.log_artifact(artifact)

        self.log_event("artifact_logged", {
            "name": name,
            "type": artifact_type,
            "path": path,
        })

    def finish(self):
        """Finish logging session."""
        if self.wandb:
            self.wandb.finish()

        self.log_event("experiment_finished", {
            "experiment_name": self.experiment_name,
        })

        self.console_logger.info(f"Experiment {self.experiment_name} finished")


def create_logger(
    experiment_name: str,
    log_dir: str = "./logs",
    use_wandb: bool = True,
    config: Optional[Dict] = None,
) -> ExperimentLogger:
    """
    Create experiment logger with standard configuration.

    Args:
        experiment_name: Name of experiment
        log_dir: Log directory
        use_wandb: Use Weights & Biases
        config: Experiment config

    Returns:
        Configured ExperimentLogger
    """
    return ExperimentLogger(
        experiment_name=experiment_name,
        log_dir=f"{log_dir}/{experiment_name}",
        use_wandb=use_wandb,
        config=config,
    )
