"""
Training loop for Ogham OCR with TrOCR.

Provides a complete training pipeline with:
- Curriculum learning support
- Mixed precision training
- Checkpoint management
- Logging and monitoring

★ Insight ─────────────────────────────────────
Training strategy for Ogham OCR:
1. Transfer learn from TrOCR-base-printed
2. Use curriculum learning (synthetic → real)
3. Monitor domain gap (synthetic vs real CER)
4. Early stopping on validation CER
─────────────────────────────────────────────────
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json
import logging
import time

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kwargs: x


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Experiment
    experiment_name: str = "ogham_ocr_v1"

    # Model
    model_name: str = "microsoft/trocr-base-stage1"
    max_length: int = 64

    # Output mode: "ogham" for Unicode output, "latin" for transliteration
    output_mode: str = "ogham"

    # Training
    num_epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Encoder freezing
    freeze_encoder_epochs: int = 5  # Freeze vision encoder for first N epochs

    # Mixed precision
    use_amp: bool = True

    # Data
    synthetic_size: int = 200000
    curriculum_schedule: str = "default"
    num_workers: int = 2

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3

    # Logging
    log_every_n_steps: int = 100
    eval_every_n_epochs: int = 1
    use_wandb: bool = True

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_cer"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "freeze_encoder_epochs": self.freeze_encoder_epochs,
            "use_amp": self.use_amp,
            "synthetic_size": self.synthetic_size,
            "curriculum_schedule": self.curriculum_schedule,
        }


class OghamTrainer:
    """
    Trainer for Ogham OCR model.

    Handles the complete training loop including:
    - Forward/backward passes
    - Gradient accumulation
    - Mixed precision
    - Curriculum learning
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        train_dataset: Any,
        val_dataset: Any,
        config: TrainingConfig,
        checkpoint_manager: Optional[Any] = None,
        logger: Optional[Any] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: TrOCR model
            processor: TrOCR processor
            train_dataset: Training dataset (MixedOghamDataset)
            val_dataset: Validation dataset
            config: Training configuration
            checkpoint_manager: Optional checkpoint manager
            logger: Optional experiment logger
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training")

        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.exp_logger = logger

        # Setup device (cuda > mps > cpu)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision scaler (only useful on CUDA)
        use_scaler = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if use_scaler else None

        # Tracking
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float("inf")
        self.patience_counter = 0

        # Encoder freezing state
        self.encoder_frozen = False

        # Risk monitor for domain gap tracking
        self.risk_monitor = None

        # Setup logging
        self.log = logging.getLogger("ogham_trainer")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Don't apply weight decay to bias and LayerNorm
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
        )

    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler."""
        from torch.optim.lr_scheduler import OneCycleLR

        steps_per_epoch = len(self.train_dataset) // self.config.batch_size
        # Account for gradient accumulation: optimizer only steps every N batches
        optimizer_steps_per_epoch = max(1, steps_per_epoch // self.config.gradient_accumulation_steps)
        total_steps = optimizer_steps_per_epoch * self.config.num_epochs

        return OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,  # 10% warmup
        )

    def _freeze_encoder(self):
        """Freeze the vision encoder parameters for stable transfer learning."""
        if hasattr(self.model, "encoder"):
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            self.encoder_frozen = True
            self.log.info("Encoder frozen - only decoder will be trained")

    def _unfreeze_encoder(self):
        """Unfreeze the vision encoder for full fine-tuning."""
        if hasattr(self.model, "encoder"):
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            self.encoder_frozen = False
            self.log.info("Encoder unfrozen - full model training enabled")

    def train(self, start_epoch: int = 0) -> Dict:
        """
        Run training loop.

        Args:
            start_epoch: Epoch to start from (for resuming)

        Returns:
            Training history
        """
        self.log.info(f"Starting training from epoch {start_epoch}")
        self.log.info(f"Device: {self.device}")
        self.log.info(f"Config: {self.config.to_dict()}")

        history = {"train_loss": [], "val_cer": [], "val_loss": []}

        # Initialize risk monitor for domain gap tracking
        from ..evaluation.analysis import RiskMonitor
        self.risk_monitor = RiskMonitor(logger=self.log)

        # Freeze encoder for early epochs if configured
        if self.config.freeze_encoder_epochs > 0 and start_epoch < self.config.freeze_encoder_epochs:
            self._freeze_encoder()

        # Create data collator
        from ..datasets.collator import OghamDataCollator
        collator = OghamDataCollator(self.processor)

        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Unfreeze encoder when freeze period ends
            if self.encoder_frozen and epoch >= self.config.freeze_encoder_epochs:
                self._unfreeze_encoder()

            # Update curriculum if using mixed dataset
            if hasattr(self.train_dataset, "set_epoch"):
                self.train_dataset.set_epoch(epoch)

            # Build DataLoader each epoch — sampler weights change with curriculum
            pin = self.device.type == "cuda"
            if hasattr(self.train_dataset, "get_weighted_sampler"):
                sampler = self.train_dataset.get_weighted_sampler()
                train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.config.batch_size,
                    sampler=sampler,
                    num_workers=self.config.num_workers,
                    collate_fn=collator,
                    pin_memory=pin,
                )
            else:
                train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=self.config.num_workers,
                    collate_fn=collator,
                    pin_memory=pin,
                )

            # Train epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            history["train_loss"].append(train_metrics["loss"])

            # Evaluate
            if (epoch + 1) % self.config.eval_every_n_epochs == 0:
                val_metrics = self.evaluate()
                history["val_cer"].append(val_metrics.get("cer", 0))
                history["val_loss"].append(val_metrics.get("loss", 0))

                # Build combined metrics for logging and risk monitoring
                epoch_metrics = {
                    "epoch": epoch,
                    **train_metrics,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }

                # Add stratified metrics if available
                if "stratified" in val_metrics:
                    strat = val_metrics["stratified"]
                    # Extract domain gap info from by_source
                    by_source = strat.get("by_source", {})
                    if "synthetic" in by_source:
                        epoch_metrics["val_cer_synthetic"] = by_source["synthetic"]["cer"]
                    if "real" in by_source:
                        epoch_metrics["val_cer_real"] = by_source["real"]["cer"]

                # Check for training risks
                if self.risk_monitor:
                    warnings = self.risk_monitor.check_epoch(epoch_metrics)
                    if warnings:
                        epoch_metrics["risk_warnings"] = warnings

                # Log metrics
                self._log_metrics(epoch_metrics)

                # Check for improvement
                current_metric = val_metrics.get(self.config.early_stopping_metric.replace("val_", ""), float("inf"))

                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0

                    # Save best checkpoint
                    if self.checkpoint_manager:
                        self.checkpoint_manager.save_checkpoint(
                            self.model, self.optimizer, epoch,
                            val_metrics, is_best=True
                        )
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    self.log.info(f"Early stopping at epoch {epoch}")
                    break

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                if self.checkpoint_manager:
                    self.checkpoint_manager.save_checkpoint(
                        self.model, self.optimizer, epoch,
                        {"train_loss": train_metrics["loss"]}
                    )

        return history

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass with mixed precision
            amp_device = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.amp.autocast(amp_device, enabled=self.config.use_amp and self.device.type == "cuda"):
                outputs = self.model(
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss

                # Scale for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            # Periodic logging
            if self.global_step % self.config.log_every_n_steps == 0:
                self._log_metrics({
                    "train/loss": total_loss / num_batches,
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/step": self.global_step,
                })

        return {"loss": total_loss / num_batches}

    def evaluate(self) -> Dict:
        """Evaluate on validation set with optional stratified analysis."""
        self.model.eval()

        from ..datasets.collator import MetadataCollator
        collator = MetadataCollator(self.processor)

        # Enable metadata return for stratified evaluation
        return_metadata = getattr(self.val_dataset, "return_metadata", False)
        if hasattr(self.val_dataset, "return_metadata"):
            self.val_dataset.return_metadata = True

        pin = self.device.type == "cuda"
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collator,
            pin_memory=pin,
        )

        total_loss = 0
        all_predictions = []
        all_references = []
        all_metadata = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    labels=labels,
                )
                total_loss += outputs.loss.item()

                # Generate predictions
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=self.config.max_length,
                )

                # Decode
                predictions = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )

                # Get references (decode labels, replacing -100)
                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = self.processor.tokenizer.pad_token_id
                references = self.processor.batch_decode(
                    labels_for_decode,
                    skip_special_tokens=True,
                )

                all_predictions.extend(predictions)
                all_references.extend(references)

                # Collect per-sample metadata if available
                if "is_synthetic" in batch:
                    for i in range(len(predictions)):
                        all_metadata.append({
                            "is_synthetic": batch["is_synthetic"][i] if "is_synthetic" in batch else False,
                            "difficulty": batch.get("difficulty", [None])[i] if "difficulty" in batch else "unknown",
                        })

        # Restore original metadata setting
        if hasattr(self.val_dataset, "return_metadata"):
            self.val_dataset.return_metadata = return_metadata

        # Compute overall metrics
        from ..evaluation.metrics import compute_cer, compute_exact_match

        cer = compute_cer(all_predictions, all_references)
        exact_match = compute_exact_match(all_predictions, all_references)

        result = {
            "loss": total_loss / len(val_loader),
            "cer": cer,
            "exact_match": exact_match,
        }

        # Stratified evaluation if metadata was collected
        if all_metadata:
            from ..evaluation.analysis import EvaluationStrategy
            result["stratified"] = EvaluationStrategy.stratified_evaluation(
                all_predictions, all_references, all_metadata
            )

        return result

    def _log_metrics(self, metrics: Dict):
        """Log metrics to experiment logger."""
        if self.exp_logger and hasattr(self.exp_logger, "log"):
            self.exp_logger.log(metrics)

        # Also log to console
        self.log.info(f"Metrics: {metrics}")

    def resume_if_possible(self) -> int:
        """
        Resume training from checkpoint if available.

        Returns:
            Start epoch (0 if no checkpoint found)
        """
        if self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_checkpoint("latest")
            if checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                self.log.info(f"Resumed from epoch {checkpoint['epoch']}")
                return start_epoch

        return 0
