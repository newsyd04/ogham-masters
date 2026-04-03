#!/usr/bin/env python3
"""
Unified training script for Ogham OCR on Google Colab (or local).

Wires together all existing src/ components:
- Tokenizer extension with Ogham Unicode tokens
- Pre-generated sharded datasets (CSV + images)
- On-the-fly synthetic generation (LazySyntheticDataset)
- Real stone data (RealOghamDataset)
- Mixed dataset with curriculum learning
- OghamTrainer with AMP, gradient accumulation, early stopping
- ColabStorageManager for Drive persistence

Usage:
    # Phase 1: Synthetic-only training (sharded data)
    python scripts/train_colab.py \
        --mode ogham --model-size small --phase 1 \
        --data-dir ogham_dataset/synthetic_200k \
        --val-data-dir ogham_dataset/synthetic_val \
        --epochs 20 --batch-size 16

    # Phase 2: Curriculum with real data
    python scripts/train_colab.py \
        --mode ogham --model-size base --phase 2 \
        --data-dir ogham_dataset/synthetic_200k \
        --real-data-dir ogham_dataset \
        --resume --epochs 50

    # Compare ogham vs latin modes
    python scripts/train_colab.py \
        --mode compare --model-size small --phase 1 \
        --data-dir ogham_dataset/synthetic_200k
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# Dataset: loads pre-generated sharded images + CSV labels
# =============================================================================

class ShardedDataset:
    """
    PyTorch Dataset loading pre-generated images from sharded directories.

    Supports both single-directory and multi-shard layouts:
    - Single: data_dir/images/*.jpg + data_dir/labels.csv
    - Sharded: data_dir/shard_XX/images/*.jpg + data_dir/shard_XX/labels.csv
    """

    def __init__(
        self,
        data_dir: str,
        processor: Any,
        tokenizer: Any,
        mode: str = "ogham",
        max_length: int = 64,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        import torch
        from PIL import Image as PILImage

        self.processor = processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.torch = torch
        self.PILImage = PILImage

        # Discover and load all samples from shards or single directory
        data_dir = Path(data_dir)
        all_rows = self._load_all_labels(data_dir)

        log.info(f"Found {len(all_rows)} total samples in {data_dir}")

        # Split into train/val
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_rows))
        val_size = max(1, int(len(all_rows) * val_ratio))

        if split == "val":
            selected = indices[:val_size]
        else:
            selected = indices[val_size:]

        self.samples = [all_rows[i] for i in selected]
        log.info(f"Loaded {len(self.samples)} {split} samples (mode: {mode})")

    def _load_all_labels(self, data_dir: Path) -> List[Dict]:
        """Load labels from all shards or a single directory."""
        rows = []

        # Check for sharded layout (shard_00, shard_01, ...)
        shard_dirs = sorted(data_dir.glob("shard_*"))
        if shard_dirs:
            for shard_dir in shard_dirs:
                csv_path = shard_dir / "labels.csv"
                if csv_path.exists():
                    rows.extend(self._load_csv(csv_path, shard_dir))
        else:
            # Single directory layout
            csv_path = data_dir / "labels.csv"
            if csv_path.exists():
                rows.extend(self._load_csv(csv_path, data_dir))

        return rows

    def _load_csv(self, csv_path: Path, base_dir: Path) -> List[Dict]:
        """Load a single CSV and resolve image paths, filtering corrupt images."""
        rows = []
        skipped = 0
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Resolve image path (check both root and images/ subdir)
                image_path = base_dir / row["image_file"]
                if not image_path.exists():
                    image_path = base_dir / "images" / row["image_file"]
                if not image_path.exists():
                    continue

                # Skip truncated files from quota errors (0-byte or tiny stubs)
                # Only validate on local disk — Drive FUSE stat() is too slow on cold cache
                is_drive = "/drive/" in str(image_path)
                if not is_drive:
                    try:
                        fsize = image_path.stat().st_size
                    except OSError:
                        skipped += 1
                        continue
                    if fsize < 500:
                        skipped += 1
                        continue

                rows.append({
                    "image_path": str(image_path),
                    "ogham_text": row["ogham_text"],
                    "latin_text": row["latin_transliteration"],
                    "difficulty": row.get("difficulty", "unknown"),
                })

        if skipped:
            log.warning(f"Skipped {skipped} corrupt images in {csv_path.parent.name}")
        return rows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            image = self.PILImage.open(sample["image_path"]).convert("RGB")
        except Exception:
            # Fallback: return a random valid sample instead of crashing
            fallback_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(fallback_idx)

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Choose target text based on mode
        text = sample["ogham_text"] if self.mode == "ogham" else sample["latin_text"]

        # Normalize U+1680 (Ogham Space Mark) → ASCII space for tokenizer compatibility
        text = text.replace("\u1680", " ")

        labels = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding with -100 for loss
        pad_id = self.tokenizer.pad_token_id
        labels = labels.clone()
        labels[labels == pad_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": text,
            "is_synthetic": True,
        }


# =============================================================================
# Dataset: loads from Hugging Face Hub (fast local cache, no Drive FUSE)
# =============================================================================

class HFDataset:
    """
    PyTorch Dataset loading from Hugging Face Hub.

    Downloads to Colab local disk on first load (~5 min), then reads from
    fast local cache. No Drive FUSE overhead on subsequent epochs.
    """

    def __init__(
        self,
        dataset_name: str,
        processor: Any,
        tokenizer: Any,
        mode: str = "ogham",
        max_length: int = 64,
        split: str = "train",
    ):
        import torch
        from datasets import load_dataset

        self.processor = processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.torch = torch

        log.info(f"Loading HF dataset: {dataset_name} (split={split})")
        self.ds = load_dataset(dataset_name, split=split, token=True)
        log.info(f"Loaded {len(self.ds)} samples from HF Hub")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]

        try:
            image = sample["image"].convert("RGB")
        except Exception:
            fallback_idx = (idx + 1) % len(self.ds)
            return self.__getitem__(fallback_idx)

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        text = sample["ogham_text"] if self.mode == "ogham" else sample["latin_transliteration"]
        text = text.replace("\u1680", " ")

        labels = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        pad_id = self.tokenizer.pad_token_id
        labels = labels.clone()
        labels[labels == pad_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": text,
            "is_synthetic": True,
        }


# =============================================================================
# Collator
# =============================================================================

class SimpleCollator:
    """Batch collator that stacks tensors and preserves metadata."""

    def __call__(self, batch):
        import torch
        result = {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }
        if any("text" in b for b in batch):
            result["texts"] = [b.get("text", "") for b in batch]
        if any("is_synthetic" in b for b in batch):
            result["is_synthetic"] = [b.get("is_synthetic", None) for b in batch]
        return result


# =============================================================================
# Training
# =============================================================================

def setup_model(mode: str, model_name: str, init_strategy: str = "latin"):
    """Setup model, processor, tokenizer based on mode."""
    if mode == "ogham":
        from src.training.tokenizer_extension import setup_ogham_model_and_tokenizer
        model, processor, tokenizer = setup_ogham_model_and_tokenizer(
            model_name, init_strategy=init_strategy
        )
    else:
        from src.training.tokenizer_extension import setup_transliteration_model
        model, processor, tokenizer = setup_transliteration_model(model_name)

    return model, processor, tokenizer


def resolve_device():
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """Compute Character Error Rate."""
    try:
        import editdistance
        total_dist = sum(editdistance.eval(p, r) for p, r in zip(predictions, references))
    except ImportError:
        # Fallback
        total_dist = 0
        for p, r in zip(predictions, references):
            m, n = len(p), len(r)
            dp = list(range(n + 1))
            for i in range(1, m + 1):
                prev, dp[0] = dp[0], i
                for j in range(1, n + 1):
                    temp = dp[j]
                    dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (0 if p[i-1] == r[j-1] else 1))
                    prev = temp
            total_dist += dp[n]

    total_chars = sum(max(len(r), 1) for r in references)
    return total_dist / total_chars if total_chars > 0 else 0.0


def train_one_epoch(model, dataloader, optimizer, device, scaler=None):
    """Train for one epoch."""
    import torch
    from tqdm import tqdm

    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="  Training", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        use_amp = scaler is not None
        amp_device = "cuda" if device.type == "cuda" else "cpu"

        with torch.amp.autocast(amp_device, enabled=use_amp):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, tokenizer, device):
    """Evaluate model, compute CER and loss."""
    import torch
    from tqdm import tqdm

    model.eval()
    total_loss = 0
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()

            generated_ids = model.generate(pixel_values, max_length=64, num_beams=4)
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_preds.extend(preds)

            # Decode references
            labels_dec = labels.clone()
            labels_dec[labels_dec == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(labels_dec, skip_special_tokens=True)
            all_refs.extend(refs)

    n_batches = max(len(all_preds) // max(1, len(batch["pixel_values"])), 1)
    avg_loss = total_loss / max(n_batches, 1)
    cer = compute_cer(all_preds, all_refs)
    exact = sum(1 for p, r in zip(all_preds, all_refs) if p.strip() == r.strip()) / max(len(all_preds), 1)

    # Show a few sample predictions
    n_show = min(3, len(all_preds))
    for i in range(n_show):
        log.info(f"  Sample {i}: ref='{all_refs[i]}' pred='{all_preds[i]}'")

    return {"loss": avg_loss, "cer": cer, "exact_match": exact}


def run_training(
    mode: str,
    model_name: str,
    data_dir: str,
    val_data_dir: Optional[str],
    real_data_dir: Optional[str],
    font_dir: str,
    phase: int,
    epochs: int,
    batch_size: int,
    lr: float,
    freeze_encoder_epochs: int,
    gradient_accumulation: int,
    init_strategy: str,
    checkpoint_dir: str,
    resume: bool,
    device_str: str,
    num_workers: int,
    hf_dataset: Optional[str] = None,
) -> Dict:
    """Run a complete training session."""
    import torch
    from torch.utils.data import DataLoader

    # Setup device
    if device_str == "auto":
        device = resolve_device()
    else:
        device = torch.device(device_str)
    log.info(f"Device: {device}")

    # Setup model
    log.info(f"Setting up model: {model_name} (mode: {mode})")
    model, processor, tokenizer = setup_model(mode, model_name, init_strategy)
    model = model.to(device)

    # Fix meta-device buffers (TrOCR-base embed_positions._float_tensor)
    for name, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            module = model
            parts = name.split(".")
            for part in parts[:-1]:
                module = getattr(module, part)
            new_buf = torch.empty(buf.shape, dtype=buf.dtype, device=device)
            module.register_buffer(parts[-1], new_buf)
            log.info(f"Fixed meta-device buffer: {name} -> {device}")

    # Load checkpoint if resuming
    start_epoch = 0
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_ckpt = checkpoint_path / f"best_{mode}"

    if resume and best_ckpt.exists():
        from transformers import VisionEncoderDecoderModel
        log.info(f"Resuming from checkpoint: {best_ckpt}")
        model = VisionEncoderDecoderModel.from_pretrained(str(best_ckpt)).to(device)

    # Create datasets (HF Hub or file-based)
    if hf_dataset:
        log.info(f"Loading from HF Hub: {hf_dataset}")
        train_dataset = HFDataset(
            dataset_name=hf_dataset,
            processor=processor,
            tokenizer=tokenizer,
            mode=mode,
            split="train",
        )
        val_dataset = HFDataset(
            dataset_name=hf_dataset,
            processor=processor,
            tokenizer=tokenizer,
            mode=mode,
            split="validation",
        )
    else:
        log.info(f"Loading training data from {data_dir}")
        train_dataset = ShardedDataset(
            data_dir=data_dir,
            processor=processor,
            tokenizer=tokenizer,
            mode=mode,
            split="train",
        )

        if val_data_dir and Path(val_data_dir).exists():
            log.info(f"Loading validation data from {val_data_dir}")
            val_dataset = ShardedDataset(
                data_dir=val_data_dir,
                processor=processor,
                tokenizer=tokenizer,
                mode=mode,
                split="val",
                val_ratio=1.0,  # Use all as validation
            )
        else:
            log.info("Using 10% of training data as validation")
            val_dataset = ShardedDataset(
                data_dir=data_dir,
                processor=processor,
                tokenizer=tokenizer,
                mode=mode,
                split="val",
            )

    log.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Phase 2: wrap with MixedOghamDataset for curriculum learning
    if phase == 2 and real_data_dir:
        log.info(f"Phase 2: Adding real data from {real_data_dir}")
        from src.datasets.real_dataset import RealOghamDataset
        from src.datasets.mixed_dataset import create_mixed_dataset

        real_dataset = RealOghamDataset(
            data_dir=real_data_dir,
            split="train",
            processor=processor,
            tokenizer=tokenizer,
            mode=mode,
        )
        log.info(f"Real dataset: {len(real_dataset)} samples")

        # Create on-the-fly synthetic for curriculum variation
        from src.datasets.synthetic_dataset import create_synthetic_dataset
        synth_onthefly = create_synthetic_dataset(
            size=len(train_dataset),
            font_dir=font_dir,
            tokenizer=tokenizer,
            processor=processor,
            mode=mode,
            lazy=True,
        )

        train_dataset = create_mixed_dataset(
            real_dataset=real_dataset,
            synthetic_dataset=synth_onthefly,
            schedule_type="default",
            total_epochs=epochs,
        )
        log.info(f"Mixed dataset: {len(train_dataset)} samples with curriculum learning")

    # DataLoaders
    # Warn if using workers with Drive paths (FUSE can deadlock with multiprocessing)
    is_drive = data_dir and "/drive/" in data_dir
    if is_drive and not hf_dataset and num_workers > 2:
        log.warning(
            f"Reducing num_workers from {num_workers} to 2 for Drive compatibility. "
            f"Use local storage for faster I/O."
        )
        num_workers = 2

    collator = SimpleCollator()
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # AMP scaler (CUDA only)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Training loop — load existing history on resume so we don't lose prior epochs
    history_path = checkpoint_path / f"history_{mode}.json"
    history = {"train_loss": [], "val_loss": [], "val_cer": [], "val_exact_match": []}
    if resume and history_path.exists():
        try:
            with open(history_path) as f:
                history = json.load(f)
            log.info(f"Loaded history: {len(history['train_loss'])} prior epochs")
        except (json.JSONDecodeError, KeyError):
            log.warning("Could not load prior history, starting fresh")
    best_cer = min(history["val_cer"]) if history["val_cer"] else float("inf")

    log.info(f"\n{'='*60}")
    log.info(f"Training: {mode} mode, {epochs} epochs, batch {batch_size}, lr {lr}")
    log.info(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Freeze/unfreeze encoder
        if epoch < freeze_encoder_epochs:
            if epoch == 0:
                for param in model.encoder.parameters():
                    param.requires_grad = False
                log.info("Encoder frozen")
        elif epoch == freeze_encoder_epochs:
            for param in model.encoder.parameters():
                param.requires_grad = True
            log.info("Encoder unfrozen")

        # Update curriculum for mixed datasets
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
            config = train_dataset.get_current_config()
            log.info(f"Curriculum: synth_ratio={config['synthetic_ratio']:.0%}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        scheduler.step()

        # Evaluate
        val_metrics = evaluate(model, val_loader, tokenizer, device)

        epoch_time = time.time() - epoch_start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_cer"].append(val_metrics["cer"])
        history["val_exact_match"].append(val_metrics["exact_match"])

        log.info(
            f"Epoch {epoch+1}/{epochs} ({epoch_time:.0f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"CER: {val_metrics['cer']:.4f} | "
            f"Exact: {val_metrics['exact_match']:.1%}"
        )

        # Save history after every epoch (survives disconnects)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Save best checkpoint
        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            save_path = checkpoint_path / f"best_{mode}"
            model.save_pretrained(str(save_path))
            tokenizer.save_pretrained(str(save_path))
            log.info(f"  New best CER: {best_cer:.4f} -> saved to {save_path}")

    # Save final checkpoint
    final_path = checkpoint_path / f"final_{mode}"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    log.info(f"\nTraining complete. Best CER: {best_cer:.4f}")
    log.info(f"  Best checkpoint: {checkpoint_path / f'best_{mode}'}")
    log.info(f"  History: {history_path}")

    return {
        "mode": mode,
        "best_cer": best_cer,
        "epochs_trained": epochs,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Ogham OCR on Colab or locally"
    )

    # Mode and model
    parser.add_argument("--mode", choices=["ogham", "latin", "compare"], default="ogham",
                        help="Output mode")
    parser.add_argument("--model-size", choices=["small", "base"], default="small",
                        help="Model size (small=62M, base=334M)")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1,
                        help="Phase 1=synthetic-only, Phase 2=curriculum with real data")

    # Data
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to synthetic training data (single dir or sharded)")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="HF Hub dataset name (e.g. username/ogham-synthetic-200k)")
    parser.add_argument("--val-data-dir", type=str, default=None,
                        help="Path to pre-generated validation data")
    parser.add_argument("--real-data-dir", type=str, default=None,
                        help="Path to real stone data (Phase 2)")
    parser.add_argument("--font-dir", type=str, default="data/fonts",
                        help="Path to Ogham fonts (for on-the-fly generation)")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=5)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--init-strategy", choices=["random", "zero", "latin"],
                        default="latin")
    parser.add_argument("--num-workers", type=int, default=2)

    # Infrastructure
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory for saving checkpoints")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name suffix for checkpoint isolation (e.g. 'no_freeze')")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from best checkpoint if available")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, or cpu")

    args = parser.parse_args()

    # Resolve model name
    model_map = {
        "small": "microsoft/trocr-small-stage1",
        "base": "microsoft/trocr-base-stage1",
    }
    model_name = model_map[args.model_size]

    # Apply experiment suffix to checkpoint dir
    if args.experiment:
        args.checkpoint_dir = str(Path(args.checkpoint_dir) / args.experiment)
        log.info(f"Experiment: {args.experiment} -> checkpoints at {args.checkpoint_dir}")

    # Validate data source
    if not args.hf_dataset and not args.data_dir:
        parser.error("Either --data-dir or --hf-dataset is required")

    if args.mode == "compare":
        log.info("Running comparison: ogham vs latin\n")
        results = {}
        for mode in ["ogham", "latin"]:
            log.info(f"\n{'#'*60}")
            log.info(f"# Mode: {mode.upper()}")
            log.info(f"{'#'*60}\n")

            results[mode] = run_training(
                mode=mode,
                model_name=model_name,
                data_dir=args.data_dir,
                val_data_dir=args.val_data_dir,
                real_data_dir=args.real_data_dir,
                font_dir=args.font_dir,
                phase=args.phase,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                freeze_encoder_epochs=args.freeze_encoder_epochs,
                gradient_accumulation=args.gradient_accumulation,
                init_strategy=args.init_strategy,
                checkpoint_dir=args.checkpoint_dir,
                resume=args.resume,
                device_str=args.device,
                num_workers=args.num_workers,
                hf_dataset=args.hf_dataset,
            )

        # Print comparison
        log.info(f"\n{'='*60}")
        log.info("COMPARISON SUMMARY")
        log.info(f"{'='*60}")
        for mode, r in results.items():
            log.info(f"  {mode:8s} | Best CER: {r['best_cer']:.4f}")

        # Save comparison report
        report_path = Path(args.checkpoint_dir) / "comparison_report.json"
        with open(report_path, "w") as f:
            json.dump({m: {"best_cer": r["best_cer"], "epochs": r["epochs_trained"]}
                       for m, r in results.items()}, f, indent=2)
        log.info(f"Report: {report_path}")

    else:
        run_training(
            mode=args.mode,
            model_name=model_name,
            data_dir=args.data_dir,
            val_data_dir=args.val_data_dir,
            real_data_dir=args.real_data_dir,
            font_dir=args.font_dir,
            phase=args.phase,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            freeze_encoder_epochs=args.freeze_encoder_epochs,
            gradient_accumulation=args.gradient_accumulation,
            init_strategy=args.init_strategy,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            device_str=args.device,
            num_workers=args.num_workers,
            hf_dataset=args.hf_dataset,
        )


if __name__ == "__main__":
    main()
