#!/usr/bin/env python3
"""
Fine-tune TrOCR for Ogham OCR.

Supports two output modes:
  - "ogham":  Extend tokenizer, model outputs Ogham Unicode (e.g., ᚋᚐᚊᚔ)
  - "latin":  Use default tokenizer, model outputs Latin transliteration (e.g., MAQI)

Designed to start from trocr-base-stage1 or trocr-small-stage1 checkpoints
(pre-trained on synthetic data, before English fine-tuning).

Usage:
    # Quick overfit test on synthetic demo data
    python scripts/finetune_trocr.py \
        --mode ogham \
        --data-dir ogham_dataset/synthetic_demo \
        --epochs 10 --batch-size 4

    # Full training with synthetic dataset
    python scripts/finetune_trocr.py \
        --mode ogham \
        --data-dir ogham_dataset/synthetic_training \
        --epochs 50 --batch-size 16

    # Compare both modes on same data
    python scripts/finetune_trocr.py \
        --mode compare \
        --data-dir ogham_dataset/synthetic_demo \
        --epochs 20

★ Insight ─────────────────────────────────────
Training strategy:
1. Start from stage1 (synthetic pre-training, not English-tuned)
2. Freeze encoder for first N epochs (stable visual features)
3. Use curriculum learning: easy → medium → hard synthetic data
4. Gradually mix in real data as training progresses
5. Monitor domain gap: CER on synthetic vs CER on real
─────────────────────────────────────────────────
"""

import argparse
import csv
import json
import logging
import sys
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
# Dataset: loads pre-generated synthetic images + CSV labels
# =============================================================================

class PreGeneratedDataset:
    """
    PyTorch Dataset that loads pre-generated synthetic images with CSV labels.

    Supports both Ogham Unicode and Latin transliteration targets.
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
        from torch.utils.data import Dataset
        from PIL import Image

        self.processor = processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.torch = torch
        self.Image = Image

        # Load labels
        data_dir = Path(data_dir)
        labels_path = data_dir / "labels.csv"

        with open(labels_path) as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)

        # Split into train/val
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_rows))
        val_size = max(1, int(len(all_rows) * val_ratio))

        if split == "val":
            selected_indices = indices[:val_size]
        else:
            selected_indices = indices[val_size:]

        self.samples = []
        for idx in selected_indices:
            row = all_rows[idx]
            image_path = data_dir / row["image_file"]
            if not image_path.exists():
                image_path = data_dir / "images" / row["image_file"]
            if image_path.exists():
                self.samples.append({
                    "image_path": str(image_path),
                    "ogham_text": row["ogham_text"],
                    "latin_text": row["latin_transliteration"],
                })

        log.info(
            f"Loaded {len(self.samples)} {split} samples from {data_dir} "
            f"(mode: {mode})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = self.Image.open(sample["image_path"]).convert("RGB")

        # Process image for TrOCR
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Choose target text based on mode
        if self.mode == "ogham":
            text = sample["ogham_text"]
        else:
            text = sample["latin_text"]

        # Tokenize
        labels = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": text,
        }


class OghamCollator:
    """Collate batch and mask padding tokens for loss."""

    def __init__(self, pad_token_id: int):
        import torch
        self.pad_token_id = pad_token_id
        self.torch = torch

    def __call__(self, batch):
        pixel_values = self.torch.stack([b["pixel_values"] for b in batch])
        labels = self.torch.stack([b["labels"] for b in batch])
        labels = labels.clone()
        labels[labels == self.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


# =============================================================================
# Training loop
# =============================================================================

def train_one_epoch(
    model, dataloader, optimizer, scheduler, device, epoch, use_amp=False
) -> Dict[str, float]:
    """Train for one epoch, return metrics."""
    import torch
    from tqdm import tqdm

    model.train()
    total_loss = 0
    num_batches = 0

    amp_context = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    progress = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch in progress:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with amp_context:
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        progress.set_postfix(loss=f"{total_loss / num_batches:.4f}")

    return {"loss": total_loss / num_batches}


def evaluate(
    model, dataloader, tokenizer, device, max_length=64
) -> Dict[str, Any]:
    """Evaluate model: compute loss, CER, and sample predictions."""
    import torch
    import editdistance

    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Loss
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1

            # Generate
            generated_ids = model.generate(pixel_values, max_length=max_length)
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Decode references
            ref_ids = labels.clone()
            ref_ids[ref_ids == -100] = tokenizer.pad_token_id
            references = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

            all_predictions.extend(predictions)
            all_references.extend(references)

    # Compute CER
    total_edit_dist = 0
    total_ref_len = 0
    exact_matches = 0

    for pred, ref in zip(all_predictions, all_references):
        pred = pred.strip()
        ref = ref.strip()
        total_edit_dist += editdistance.eval(pred, ref)
        total_ref_len += max(len(ref), 1)
        if pred == ref:
            exact_matches += 1

    cer = total_edit_dist / total_ref_len if total_ref_len > 0 else 1.0
    exact_match_rate = exact_matches / len(all_predictions) if all_predictions else 0.0

    # Sample predictions for logging
    sample_preds = []
    for i in range(min(5, len(all_predictions))):
        sample_preds.append({
            "reference": all_references[i].strip(),
            "prediction": all_predictions[i].strip(),
            "match": all_predictions[i].strip() == all_references[i].strip(),
        })

    return {
        "loss": total_loss / max(num_batches, 1),
        "cer": cer,
        "exact_match": exact_match_rate,
        "num_samples": len(all_predictions),
        "sample_predictions": sample_preds,
    }


def run_training(
    mode: str,
    model_name: str,
    data_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    freeze_encoder_epochs: int,
    device_str: str,
    output_dir: str,
    eval_every: int = 1,
) -> Dict[str, Any]:
    """Run complete training pipeline for one mode."""
    import torch
    from torch.utils.data import DataLoader

    # Resolve device
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    device = torch.device(device_str)
    log.info(f"Device: {device}")

    # Setup model
    if mode == "ogham":
        from src.training.tokenizer_extension import setup_ogham_model_and_tokenizer
        model, processor, tokenizer = setup_ogham_model_and_tokenizer(model_name)
    else:
        from src.training.tokenizer_extension import setup_transliteration_model
        model, processor, tokenizer = setup_transliteration_model(model_name)

    model = model.to(device)

    # Create datasets
    train_dataset = PreGeneratedDataset(
        data_dir, processor, tokenizer, mode=mode, split="train"
    )
    val_dataset = PreGeneratedDataset(
        data_dir, processor, tokenizer, mode=mode, split="val"
    )

    collator = OghamCollator(tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=0,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.1
    )

    # Freeze encoder initially
    if freeze_encoder_epochs > 0:
        log.info(f"Freezing encoder for first {freeze_encoder_epochs} epochs")
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_cer": [], "val_exact_match": []}
    best_cer = float("inf")

    for epoch in range(1, epochs + 1):
        # Unfreeze encoder when freeze period ends
        if epoch == freeze_encoder_epochs + 1:
            log.info("Unfreezing encoder — full model training")
            for param in model.encoder.parameters():
                param.requires_grad = True

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        history["train_loss"].append(train_metrics["loss"])

        # Evaluate
        if epoch % eval_every == 0 or epoch == epochs:
            val_metrics = evaluate(model, val_loader, tokenizer, device)
            history["val_loss"].append(val_metrics["loss"])
            history["val_cer"].append(val_metrics["cer"])
            history["val_exact_match"].append(val_metrics["exact_match"])

            log.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val CER: {val_metrics['cer']:.4f} | "
                f"Val Exact: {val_metrics['exact_match']:.1%}"
            )

            # Show sample predictions
            for sp in val_metrics["sample_predictions"][:3]:
                mark = "✓" if sp["match"] else "✗"
                log.info(f"  {mark} ref='{sp['reference']}' pred='{sp['prediction']}'")

            # Save best model
            if val_metrics["cer"] < best_cer:
                best_cer = val_metrics["cer"]
                save_dir = Path(output_dir) / f"best_{mode}"
                save_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                log.info(f"  New best CER: {best_cer:.4f} — saved to {save_dir}")

    # Save final model
    save_dir = Path(output_dir) / f"final_{mode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save history
    history_path = Path(output_dir) / f"history_{mode}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return {
        "mode": mode,
        "best_cer": best_cer,
        "final_train_loss": history["train_loss"][-1],
        "final_val_cer": history["val_cer"][-1] if history["val_cer"] else None,
        "final_val_exact_match": history["val_exact_match"][-1] if history["val_exact_match"] else None,
        "history": history,
    }


# =============================================================================
# Comparison report
# =============================================================================

def generate_comparison_report(results: Dict[str, Dict], output_dir: str):
    """Generate a comparison report between ogham and latin modes."""
    report_lines = [
        "=" * 70,
        "OGHAM OCR: Unicode vs Transliteration Comparison Report",
        "=" * 70,
        "",
    ]

    for mode, r in results.items():
        report_lines.extend([
            f"Mode: {mode.upper()}",
            f"  Best CER:           {r['best_cer']:.4f}",
            f"  Final Train Loss:   {r['final_train_loss']:.4f}",
            f"  Final Val CER:      {r.get('final_val_cer', 'N/A')}",
            f"  Final Val Exact:    {r.get('final_val_exact_match', 'N/A')}",
            "",
        ])

    # Analysis
    if "ogham" in results and "latin" in results:
        ogham_cer = results["ogham"]["best_cer"]
        latin_cer = results["latin"]["best_cer"]

        report_lines.extend([
            "-" * 70,
            "ANALYSIS",
            "-" * 70,
            f"CER difference: {abs(ogham_cer - latin_cer):.4f}",
        ])

        if latin_cer < ogham_cer:
            report_lines.append(
                f"Latin transliteration achieved {((ogham_cer - latin_cer) / ogham_cer * 100):.1f}% lower CER"
            )
        elif ogham_cer < latin_cer:
            report_lines.append(
                f"Ogham Unicode achieved {((latin_cer - ogham_cer) / latin_cer * 100):.1f}% lower CER"
            )
        else:
            report_lines.append("Both modes achieved identical CER")

        report_lines.extend([
            "",
            "NOTE: This comparison uses identical training data and hyperparameters.",
            "The difference reflects only the impact of output representation.",
            "",
        ])

    report = "\n".join(report_lines)
    log.info(f"\n{report}")

    report_path = Path(output_dir) / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"Report saved to {report_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR for Ogham OCR")
    parser.add_argument(
        "--mode", choices=["ogham", "latin", "compare"], default="ogham",
        help="'ogham' for Unicode output, 'latin' for transliteration, 'compare' for both"
    )
    parser.add_argument(
        "--model", default="microsoft/trocr-base-stage1",
        help="HuggingFace model checkpoint"
    )
    parser.add_argument(
        "--data-dir", default="ogham_dataset/synthetic_demo",
        help="Directory with images + labels.csv"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir", default="checkpoints",
        help="Directory to save models and reports"
    )
    parser.add_argument("--eval-every", type=int, default=1)

    args = parser.parse_args()

    if args.mode == "compare":
        results = {}
        for mode in ["ogham", "latin"]:
            log.info(f"\n{'#' * 60}")
            log.info(f"# Training: {mode.upper()} mode")
            log.info(f"{'#' * 60}\n")

            results[mode] = run_training(
                mode=mode,
                model_name=args.model,
                data_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                freeze_encoder_epochs=args.freeze_encoder_epochs,
                device_str=args.device,
                output_dir=args.output_dir,
                eval_every=args.eval_every,
            )

        generate_comparison_report(results, args.output_dir)
    else:
        run_training(
            mode=args.mode,
            model_name=args.model,
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            freeze_encoder_epochs=args.freeze_encoder_epochs,
            device_str=args.device,
            output_dir=args.output_dir,
            eval_every=args.eval_every,
        )


if __name__ == "__main__":
    main()
