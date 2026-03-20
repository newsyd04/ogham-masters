#!/usr/bin/env python3
"""
Overfit-one-sample litmus test for Ogham OCR.

This script verifies end-to-end that TrOCR can learn to output Ogham Unicode
characters by overfitting on a single synthetic image. If the model can
perfectly reproduce one sample, the tokenizer extension and training pipeline
are working correctly.

Usage:
    # Ogham Unicode output (extended tokenizer):
    python scripts/overfit_one_sample.py --mode ogham

    # Latin transliteration output (default tokenizer):
    python scripts/overfit_one_sample.py --mode latin

    # Use small model for faster iteration:
    python scripts/overfit_one_sample.py --mode ogham --model microsoft/trocr-small-stage1

    # Custom sample from the synthetic dataset:
    python scripts/overfit_one_sample.py --mode ogham --sample-idx 42

★ Insight ─────────────────────────────────────
Why overfit one sample first?
1. It's the fastest way to verify the pipeline works end-to-end
2. If the model can't memorise ONE sample, something is fundamentally broken
   (tokenizer, loss computation, generation config, etc.)
3. Expected behaviour: loss should drop to ~0 within 50-200 steps
4. If loss plateaus above 0, check tokenizer encoding of the target text
─────────────────────────────────────────────────
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_sample_image(dataset_dir: Path, sample_idx: int = 0):
    """Load a single synthetic sample (image + label)."""
    import csv
    from PIL import Image

    labels_path = dataset_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}. "
            f"Run scripts/generate_demo_dataset.py first."
        )

    with open(labels_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if sample_idx >= len(rows):
        raise IndexError(
            f"Sample index {sample_idx} out of range (dataset has {len(rows)} samples)"
        )

    row = rows[sample_idx]
    image_path = dataset_dir / row["image_file"]
    # Also check images/ subdirectory (common layout)
    if not image_path.exists():
        image_path = dataset_dir / "images" / row["image_file"]
    ogham_text = row["ogham_text"]
    latin_text = row["latin_transliteration"]

    image = Image.open(image_path).convert("RGB")

    log.info(f"Loaded sample {sample_idx}:")
    log.info(f"  Image: {image_path} ({image.size[0]}x{image.size[1]})")
    log.info(f"  Ogham: {ogham_text}")
    log.info(f"  Latin: {latin_text}")

    return image, ogham_text, latin_text


def run_overfit_test(
    mode: str = "ogham",
    model_name: str = "microsoft/trocr-base-stage1",
    sample_idx: int = 0,
    num_steps: int = 200,
    lr: float = 5e-5,
    eval_every: int = 10,
    device: str = "auto",
    init_strategy: str = "latin",
):
    """
    Run the overfit-one-sample test.

    Args:
        mode: "ogham" for Unicode output, "latin" for transliteration
        model_name: HuggingFace model checkpoint
        sample_idx: Which synthetic sample to use
        num_steps: Number of training steps
        lr: Learning rate
        eval_every: Evaluate (generate) every N steps
        device: "auto", "cuda", "mps", or "cpu"
    """
    import torch

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    log.info(f"Using device: {device}")

    # Load sample
    dataset_dir = PROJECT_ROOT / "ogham_dataset" / "synthetic_demo"
    image, ogham_text, latin_text = load_sample_image(dataset_dir, sample_idx)

    # Choose target text based on mode
    target_text = ogham_text if mode == "ogham" else latin_text
    log.info(f"\nMode: {mode}")
    log.info(f"Target text: {target_text}")
    log.info(f"Target length: {len(target_text)} characters")

    # Setup model and tokenizer
    if mode == "ogham":
        from src.training.tokenizer_extension import setup_ogham_model_and_tokenizer
        model, processor, tokenizer = setup_ogham_model_and_tokenizer(
            model_name, init_strategy=init_strategy
        )
        log.info(f"Embedding init strategy: {init_strategy}")
    else:
        from src.training.tokenizer_extension import setup_transliteration_model
        model, processor, tokenizer = setup_transliteration_model(model_name)

    model = model.to(device)
    model.train()

    # Verify tokenization of target text
    encoded = tokenizer.encode(target_text, add_special_tokens=False)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    log.info(f"\nTokenization check:")
    log.info(f"  Encoded token IDs: {encoded}")
    log.info(f"  Num tokens: {len(encoded)}")
    log.info(f"  Decoded back: '{decoded}'")
    log.info(f"  Roundtrip match: {decoded.strip() == target_text}")

    if decoded.strip() != target_text:
        log.warning(
            "WARNING: Tokenizer roundtrip failed! "
            "The model may not be able to perfectly reproduce the target."
        )

    # Prepare inputs
    pixel_values = processor(
        images=image,
        return_tensors="pt",
    ).pixel_values.to(device)

    labels = tokenizer(
        target_text,
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    # Replace padding with -100 for loss computation
    labels[labels == tokenizer.pad_token_id] = -100

    log.info(f"\nInput shape: {pixel_values.shape}")
    log.info(f"Labels shape: {labels.shape}")
    log.info(f"Non-padding labels: {(labels != -100).sum().item()}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Use appropriate autocast based on device
    use_amp = device.type == "cuda"
    amp_context = torch.amp.autocast("cuda") if use_amp else torch.amp.autocast("cpu", enabled=False)

    # Training loop
    log.info(f"\n{'='*60}")
    log.info(f"Starting overfit test: {num_steps} steps, lr={lr}")
    log.info(f"{'='*60}\n")

    best_prediction = ""
    best_loss = float("inf")

    for step in range(1, num_steps + 1):
        model.train()
        optimizer.zero_grad()

        with amp_context:
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss

        # Periodic evaluation
        if step % eval_every == 0 or step == 1 or step == num_steps:
            model.eval()
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values,
                    max_length=64,
                    num_beams=4,
                )
                prediction = tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )

            # Character-level accuracy
            correct_chars = sum(
                1 for a, b in zip(prediction, target_text) if a == b
            )
            total_chars = max(len(prediction), len(target_text))
            char_acc = correct_chars / total_chars if total_chars > 0 else 0.0

            exact_match = prediction.strip() == target_text
            if exact_match:
                best_prediction = prediction

            status = "EXACT MATCH!" if exact_match else ""
            log.info(
                f"Step {step:4d} | Loss: {current_loss:.6f} | "
                f"Char Acc: {char_acc:.1%} | "
                f"Pred: '{prediction}' | {status}"
            )

            if exact_match:
                log.info(f"\n{'='*60}")
                log.info(f"SUCCESS: Model perfectly reproduced target at step {step}")
                log.info(f"  Target: '{target_text}'")
                log.info(f"  Output: '{prediction}'")
                log.info(f"  Loss:    {current_loss:.6f}")
                log.info(f"{'='*60}")
                break

            model.train()

    # Final summary
    log.info(f"\n{'='*60}")
    log.info(f"OVERFIT TEST SUMMARY ({mode} mode)")
    log.info(f"{'='*60}")
    log.info(f"Model:       {model_name}")
    log.info(f"Target:      '{target_text}'")
    log.info(f"Best loss:   {best_loss:.6f}")
    log.info(f"Final pred:  '{prediction}'")
    log.info(f"Exact match: {prediction.strip() == target_text}")

    if prediction.strip() != target_text:
        log.info(f"\nDiagnostics:")
        log.info(f"  Target chars: {[c for c in target_text]}")
        log.info(f"  Pred chars:   {[c for c in prediction]}")

        # Show character-by-character diff
        log.info(f"\n  Character diff:")
        max_len = max(len(target_text), len(prediction))
        for i in range(max_len):
            t = target_text[i] if i < len(target_text) else "∅"
            p = prediction[i] if i < len(prediction) else "∅"
            match = "✓" if t == p else "✗"
            log.info(f"    [{i:2d}] target='{t}' pred='{p}' {match}")

    return {
        "mode": mode,
        "model_name": model_name,
        "target": target_text,
        "prediction": prediction.strip(),
        "exact_match": prediction.strip() == target_text,
        "best_loss": best_loss,
        "steps": step,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Overfit TrOCR on one Ogham sample as a litmus test"
    )
    parser.add_argument(
        "--mode",
        choices=["ogham", "latin"],
        default="ogham",
        help="Output mode: 'ogham' for Unicode, 'latin' for transliteration",
    )
    parser.add_argument(
        "--model",
        default="microsoft/trocr-base-stage1",
        help="HuggingFace model checkpoint (default: trocr-base-stage1)",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Index of synthetic sample to use (default: 0)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of training steps (default: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Generate prediction every N steps (default: 10)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cuda, mps, or cpu (default: auto)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both modes and compare results",
    )
    parser.add_argument(
        "--init-strategy",
        choices=["random", "zero", "latin"],
        default="latin",
        help="Embedding init for Ogham tokens: random, zero, or latin (default: latin)",
    )
    parser.add_argument(
        "--compare-init",
        action="store_true",
        help="Compare all three init strategies (random, zero, latin) in ogham mode",
    )

    args = parser.parse_args()

    if args.compare_init:
        log.info("Comparing embedding init strategies: random vs zero vs latin\n")

        results = {}
        for strategy in ["random", "zero", "latin"]:
            log.info(f"\n{'#'*60}")
            log.info(f"# Init strategy: {strategy.upper()}")
            log.info(f"{'#'*60}\n")

            results[strategy] = run_overfit_test(
                mode="ogham",
                model_name=args.model,
                sample_idx=args.sample_idx,
                num_steps=args.steps,
                lr=args.lr,
                eval_every=args.eval_every,
                device=args.device,
                init_strategy=strategy,
            )

        log.info(f"\n{'='*60}")
        log.info(f"INIT STRATEGY COMPARISON")
        log.info(f"{'='*60}")
        for strategy, r in results.items():
            log.info(
                f"  {strategy:8s} | Match: {r['exact_match']} | "
                f"Loss: {r['best_loss']:.6f} | Steps: {r['steps']}"
            )

    elif args.compare:
        log.info("Running comparison: Ogham Unicode vs Latin transliteration\n")

        results = {}
        for mode in ["ogham", "latin"]:
            log.info(f"\n{'#'*60}")
            log.info(f"# Running {mode.upper()} mode")
            log.info(f"{'#'*60}\n")

            results[mode] = run_overfit_test(
                mode=mode,
                model_name=args.model,
                sample_idx=args.sample_idx,
                num_steps=args.steps,
                lr=args.lr,
                eval_every=args.eval_every,
                device=args.device,
                init_strategy=args.init_strategy,
            )

        # Comparison summary
        log.info(f"\n{'='*60}")
        log.info(f"COMPARISON SUMMARY")
        log.info(f"{'='*60}")
        for mode, r in results.items():
            log.info(
                f"  {mode:8s} | Match: {r['exact_match']} | "
                f"Loss: {r['best_loss']:.6f} | Steps: {r['steps']}"
            )

    else:
        run_overfit_test(
            mode=args.mode,
            model_name=args.model,
            sample_idx=args.sample_idx,
            num_steps=args.steps,
            lr=args.lr,
            eval_every=args.eval_every,
            device=args.device,
            init_strategy=args.init_strategy,
        )


if __name__ == "__main__":
    main()
