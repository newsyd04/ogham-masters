"""Generate synthetic Ogham datasets for training.

Renders synthetic Ogham inscription images using the sequence sampler
and renderer, with stone texture and weathering augmentation for realism.

Supports sharded output for large datasets and JPEG format for storage efficiency.

Requires: numpy, Pillow, opencv-python-headless
No torch/transformers needed.

Usage:
    # Small demo dataset (100 PNG images)
    python scripts/generate_demo_dataset.py --n 100 --difficulty mixed --seed 42

    # Large sharded training set (200k JPEG images in 10 shards)
    python scripts/generate_demo_dataset.py \
        --n 200000 --format jpeg --shards 10 --workers 4 \
        --difficulty mixed --realism medium --seed 42 \
        --output-dir ogham_dataset/synthetic_200k

    # Validation set with different seed
    python scripts/generate_demo_dataset.py \
        --n 5000 --format jpeg --seed 99999 \
        --output-dir ogham_dataset/synthetic_val

    # Clean images for debugging
    python scripts/generate_demo_dataset.py --n 50 --realism clean

Output structure (single shard / no sharding):
    output_dir/
    ├── images/
    │   ├── synth_000000.png
    │   └── ...
    ├── labels.csv
    └── summary.json

Output structure (sharded):
    output_dir/
    ├── shard_00/
    │   ├── images/
    │   ├── labels.csv
    │   └── summary.json
    ├── shard_01/
    │   └── ...
    └── generation_config.json
"""

import argparse
import csv
import json
import multiprocessing
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image

from src.generation.renderer import create_renderer
from src.generation.sequence_sampler import DifficultyAwareSequenceSampler
from src.utils.ogham import OGHAM_TO_LATIN


def transliterate(ogham_text: str) -> str:
    """Convert Ogham Unicode to Latin transliteration."""
    return "".join(OGHAM_TO_LATIN.get(ch, "?") for ch in ogham_text)


class StoneWeathering:
    """Apply realistic stone weathering effects to rendered Ogham images.

    Simulates the visual characteristics of real stone inscriptions:
    surface texture, weathering erosion, lichen, uneven lighting, and
    camera-like blur/noise.
    """

    def __init__(self, severity: str = "medium", seed: int = 42):
        self.severity = severity
        self.rng = np.random.default_rng(seed)

        # Severity presets
        self.configs = {
            "clean": {},  # No augmentation
            "light": {
                "texture_intensity": 0.10,
                "noise_std": 8,
                "brightness_delta": 0.08,
                "blur_prob": 0.2,
                "shadow_prob": 0.2,
                "lichen_prob": 0.1,
                "erosion_prob": 0.1,
            },
            "medium": {
                "texture_intensity": 0.20,
                "noise_std": 18,
                "brightness_delta": 0.15,
                "blur_prob": 0.4,
                "shadow_prob": 0.5,
                "lichen_prob": 0.3,
                "erosion_prob": 0.3,
            },
            "heavy": {
                "texture_intensity": 0.30,
                "noise_std": 30,
                "brightness_delta": 0.25,
                "blur_prob": 0.6,
                "shadow_prob": 0.7,
                "lichen_prob": 0.5,
                "erosion_prob": 0.5,
            },
        }

    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.severity == "clean":
            return image

        cfg = self.configs[self.severity]
        result = image.copy()

        # 1. Stone surface texture (multi-scale noise)
        result = self._add_stone_texture(result, cfg["texture_intensity"])

        # 2. Uneven lighting / shadow bands
        if self.rng.random() < cfg["shadow_prob"]:
            result = self._add_shadows(result)

        # 3. Lichen/moss patches
        if self.rng.random() < cfg["lichen_prob"]:
            result = self._add_lichen(result)

        # 4. Erosion (fade inscription strokes)
        if self.rng.random() < cfg["erosion_prob"]:
            result = self._add_erosion(result)

        # 5. Gaussian noise (camera sensor)
        result = self._add_noise(result, cfg["noise_std"])

        # 6. Brightness/contrast shift
        result = self._adjust_brightness(result, cfg["brightness_delta"])

        # 7. Slight blur (focus variation)
        if self.rng.random() < cfg["blur_prob"]:
            k = self.rng.choice([3, 5])
            result = cv2.GaussianBlur(result, (k, k), 0)

        return result

    def _add_stone_texture(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Multi-scale Perlin-like noise simulating stone grain."""
        h, w = image.shape[:2]
        noise = np.zeros((h, w), dtype=np.float32)

        for scale in [8, 16, 32, 64]:
            rows = h // scale + 2
            cols = w // scale + 2
            noise_low = self.rng.random((rows, cols)).astype(np.float32)
            noise_scaled = cv2.resize(noise_low, (w, h), interpolation=cv2.INTER_CUBIC)
            noise += noise_scaled

        # Normalize and apply
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        noise = ((noise - 0.5) * 255 * intensity).astype(np.int32)

        result = image.astype(np.int32)
        for c in range(3):
            result[:, :, c] += noise
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_shadows(self, image: np.ndarray) -> np.ndarray:
        """Simulate uneven outdoor lighting with gradient shadows."""
        h, w = image.shape[:2]
        shadow = np.ones((h, w), dtype=np.float32)

        n_bands = self.rng.integers(1, 4)
        for _ in range(n_bands):
            # Random diagonal gradient
            angle = self.rng.uniform(0, np.pi)
            cx = self.rng.uniform(0.2, 0.8) * w
            cy = self.rng.uniform(0.2, 0.8) * h
            spread = self.rng.uniform(0.2, 0.6) * max(h, w)
            darkness = self.rng.uniform(0.15, 0.4)

            yy, xx = np.mgrid[:h, :w]
            dist = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
            band = np.exp(-0.5 * (dist / spread) ** 2)
            shadow -= band * darkness

        shadow = np.clip(shadow, 0.4, 1.0)
        result = image.astype(np.float32)
        for c in range(3):
            result[:, :, c] *= shadow
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_lichen(self, image: np.ndarray) -> np.ndarray:
        """Add lichen/moss patches (greenish-yellow blobs)."""
        h, w = image.shape[:2]
        result = image.copy()

        n_patches = self.rng.integers(2, 8)
        for _ in range(n_patches):
            cx = self.rng.integers(0, w)
            cy = self.rng.integers(0, h)
            radius = self.rng.integers(5, max(6, min(h, w) // 8))
            opacity = self.rng.uniform(0.3, 0.7)

            # Lichen color: greens, yellows, pale gray-green
            color = self.rng.choice([
                [90, 120, 70],    # Green
                [130, 140, 80],   # Yellow-green
                [160, 165, 140],  # Pale lichen
                [80, 100, 60],    # Dark moss
                [170, 170, 120],  # Yellow
            ])

            # Create soft blob mask
            yy, xx = np.mgrid[:h, :w]
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            mask = np.clip(1.0 - dist / radius, 0, 1) ** 2
            # Add irregularity
            noise = cv2.resize(
                self.rng.random((h // 8 + 1, w // 8 + 1)).astype(np.float32),
                (w, h), interpolation=cv2.INTER_CUBIC,
            )
            mask *= (noise > 0.4).astype(np.float32)

            for c in range(3):
                result[:, :, c] = (
                    result[:, :, c] * (1 - mask * opacity)
                    + color[c] * mask * opacity
                ).astype(np.uint8)

        return result

    def _add_erosion(self, image: np.ndarray) -> np.ndarray:
        """Fade inscription strokes to simulate weathering erosion."""
        # Reduce contrast: push everything toward the mean
        mean_val = image.mean()
        factor = self.rng.uniform(0.6, 0.85)
        result = image.astype(np.float32)
        result = mean_val + (result - mean_val) * factor
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_noise(self, image: np.ndarray, std: float) -> np.ndarray:
        """Gaussian sensor noise."""
        actual_std = self.rng.uniform(std * 0.5, std)
        noise = self.rng.normal(0, actual_std, image.shape)
        result = image.astype(np.float32) + noise
        return np.clip(result, 0, 255).astype(np.uint8)

    def _adjust_brightness(self, image: np.ndarray, delta: float) -> np.ndarray:
        """Random brightness and contrast shift."""
        brightness = self.rng.uniform(-delta, delta)
        contrast = self.rng.uniform(1 - delta, 1 + delta)
        result = image.astype(np.float32)
        result = 128 + (result - 128) * contrast + brightness * 255
        return np.clip(result, 0, 255).astype(np.uint8)


def generate_chunk(args_tuple):
    """Generate a chunk of images (for multiprocessing).

    Each worker creates its own renderer/sampler/weathering instances
    to avoid sharing state across processes.
    """
    (chunk_indices, schedule_chunk, base_seed, font_dir, image_height,
     with_texture, realism, img_format, jpeg_quality, output_dir) = args_tuple

    # Create per-worker renderer and weathering
    renderer = create_renderer(
        font_dir=font_dir,
        image_height=image_height,
        with_texture=with_texture,
        seed=base_seed,
    )
    weathering = StoneWeathering(severity=realism, seed=base_seed)

    # Create samplers for each difficulty in this chunk
    difficulties_needed = set(schedule_chunk)
    samplers = {
        d: DifficultyAwareSequenceSampler(difficulty=d, seed=base_seed)
        for d in difficulties_needed
    }

    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for local_idx, (global_idx, difficulty) in enumerate(zip(chunk_indices, schedule_chunk)):
        sample_seed = base_seed + global_idx
        samplers[difficulty].set_seed(sample_seed)
        renderer.set_seed(sample_seed)
        weathering.set_seed(sample_seed + 10000)

        ogham_text = samplers[difficulty].sample()
        img_array, render_info = renderer.render(ogham_text)
        img_array = weathering(img_array)

        # Save image
        ext = "jpg" if img_format == "jpeg" else "png"
        filename = f"synth_{global_idx:06d}.{ext}"
        img = Image.fromarray(img_array)
        if img_format == "jpeg":
            img.save(images_dir / filename, quality=jpeg_quality)
        else:
            img.save(images_dir / filename)

        latin = transliterate(ogham_text)
        rows.append({
            "image_file": filename,
            "ogham_text": ogham_text,
            "latin_transliteration": latin,
            "difficulty": difficulty,
            "realism": realism,
            "text_length": len(ogham_text),
            "image_width": img_array.shape[1],
            "font_size": render_info["font_size"],
        })

    return rows


def generate_shard(shard_idx, indices, schedule, args, shard_dir):
    """Generate one shard of the dataset."""
    start = time.time()

    shard_dir = Path(shard_dir)
    n_images = len(indices)

    if args.workers > 1:
        # Split into chunks for multiprocessing
        chunk_size = max(1, n_images // args.workers)
        chunks = []
        for i in range(0, n_images, chunk_size):
            chunk_indices = indices[i:i + chunk_size]
            chunk_schedule = schedule[i:i + chunk_size]
            chunks.append((
                chunk_indices, chunk_schedule, args.seed, args.font_dir,
                args.image_height, args.texture, args.realism,
                args.format, args.jpeg_quality, str(shard_dir),
            ))

        with multiprocessing.Pool(processes=args.workers) as pool:
            chunk_results = pool.map(generate_chunk, chunks)

        # Flatten results
        all_rows = []
        for rows in chunk_results:
            all_rows.extend(rows)
    else:
        # Single-threaded generation
        all_rows = generate_chunk((
            indices, schedule, args.seed, args.font_dir,
            args.image_height, args.texture, args.realism,
            args.format, args.jpeg_quality, str(shard_dir),
        ))

    # Sort by global index for deterministic CSV ordering
    all_rows.sort(key=lambda r: r["image_file"])

    # Write labels CSV
    csv_path = shard_dir / "labels.csv"
    fieldnames = ["image_file", "ogham_text", "latin_transliteration",
                  "difficulty", "realism", "text_length", "image_width", "font_size"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    elapsed = time.time() - start

    # Difficulty counts
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    total_chars = 0
    for row in all_rows:
        difficulty_counts[row["difficulty"]] = difficulty_counts.get(row["difficulty"], 0) + 1
        total_chars += row["text_length"]

    # Write summary
    summary = {
        "shard_index": shard_idx,
        "total_images": n_images,
        "difficulty_distribution": difficulty_counts,
        "realism": args.realism,
        "format": args.format,
        "texture_renderer": args.texture,
        "avg_text_length": round(total_chars / max(1, n_images), 1),
        "image_height": args.image_height,
        "seed": args.seed,
        "index_range": [int(indices[0]), int(indices[-1])],
        "elapsed_seconds": round(elapsed, 2),
        "workers": args.workers,
    }
    with open(shard_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Ogham dataset")
    parser.add_argument("--n", type=int, default=100, help="Number of images to generate")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed",
                        help="Sequence difficulty (mixed = equal split of all three)")
    parser.add_argument("--realism", choices=["clean", "light", "medium", "heavy"], default="medium",
                        help="Visual realism level (clean = no augmentation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./ogham_dataset/synthetic_demo",
                        help="Output directory")
    parser.add_argument("--font-dir", type=str, default="./data/fonts",
                        help="Directory containing Ogham fonts")
    parser.add_argument("--image-height", type=int, default=384,
                        help="Image height in pixels")
    parser.add_argument("--texture", action="store_true", default=True,
                        help="Use stone texture renderer (default: on)")
    parser.add_argument("--no-texture", dest="texture", action="store_false",
                        help="Disable stone texture renderer")
    parser.add_argument("--format", choices=["png", "jpeg"], default="png",
                        help="Image format (jpeg is ~4x smaller)")
    parser.add_argument("--jpeg-quality", type=int, default=85,
                        help="JPEG quality (1-100, default 85)")
    parser.add_argument("--shards", type=int, default=1,
                        help="Number of shards (1 = no sharding)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for generation")
    args = parser.parse_args()

    start = time.time()
    output_dir = Path(args.output_dir)

    # Build full difficulty schedule
    if args.difficulty == "mixed":
        difficulties = ["easy", "medium", "hard"]
        schedule = [difficulties[i % 3] for i in range(args.n)]
    else:
        schedule = [args.difficulty] * args.n

    all_indices = list(range(args.n))

    print(f"Generating {args.n} synthetic images ({args.format.upper()}, "
          f"{args.shards} shard(s), {args.workers} worker(s))")
    print(f"  Output: {output_dir}")
    print(f"  Difficulty: {args.difficulty}, Realism: {args.realism}")

    if args.shards <= 1:
        # No sharding -- same as original behavior
        summary = generate_shard(0, all_indices, schedule, args, output_dir)
        elapsed = time.time() - start
        print(f"\nDone! Generated {args.n} images in {elapsed:.1f}s")
        dc = summary["difficulty_distribution"]
        print(f"  Difficulty -- Easy: {dc.get('easy', 0)}, Medium: {dc.get('medium', 0)}, Hard: {dc.get('hard', 0)}")
        print(f"  Avg text length: {summary['avg_text_length']} chars")
    else:
        # Sharded output
        shard_size = args.n // args.shards
        remainder = args.n % args.shards

        shard_summaries = []
        for shard_idx in range(args.shards):
            # Distribute remainder across first shards
            shard_start = shard_idx * shard_size + min(shard_idx, remainder)
            extra = 1 if shard_idx < remainder else 0
            shard_end = shard_start + shard_size + extra

            shard_indices = all_indices[shard_start:shard_end]
            shard_schedule = schedule[shard_start:shard_end]
            shard_dir = output_dir / f"shard_{shard_idx:02d}"

            print(f"\n  Shard {shard_idx:02d}: {len(shard_indices)} images "
                  f"(indices {shard_start}-{shard_end - 1})")

            summary = generate_shard(
                shard_idx, shard_indices, shard_schedule, args, shard_dir
            )
            shard_summaries.append(summary)
            print(f"    Done in {summary['elapsed_seconds']:.1f}s")

        # Write top-level generation config
        elapsed = time.time() - start
        gen_config = {
            "total_images": args.n,
            "num_shards": args.shards,
            "shard_size": shard_size,
            "format": args.format,
            "difficulty": args.difficulty,
            "realism": args.realism,
            "seed": args.seed,
            "image_height": args.image_height,
            "workers": args.workers,
            "total_elapsed_seconds": round(elapsed, 2),
            "shards": [
                {"shard_dir": f"shard_{s['shard_index']:02d}", **s}
                for s in shard_summaries
            ],
        }
        with open(output_dir / "generation_config.json", "w") as f:
            json.dump(gen_config, f, indent=2)

        print(f"\nDone! Generated {args.n} images across {args.shards} shards in {elapsed:.1f}s")
        print(f"  Config: {output_dir / 'generation_config.json'}")


if __name__ == "__main__":
    main()
