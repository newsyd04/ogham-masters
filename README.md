# Ogham OCR Dataset Pipeline

A complete dataset pipeline for fine-tuning TrOCR on Ogham inscription recognition.

## Overview

This project provides tools and infrastructure for building an OCR system specialized in reading ancient Ogham inscriptions from stone photographs. Given the limited availability of real data (~400 surviving stones), the pipeline combines:

1. **Real Image Collection**: Ethical scraping from academic and heritage sources
2. **Synthetic Data Generation**: On-the-fly procedural generation
3. **Curriculum Learning**: Progressive training from synthetic to real data
4. **TrOCR Fine-tuning**: Transfer learning from pre-trained OCR models

## Project Structure

```
ogham_ocr/
├── src/
│   ├── datasets/          # PyTorch dataset classes
│   ├── generation/        # Synthetic data generation
│   ├── preprocessing/     # Image preprocessing pipeline
│   ├── scrapers/          # Web scrapers for image collection
│   ├── training/          # Training infrastructure
│   ├── evaluation/        # Metrics and analysis
│   └── utils/             # Ogham character utilities
├── annotation_tool/       # Streamlit annotation interface
├── notebooks/             # Jupyter/Colab notebooks
├── configs/               # Configuration files
└── tests/                 # Unit tests
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### 1. Data Collection

```python
from ogham_ocr.src.scrapers import CISPScraper, ScraperConfig

config = ScraperConfig(output_dir="./ogham_dataset")
scraper = CISPScraper(config)
results = scraper.download_all(max_stones=50)
```

### 2. Annotation

```bash
streamlit run annotation_tool/app.py
```

### 3. Create Data Splits

```python
from ogham_ocr.src.datasets import create_splits

create_splits(
    data_dir="./ogham_dataset",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify_by="region"
)
```

### 4. Training

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ogham_ocr.src.datasets import RealOghamDataset, SyntheticOghamDataset, MixedOghamDataset
from ogham_ocr.src.training import OghamTrainer, TrainingConfig

# Load model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Create datasets
real_dataset = RealOghamDataset(data_dir, "train", processor)
synthetic_dataset = SyntheticOghamDataset(size=50000, font_paths, processor.tokenizer)
train_dataset = MixedOghamDataset(real_dataset, synthetic_dataset)

# Train
config = TrainingConfig(experiment_name="ogham_v1")
trainer = OghamTrainer(model, processor, train_dataset, val_dataset, config)
trainer.train()
```

## Ogham Character Reference

```
┌────────────────────────────────────────────────────────────────────┐
│ AICME BEITHE (B Group) - Strokes right of stemline                 │
│ ᚁ B   ᚂ L   ᚃ F/V   ᚄ S   ᚅ N                                     │
│                                                                     │
│ AICME HÚATHA (H Group) - Strokes left of stemline                  │
│ ᚆ H   ᚇ D   ᚈ T   ᚉ C   ᚊ Q                                        │
│                                                                     │
│ AICME MUINE (M Group) - Diagonal strokes                           │
│ ᚋ M   ᚌ G   ᚍ NG   ᚎ Z   ᚏ R                                       │
│                                                                     │
│ AICME AILME (Vowels) - Notches or cross-strokes                    │
│ ᚐ A   ᚑ O   ᚒ U   ᚓ E   ᚔ I                                        │
└────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Stone-Level Splitting
All images from the same stone must be in the same split to prevent data leakage. Multiple images of the same inscription could inflate metrics if split across train/test.

### Curriculum Learning
Training progresses through phases:
1. **Bootstrap**: Easy synthetic (short, clean)
2. **Transition**: Medium synthetic + real data introduction
3. **Refinement**: Hard synthetic (weathered) + balanced real

### On-the-Fly Generation
Synthetic images are generated procedurally during training, avoiding disk storage for millions of images while maintaining reproducibility through seeding.

## Recommended Fonts

For synthetic generation, use Ogham-compatible Unicode fonts:
- BabelStone Ogham
- Noto Sans Ogham
- Aboriginal Sans

## Ethical Considerations

- All scrapers respect `robots.txt` and rate limits
- License metadata is stored with every downloaded image
- Academic sources are used with appropriate attribution
- Contact archive maintainers for bulk academic use

## License

This project is for academic research purposes. Individual images retain their original licenses from source archives.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{ogham_ocr_pipeline,
  title={Ogham OCR Dataset Pipeline},
  year={2024},
  url={https://github.com/your-repo/ogham-ocr}
}
```
