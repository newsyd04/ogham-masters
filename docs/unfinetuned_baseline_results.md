# Unfine-tuned Baseline Results — 2026-04-18

## Experiment: TrOCR-small-stage1 Before Any Training

### Configuration
- **Model**: `microsoft/trocr-small-stage1` (62M params)
- **Tokenizer**: Extended with 28 Ogham tokens (latin-init embeddings)
- **Training**: None — pretrained weights only (Stage 1 synthetic English pretraining)
- **Evaluation**: 5,000 synthetic validation images from `DaraTraining/ogham-synthetic-200k`

### Results

| Metric | Value |
|--------|-------|
| Mean CER | **100.12%** |
| Exact Match | **0.0%** |
| Samples | 5,000 |

### Sample Predictions

| Reference | Prediction |
|-----------|------------|
| ᚐᚌᚊᚊ | . |
| ᚋᚐᚊᚔᚈᚈᚐᚄ ᚋᚐᚊᚔ ᚋᚒᚉᚑᚔ ᚉᚑᚏᚁᚁᚔ | . |
| ᚔᚌᚐᚉᚁᚆᚐᚋᚁᚐᚔᚑᚉ | . |
| ᚉᚋᚊᚒ | . |
| ᚁᚏᚒᚄᚉᚉᚑᚄ ᚋᚐᚊᚔ ᚉᚐᚂᚔᚐᚉᚔ | --- |

Every prediction is either `"."` or `"---"` — the pretrained RoBERTa decoder defaults to common English punctuation tokens when shown images it has never been trained on.

### Fine-tuning Impact

| Model | CER | Exact Match | Improvement |
|-------|-----|-------------|-------------|
| Unfine-tuned (0 epochs) | 100.12% | 0.0% | — |
| Fine-tuned frozen (20 epochs) | 0.14% | 99.5% | 715x |
| Fine-tuned unfrozen (20 epochs) | 0.06% | 99.8% | 1,669x |

Fine-tuning on 200k synthetic images for 20 epochs produces a **1,669x reduction in CER** — from complete failure (100%) to near-perfect (0.06%).

### Key Finding

The pretrained TrOCR-small model has **zero Ogham capability**. The 0.06% CER achieved after fine-tuning is entirely learned from synthetic training data. This confirms that domain-specific fine-tuning is essential and that the pretrained model contributes only general visual feature extraction (from the ViT encoder) and text generation patterns (from the RoBERTa decoder), not any script-specific knowledge.
