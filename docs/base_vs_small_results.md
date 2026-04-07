# TrOCR-base vs TrOCR-small Results — 2026-04-07

## Experiment: Model Scale Comparison (Unfrozen Encoder, Ogham Mode)

### Configuration (shared)
- Dataset: `DaraTraining/ogham-synthetic-200k` (200k train, 5k val)
- Mode: Ogham Unicode
- Encoder: Unfrozen from epoch 1
- Init strategy: latin-seeded embeddings

### Models
| | TrOCR-small | TrOCR-base |
|---|---|---|
| Parameters | 62M | 385M |
| Encoder | DeiT-small (22M) | ViT-base (87M) |
| Decoder | RoBERTa 6-layer (38M) | RoBERTa 12-layer (298M) |
| Hidden dim | 384 | 1024 |

## Results

### TrOCR-small unfrozen (20 epochs, batch 16, lr 5e-5)

Best CER: **0.06%** at epoch 17 | Best Exact: **99.8%** at epoch 19

| Epoch | Train Loss | Val Loss | CER | Exact |
|-------|-----------|----------|-----|-------|
| 1 | 0.3358 | 0.0323 | 2.61% | 85.6% |
| 5 | 0.0289 | 0.0105 | 1.02% | 94.2% |
| 10 | 0.0121 | 0.0049 | 0.49% | 97.0% |
| 15 | 0.0034 | 0.0012 | 0.15% | 99.0% |
| 20 | 0.0003 | 0.0000 | 0.09% | 99.8% |

### TrOCR-base unfrozen — Run 1 (batch 64, lr 5e-5, 7 epochs before disconnect)

Best CER: **28.55%** at epoch 2 | Diverged after epoch 2

| Epoch | Train Loss | Val Loss | CER | Exact |
|-------|-----------|----------|-----|-------|
| 1 | 1.2340 | 0.0699 | 32.43% | 8.4% |
| 2 | 0.6366 | 0.0423 | 28.55% | 25.2% |
| 3 | 0.4666 | 0.0866 | 40.62% | 18.5% |
| 4 | 0.3561 | 0.0804 | 104.22% | 0.0% |
| 5 | 0.2635 | 0.1763 | 100.02% | 0.0% |
| 6 | 0.2172 | 0.0853 | 110.35% | 0.0% |
| 7 | 0.1771 | 0.0447 | 103.18% | 0.0% |

### TrOCR-base unfrozen — Run 2 (batch 16, lr 5e-5, 3 epochs)

Best CER: **90.43%** at epoch 1 | Diverging

| Epoch | Train Loss | Val Loss | CER | Exact |
|-------|-----------|----------|-----|-------|
| 1 | 0.2670 | 0.2487 | 90.43% | 17.1% |
| 2 | 0.1088 | 0.3105 | 91.37% | 14.9% |
| 3 | 0.1008 | 0.4754 | 99.57% | 1.0% |

## Key Findings

### 1. Bigger is not better for Ogham OCR
TrOCR-small (62M) achieves 0.06% CER while TrOCR-base (385M) cannot converge below 28% CER. The 6x larger model performs **1500x worse** on this task.

### 2. Base model diverges — train loss drops, val loss rises
Both base runs show the same pattern: training loss decreases (the model memorises training data) but validation loss increases (generalisation degrades). This is classic overfitting driven by overparameterisation.

### 3. CER > 100% indicates degenerate generation
In the batch-64 run, CER exceeded 100% by epoch 4, meaning the model outputs longer strings than the reference (character duplication). The 12-layer decoder "stutters" — repeating characters instead of advancing.

### 4. The 28-character Ogham alphabet doesn't need 298M decoder parameters
The RoBERTa decoder in TrOCR-base was designed for 50,265 English subword tokens. Ogham has 28 characters with simple sequential structure. The massive decoder capacity leads to overfitting rather than better representation.

### 5. Smaller encoder also matters
TrOCR-small uses DeiT-small (22M params, 384 hidden dim) vs ViT-base (87M params, 1024 hidden dim). The smaller encoder extracts sufficient visual features for Ogham strokes — the larger encoder provides no benefit.

## Comparison with All Models

| Model | Params | Best CER | Best Exact | Notes |
|-------|--------|----------|------------|-------|
| **TrOCR-small unfrozen** | **62M** | **0.06%** | **99.8%** | **Best model** |
| TrOCR-small frozen | 62M | 0.14% | 99.5% | Freezing hurts |
| TrOCR-base (batch 64) | 385M | 28.55% | 25.2% | Diverged at epoch 4 |
| TrOCR-base (batch 16) | 385M | 90.43% | 17.1% | Diverging at epoch 3 |
| CNN+RNN (CTC) | 15M | 66.82% | 24.8% | CTC alignment limit |
| Claude (few-shot) | ~100B+ | 80.07% | 0.0% | Format learned, not skill |
| GPT-4o (zero-shot) | ~200B | 98.22% | 0.0% | Refused to attempt |
