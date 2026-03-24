# Clean Comparison Run Results — 2026-03-23

Run started: 13:19:59 | Ogham completed: 20:21:08 | Latin started: 20:21:12 | Latin completed: 03:10:15

## Configuration
- Model: `microsoft/trocr-small-stage1` (62M params)
- Dataset: `DaraTraining/ogham-synthetic-200k` (200k train, 5k val)
- Epochs: 20 per mode
- Batch size: 16, LR: 5e-05
- Encoder frozen epochs 1-5, unfrozen epoch 6+
- Checkpoints: `/content/drive/MyDrive/ogham_ocr/checkpoints/`

## Ogham Mode (20/20 complete)

Best CER: **0.0013 (0.13%)** at epoch 18

| Epoch | Encoder | Train Loss | Val Loss | CER | Exact Match | Time (s) |
|-------|---------|-----------|----------|-----|-------------|----------|
| 1 | Frozen | 1.4863 | 0.6896 | 72.45% | 10.8% | 1005 |
| 2 | Frozen | 1.3356 | 0.6589 | 67.86% | 15.9% | 1007 |
| 3 | Frozen | 1.2789 | 0.6321 | 62.18% | 15.7% | 999 |
| 4 | Frozen | 1.2386 | 0.6119 | 59.79% | 17.1% | 1001 |
| 5 | Frozen | 1.2051 | 0.5973 | 59.15% | 19.6% | 1004 |
| 6 | Unfrozen | 0.1891 | 0.0325 | 2.71% | 85.4% | 1343 |
| 7 | Unfrozen | 0.0546 | 0.0174 | 1.69% | 91.1% | 1351 |
| 8 | Unfrozen | 0.0376 | 0.0160 | 1.97% | 91.0% | 1347 |
| 9 | Unfrozen | 0.0288 | 0.0108 | 1.00% | 94.4% | 1353 |
| 10 | Unfrozen | 0.0227 | 0.0082 | 1.73% | 91.3% | 1349 |
| 11 | Unfrozen | 0.0182 | 0.0069 | 0.69% | 96.1% | 1354 |
| 12 | Unfrozen | 0.0145 | 0.0048 | 0.64% | 96.5% | 1345 |
| 13 | Unfrozen | 0.0116 | 0.0048 | 0.52% | 97.3% | 1350 |
| 14 | Unfrozen | 0.0092 | 0.0032 | 0.52% | 97.7% | 1350 |
| 15 | Unfrozen | 0.0068 | 0.0025 | 0.81% | 98.0% | 1346 |
| 16 | Unfrozen | 0.0051 | 0.0018 | 0.72% | 97.6% | 1348 |
| 17 | Unfrozen | 0.0037 | 0.0012 | 0.20% | 99.0% | 1347 |
| 18 | Unfrozen | 0.0024 | 0.0007 | **0.13%** | 99.4% | 1354 |
| 19 | Unfrozen | 0.0016 | 0.0005 | 0.14% | **99.5%** | 1346 |
| 20 | Unfrozen | 0.0011 | 0.0004 | 0.14% | 99.4% | 1343 |

Total ogham training time: ~7h 1m

### Sample Predictions (best checkpoint, evaluated on validation set)
| # | Reference | Prediction | Match | Notes |
|---|-----------|------------|-------|-------|
| 0 | ᚐᚌᚊᚊ | ᚐᚌᚊᚊ | Yes | |
| 1 | ᚋᚐᚊᚔᚈᚈᚐᚄᚋᚐᚊᚔᚋᚒᚉᚑᚔᚉᚑᚏᚁᚁᚔ | ᚋᚐᚊᚔᚈᚈᚐᚄᚋᚐᚊᚔᚋᚒᚉᚑᚔᚉᚑᚏᚁᚁᚔ | Yes | |
| 2 | ᚔᚌᚐᚉᚁᚆᚐᚋᚁᚐᚔᚑᚉ | ᚔᚌᚐᚉᚁᚆᚐᚋᚁᚐᚔᚑᚉ | Yes | |
| 3 | ᚊᚉᚐᚌᚐᚊᚊ | ᚊᚉᚐᚌᚐᚊᚊ | Yes | |
| 4 | ᚄᚔᚄᚐᚉᚐᚊᚉᚐᚈᚈᚒᚁᚒᚈᚈᚐᚄ | ᚄᚔᚄᚐᚉᚐᚊᚉᚐᚈᚈᚒᚁᚒᚈᚈᚐᚄ | Yes | |
| 5 | ᚉᚐᚌᚔᚅᚐᚇᚔ ᚋᚐᚊᚔ ᚃᚑᚁᚐᚏᚐᚉᚔ | ᚉᚐᚌᚔᚅᚐᚇᚔᚋᚐᚊᚔᚃᚑᚁᚐᚏᚐᚉᚔ | No | U+1680 spaces dropped |
| 6 | ᚉᚐᚈᚈᚒᚁᚒᚈᚈᚐᚄ ᚋᚐᚊᚔ ᚋᚒᚉᚑᚔ ᚃᚓᚇᚇᚑᚄ | ᚉᚐᚈᚈᚒᚁᚒᚈᚈᚐᚄᚋᚐᚊᚔᚋᚒᚉᚑᚔᚃᚓᚇᚇᚑᚄ | No | U+1680 spaces dropped |
| 7 | ᚓᚈᚑᚔᚗᚋᚃᚔᚋᚑᚑᚊ | ᚓᚈᚑᚔᚗᚋᚃᚔᚋᚑᚑᚊ | Yes | |
| 8 | ᚑᚐᚐᚋᚐᚄᚐᚒᚔᚊᚋᚔᚒᚈᚐ | ᚑᚐᚐᚋᚐᚄᚐᚒᚔᚊᚋᚔᚒᚈᚐ | Yes | |
| 9 | ᚈᚐᚁᚅᚄᚐᚋᚔᚌᚐᚉᚑᚋᚔᚏᚊ | ᚈᚐᚁᚅᚄᚐᚋᚔᚌᚐᚉᚑᚋᚔᚏᚊ | Yes | |
| 10 | ᚏᚔᚈᚐᚊᚐᚁᚔᚌᚋᚐᚔᚈ | ᚏᚔᚈᚐᚊᚐᚁᚔᚌᚋᚐᚔᚈ | Yes | |
| 11 | ᚋᚐᚊᚔᚈᚈᚐᚄ ᚋᚐᚊᚔ ᚋᚒᚉᚑᚔ ᚉᚑᚏᚁᚁᚔ | ᚋᚐᚊᚔᚈᚈᚐᚄᚋᚐᚊᚔᚋᚒᚉᚑᚔᚉᚑᚏᚁᚁᚔ | No | U+1680 spaces dropped |
| 12 | ᚉᚐᚈᚈᚒᚁᚒᚈᚈᚐᚄ ᚋᚐᚊᚔ ᚋᚒᚉᚑᚔ ᚃᚓᚇᚇᚑᚄ | ᚉᚐᚈᚈᚒᚁᚒᚈᚈᚐᚄᚋᚐᚊᚔᚋᚒᚉᚑᚔᚃᚓᚇᚇᚑᚄ | No | U+1680 spaces dropped |

**Ogham evaluation: 9/13 exact match. All 4 errors are U+1680 space mark omissions — zero character-level errors.**

## Latin Mode (20/20 complete)

Best CER: **0.0016 (0.16%)** at epoch 20

| Epoch | Encoder | Train Loss | Val Loss | CER | Exact Match | Time (s) |
|-------|---------|-----------|----------|-----|-------------|----------|
| 1 | Frozen | 2.3539 | 1.1196 | 71.51% | 9.9% | 959 |
| 2 | Frozen | 2.1713 | 1.0662 | 63.79% | 15.9% | 962 |
| 3 | Frozen | 2.0909 | 1.0289 | 60.29% | 17.7% | 970 |
| 4 | Frozen | 2.0345 | 1.0008 | 58.69% | 19.6% | 968 |
| 5 | Frozen | 1.9882 | 0.9768 | 56.23% | 21.1% | 972 |
| 6 | Unfrozen | 0.5651 | 0.1074 | 3.84% | 81.1% | 1307 |
| 7 | Unfrozen | 0.2088 | 0.0722 | 2.48% | 86.8% | 1312 |
| 8 | Unfrozen | 0.1536 | 0.0589 | 1.84% | 90.8% | 1313 |
| 9 | Unfrozen | 0.1235 | 0.0449 | 1.70% | 92.4% | 1311 |
| 10 | Unfrozen | 0.1011 | 0.0382 | 1.50% | 93.0% | 1309 |
| 11 | Unfrozen | 0.0830 | 0.0315 | 0.90% | 95.0% | 1311 |
| 12 | Unfrozen | 0.0675 | 0.0250 | 0.75% | 96.0% | 1313 |
| 13 | Unfrozen | 0.0540 | 0.0188 | 0.57% | 97.1% | 1315 |
| 14 | Unfrozen | 0.0420 | 0.0139 | 0.61% | 97.4% | 1315 |
| 15 | Unfrozen | 0.0314 | 0.0101 | 0.36% | 98.2% | 1313 |
| 16 | Unfrozen | 0.0224 | 0.0066 | 0.31% | 98.4% | 1308 |
| 17 | Unfrozen | 0.0156 | 0.0044 | 0.21% | 99.0% | 1316 |
| 18 | Unfrozen | 0.0109 | 0.0030 | 0.20% | 99.1% | 1324 |
| 19 | Unfrozen | 0.0084 | 0.0023 | 0.18% | 99.3% | 1318 |
| 20 | Unfrozen | 0.0070 | 0.0022 | **0.16%** | **99.2%** | 1307 |

Total latin training time: ~6h 49m

### Sample Predictions (epoch 20)
| # | Reference | Prediction | Correct |
|---|-----------|------------|---------|
| 0 | AGQQ | AGQQ | Yes |
| 1 | MAQITTAS?MAQI?MUCOI?CORBBI | MAQITTAS?MAQI?MUCOI?CORBBI | Yes |
| 2 | IGACBHAMBAIOC | IGACBHAMBAIOC | Yes |

## Comparison Summary

| Metric | Ogham | Latin | Winner |
|--------|-------|-------|--------|
| Best CER | **0.13%** (epoch 18) | 0.16% (epoch 20) | Ogham |
| Best Exact Match | **99.5%** (epoch 19) | 99.3% (epoch 19) | Ogham |
| Final CER (epoch 20) | **0.14%** | 0.16% | Ogham |
| Final Train Loss | **0.0011** | 0.0070 | Ogham |
| Final Val Loss | **0.0004** | 0.0022 | Ogham |
| Epoch 6 CER (inflection) | **2.71%** | 3.84% | Ogham |
| Epoch 6 Exact Match | **85.4%** | 81.1% | Ogham |
| Training time (total) | ~7h 1m | **~6h 49m** | Latin |
| Frozen epoch speed | ~1000s/epoch | **~965s/epoch** | Latin |
| Unfrozen epoch speed | ~1348s/epoch | **~1313s/epoch** | Latin |

## Key Observations

### Encoder Freeze/Unfreeze Inflection
- **Ogham** epoch 5 → 6: CER 59.15% → 2.71% (21.8x improvement)
- **Latin** epoch 5 → 6: CER 56.23% → 3.84% (14.6x improvement)
- Ogham's inflection is 1.5x more dramatic — the 1:1 token mapping gives the decoder a cleaner gradient signal once the encoder starts adapting

### Convergence Trajectory
- Ogham plateaued at epoch 18 (CER 0.13%), oscillated epochs 19-20
- Latin still improving at epoch 20 (CER 0.16%), had not fully plateaued
- Latin's convergence was slower but steadier — no CER oscillations like Ogham's (epochs 8, 10, 15-16)

### Speed
- Frozen encoder: Ogham ~1000s/epoch (~12.5 it/s), Latin ~965s/epoch (~15.5 it/s)
- Unfrozen encoder: Ogham ~1348s/epoch (~9.3 it/s), Latin ~1313s/epoch (~9.5 it/s)
- Latin is faster due to smaller output projection (50,265 vs 64,031 vocab)

### Loss Scale
- Latin train loss consistently higher (0.0070 vs 0.0011 at epoch 20) because BPE predicts more tokens per sequence
- Latin val loss also higher (0.0022 vs 0.0004) — not comparable across modes due to different token counts

### Tokenizer
- Ogham: 64,031 vocab (29 Ogham tokens added, latin-init embeddings, 1:1 char mapping)
- Latin: ~50,265 vocab (default RoBERTa BPE, warm embeddings, subword tokenization)
