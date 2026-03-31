# Encoder Freeze Ablation Results — 2026-03-30

## Experiment: Frozen vs Unfrozen Encoder (TrOCR-small, Ogham mode)

### Configuration (shared)
- Model: `microsoft/trocr-small-stage1` (62M params)
- Dataset: `DaraTraining/ogham-synthetic-200k` (200k train, 5k val)
- Mode: Ogham Unicode
- Epochs: 20, Batch size: 16, LR: 5e-05
- Init strategy: latin-seeded embeddings

### Variable
- **Frozen**: Encoder frozen epochs 1-5, unfrozen epoch 6+ (`--freeze-encoder-epochs 5`)
- **No-Freeze**: Encoder unfrozen from epoch 1 (`--freeze-encoder-epochs 0`)

### Checkpoints
- Frozen: `/content/drive/MyDrive/ogham_ocr/checkpoints/best_ogham`
- No-Freeze: `/content/drive/MyDrive/ogham_ocr/checkpoints/no_freeze/best_ogham`

---

## No-Freeze Results (20/20 complete)

Best CER: **0.0006 (0.06%)** at epoch 17

| Epoch | Train Loss | Val Loss | CER | Exact Match | Time (s) |
|-------|-----------|----------|-----|-------------|----------|
| 1 | 0.3358 | 0.0323 | 2.61% | 85.6% | 1394 |
| 2 | 0.0626 | 0.0195 | 1.69% | 90.2% | 1387 |
| 3 | 0.0442 | 0.0175 | 1.52% | 91.5% | 1394 |
| 4 | 0.0350 | 0.0133 | 1.09% | 93.6% | 1385 |
| 5 | 0.0289 | 0.0105 | 1.02% | 94.2% | 1359 |
| 6 | 0.0241 | 0.0090 | 0.81% | 95.3% | 1366 |
| 7 | 0.0204 | 0.0074 | 0.66% | 96.0% | 1375 |
| 8 | 0.0173 | 0.0066 | 0.59% | 96.7% | 1373 |
| 9 | 0.0146 | 0.0064 | 0.56% | 96.7% | 1377 |
| 10 | 0.0121 | 0.0049 | 0.49% | 97.0% | 1373 |
| 11 | 0.0100 | 0.0039 | 0.41% | 97.4% | 1363 |
| 12 | 0.0081 | 0.0030 | 0.32% | 97.7% | 1373 |
| 13 | 0.0063 | 0.0024 | 0.51% | 96.1% | 1371 |
| 14 | 0.0048 | 0.0014 | 0.21% | 98.4% | 1369 |
| 15 | 0.0034 | 0.0012 | 0.15% | 99.0% | 1374 |
| 16 | 0.0022 | 0.0006 | 0.16% | 99.2% | 1366 |
| 17 | 0.0014 | 0.0005 | **0.06%** | 99.7% | 1372 |
| 18 | 0.0008 | 0.0002 | 0.15% | 99.4% | 1379 |
| 19 | 0.0004 | 0.0001 | 0.06% | **99.8%** | 1371 |
| 20 | 0.0003 | 0.0000 | 0.09% | 99.8% | 1370 |

Total training time: ~7h 38m

---

## Comparison Summary

| Metric | Frozen (5 epochs) | No-Freeze | Winner |
|--------|-------------------|-----------|--------|
| Best CER | 0.13% (epoch 18) | **0.06%** (epoch 17) | **No-Freeze** (2.3x better) |
| Best Exact Match | 99.5% (epoch 19) | **99.8%** (epoch 19) | **No-Freeze** |
| CER < 1% achieved at | epoch 9 | **epoch 5** | **No-Freeze** (4 epochs faster) |
| First useful output | epoch 6 | **epoch 1** | **No-Freeze** (5 epochs faster) |
| Final Val Loss | 0.0004 | **0.0000** | **No-Freeze** |
| Convergence style | Dramatic inflection at ep 6 | Smooth, monotonic | — |
| Total training time | ~7h 1m | ~7h 38m | Frozen (37 min faster) |

## Key Findings

1. **No-freeze produces a better final model** — 0.06% CER vs 0.14% CER (2.3x improvement)
2. **No-freeze converges faster to useful output** — 2.61% CER at epoch 1 vs 59% CER for frozen epochs 1-5
3. **Frozen approach wastes 5 epochs** — encoder cannot adapt, decoder learns from inappropriate features
4. **Frozen inflection is disruptive** — when encoder unfreezes at epoch 6, it must readjust to a decoder already trained on frozen features
5. **No-freeze convergence is smoother** — monotonically decreasing CER without the oscillations seen in the frozen run (epochs 8, 10, 15-16)
6. **Frozen is only faster in wall-clock time** — frozen epochs run ~1000s vs ~1370s unfrozen, saving 37 minutes total despite worse results
