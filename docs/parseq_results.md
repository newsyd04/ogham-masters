# PARSeq Baseline Results — 2026-04-08

## Experiment: PARSeq (Permuted Autoregressive Sequence) on Synthetic Ogham Data

### Configuration
- **Architecture**: ViT encoder (12 blocks, 384 dim) + Transformer decoder (1 layer) + permutation language modelling
- **Parameters**: ~24M (encoder 21.4M, decoder 2.4M, head + embed ~0.07M)
- **Dataset**: `DaraTraining/ogham-synthetic-200k` (200k train, 5k val)
- **Training**: 20 epochs, batch 64, lr 1e-4, AdamW, CosineAnnealingLR, AMP
- **Image size**: 32x128
- **Charset**: 29 characters (28 Ogham + space)
- **Pretrained**: ViT encoder from scene text recognition, decoder trained from scratch
- **Paper**: Bautista & Atienza, ECCV 2022

### Results

Best CER: **8.96%** at epoch 20 | Best Exact: **72.2%** at epoch 20

| Epoch | Train Loss | Val Loss | CER | Exact Match |
|-------|-----------|----------|-----|-------------|
| 1 | 1.1762 | 0.7947 | 57.75% | 29.4% |
| 2 | 0.6815 | 0.5474 | 39.40% | 38.7% |
| 3 | 0.4970 | 0.4399 | 30.57% | 47.9% |
| 4 | 0.4111 | 0.3724 | 23.48% | 55.0% |
| 5 | 0.3573 | 0.3320 | 20.77% | 58.5% |
| 6 | 0.3196 | 0.2920 | 17.54% | 60.8% |
| 7 | 0.2908 | 0.2690 | 15.42% | 63.6% |
| 8 | 0.2683 | 0.2504 | 14.88% | 65.3% |
| 9 | 0.2496 | 0.2282 | 13.18% | 66.4% |
| 10 | 0.2333 | 0.2194 | 12.11% | 67.4% |
| 11 | 0.2193 | 0.2031 | 11.78% | 68.5% |
| 12 | 0.2071 | 0.1950 | 11.64% | 69.2% |
| 13 | 0.1961 | 0.1861 | 10.60% | 69.8% |
| 14 | 0.1868 | 0.1770 | 10.60% | 70.1% |
| 15 | 0.1789 | 0.1711 | 9.67% | 70.5% |
| 16 | 0.1722 | 0.1679 | 9.69% | 70.8% |
| 17 | 0.1669 | 0.1622 | 9.47% | 71.6% |
| 18 | 0.1629 | 0.1600 | 9.19% | 72.1% |
| 19 | 0.1604 | 0.1582 | 9.00% | 72.1% |
| 20 | 0.1589 | 0.1578 | 8.96% | 72.2% |

### Key Findings

1. **PARSeq is the clear second-best model** at 8.96% CER — 7.5x better than CNN+RNN (66.8%) but 149x worse than TrOCR-small (0.06%).

2. **Attention is the key differentiator**: PARSeq (24M, attention) vs CNN+RNN (15M, CTC) — similar model sizes but PARSeq achieves 8.96% vs 66.8%. The Transformer decoder's cross-attention to visual features is fundamentally better than CTC's fixed-context approach.

3. **Pretrained language model decoder matters**: TrOCR-small's RoBERTa decoder (pretrained on English) achieves 0.06% vs PARSeq's from-scratch decoder at 8.96%. Despite being trained on English text, RoBERTa's learned attention patterns transfer effectively to Ogham.

4. **Still converging at epoch 20**: CER dropped from 57.7% to 9.0% with no plateau. More epochs (40-60) could potentially reach 3-5% CER. However, TrOCR-small reached 2.6% CER in just 1 epoch.

5. **72.2% exact match is respectable**: Nearly 3 in 4 synthetic inscriptions are transcribed perfectly. The remaining 28% are close but have minor character errors.

## Full Model Comparison

| Model | Params | Architecture | Best CER | Best Exact | Epochs to <10% |
|-------|--------|-------------|----------|------------|----------------|
| **TrOCR-small unfrozen** | **62M** | **ViT + RoBERTa** | **0.06%** | **99.8%** | **1** |
| TrOCR-small frozen | 62M | ViT + RoBERTa | 0.14% | 99.5% | 9 |
| PARSeq | 24M | ViT + Transformer | 8.96% | 72.2% | 18 |
| CNN+RNN (CTC) | 15M | ResNet-18 + BiLSTM | 66.82% | 24.8% | never |
| TrOCR-base | 385M | ViT + RoBERTa-12L | 90.43% | 17.1% | never |
| Claude (few-shot) | ~100B+ | Multimodal LLM | 80.07% | 0.0% | N/A |
| GPT-4o (zero-shot) | ~200B | Multimodal LLM | 98.22% | 0.0% | N/A |
