# Phase 2 Three-Architecture Comparison

All three OCR architectures were Phase-1 fine-tuned on 200k synthetic Ogham inscriptions, then Phase-2 fine-tuned on 280 synthetic-freeform samples. Evaluated on the same 35-sample held-out test split.

## Results

| Model | Architecture | Phase 1 synth CER | Pre-P2 freeform CER | Post-P2 freeform CER | Lift (pp) |
|---|---|---|---|---|---|
| **TrOCR-small** | Attention (TrOCR encoder + RoBERTa decoder) | 0.06% | 14.34% | **1.34%** | ↓ 13.0 |
| **PARSeq** | Attention (ViT + Permutation-LM decoder) | 8.96% | 37.90% | **29.17%** | ↓ 8.7 |
| **CNN+RNN** | CTC (ResNet-18 + BiLSTM) | 66.82% | 68.58% | **67.24%** | ↓ 1.3 |

## Key finding

Phase 2 adaptation benefit scales with Phase 1 capability. Attention-based models (TrOCR-small, PARSeq) adapt substantially to the freeform distribution; CNN+RNN's CTC-based architecture cannot meaningfully adapt despite its training loss collapsing (memorisation without generalisation).

## Methodology

- **Training data for Phase 2**: 280 synthetic-freeform samples (elastic-deformation-augmented   Ogham inscriptions rendered via OghamRenderer, then warped with albumentations ElasticTransform)
- **Validation**: 35 samples for checkpoint selection during training
- **Test**: 35 pristine samples, never used for training or tuning
- **Split seed**: 42 (deterministic, consistent across all three architectures)
- **CER computation**: whitespace excluded from both reference and prediction
- **Training**: 3-5 epochs at lower LR than Phase 1, fine-tuning from Phase 1 best checkpoint

## Figures

- `figures/phase2_pre_vs_post.png` — grouped bar chart of pre- vs post-Phase-2 CER
- `figures/phase2_lift_vs_capability.png` — scatter plot: Phase 1 capability vs Phase 2 lift
- `figures/phase2_architecture_comparison.png` — combined figure for thesis use
