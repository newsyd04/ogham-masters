# Ogham OCR Output Representation: Unicode vs Latin Transliteration

## Overview

When fine-tuning TrOCR for Ogham inscription recognition, a fundamental design
decision is the output representation. Two approaches are considered:

1. **Ogham Unicode**: The model decodes directly to Ogham Unicode characters
   (e.g., `᚛ᚉᚒᚅᚐᚃᚐᚂᚔ᚜`)
2. **Latin Transliteration**: The model decodes to Latin character equivalents
   (e.g., `CUNAVALI`)

Both approaches use the same visual encoder (ViT) and the same input images.
The difference lies entirely in the decoder's target vocabulary and output space.

---

## Approach 1: Ogham Unicode Output

### Description
The TrOCR tokenizer (RoBERTa-based BPE) is extended with 29 new tokens
covering the full Ogham Unicode block (U+1680–U+169C). The model's decoder
embedding matrix is resized accordingly. Each Ogham character maps to exactly
one token.

### Pros

| Advantage | Explanation |
|-----------|-------------|
| **Preserves original script** | Output is directly usable for scholarly work, database entry, and Unicode-compliant digital epigraphy without post-processing |
| **Compact output vocabulary** | Only ~29 Ogham tokens + special tokens. The decoder's effective search space is much smaller than the full RoBERTa vocabulary (~50k tokens) |
| **1:1 character-token mapping** | Each Ogham stroke pattern corresponds to exactly one output token, making the decoding task cleanly structured |
| **No ambiguity in mapping** | Ogham → Unicode is unambiguous. There are no spelling variants or case sensitivity issues |
| **Compact sequences** | Output sequences are short (typically 5–25 tokens), within easy reach of the decoder |
| **Enables end-to-end digital epigraphy** | A researcher can photograph a stone and immediately get a Unicode transcription |

### Cons

| Disadvantage | Explanation |
|--------------|-------------|
| **Requires tokenizer modification** | Adding new tokens to a pre-trained tokenizer means new embedding vectors are randomly initialized — they have no pre-trained semantic knowledge |
| **Cold-start embeddings** | The 29 new token embeddings start from random initialization, requiring the model to learn them from scratch. This may require more training data |
| **No transfer from language model** | The pre-trained decoder's language modelling capability (learned on English) provides zero benefit for Ogham sequence patterns |
| **Harder to evaluate informally** | Ogham Unicode characters are not human-readable for most reviewers without a reference chart |
| **Font rendering issues** | Not all environments render Ogham Unicode correctly, making debugging and display harder |

---

## Approach 2: Latin Transliteration Output

### Description
The model outputs Latin characters representing the Ogham transliteration
(e.g., `MAQI MUCOI CUNACATOS`). The standard RoBERTa tokenizer already
handles these characters natively — no vocabulary extension needed.

### Pros

| Advantage | Explanation |
|-----------|-------------|
| **No tokenizer modification** | Latin characters are already in the vocabulary. The pre-trained embeddings are immediately useful |
| **Warm-start embeddings** | All Latin character tokens have meaningful pre-trained representations from RoBERTa's training on English text |
| **Transfer learning benefit** | The decoder's language model may recognise partial English-like patterns in Ogham transliterations (many are Celtic personal names with recognisable structure) |
| **Human-readable output** | Researchers and non-specialists can read the output directly without Unicode lookup tables |
| **Easier debugging** | Print statements, logs, and visualisations are immediately interpretable |
| **Broader compatibility** | Latin characters render correctly in every environment |

### Cons

| Disadvantage | Explanation |
|--------------|-------------|
| **Information loss** | Some Ogham distinctions may be ambiguous in Latin (e.g., F/V both map to ᚃ; transliteration must choose one) |
| **Multi-character tokens** | BPE may tokenize `MAQI` as one or two tokens rather than four characters, creating a misalignment between visual stroke patterns and token boundaries |
| **Requires post-processing for Unicode** | If Unicode output is ultimately needed, an additional transliteration step introduces potential errors |
| **Larger effective vocabulary** | The decoder can potentially output any of ~50k RoBERTa tokens, making the search space larger (though beam search mitigates this) |
| **Digraph ambiguity** | Transliterations like `NG` (one Ogham character ᚍ) vs `N`+`G` (two characters ᚅᚌ) create tokenization ambiguity |
| **Case sensitivity** | Ogham has no case distinction, but the Latin tokenizer does — the model must learn to always produce uppercase |
| **Scholarly convention mismatch** | Published Ogham scholarship increasingly uses Unicode. Latin-only output requires an extra conversion step for academic use |

---

## Experimental Comparison Framework

The `scripts/finetune_trocr.py --mode compare` script trains both approaches
with identical hyperparameters and data, then compares:

| Metric | What it measures |
|--------|-----------------|
| **CER (Character Error Rate)** | Edit distance between predicted and reference strings, normalized by reference length |
| **Exact Match Rate** | Fraction of samples where prediction exactly matches reference |
| **Convergence Speed** | How many epochs to reach a given CER threshold |
| **Overfit-One-Sample Steps** | How many gradient steps to perfectly memorize one sample (litmus test) |

### Expected Hypotheses

1. **Latin may converge faster** due to warm-start embeddings, especially with
   limited training data (<10k samples)
2. **Ogham Unicode may achieve lower final CER** on larger datasets because the
   compact vocabulary and 1:1 mapping reduce the decoder's search space
3. **Both should pass the overfit-one-sample test** if the pipeline is correct
4. **The gap should narrow** as training data increases (100k+ samples), since
   the cold-start embedding disadvantage diminishes with more gradient updates

---

## Recommendation

For a research project on Ogham OCR, **implement and evaluate both approaches**:

- **Latin transliteration** is the pragmatic first choice for rapid iteration.
  It will likely converge faster with small datasets and is easier to debug.
- **Ogham Unicode** is the more principled approach for a production system or
  scholarly tool. It preserves the full information content and aligns with
  digital epigraphy standards.

The ideal pipeline may ultimately be a **two-head approach** or **post-hoc
conversion**, where the model outputs Latin transliterations during training
(leveraging warm embeddings) and a deterministic mapping converts to Ogham
Unicode at inference time. This gives the best of both worlds: fast training
convergence and Unicode-compliant output.

---

## Data Requirements Estimate

| Dataset Size | Expected Outcome |
|-------------|-----------------|
| 1 sample | Overfit test — validates pipeline works |
| 100–1,000 | Model learns basic stroke patterns on synthetic data |
| 10,000–50,000 | Reasonable CER on synthetic data, limited generalisation |
| 100,000–200,000 | Strong synthetic performance, basis for real-data fine-tuning |
| 200,000+ synthetic + 132 real | Target for publication-quality results |

The synthetic dataset generator (`src/generation/`) can produce unlimited
training data. The recommendation is to start with the overfit test, scale to
~200k synthetic images, then gradually mix in the 132 real stone annotations
using curriculum learning.
