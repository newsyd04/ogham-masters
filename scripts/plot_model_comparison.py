#!/usr/bin/env python3
"""Plot all model comparison charts: TrOCR-small (frozen/unfrozen), CNN+RNN, large models."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

output_dir = Path("docs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# === Data ===

epochs = list(range(1, 21))

# TrOCR-small frozen
small_frozen_cer = [
    72.45, 67.86, 62.18, 59.79, 59.15,
    2.71, 1.69, 1.97, 1.00, 1.73,
    0.69, 0.64, 0.52, 0.52, 0.81,
    0.72, 0.20, 0.13, 0.14, 0.14,
]

# TrOCR-small unfrozen
small_unfrozen_cer = [
    2.61, 1.69, 1.52, 1.09, 1.02,
    0.81, 0.66, 0.59, 0.56, 0.49,
    0.41, 0.32, 0.51, 0.21, 0.15,
    0.16, 0.06, 0.15, 0.06, 0.09,
]

# CNN+RNN (CTC)
cnn_rnn_cer = [
    71.42, 68.58, 70.70, 68.33, 68.63,
    68.02, 68.85, 68.12, 68.31, 67.84,
    67.88, 67.14, 67.87, 66.96, 67.05,
    66.87, 66.95, 67.15, 66.86, 66.82,
]

cnn_rnn_exact = [
    18.1, 20.8, 21.2, 22.1, 22.3,
    22.8, 22.9, 23.1, 23.3, 23.6,
    23.9, 24.0, 24.2, 24.4, 24.6,
    24.7, 24.8, 24.8, 24.8, 24.8,
]

small_frozen_exact = [
    10.8, 15.9, 15.7, 17.1, 19.6,
    85.4, 91.1, 91.0, 94.4, 91.3,
    96.1, 96.5, 97.3, 97.7, 98.0,
    97.6, 99.0, 99.4, 99.5, 99.4,
]

small_unfrozen_exact = [
    85.6, 90.2, 91.5, 93.6, 94.2,
    95.3, 96.0, 96.7, 96.7, 97.0,
    97.4, 97.7, 96.1, 98.4, 99.0,
    99.2, 99.7, 99.4, 99.8, 99.8,
]

# PARSeq
parseq_cer = [
    57.75, 39.40, 30.57, 23.48, 20.77,
    17.54, 15.42, 14.88, 13.18, 12.11,
    11.78, 11.64, 10.60, 10.60, 9.67,
    9.69, 9.47, 9.19, 9.00, 8.96,
]
parseq_exact = [
    29.4, 38.7, 47.9, 55.0, 58.5,
    60.8, 63.6, 65.3, 66.4, 67.4,
    68.5, 69.2, 69.8, 70.1, 70.5,
    70.8, 71.6, 72.1, 72.1, 72.2,
]

# Large models — Phase 1 clean-synth (30 samples, few-shot prompting).
# Source: docs/extended_large_model_results.json
large_models = {
    "GPT-4o\n(few-shot)": 102.71,
    "Claude\n(few-shot)": 90.64,
    "Gemini\n(few-shot)": 110.10,
}

plt.style.use("seaborn-v0_8-whitegrid")
colors = {
    "frozen": "#2196F3",
    "unfrozen": "#FF5722",
    "cnn_rnn": "#9C27B0",
    "parseq": "#4CAF50",
    "large": "#607D8B",
}

# === Figure 1: All models CER over epochs ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Full range
ax1.plot(epochs, small_frozen_cer, "o-", color=colors["frozen"], label="TrOCR-small frozen", linewidth=2, markersize=4)
ax1.plot(epochs, small_unfrozen_cer, "s-", color=colors["unfrozen"], label="TrOCR-small unfrozen", linewidth=2, markersize=4)
ax1.plot(epochs, parseq_cer, "D-", color=colors["parseq"], label="PARSeq", linewidth=2, markersize=4)
ax1.plot(epochs, cnn_rnn_cer, "^-", color=colors["cnn_rnn"], label="CNN+RNN (CTC)", linewidth=2, markersize=4)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("CER (%)")
ax1.set_title("CER Over Training — All Models")
ax1.legend(fontsize=9)
ax1.set_ylim(-2, 80)

# Zoomed to TrOCR + PARSeq convergence
ax2.plot(epochs[5:], small_frozen_cer[5:], "o-", color=colors["frozen"], label="TrOCR-small frozen", linewidth=2, markersize=5)
ax2.plot(epochs, small_unfrozen_cer, "s-", color=colors["unfrozen"], label="TrOCR-small unfrozen", linewidth=2, markersize=5)
ax2.plot(epochs, parseq_cer, "D-", color=colors["parseq"], label="PARSeq", linewidth=2, markersize=5)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("CER (%)")
ax2.set_title("CER Zoomed — Attention Models")
ax2.legend(fontsize=9)
ax2.set_ylim(-0.5, 60)

fig.suptitle("Model Comparison: Character Error Rate", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(output_dir / "model_comparison_cer.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'model_comparison_cer.png'}")

# === Figure 2: Exact Match over epochs ===
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(epochs, small_frozen_exact, "o-", color=colors["frozen"], label="TrOCR-small frozen", linewidth=2, markersize=4)
ax.plot(epochs, small_unfrozen_exact, "s-", color=colors["unfrozen"], label="TrOCR-small unfrozen", linewidth=2, markersize=4)
ax.plot(epochs, parseq_exact, "D-", color=colors["parseq"], label="PARSeq", linewidth=2, markersize=4)
ax.plot(epochs, cnn_rnn_exact, "^-", color=colors["cnn_rnn"], label="CNN+RNN (CTC)", linewidth=2, markersize=4)
ax.axhline(y=99, color="green", linestyle=":", alpha=0.4, label="99% target")
ax.set_xlabel("Epoch")
ax.set_ylabel("Exact Match (%)")
ax.set_title("Model Comparison: Exact Match Rate", fontsize=14, fontweight="bold")
ax.legend()
ax.set_ylim(0, 101)

plt.tight_layout()
plt.savefig(output_dir / "model_comparison_exact_match.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'model_comparison_exact_match.png'}")

# === Figure 3: Final CER bar chart (all models including large) ===
fig, ax = plt.subplots(figsize=(12, 5))

all_models = {
    "TrOCR-small\nunfrozen":      0.06,
    "TrOCR-small\nfrozen":        0.14,
    "PARSeq":                     8.96,
    "TrOCR-base":                44.25,
    "CNN+RNN\n(CTC)":             66.82,
    "TrOCR-small\nUNTRAINED":   100.12,
    "GPT-4o\n(few-shot)":       102.71,
    "Claude\n(few-shot)":        90.64,
    "Gemini\n(few-shot)":       110.10,
}

model_colors = [
    colors["unfrozen"], colors["frozen"], colors["parseq"],
    "#03A9F4",  # TrOCR-base — distinct blue
    colors["cnn_rnn"],
    "#BDBDBD", colors["large"], colors["large"], colors["large"],
]

bars = ax.bar(all_models.keys(), all_models.values(), color=model_colors, edgecolor="white", linewidth=0.5)

# Add value labels on bars
for bar, val in zip(bars, all_models.values()):
    if val < 5:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5,
                f"{val:.1f}%", ha="center", va="top", fontsize=9, fontweight="bold", color="white")

ax.set_ylabel("Character Error Rate (%)")
ax.set_title("Phase 1 — CER on clean synthetic Ogham",
             fontsize=14, fontweight="bold")
ax.axhline(100, linestyle=":", color="red", alpha=0.4)
ax.set_ylim(0, 120)

plt.tight_layout()
plt.savefig(output_dir / "model_comparison_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'model_comparison_bar.png'}")


# === Figure 3b: Phase 2 bar chart — same style, synthetic-freeform test ===
fig, ax = plt.subplots(figsize=(12, 5))

# Phase 2 post-curriculum CER on the 35-sample freeform test split.
# Large models: few-shot freeform eval from docs/extended_large_model_results.json.
phase2_models = {
    "TrOCR-small\n(phase 2)":     1.34,
    "TrOCR-base\n(phase 2)":      8.24,
    "PARSeq\n(phase 2)":         29.17,
    "CNN+RNN\n(phase 2)":        67.24,
    "GPT-4o\n(few-shot)":       141.57,
    "Claude\n(few-shot)":       109.00,
    "Gemini\n(few-shot)":        85.63,
}

phase2_colors = [
    colors["unfrozen"], "#03A9F4", colors["parseq"], colors["cnn_rnn"],
    colors["large"], colors["large"], colors["large"],
]

YMAX = 160  # give headroom for GPT-4o 141.57
bars = ax.bar(phase2_models.keys(), phase2_models.values(),
              color=phase2_colors, edgecolor="white", linewidth=0.5)

for bar, val in zip(bars, phase2_models.values()):
    if val < 5:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2.5,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 6,
                f"{val:.1f}%", ha="center", va="top", fontsize=9,
                fontweight="bold", color="white")

ax.set_ylabel("Character Error Rate (%)")
ax.set_title("Phase 2 — CER on freeform trace",
             fontsize=14, fontweight="bold")
ax.axhline(100, linestyle=":", color="red", alpha=0.4)
ax.set_ylim(0, YMAX)

plt.tight_layout()
plt.savefig(output_dir / "model_comparison_bar_phase2.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'model_comparison_bar_phase2.png'}")


# === Figure 3c: Phase 2 log-scale bar ===
fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(phase2_models.keys(), phase2_models.values(),
       color=phase2_colors, edgecolor="white", linewidth=0.5)

for bar, val in zip(ax.patches, phase2_models.values()):
    y_pos = val * 1.15
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            f"{val:.2f}%" if val < 1 else f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Character Error Rate (%, log scale)")
ax.set_title("Phase 2 — CER on freeform trace (log scale)",
             fontsize=14, fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(0.5, 300)

plt.tight_layout()
plt.savefig(output_dir / "model_comparison_bar_phase2_log.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'model_comparison_bar_phase2_log.png'}")

# === Figure 4: Log-scale CER bar (to show TrOCR detail) ===
fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(all_models.keys(), all_models.values(), color=model_colors, edgecolor="white", linewidth=0.5)

for bar, val in zip(ax.patches, all_models.values()):
    y_pos = val * 1.3 if val > 0.5 else val + 0.02
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            f"{val:.2f}%" if val < 1 else f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Character Error Rate (%, log scale)")
ax.set_title("Phase 1 — CER on clean synthetic Ogham (log scale)",
             fontsize=14, fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(0.01, 200)

plt.tight_layout()
plt.savefig(output_dir / "model_comparison_bar_log.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'model_comparison_bar_log.png'}")

print("\nAll model comparison charts saved to docs/figures/")
