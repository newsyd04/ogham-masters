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

# Large models (single point, not epoch-based)
large_models = {
    "GPT-4o\n(zero-shot)": 98.22,
    "GPT-4o\n(few-shot)": 93.23,
    "Claude\n(zero-shot)": 97.33,
    "Claude\n(few-shot)": 80.07,
}

plt.style.use("seaborn-v0_8-whitegrid")
colors = {
    "frozen": "#2196F3",
    "unfrozen": "#FF5722",
    "cnn_rnn": "#9C27B0",
    "large": "#607D8B",
}

# === Figure 1: All models CER over epochs ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Full range
ax1.plot(epochs, small_frozen_cer, "o-", color=colors["frozen"], label="TrOCR-small frozen", linewidth=2, markersize=4)
ax1.plot(epochs, small_unfrozen_cer, "s-", color=colors["unfrozen"], label="TrOCR-small unfrozen", linewidth=2, markersize=4)
ax1.plot(epochs, cnn_rnn_cer, "^-", color=colors["cnn_rnn"], label="CNN+RNN (CTC)", linewidth=2, markersize=4)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("CER (%)")
ax1.set_title("CER Over Training — All Models")
ax1.legend(fontsize=9)
ax1.set_ylim(-2, 80)

# Zoomed to TrOCR only (epochs 6+)
ax2.plot(epochs[5:], small_frozen_cer[5:], "o-", color=colors["frozen"], label="TrOCR-small frozen", linewidth=2, markersize=5)
ax2.plot(epochs, small_unfrozen_cer, "s-", color=colors["unfrozen"], label="TrOCR-small unfrozen", linewidth=2, markersize=5)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("CER (%)")
ax2.set_title("CER Zoomed — TrOCR Models Only")
ax2.legend(fontsize=9)
ax2.set_ylim(-0.1, 3.5)

fig.suptitle("Model Comparison: Character Error Rate", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(output_dir / "model_comparison_cer.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'model_comparison_cer.png'}")

# === Figure 2: Exact Match over epochs ===
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(epochs, small_frozen_exact, "o-", color=colors["frozen"], label="TrOCR-small frozen", linewidth=2, markersize=4)
ax.plot(epochs, small_unfrozen_exact, "s-", color=colors["unfrozen"], label="TrOCR-small unfrozen", linewidth=2, markersize=4)
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
    "TrOCR-small\nunfrozen": 0.06,
    "TrOCR-small\nfrozen": 0.14,
    "CNN+RNN\n(CTC)": 66.82,
    "Claude\n(few-shot)": 80.07,
    "GPT-4o\n(few-shot)": 93.23,
    "Claude\n(zero-shot)": 97.33,
    "GPT-4o\n(zero-shot)": 98.22,
}

model_colors = [
    colors["unfrozen"], colors["frozen"], colors["cnn_rnn"],
    colors["large"], colors["large"], colors["large"], colors["large"],
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
ax.set_title("Final CER Comparison — All Models", fontsize=14, fontweight="bold")
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig(output_dir / "model_comparison_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'model_comparison_bar.png'}")

# === Figure 4: Log-scale CER bar (to show TrOCR detail) ===
fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(all_models.keys(), all_models.values(), color=model_colors, edgecolor="white", linewidth=0.5)

for bar, val in zip(ax.patches, all_models.values()):
    y_pos = val * 1.3 if val > 0.5 else val + 0.02
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            f"{val:.2f}%" if val < 1 else f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Character Error Rate (%, log scale)")
ax.set_title("Final CER Comparison — Log Scale", fontsize=14, fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(0.01, 200)

plt.tight_layout()
plt.savefig(output_dir / "model_comparison_bar_log.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'model_comparison_bar_log.png'}")

print("\nAll model comparison charts saved to docs/figures/")
