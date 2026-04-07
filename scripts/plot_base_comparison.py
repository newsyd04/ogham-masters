#!/usr/bin/env python3
"""Plot TrOCR-small vs TrOCR-base comparison charts."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

output_dir = Path("docs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

epochs = list(range(1, 21))

# TrOCR-small unfrozen (complete 20 epochs)
small_cer = [
    2.61, 1.69, 1.52, 1.09, 1.02,
    0.81, 0.66, 0.59, 0.56, 0.49,
    0.41, 0.32, 0.51, 0.21, 0.15,
    0.16, 0.06, 0.15, 0.06, 0.09,
]
small_train_loss = [
    0.3358, 0.0626, 0.0442, 0.0350, 0.0289,
    0.0241, 0.0204, 0.0173, 0.0146, 0.0121,
    0.0100, 0.0081, 0.0063, 0.0048, 0.0034,
    0.0022, 0.0014, 0.0008, 0.0004, 0.0003,
]
small_exact = [
    85.6, 90.2, 91.5, 93.6, 94.2,
    95.3, 96.0, 96.7, 96.7, 97.0,
    97.4, 97.7, 96.1, 98.4, 99.0,
    99.2, 99.7, 99.4, 99.8, 99.8,
]
small_val_loss = [
    0.0323, 0.0195, 0.0175, 0.0133, 0.0105,
    0.0090, 0.0074, 0.0066, 0.0064, 0.0049,
    0.0039, 0.0030, 0.0024, 0.0014, 0.0012,
    0.0006, 0.0005, 0.0002, 0.0001, 0.0000,
]

# TrOCR-base unfrozen batch-16 lr 5e-5 (latest run, 3 epochs so far)
base_epochs_b16 = [1, 2, 3]
base_cer_b16 = [90.43, 91.37, 99.57]
base_train_loss_b16 = [0.2670, 0.1088, 0.1008]
base_exact_b16 = [17.1, 14.9, 1.0]
base_val_loss_b16 = [0.2487, 0.3105, 0.4754]

# TrOCR-base unfrozen batch-64 (earlier run, 7 epochs before disconnect)
base_epochs_b64 = [1, 2, 3, 4, 5, 6, 7]
base_cer_b64 = [32.43, 28.55, 40.62, 104.22, 100.02, 110.35, 103.18]
base_train_loss_b64 = [1.2340, 0.6366, 0.4666, 0.3561, 0.2635, 0.2172, 0.1771]
base_exact_b64 = [8.4, 25.2, 18.5, 0.0, 0.0, 0.0, 0.0]

# CNN+RNN for reference
cnn_rnn_cer = [
    71.42, 68.58, 70.70, 68.33, 68.63,
    68.02, 68.85, 68.12, 68.31, 67.84,
    67.88, 67.14, 67.87, 66.96, 67.05,
    66.87, 66.95, 67.15, 66.86, 66.82,
]

plt.style.use("seaborn-v0_8-whitegrid")
colors = {
    "small": "#FF5722",
    "base_b16": "#2196F3",
    "base_b64": "#03A9F4",
    "cnn_rnn": "#9C27B0",
}

# === Figure 1: CER comparison (all models, full range) ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: full range showing how bad base is
ax1.plot(epochs, small_cer, "s-", color=colors["small"], label="TrOCR-small (62M)", linewidth=2, markersize=5)
ax1.plot(base_epochs_b16, base_cer_b16, "o-", color=colors["base_b16"], label="TrOCR-base batch=16 (385M)", linewidth=2, markersize=6)
ax1.plot(base_epochs_b64, base_cer_b64, "^--", color=colors["base_b64"], label="TrOCR-base batch=64 (385M)", linewidth=1.5, markersize=5, alpha=0.7)
ax1.plot(epochs, cnn_rnn_cer, "d-", color=colors["cnn_rnn"], label="CNN+RNN (15M)", linewidth=1.5, markersize=4, alpha=0.6)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("CER (%)")
ax1.set_title("CER — All Models")
ax1.legend(fontsize=8)
ax1.set_ylim(-2, 115)
ax1.axhline(y=100, color="red", linestyle=":", alpha=0.3)

# Right: zoomed to TrOCR-small only
ax2.plot(epochs, small_cer, "s-", color=colors["small"], label="TrOCR-small (62M)", linewidth=2, markersize=5)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("CER (%)")
ax2.set_title("CER — TrOCR-small Detail")
ax2.legend()
ax2.set_ylim(-0.05, 3)

fig.suptitle("TrOCR-small vs TrOCR-base: Character Error Rate", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(output_dir / "base_vs_small_cer.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'base_vs_small_cer.png'}")

# === Figure 2: Train loss comparison ===
fig, ax = plt.subplots(figsize=(10, 5))

ax.semilogy(epochs, small_train_loss, "s-", color=colors["small"], label="TrOCR-small (62M)", linewidth=2, markersize=5)
ax.semilogy(base_epochs_b16, base_train_loss_b16, "o-", color=colors["base_b16"], label="TrOCR-base batch=16 (385M)", linewidth=2, markersize=6)
ax.semilogy(base_epochs_b64, base_train_loss_b64, "^--", color=colors["base_b64"], label="TrOCR-base batch=64 (385M)", linewidth=1.5, markersize=5, alpha=0.7)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss (log scale)")
ax.set_title("Training Loss: Small vs Base", fontsize=14, fontweight="bold")
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / "base_vs_small_train_loss.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'base_vs_small_train_loss.png'}")

# === Figure 3: Exact match comparison ===
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(epochs, small_exact, "s-", color=colors["small"], label="TrOCR-small (62M)", linewidth=2, markersize=5)
ax.plot(base_epochs_b16, base_exact_b16, "o-", color=colors["base_b16"], label="TrOCR-base batch=16 (385M)", linewidth=2, markersize=6)
ax.plot(base_epochs_b64, base_exact_b64, "^--", color=colors["base_b64"], label="TrOCR-base batch=64 (385M)", linewidth=1.5, markersize=5, alpha=0.7)
ax.plot(epochs, [24.8]*20, "d--", color=colors["cnn_rnn"], label="CNN+RNN (15M)", linewidth=1, alpha=0.5)
ax.axhline(y=99, color="green", linestyle=":", alpha=0.4, label="99% target")
ax.set_xlabel("Epoch")
ax.set_ylabel("Exact Match (%)")
ax.set_title("Exact Match Rate: Small vs Base", fontsize=14, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(-2, 102)

plt.tight_layout()
plt.savefig(output_dir / "base_vs_small_exact.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'base_vs_small_exact.png'}")

# === Figure 4: Val loss divergence ===
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(epochs, small_val_loss, "s-", color=colors["small"], label="TrOCR-small (62M)", linewidth=2, markersize=5)
ax.plot(base_epochs_b16, base_val_loss_b16, "o-", color=colors["base_b16"], label="TrOCR-base batch=16 (385M)", linewidth=2, markersize=6)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Loss")
ax.set_title("Validation Loss: Small Converges, Base Diverges", fontsize=14, fontweight="bold")
ax.legend()
ax.annotate("Base: val loss increasing\n(overfitting to train)", xy=(3, 0.48), fontsize=10,
            color=colors["base_b16"], ha="center",
            arrowprops=dict(arrowstyle="->", color=colors["base_b16"]),
            xytext=(5, 0.55))

plt.tight_layout()
plt.savefig(output_dir / "base_vs_small_val_loss.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'base_vs_small_val_loss.png'}")

# === Figure 5: Final summary bar chart ===
fig, ax = plt.subplots(figsize=(10, 5))

models = ["TrOCR-small\nunfrozen\n(62M)", "TrOCR-base\nbatch=16\n(385M)", "TrOCR-base\nbatch=64\n(385M)", "CNN+RNN\n(15M)"]
best_cers = [0.06, 90.43, 28.55, 66.82]
bar_colors = [colors["small"], colors["base_b16"], colors["base_b64"], colors["cnn_rnn"]]

bars = ax.bar(models, best_cers, color=bar_colors, edgecolor="white", linewidth=0.5)

for bar, val in zip(bars, best_cers):
    if val < 5:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 4,
                f"{val:.1f}%", ha="center", va="top", fontsize=11, fontweight="bold", color="white")

ax.set_ylabel("Best CER (%)")
ax.set_title("Best CER by Model Size — Bigger Is Not Better", fontsize=14, fontweight="bold")
ax.set_ylim(0, 115)

plt.tight_layout()
plt.savefig(output_dir / "model_size_cer_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'model_size_cer_bar.png'}")

print("\nAll base vs small comparison charts saved to docs/figures/")
