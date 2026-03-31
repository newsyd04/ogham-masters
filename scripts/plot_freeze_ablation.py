#!/usr/bin/env python3
"""Plot freeze vs no-freeze encoder ablation results."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Frozen encoder (5 epochs frozen, then unfrozen) — from Phase 1
frozen_epochs = list(range(1, 21))
frozen_cer = [
    72.45, 67.86, 62.18, 59.79, 59.15,  # Frozen
    2.71, 1.69, 1.97, 1.00, 1.73,        # Unfrozen
    0.69, 0.64, 0.52, 0.52, 0.81,
    0.72, 0.20, 0.13, 0.14, 0.14,
]
frozen_val_loss = [
    0.6896, 0.6589, 0.6321, 0.6119, 0.5973,
    0.0325, 0.0174, 0.0160, 0.0108, 0.0082,
    0.0069, 0.0048, 0.0048, 0.0032, 0.0025,
    0.0018, 0.0012, 0.0007, 0.0005, 0.0004,
]
frozen_train_loss = [
    1.4863, 1.3356, 1.2789, 1.2386, 1.2051,
    0.1891, 0.0546, 0.0376, 0.0288, 0.0227,
    0.0182, 0.0145, 0.0116, 0.0092, 0.0068,
    0.0051, 0.0037, 0.0024, 0.0016, 0.0011,
]
frozen_exact = [
    10.8, 15.9, 15.7, 17.1, 19.6,
    85.4, 91.1, 91.0, 94.4, 91.3,
    96.1, 96.5, 97.3, 97.7, 98.0,
    97.6, 99.0, 99.4, 99.5, 99.4,
]

# No-freeze encoder — from ablation run
nofreeze_epochs = list(range(1, 21))
nofreeze_cer = [
    2.61, 1.69, 1.52, 1.09, 1.02,
    0.81, 0.66, 0.59, 0.56, 0.49,
    0.41, 0.32, 0.51, 0.21, 0.15,
    0.16, 0.06, 0.15, 0.06, 0.09,
]
nofreeze_val_loss = [
    0.0323, 0.0195, 0.0175, 0.0133, 0.0105,
    0.0090, 0.0074, 0.0066, 0.0064, 0.0049,
    0.0039, 0.0030, 0.0024, 0.0014, 0.0012,
    0.0006, 0.0005, 0.0002, 0.0001, 0.0000,
]
nofreeze_train_loss = [
    0.3358, 0.0626, 0.0442, 0.0350, 0.0289,
    0.0241, 0.0204, 0.0173, 0.0146, 0.0121,
    0.0100, 0.0081, 0.0063, 0.0048, 0.0034,
    0.0022, 0.0014, 0.0008, 0.0004, 0.0003,
]
nofreeze_exact = [
    85.6, 90.2, 91.5, 93.6, 94.2,
    95.3, 96.0, 96.7, 96.7, 97.0,
    97.4, 97.7, 96.1, 98.4, 99.0,
    99.2, 99.7, 99.4, 99.8, 99.8,
]

output_dir = Path("docs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
colors = {"frozen": "#2196F3", "nofreeze": "#FF5722"}

# --- Figure 1: CER Comparison ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Full range
ax1.plot(frozen_epochs, frozen_cer, "o-", color=colors["frozen"], label="Frozen (5 ep)", linewidth=2, markersize=5)
ax1.plot(nofreeze_epochs, nofreeze_cer, "s-", color=colors["nofreeze"], label="No Freeze", linewidth=2, markersize=5)
ax1.axvline(x=5.5, color="gray", linestyle="--", alpha=0.5, label="Encoder unfreezes")
ax1.axhspan(0, 1, alpha=0.05, color="green")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("CER (%)")
ax1.set_title("CER — Full Range")
ax1.legend()
ax1.set_ylim(bottom=-1)

# Zoomed (epoch 6+)
ax2.plot(frozen_epochs[5:], frozen_cer[5:], "o-", color=colors["frozen"], label="Frozen (5 ep)", linewidth=2, markersize=5)
ax2.plot(nofreeze_epochs, nofreeze_cer, "s-", color=colors["nofreeze"], label="No Freeze", linewidth=2, markersize=5)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("CER (%)")
ax2.set_title("CER — Zoomed (post-unfreeze)")
ax2.legend()
ax2.set_ylim(-0.1, 3.5)

fig.suptitle("Freeze vs No-Freeze: Character Error Rate", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(output_dir / "freeze_ablation_cer.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'freeze_ablation_cer.png'}")

# --- Figure 2: Validation Loss ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(frozen_epochs, frozen_val_loss, "o-", color=colors["frozen"], label="Frozen (5 ep)", linewidth=2, markersize=5)
ax1.plot(nofreeze_epochs, nofreeze_val_loss, "s-", color=colors["nofreeze"], label="No Freeze", linewidth=2, markersize=5)
ax1.axvline(x=5.5, color="gray", linestyle="--", alpha=0.5, label="Encoder unfreezes")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Validation Loss")
ax1.set_title("Val Loss — Full Range")
ax1.legend()

ax2.plot(frozen_epochs[5:], frozen_val_loss[5:], "o-", color=colors["frozen"], label="Frozen (5 ep)", linewidth=2, markersize=5)
ax2.plot(nofreeze_epochs, nofreeze_val_loss, "s-", color=colors["nofreeze"], label="No Freeze", linewidth=2, markersize=5)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation Loss")
ax2.set_title("Val Loss — Zoomed (post-unfreeze)")
ax2.legend()
ax2.set_ylim(-0.001, 0.04)

fig.suptitle("Freeze vs No-Freeze: Validation Loss", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(output_dir / "freeze_ablation_val_loss.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'freeze_ablation_val_loss.png'}")

# --- Figure 3: Exact Match ---
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(frozen_epochs, frozen_exact, "o-", color=colors["frozen"], label="Frozen (5 ep)", linewidth=2, markersize=5)
ax.plot(nofreeze_epochs, nofreeze_exact, "s-", color=colors["nofreeze"], label="No Freeze", linewidth=2, markersize=5)
ax.axvline(x=5.5, color="gray", linestyle="--", alpha=0.5, label="Encoder unfreezes")
ax.axhline(y=99, color="green", linestyle=":", alpha=0.4, label="99% target")
ax.set_xlabel("Epoch")
ax.set_ylabel("Exact Match (%)")
ax.set_title("Freeze vs No-Freeze: Exact Match Rate", fontsize=14, fontweight="bold")
ax.legend()
ax.set_ylim(0, 101)

plt.tight_layout()
plt.savefig(output_dir / "freeze_ablation_exact_match.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'freeze_ablation_exact_match.png'}")

# --- Figure 4: Train Loss (log scale) ---
fig, ax = plt.subplots(figsize=(10, 5))

ax.semilogy(frozen_epochs, frozen_train_loss, "o-", color=colors["frozen"], label="Frozen (5 ep)", linewidth=2, markersize=5)
ax.semilogy(nofreeze_epochs, nofreeze_train_loss, "s-", color=colors["nofreeze"], label="No Freeze", linewidth=2, markersize=5)
ax.axvline(x=5.5, color="gray", linestyle="--", alpha=0.5, label="Encoder unfreezes")
ax.set_xlabel("Epoch")
ax.set_ylabel("Train Loss (log scale)")
ax.set_title("Freeze vs No-Freeze: Training Loss", fontsize=14, fontweight="bold")
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / "freeze_ablation_train_loss.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {output_dir / 'freeze_ablation_train_loss.png'}")

print("\nAll freeze ablation charts saved to docs/figures/")
