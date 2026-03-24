"""Plot training comparison charts for Ogham vs Latin modes."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# --- Data from comparison run 2026-03-23 ---

epochs = list(range(1, 21))

ogham = {
    "train_loss": [1.4863, 1.3356, 1.2789, 1.2386, 1.2051, 0.1891, 0.0546, 0.0376, 0.0288, 0.0227,
                   0.0182, 0.0145, 0.0116, 0.0092, 0.0068, 0.0051, 0.0037, 0.0024, 0.0016, 0.0011],
    "val_loss":   [0.6896, 0.6589, 0.6321, 0.6119, 0.5973, 0.0325, 0.0174, 0.0160, 0.0108, 0.0082,
                   0.0069, 0.0048, 0.0048, 0.0032, 0.0025, 0.0018, 0.0012, 0.0007, 0.0005, 0.0004],
    "cer":        [72.45, 67.86, 62.18, 59.79, 59.15, 2.71, 1.69, 1.97, 1.00, 1.73,
                   0.69, 0.64, 0.52, 0.52, 0.81, 0.72, 0.20, 0.13, 0.14, 0.14],
    "exact":      [10.8, 15.9, 15.7, 17.1, 19.6, 85.4, 91.1, 91.0, 94.4, 91.3,
                   96.1, 96.5, 97.3, 97.7, 98.0, 97.6, 99.0, 99.4, 99.5, 99.4],
}

latin = {
    "train_loss": [2.3539, 2.1713, 2.0909, 2.0345, 1.9882, 0.5651, 0.2088, 0.1536, 0.1235, 0.1011,
                   0.0830, 0.0675, 0.0540, 0.0420, 0.0314, 0.0224, 0.0156, 0.0109, 0.0084, 0.0070],
    "val_loss":   [1.1196, 1.0662, 1.0289, 1.0008, 0.9768, 0.1074, 0.0722, 0.0589, 0.0449, 0.0382,
                   0.0315, 0.0250, 0.0188, 0.0139, 0.0101, 0.0066, 0.0044, 0.0030, 0.0023, 0.0022],
    "cer":        [71.51, 63.79, 60.29, 58.69, 56.23, 3.84, 2.48, 1.84, 1.70, 1.50,
                   0.90, 0.75, 0.57, 0.61, 0.36, 0.31, 0.21, 0.20, 0.18, 0.16],
    "exact":      [9.9, 15.9, 17.7, 19.6, 21.1, 81.1, 86.8, 90.8, 92.4, 93.0,
                   95.0, 96.0, 97.1, 97.4, 98.2, 98.4, 99.0, 99.1, 99.3, 99.2],
}

out_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

# Colours
OGHAM_C = "#2E86AB"  # blue
LATIN_C = "#E8430C"  # orange-red
FREEZE_C = "#DDDDDD"

def add_freeze_band(ax):
    """Shade the frozen-encoder region (epochs 1-5)."""
    ax.axvspan(0.5, 5.5, alpha=0.15, color=FREEZE_C, label="Encoder frozen")
    ax.axvline(5.5, color="#999999", linestyle="--", linewidth=0.8, alpha=0.6)


# ── Figure 1: CER over epochs (two panels: full + zoomed) ──────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: full range
ax1.plot(epochs, ogham["cer"], "o-", color=OGHAM_C, markersize=4, linewidth=1.5, label="Ogham Unicode")
ax1.plot(epochs, latin["cer"], "s-", color=LATIN_C, markersize=4, linewidth=1.5, label="Latin BPE")
add_freeze_band(ax1)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("CER (%)")
ax1.set_title("Character Error Rate — Full Range")
ax1.set_xticks(epochs)
ax1.legend(loc="upper right")
ax1.set_ylim(-1, 80)
ax1.grid(True, alpha=0.3)

# Right: epochs 6-20 zoomed
ax2.plot(epochs[5:], ogham["cer"][5:], "o-", color=OGHAM_C, markersize=5, linewidth=1.5, label="Ogham Unicode")
ax2.plot(epochs[5:], latin["cer"][5:], "s-", color=LATIN_C, markersize=5, linewidth=1.5, label="Latin BPE")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("CER (%)")
ax2.set_title("CER — Post-Unfreeze (epochs 6–20)")
ax2.set_xticks(epochs[5:])
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

# Annotate best CER
ax2.annotate(f"0.13%\n(ep 18)", xy=(18, 0.13), fontsize=8, color=OGHAM_C,
             ha="center", va="bottom", fontweight="bold",
             xytext=(18, 0.6), arrowprops=dict(arrowstyle="->", color=OGHAM_C, lw=0.8))
ax2.annotate(f"0.16%\n(ep 20)", xy=(20, 0.16), fontsize=8, color=LATIN_C,
             ha="center", va="bottom", fontweight="bold",
             xytext=(20, 0.9), arrowprops=dict(arrowstyle="->", color=LATIN_C, lw=0.8))

fig.suptitle("Ogham vs Latin — CER Comparison", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(out_dir / "cer_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved: {out_dir / 'cer_comparison.png'}")


# ── Figure 2: Validation Loss ──────────────────────────────────────────

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

ax3.plot(epochs, ogham["val_loss"], "o-", color=OGHAM_C, markersize=4, linewidth=1.5, label="Ogham Unicode")
ax3.plot(epochs, latin["val_loss"], "s-", color=LATIN_C, markersize=4, linewidth=1.5, label="Latin BPE")
add_freeze_band(ax3)
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Validation Loss")
ax3.set_title("Validation Loss — Full Range")
ax3.set_xticks(epochs)
ax3.legend(loc="upper right")
ax3.grid(True, alpha=0.3)

ax4.plot(epochs[5:], ogham["val_loss"][5:], "o-", color=OGHAM_C, markersize=5, linewidth=1.5, label="Ogham Unicode")
ax4.plot(epochs[5:], latin["val_loss"][5:], "s-", color=LATIN_C, markersize=5, linewidth=1.5, label="Latin BPE")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Validation Loss")
ax4.set_title("Validation Loss — Post-Unfreeze (epochs 6–20)")
ax4.set_xticks(epochs[5:])
ax4.legend(loc="upper right")
ax4.grid(True, alpha=0.3)

fig2.suptitle("Ogham vs Latin — Validation Loss", fontsize=14, fontweight="bold")
fig2.tight_layout()
fig2.savefig(out_dir / "val_loss_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved: {out_dir / 'val_loss_comparison.png'}")


# ── Figure 3: Exact Match ──────────────────────────────────────────────

fig3, ax5 = plt.subplots(figsize=(10, 5))

ax5.plot(epochs, ogham["exact"], "o-", color=OGHAM_C, markersize=5, linewidth=1.5, label="Ogham Unicode")
ax5.plot(epochs, latin["exact"], "s-", color=LATIN_C, markersize=5, linewidth=1.5, label="Latin BPE")
add_freeze_band(ax5)
ax5.set_xlabel("Epoch")
ax5.set_ylabel("Exact Match (%)")
ax5.set_title("Ogham vs Latin — Exact Match Rate", fontsize=14, fontweight="bold")
ax5.set_xticks(epochs)
ax5.set_ylim(0, 105)
ax5.legend(loc="lower right")
ax5.grid(True, alpha=0.3)

# Annotate the unfreeze jump
ax5.annotate("Encoder\nunfrozen", xy=(5.5, 50), fontsize=9, color="#666666",
             ha="center", va="center", style="italic")

fig3.tight_layout()
fig3.savefig(out_dir / "exact_match_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved: {out_dir / 'exact_match_comparison.png'}")


# ── Figure 4: Train Loss (log scale) ──────────────────────────────────

fig4, ax6 = plt.subplots(figsize=(10, 5))

ax6.semilogy(epochs, ogham["train_loss"], "o-", color=OGHAM_C, markersize=5, linewidth=1.5, label="Ogham Unicode")
ax6.semilogy(epochs, latin["train_loss"], "s-", color=LATIN_C, markersize=5, linewidth=1.5, label="Latin BPE")
add_freeze_band(ax6)
ax6.set_xlabel("Epoch")
ax6.set_ylabel("Train Loss (log scale)")
ax6.set_title("Ogham vs Latin — Training Loss", fontsize=14, fontweight="bold")
ax6.set_xticks(epochs)
ax6.legend(loc="upper right")
ax6.grid(True, alpha=0.3, which="both")

fig4.tight_layout()
fig4.savefig(out_dir / "train_loss_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved: {out_dir / 'train_loss_comparison.png'}")

print(f"\nAll figures saved to {out_dir}/")
