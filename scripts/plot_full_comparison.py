#!/usr/bin/env python3
"""Regenerate Phase 1 and Phase 2 comparison bar charts.

Phase 1 chart: all architectures + large models on synth (clean)
Phase 2 chart: all architectures + large models on synth-freeform

Data comes from:
- docs/phase2_comparison.json (fine-tuned models)
- EXTENDED_LM_RESULTS (below, fill in after running notebook 05 extended cells)

Usage:
    python scripts/plot_full_comparison.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
DOCS = PROJECT / "docs"
FIGURES = DOCS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


# Fine-tuned results (from phase2_comparison.json)
with open(DOCS / "phase2_comparison.json") as f:
    phase2_data = json.load(f)

# Large-model results from real-stone eval (previously measured)
# These use the REAL stone test set (29 images). Kept for reference.
REAL_STONE_LM = {
    "GPT-4o (few-shot)": 93.23,
    "Claude (few-shot)": 80.07,
    "Gemini": None,  # rate-limited
}

# Extended eval results (large models on synth + synth-freeform).
# Fill these in AFTER running the notebook 05 extended cells.
# Structure: {model_name: {'synth_cer': float, 'synth_exact': float,
#                          'freeform_cer': float, 'freeform_exact': float}}
EXTENDED_LM_RESULTS = {
    "GPT-4o (few-shot)":  {"synth_cer": None, "synth_exact": None,
                            "freeform_cer": None, "freeform_exact": None},
    "Claude (few-shot)":  {"synth_cer": None, "synth_exact": None,
                            "freeform_cer": None, "freeform_exact": None},
    "Gemini (few-shot)":  {"synth_cer": None, "synth_exact": None,
                            "freeform_cer": None, "freeform_exact": None},
}

# Try to auto-load from Drive-path-style export if present
EXTENDED_JSON_LOCAL = DOCS / "extended_large_model_results.json"
if EXTENDED_JSON_LOCAL.exists():
    with open(EXTENDED_JSON_LOCAL) as f:
        raw = json.load(f)
    # Map raw keys into EXTENDED_LM_RESULTS
    for disp_name, raw_key in [
        ("GPT-4o (few-shot)", "gpt4o_few_shot"),
        ("Claude (few-shot)", "claude_few_shot"),
        ("Gemini (few-shot)", "gemini_few_shot"),
    ]:
        ff = raw.get("synth_freeform", {}).get(raw_key, {}).get("aggregate")
        sy = raw.get("synth_clean", {}).get(raw_key, {}).get("aggregate")
        if ff:
            EXTENDED_LM_RESULTS[disp_name]["freeform_cer"] = ff.get("mean_cer_no_sp_pct")
            EXTENDED_LM_RESULTS[disp_name]["freeform_exact"] = ff.get("exact_match_pct")
        if sy:
            EXTENDED_LM_RESULTS[disp_name]["synth_cer"] = sy.get("mean_cer_no_sp_pct")
            EXTENDED_LM_RESULTS[disp_name]["synth_exact"] = sy.get("exact_match_pct")
    print(f"Loaded extended large-model results from {EXTENDED_JSON_LOCAL}")
else:
    print(f"(Extended LM results not found at {EXTENDED_JSON_LOCAL} — "
          f"download from Drive after running notebook 05 cells)")


def chart_phase1_comparison():
    """Bar chart: all models' CER on clean synthetic."""
    models = [
        ("TrOCR-small\n(fine-tuned)", phase2_data["models"]["trocr_small"]["phase1_synth_cer_pct"], "#2ca02c"),
        ("PARSeq\n(fine-tuned)",      phase2_data["models"]["parseq"]["phase1_synth_cer_pct"],       "#ff7f0e"),
        ("CNN+RNN\n(fine-tuned)",     phase2_data["models"]["cnn_rnn"]["phase1_synth_cer_pct"],      "#d62728"),
    ]
    # Add large models if measurements exist
    for name, vals in EXTENDED_LM_RESULTS.items():
        if vals["synth_cer"] is not None:
            models.append((name, vals["synth_cer"], "#7f7f7f"))
    # TrOCR unfinetuned floor for context
    models.append(("TrOCR-small\n(unfinetuned)", 100.12, "#999999"))

    labels = [m[0] for m in models]
    cers = [m[1] for m in models]
    colors = [m[2] for m in models]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(labels)), cers, color=colors, edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=0)
    ax.set_ylabel("Character Error Rate (%)", fontsize=11)
    ax.set_title("Phase 1 comparison — CER on clean synthetic Ogham test set",
                 fontsize=13, pad=12)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(100, linestyle=":", color="red", alpha=0.5)

    for bar, cer in zip(bars, cers):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{cer:.2f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = FIGURES / "phase1_full_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def chart_phase2_comparison():
    """Bar chart: all models' CER on synth-freeform test."""
    models = [
        ("TrOCR-small",    phase2_data["models"]["trocr_small"]["post_p2_freeform_cer_pct"], "#2ca02c"),
        ("PARSeq",         phase2_data["models"]["parseq"]["post_p2_freeform_cer_pct"],       "#ff7f0e"),
        ("CNN+RNN",        phase2_data["models"]["cnn_rnn"]["post_p2_freeform_cer_pct"],      "#d62728"),
    ]
    for name, vals in EXTENDED_LM_RESULTS.items():
        if vals["freeform_cer"] is not None:
            models.append((name, vals["freeform_cer"], "#7f7f7f"))

    if not any(vals["freeform_cer"] is not None for vals in EXTENDED_LM_RESULTS.values()):
        print("  (Phase 2 chart: no large-model numbers yet — showing fine-tuned only)")

    labels = [m[0] for m in models]
    cers = [m[1] for m in models]
    colors = [m[2] for m in models]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(labels)), cers, color=colors, edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Character Error Rate (%)", fontsize=11)
    ax.set_title("Phase 2 comparison — CER on synthetic-freeform test set",
                 fontsize=13, pad=12)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, cer in zip(bars, cers):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{cer:.2f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = FIGURES / "phase2_full_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    chart_phase1_comparison()
    chart_phase2_comparison()
