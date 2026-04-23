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

# Track API-error placeholders separately so we can annotate them in charts
EXTENDED_LM_STATUS = {}

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
            EXTENDED_LM_STATUS.setdefault(disp_name, {})["freeform_status"] = ff.get("status", "valid_eval")
        if sy:
            EXTENDED_LM_RESULTS[disp_name]["synth_cer"] = sy.get("mean_cer_no_sp_pct")
            EXTENDED_LM_RESULTS[disp_name]["synth_exact"] = sy.get("exact_match_pct")
            EXTENDED_LM_STATUS.setdefault(disp_name, {})["synth_status"] = sy.get("status", "valid_eval")
    print(f"Loaded extended large-model results from {EXTENDED_JSON_LOCAL}")
else:
    print(f"(Extended LM results not found at {EXTENDED_JSON_LOCAL} — "
          f"download from Drive after running notebook 05 cells)")


def _status_for(model_name, split_key):
    """Return API-error annotation label if this cell was a placeholder."""
    st = EXTENDED_LM_STATUS.get(model_name, {}).get(split_key)
    if st and st != "valid_eval":
        return "API error"
    return None


def chart_phase1_comparison():
    """Bar chart: all models' CER on clean synthetic."""
    models = [
        ("TrOCR-small\n(fine-tuned)", phase2_data["models"]["trocr_small"]["phase1_synth_cer_pct"], "#2ca02c", None),
        ("PARSeq\n(fine-tuned)",      phase2_data["models"]["parseq"]["phase1_synth_cer_pct"],       "#ff7f0e", None),
        ("CNN+RNN\n(fine-tuned)",     phase2_data["models"]["cnn_rnn"]["phase1_synth_cer_pct"],      "#d62728", None),
    ]
    for name, vals in EXTENDED_LM_RESULTS.items():
        if vals["synth_cer"] is not None:
            status = _status_for(name, "synth_status")
            models.append((name, vals["synth_cer"], "#7f7f7f", status))
    models.append(("TrOCR-small\n(unfinetuned)", 100.12, "#999999", None))

    labels = [m[0] for m in models]
    cers = [m[1] for m in models]
    colors = [m[2] for m in models]
    statuses = [m[3] for m in models]

    # Cap display at 110 for visual clarity; annotate bars that exceed
    display_cers = [min(c, 110) for c in cers]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(labels)), display_cers, color=colors, edgecolor="black")
    # Stripe pattern for API-errored bars
    for bar, status in zip(bars, statuses):
        if status == "API error":
            bar.set_hatch("///")
            bar.set_alpha(0.6)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=0)
    ax.set_ylabel("Character Error Rate (%)", fontsize=11)
    ax.set_title("Phase 1 comparison — CER on clean synthetic Ogham test set",
                 fontsize=13, pad=12)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(100, linestyle=":", color="red", alpha=0.5)

    for bar, cer, status in zip(bars, cers, statuses):
        h = bar.get_height()
        label_top = f"{cer:.2f}%" if cer <= 110 else f"{cer:.2f}%↑"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, label_top,
                ha="center", va="bottom", fontsize=9)
        if status == "API error":
            ax.text(bar.get_x() + bar.get_width() / 2, h / 2, "API\nerror",
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold", rotation=90)

    plt.tight_layout()
    out = FIGURES / "phase1_full_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def chart_phase2_comparison():
    """Bar chart: all models' CER on freeform trace test."""
    models = [
        ("TrOCR-small",    phase2_data["models"]["trocr_small"]["post_p2_freeform_cer_pct"], "#2ca02c", None),
        ("PARSeq",         phase2_data["models"]["parseq"]["post_p2_freeform_cer_pct"],       "#ff7f0e", None),
        ("CNN+RNN",        phase2_data["models"]["cnn_rnn"]["post_p2_freeform_cer_pct"],      "#d62728", None),
    ]
    for name, vals in EXTENDED_LM_RESULTS.items():
        if vals["freeform_cer"] is not None:
            status = _status_for(name, "freeform_status")
            models.append((name, vals["freeform_cer"], "#7f7f7f", status))

    if not any(vals["freeform_cer"] is not None for vals in EXTENDED_LM_RESULTS.values()):
        print("  (Phase 2 chart: no large-model numbers yet — showing fine-tuned only)")

    labels = [m[0] for m in models]
    cers = [m[1] for m in models]
    colors = [m[2] for m in models]
    statuses = [m[3] for m in models]
    display_cers = [min(c, 150) for c in cers]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(labels)), display_cers, color=colors, edgecolor="black")
    for bar, status in zip(bars, statuses):
        if status == "API error":
            bar.set_hatch("///")
            bar.set_alpha(0.6)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Character Error Rate (%)", fontsize=11)
    ax.set_title("Phase 2 comparison — CER on synthetic-freeform test set",
                 fontsize=13, pad=12)
    ax.set_ylim(0, 160)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.axhline(100, linestyle=":", color="red", alpha=0.5, label="100% CER floor")

    for bar, cer, status in zip(bars, cers, statuses):
        h = bar.get_height()
        label_top = f"{cer:.2f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, label_top,
                ha="center", va="bottom", fontsize=9)
        if status == "API error":
            ax.text(bar.get_x() + bar.get_width() / 2, h / 2, "API\nerror",
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold", rotation=90)

    plt.tight_layout()
    out = FIGURES / "phase2_full_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def chart_phase1_exact_match():
    """Bar chart: all models' exact-match on clean synthetic."""
    models = [
        ("TrOCR-small\n(fine-tuned)", phase2_data["models"]["trocr_small"].get("phase1_synth_exact_pct", 99.8), "#2ca02c"),
        ("PARSeq\n(fine-tuned)",      phase2_data["models"]["parseq"].get("phase1_synth_exact_pct", 72.2),       "#ff7f0e"),
        ("CNN+RNN\n(fine-tuned)",     phase2_data["models"]["cnn_rnn"].get("phase1_synth_exact_pct", 24.8),      "#d62728"),
    ]
    for name, vals in EXTENDED_LM_RESULTS.items():
        if vals["synth_exact"] is not None:
            models.append((name, vals["synth_exact"], "#7f7f7f"))
    models.append(("TrOCR-small\n(unfinetuned)", 0.0, "#999999"))

    labels = [m[0] for m in models]
    exacts = [m[1] for m in models]
    colors = [m[2] for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(labels)), exacts, color=colors, edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=0)
    ax.set_ylabel("Exact match rate (%)", fontsize=11)
    ax.set_title("Phase 1 comparison — exact-match on clean synthetic Ogham test set",
                 fontsize=13, pad=12)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, ex in zip(bars, exacts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{ex:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = FIGURES / "phase1_exact_match_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def chart_phase2_exact_match():
    """Bar chart: all models' exact-match on synth-freeform test."""
    models = [
        ("TrOCR-small", phase2_data["models"]["trocr_small"]["post_p2_freeform_exact_pct"], "#2ca02c"),
        ("PARSeq",      phase2_data["models"]["parseq"]["post_p2_freeform_exact_pct"],       "#ff7f0e"),
        ("CNN+RNN",     phase2_data["models"]["cnn_rnn"]["post_p2_freeform_exact_pct"],      "#d62728"),
    ]
    for name, vals in EXTENDED_LM_RESULTS.items():
        if vals["freeform_exact"] is not None:
            models.append((name, vals["freeform_exact"], "#7f7f7f"))

    labels = [m[0] for m in models]
    exacts = [m[1] for m in models]
    colors = [m[2] for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(labels)), exacts, color=colors, edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Exact match rate (%)", fontsize=11)
    ax.set_title("Phase 2 comparison — exact-match on synthetic-freeform test set",
                 fontsize=13, pad=12)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for bar, ex in zip(bars, exacts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{ex:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = FIGURES / "phase2_exact_match_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def chart_phase1_dual():
    """Two-panel figure: CER and exact-match for Phase 1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: CER
    p1_models_cer = [
        ("TrOCR-small\n(fine-tuned)", phase2_data["models"]["trocr_small"]["phase1_synth_cer_pct"], "#2ca02c"),
        ("PARSeq\n(fine-tuned)",      phase2_data["models"]["parseq"]["phase1_synth_cer_pct"],       "#ff7f0e"),
        ("CNN+RNN\n(fine-tuned)",     phase2_data["models"]["cnn_rnn"]["phase1_synth_cer_pct"],      "#d62728"),
    ]
    for name, vals in EXTENDED_LM_RESULTS.items():
        if vals["synth_cer"] is not None:
            p1_models_cer.append((name, vals["synth_cer"], "#7f7f7f"))
    p1_models_cer.append(("TrOCR-small\n(unfinetuned)", 100.12, "#999999"))

    labels = [m[0] for m in p1_models_cer]
    cers = [m[1] for m in p1_models_cer]
    colors = [m[2] for m in p1_models_cer]
    display_cers = [min(c, 115) for c in cers]

    bars1 = ax1.bar(range(len(labels)), display_cers, color=colors, edgecolor="black")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8, rotation=0)
    ax1.set_ylabel("Character Error Rate (%)", fontsize=10)
    ax1.set_title("(a) CER — lower is better", fontsize=11)
    ax1.set_ylim(0, 120)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.axhline(100, linestyle=":", color="red", alpha=0.5)
    for bar, cer in zip(bars1, cers):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                 f"{cer:.2f}%", ha="center", va="bottom", fontsize=8)

    # Right: exact match
    p1_models_ex = [
        ("TrOCR-small\n(fine-tuned)", phase2_data["models"]["trocr_small"].get("phase1_synth_exact_pct", 99.8), "#2ca02c"),
        ("PARSeq\n(fine-tuned)",      phase2_data["models"]["parseq"].get("phase1_synth_exact_pct", 72.2),       "#ff7f0e"),
        ("CNN+RNN\n(fine-tuned)",     phase2_data["models"]["cnn_rnn"].get("phase1_synth_exact_pct", 24.8),      "#d62728"),
    ]
    for name, vals in EXTENDED_LM_RESULTS.items():
        if vals["synth_exact"] is not None:
            p1_models_ex.append((name, vals["synth_exact"], "#7f7f7f"))
    p1_models_ex.append(("TrOCR-small\n(unfinetuned)", 0.0, "#999999"))

    exacts = [m[1] for m in p1_models_ex]
    bars2 = ax2.bar(range(len(labels)), exacts, color=colors, edgecolor="black")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8, rotation=0)
    ax2.set_ylabel("Exact match rate (%)", fontsize=10)
    ax2.set_title("(b) Exact match — higher is better", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, ex in zip(bars2, exacts):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1,
                 f"{ex:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Phase 1 comparison on 5,000-sample clean synthetic Ogham val set",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES / "phase1_dual_metric.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def chart_phase2_dual():
    """Two-panel figure: CER and exact-match for Phase 2."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: CER
    p2_models_cer = [
        ("TrOCR-small", phase2_data["models"]["trocr_small"]["post_p2_freeform_cer_pct"], "#2ca02c"),
        ("PARSeq",      phase2_data["models"]["parseq"]["post_p2_freeform_cer_pct"],       "#ff7f0e"),
        ("CNN+RNN",     phase2_data["models"]["cnn_rnn"]["post_p2_freeform_cer_pct"],      "#d62728"),
    ]
    for name, vals in EXTENDED_LM_RESULTS.items():
        if vals["freeform_cer"] is not None:
            p2_models_cer.append((name, vals["freeform_cer"], "#7f7f7f"))

    labels = [m[0] for m in p2_models_cer]
    cers = [m[1] for m in p2_models_cer]
    colors = [m[2] for m in p2_models_cer]
    display_cers = [min(c, 150) for c in cers]

    bars1 = ax1.bar(range(len(labels)), display_cers, color=colors, edgecolor="black")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax1.set_ylabel("Character Error Rate (%)", fontsize=10)
    ax1.set_title("(a) CER — lower is better", fontsize=11)
    ax1.set_ylim(0, 160)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.axhline(100, linestyle=":", color="red", alpha=0.5)
    for bar, cer in zip(bars1, cers):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                 f"{cer:.2f}%", ha="center", va="bottom", fontsize=8)

    # Right: exact match
    p2_models_ex = [
        ("TrOCR-small", phase2_data["models"]["trocr_small"]["post_p2_freeform_exact_pct"], "#2ca02c"),
        ("PARSeq",      phase2_data["models"]["parseq"]["post_p2_freeform_exact_pct"],       "#ff7f0e"),
        ("CNN+RNN",     phase2_data["models"]["cnn_rnn"]["post_p2_freeform_exact_pct"],      "#d62728"),
    ]
    for name, vals in EXTENDED_LM_RESULTS.items():
        if vals["freeform_exact"] is not None:
            p2_models_ex.append((name, vals["freeform_exact"], "#7f7f7f"))

    exacts = [m[1] for m in p2_models_ex]
    bars2 = ax2.bar(range(len(labels)), exacts, color=colors, edgecolor="black")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax2.set_ylabel("Exact match rate (%)", fontsize=10)
    ax2.set_title("(b) Exact match — higher is better", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, ex in zip(bars2, exacts):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1,
                 f"{ex:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Phase 2 comparison on 35-sample synthetic-freeform test set",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES / "phase2_dual_metric.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    chart_phase1_comparison()
    chart_phase2_comparison()
    chart_phase1_exact_match()
    chart_phase2_exact_match()
    chart_phase1_dual()
    chart_phase2_dual()
