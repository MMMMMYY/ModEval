#!/usr/bin/env python3
"""
Figure 6 — Gender per-term signed skew.

4 panels (one per occupation), each showing 25 modifier terms grouped by
5 categories. For each term, 3 coloured markers = FLUX / SD 3.5 / MJ V7.
Filled marker = χ² p<0.05 vs. that model's baseline. Horizontal dashed
coloured lines = per-model baseline G value.
"""

import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
FAIRFACE = ROOT / "fairface_results.json"
STATS = ROOT / "bias_stats.csv"
OUT_PDF = ROOT / "figure6_gender.pdf"
OUT_PNG = ROOT / "figure6_gender.png"

# ── Term ordering (matches section5_rq2.md) ─────────────────────────────────

CATEGORIES = {
    "Artist":     ["artgerm", "greg rutkowski", "wlop",
                   "alphonse mucha", "rossdraws"],
    "Medium":     ["digital painting", "illustration", "digital art",
                   "matte painting", "oil painting"],
    "Movement":   ["fantasy", "cyberpunk", "anime", "sci-fi", "futuristic"],
    "Trending":   ["trending on artstation", "cgsociety", "deviantart",
                   "pixiv", "hearthstone"],
    "Atmosphere": ["elegant", "cinematic", "dramatic lighting",
                   "epic", "volumetric lighting"],
}
CAT_ORDER = list(CATEGORIES.keys())

MODELS = ["flux", "sd35", "midjourney"]
MODEL_LABEL = {"flux": "FLUX.1", "sd35": "SD 3.5", "midjourney": "MJ V7"}
MODEL_COLOR = {"flux": "#D62728", "sd35": "#1F77B4", "midjourney": "#2CA02C"}
MODEL_MARKER = {"flux": "o", "sd35": "s", "midjourney": "D"}

OCCUPATIONS = ["CEO", "engineer", "nurse", "teacher"]
OCC_LABEL = {"CEO": "CEO", "engineer": "Engineer",
             "nurse": "Nurse", "teacher": "Teacher"}


def slug(term: str) -> str:
    return term.lower().replace(" ", "_").replace("-", "_")


# ── Load data ───────────────────────────────────────────────────────────────

with open(FAIRFACE) as f:
    fairface = json.load(f)

# Load Cramér's V & significance from bias_stats.csv
sig_lookup = {}  # (model, occ, category, term) -> (p, V, sig_bool)
with open(STATS) as f:
    for row in csv.DictReader(f):
        key = (row["Model"], row["Occupation"],
               row["Category"], row["Term"])
        sig_lookup[key] = (
            float(row["chi2_p_gender"]),
            float(row["cramers_v_gender"]),
            bool(row["sig_gender"].strip()),
        )


def signed_g(entries):
    n_m = n_f = 0
    for e in entries:
        if not e.get("face_detected", True):
            continue
        g = e.get("gender")
        if g in ("Man", "Male"):
            n_m += 1
        elif g in ("Woman", "Female"):
            n_f += 1
    n = n_m + n_f
    return (n_m - n_f) / n if n else np.nan


def get_entries(model, occ, category, term):
    """Handle lowercase MJ category keys and term slug matching."""
    m_data = fairface[model][occ]
    # category key varies: FLUX/SD = title-case; MJ = lowercase
    for ck in m_data:
        if ck.lower() == category.lower():
            c_data = m_data[ck]
            break
    else:
        return []
    ts = slug(term)
    return c_data.get(ts, [])


def get_baseline_g(model, occ):
    m_data = fairface[model][occ]
    bl = {}
    for ck in m_data:
        if ck.lower() == "baseline":
            bl = m_data[ck]
            break
    entries = []
    for _, es in bl.items():
        entries.extend(es)
    return signed_g(entries)


# ── Build figure ────────────────────────────────────────────────────────────

# Flat term list + x positions.
# Each category gets 5 term slots + 1 summary slot + 1 gap.
X_POS = []
TERM_LIST = []            # (category, term, x)
CAT_SPANS = []            # (category, x_start, x_end_terms, x_summary)
x = 0
for cat in CAT_ORDER:
    start = x
    for t in CATEGORIES[cat]:
        TERM_LIST.append((cat, t, x))
        X_POS.append(x)
        x += 1
    end_terms = x - 1
    x_summary = x + 0.1       # summary marker sits just after last term
    CAT_SPANS.append((cat, start, end_terms, x_summary))
    x += 2                    # gap (reserve 2 slots: summary + separator)

# 1×4 grid of panels
fig, axes = plt.subplots(1, 4, figsize=(22, 5.6),
                         sharey=True, sharex=True)

JITTER = 0.18  # horizontal offset between 3 model markers

# Alternating pale background for category separation
CAT_BG = ["#FFFFFF", "#F4F4F4"]

for panel_i, occ in enumerate(OCCUPATIONS):
    ax = axes[panel_i]

    # category background bands (alternating)
    for ci, (cat, s, e, _) in enumerate(CAT_SPANS):
        ax.axvspan(s - 0.5, e + 0.5,
                   color=CAT_BG[ci % 2], alpha=0.6, zorder=0)

    # parity band
    ax.axhspan(-0.05, 0.05, color="#CCCCCC", alpha=0.55, zorder=1)
    ax.axhline(0, color="#777777", linewidth=0.7, zorder=1.5)

    # per-model baseline lines (slightly offset if they coincide, so
    # overlapping baselines don't collapse into a single colour stripe)
    baseline_vals = [(m, get_baseline_g(m, occ)) for m in MODELS]
    # Detect clusters of ≤0.02 apart, apply ±0.012 vertical offset
    sorted_bv = sorted(enumerate(baseline_vals),
                        key=lambda x: x[1][1])
    shifts = [0.0] * len(MODELS)
    for i in range(1, len(sorted_bv)):
        idx_prev, (_, v_prev) = sorted_bv[i-1]
        idx_curr, (_, v_curr) = sorted_bv[i]
        if abs(v_curr - v_prev) < 0.02:
            shifts[idx_curr] = shifts[idx_prev] + 0.025
    for mi, (m, g_bl) in enumerate(baseline_vals):
        ax.axhline(g_bl + shifts[mi], color=MODEL_COLOR[m],
                   linewidth=1.4, linestyle="--", alpha=0.75,
                   zorder=1.5)

    # plot each term × model (small per-term markers)
    for cat, term, xc in TERM_LIST:
        for mi, m in enumerate(MODELS):
            offset = (mi - 1) * JITTER
            entries = get_entries(m, occ, cat, term)
            g_val = signed_g(entries)
            if np.isnan(g_val):
                continue
            key = (m, occ, cat, term)
            _, cv, sig = sig_lookup.get(key, (1.0, 0.0, False))
            face = MODEL_COLOR[m] if sig else "white"
            edge = MODEL_COLOR[m]
            size = 30 + cv * 120
            ax.scatter(xc + offset, g_val,
                       marker=MODEL_MARKER[m],
                       s=size,
                       facecolors=face,
                       edgecolors=edge,
                       linewidths=1.1,
                       alpha=0.85,
                       zorder=3)

    # vertical separator between categories
    for cat, s, e, _xs in CAT_SPANS:
        ax.axvline(e + 0.25, color="#999999",
                   linewidth=0.7, linestyle="-", alpha=0.6, zorder=1)

    # category label strip at the top of each panel (inside axes)
    for cat, s, e, _xs in CAT_SPANS:
        mid = (s + e) / 2
        ax.text(mid, 1.05, cat, ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#333333",
                transform=ax.get_xaxis_transform())

    # occupation title (above category strip, so pad title upward)
    bl_str = "  ".join(
        f"{MODEL_LABEL[m]}:{get_baseline_g(m, occ):+.2f}" for m in MODELS)
    ax.set_title(f"{OCC_LABEL[occ]}   [baseline  {bl_str}]",
                 fontsize=10.5, pad=28, fontweight="bold")

    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_xlim(-0.8, TERM_LIST[-1][2] + 0.8)

    xs = [xc for _, _, xc in TERM_LIST]
    labels = [t for _, t, _ in TERM_LIST]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7.5)

    if panel_i == 0:
        ax.set_ylabel("signed gender skew  (+ male / − female)",
                      fontsize=10)

# Legend — one merged block: model colour + marker + baseline + fill semantics
legend_elems = []
# Model colour + marker rows (filled version = significance demo)
for m in MODELS:
    legend_elems.append(plt.Line2D([0], [0],
        marker=MODEL_MARKER[m], color="w",
        markerfacecolor=MODEL_COLOR[m],
        markeredgecolor=MODEL_COLOR[m],
        markersize=10, linestyle="",
        label=f"{MODEL_LABEL[m]}"))
# Fill semantics
legend_elems.append(plt.Line2D([0], [0],
    marker="o", color="w", markerfacecolor="#555555",
    markeredgecolor="#555555", markersize=10, linestyle="",
    label="filled: significantly differs from baseline (χ², p < 0.05)"))
legend_elems.append(plt.Line2D([0], [0],
    marker="o", color="w", markerfacecolor="white",
    markeredgecolor="#555555", markersize=10, linestyle="",
    label="hollow: not significant"))
# Baseline line
legend_elems.append(plt.Line2D([0], [0], color="#555555",
    linestyle="--", linewidth=1.4,
    label="per-model baseline G (no modifier)"))
# Size cue
legend_elems.append(plt.Line2D([0], [0],
    marker="o", color="w", markerfacecolor="#555555",
    markeredgecolor="#555555", markersize=12, linestyle="",
    label="marker size ∝ Cramér's V"))

fig.legend(handles=legend_elems, loc="upper center",
           ncol=7, frameon=True, fontsize=9.5,
           bbox_to_anchor=(0.5, 1.02),
           handletextpad=0.5, columnspacing=1.2)

plt.tight_layout(rect=[0, 0.02, 1, 0.88])
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Wrote {OUT_PDF}")
print(f"Wrote {OUT_PNG}")
