#!/usr/bin/env python3
"""
Figure 7 — Race per-term composition.

4 rows (occupations) × 3 cols (models) grid of stacked-bar subplots.
Each subplot: 1 baseline bar + 25 term bars grouped by 5 categories.
Each bar's 4 stacked segments = race proportions (White / Black /
Asian / Other). A black * above a term bar marks χ² p<0.05 vs. that
model's baseline.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

ROOT = Path(__file__).parent
FAIRFACE = ROOT / "fairface_results.json"
OUT_PDF = ROOT / "figure7_race.pdf"
OUT_PNG = ROOT / "figure7_race.png"

# ── Term ordering (matches Figure 6) ────────────────────────────────────────
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
MODEL_LABEL = {"flux": "FLUX.1 [dev]",
               "sd35": "SD 3.5 Large",
               "midjourney": "Midjourney V7"}
OCCUPATIONS = ["CEO", "engineer", "nurse", "teacher"]
OCC_LABEL = {"CEO": "CEO", "engineer": "Engineer",
             "nurse": "Nurse", "teacher": "Teacher"}

RACE_ORDER = ["White", "Black", "Asian", "Other"]
RACE_COLOR = {
    "White": "#FAD9B8",   # pale peach
    "Black": "#5D4037",   # warm brown
    "Asian": "#F5C83F",   # gold-yellow
    "Other": "#6FA8DC",   # muted blue
}


def slug(term):
    return term.lower().replace(" ", "_").replace("-", "_")


# ── Load data ──────────────────────────────────────────────────────────────
with open(FAIRFACE) as f:
    fairface = json.load(f)


def race_counts(entries):
    counts = {r: 0 for r in RACE_ORDER}
    for e in entries:
        if not e.get("face_detected", True):
            continue
        r = e.get("race_group", "Other")
        if r not in counts:
            r = "Other"
        counts[r] += 1
    return counts


def race_proportions(entries):
    c = race_counts(entries)
    total = sum(c.values())
    if total == 0:
        return {r: 0.0 for r in RACE_ORDER}
    return {r: c[r] / total for r in RACE_ORDER}


def race_chi2(entries_base, entries_term):
    """χ² on 2 × K contingency (baseline vs term, K = race groups
    with any non-zero count across both rows). Returns (p, V)."""
    cb = race_counts(entries_base)
    ct = race_counts(entries_term)
    row_b = np.array([cb[r] for r in RACE_ORDER], dtype=float)
    row_t = np.array([ct[r] for r in RACE_ORDER], dtype=float)
    tab = np.vstack([row_b, row_t])
    # drop race cols with zero in both rows (chi2 undefined)
    mask = tab.sum(axis=0) > 0
    tab = tab[:, mask]
    if tab.shape[1] < 2 or row_b.sum() == 0 or row_t.sum() == 0:
        return (1.0, 0.0)
    try:
        chi2, p, _, _ = chi2_contingency(tab)
    except ValueError:
        return (1.0, 0.0)
    n = tab.sum()
    k = min(tab.shape)
    V = float(np.sqrt(chi2 / (n * (k - 1)))) if n > 0 and k > 1 else 0.0
    return (float(p), V)


def get_entries(model, occ, category, term):
    m_data = fairface[model][occ]
    # handle case-insensitive category key
    for ck in m_data:
        if ck.lower() == category.lower():
            c_data = m_data[ck]
            break
    else:
        return []
    return c_data.get(slug(term), [])


def get_baseline_entries(model, occ):
    m_data = fairface[model][occ]
    for ck in m_data:
        if ck.lower() == "baseline":
            bl = m_data[ck]
            break
    else:
        return []
    entries = []
    for _, es in bl.items():
        entries.extend(es)
    return entries


# ── Build x-positions ──────────────────────────────────────────────────────
# Layout: [ baseline ] [gap] [cat1 × 5] [gap] [cat2 × 5] ...
BARS = []  # (x, short_label, full_term, is_baseline, cat)
x = 0.0
BARS.append((x, "BL", "baseline", True, None))
x += 1.8  # gap after baseline

CAT_SPANS = []  # (cat, x_start, x_end)
for cat in CAT_ORDER:
    start = x
    for t in CATEGORIES[cat]:
        BARS.append((x, t, t, False, cat))
        x += 1
    end = x - 1
    CAT_SPANS.append((cat, start, end))
    x += 0.8  # small gap between categories

X_MAX = BARS[-1][0] + 0.8


# ── Plot ───────────────────────────────────────────────────────────────────
nrows, ncols = len(OCCUPATIONS), len(MODELS)
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(23, 9.5),
                         sharey=True, sharex=True)
plt.subplots_adjust(left=0.055, right=0.995, top=0.85, bottom=0.12,
                    hspace=0.22, wspace=0.06)

for r, occ in enumerate(OCCUPATIONS):
    for c, model in enumerate(MODELS):
        ax = axes[r, c]
        bl_entries = get_baseline_entries(model, occ)

        # alternating pale category background
        for ci, (cat, s, e) in enumerate(CAT_SPANS):
            if ci % 2 == 1:
                ax.axvspan(s - 0.5, e + 0.5,
                           color="#F4F4F4", alpha=0.7, zorder=0)

        # subtle separator before baseline area
        ax.axvspan(-0.7, 0.7, color="#FFF4E6", alpha=0.5, zorder=0)

        # draw bars
        for xc, lbl, term, is_bl, cat in BARS:
            if is_bl:
                props = race_proportions(bl_entries)
            else:
                props = race_proportions(
                    get_entries(model, occ, cat, term))

            bottom = 0
            for race in RACE_ORDER:
                h = props[race]
                if h > 0:
                    ax.bar(xc, h, bottom=bottom, width=0.94,
                           color=RACE_COLOR[race],
                           edgecolor="white", linewidth=0.3,
                           zorder=2)
                bottom += h

            # baseline marked with thicker black outline
            if is_bl:
                ax.bar(xc, 1.0, bottom=0, width=0.94,
                       color="none", edgecolor="black",
                       linewidth=1.6, zorder=3)

            # significance asterisk — compute χ² live from fairface data
            if not is_bl:
                term_entries = get_entries(model, occ, cat, term)
                p, _ = race_chi2(bl_entries, term_entries)
                if p < 0.05:
                    ax.text(xc, 1.015, "*", ha="center", va="bottom",
                            fontsize=14, fontweight="bold",
                            color="black")

        # category strip labels only on top row (shared x, same grouping)
        if r == 0:
            for cat, s, e in CAT_SPANS:
                mid = (s + e) / 2
                ax.text(mid, 1.12, cat, ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#333333",
                        transform=ax.get_xaxis_transform())
            ax.text(0, 1.12, "BL", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333",
                    transform=ax.get_xaxis_transform())

        # y-label (left col)
        if c == 0:
            ax.set_ylabel("race proportion", fontsize=10)

        ax.set_ylim(0, 1.20)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(-0.9, X_MAX)
        ax.grid(axis="y", alpha=0.15, zorder=0)

        # x-tick labels only on bottom row
        if r == nrows - 1:
            xs = [b[0] for b in BARS]
            labels = [b[1] for b in BARS]
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=55, ha="right",
                               fontsize=7.2)
        else:
            ax.set_xticks([])


# ── Legend ─────────────────────────────────────────────────────────────────
legend_elems = []
for race in RACE_ORDER:
    legend_elems.append(plt.Rectangle(
        (0, 0), 1, 1,
        facecolor=RACE_COLOR[race], edgecolor="white",
        label=race))
legend_elems.append(plt.Rectangle(
    (0, 0), 1, 1,
    facecolor="none", edgecolor="black", linewidth=1.6,
    label="baseline (no modifier)"))
legend_elems.append(plt.Line2D(
    [0], [0], marker="$*$", color="black",
    markersize=14, linestyle="",
    label="significantly differs from baseline (χ², p < 0.05)"))

fig.legend(handles=legend_elems, loc="upper center",
           ncol=6, frameon=True, fontsize=11,
           bbox_to_anchor=(0.5, 0.99),
           handletextpad=0.6, columnspacing=1.8)

# Column headers (models) — placed in figure coords above top row
for c, model in enumerate(MODELS):
    x_mid = (axes[0, c].get_position().x0 +
             axes[0, c].get_position().x1) / 2
    fig.text(x_mid, 0.92, MODEL_LABEL[model],
             ha="center", va="bottom",
             fontsize=14, fontweight="bold", color="#111111")

# Row labels (occupations) on the left of each row, outside axes
for r, occ in enumerate(OCCUPATIONS):
    bbox_top = axes[r, 0].get_position().y1
    bbox_bot = axes[r, 0].get_position().y0
    y_mid = (bbox_top + bbox_bot) / 2
    fig.text(0.012, y_mid, OCC_LABEL[occ],
             ha="left", va="center",
             fontsize=14, fontweight="bold",
             rotation=90, color="#222222")

plt.savefig(OUT_PDF)
plt.savefig(OUT_PNG, dpi=150)
print(f"Wrote {OUT_PDF}")
print(f"Wrote {OUT_PNG}")
