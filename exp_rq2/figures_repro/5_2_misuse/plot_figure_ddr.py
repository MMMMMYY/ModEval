#!/usr/bin/env python3
"""
Figure (5.2) — DDR grouped bar chart.

Two panels (Person / City street). Each panel groups DDR bars by
modifier category (6: baseline + 5) with three models per group.
Per-model dashed baseline line overlays the panel. † annotates
bars whose Fisher p<0.05 vs. that model's baseline.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
STATS = ROOT / "table8_ddr_stats.csv"
DDR = ROOT / "table8_ddr.csv"
OUT_PDF = ROOT / "figure_ddr.pdf"
OUT_PNG = ROOT / "figure_ddr.png"

CATS = ["baseline", "Artist", "Medium", "Movement", "Trending", "Atmosphere"]
CAT_LABEL = {"baseline": "Baseline", "Artist": "Artist",
             "Medium": "Medium", "Movement": "Movement",
             "Trending": "Trending", "Atmosphere": "Atmosphere"}
MODELS = ["flux", "sd35", "midjourney"]
MODEL_LABEL = {"flux": "FLUX.1 [dev]",
               "sd35": "SD 3.5 Large",
               "midjourney": "Midjourney V7"}
MODEL_COLOR = {"flux": "#D62728",        # red  (matches Fig 6)
               "sd35": "#1F77B4",        # blue
               "midjourney": "#2CA02C"}  # green
SUBJECTS = [("person", "Person"), ("city_street", "City street")]


# ── Load data ──────────────────────────────────────────────────────────────
ddr = {}
for row in csv.DictReader(open(DDR)):
    ddr[(row["Model"], row["Subject"], row["Category"])] = float(row["DDR"])

sig = {}  # (model, subject, category) -> "*" or ""
for row in csv.DictReader(open(STATS)):
    sig[(row["Model"], row["Subject"], row["Category"])] = \
        row["sig"] if row["sig"] else ""


# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.6),
                         sharey=True)
plt.subplots_adjust(left=0.06, right=0.985, top=0.92, bottom=0.20,
                    wspace=0.05)

bar_w = 0.26
x_cat = np.arange(len(CATS))

for ax_i, (subj_key, subj_label) in enumerate(SUBJECTS):
    ax = axes[ax_i]

    # baseline reference line per model (horizontal dashed)
    for m in MODELS:
        base = ddr[(m, subj_key, "baseline")]
        ax.axhline(base, color=MODEL_COLOR[m], linestyle="--",
                   linewidth=1.0, alpha=0.55, zorder=1)

    # faint divider between baseline column and modifier columns
    ax.axvline(0.5, color="#999999", linewidth=0.6,
               linestyle=":", alpha=0.7, zorder=0)

    # grouped bars
    for mi, m in enumerate(MODELS):
        xs = x_cat + (mi - 1) * bar_w
        heights = [ddr[(m, subj_key, c)] for c in CATS]
        bars = ax.bar(xs, heights, bar_w,
                      color=MODEL_COLOR[m], edgecolor="white",
                      linewidth=0.6, label=MODEL_LABEL[m],
                      zorder=3)
        # annotate † on significant modifier bars
        for xc, c, h in zip(xs, CATS, heights):
            if c == "baseline":
                continue
            if sig.get((m, subj_key, c)):
                ax.text(xc, h + 0.005, "*",
                        ha="center", va="bottom",
                        fontsize=14, fontweight="bold",
                        color="black", zorder=4)

    ax.set_xticks(x_cat)
    ax.set_xticklabels([CAT_LABEL[c] for c in CATS],
                       rotation=20, ha="right", fontsize=9.5)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", alpha=0.18, zorder=0)
    if ax_i == 0:
        ax.set_ylabel("Deepfake Detection Rate (DDR)", fontsize=10.5)
    ax.set_title(subj_label, fontsize=12, fontweight="bold", pad=6)

    # subtle shading over the baseline column
    ax.axvspan(-0.5, 0.5, color="#FFF4E6", alpha=0.45, zorder=0)


# ── Legend inside the Person panel (low-left, which is empty space) ───────
handles = [plt.Rectangle((0, 0), 1, 1,
                         facecolor=MODEL_COLOR[m], edgecolor="white",
                         label=MODEL_LABEL[m])
           for m in MODELS]
handles.append(plt.Line2D([0], [0], color="gray", linestyle="--",
                          linewidth=1.2,
                          label="per-model baseline"))
handles.append(plt.Line2D([0], [0], marker="$*$", color="black",
                          markersize=13, linestyle="",
                          label="sig. vs. baseline (Fisher p<0.05)"))
axes[0].legend(handles=handles, loc="upper right",
               ncol=1, frameon=True, fontsize=8.5,
               handletextpad=0.55, labelspacing=0.35,
               framealpha=0.92, borderpad=0.5)

plt.savefig(OUT_PDF)
plt.savefig(OUT_PNG, dpi=160)
print(f"Wrote {OUT_PDF}")
print(f"Wrote {OUT_PNG}")
