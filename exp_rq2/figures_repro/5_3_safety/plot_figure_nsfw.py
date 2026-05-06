#!/usr/bin/env python3
"""
Figure (5.3) — NSFW grouped bar chart, two panels.

Left  panel: Sexually explicit NSFW rate (%, NudeNet)
Right panel: Gore / violent NSFW rate (%, Q16)

Grouped bars by modifier category (baseline + 5) with three model
bars per group (FLUX.1 [dev] / SD 3.5 Large / Midjourney V7).
Per-model dashed line = baseline NSFW rate. A `*` above a bar
indicates Fisher exact p<0.05 vs. that model's baseline.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
EXPL = ROOT / "table9_explicit.csv"
GORE = ROOT / "table10_gore.csv"
STATS = ROOT / "safety_stats.csv"
OUT_PDF = ROOT / "figure9_nsfw.pdf"
OUT_PNG = ROOT / "figure9_nsfw.png"

CATS = ["Artist", "Medium", "Movement", "Trending", "Atmosphere"]
CAT_LABEL = {"Artist": "Artist", "Medium": "Medium",
             "Movement": "Movement", "Trending": "Trending",
             "Atmosphere": "Atmosphere"}
MODELS = ["flux", "sd35", "midjourney"]
MODEL_LABEL = {"flux": "FLUX.1 [dev]",
               "sd35": "SD 3.5 Large",
               "midjourney": "Midjourney V7"}
MODEL_COLOR = {"flux": "#D62728",
               "sd35": "#1F77B4",
               "midjourney": "#2CA02C"}


# ── Load per-category rates ────────────────────────────────────────────────
def load_rates(path):
    d = {}
    for row in csv.DictReader(open(path)):
        d[row["Category"]] = {m: float(row[m]) for m in MODELS}
    return d


rates_expl = load_rates(EXPL)
rates_gore = load_rates(GORE)

# ── Load per-cell significance ─────────────────────────────────────────────
sig_expl, sig_gore = {}, {}
for r in csv.DictReader(open(STATS)):
    d = sig_expl if r["ContentType"] == "explicit" else sig_gore
    d[(r["Model"], r["Category"])] = r["sig"] if r["sig"] else ""


def panel(ax, rates, sig, title):
    bar_w = 0.26
    x_cat = np.arange(len(CATS))

    # baseline reference line per model (nudge overlapping lines apart)
    bases = [(m, rates["baseline"][m]) for m in MODELS]
    # group near-identical baselines and vertically separate them a bit
    bases_sorted = sorted(bases, key=lambda x: x[1])
    offsets = {}
    last_v = None
    stack = 0
    for m, v in bases_sorted:
        if last_v is not None and abs(v - last_v) < 0.8:
            stack += 1
        else:
            stack = 0
        offsets[m] = stack * 1.1  # small vertical offset in % units
        last_v = v
    for m in MODELS:
        base = rates["baseline"][m] + offsets[m]
        ax.axhline(base, color=MODEL_COLOR[m], linestyle="--",
                   linewidth=1.2, alpha=0.75, zorder=1)

    # grouped bars (modifier categories only)
    for mi, m in enumerate(MODELS):
        xs = x_cat + (mi - 1) * bar_w
        heights = [rates[c][m] for c in CATS]
        ax.bar(xs, heights, bar_w, color=MODEL_COLOR[m],
               edgecolor="white", linewidth=0.6,
               label=MODEL_LABEL[m], zorder=3)
        for xc, c, h in zip(xs, CATS, heights):
            if sig.get((m, c)):
                ax.text(xc, h + 1.2, "*",
                        ha="center", va="bottom",
                        fontsize=14, fontweight="bold",
                        color="black", zorder=4)

    ax.set_xticks(x_cat)
    ax.set_xticklabels([CAT_LABEL[c] for c in CATS],
                       rotation=20, ha="right", fontsize=9.5)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", alpha=0.18, zorder=0)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)


fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), sharey=True)
plt.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.20,
                    wspace=0.05)

panel(axes[0], rates_expl, sig_expl, "Sexually explicit (NudeNet)")
panel(axes[1], rates_gore, sig_gore, "Gore / violent (Q16)")

axes[0].set_ylabel("NSFW rate (%)", fontsize=10.5)

# legend inside left panel (upper-left empty area — explicit bars
# cluster on the right side of panel A)
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
axes[0].legend(handles=handles, loc="upper left",
               ncol=1, frameon=True, fontsize=8.5,
               handletextpad=0.55, labelspacing=0.35,
               framealpha=0.92, borderpad=0.5)

plt.savefig(OUT_PDF)
plt.savefig(OUT_PNG, dpi=160)
print(f"Wrote {OUT_PDF}")
print(f"Wrote {OUT_PNG}")
