#!/usr/bin/env python3
"""
Figure 10 — Cross-risk composition (100%-stacked horizontal bar chart).

Reads upstream CSVs from ../5_1_bias, ../5_2_misuse, ../5_3_safety and
computes per-(category, dimension) mean effect magnitudes live. No
values are hard-coded.

Pipeline:
  1. Per (category, dimension) scalar =
        Bias    : mean V̄  over (gender ∪ race) × 4 occ × 3 models
        Misuse  : mean |ΔDDR| over 2 subjects × 3 models
        Safety  : mean |Δrate| over 2 content types × 3 models
  2. Column-wise min-max (each column / column max)
  3. Row-normalize so each category's three shares sum to 1

Outputs figure10.pdf / figure10.png into the current folder.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent

BIAS_CSV   = ROOT / "5_1_bias"   / "bias_stats.csv"
DDR_CSV    = ROOT / "5_2_misuse" / "table8_ddr_stats.csv"
SAFETY_CSV = ROOT / "5_3_safety" / "safety_stats.csv"

OUT_PDF = HERE / "figure10.pdf"
OUT_PNG = HERE / "figure10.png"

CATS = ["Artist", "Medium", "Movement", "Trending", "Atmosphere"]
MODELS = ["flux", "sd35", "midjourney"]
DIMS = ["Bias", "Misuse", "Safety"]
DIM_COLOR = {"Bias": "#4C78A8",
             "Misuse": "#F58518",
             "Safety": "#E45756"}


# ── Step 1a: Bias — mean V̄ over gender + race × (4 occ × 3 models) ─────────
def compute_bias():
    """Mean Cramér's V̄ across 24 cells per category."""
    out = {c: [] for c in CATS}
    with open(BIAS_CSV) as f:
        for row in csv.DictReader(f):
            if row["Category"] not in out:
                continue
            for key in ("cramers_v_gender", "cramers_v_race"):
                v = row[key]
                if v == "" or v is None:
                    continue
                try:
                    out[row["Category"]].append(float(v))
                except ValueError:
                    pass
    return {c: float(np.mean(out[c])) for c in CATS}


# ── Step 1b: Misuse — mean |ΔDDR| over 6 cells per category ────────────────
def compute_misuse():
    out = {c: [] for c in CATS}
    with open(DDR_CSV) as f:
        for row in csv.DictReader(f):
            if row["Category"] not in out:
                continue
            out[row["Category"]].append(abs(float(row["ΔDDR"])))
    return {c: float(np.mean(out[c])) for c in CATS}


# ── Step 1c: Safety — mean |Δrate| over 6 cells per category ───────────────
def compute_safety():
    out = {c: [] for c in CATS}
    with open(SAFETY_CSV) as f:
        for row in csv.DictReader(f):
            if row["Category"] not in out:
                continue
            # Δ_% is in percentage points; divide by 100 to match proportion
            out[row["Category"]].append(abs(float(row["Δ_%"])) / 100.0)
    return {c: float(np.mean(out[c])) for c in CATS}


bias = compute_bias()
misuse = compute_misuse()
safety = compute_safety()

# raw matrix
mat = np.array([[bias[c], misuse[c], safety[c]] for c in CATS])

# ── Step 2: column min-max ─────────────────────────────────────────────────
col_max = mat.max(axis=0)
mat_rel = mat / col_max

# ── Step 3: row-normalize ──────────────────────────────────────────────────
row_sum = mat_rel.sum(axis=1, keepdims=True)
mat_prop = mat_rel / row_sum

# sort by max component (most concentrated first)
order = np.argsort(-mat_prop.max(axis=1))
cats_sorted = [CATS[i] for i in order]
props = mat_prop[order]

# ── Print verification table ───────────────────────────────────────────────
print(f"{'Category':<12} {'Bias':>7} {'Misuse':>7} {'Safety':>7}"
      f"    {'Bias%':>6} {'Misuse%':>8} {'Safety%':>8}")
print("-" * 70)
for i, c in enumerate(CATS):
    idx = cats_sorted.index(c)
    print(f"{c:<12} "
          f"{bias[c]:>7.3f} {misuse[c]:>7.3f} {safety[c]:>7.3f}    "
          f"{props[idx,0]*100:>5.1f}% {props[idx,1]*100:>7.1f}% "
          f"{props[idx,2]*100:>7.1f}%")
print("-" * 70)
print(f"Column max: Bias={col_max[0]:.3f}  Misuse={col_max[1]:.3f}  "
      f"Safety={col_max[2]:.3f}\n")

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.8, 2.8))
y = np.arange(len(cats_sorted))
left = np.zeros(len(cats_sorted))

for di, dim in enumerate(DIMS):
    widths = props[:, di]
    ax.barh(y, widths, left=left, height=0.62,
            color=DIM_COLOR[dim], edgecolor="white",
            linewidth=1.2, label=dim)
    for yi, (w, lf) in enumerate(zip(widths, left)):
        if w >= 0.12:
            ax.text(lf + w / 2, yi, f"{w*100:.0f}%",
                    ha="center", va="center",
                    fontsize=9.5, color="white", fontweight="bold")
    left += widths

ax.set_yticks(y)
ax.set_yticklabels(cats_sorted, fontsize=11)
ax.invert_yaxis()
ax.set_xlim(0, 1.0)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
ax.set_xlabel("Share of effect magnitude "
              "(column-normalized, then row-normalized)",
              fontsize=9.5)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=3)
ax.grid(axis="x", alpha=0.18, zorder=0)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13),
          ncol=3, frameon=False, fontsize=10, handlelength=1.4,
          handletextpad=0.6, columnspacing=1.6)

plt.tight_layout()
plt.savefig(OUT_PDF)
plt.savefig(OUT_PNG, dpi=180)
print(f"Wrote {OUT_PDF}")
print(f"Wrote {OUT_PNG}")
