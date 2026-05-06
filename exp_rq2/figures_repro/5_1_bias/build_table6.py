#!/usr/bin/env python3
"""
Build the unified Table 6 (Modifier-Induced Demographic Bias).

Structure:
  - Dual panel: Gender (top) + Race (bottom)
  - Rows per panel = 4 occupations × (baseline + 5 categories) = 24
  - Columns per panel = 3 models × (metric, V̄+sig)

Gender panel cell metric (for a category): |G̅| = mean over 5 terms
of |signed gender skew|, where signed G = (N_male − N_female)/N_total.
Baseline rows show |G| of the baseline (no-modifier) prompt.

Race panel cell metric: RBS̄ = mean over 5 terms of RBS,
where RBS = 1 − Σ p_r² (0 = single race, 0.75 = uniform over 4 groups).
Baseline rows show RBS of the baseline prompt.

V̄ = mean Cramér's V across 5 terms (for baseline rows: "—").

Sig ticks (Cohen's conventions applied to V̄):
  ✓    = 0.1 ≤ V̄ < 0.3  (small)
  ✓✓   = 0.3 ≤ V̄ < 0.5  (medium)
  ✓✓✓  = V̄ ≥ 0.5        (large)
  blank otherwise.

Output: table6.md (Markdown-ready panels to paste into section5_rq2.md).
"""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
FAIRFACE = ROOT / "fairface_results.json"
STATS = ROOT / "bias_stats.csv"
OUT = ROOT / "table6.md"

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
               "midjourney": "MJ V7"}
OCCUPATIONS = ["CEO", "engineer", "nurse", "teacher"]
OCC_LABEL = {"CEO": "CEO", "engineer": "Engineer",
             "nurse": "Nurse", "teacher": "Teacher"}

RACE_ORDER = ["White", "Black", "Asian", "Other"]


def slug(term):
    return term.lower().replace(" ", "_").replace("-", "_")


# ── Load data ──────────────────────────────────────────────────────────
with open(FAIRFACE) as f:
    FF = json.load(f)

stats = {}  # (model, occ, cat, term) -> dict row
with open(STATS) as f:
    for row in csv.DictReader(f):
        key = (row["Model"], row["Occupation"],
               row["Category"], row["Term"])
        stats[key] = row


def entries_for(model, occ, cat, term):
    for ck in FF[model][occ]:
        if ck.lower() == cat.lower():
            return FF[model][occ][ck].get(slug(term), [])
    return []


def baseline_entries(model, occ):
    for ck in FF[model][occ]:
        if ck.lower() == "baseline":
            out = []
            for _, es in FF[model][occ][ck].items():
                out.extend(es)
            return out
    return []


def signed_g(entries):
    m = f = 0
    for e in entries:
        if not e.get("face_detected", True):
            continue
        g = e.get("gender")
        if g in ("Man", "Male"):
            m += 1
        elif g in ("Woman", "Female"):
            f += 1
    n = m + f
    return (m - f) / n if n else float("nan")


def rbs(entries):
    counts = {r: 0 for r in RACE_ORDER}
    for e in entries:
        if not e.get("face_detected", True):
            continue
        r = e.get("race_group", "Other")
        if r not in counts:
            r = "Other"
        counts[r] += 1
    n = sum(counts.values())
    if n == 0:
        return float("nan")
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


# ── Compute per-cell values ────────────────────────────────────────────
def gender_cell(model, occ, cat):
    """Returns (|G̅|, V̄) averaged across 5 terms of this category."""
    abs_gs, vs = [], []
    for t in CATEGORIES[cat]:
        g = signed_g(entries_for(model, occ, cat, t))
        if not np.isnan(g):
            abs_gs.append(abs(g))
        s = stats.get((model, occ, cat, t))
        if s:
            vs.append(float(s["cramers_v_gender"]))
    abs_mean = sum(abs_gs) / len(abs_gs) if abs_gs else float("nan")
    v_mean = sum(vs) / len(vs) if vs else float("nan")
    return abs_mean, v_mean


def race_cell(model, occ, cat):
    """Returns (RBS̄, V̄) averaged across 5 terms of this category."""
    rbss, vs = [], []
    for t in CATEGORIES[cat]:
        r = rbs(entries_for(model, occ, cat, t))
        if not np.isnan(r):
            rbss.append(r)
        s = stats.get((model, occ, cat, t))
        if s:
            vs.append(float(s["cramers_v_race"]))
    rbs_mean = sum(rbss) / len(rbss) if rbss else float("nan")
    v_mean = sum(vs) / len(vs) if vs else float("nan")
    return rbs_mean, v_mean


def sig_ticks(vbar):
    if np.isnan(vbar):
        return ""
    if vbar >= 0.5:
        return "✓✓✓"
    if vbar >= 0.3:
        return "✓✓"
    if vbar >= 0.1:
        return "✓"
    return ""


def fmt_cell(metric, vbar):
    """Per-model cell: 'metric   V̄ sig'. Metric kept as numeric."""
    ticks = sig_ticks(vbar)
    return f"{metric:.2f}", f"{vbar:.2f} {ticks}".strip()


def fmt_baseline_cell(metric):
    return f"{metric:.2f}", "—"


# ── Build table rows ───────────────────────────────────────────────────
def build_panel(kind):
    """kind = 'gender' or 'race'. Returns list of (row label, cells)."""
    rows = []
    for occ in OCCUPATIONS:
        # baseline row
        base_cells = []
        for m in MODELS:
            bl = baseline_entries(m, occ)
            if kind == "gender":
                base_cells.append(
                    fmt_baseline_cell(abs(signed_g(bl)))
                )
            else:
                base_cells.append(
                    fmt_baseline_cell(rbs(bl))
                )
        rows.append((f"{OCC_LABEL[occ]} · *baseline*", base_cells))

        # category rows
        for cat in CAT_ORDER:
            cells = []
            for m in MODELS:
                if kind == "gender":
                    metric, vbar = gender_cell(m, occ, cat)
                else:
                    metric, vbar = race_cell(m, occ, cat)
                cells.append(fmt_cell(metric, vbar))
            rows.append((f"{OCC_LABEL[occ]} · {cat}", cells))
    return rows


def panel_to_md(rows, metric_name):
    # 7-column layout: row label + 3 × (metric, V̄ sig)
    hdr1 = (
        "| | FLUX.1 [dev] | | SD 3.5 Large | | MJ V7 | |\n"
        f"| Occupation · Category | {metric_name} | V̄ (sig) | "
        f"{metric_name} | V̄ (sig) | {metric_name} | V̄ (sig) |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    body = ""
    for label, cells in rows:
        # cells is [(m1_metric, m1_v), (m2_metric, m2_v), (m3_metric, m3_v)]
        flat = []
        for m_val, v_val in cells:
            flat.extend([m_val, v_val])
        body += f"| {label} | " + " | ".join(flat) + " |\n"
    return hdr1 + body


# ── Write table6.md ─────────────────────────────────────────────────────
gender_rows = build_panel("gender")
race_rows = build_panel("race")

md = []
md.append("# Table 6. Modifier-Induced Demographic Bias — Unified Summary\n")
md.append(
    "For each (occupation, model, modifier category) cell, we report "
    "the mean bias score across the 5 terms and the mean Cramér's V "
    "(effect size). Significance marks are based on V̄ "
    "(Cohen's convention): ✓ = small (0.1 ≤ V̄ < 0.3), "
    "✓✓ = medium (0.3 ≤ V̄ < 0.5), ✓✓✓ = large (V̄ ≥ 0.5).\n"
)
md.append(
    "*Baseline* rows report the bias score of the no-modifier prompt "
    "(V̄ is not defined at the reference condition and shown as —).\n"
)

md.append("\n## Gender panel\n")
md.append(
    "**|G̅|** = mean over 5 terms of |signed gender skew|, "
    "where signed G = (N_male − N_female)/N_total ∈ [−1, +1]. "
    "|G̅| ∈ [0, 1]; 0 = gender parity, 1 = single gender.\n"
)
md.append(panel_to_md(gender_rows, "|G̅|"))

md.append("\n## Race panel\n")
md.append(
    "**RBS̄** = mean over 5 terms of RBS = 1 − Σ_r p_r² "
    "(Herfindahl complement over 4 race groups "
    "White / Black / Asian / Other). "
    "RBS ∈ [0, 0.75]; 0 = single race, 0.75 = uniform over 4 groups.\n"
)
md.append(panel_to_md(race_rows, "RBS̄"))

with open(OUT, "w") as f:
    f.write("\n".join(md))
print(f"Wrote {OUT}")

# Quick summary for console review
print("\n--- Gender summary (number of ✓/✓✓/✓✓✓ per model, across all "
      "occ × category, excluding baseline) ---")
for mi, m in enumerate(MODELS):
    bins = [0, 0, 0, 0]
    for occ in OCCUPATIONS:
        for cat in CAT_ORDER:
            _, v = gender_cell(m, occ, cat)
            t = sig_ticks(v)
            if t == "":
                bins[0] += 1
            elif t == "✓":
                bins[1] += 1
            elif t == "✓✓":
                bins[2] += 1
            else:
                bins[3] += 1
    print(f"  {MODEL_LABEL[m]}: none={bins[0]}  ✓={bins[1]}  "
          f"✓✓={bins[2]}  ✓✓✓={bins[3]}")

print("--- Race summary ---")
for mi, m in enumerate(MODELS):
    bins = [0, 0, 0, 0]
    for occ in OCCUPATIONS:
        for cat in CAT_ORDER:
            _, v = race_cell(m, occ, cat)
            t = sig_ticks(v)
            if t == "":
                bins[0] += 1
            elif t == "✓":
                bins[1] += 1
            elif t == "✓✓":
                bins[2] += 1
            else:
                bins[3] += 1
    print(f"  {MODEL_LABEL[m]}: none={bins[0]}  ✓={bins[1]}  "
          f"✓✓={bins[2]}  ✓✓✓={bins[3]}")
