#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
#  ModEval — one-shot reproducer for ALL paper figures and tables.
#
#  This script regenerates every figure / table in the paper from the
#  frozen derived data shipped in this repository (CSVs + JSONs under
#  exp_rq1/outputs/ and exp_rq2/outputs/). It does NOT re-run image
#  generation or classifier inference — those upstream stages require
#  GPU resources and external API access; see run_all_pipeline.sh for
#  the full pipeline.
#
#  Wall time:  ~30 seconds on a CPU laptop.
#  Disk req.:  none beyond the repo itself.
# ──────────────────────────────────────────────────────────────────
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "════════════════════════════════════════════════════════════"
echo "  ModEval — regenerating all figures and tables"
echo "════════════════════════════════════════════════════════════"

# ── RQ1 (steering identification) ────────────────────────────────
# Table 1 (dual-metric), Table 7 (frequency), Table 6 (taxonomy)
# are produced by section_4_3_steering.py and are also
# kept as frozen CSVs under exp_rq1/outputs/4_3/.
echo ""
echo "── RQ1 derived tables already frozen at exp_rq1/outputs/4_3/"
echo "    table1_frequency.csv, table2_image_metrics_*.csv, table3_mscs_*.csv"

# ── RQ2 (social-risk eval) figures + tables ──────────────────────
echo ""
echo "── Re-rendering RQ2 figures (Tables 2,3,4 + Figures 2,3,4) …"
( cd "${HERE}/exp_rq2/figures_repro" && bash run_all.sh )

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Done. Outputs under:"
echo "    exp_rq2/figures_repro/5_1_bias/       (Figs 2-3 bias, Table 2)"
echo "    exp_rq2/figures_repro/5_2_misuse/     (Fig — DDR,    Table 3)"
echo "    exp_rq2/figures_repro/5_3_safety/     (Fig — NSFW,   Table 4)"
echo "    exp_rq2/figures_repro/5_4_cross_risk/ (Fig 4 cross-risk)"
echo "════════════════════════════════════════════════════════════"
