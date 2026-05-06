#!/usr/bin/env bash
# Section 5.2 — Misuse (deepfake detectability) experiment
# screen -S rq2_misuse && bash run_5_2.sh

set -euo pipefail
LOG_DIR="outputs/5_2/logs"
mkdir -p "$LOG_DIR"
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

run_step() {
    local label="$1"; shift
    echo "================================================================"
    echo "[$(timestamp)]  START  $label"
    python section_5_2_misuse.py "$@" 2>&1 | tee "$LOG_DIR/${label}.log"
    echo "[$(timestamp)]  DONE   $label"
    echo ""
}

echo "5.2 Misuse pipeline started at $(timestamp)"

# Standard generation (no watermark)
run_step "gen_flux"       --mode generate    --model flux
run_step "gen_sd35"       --mode generate    --model sd35

# Watermarked generation (Tree-Ring)
# Requires tree_ring package on PYTHONPATH — skip if not installed
run_step "gen_wm_flux"    --mode generate_wm --model flux  || echo "  [SKIP] Tree-Ring not installed for flux"
run_step "gen_wm_sd35"    --mode generate_wm --model sd35  || echo "  [SKIP] Tree-Ring not installed for sd35"

# Evaluation (all detectors)
run_step "evaluate"       --mode evaluate

# Report
run_step "report"         --mode report

echo "================================================================"
echo "DONE at $(timestamp)"
echo "Sync back: outputs/5_2/detector_results.json table8_ddr.csv"
