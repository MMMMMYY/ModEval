#!/usr/bin/env bash
# Section 5.3 — Safety (NSFW) experiment
# screen -S rq2_safety && bash run_5_3.sh

set -euo pipefail
LOG_DIR="outputs/5_3/logs"
mkdir -p "$LOG_DIR"
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

run_step() {
    local label="$1"; shift
    echo "================================================================"
    echo "[$(timestamp)]  START  $label"
    python section_5_3_safety.py "$@" 2>&1 | tee "$LOG_DIR/${label}.log"
    echo "[$(timestamp)]  DONE   $label"
    echo ""
}

echo "5.3 Safety pipeline started at $(timestamp)"

run_step "gen_flux"  --mode generate --model flux
run_step "gen_sd35"  --mode generate --model sd35
run_step "evaluate"  --mode evaluate
run_step "report"    --mode report

echo "================================================================"
echo "DONE at $(timestamp)"
echo "Sync back: outputs/5_3/safety_results.json table9_explicit.csv table10_gore.csv"
