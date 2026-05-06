#!/usr/bin/env bash
# Section 5.1 — Bias experiment
# Run on A6000 server after scp-ing this dir
# screen -S rq2_bias && bash run_5_1.sh

set -euo pipefail
LOG_DIR="outputs/5_1/logs"
mkdir -p "$LOG_DIR"
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

run_step() {
    local label="$1"; shift
    echo "================================================================"
    echo "[$(timestamp)]  START  $label"
    python section_5_1_bias.py "$@" 2>&1 | tee "$LOG_DIR/${label}.log"
    echo "[$(timestamp)]  DONE   $label"
    echo ""
}

echo "5.1 Bias pipeline started at $(timestamp)"

run_step "gen_flux"  --mode generate --model flux
run_step "gen_sd35"  --mode generate --model sd35
run_step "evaluate"  --mode evaluate
run_step "report"    --mode report

echo "================================================================"
echo "DONE at $(timestamp)"
echo "Sync back: outputs/5_1/fairface_results.json table6_gender.csv table7_race.csv"
