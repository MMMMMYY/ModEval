#!/usr/bin/env bash
# ============================================================
# Section 4.3 — Image Generation + Evaluation
# Usage:  screen -S rq1_4_3
#         bash run_4_3.sh
# ============================================================

set -euo pipefail

SCRIPT="section_4_3_steering.py"
LOG_DIR="outputs/4_3/logs"
mkdir -p "$LOG_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

run_step() {
    local label="$1"; shift
    local logfile="$LOG_DIR/${label}.log"
    echo "================================================================"
    echo "[$(timestamp)]  START  $label"
    echo "================================================================"
    python "$SCRIPT" "$@" 2>&1 | tee "$logfile"
    local status=${PIPESTATUS[0]}
    if [ $status -ne 0 ]; then
        echo "================================================================"
        echo "[$(timestamp)]  FAILED  $label  (exit $status)"
        echo "================================================================"
        exit $status
    fi
    echo "================================================================"
    echo "[$(timestamp)]  DONE   $label"
    echo "================================================================"
    echo ""
}

echo "Section 4.3 pipeline started at $(timestamp)"
echo "Logs → $LOG_DIR"
echo ""

# ── Step 1: FLUX generation ───────────────────────────────────────────────────
run_step "generate_flux"   --mode generate --model flux

# ── Step 2: SDXL generation ───────────────────────────────────────────────────
run_step "generate_sdxl"   --mode generate --model sdxl

# ── Step 3: FLUX evaluation ───────────────────────────────────────────────────
run_step "evaluate_flux"   --mode evaluate --model flux

# ── Step 4: SDXL evaluation ───────────────────────────────────────────────────
run_step "evaluate_sdxl"   --mode evaluate --model sdxl

echo "================================================================"
echo "ALL STEPS COMPLETE at $(timestamp)"
echo "Download these two files to local machine:"
echo "  outputs/4_3/metrics_flux.json"
echo "  outputs/4_3/metrics_sdxl.json"
echo "================================================================"
