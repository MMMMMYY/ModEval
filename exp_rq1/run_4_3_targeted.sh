#!/usr/bin/env bash
# ============================================================
# Section 4.3 — Targeted sampling for Repeating and Magic
# Run AFTER the main sampled_25 files are already on server.
#
# Local pre-step (run on laptop before scp to server):
#   python section_4_3_steering.py --mode sample_targeted
#   scp outputs/4_3/targeted_repeating_prompts.jsonl server:...
#   scp outputs/4_3/targeted_magic_prompts.jsonl server:...
#
# Then on server:
#   screen -S rq1_targeted
#   bash run_4_3_targeted.sh
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

echo "Section 4.3 targeted pipeline started at $(timestamp)"
echo "Logs → $LOG_DIR"
echo ""

# ── Step 1: FLUX generation (Repeating) ──────────────────────────────────────
run_step "gen_targeted_repeating_flux"  --mode generate_targeted --cat Repeating --model flux

# ── Step 2: FLUX generation (Magic) ──────────────────────────────────────────
run_step "gen_targeted_magic_flux"      --mode generate_targeted --cat Magic     --model flux

# ── Step 3: SDXL generation (Repeating) ──────────────────────────────────────
run_step "gen_targeted_repeating_sdxl"  --mode generate_targeted --cat Repeating --model sdxl

# ── Step 4: SDXL generation (Magic) ──────────────────────────────────────────
run_step "gen_targeted_magic_sdxl"      --mode generate_targeted --cat Magic     --model sdxl

# ── Step 5: FLUX evaluation (Repeating) ──────────────────────────────────────
run_step "eval_targeted_repeating_flux" --mode evaluate_targeted --cat Repeating --model flux

# ── Step 6: FLUX evaluation (Magic) ──────────────────────────────────────────
run_step "eval_targeted_magic_flux"     --mode evaluate_targeted --cat Magic     --model flux

# ── Step 7: SDXL evaluation (Repeating) ──────────────────────────────────────
run_step "eval_targeted_repeating_sdxl" --mode evaluate_targeted --cat Repeating --model sdxl

# ── Step 8: SDXL evaluation (Magic) ──────────────────────────────────────────
run_step "eval_targeted_magic_sdxl"     --mode evaluate_targeted --cat Magic     --model sdxl

echo "================================================================"
echo "ALL TARGETED STEPS COMPLETE at $(timestamp)"
echo "Download these four files to local machine:"
echo "  outputs/4_3/metrics_targeted_repeating_flux.json"
echo "  outputs/4_3/metrics_targeted_repeating_sdxl.json"
echo "  outputs/4_3/metrics_targeted_magic_flux.json"
echo "  outputs/4_3/metrics_targeted_magic_sdxl.json"
echo "Then run locally:"
echo "  python section_4_3_steering.py --mode report_targeted"
echo "================================================================"
