#!/usr/bin/env bash
# ============================================================
# RQ1 full pipeline runner
# Usage: bash run_rq1.sh [step]
#   step: all | 4.1 | 4.2_classify | 4.2_validate | 4.2_freq | 4.3_sample | 4.3_generate | 4.3_eval
# ============================================================
set -euo pipefail

STEP=${1:-all}
GOLD_PATH=${GOLD_PATH:-"data/gold_annotations_150.json"}

echo "=============================="
echo "RQ1 pipeline — step: $STEP"
echo "=============================="

mkdir -p outputs

run_4_1() {
    echo ""
    echo "[4.1] Category harmonization …"
    python section_4_1_harmonization.py
    echo "[4.1] Done. Review outputs/4_1/figure3_heatmap.pdf and"
    echo "      edit CLUSTER_NAME_OVERRIDE in section_4_1_harmonization.py,"
    echo "      then re-run to assign unified category names."
}

run_4_2_classify() {
    echo ""
    echo "[4.2] LLM classification (80K prompts) …"
    python section_4_2_classification.py --mode classify
}

run_4_2_validate() {
    echo ""
    echo "[4.2] Validation against gold annotations …"
    python section_4_2_classification.py --mode validate --gold_path "$GOLD_PATH"
}

run_4_2_freq() {
    echo ""
    echo "[4.2] Frequency statistics …"
    python section_4_2_classification.py --mode frequency
}

run_4_3_sample() {
    echo ""
    echo "[4.3] Prompt sampling …"
    python section_4_3_steering.py --step sample
}

run_4_3_generate() {
    echo ""
    echo "[4.3] Image generation (FLUX + SD 3.5) …"
    python section_4_3_steering.py --step generate
}

run_4_3_eval() {
    echo ""
    echo "[4.3] Evaluation (similarity + MSCS + steering confirmation) …"
    python section_4_3_steering.py --step evaluate
}

case $STEP in
    all)
        run_4_1
        run_4_2_classify
        run_4_2_validate
        run_4_2_freq
        run_4_3_sample
        run_4_3_generate
        run_4_3_eval
        ;;
    4.1)            run_4_1 ;;
    4.2_classify)   run_4_2_classify ;;
    4.2_validate)   run_4_2_validate ;;
    4.2_freq)       run_4_2_freq ;;
    4.3_sample)     run_4_3_sample ;;
    4.3_generate)   run_4_3_generate ;;
    4.3_eval)       run_4_3_eval ;;
    *)
        echo "Unknown step: $STEP"
        echo "Valid: all | 4.1 | 4.2_classify | 4.2_validate | 4.2_freq | 4.3_sample | 4.3_generate | 4.3_eval"
        exit 1
        ;;
esac

echo ""
echo "=============================="
echo "Step '$STEP' complete."
echo "=============================="
