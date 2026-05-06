#!/usr/bin/env bash
# Reproduce every RQ2 figure + Table 6 from frozen derived data.
# Usage: bash run_all.sh

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {
    local dir="$1"
    local script="$2"
    echo ""
    echo "────────────────────────────────────────────"
    echo "  ${dir}/${script}"
    echo "────────────────────────────────────────────"
    (cd "${HERE}/${dir}" && python "${script}")
}

run 5_1_bias        plot_figure6_gender.py
run 5_1_bias        plot_figure7_race.py
run 5_1_bias        build_table6.py
run 5_2_misuse      plot_figure_ddr.py
run 5_3_safety      plot_figure_nsfw.py
run 5_4_cross_risk  plot_figure10.py

echo ""
echo "All figures regenerated under figures_repro/*/."
