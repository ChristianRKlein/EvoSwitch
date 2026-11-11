#!/usr/bin/env bash
# Run all analyses of the manuscript and create all figures.
# Results are written to results/, figures to figures/.
set -euo pipefail
cd "$(dirname "$0")/analyses"
for script in 01_timing_example 02_timing_scan 03_drug_effect_scan 04_tumors_A_B \
              05_response_matched_map 06_resistance_limits 07_max_compatible_fraction \
              08_nadir_criteria 09_extreme_scenarios 10_convergence; do
    python "${script}.py"
done
python make_figures.py
echo "All analyses completed."
