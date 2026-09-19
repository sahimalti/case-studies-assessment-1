#!/usr/bin/env bash
# Reproduce every Task 2 (Part 2) result, table and figure from the raw data.
# Run from the project root:   bash run_task2.sh
# Approximate run time on a single CPU core: 1.5-2.5 hours (the 100-seed hold-out
# experiment, nested CV and the BRFSS random forests dominate).
# Every stage caches its output in outputs/task2/, so an interrupted run resumes.
set -euo pipefail
cd "$(dirname "$0")"

python src/task2_cv_sampling.py            # repeated / nested CV, 100-seed hold-out, BRFSS CV
python src/task2_learning_curves.py        # learning-curve experiments
python src/task2_plot_learning_curves.py   # learning-curve figure + summary
python src/task2_fairness.py               # Fairlearn audit + mitigation + figure
python src/task2_make_tables.py            # LaTeX tables for the report
