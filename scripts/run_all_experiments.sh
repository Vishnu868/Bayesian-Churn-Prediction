#!/usr/bin/env bash
# Reproduces every number in REPORT.md. Each step caches its per-unit results, so re-running resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/download_data.py
python scripts/make_figures.py                                                    # dataset (EDA) figures
python src/main.py --out results/main                                             # corrected pipeline
python src/main.py --legacy-keep-id --skip-full --out results/legacy_with_customerID   # original README setup
python scripts/run_cv.py --folds 5 --out results/cv
python scripts/run_prior_sensitivity.py --out results/prior_sensitivity
python scripts/run_data_size.py --out results/data_size
python scripts/run_synthetic.py --reps 10 --out results/synthetic_sanity_check    # SYNTHETIC sanity check
python -m pytest -q tests
