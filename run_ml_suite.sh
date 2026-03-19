#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Activating venv + env vars..."
source .venv/Scripts/activate 2>/dev/null || echo "Using global Python"

export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
export TF_CPP_MIN_LOG_LEVEL=3

echo "Running ML suite: rnn,lstm,gan,lgbm,xgb on iris,heart,breast,wine + ALL"
python tests/run_all_tests.py

echo "JSON results saved: test_*_results.json"

