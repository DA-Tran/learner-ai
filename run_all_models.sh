#!/bin/bash
# ML test suite runner - activates venv, runs all models/datasets

cd "$(dirname "$0")"
source .venv/Scripts/activate || echo "No .venv, using global Python"

# Set sklearn env
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Datasets
DATASETS="iris heart breast wine"

echo "ML Suite: $DATASETS on rnn lstm gan lgbm xgb + ALL"

python tests/run_all_tests.py

