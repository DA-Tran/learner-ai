#!/bin/bash
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
source .venv/Scripts/activate
python Main.py

