#!/usr/bin/env bash
# Set up the Python environment for the paper's code and build the SimBench
# instances T2, T4, T6, T8 (experiments/data/models/, ~130 MB each).
# Requires: uv, Python 3.11, a C++ compiler with OpenMP (g++).
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv -p 3.11 env          # matches .envrc (source env/bin/activate)
source env/bin/activate
uv pip install -r requirements-lock.txt

# Compile the paper's C++ extensions (qe_builder, qe_sparse)
uv pip install --no-build-isolation -e package

# Run the SimBench power flow and create the instances (~30 s)
(cd experiments && python create_models.py)
