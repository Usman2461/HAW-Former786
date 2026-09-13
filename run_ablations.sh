#!/usr/bin/env bash
# Component ablations for both datasets (Tables VII and VIII).
set -eu
cd "$(dirname "$0")"
python3 scripts/run_ablation.py --config configs/qdb19.yaml  --seed 0
python3 scripts/run_ablation.py --config configs/tdrive.yaml --seed 0
python3 scripts/ablation_table.py
