#!/usr/bin/env bash
# T-Drive headline result: three seeds of the complete model.
# Reproduces results/tdrive/. Set the data paths in configs/tdrive.yaml first.
set -eu
cd "$(dirname "$0")"
for s in 0 1 2; do
  echo "=== T-Drive seed $s ==="
  python3 scripts/train.py --config configs/tdrive.yaml --tag final --seed "$s"
done
python3 scripts/verify_results.py --dataset tdrive
