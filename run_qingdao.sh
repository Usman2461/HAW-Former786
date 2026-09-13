#!/usr/bin/env bash
# Qingdao headline result: three seeds of the complete model.
# Reproduces results/qdb19/. Sequential -- the graph stage is memory-hungry.
set -eu
cd "$(dirname "$0")"
for s in 0 1 2; do
  echo "=== Qingdao seed $s ==="
  python3 scripts/train.py --config configs/qdb19.yaml --tag final --seed "$s"
done
python3 scripts/verify_results.py --dataset qdb19
