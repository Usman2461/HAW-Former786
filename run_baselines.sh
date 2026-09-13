#!/usr/bin/env bash
# All nine competitors, retrained on the same partitions under one budget.
set -eu
cd "$(dirname "$0")"
for cfg in configs/qdb19.yaml configs/tdrive.yaml; do
  python3 scripts/baselines.py      --config "$cfg"   # HA, ARIMA
  python3 scripts/deep_baselines.py --config "$cfg"   # LSTM
  python3 scripts/gnn_baselines.py  --config "$cfg"   # DCRNN, STGCN, ASTGCN, Graph WaveNet
  python3 scripts/adapgl_baseline.py --config "$cfg"  # AdapGL
done
python3 scripts/make_tables.py
