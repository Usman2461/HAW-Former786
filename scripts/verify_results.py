#!/usr/bin/env python3
"""Recompute the headline metrics from the stored prediction files.

Every number reported for HAWFormer comes from a stored prediction tensor
re-scored under one metric implementation.  This script re-runs that scoring so
anyone can check the table without retraining, and so a fresh run can be
compared against the released one.

    python3 scripts/verify_results.py                  # both datasets
    python3 scripts/verify_results.py --dataset qdb19
    python3 scripts/verify_results.py --runs runs      # score your own runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hawformer.metrics import adapstwt_metrics  # noqa: E402


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    keys = list(z.keys())
    preds = z["preds"] if "preds" in z else z[keys[0]]
    target = z["targets"] if "targets" in z else z[keys[1]]
    return preds, target


def score(root: Path, dataset: str, pattern: str) -> None:
    seeds = sorted(root.glob(pattern))
    if not seeds:
        print(f"  {dataset}: no prediction files under {root}")
        return
    rows = []
    for d in seeds:
        preds, target = load(d)
        m = adapstwt_metrics(preds, target)
        rows.append(m)
        print(f"  {d.parent.name:<12} MAE {m['MAE']:8.4f}   "
              f"MAPE {m['MAPE']:7.4f}   RMSE {m['RMSE']:8.4f}")
    if len(rows) > 1:
        print(f"  {'mean +/- sd':<12} "
              f"MAE {np.mean([r['MAE'] for r in rows]):8.4f} "
              f"+/- {np.std([r['MAE'] for r in rows], ddof=1):.4f}   "
              f"MAPE {np.mean([r['MAPE'] for r in rows]):7.4f}   "
              f"RMSE {np.mean([r['RMSE'] for r in rows]):8.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["qdb19", "tdrive"], default=None,
                    help="score one dataset instead of both")
    ap.add_argument("--runs", default="results",
                    help="directory holding the runs (default: results)")
    args = ap.parse_args()

    base = ROOT / args.runs
    if not base.exists():
        print(f"no such directory: {base}")
        return 1

    datasets = [args.dataset] if args.dataset else ["qdb19", "tdrive"]
    for ds in datasets:
        print(f"\n{ds}")
        score(base / ds, ds, "*/predictions.npz")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
