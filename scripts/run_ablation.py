#!/usr/bin/env python3
"""Run the ablation grid of Section VII-E and print a comparison table.

Each variant disables exactly ONE mechanism, so the change in error is
attributable to that mechanism alone.  Variants map one-to-one onto Table V of
the paper.

    python scripts/run_ablation.py --config configs/chengdu.yaml --seeds 0 1 2
    python scripts/run_ablation.py --config configs/qingdao.yaml \
        --only full gamma1 static_hier
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# name -> (dotted overrides, what the paper calls it, requirement tested)
VARIANTS: Dict[str, tuple] = {
    "full":        ({}, "HAWFormer", "-"),
    "no_micro":    ({"model.use_micro": False}, "w/o Micro", "R1"),
    "no_high":     ({"model.use_high": False}, "w/o High", "R1"),
    "no_meso":     ({"model.use_meso": False}, "w/o Meso", "R2"),
    "static_hier": ({"model.static_hierarchy": True}, "static-Hier", "R2"),
    "hard_hs":     ({"model.soft_membership": False}, "hard-H_s", "R2"),
    "identity_aff":({"model.identity_affinity": True}, "Lambda = I", "R2"),
    "no_spectral": ({"model.use_spectral_descriptor": False}, "w/o Spectral", "R2"),
    "gamma1":      ({"model.gamma": 1.0}, "gamma = 1", "R3"),
    "no_wavelet":  ({"model.use_wavelet": False}, "w/o Wavelet", "R3"),
    "no_dla":      ({"model.use_dla": False}, "w/o DLA", "R3"),
    "no_gcn":      ({"model.use_gcn": False}, "w/o GCN", "-"),
    "no_bank":     ({"train.use_bank": False}, "w/o Bank", "-"),
    "no_alt":      ({"train.alternating": False}, "w/o Alt", "-"),
}


def run_one(config: str, name: str, overrides: dict, seed: int,
            epochs: int, extra: List[str]) -> Path:
    cmd = [sys.executable, "scripts/train.py", "--config", config,
           "--tag", name, "--seed", str(seed)]
    if epochs:
        cmd += ["--epochs", str(epochs)]
    for k, v in overrides.items():
        cmd += ["--set", f"{k}={json.dumps(v)}"]
    cmd += extra
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    cfg_name = Path(config).stem
    return Path("runs") / cfg_name / f"{name}_seed{seed}" / "report.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()

    names = a.only or list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}\nknown: {list(VARIANTS)}")

    rows = []
    for name in names:
        ov, label, req = VARIANTS[name]
        per_seed = []
        for s in a.seeds:
            rp = run_one(a.config, name, ov, s, a.epochs, a.extra)
            with open(rp) as f:
                rep = json.load(f)["test"]["overall"]
            per_seed.append(rep)
        agg = {}
        for k in ("MAE", "RMSE", "WAPE"):
            v = np.array([p[k] for p in per_seed], dtype=float)
            agg[k] = (float(np.nanmean(v)), float(np.nanstd(v)))
        rows.append((label, req, agg))

    print("\n" + "=" * 78)
    print(f"{'Variant':<16}{'Req':<5}{'MAE':>18}{'RMSE':>18}{'WAPE %':>18}")
    print("-" * 78)
    base = rows[0][2] if names[0] == "full" else None
    for label, req, agg in rows:
        cells = []
        for k in ("MAE", "RMSE", "WAPE"):
            m, sd = agg[k]
            cells.append(f"{m:9.4f}+-{sd:<6.4f}")
        print(f"{label:<16}{req:<5}" + "".join(f"{c:>18}" for c in cells))
    print("=" * 78)
    if base is not None and len(a.seeds) > 1:
        print("\nA variant whose difference from HAWFormer is smaller than the "
              "seed standard deviation is inconclusive, not evidence.")


if __name__ == "__main__":
    main()
