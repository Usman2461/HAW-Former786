#!/usr/bin/env python3
"""The two ablation series, scored the same way as every other table.

Reads the ablation run directories, re-scores their predictions under the
Adap-STWT protocol -- rather than trusting each run's own report -- and emits
one JSON the paper's ablation tables and Figures 12 and 13 are built from.

Each series is reported as a cumulative build-up, so the number that matters is
the change from the row above, not the absolute value: that is what "the
contribution of each component" means, and it is the only reading that survives
a reviewer asking why the rows are not independent.

A variant with no completed run is omitted with a reason.  Nothing is
interpolated between variants.

    python scripts/ablation_table.py --out tables/ablation_values.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hawformer.data.dataset import load_bundle, make_loaders            # noqa: E402
from hawformer.metrics import adapstwt_metrics                          # noqa: E402
from hawformer.utils import get_logger, load_config                     # noqa: E402

LOG = get_logger()

# (run tag, the row label the paper prints).  Order is the build-up order.
HAGL_SERIES = [
    ("abl_static",      "Static Graph + BWST"),
    ("abl_local",       "Local Graph + BWST"),
    ("abl_macro",       "Macro Graph + BWST"),
    ("abl_macromicro",  "Macro+Micro + BWST"),
    ("abl_noaffinity",  "HAGL w/o affinity + BWST"),
    ("FULL",            "HAWFormer"),
]
BWST_SERIES = [
    ("abl_stt",         "HAGL + STT"),
    ("abl_nowavelet",   "HAGL + Emb + STT"),
    ("abl_nodla",       "HAGL + Wavelet-STT"),
    ("abl_gamma1",      r"HAGL + BWST ($\gamma=1$)"),
    ("FULL",            "HAWFormer"),
]
TARGETED = [
    ("abl_hardmem",     "hard semantic membership"),
    ("abl_nogcn",       "no Chebyshev branch"),
    ("abl_statichier",  "hierarchy never refreshed"),
]


def seed_dirs(root: Path, tag: str) -> List[Path]:
    return sorted(p for p in root.glob(f"{tag}_seed*")
                  if (p / "predictions.npz").exists())


def agg(v: List[float]) -> Dict[str, float]:
    a = np.asarray(v, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}


def score(root: Path, tag: str, y: np.ndarray) -> Optional[Dict]:
    ds = seed_dirs(root, tag)
    if not ds:
        return None
    ms = [adapstwt_metrics(y, np.load(p / "predictions.npz")["pred"]) for p in ds]
    return {k: agg([m[k] for m in ms]) for k in ("MAE", "MAPE", "RMSE")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--configs", nargs="+",
                    default=["configs/qdb19.yaml", "configs/tdrive.yaml"])
    ap.add_argument("--out", default="tables/ablation_values.json")
    ap.add_argument("--prefix", default="abl_",
                    help="tag prefix for the ablation runs (pin_ for the pinned schedule)")
    ap.add_argument("--full-tag", default=None,
                    help="tag of the full model these ablations belong to")
    a = ap.parse_args()

    out: Dict = {}
    for cfg_path in a.configs:
        cfg = load_config(cfg_path)
        name = cfg.get("name", Path(cfg_path).stem)
        b = load_bundle(f"processed/{name}/data.npz", tuple(cfg["split_ratio"]),
                        bool(cfg.get("smooth", False)))
        dc = cfg["data"]
        H, P = int(dc["t_history"]), int(dc["t_horizon"])
        L = make_loaders(b, H, P, int(dc["batch_size"]), 0,
                         int(dc.get("train_stride", 1)))
        st = np.asarray(L["test"].dataset.starts)
        y = np.stack([b.flow[s + H:s + H + P] for s in st])

        root = Path(a.runs) / name
        full_tag = a.full_tag or ("final" if seed_dirs(root, "final") else "paper")
        rec: Dict = {"dataset": name, "full_tag": full_tag,
                     "series": {}, "targeted": {}, "missing": []}

        for series_name, series in (("HAGL", HAGL_SERIES), ("BWST", BWST_SERIES)):
            rows = []
            for tag, label in series:
                t = full_tag if tag == "FULL" else a.prefix + tag[4:]
                m = score(root, t, y)
                if m is None:
                    rec["missing"].append(f"{series_name}: {label} ({t})")
                    continue
                rows.append({"tag": t, "label": label, **m})
            # the change each component buys, relative to the row above it
            for i in range(1, len(rows)):
                prev, cur = rows[i - 1]["MAE"]["mean"], rows[i]["MAE"]["mean"]
                rows[i]["delta_MAE_pct"] = float(100.0 * (prev - cur) / prev)
            if rows:
                first, last = rows[0]["MAE"]["mean"], rows[-1]["MAE"]["mean"]
                rec["series"][series_name] = {
                    "rows": rows,
                    "total_MAE_pct": float(100.0 * (first - last) / first)}

        base = score(root, full_tag, y)
        for tag, label in TARGETED:
            t = a.prefix + tag[4:]
            m = score(root, t, y)
            if m is None:
                rec["missing"].append(f"targeted: {label} ({t})")
                continue
            d = {"tag": t, "label": label, **m}
            if base:
                d["delta_MAE_pct"] = float(
                    100.0 * (m["MAE"]["mean"] - base["MAE"]["mean"])
                    / base["MAE"]["mean"])
            rec["targeted"][tag] = d
        out[name] = rec

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    for name, rec in out.items():
        print(f"\n=== {name}  (full model = {rec['full_tag']})")
        for sn, s in rec["series"].items():
            print(f"  -- {sn} series, cumulative build-up "
                  f"(total {s['total_MAE_pct']:+.1f}% MAE)")
            for r in s["rows"]:
                d = (f"{r['delta_MAE_pct']:+6.2f}%" if "delta_MAE_pct" in r
                     else "   base")
                print(f"     {r['label']:<30} MAE {r['MAE']['mean']:8.4f}"
                      f" +-{r['MAE']['std']:.4f} (n={r['MAE']['n']})   {d}")
        if rec["targeted"]:
            print("  -- targeted (positive = worse than the full model)")
            for t in rec["targeted"].values():
                print(f"     {t['label']:<30} MAE {t['MAE']['mean']:8.4f}"
                      f" +-{t['MAE']['std']:.4f}   "
                      f"{t.get('delta_MAE_pct', float('nan')):+6.2f}%")
        for m in rec["missing"]:
            print(f"  not run: {m}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
