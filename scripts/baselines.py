#!/usr/bin/env python3
"""Zero-training reference predictors, evaluated on exactly the test windows.

Reporting a model's MAE without an anchor tells a reader very little: is 18.9
vehicles per five minutes good?  These two baselines answer that, cost seconds
rather than hours, and are standard:

  copy_last   persistence -- repeat the last observed step across the horizon.
              Hard to beat at short horizons, and the honest floor for "did the
              model learn anything beyond the current level?"
  ha          historical average -- for each node and time-of-day slot, the mean
              over the TRAINING split only.  Captures the daily profile and
              nothing else, so the gap between HA and a model is the part of the
              signal that is not simply "this is what this node does at 8am".

Both are evaluated through the same windows, mask and metrics as a trained run,
and written as report.json so they drop into the same tables and figures.

    python scripts/baselines.py --config configs/qingdao.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hawformer.data.dataset import load_bundle, make_loaders            # noqa: E402
from hawformer.evaluate import format_report, full_report               # noqa: E402
from hawformer.utils import get_logger, load_config, save_json          # noqa: E402

LOG = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--processed", default="processed")
    a = ap.parse_args()

    cfg = load_config(a.config)
    name = cfg.get("name", Path(a.config).stem)
    bundle = load_bundle(str(Path(a.processed) / name / "data.npz"),
                         tuple(cfg["split_ratio"]), bool(cfg["smooth"]))
    dc, ev = cfg["data"], cfg.get("eval", {})
    th, tp = int(dc["t_history"]), int(dc["t_horizon"])
    loaders = make_loaders(bundle, th, tp, int(dc["batch_size"]), 0,
                           int(dc.get("train_stride", 1)))
    starts = loaders["test"].dataset.starts
    flow, valid, tod = bundle.flow, bundle.valid, bundle.tod

    # historical average over the training split only -- using the whole series
    # would leak the test period into the baseline
    tr0, tr1 = bundle.splits["train"]
    n_slot = int(bundle.steps_per_day)
    N = bundle.num_nodes
    ha_tab = np.full((n_slot, N), np.nan)
    for slot in range(n_slot):
        sel = np.where(tod[tr0:tr1] == slot)[0] + tr0
        if len(sel):
            v = flow[sel]
            m = valid[sel]
            with np.errstate(invalid="ignore"):
                ha_tab[slot] = np.where(m.sum(0) > 0,
                                        np.nansum(np.where(m, v, 0.0), 0)
                                        / np.maximum(m.sum(0), 1), np.nan)
    gm = np.nanmean(flow[tr0:tr1][valid[tr0:tr1]]) if valid[tr0:tr1].any() else 0.0
    ha_tab = np.where(np.isfinite(ha_tab), ha_tab, gm)

    B = len(starts)
    y_true = np.empty((B, tp, N), np.float32)
    mask = np.empty((B, tp, N), bool)
    p_last = np.empty((B, tp, N), np.float32)
    p_ha = np.empty((B, tp, N), np.float32)
    for b, s in enumerate(starts):
        e = int(s) + th
        y_true[b] = flow[e:e + tp]
        mask[b] = valid[e:e + tp]
        p_last[b] = np.repeat(flow[e - 1][None, :], tp, axis=0)
        p_ha[b] = ha_tab[tod[e:e + tp]]

    for tag, pred in (("copy_last", p_last), ("ha", p_ha)):
        out = Path(a.runs) / name / f"{tag}_seed0"
        out.mkdir(parents=True, exist_ok=True)
        rep = full_report(y_true, pred, mask, bundle, starts, th,
                          horizons=tuple(ev.get("horizons", (3, 6, 9, 12))),
                          mape_threshold=float(ev.get("mape_threshold", 5.0)))
        save_json({"config": {"name": name, "baseline": tag},
                   "best": {"val_MAE": None, "epoch": 0}, "test": rep},
                  str(out / "report.json"))
        np.savez_compressed(out / "predictions.npz",
                            pred=pred, true=y_true, mask=mask)
        print(format_report(rep, f"{name} / {tag}"))
        print()


if __name__ == "__main__":
    main()
