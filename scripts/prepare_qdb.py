#!/usr/bin/env python3
"""Build the Qingdao benchmark exactly as Adap-STWT's pipeline defines it.

The point of this script is comparability, not convenience: every choice below
is taken from the Adap-STWT release (config.py, preprocess/, utils/tools.py)
rather than from our own preferences, so that a number produced under it can be
set beside theirs.

    period      2019-08-01 .. 2019-08-14, the preliminary-round files, 07:00
                to 19:00, 144 five-minute slices per day.  Identified from
                their `time_embedding()`, which iterates August 1-14, and
                confirmed against the raw timestamps.
    flow        DISTINCT VEHICLES per crossroad per slice -- they drop
                duplicates on (crossroadID, vehicleID, time_slice) before
                counting.  A vehicle seen twice in one slice counts once.
    gaps        a crossroad-slice with no record is 0, via fillna(0); it is not
                treated as missing.
    smoothing   tsmoothie KalmanSmoother(level_longseason, level noise 0.3,
                longseason noise 0.2, n_longseasons=144), applied per column
                before node selection.
    nodes       mean flow >= 20, then their coordinate list.  The coordinate
                file is not in the release, so we take the 134 busiest of the
                141 that clear the threshold, matching their node_num.
    windows     12 -> 12, built WITHIN each day.  Their per-day sliding window
                never spans the overnight gap, and neither does ours: the
                contiguity check in WindowDataset drops those windows for us.
    split       0.7 / 0.1 / 0.2, chronological.

Aggregation and smoothing run on the machine that holds the raw files; this
script converts their output into the array bundle the rest of the pipeline
reads.
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hawformer.data.build import build_adjacency                  # noqa: E402
from hawformer.utils import get_logger                            # noqa: E402

LOG = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/qdb")
    ap.add_argument("--out", default="processed/qdb")
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--smoothed", action="store_true", default=True,
                    help="use their Kalman-smoothed series (the default)")
    ap.add_argument("--raw-flow", dest="smoothed", action="store_false")
    a = ap.parse_args()

    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(raw / "qd_adapstwt.npz", allow_pickle=True)
    flow = z["flow" if a.smoothed else "flow_raw"].astype(np.float32)
    node_ids = [str(x) for x in z["node_ids"]]
    epoch = z["epoch"].astype(np.int64)
    T, N = flow.shape
    LOG.info("flow %s over %d crossroads, %s .. %s", flow.shape, N,
             pd.Timestamp(epoch[0], unit="s"), pd.Timestamp(epoch[-1], unit="s"))

    # Their pipeline has no notion of a missing cell: fillna(0) makes every
    # entry observed.  We keep that, so the evaluation sees what theirs does.
    valid = np.ones_like(flow, dtype=bool)

    bins = pd.DatetimeIndex(pd.to_datetime(epoch, unit="s"))
    steps_per_day = 288                       # 5-minute grid over a 24h clock
    tod = (bins.hour * 60 + bins.minute) // 5

    # Adjacency from the published road network, contracting paths that leave
    # the retained node set and return to it -- the link exists, it just runs
    # through a crossroad we do not model.
    adj = build_adjacency(
        np.full((N, 2), np.nan), n_nodes=N, node_ids=node_ids,
        edge_list={"path": str(raw / "qd_roadnet.csv"),
                   "from_column": "uproadID", "to_column": "downroadID",
                   "max_hops": a.max_hops},
    )
    LOG.info("adjacency density %.3f, mean out-degree %.2f",
             (adj > 0).mean(), (adj > 0).sum(1).mean() - np.mean(np.diag(adj) > 0))

    np.savez_compressed(
        out / "data.npz",
        flow=flow, valid=valid, epoch=epoch,
        tod=np.asarray(tod, dtype=np.int64),
        dow=np.asarray(bins.dayofweek, dtype=np.int64),
        node_ids=np.array(node_ids), node_latlon=np.full((N, 2), np.nan),
        adj=adj.astype(np.float32),
        steps_per_day=np.array(steps_per_day),
    )
    src = raw / "qd_corpus.txt.gz"
    with gzip.open(src, "rb") as fi, open(out / "corpus.txt", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    n = sum(1 for _ in open(out / "corpus.txt"))
    LOG.info("wrote %s and %d movement sequences", out / "data.npz", n)
    print("next: python3 scripts/prepare_micro.py --config configs/qdb.yaml")


if __name__ == "__main__":
    main()
