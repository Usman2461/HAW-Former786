#!/usr/bin/env python3
"""Choose an ARIMA order on the training split, by AIC.

The Adap-STWT comparison fixes (2,1,2) on Qingdao and (1,1,2) on Chengdu.  A
third dataset needs an order chosen the same way rather than borrowed, so this
fits a small (p,d,q) grid to a sample of nodes' training series and keeps the
order with the lowest mean AIC.  d is restricted to {0,1}: five-minute flow at a
single node is close to stationary within a day, and higher differencing mostly
amplifies the sampling noise.

Writes tables/arima_order_<dataset>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hawformer.data.dataset import load_bundle                          # noqa: E402
from hawformer.utils import get_logger, load_config                     # noqa: E402

LOG = get_logger()
warnings.filterwarnings("ignore")

GRID = [(p, d, q) for d in (0, 1) for p in (1, 2, 3) for q in (0, 1, 2)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--nodes", type=int, default=25, help="sample of busiest nodes")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()

    from statsmodels.tsa.arima.model import ARIMA

    cfg = load_config(a.config)
    name = cfg.get("name", Path(a.config).stem)
    b = load_bundle(f"processed/{name}/data.npz", tuple(cfg["split_ratio"]),
                    bool(cfg.get("smooth", False)))
    tr0, tr1 = b.splits["train"]
    train = b.flow[tr0:tr1].astype(np.float64)
    pick = np.argsort(-train.mean(0))[:min(a.nodes, train.shape[1])]

    scores = {}
    for order in GRID:
        aics = []
        for n in pick:
            try:
                aics.append(float(ARIMA(train[:, n], order=order,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False).fit().aic))
            except Exception:                                    # noqa: BLE001
                pass
        if len(aics) >= max(3, len(pick) // 2):
            scores[order] = float(np.mean(aics))
            LOG.info("order %s  mean AIC %.1f  (%d/%d nodes)", order, scores[order],
                     len(aics), len(pick))

    if not scores:
        LOG.error("no order fitted; leaving the default in place")
        sys.exit(1)
    best = min(scores, key=scores.get)
    LOG.info("selected %s for %s", best, name)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"dataset": name, "order": list(best),
               "order_csv": ",".join(str(v) for v in best),
               "nodes_sampled": int(len(pick)),
               "mean_aic": {str(k): v for k, v in sorted(scores.items(),
                                                         key=lambda kv: kv[1])}},
              open(out / f"arima_order_{name}.json", "w"), indent=1)
    print(",".join(str(v) for v in best))


if __name__ == "__main__":
    main()
