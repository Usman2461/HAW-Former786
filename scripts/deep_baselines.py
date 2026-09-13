#!/usr/bin/env python3
"""The two baselines from the Adap-STWT table that this machine can honestly run.

    python3 scripts/deep_baselines.py --config configs/qdb19.yaml --model lstm --seed 0
    python3 scripts/deep_baselines.py --config configs/qdb19.yaml --model arima

Both consume exactly the windows, splits, normalisation statistics and metrics
that a HAWFormer run consumes, and both write runs/<name>/<tag>_seed<k>/
predictions.npz in the same layout, so the table generator cannot tell them
apart from a trained run and no separate evaluation path can drift.

  lstm    Two stacked LSTM layers over the 12-step history of a single node,
          weights shared across nodes -- the univariate sequence baseline as it
          is normally specified in this literature.  It has no access to the
          graph, which is the point of including it.  Trained on z-scored
          targets with Huber loss, Adam, early stopping on validation MAE.
  arima   One model per node, order taken from --order, fitted once on the
          training split and then applied to each test window's history before
          forecasting 12 steps.  Refitting per window would be 61k fits; the
          apply-and-forecast route is the standard compromise and is what the
          published comparisons describe.

Deliberately NOT here: DCRNN, STGCN, ASTGCN, Graph WaveNet, AdapGL and
Adap-STWT.  Faithful versions of those need weeks of tuning or a GPU, and a
hasty reimplementation would produce a number that flatters this paper for the
wrong reason.  They stay unfilled.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hawformer.data.dataset import load_bundle, make_loaders            # noqa: E402
from hawformer.metrics import adapstwt_metrics                          # noqa: E402
from hawformer.utils import get_logger, load_config                     # noqa: E402

LOG = get_logger()
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------

def windows(bundle, starts, H, P):
    """[B, H, N] history and [B, P, N] target, original scale."""
    x = np.stack([bundle.flow[s:s + H] for s in starts]).astype(np.float32)
    y = np.stack([bundle.flow[s + H:s + H + P] for s in starts]).astype(np.float32)
    return x, y


class LSTMBaseline(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 2, horizon: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):                       # x: [B, H, N] normalised
        B, H, N = x.shape
        z = x.permute(0, 2, 1).reshape(B * N, H, 1)
        o, _ = self.lstm(z)
        y = self.head(o[:, -1])                 # [B*N, P]
        return y.reshape(B, N, -1).permute(0, 2, 1)


def run_lstm(bundle, sets, H, P, seed, epochs, patience, lr, batch):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = bundle.normalizer.mean, bundle.normalizer.std
    dev = torch.device("cpu")
    net = LSTMBaseline(horizon=P).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=1.0)

    def to_t(a):
        return torch.from_numpy((a - mu) / sd)

    Xtr, Ytr = (to_t(v) for v in sets["train"])
    Xva, Yva = sets["val"]
    Xva_t = to_t(Xva)

    best, best_state, bad = float("inf"), None, 0
    n = Xtr.shape[0]
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n)
        t0, tot = time.time(), 0.0
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            opt.zero_grad()
            out = net(Xtr[j])
            loss = lossf(out, Ytr[j])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(j)
        net.eval()
        with torch.no_grad():
            pv = net(Xva_t).numpy() * sd + mu
        vm = adapstwt_metrics(Yva, pv)["MAE"]
        LOG.info("epoch %3d | train %.4f | val MAE %.4f | %.1fs", ep, tot / n, vm,
                 time.time() - t0)
        if vm < best - 1e-6:
            best, bad = vm, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                LOG.info("early stop at epoch %d (best val MAE %.4f)", ep, best)
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(to_t(sets["test"][0])).numpy() * sd + mu
    return pred, {"best_val_MAE": float(best)}


def run_arima(bundle, sets, H, P, order):
    from statsmodels.tsa.arima.model import ARIMA
    tr0, tr1 = bundle.splits["train"]
    train = bundle.flow[tr0:tr1]
    Xte = sets["test"][0]
    B, _, N = Xte.shape
    pred = np.zeros((B, P, N), dtype=np.float32)
    t0 = time.time()
    for n in range(N):
        col = train[:, n].astype(np.float64)
        try:
            fit = ARIMA(col, order=order,
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit()
        except Exception as e:                                   # noqa: BLE001
            LOG.warning("node %d: ARIMA fit failed (%s); falling back to persistence", n, e)
            pred[:, :, n] = Xte[:, -1, n][:, None]
            continue
        for b in range(B):
            try:
                pred[b, :, n] = fit.apply(Xte[b, :, n].astype(np.float64),
                                          refit=False).forecast(P)
            except Exception:                                    # noqa: BLE001
                pred[b, :, n] = Xte[b, -1, n]
        if (n + 1) % 20 == 0:
            LOG.info("  %d/%d nodes, %.1fs elapsed", n + 1, N, time.time() - t0)
    pred = np.clip(pred, 0.0, None)
    return pred, {"order": list(order)}


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", choices=["lstm", "arima"], required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--order", default="2,1,2")
    a = ap.parse_args()

    cfg = load_config(a.config)
    name = cfg.get("name", Path(a.config).stem)
    b = load_bundle(f"processed/{name}/data.npz", tuple(cfg["split_ratio"]),
                    bool(cfg.get("smooth", False)))
    dc = cfg["data"]
    H, P = int(dc["t_history"]), int(dc["t_horizon"])
    L = make_loaders(b, H, P, int(dc["batch_size"]), 0, int(dc.get("train_stride", 1)))
    sets = {k: windows(b, L[k].dataset.starts, H, P) for k in ("train", "val", "test")}
    LOG.info("%s: %s -> %d/%d/%d windows over %d nodes", name, a.model,
             *[sets[k][0].shape[0] for k in ("train", "val", "test")], b.num_nodes)

    if a.model == "lstm":
        pred, extra = run_lstm(b, sets, H, P, a.seed, a.epochs, a.patience, a.lr,
                               int(dc["batch_size"]))
    else:
        pred, extra = run_arima(b, sets, H, P, tuple(int(v) for v in a.order.split(",")))

    y = sets["test"][1]
    m = adapstwt_metrics(y, pred)
    LOG.info("TEST  MAE %.3f  MAPE %.3f  RMSE %.3f", m["MAE"], m["MAPE"], m["RMSE"])

    tag = a.tag or a.model
    out = Path(a.runs) / name / f"{tag}_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", pred=pred.astype(np.float32),
                        true=y.astype(np.float32),
                        mask=np.isfinite(y))
    json.dump({"model": a.model, "seed": a.seed, "adapstwt": m, **extra},
              open(out / "report.json", "w"), indent=1)
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
