#!/usr/bin/env python3
"""Train and evaluate the four graph baselines on our benchmark.

    python3 scripts/gnn_baselines.py --config configs/qdb19.yaml --model stgcn --seed 0
    python3 scripts/gnn_baselines.py --config configs/qdb19.yaml --model stgcn --micro

They see exactly the windows, splits, normalisation statistics and metrics that
HAWFormer and Adap-STWT see, and write runs/<name>/<tag>_seed<k>/predictions.npz
in the same layout, so one table generator reads all of them and no separate
evaluation path can drift.

--micro additionally hands the model the microscopic trajectory graph, which is
what Table V compares: the point is to separate the value of trajectory evidence
from the value of the hierarchy built on top of it.  The road graph and the
trajectory graph are each row-normalised before being averaged, so neither is
scaled out of the fusion by its own edge-weight units.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hawformer.data.dataset import load_bundle, make_loaders                 # noqa: E402
from hawformer.metrics import adapstwt_metrics                               # noqa: E402
from hawformer.utils import configure_backends, get_logger, load_config      # noqa: E402
from gnn_models import (STGCN, DCRNN, ASTGCN, GraphWaveNet,                   # noqa: E402
                        cheb_supports, transition_supports)

LOG = get_logger()


def build(model: str, N: int, T: int, H: int, adj: np.ndarray, dev):
    if model == "stgcn":
        return STGCN(N, H, T, cheb_supports(adj, 3)).to(dev)
    if model == "dcrnn":
        return DCRNN(N, H, transition_supports(adj), hidden=64, layers=2).to(dev)
    if model == "astgcn":
        return ASTGCN(N, H, T, cheb_supports(adj, 3), channels=64, blocks=2).to(dev)
    if model == "gwnet":
        return GraphWaveNet(N, H, transition_supports(adj)).to(dev)
    raise ValueError(model)


def row_norm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a / np.maximum(a.sum(1, keepdims=True), 1e-12)


def topk_rows(a: np.ndarray, k: int) -> np.ndarray:
    """Keep the k strongest entries per row, drop the rest.

    The trajectory graph is an all-pairs similarity, so fusing it raw makes the
    adjacency 100% dense -- every baseline then does full N^2 propagation and
    Table V would be comparing density, not trajectory evidence.  HAWFormer
    sparsifies this graph before using it, so the baselines get the same budget.
    """
    a = np.asarray(a, dtype=np.float64).copy()
    if k >= a.shape[1]:
        return a
    cut = np.partition(a, -k, axis=1)[:, -k][:, None]
    return np.where(a >= cut, a, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True,
                    choices=["stgcn", "dcrnn", "astgcn", "gwnet"])
    ap.add_argument("--micro", action="store_true",
                    help="fuse the microscopic trajectory graph (Table V)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--processed", default="processed")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    configure_backends()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(a.config)
    name = cfg.get("name", Path(a.config).stem)
    pdir = Path(a.processed) / name
    b = load_bundle(str(pdir / "data.npz"), tuple(cfg["split_ratio"]),
                    bool(cfg.get("smooth", False)))
    dc = cfg["data"]
    T, H = int(dc["t_history"]), int(dc["t_horizon"])
    L = make_loaders(b, T, H, int(dc["batch_size"]), 0, int(dc.get("train_stride", 1)))

    adj = np.asarray(b.adj, dtype=np.float64)
    if a.micro:
        mp = pdir / "micro.npy"
        if not mp.exists():
            sys.exit(f"--micro needs {mp}")
        k = int(cfg.get("model", {}).get("topk", 20))
        mi = topk_rows(np.load(mp).astype(np.float64), k)
        adj = 0.5 * (row_norm(adj) + row_norm(mi))
        LOG.info("fused road + microscopic graph (top-%d per row, density %.3f)",
                 k, (adj > 0).mean())

    mu, sd = b.normalizer.mean, b.normalizer.std
    sets = {}
    for k in ("train", "val", "test"):
        st = L[k].dataset.starts
        x = np.stack([b.flow[s:s + T] for s in st]).astype(np.float32)
        y = np.stack([b.flow[s + T:s + T + H] for s in st]).astype(np.float32)
        sets[k] = (torch.from_numpy((x - mu) / sd), torch.from_numpy((y - mu) / sd),
                   y.astype(np.float64))
    LOG.info("%s / %s%s: %d/%d/%d windows over %d nodes", name, a.model,
             " +micro" if a.micro else "",
             *[sets[k][0].shape[0] for k in ("train", "val", "test")], b.num_nodes)

    model = build(a.model, b.num_nodes, T, H, adj, dev)
    LOG.info("parameters: %d", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    Xtr, Ytr, _ = sets["train"]
    Xva, _, Yva_raw = sets["val"]
    bs = int(dc["batch_size"])
    best, best_state, bad, hist = float("inf"), None, 0, []

    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(Xtr.shape[0])
        tot, t0 = 0.0, time.time()
        # DCRNN's scheduled sampling: teacher forcing decayed over training
        teacher = max(0.0, 1.0 - ep / max(a.epochs * 0.5, 1)) if a.model == "dcrnn" else 0.0
        for i in range(0, Xtr.shape[0], bs):
            j = perm[i:i + bs]
            xb, yb = Xtr[j].to(dev), Ytr[j].to(dev)
            out = model(xb, yb, teacher) if a.model == "dcrnn" else model(xb)
            loss = nn.functional.l1_loss(out, yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += float(loss.detach()) * len(j)
        model.eval()
        with torch.no_grad():
            pv = np.concatenate([model(Xva[i:i + bs].to(dev)).cpu().numpy()
                                 for i in range(0, Xva.shape[0], bs)]) * sd + mu
        vm = adapstwt_metrics(Yva_raw, pv)["MAE"]
        sched.step(vm)
        hist.append({"epoch": ep, "train_loss": tot / Xtr.shape[0], "val_MAE": float(vm),
                     "secs": round(time.time() - t0, 1)})
        LOG.info("epoch %3d | train %.4f | val MAE %.4f | %.1fs", ep, tot / Xtr.shape[0],
                 vm, time.time() - t0)
        if vm < best - 1e-6:
            best, bad = vm, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                LOG.info("early stop at epoch %d (best val MAE %.4f)", ep, best); break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    Xte, _, Yte_raw = sets["test"]
    t0 = time.time()
    with torch.no_grad():
        pred = np.concatenate([model(Xte[i:i + bs].to(dev)).cpu().numpy()
                               for i in range(0, Xte.shape[0], bs)]) * sd + mu
    infer_s = time.time() - t0
    m = adapstwt_metrics(Yte_raw, pred)
    LOG.info("TEST  MAE %.3f  MAPE %.3f  RMSE %.3f", m["MAE"], m["MAPE"], m["RMSE"])

    tag = a.tag or (("micro_" if a.micro else "") + a.model)
    out = Path(a.runs) / name / f"{tag}_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", pred=pred.astype(np.float32),
                        true=Yte_raw.astype(np.float32), mask=np.isfinite(Yte_raw))
    # epoch_budget is stamped so the runner can tell a finished run apart from a
    # finished run under a *different* budget.  Every row of Tables III and V
    # has to share one budget or the comparison is between schedules, not
    # models -- and the first pass stopped STGCN while its validation curve was
    # still falling steeply.
    json.dump({"model": a.model, "micro": a.micro, "seed": a.seed, "adapstwt": m,
               "epoch_budget": int(a.epochs), "patience": int(a.patience),
               "best_val_MAE": float(best), "epochs_run": len(hist),
               "inference_seconds": round(infer_s, 3),
               "parameters": sum(p.numel() for p in model.parameters()),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)
               if torch.cuda.is_available() else None,
               "history": hist}, open(out / "report.json", "w"), indent=1)
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
