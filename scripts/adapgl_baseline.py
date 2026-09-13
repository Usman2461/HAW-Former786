#!/usr/bin/env python3
"""AdapGL on our benchmark, using the authors' released model.

    python3 scripts/adapgl_baseline.py --config configs/qdb19.yaml --seed 0

AdapGL alternates two objectives: the predictor (AdapGLA) is trained on a fixed
adjacency, then the graph learner (GraphLearn) proposes a new one, and the graph
that scores best on validation is kept.  That schedule is reproduced here from
their paper; the modules themselves are imported from the release unchanged.

One caveat travels with every number this produces: model/ASTGCN.py is missing
from the release, so AdapGL cannot be imported as shipped.  The TemporalAttention
it needs is reconstructed in adapstwt/Adap-STWT-main/model/ASTGCN.py, with the
interface pinned by AdapGL's own call site.  It is a faithful reconstruction of a
missing file, not the authors' code.

Writes runs/<name>/adapgl_seed<k>/ in the same layout as every other model.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from hawformer.data.dataset import load_bundle, make_loaders                 # noqa: E402
from hawformer.metrics import adapstwt_metrics                               # noqa: E402
from hawformer.utils import get_logger, load_config                          # noqa: E402

LOG = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", default=None, help="Adap-STWT release root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--processed", default="processed")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--pred-epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--graph-lr", type=float, default=1e-3)
    ap.add_argument("--feature-num", type=int, default=32)
    ap.add_argument("--micro", action="store_true",
                    help="fuse the microscopic trajectory graph (Table V)")
    a = ap.parse_args()

    # Absolute, before anything touches sys.path: the import below runs after a
    # chdir into src, so a relative --src ("../adapstwt/...") would already be
    # dangling on sys.path by the time Python resolves it -- which is how every
    # AdapGL run failed with ModuleNotFoundError while the path check passed.
    src = (Path(a.src) if a.src else
           (HERE.parent.parent.parent / "adapstwt" / "Adap-STWT-main")).resolve()
    if not (src / "model" / "AdapGL" / "AdapGL.py").exists():
        sys.exit(f"AdapGL source not found under {src}")
    sys.path.insert(0, str(src))
    cwd = os.getcwd(); os.chdir(src)
    from model.AdapGL.AdapGL import AdapGLA, GraphLearn                       # noqa: E402
    os.chdir(cwd)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(a.config)
    name = cfg.get("name", Path(a.config).stem)
    b = load_bundle(str(Path(a.processed) / name / "data.npz"),
                    tuple(cfg["split_ratio"]), bool(cfg.get("smooth", False)))
    dc = cfg["data"]
    T, H, bs = int(dc["t_history"]), int(dc["t_horizon"]), int(dc["batch_size"])
    L = make_loaders(b, T, H, bs, 0, int(dc.get("train_stride", 1)))
    N = b.num_nodes
    mu, sd = b.normalizer.mean, b.normalizer.std

    sets = {}
    for k in ("train", "val", "test"):
        st = L[k].dataset.starts
        x = np.stack([b.flow[s:s + T] for s in st]).astype(np.float32)
        y = np.stack([b.flow[s + T:s + T + H] for s in st]).astype(np.float32)
        sets[k] = (torch.from_numpy((x - mu) / sd).unsqueeze(-1),   # [B,T,N,1]
                   torch.from_numpy((y - mu) / sd), y.astype(np.float64))
    LOG.info("%s adapgl: %d/%d/%d windows over %d nodes",
             name, *[sets[k][0].shape[0] for k in ("train", "val", "test")], N)

    pred = AdapGLA(num_nodes=N, step_num_in=T, step_num_out=H, input_size=1,
                   num_block=2, num_cheb_filter=64, num_time_filter=64,
                   K=3, conv_type="cheb").to(dev)
    gl = GraphLearn(N, a.feature_num).to(dev)
    LOG.info("parameters: predictor %d, graph learner %d",
             sum(p.numel() for p in pred.parameters()),
             sum(p.numel() for p in gl.parameters()))
    opt_p = torch.optim.Adam(pred.parameters(), lr=a.lr, weight_decay=1e-4)
    opt_g = torch.optim.Adam(gl.parameters(), lr=a.graph_lr)

    adj0 = np.asarray(b.adj, dtype=np.float64)
    if a.micro:
        sys.path.insert(0, str(HERE))
        from gnn_baselines import row_norm, topk_rows
        mp = Path(a.processed) / name / "micro.npy"
        if not mp.exists():
            sys.exit(f"--micro needs {mp}")
        k = int(cfg.get("model", {}).get("topk", 20))
        adj0 = 0.5 * (row_norm(adj0) + row_norm(topk_rows(np.load(mp).astype(np.float64), k)))
        LOG.info("fused road + microscopic graph (top-%d per row, density %.3f)",
                 k, (adj0 > 0).mean())
    A0 = torch.tensor(adj0.astype(np.float32), device=dev)
    A0 = A0 / torch.clamp(A0.sum(1, keepdim=True), min=1e-6)
    best_adj = A0.clone()

    def run_eval(split, adj):
        pred.eval()
        X, _, Yraw = sets[split]
        with torch.no_grad():
            p = np.concatenate([pred(X[i:i + bs].to(dev), adj).cpu().numpy()
                                for i in range(0, X.shape[0], bs)])
        p = p.reshape(p.shape[0], H, N) * sd + mu
        return adapstwt_metrics(Yraw, p), p

    Xtr, Ytr, _ = sets["train"]
    best, bad, hist = float("inf"), 0, []
    best_state, best_graph = None, best_adj.clone()
    t_start = time.time()

    for rnd in range(1, a.rounds + 1):
        pred.train()
        for _ in range(a.pred_epochs):
            perm = torch.randperm(Xtr.shape[0])
            for i in range(0, Xtr.shape[0], bs):
                j = perm[i:i + bs]
                out = pred(Xtr[j].to(dev), best_adj).reshape(-1, H, N)
                loss = nn.functional.l1_loss(out, Ytr[j].to(dev))
                opt_p.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(pred.parameters(), 5.0); opt_p.step()

        # graph stage: the predictor is frozen while the adjacency is proposed
        gl.train(); pred.eval()
        for p_ in pred.parameters():
            p_.requires_grad_(False)
        perm = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0], bs):
            j = perm[i:i + bs]
            A_new = gl(A0)
            out = pred(Xtr[j].to(dev), A_new).reshape(-1, H, N)
            loss = nn.functional.l1_loss(out, Ytr[j].to(dev))
            opt_g.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(gl.parameters(), 5.0); opt_g.step()
        for p_ in pred.parameters():
            p_.requires_grad_(True)

        with torch.no_grad():
            cand = gl(A0).detach()
        # keep whichever of the two graphs validation prefers
        m_old, _ = run_eval("val", best_adj)
        m_new, _ = run_eval("val", cand)
        if m_new["MAE"] < m_old["MAE"]:
            best_adj = cand
        cur = min(m_old["MAE"], m_new["MAE"])
        hist.append({"round": rnd, "val_MAE": float(cur),
                     "elapsed_s": round(time.time() - t_start, 1)})
        LOG.info("round %2d | val MAE %.4f (kept %s) | %.0fs", rnd, cur,
                 "learned" if m_new["MAE"] < m_old["MAE"] else "previous",
                 time.time() - t_start)
        if cur < best - 1e-6:
            best, bad = cur, 0
            best_state = {k: v.detach().clone() for k, v in pred.state_dict().items()}
            best_graph = best_adj.clone()
        else:
            bad += 1
            if bad >= a.patience:
                LOG.info("stopping: %d rounds without improvement", bad); break

    if best_state is not None:
        pred.load_state_dict(best_state)
    t0 = time.time()
    m, p = run_eval("test", best_graph)
    infer_s = time.time() - t0
    LOG.info("TEST  MAE %.3f  MAPE %.3f  RMSE %.3f", m["MAE"], m["MAPE"], m["RMSE"])

    tag = a.tag or ("micro_adapgl" if a.micro else "adapgl")
    out = Path(a.runs) / name / f"{tag}_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    Yraw = sets["test"][2]
    np.savez_compressed(out / "predictions.npz", pred=p.astype(np.float32),
                        true=Yraw.astype(np.float32), mask=np.isfinite(Yraw))
    json.dump({"model": "AdapGL", "micro": a.micro, "seed": a.seed, "adapstwt": m,
               "epoch_budget": int(a.rounds) * int(a.pred_epochs),
               "rounds": int(a.rounds), "pred_epochs": int(a.pred_epochs),
               "patience": int(a.patience),
               "best_val_MAE": float(best), "rounds_run": len(hist),
               "inference_seconds": round(infer_s, 3),
               "reconstructed_dependency": "model/ASTGCN.py (TemporalAttention) is "
                                           "absent from the release and was rebuilt",
               "parameters": sum(x.numel() for x in pred.parameters())
                             + sum(x.numel() for x in gl.parameters()),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)
               if torch.cuda.is_available() else None,
               "history": hist}, open(out / "report.json", "w"), indent=1)
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
