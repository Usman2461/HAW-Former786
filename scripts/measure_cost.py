#!/usr/bin/env python3
"""Measure training and inference cost, per component and per model.

    python scripts/measure_cost.py --configs configs/qingdao.yaml configs/tdrive.yaml

The aggregate cost of the framework conceals the fact that its parts have very
different profiles: the microscopic branch is expensive but runs offline, while
the predictor is the only thing on the inference path.  A single "seconds per
epoch" number would hide exactly the property the paper claims.  So each part
is timed separately, on real batch shapes from the real datasets:

  Macro / Meso branch   forward+backward of that branch of the graph learner,
                        which runs on the graph-learner stage of the schedule.
  Micro branch          building the transition graph from the trajectory
                        corpus.  This is preprocessing -- it happens once,
                        before training -- so it is reported as a one-off in
                        the notes rather than amortised into a per-epoch cost.
  BWST predictor        forward+backward of the predictor alone, the quantity
                        that governs both training and deployment.

Inference is timed with the graph and partition frozen, which is the deployed
configuration: the graph learner contributes nothing to the forward pass.

Memory is peak RSS of the process, measured as the maximum over the timed
region, and reported per configuration rather than per component -- attributing
resident memory to a submodule of a shared process would be a fiction.

Writes tables/table_cost.tex and a machine-readable cost.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hawformer.data.dataset import load_bundle, make_loaders            # noqa: E402
from hawformer.models.hawformer import huber_loss                       # noqa: E402
from hawformer.utils import get_logger, load_config, save_json          # noqa: E402

# The model is built by train.py's own builder rather than a copy of it: a
# second construction path would be free to drift, and then the cost table
# would describe a model that was never trained.
from train import build_model                                           # noqa: E402
from hawformer.train import _predictor_params                           # noqa: E402

LOG = get_logger()


def peak_rss_gb() -> float:
    """Peak resident set size of this process, in GB.

    ru_maxrss is kilobytes on Linux and bytes on macOS; this runs on Linux.
    It is a high-water mark for the whole process, so it is only meaningful
    when compared between runs of the same script, which is how it is used.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def timeit(fn, n: int, warmup: int = 2) -> float:
    """Median seconds per call, discarding warm-up iterations.

    The median rather than the mean: on a shared container a single descheduled
    iteration would otherwise dominate the average and be reported as cost.
    """
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _node_cells(cfg: dict, bundle):
    """The tokens the corpus uses for each node.

    Qingdao's records already name a crossroad, so the node id *is* the token;
    T-Drive and Porto are raw GPS, so the token is the node's GeoHash cell.
    """
    latlon = bundle.node_latlon
    if latlon is None or not np.isfinite(latlon).all():
        return [str(n) for n in bundle.node_ids]
    from hawformer.data.geohash import encode
    p = int(cfg["raw"].get("geohash_precision", 7))
    return [encode(float(la), float(lo), p) for la, lo in latlon]


def measure(cfg_path: str, runs_dir: str, processed: str,
            n_iter: int, skip_micro: bool = False) -> Dict[str, float]:
    cfg = load_config(cfg_path)
    name = cfg.get("name", Path(cfg_path).stem)
    device = "cpu"
    bundle = load_bundle(str(Path(processed) / name / "data.npz"),
                         tuple(cfg["split_ratio"]), bool(cfg["smooth"]))
    dc = cfg["data"]
    loaders = make_loaders(bundle, int(dc["t_history"]), int(dc["t_horizon"]),
                           int(dc["batch_size"]), 0,
                           int(dc.get("train_stride", 1)))
    model = build_model(cfg, bundle, Path(processed) / name, device)
    n_batches = len(loaders["train"])
    # WindowDataset yields (x, y, mask, t_in, t_out); the predictor takes the
    # history time index.
    x, y, m, t_idx, _ = next(iter(loaders["train"]))
    x, y, m, t_idx = x.to(device), y.to(device), m.to(device), t_idx.to(device)

    out: Dict[str, float] = {
        "name": name, "num_nodes": int(bundle.num_nodes),
        "params_total": int(sum(p.numel() for p in model.parameters())),
        "train_batches_per_epoch": n_batches,
        "batch_size": int(dc["batch_size"]),
    }

    # -- predictor: forward + backward, the per-batch training cost ----------
    A = model.A_frozen

    def predictor_step():
        model.zero_grad(set_to_none=True)
        loss = huber_loss(model(x, t_idx, A)["pred"], y, m)
        loss.backward()

    out["predictor_s_per_batch"] = timeit(predictor_step, n_iter)

    # -- graph learner: the ACTUAL stage-2 step, not just the HAGL forward ---
    # Timing `learn_graph` alone would understate this stage by an order of
    # magnitude, because stage 2 also pushes a full predictor forward through
    # the learned graph and backpropagates into the learner.  It runs
    # `graph_steps` batches per epoch, not once, so the per-epoch figure is
    # that product -- getting this wrong would let the paper claim the
    # structural machinery is free when it is not.
    xs = x.squeeze(-1)
    tcfg = cfg.get("train", {})
    graph_steps = int(tcfg.get("graph_steps", 20))
    lam_a = float(tcfg.get("lambda_a", 1e-5))
    lam_c = float(tcfg.get("lambda_c", 1e-4))
    delta = float(tcfg.get("huber_delta", 1.0))

    # Stage 2 freezes the predictor's parameters for its duration, so backward
    # populates gradients for the graph learner only.  Timing it without that
    # freeze measures a backward pass through the whole model and overstates
    # the stage by 50-90%, which would turn a real cost into an inflated one.
    opt_graph = torch.optim.AdamW(model.hagl.parameters(),
                                  lr=float(tcfg.get("lr_graph", 1e-3)))
    clip = float(tcfg.get("grad_clip", 5.0))
    pred_params = _predictor_params(model)

    def graph_step():
        for q in pred_params:
            q.requires_grad_(False)
        try:
            g = model.learn_graph(x)
            o = model(x, t_idx, A=g["A_sparse"])
            pred = bundle.normalizer.inverse(o["pred"])
            loss = huber_loss(pred, y, m, delta)
            loss = loss + model.hagl.regularisation(g["A_star"], lam_a, lam_c)
            opt_graph.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.hagl.parameters(), clip)
            opt_graph.step()
        finally:
            for q in pred_params:
                q.requires_grad_(True)

    try:
        out["graph_s_per_call"] = timeit(graph_step, max(3, n_iter // 4))
    except Exception as e:                                  # pragma: no cover
        LOG.warning("graph timing failed: %s", e)
        out["graph_s_per_call"] = float("nan")
    out["graph_steps"] = graph_steps
    out["graph_epoch"] = out["graph_s_per_call"] * graph_steps

    # -- macro and meso separately, so the branch costs are attributable ----
    hagl = model.hagl

    def macro_step():
        hagl.zero_grad(set_to_none=True)
        a = hagl.low(model.A_pre)
        if hagl.high is not None:
            a = a + hagl.high(xs, model.A_pre)
        a.sum().backward()

    out["macro_s_per_call"] = timeit(macro_step, max(3, n_iter // 4))

    if hagl.meso is not None:
        def meso_step():
            hagl.zero_grad(set_to_none=True)
            hagl.meso(model.H_s, model.H_g).sum().backward()
        out["meso_s_per_call"] = timeit(meso_step, max(3, n_iter // 4))
    else:
        out["meso_s_per_call"] = float("nan")

    # -- inference: graph frozen, no grad; this is the deployed path --------
    model.eval()
    with torch.no_grad():
        out["infer_s_per_batch"] = timeit(lambda: model(x, t_idx, A), n_iter)
    n_test = len(loaders["test"])
    out["infer_s_test_split"] = out["infer_s_per_batch"] * n_test
    out["test_batches"] = n_test

    # -- one epoch, from the measured per-batch costs -----------------------
    # An epoch is n_batches predictor steps plus, on the schedule the paper
    # uses, one graph-learner pass; reporting the sum makes the split visible.
    out["predictor_epoch"] = out["predictor_s_per_batch"] * n_batches
    out["train_s_per_epoch"] = out["predictor_epoch"] + out["graph_epoch"]
    # macro/meso are components of one graph-learner call, so their per-epoch
    # share follows the same multiplier
    out["macro_epoch"] = out["macro_s_per_call"] * graph_steps
    out["meso_epoch"] = out["meso_s_per_call"] * graph_steps
    out["peak_rss_gb"] = peak_rss_gb()

    # Microscopic branch: a one-off preprocessing cost.  It is rebuilt here from
    # the corpus that is already on disk and timed once -- rather than read from
    # a log or estimated -- so the number in the table is measured on the same
    # machine as the rest of the column.  The result is discarded; micro.npy is
    # not overwritten, because the runs were trained against the existing one.
    corpus = Path(processed) / name / "corpus.txt"
    out["micro_build_s"] = float("nan")
    if not skip_micro and corpus.exists():
        try:
            from hawformer.data.micro import build_micro_graph
            mc = cfg.get("micro", {})
            cells = _node_cells(cfg, bundle)
            t0 = time.perf_counter()
            build_micro_graph(
                cells, str(corpus), dim=int(mc.get("dim", 100)),
                window=int(mc.get("window", 5)),
                min_count=int(mc.get("min_count", 2)),
                epochs=int(mc.get("epochs", 5)), seed=int(cfg.get("seed", 42)),
                corpus_limit=mc.get("corpus_limit"))
            out["micro_build_s"] = time.perf_counter() - t0
        except Exception as e:
            LOG.warning("micro-branch timing skipped: %s", e)

    del model
    gc.collect()
    return out


def fmt(v: float, nd: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "---"
    return f"{v:.{nd}f}"


def emit_table(rows: List[Dict], out: Path) -> None:
    """Write the cost table.

    Units are chosen per row rather than per column: the graph branches cost
    milliseconds and the predictor costs tens of seconds, so one shared unit
    would print either "0.00" for the branches or a five-digit number for the
    predictor.  Each row therefore states its own unit.
    """
    disp = {"qingdao": "Qingdao", "tdrive": "T-Drive", "chengdu": "Chengdu",
            "porto": "Porto"}
    names = [disp.get(r["name"], r["name"].capitalize()) for r in rows]
    n = len(rows)
    L = [
        "% generated by scripts/measure_cost.py -- do not edit by hand",
        "\\begin{tabular}{@{}ll" + " c" * n + "}",
        "\\toprule",
        "Component & Unit & " + " & ".join(names) + "\\\\",
        "\\midrule",
    ]

    def line(label, unit, key, scale, digits):
        c = [label, unit]
        for r in rows:
            c.append(fmt(r.get(key, float("nan")) * scale, digits))
        return " & ".join(c) + "\\\\"

    # Graph branches run once per epoch on the graph-learner stage, not once
    # per batch, and none of them is on the inference path.
    L.append(line("Macro branch", "ms / epoch", "macro_epoch", 1000.0, 1))
    L.append(line("Meso branch", "ms / epoch", "meso_epoch", 1000.0, 1))
    L.append(line("Graph stage (total)", "s / epoch", "graph_epoch", 1.0, 2))
    L.append(line("Micro branch", "s, one-off", "micro_build_s", 1.0, 1))
    L.append(line("BWST predictor", "s / epoch", "predictor_epoch", 1.0, 1))
    L.append("\\midrule")
    L.append(line("HAWFormer, training", "s / epoch", "train_s_per_epoch", 1.0, 1))
    L.append(line("HAWFormer, inference", "ms / batch", "infer_s_per_batch",
                  1000.0, 1))
    L.append(line("Parameters", "thousand", "params_total", 1e-3, 1))
    L += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_cost.tex").write_text("\n".join(L) + "\n")
    LOG.info("wrote %s", out / "table_cost.tex")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--processed", default="processed")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--skip-micro", action="store_true",
                    help="do not re-time the skip-gram step (it is the slowest "
                         "part of this script)")
    a = ap.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    rows = []
    for c in a.configs:
        LOG.info("measuring %s", c)
        r = measure(c, a.runs, a.processed, a.iters, a.skip_micro)
        rows.append(r)
        LOG.info("  predictor %.3f s/batch x %d batches | graph %.3f s | "
                 "infer %.1f ms/batch | %.2f GB peak",
                 r["predictor_s_per_batch"], r["train_batches_per_epoch"],
                 r["graph_s_per_call"], r["infer_s_per_batch"] * 1000,
                 r["peak_rss_gb"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json({"rows": rows, "threads": torch.get_num_threads(),
               "cpu_count": os.cpu_count()}, str(out / "cost.json"))
    emit_table(rows, out)


if __name__ == "__main__":
    main()
