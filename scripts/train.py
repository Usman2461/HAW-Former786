#!/usr/bin/env python3
"""Train and evaluate HAWFormer.

    python scripts/train.py --config configs/chengdu.yaml
    python scripts/train.py --config configs/qingdao.yaml --seed 1 --tag run1
    python scripts/train.py --config configs/chengdu.yaml \
        --set model.gamma=1.0 --tag gamma1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hawformer.data.dataset import load_bundle, make_loaders, WindowDataset  # noqa: E402
from hawformer.evaluate import format_report, full_report                    # noqa: E402
from hawformer.models.hawformer import HAWFormer, ModelConfig                # noqa: E402
from hawformer.models.hierarchy import (build_hierarchy, semantic_descriptor,  # noqa: E402
                                        suggest_P)
from hawformer.train import TrainConfig, evaluate, train                     # noqa: E402
from hawformer.utils import (configure_backends, count_parameters,          # noqa: E402
                             get_logger, load_config, pick_device, save_json,
                             set_seed)

LOG = get_logger()


def apply_sets(cfg: dict, sets: list) -> dict:
    """--set a.b=value overrides, parsed as JSON when possible."""
    for s in sets or []:
        key, _, val = s.partition("=")
        try:
            v = json.loads(val)
        except json.JSONDecodeError:
            v = val
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return cfg


def build_model(cfg: dict, bundle, npz_dir: Path, device) -> HAWFormer:
    mc = cfg["model"]
    dc = cfg["data"]

    # Semantic group count: silhouette initialisation when not fixed.
    tr0, tr1 = bundle.splits["train"]
    flow_tr = bundle.flow[tr0:tr1]
    P = mc.get("P")
    if P in (None, "null", "auto"):
        P = suggest_P(semantic_descriptor(flow_tr))
    P = int(P)

    model_cfg = ModelConfig(
        num_nodes=bundle.num_nodes,
        t_history=int(dc["t_history"]), t_horizon=int(dc["t_horizon"]),
        d_model=int(mc["d_model"]), n_layers=int(mc["n_layers"]),
        n_heads=int(mc["n_heads"]), sim_heads=int(mc["sim_heads"]),
        wavelet_levels=int(mc["wavelet_levels"]), cheb_k=int(mc["cheb_k"]),
        dropout=float(mc["dropout"]), time_dim=int(mc["time_dim"]),
        steps_per_day=int(bundle.steps_per_day), lap_pe_dim=int(mc["lap_pe_dim"]),
        kappa0=int(mc["kappa0"]), gamma=float(mc["gamma"]),
        kappa_min=int(mc["kappa_min"]), geo_topk=int(mc["geo_topk"]),
        topk=int(mc["topk"]), low_dim=int(mc["low_dim"]),
        high_dim=int(mc["high_dim"]), P=P, Q=int(mc["Q"]),
        use_micro=bool(mc["use_micro"]), use_meso=bool(mc["use_meso"]),
        use_high=bool(mc["use_high"]), use_gcn=bool(mc["use_gcn"]),
        use_dla=bool(mc["use_dla"]), use_wavelet=bool(mc["use_wavelet"]),
        identity_affinity=bool(mc["identity_affinity"]),
        soft_membership=bool(mc["soft_membership"]),
        use_spectral_descriptor=bool(mc["use_spectral_descriptor"]),
        static_hierarchy=bool(mc["static_hierarchy"]),
        membership_temperature=float(mc.get("membership_temperature", 1.0)),
        reg_covar=float(mc.get("reg_covar", 1e-2)),
        residual_from_last=bool(mc.get("residual_from_last", False)),
        grad_checkpoint=bool(mc.get("grad_checkpoint", False)),
        relap_on_freeze=bool(mc.get("relap_on_freeze", True)),
    )
    model = HAWFormer(model_cfg).to(device)

    A_mi = None
    mp = npz_dir / "micro.npy"
    if mp.exists():
        A_mi = np.load(mp)
        if not np.isfinite(A_mi).all() or np.abs(A_mi).sum() < 1e-9:
            LOG.warning("micro.npy is empty; running without movement evidence")
            A_mi = None
    if A_mi is None and model_cfg.use_micro:
        LOG.warning("no microscopic graph available -> forcing use_micro=false "
                    "(this is the HAWFormer-minus configuration)")
        model.cfg.use_micro = False
        model.hagl.use_micro = False

    model.set_static_inputs(bundle.adj, A_mi, bundle.node_latlon)

    # Initial hierarchy: on the predefined graph when static, else it will be
    # refreshed onto the learned graph on the first refresh tick.
    h = build_hierarchy(
        flow_tr, bundle.adj, P=P, Q=int(mc["Q"]),
        wavelet_levels=model_cfg.wavelet_levels,
        use_spectral_descriptor=model_cfg.use_spectral_descriptor,
        soft_membership=model_cfg.soft_membership, topk=model_cfg.topk,
        seed=int(cfg.get("seed", 42)),
        membership_temperature=model_cfg.membership_temperature,
        reg_covar=model_cfg.reg_covar,
    )
    model.set_hierarchy(h.H_s, h.H_g)
    model.to(device)
    LOG.info("model built: N=%d P=%d Q=%d bands=%d budgets=%s params=%d",
             bundle.num_nodes, h.P, h.Q, model.num_bands, model.budgets,
             count_parameters(model))
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default=None, help="processed dir (default processed/<name>)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--set", action="append", default=[],
                    help="dotted override, e.g. --set model.gamma=1.0")
    a = ap.parse_args()

    cfg = apply_sets(load_config(a.config), a.set)
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.epochs is not None:
        cfg["train"]["epochs"] = a.epochs
    name = cfg.get("name", Path(a.config).stem)
    npz_dir = Path(a.data or os.path.join("processed", name))
    out_dir = Path(a.out or os.path.join("runs", name, f"{a.tag}_seed{cfg['seed']}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg["seed"]))
    configure_backends()
    device = pick_device(cfg.get("device", "auto"))
    LOG.info("device: %s | amp %s | grad_checkpoint %s | batch %s", device,
             cfg["train"].get("amp", "off"),
             cfg["model"].get("grad_checkpoint", False),
             cfg["data"]["batch_size"])

    bundle = load_bundle(str(npz_dir / "data.npz"),
                         tuple(cfg["split_ratio"]), bool(cfg["smooth"]))
    loaders = make_loaders(bundle, int(cfg["data"]["t_history"]),
                           int(cfg["data"]["t_horizon"]),
                           int(cfg["data"]["batch_size"]),
                           int(cfg["data"]["num_workers"]),
                           int(cfg["data"].get("train_stride", 1)))

    model = build_model(cfg, bundle, npz_dir, device)
    tcfg = TrainConfig(**{k: v for k, v in cfg["train"].items()
                          if k in TrainConfig.__dataclass_fields__})
    res = train(model, loaders, bundle, tcfg, device, str(out_dir),
                hierarchy_kwargs={"seed": int(cfg["seed"])})

    ev = cfg.get("eval", {})
    metrics, arrays = evaluate(model, loaders["test"], bundle, device,
                               float(ev.get("mape_threshold", 5.0)),
                               return_arrays=True)
    test_ds: WindowDataset = loaders["test"].dataset
    rep = full_report(arrays["true"], arrays["pred"], arrays["mask"], bundle,
                      test_ds.starts, int(cfg["data"]["t_history"]),
                      horizons=tuple(ev.get("horizons", (3, 6, 9, 12))),
                      mape_threshold=float(ev.get("mape_threshold", 5.0)))
    print()
    print(format_report(rep, f"{name} / {a.tag} / seed {cfg['seed']}"))

    save_json({"config": cfg, "best": res["best"], "test": rep},
              str(out_dir / "report.json"))
    np.savez_compressed(out_dir / "predictions.npz", **arrays)
    LOG.info("wrote %s", out_dir / "report.json")


if __name__ == "__main__":
    main()
