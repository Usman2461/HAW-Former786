#!/usr/bin/env python3
"""Raw files -> processed/<name>/{data.npz, corpus.txt, micro.npy}.

    python scripts/prepare_data.py --config configs/chengdu.yaml
    python scripts/prepare_data.py --config configs/qingdao.yaml --skip-micro
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hawformer.data.build import build_dataset          # noqa: E402
from hawformer.data.micro import build_micro_graph      # noqa: E402
from hawformer.utils import get_logger, load_config     # noqa: E402

LOG = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="output dir (default processed/<name>)")
    ap.add_argument("--skip-micro", action="store_true",
                    help="skip skip-gram training; runs as HAWFormer-minus")
    ap.add_argument("--corpus-limit", type=int, default=None,
                    help="cap corpus sequences for a fast first pass")
    a = ap.parse_args()

    cfg = load_config(a.config)
    name = cfg.get("name", Path(a.config).stem)
    out_dir = Path(a.out or os.path.join("processed", name))

    npz = build_dataset(cfg, str(out_dir))
    z = np.load(npz, allow_pickle=True)
    node_ids = [str(x) for x in z["node_ids"]]
    N = len(node_ids)

    micro_path = out_dir / "micro.npy"
    corpus = out_dir / "corpus.txt"
    mc = cfg.get("micro", {})
    if a.skip_micro or not mc.get("enabled", True) or not corpus.exists() \
            or corpus.stat().st_size == 0:
        LOG.warning("microscopic graph skipped -> zeros; the model will run "
                    "as HAWFormer-minus unless you set model.use_micro: false")
        np.save(micro_path, np.zeros((N, N), dtype=np.float32))
    else:
        # In grid mode node ids ARE geohash cells, so they index the corpus
        # vocabulary directly.  In node_records mode they are sensor ids, and
        # each is mapped to the cell containing it.
        if cfg["raw"].get("mode") == "grid_from_trajectories" or \
                cfg["raw"].get("corpus_tokens") == "node_ids":
            # Tokens already ARE the node identifiers -- either GeoHash cells
            # (grid mode) or, for passage records, the node ids themselves.
            # No spatial mapping is needed or wanted.
            cells = node_ids
        else:
            from hawformer.data.geohash import encode
            latlon = z["node_latlon"]
            if not np.isfinite(latlon).all():
                LOG.error("node coordinates missing: cannot map sensors to grid "
                          "cells, so the microscopic branch is unavailable. "
                          "Provide raw.flow.node_meta with lat/lon columns.")
                np.save(micro_path, np.zeros((N, N), dtype=np.float32))
                LOG.info("wrote %s", micro_path)
                return
            p = int(cfg["raw"].get("geohash_precision", 7))
            cells = [encode(float(la), float(lo), p) for la, lo in latlon]

        A_mi = build_micro_graph(
            cells, str(corpus),
            dim=int(mc.get("dim", 100)), window=int(mc.get("window", 5)),
            min_count=int(mc.get("min_count", 2)), epochs=int(mc.get("epochs", 5)),
            seed=int(cfg.get("seed", 42)),
            corpus_limit=a.corpus_limit or mc.get("corpus_limit"),
        )
        np.save(micro_path, A_mi)
    LOG.info("wrote %s", micro_path)
    LOG.info("done. next: python scripts/train.py --config %s", a.config)


if __name__ == "__main__":
    main()
