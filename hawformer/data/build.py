"""Raw data -> standardised arrays.

The two target datasets ship in different shapes, and neither publishes a
schema that is stable across mirrors, so this module is deliberately
schema-driven: column names come from the YAML config rather than being
hard-coded.  Run `scripts/inspect_raw.py` first to see what your download
actually contains, then fill in the `columns:` blocks.

Two build modes:

  node_records
      Flow is published directly as records at named nodes (Qingdao's sensor
      passage records).  Trajectories are read separately and used only for the
      microscopic branch.

  grid_from_trajectories
      There are no node-level counts; nodes are grid cells and flow is obtained
      by aggregating the trajectories themselves (Chengdu, and any other raw
      GPS corpus -- T-Drive, Porto, TaxiBJ21).  Flow and transition structure
      then come from one source, so node definitions and the microscopic graph
      are mutually consistent by construction.

Output is a single .npz plus a plain-text trajectory corpus:

  flow        [T, N]  float32   aggregated flow, before smoothing
  valid       [T, N]  bool      False where the bin had no coverage
  epoch       [T]     int64     bin start, seconds
  tod         [T]     int64     time-of-day index
  dow         [T]     int64     day of week, 0 = Monday
  node_ids    [N]     str
  node_latlon [N, 2]  float64
  adj         [N, N]  float32   predefined adjacency (initialisation only)
  corpus.txt          one whitespace-separated token sequence per line
"""
from __future__ import annotations

import ast
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import geohash
from ..utils import get_logger

LOG = get_logger()


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def _parse_timestamps(s: pd.Series, fmt: str) -> pd.Series:
    if fmt == "epoch_s":
        return pd.to_datetime(s.astype("int64"), unit="s")
    if fmt == "epoch_ms":
        return pd.to_datetime(s.astype("int64"), unit="ms")
    if fmt in ("auto", "", None):
        return pd.to_datetime(s, errors="coerce")
    return pd.to_datetime(s, format=fmt, errors="coerce")


def _expand_paths(path: str) -> List[str]:
    if os.path.isdir(path):
        files = sorted(
            glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
            + glob.glob(os.path.join(path, "**", "*.txt"), recursive=True)
        )
        if not files:
            raise FileNotFoundError(f"no .csv/.txt files under {path}")
        return files
    files = sorted(glob.glob(path))
    if not files:
        raise FileNotFoundError(f"no files match {path}")
    return files


def load_trajectories(cfg: Dict) -> pd.DataFrame:
    """Return a DataFrame with columns [traj_id, ts, lat, lon], time-sorted."""
    fmt = cfg.get("format", "csv")
    path = cfg["path"]
    if fmt == "porto_polyline":
        return _load_porto(cfg)
    if fmt == "tdrive_dir":
        return _load_tdrive(cfg)
    return _load_generic_csv(cfg, _expand_paths(path))


def _load_generic_csv(cfg: Dict, files: Sequence[str]) -> pd.DataFrame:
    cols = cfg["columns"]
    header = cfg.get("header", "infer")
    names = cfg.get("names")            # for headerless files
    sep = cfg.get("sep", ",")
    usecols = [cols["traj_id"], cols["timestamp"], cols["lat"], cols["lon"]]
    frames = []
    for i, f in enumerate(files):
        df = pd.read_csv(f, sep=sep, header=header, names=names, low_memory=False)
        missing = [c for c in usecols if c not in df.columns]
        if missing:
            raise KeyError(
                f"{f}: columns {missing} not found. Present: {list(df.columns)}. "
                "Run scripts/inspect_raw.py and fix the `columns:` block."
            )
        df = df[usecols]
        df.columns = ["traj_id", "ts", "lat", "lon"]
        frames.append(df)
        if (i + 1) % 200 == 0:
            LOG.info("read %d/%d trajectory files", i + 1, len(files))
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = _parse_timestamps(df["ts"], cfg.get("timestamp_format", "auto"))
    df = df.dropna(subset=["ts", "lat", "lon"])
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    return df.sort_values(["traj_id", "ts"], kind="mergesort").reset_index(drop=True)


def _load_tdrive(cfg: Dict) -> pd.DataFrame:
    """T-Drive: one file per taxi, lines `id,datetime,longitude,latitude`."""
    files = _expand_paths(cfg["path"])
    frames = []
    for i, f in enumerate(files):
        try:
            df = pd.read_csv(f, header=None, names=["traj_id", "ts", "lon", "lat"])
        except Exception:
            continue
        frames.append(df)
        if (i + 1) % 500 == 0:
            LOG.info("read %d/%d T-Drive files", i + 1, len(files))
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "lat", "lon"])
    return df[["traj_id", "ts", "lat", "lon"]].sort_values(
        ["traj_id", "ts"], kind="mergesort").reset_index(drop=True)


def _load_porto(cfg: Dict) -> pd.DataFrame:
    """Porto / ECML-PKDD 2015: POLYLINE is a JSON list of [lon, lat] at 15 s."""
    cols = cfg.get("columns", {})
    id_col = cols.get("traj_id", "TRIP_ID")
    ts_col = cols.get("timestamp", "TIMESTAMP")
    poly_col = cols.get("polyline", "POLYLINE")
    step = float(cfg.get("polyline_step_seconds", 15.0))
    rows = []
    for f in _expand_paths(cfg["path"]):
        df = pd.read_csv(f, usecols=[id_col, ts_col, poly_col], low_memory=False)
        for tid, t0, poly in df.itertuples(index=False):
            if not isinstance(poly, str) or len(poly) < 5:
                continue
            try:
                pts = json.loads(poly)
            except Exception:
                try:
                    pts = ast.literal_eval(poly)
                except Exception:
                    continue
            for k, p in enumerate(pts):
                rows.append((tid, float(t0) + k * step, p[1], p[0]))
    df = pd.DataFrame(rows, columns=["traj_id", "ts", "lat", "lon"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    return df.sort_values(["traj_id", "ts"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Spatial filtering and gridding
# ---------------------------------------------------------------------------

def apply_bbox(df: pd.DataFrame, bbox: Optional[Sequence[float]]) -> pd.DataFrame:
    """bbox = [lat_min, lon_min, lat_max, lon_max]."""
    if not bbox:
        return df
    la0, lo0, la1, lo1 = bbox
    m = (df.lat.between(la0, la1)) & (df.lon.between(lo0, lo1))
    LOG.info("bbox keeps %.1f%% of points", 100.0 * m.mean())
    return df[m].reset_index(drop=True)


def add_cells(df: pd.DataFrame, precision: int) -> pd.DataFrame:
    """Attach a GeoHash cell token to every point.

    Encoding is vectorised over the unique rounded coordinate pairs rather than
    per row, which matters: raw corpora have tens of millions of points but far
    fewer distinct positions at the resolution we care about.
    """
    # Quantise to ~0.1 m before de-duplicating: raw corpora have tens of
    # millions of points but far fewer distinct positions at the resolution a
    # GeoHash cell cares about, so this turns a per-row Python loop into one
    # encode per distinct location.
    q = 10 ** 6
    key = (np.round(df.lat.values * q).astype(np.int64) << np.int64(32)) ^ \
        np.round(df.lon.values * q).astype(np.int64)
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    lats = df.lat.values[first]
    lons = df.lon.values[first]
    LOG.info("geohashing %d distinct positions (from %d points)",
             len(first), len(df))
    cells = np.array([geohash.encode(a, o, precision) for a, o in zip(lats, lons)])
    df = df.copy()
    df["cell"] = cells[inv]
    return df


# ---------------------------------------------------------------------------
# Node selection
# ---------------------------------------------------------------------------

def select_nodes(
    df: pd.DataFrame,
    cfg: Dict,
    precision: int,
) -> Tuple[List[str], np.ndarray]:
    """Return (node cell tokens, [N,2] lat/lon centres).

    Two strategies:

    `poi_file` -- a CSV of points of interest (subway stations for Chengdu);
        each is mapped to the cell containing it and duplicates are merged.
        This is the strategy the paper describes: citywide grids are numerous
        and mostly too sparse to be representative, so anchoring nodes to
        transport hotspots keeps the node set meaningful.

    `busiest`  -- the top-K cells by point count.  Use when no POI list is
        available; it is a weaker definition of "node" and should be reported
        as such.
    """
    source = cfg.get("source", "busiest")
    if source == "poi_file":
        poi = pd.read_csv(cfg["path"])
        latc = cfg.get("lat_column", "lat")
        lonc = cfg.get("lon_column", "lon")
        cells = [geohash.encode(float(a), float(o), precision)
                 for a, o in zip(poi[latc], poi[lonc])]
        seen, keep = set(), []
        for c in cells:
            if c not in seen:
                seen.add(c)
                keep.append(c)
        present = set(df.cell.unique())
        missing = [c for c in keep if c not in present]
        if missing:
            LOG.warning("%d/%d POI cells have no trajectory coverage; dropped",
                        len(missing), len(keep))
        keep = [c for c in keep if c in present]
    elif source == "node_file":
        nf = pd.read_csv(cfg["path"])
        latc = cfg.get("lat_column", "lat")
        lonc = cfg.get("lon_column", "lon")
        keep = [geohash.encode(float(a), float(o), precision)
                for a, o in zip(nf[latc], nf[lonc])]
    else:
        k = int(cfg.get("num_nodes", 150))
        counts = df.cell.value_counts()
        keep = list(counts.index[:k])
    centres = np.array([geohash.decode(c) for c in keep], dtype=np.float64)
    LOG.info("selected %d nodes (precision %d)", len(keep), precision)
    return keep, centres


# ---------------------------------------------------------------------------
# Flow aggregation
# ---------------------------------------------------------------------------

def aggregate_flow_from_trajectories(
    df: pd.DataFrame,
    nodes: Sequence[str],
    freq: str = "5min",
    daily_window: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Count distinct vehicles entering each node cell per time bin.

    Counting *distinct trajectories* rather than raw points is important: GPS
    sampling rates differ between vehicles and between corpora, so a raw point
    count measures the sampling process as much as the traffic.
    """
    node_index = {c: i for i, c in enumerate(nodes)}
    d = df[df.cell.isin(node_index)].copy()
    d["bin"] = d.ts.dt.floor(freq)
    d["node"] = d.cell.map(node_index)
    g = d.groupby(["bin", "node"])["traj_id"].nunique()

    bins = pd.date_range(df.ts.min().floor(freq), df.ts.max().floor(freq), freq=freq)
    if daily_window is not None:
        h0, h1 = daily_window
        bins = bins[(bins.hour >= h0) & (bins.hour < h1)]
    bin_index = {b: i for i, b in enumerate(bins)}

    flow = np.zeros((len(bins), len(nodes)), dtype=np.float32)
    for (b, n), v in g.items():
        i = bin_index.get(b)
        if i is not None:
            flow[i, n] = v

    # A bin is "covered" if the corpus contains any point in it at all; bins
    # with no coverage anywhere are data gaps, not genuine zero flow.
    covered_bins = set(d["bin"].unique())
    valid = np.zeros_like(flow, dtype=bool)
    for b, i in bin_index.items():
        if b in covered_bins:
            valid[i, :] = True
    LOG.info("aggregated flow %s, %.1f%% of bins covered",
             flow.shape, 100.0 * valid.any(axis=1).mean())
    return flow, valid, bins


def load_node_records(
    cfg: Dict,
    freq: str = "5min",
    daily_window: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str], np.ndarray]:
    """Load flow published directly as node records or as a wide matrix."""
    layout = cfg.get("layout", "long")
    if layout == "wide":
        df = pd.read_csv(cfg["path"], index_col=0)
        df.index = _parse_timestamps(pd.Series(df.index),
                                     cfg.get("timestamp_format", "auto")).values
        df = df.sort_index()
        node_ids = [str(c) for c in df.columns]
        flow = df.values.astype(np.float32)
        bins = pd.DatetimeIndex(df.index)
        valid = np.isfinite(flow)
    else:
        cols = cfg["columns"]
        df = pd.read_csv(cfg["path"], low_memory=False)
        df = df[[cols["node_id"], cols["timestamp"], cols["count"]]]
        df.columns = ["node", "ts", "count"]
        df["ts"] = _parse_timestamps(df["ts"], cfg.get("timestamp_format", "auto"))
        df = df.dropna(subset=["ts"])
        df["bin"] = df.ts.dt.floor(freq)
        piv = df.pivot_table(index="bin", columns="node", values="count",
                             aggfunc="sum")
        piv = piv.sort_index()
        node_ids = [str(c) for c in piv.columns]
        flow = piv.values.astype(np.float32)
        bins = pd.DatetimeIndex(piv.index)
        valid = np.isfinite(flow)

    if daily_window is not None:
        h0, h1 = daily_window
        keep = (bins.hour >= h0) & (bins.hour < h1)
        flow, valid, bins = flow[keep], valid[keep], bins[keep]

    latlon = np.full((len(node_ids), 2), np.nan)
    meta_path = cfg.get("node_meta")
    if meta_path:
        meta = pd.read_csv(meta_path)
        idc = cfg.get("node_id_column", "node_id")
        latc = cfg.get("lat_column", "lat")
        lonc = cfg.get("lon_column", "lon")
        m = {str(r[idc]): (float(r[latc]), float(r[lonc])) for _, r in meta.iterrows()}
        for i, nid in enumerate(node_ids):
            if nid in m:
                latlon[i] = m[nid]
    return flow, valid, bins, node_ids, latlon


# ---------------------------------------------------------------------------
# Predefined adjacency
# ---------------------------------------------------------------------------

def build_adjacency(
    latlon: np.ndarray,
    sigma: Optional[float] = None,
    epsilon: float = 0.1,
    adj_file: Optional[str] = None,
    n_nodes: Optional[int] = None,
    node_ids: Optional[Sequence[str]] = None,
    edge_list: Optional[Dict] = None,
) -> np.ndarray:
    """Gaussian-kernel adjacency on great-circle distance, thresholded.

    This matrix is an *initialisation only*: HAGL treats it as a starting point
    and the low-order branch is explicitly masked to its support, while the
    high-order and microscopic branches are free to discover relations it does
    not contain.
    """
    if edge_list and node_ids is not None:
        # A published road network: rows are directed links between nodes.  This
        # is the honest "predefined adjacency" -- connectivity, which the paper
        # argues is a poor proxy for dependency and which HAGL is meant to
        # improve on.  It is kept directed; the model symmetrises where it needs
        # to (partitioning, Laplacian PE) and uses both directions in ChebConv.
        df = pd.read_csv(edge_list["path"])
        fc = edge_list.get("from_column", df.columns[0])
        tc = edge_list.get("to_column", df.columns[1])
        idx = {str(n): i for i, n in enumerate(node_ids)}
        N = len(node_ids)
        a = np.zeros((N, N), dtype=np.float32)
        pairs = list(zip(df[fc].astype(str), df[tc].astype(str)))
        hit = 0
        for u, v in pairs:
            i, j = idx.get(u), idx.get(v)
            if i is not None and j is not None:
                a[i, j] = 1.0
                hit += 1

        # Monitoring covers only a subset of intersections, so most published
        # links have at least one endpoint outside the node set and a direct
        # match leaves an implausibly sparse graph.  The physical connection
        # still exists -- it just runs through an unmonitored intersection --
        # so we contract paths that leave the node set and immediately return
        # to it, up to `max_hops` intermediate nodes.  Without this the
        # predefined adjacency would understate the road network rather than
        # merely being an imperfect proxy for dependency.
        hops = int(edge_list.get("max_hops", 0))
        if hops > 0:
            from collections import defaultdict, deque
            out = defaultdict(list)
            for u, v in pairs:
                out[u].append(v)
            added = 0
            for u, i in idx.items():
                seen = {u}
                q = deque((w, 1) for w in out.get(u, ()))
                while q:
                    w, d = q.popleft()
                    if w in seen:
                        continue
                    seen.add(w)
                    j = idx.get(w)
                    if j is not None:
                        if a[i, j] == 0.0:
                            a[i, j] = 1.0
                            added += 1
                        continue          # stop at the first monitored node
                    if d < hops:
                        q.extend((x, d + 1) for x in out.get(w, ()))
            LOG.info("contracted %d additional links through unmonitored "
                     "intersections (<= %d hops)", added, hops)

        np.fill_diagonal(a, 1.0)
        LOG.info("adjacency from edge list: %d/%d direct links inside the node "
                 "set, final density %.3f, mean out-degree %.1f",
                 hit, len(df), (a > 0).mean(), (a > 0).sum(1).mean() - 1)
        if (a > 0).sum() <= N:
            LOG.warning("adjacency is essentially empty -- check that the id "
                        "columns of %s use the same ids as the flow file",
                        edge_list["path"])
        return a

    if adj_file:
        a = np.load(adj_file) if adj_file.endswith(".npy") else \
            pd.read_csv(adj_file, header=None).values
        return a.astype(np.float32)

    if latlon is None or not np.isfinite(latlon).all():
        n = n_nodes or (0 if latlon is None else latlon.shape[0])
        LOG.warning("no usable coordinates; falling back to identity adjacency")
        return np.eye(n, dtype=np.float32)

    lat = latlon[:, 0][:, None]
    lon = latlon[:, 1][:, None]
    d = geohash.haversine(lat, lon, lat.T, lon.T)
    off = d[~np.eye(d.shape[0], dtype=bool)]
    if sigma is None:
        sigma = float(off.std()) if off.size else 1.0
    a = np.exp(-(d ** 2) / (sigma ** 2 + 1e-12))
    a[a < epsilon] = 0.0
    np.fill_diagonal(a, 1.0)
    LOG.info("adjacency density %.3f (sigma=%.1f m, eps=%.2f)",
             (a > 0).mean(), sigma, epsilon)
    return a.astype(np.float32)


# ---------------------------------------------------------------------------
# Trajectory corpus for the microscopic branch
# ---------------------------------------------------------------------------

def build_corpus(
    df: pd.DataFrame,
    out_path: str,
    min_len: int = 3,
    max_gap_minutes: float = 30.0,
) -> int:
    """Write one token sequence per trajectory segment.

    Consecutive repeats are collapsed so that a sequence records *movement*
    rather than dwell time -- a taxi idling in one cell for twenty minutes
    would otherwise dominate the skip-gram context windows.  A trajectory is
    also split wherever the time gap exceeds `max_gap_minutes`, since the two
    sides of a long gap are not evidence of a transition.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    gap = pd.Timedelta(minutes=max_gap_minutes)
    with open(out_path, "w") as fh:
        for _, grp in df.groupby("traj_id", sort=False):
            cells = grp.cell.values
            times = grp.ts.values
            seg: List[str] = []
            prev_t = None
            for c, t in zip(cells, times):
                if prev_t is not None and (t - prev_t) > gap.to_timedelta64():
                    if len(seg) >= min_len:
                        fh.write(" ".join(seg) + "\n")
                        n_written += 1
                    seg = []
                if not seg or seg[-1] != c:
                    seg.append(c)
                prev_t = t
            if len(seg) >= min_len:
                fh.write(" ".join(seg) + "\n")
                n_written += 1
    LOG.info("wrote %d trajectory token sequences to %s", n_written, out_path)
    return n_written


# ---------------------------------------------------------------------------
# Top-level build
# ---------------------------------------------------------------------------

def build_dataset(cfg: Dict, out_dir: str) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = cfg["raw"]
    freq = cfg.get("freq", "5min")
    precision = int(raw.get("geohash_precision", 7))
    dw = raw.get("daily_window")
    dw = tuple(dw) if dw else None

    traj = None
    if "trajectory" in raw and raw["trajectory"].get("path"):
        LOG.info("loading trajectories ...")
        traj = load_trajectories(raw["trajectory"])
        traj = apply_bbox(traj, raw.get("bbox"))
        LOG.info("%d trajectory points, %d distinct vehicles",
                 len(traj), traj.traj_id.nunique())
        traj = add_cells(traj, precision)

    mode = raw.get("mode", "grid_from_trajectories")
    if mode == "grid_from_trajectories":
        if traj is None:
            raise ValueError("grid_from_trajectories requires raw.trajectory")
        nodes, latlon = select_nodes(traj, raw.get("nodes", {}), precision)
        flow, valid, bins = aggregate_flow_from_trajectories(traj, nodes, freq, dw)
        node_ids = list(nodes)
    elif mode == "node_records":
        flow, valid, bins, node_ids, latlon = load_node_records(
            raw["flow"], freq, dw)
        LOG.info("node records: flow %s over %d nodes", flow.shape, len(node_ids))
    else:
        raise ValueError(f"unknown raw.mode {mode!r}")

    adj = build_adjacency(
        latlon,
        sigma=raw.get("adj_sigma"),
        epsilon=float(raw.get("adj_epsilon", 0.1)),
        adj_file=raw.get("adj_file"),
        n_nodes=len(node_ids),
        node_ids=node_ids,
        edge_list=raw.get("edge_list"),
    )

    corpus_path = str(out_dir / "corpus.txt")
    pre = raw.get("corpus_file")
    if pre:
        # A corpus prepared upstream (e.g. aggregated on the machine that holds
        # the raw data).  Accepted plain or gzipped.
        import gzip, shutil
        op = gzip.open if str(pre).endswith(".gz") else open
        with op(pre, "rb") as fi, open(corpus_path, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        n = sum(1 for _ in open(corpus_path))
        LOG.info("using prebuilt corpus %s (%d sequences)", pre, n)
    elif traj is not None:
        build_corpus(traj, corpus_path,
                     min_len=int(raw.get("corpus_min_len", 3)),
                     max_gap_minutes=float(raw.get("corpus_max_gap_minutes", 30)))
    else:
        LOG.warning("no trajectories supplied; the microscopic branch will be "
                    "disabled and the model runs as HAWFormer-minus")
        open(corpus_path, "w").close()

    steps_per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))
    tod = (bins.hour * 60 + bins.minute) // int(pd.Timedelta(freq).seconds // 60)
    out_npz = str(out_dir / "data.npz")
    np.savez_compressed(
        out_npz,
        flow=flow.astype(np.float32),
        valid=valid,
        # pandas 2.x keeps second-resolution datetimes as datetime64[s], so
        # the old ns-based `// 10**9` collapsed every timestamp to 1.  Convert
        # through an explicit second resolution instead of assuming ns.
        epoch=pd.DatetimeIndex(bins).as_unit("s").astype("int64").values,
        tod=np.asarray(tod, dtype=np.int64),
        dow=np.asarray(bins.dayofweek, dtype=np.int64),
        node_ids=np.array([str(x) for x in node_ids]),
        node_latlon=latlon.astype(np.float64),
        adj=adj,
        steps_per_day=np.array(steps_per_day),
    )
    LOG.info("wrote %s  flow=%s  nodes=%d", out_npz, flow.shape, len(node_ids))
    return out_npz
