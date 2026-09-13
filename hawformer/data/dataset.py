"""Windowing, chronological splitting and normalisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils import Normalizer, get_logger
from .kalman import kalman_smooth

LOG = get_logger()


@dataclass
class Bundle:
    """Everything a run needs, in one object."""
    flow: np.ndarray          # [T, N] smoothed, original scale
    flow_raw: np.ndarray      # [T, N] before smoothing
    valid: np.ndarray         # [T, N] bool
    tod: np.ndarray           # [T]
    dow: np.ndarray           # [T]
    node_ids: np.ndarray      # [N]
    node_latlon: np.ndarray   # [N, 2]
    adj: np.ndarray           # [N, N]
    steps_per_day: int
    splits: Dict[str, Tuple[int, int]]
    normalizer: Normalizer
    epoch: Optional[np.ndarray] = None   # [T] unix seconds per bin, if recorded

    @property
    def num_nodes(self) -> int:
        return self.flow.shape[1]


def load_bundle(
    npz_path: str,
    split_ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    smooth: bool = True,
) -> Bundle:
    z = np.load(npz_path, allow_pickle=True)
    flow_raw = z["flow"].astype(np.float64)
    valid = z["valid"].astype(bool)

    # Bins with no coverage are gaps, not zeros: mark them NaN so the Kalman
    # smoother imputes rather than the model learning a spurious zero level.
    work = flow_raw.copy()
    work[~valid] = np.nan
    flow = kalman_smooth(work) if smooth else np.nan_to_num(work)

    T = flow.shape[0]
    a, b, _ = split_ratio
    i_tr, i_va = int(T * a), int(T * (a + b))
    splits = {"train": (0, i_tr), "val": (i_tr, i_va), "test": (i_va, T)}

    norm = Normalizer.fit(flow[:i_tr], mask=valid[:i_tr])
    LOG.info("splits train=%d val=%d test=%d | mean=%.3f std=%.3f",
             i_tr, i_va - i_tr, T - i_va, norm.mean, norm.std)

    return Bundle(
        flow=flow, flow_raw=flow_raw, valid=valid,
        tod=z["tod"], dow=z["dow"],
        node_ids=z["node_ids"], node_latlon=z["node_latlon"],
        adj=z["adj"].astype(np.float32),
        steps_per_day=int(z["steps_per_day"]),
        splits=splits, normalizer=norm,
        epoch=(z["epoch"].astype(np.int64) if "epoch" in z.files else None),
    )


class WindowDataset(Dataset):
    """Sliding windows of (history, horizon).

    Windows never straddle a split boundary: a window is assigned to the split
    containing its *last history step*, and windows whose horizon would cross
    into the next split are dropped.  Without this, test windows would carry
    training observations in their input and the evaluation would leak.
    """

    def __init__(
        self,
        bundle: Bundle,
        split: str,
        t_history: int = 12,
        t_horizon: int = 12,
        drop_invalid_targets: bool = True,
        stride: int = 1,
    ):
        self.b = bundle
        self.th, self.tp = t_history, t_horizon
        lo, hi = bundle.splits[split]
        # Striding thins the TRAINING windows only.  Consecutive windows overlap
        # in 11 of 12 history steps, so most of an epoch re-reads nearly
        # identical inputs; taking every k-th start keeps coverage of the whole
        # period at a fraction of the cost.  Validation and test are never
        # strided -- the evaluation must stay complete.
        step = max(1, int(stride)) if split == "train" else 1

        # A window must be CONTIGUOUS IN TIME, not merely contiguous in index.
        # Qingdao publishes 07:00-19:00 only, so consecutive rows jump from
        # 18:55 to 07:00 the next morning.  Indexing straight through that gap
        # builds windows whose "next hour" is fourteen hours away, which
        # inflates the error of every method and measures something other than
        # one-hour-ahead prediction.  Where per-bin timestamps are available we
        # require the whole span to be evenly spaced; where they are not
        # (synthetic data), every window is contiguous by construction.
        ep = bundle.epoch
        dt = None
        if ep is not None and len(ep) > 1:
            d = np.diff(ep.astype(np.int64))
            d = d[d > 0]
            dt = int(np.median(d)) if d.size else None

        starts, dropped_gap = [], 0
        for s in range(lo, hi, step):
            e = s + t_history
            f = e + t_horizon
            if f > hi:
                continue
            if dt is not None and ep is not None:
                span = ep[s:f].astype(np.int64)
                if span.size < 2 or int(span[-1] - span[0]) != dt * (f - s - 1):
                    dropped_gap += 1
                    continue
            if drop_invalid_targets and not bundle.valid[e:f].any():
                continue
            starts.append(s)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.x = bundle.normalizer.transform(bundle.flow).astype(np.float32)
        self.y = bundle.flow.astype(np.float32)
        if dropped_gap:
            LOG.info("%s split: %d windows (%d dropped for spanning a "
                     "recording gap)", split, len(self.starts), dropped_gap)
        else:
            LOG.info("%s split: %d windows", split, len(self.starts))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = int(self.starts[i])
        e, f = s + self.th, s + self.th + self.tp
        x = self.x[s:e]                            # [Th, N] normalised
        y = self.y[e:f]                            # [Tp, N] original scale
        m = self.b.valid[e:f]                      # [Tp, N]
        tod_in = self.b.tod[s:e]
        dow_in = self.b.dow[s:e]
        tod_out = self.b.tod[e:f]
        dow_out = self.b.dow[e:f]
        return (
            torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(-1),  # [Th,N,1]
            torch.from_numpy(np.ascontiguousarray(y)),                # [Tp,N]
            torch.from_numpy(np.ascontiguousarray(m)),                # [Tp,N]
            torch.from_numpy(np.stack([tod_in, dow_in], -1)).long(),  # [Th,2]
            torch.from_numpy(np.stack([tod_out, dow_out], -1)).long(),# [Tp,2]
        )


def make_loaders(
    bundle: Bundle,
    t_history: int = 12,
    t_horizon: int = 12,
    batch_size: int = 16,
    num_workers: int = 0,
    stride: int = 1,
) -> Dict[str, DataLoader]:
    out = {}
    for split in ("train", "val", "test"):
        ds = WindowDataset(bundle, split, t_history, t_horizon, stride=stride)
        out[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, drop_last=False, pin_memory=False,
        )
    return out
