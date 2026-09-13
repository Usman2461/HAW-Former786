"""Evaluation metrics.

Metric choice is driven by a property of trajectory-derived flow: aggregating
GPS traces into five-minute windows at individual nodes produces many small
integers and some exact zeros.  MAPE is undefined at zero and explodes just
above it, so a suite built around MAPE alone is dominated by the least
informative observations in the test set.

We therefore report:

  MAE   -- primary, in vehicles / 5 min; what an operator reads directly.
  RMSE  -- penalises large deviations quadratically, so it is the metric most
           sensitive to incident-driven excursions.  A model that improves MAE
           while worsening RMSE is smoothing away the events that matter.
  WAPE  -- primary scale-free metric: sum |err| / sum |y|.  Well defined with
           zeros present, and comparable across networks of different volume.
  MAPE  -- retained for comparability with the literature, but computed only
           over observations above `mape_threshold`.  `mape_coverage` reports
           what fraction of observations survived that filter, so the number
           can be interpreted rather than quoted blind.
  R2    -- goodness of fit, exposing systematic bias rather than pure variance.

All functions take arrays in the ORIGINAL flow scale (inverse-normalised) with
shape [..., T_p, N] and an optional boolean validity mask of the same shape.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _prep(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray]):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if mask is None:
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true, y_pred, mask


def mae(y_true, y_pred, mask=None) -> float:
    y_true, y_pred, mask = _prep(y_true, y_pred, mask)
    if mask.sum() == 0:
        return float("nan")
    return float(np.abs(y_true[mask] - y_pred[mask]).mean())


def rmse(y_true, y_pred, mask=None) -> float:
    y_true, y_pred, mask = _prep(y_true, y_pred, mask)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(((y_true[mask] - y_pred[mask]) ** 2).mean()))


def wape(y_true, y_pred, mask=None) -> float:
    """Weighted absolute percentage error, in percent."""
    y_true, y_pred, mask = _prep(y_true, y_pred, mask)
    denom = np.abs(y_true[mask]).sum()
    if denom < 1e-9:
        return float("nan")
    return float(np.abs(y_true[mask] - y_pred[mask]).sum() / denom * 100.0)


def mape(y_true, y_pred, mask=None, threshold: float = 5.0) -> float:
    """MAPE in percent, over observations with |y| > threshold only."""
    y_true, y_pred, mask = _prep(y_true, y_pred, mask)
    sel = mask & (np.abs(y_true) > threshold)
    if sel.sum() == 0:
        return float("nan")
    return float((np.abs(y_true[sel] - y_pred[sel]) / np.abs(y_true[sel])).mean() * 100.0)


def mape_coverage(y_true, mask=None, threshold: float = 5.0) -> float:
    """Fraction of valid observations that the MAPE threshold retains."""
    y_true = np.asarray(y_true, dtype=np.float64)
    if mask is None:
        mask = np.isfinite(y_true)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(y_true)
    if mask.sum() == 0:
        return float("nan")
    return float((mask & (np.abs(y_true) > threshold)).sum() / mask.sum())


def r2(y_true, y_pred, mask=None) -> float:
    y_true, y_pred, mask = _prep(y_true, y_pred, mask)
    if mask.sum() < 2:
        return float("nan")
    yt, yp = y_true[mask], y_pred[mask]
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def bias(y_true, y_pred, mask=None) -> float:
    """Mean signed error; negative means systematic under-prediction."""
    y_true, y_pred, mask = _prep(y_true, y_pred, mask)
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] - y_true[mask]).mean())


def all_metrics(y_true, y_pred, mask=None, mape_threshold: float = 5.0) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred, mask),
        "RMSE": rmse(y_true, y_pred, mask),
        "WAPE": wape(y_true, y_pred, mask),
        "MAPE": mape(y_true, y_pred, mask, mape_threshold),
        "MAPE_coverage": mape_coverage(y_true, mask, mape_threshold),
        "R2": r2(y_true, y_pred, mask),
        "Bias": bias(y_true, y_pred, mask),
    }


def horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
    steps: Optional[list] = None,
    mape_threshold: float = 5.0,
) -> Dict[str, Dict[str, float]]:
    """Metrics at individual prediction steps.

    y_true / y_pred: [B, T_p, N].  `steps` are 1-indexed horizons; default all.
    """
    T_p = y_true.shape[1]
    steps = steps or list(range(1, T_p + 1))
    out = {}
    for s in steps:
        i = s - 1
        m = None if mask is None else mask[:, i]
        out[f"step_{s}"] = all_metrics(y_true[:, i], y_pred[:, i], m, mape_threshold)
    return out


# ---------------------------------------------------------------------------
# Condition-conditioned evaluation (Section VII-F of the paper)
# ---------------------------------------------------------------------------

def congestion_mask(
    flow_window: np.ndarray,
    speed_window: Optional[np.ndarray] = None,
    speed_threshold: Optional[float] = None,
    flow_quantile: float = 0.75,
) -> np.ndarray:
    """Boolean mask selecting congested samples.

    If per-node speed is available, congestion is speed < `speed_threshold`.
    Otherwise we fall back to a flow-based proxy: a node-time is congested when
    its flow exceeds the per-node `flow_quantile` of the training distribution.
    The proxy is coarser and should be reported as such.
    """
    if speed_window is not None and speed_threshold is not None:
        return speed_window < speed_threshold
    thr = np.quantile(flow_window, flow_quantile, axis=0, keepdims=True)
    return flow_window > thr


def incident_flags(
    flow: np.ndarray,
    percentile: float = 95.0,
    horizon_steps: int = 12,
) -> np.ndarray:
    """Flag incident-like events on a [T, N] flow series.

    An event is two consecutive step-to-step changes whose absolute magnitude
    both exceed the empirical `percentile` of |delta| for that node.  Returns a
    boolean [T, N] array that is True for the `horizon_steps` steps following
    each flagged time, i.e. the window over which post-event error is measured.
    """
    # The raw series carries NaN wherever a bin had no coverage.  np.percentile
    # propagates that to NaN and every comparison then returns False, which
    # silently reports "no incidents" rather than failing -- exactly the kind of
    # empty result that looks like a finding.  Use nan-aware statistics and
    # treat gaps as non-events.
    d = np.abs(np.diff(flow, axis=0))                      # [T-1, N]
    with np.errstate(invalid="ignore"):
        thr = np.nanpercentile(np.where(np.isfinite(d), d, np.nan),
                               percentile, axis=0, keepdims=True)
    thr = np.where(np.isfinite(thr), thr, np.inf)
    hot = np.isfinite(d) & (d > thr)                       # [T-1, N]
    both = hot[:-1] & hot[1:]                              # [T-2, N]
    T, N = flow.shape
    out = np.zeros((T, N), dtype=bool)
    idx = np.argwhere(both)
    for t, n in idx:
        t0 = t + 2
        out[t0:t0 + horizon_steps, n] = True
    return out


# ---------------------------------------------------------------------------
# Adap-STWT's evaluation protocol
# ---------------------------------------------------------------------------
# Their metrics are not the same functions as ours, and the difference is not
# cosmetic: MAE and RMSE are taken over ground-truth entries that are NON-ZERO,
# and MAPE over |y| > 0.5 (utils/tools.py:85-99).  Because their preprocessing
# fills absent crossroad-slices with 0, that filter removes both the genuine
# zeros and the missing data from the score.  Reproducing their numbers means
# reproducing this, so it lives here as an explicit, named protocol rather than
# as a silent flag on our own metrics.

def adapstwt_metrics(y_true, y_pred, mape_floor: float = 0.5) -> Dict[str, float]:
    """MAE / RMSE over non-zero targets, MAPE over |y| > 0.5, as they compute them."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    nz = (y_true != 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    mp = (np.abs(y_true) > mape_floor) & np.isfinite(y_true) & np.isfinite(y_pred)
    out = {
        "MAE": float(np.abs(y_true[nz] - y_pred[nz]).mean()) if nz.any() else float("nan"),
        "RMSE": float(np.sqrt(((y_true[nz] - y_pred[nz]) ** 2).mean())) if nz.any() else float("nan"),
        "MAPE": float(np.abs((y_true[mp] - y_pred[mp]) / y_true[mp]).mean()) if mp.any() else float("nan"),
        "n_nonzero": int(nz.sum()),
        "nonzero_frac": float(nz.mean()),
    }
    if nz.sum() > 1:
        yt = y_true[nz]
        ss_res = ((yt - y_pred[nz]) ** 2).sum()
        ss_tot = ((yt - yt.mean()) ** 2).sum()
        out["R2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return out
