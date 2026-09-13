"""Kalman filtering and RTS smoothing for trajectory-derived flow series.

Flow obtained by aggregating GPS traces into five-minute bins at a node is
noisy for two reasons that have nothing to do with traffic: sampling (only a
fraction of vehicles are probes) and assignment error (a trace near a boundary
may be binned to the wrong cell).  Both are approximately additive and
zero-mean at the five-minute scale, which is exactly the regime a local-level
state-space model handles well.

Model (per node, independently):

    x_t = x_{t-1} + w_t,   w_t ~ N(0, q)     latent flow level
    y_t = x_t     + v_t,   v_t ~ N(0, r)     observed count

The smoother is run forwards (filter) then backwards (RTS), so every estimate
uses the whole series rather than only the past -- appropriate here because
smoothing is offline preprocessing, not an online filter.

`q` and `r` are estimated per node from the observed series: r from the
high-frequency component of the first difference, q from the residual
variance after removing it.  This keeps the smoother from flattening genuine
peak-hour structure, which a globally tuned q/r ratio tends to do.
"""
from __future__ import annotations

import numpy as np


def _estimate_qr(y: np.ndarray) -> tuple:
    """Estimate observation and process noise from a single series."""
    y = y[np.isfinite(y)]
    if y.size < 8:
        return 1.0, 1.0
    d1 = np.diff(y)
    # For a local-level model, Var(d1) = q + 2r and Cov(d1_t, d1_{t+1}) = -r.
    v = float(np.var(d1))
    c = float(np.mean((d1[:-1] - d1[:-1].mean()) * (d1[1:] - d1[1:].mean())))
    r = max(-c, 1e-3)
    q = max(v - 2 * r, 1e-3)
    return q, r


def kalman_smooth_series(y: np.ndarray, q: float = None, r: float = None) -> np.ndarray:
    """RTS-smooth a 1-D series with NaNs allowed (treated as missing)."""
    y = np.asarray(y, dtype=np.float64)
    T = y.shape[0]
    if q is None or r is None:
        q_e, r_e = _estimate_qr(y)
        q = q_e if q is None else q
        r = r_e if r is None else r

    xf = np.zeros(T)
    Pf = np.zeros(T)
    xp = np.zeros(T)
    Pp = np.zeros(T)

    obs = y[np.isfinite(y)]
    x, P = (float(obs[0]) if obs.size else 0.0), 1e4

    for t in range(T):
        # predict
        x_pred = x
        P_pred = P + q
        xp[t], Pp[t] = x_pred, P_pred
        # update
        if np.isfinite(y[t]):
            K = P_pred / (P_pred + r)
            x = x_pred + K * (y[t] - x_pred)
            P = (1 - K) * P_pred
        else:
            x, P = x_pred, P_pred
        xf[t], Pf[t] = x, P

    # RTS backward pass
    xs = xf.copy()
    Ps = Pf.copy()
    for t in range(T - 2, -1, -1):
        C = Pf[t] / max(Pp[t + 1], 1e-12)
        xs[t] = xf[t] + C * (xs[t + 1] - xp[t + 1])
        Ps[t] = Pf[t] + C * C * (Ps[t + 1] - Pp[t + 1])
    return xs


def kalman_smooth(flow: np.ndarray, clip_nonnegative: bool = True) -> np.ndarray:
    """Smooth a [T, N] flow matrix column-wise.

    Missing entries (NaN) are imputed by the smoother, which is the point of
    using a state-space model here rather than a moving average.
    """
    flow = np.asarray(flow, dtype=np.float64)
    out = np.empty_like(flow)
    for n in range(flow.shape[1]):
        out[:, n] = kalman_smooth_series(flow[:, n])
    if clip_nonnegative:
        out = np.maximum(out, 0.0)
    return out
