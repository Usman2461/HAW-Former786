"""Test-set reporting: overall, per horizon, and conditioned on traffic state."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .data.dataset import Bundle
from .metrics import all_metrics, horizon_metrics, incident_flags
from .utils import get_logger

LOG = get_logger()


def condition_masks(
    bundle: Bundle,
    y_shape,
    starts: np.ndarray,
    t_history: int,
    congestion_quantile: float = 0.75,
    incident_percentile: float = 95.0,
) -> Dict[str, np.ndarray]:
    """Build [B, T_p, N] boolean masks for free-flow / congested / incident.

    Congestion is defined by a per-node flow quantile taken from the TRAINING
    split, not from the test set: using test quantiles would define the regimes
    using information the model is being evaluated on.

    Incidents are flagged on the raw (unsmoothed) series, since smoothing is
    designed to remove exactly the abrupt changes the flag looks for.
    """
    B, T_p, N = y_shape
    tr0, tr1 = bundle.splits["train"]
    thr = np.quantile(bundle.flow[tr0:tr1], congestion_quantile, axis=0)   # [N]

    inc_full = incident_flags(bundle.flow_raw, incident_percentile, T_p)    # [T,N]

    cong = np.zeros((B, T_p, N), dtype=bool)
    inc = np.zeros((B, T_p, N), dtype=bool)
    for b, s in enumerate(starts):
        e = int(s) + t_history
        cong[b] = bundle.flow[e:e + T_p] > thr[None, :]
        inc[b] = inc_full[e:e + T_p]
    return {"free": ~cong, "congested": cong, "incident": inc}


def full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid: np.ndarray,
    bundle: Bundle,
    starts: np.ndarray,
    t_history: int,
    horizons=(3, 6, 9, 12),
    mape_threshold: float = 5.0,
) -> Dict:
    rep: Dict = {"overall": all_metrics(y_true, y_pred, valid, mape_threshold)}
    rep["by_horizon"] = horizon_metrics(y_true, y_pred, valid,
                                        list(horizons), mape_threshold)
    cm = condition_masks(bundle, y_true.shape, starts, t_history)
    rep["by_condition"] = {}
    for name, m in cm.items():
        mm = m & valid
        n = int(mm.sum())
        rep["by_condition"][name] = {
            "n_observations": n,
            **all_metrics(y_true, y_pred, mm, mape_threshold),
        }
    return rep


def format_report(rep: Dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"=== {title} ===")
    o = rep["overall"]
    lines.append(
        f"overall   MAE {o['MAE']:8.4f}  RMSE {o['RMSE']:8.4f}  "
        f"WAPE {o['WAPE']:6.2f}%  MAPE {o['MAPE']:6.2f}% "
        f"(cov {o['MAPE_coverage']*100:.0f}%)  R2 {o['R2']:6.3f}  "
        f"Bias {o['Bias']:+.3f}"
    )
    for k, v in rep.get("by_horizon", {}).items():
        lines.append(
            f"  {k:>8}  MAE {v['MAE']:8.4f}  RMSE {v['RMSE']:8.4f}  "
            f"WAPE {v['WAPE']:6.2f}%  R2 {v['R2']:6.3f}"
        )
    for k, v in rep.get("by_condition", {}).items():
        lines.append(
            f"  {k:>9}  n={v['n_observations']:>8}  MAE {v['MAE']:8.4f}  "
            f"RMSE {v['RMSE']:8.4f}  WAPE {v['WAPE']:6.2f}%"
        )
    return "\n".join(lines)
