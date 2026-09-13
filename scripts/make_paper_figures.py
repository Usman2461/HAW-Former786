#!/usr/bin/env python3
"""The figures that can be drawn from the measured values file alone.

tables/paper_values.json already holds every model's horizon-wise metrics, its
per-horizon R^2, and its cost, so these figures are built from the same numbers
the tables print rather than from a second pass over the predictions -- a figure
that disagrees with the table beside it is worse than no figure.

    python scripts/make_paper_figures.py --out ../../newpaper/figures

Historical Average is excluded from the horizon and R^2 panels.  It is a
constant predictor an order of magnitude worse than everything else, and leaving
it in compresses every other curve into the bottom of the axis; it remains in
the tables, where it belongs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

ORDER = ["DCRNN", "STGCN", "ASTGCN", "Graph WaveNet", "AdapGL", "Adap-STWT",
         "HAWFormer"]
DSNAME = {"qdb19": "Qingdao", "tdrive": "T-Drive"}


def style(model: str) -> dict:
    """HAWFormer is drawn heavier than the baselines so it reads at a glance."""
    if model == "HAWFormer":
        return dict(color="#C1121F", lw=2.0, marker="o", ms=4, zorder=5)
    palette = {"DCRNN": "#4C6EF5", "STGCN": "#12B886", "ASTGCN": "#F59F00",
               "Graph WaveNet": "#7048E8", "AdapGL": "#0CA678",
               "Adap-STWT": "#495057"}
    return dict(color=palette.get(model, "#868E96"), lw=1.0, marker="s", ms=2.5,
                alpha=0.85)


def present(v: dict, ds: str) -> list:
    return [m for m in ORDER if m in v[ds]["models"]]


def fig_horizon(v: dict, ds: str, out: Path) -> Path:
    steps = [("15min", 15), ("30min", 30), ("45min", 45), ("60min", 60)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        for m in present(v, ds):
            rec = v[ds]["models"][m]["horizon"]
            xs = [x for _, x in steps if _ in rec]
            ys = [rec[k][metric]["mean"] for k, _ in steps if k in rec]
            es = [rec[k][metric]["std"] for k, _ in steps if k in rec]
            st = style(m)
            ax.errorbar(xs, ys, yerr=es, capsize=1.5, label=m, **st)
        ax.set_xlabel("prediction horizon (min)")
        ax.set_ylabel(metric)
        ax.set_xticks([x for _, x in steps])
    axes[0].legend(loc="upper left", frameon=False, ncol=1)
    fig.suptitle(f"{DSNAME[ds]}", y=1.02, fontsize=9)
    p = out / f"fig_horizon_{ds}.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_r2(v: dict, out: Path) -> Path:
    dss = [d for d in ("qdb19", "tdrive") if d in v]
    fig, axes = plt.subplots(1, len(dss), figsize=(7.0, 2.6))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, dss):
        models = [m for m in present(v, ds)
                  if "r2_by_horizon" in v[ds]["models"][m]]
        data = [np.asarray(v[ds]["models"][m]["r2_by_horizon"], dtype=float)
                for m in models]
        bp = ax.boxplot(data, labels=models, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", lw=1.0),
                        flierprops=dict(marker=".", ms=2))
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(style(m)["color"])
            patch.set_alpha(0.95 if m == "HAWFormer" else 0.55)
            patch.set_linewidth(1.2 if m == "HAWFormer" else 0.6)
        ax.set_title(DSNAME[ds])
        ax.set_ylabel("$R^2$ across the 12 horizons")
        ax.tick_params(axis="x", rotation=38)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
    p = out / "fig_r2.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_cost(v: dict, out: Path) -> Path:
    """Accuracy against training cost: the trade-off, not two separate tables."""
    dss = [d for d in ("qdb19", "tdrive") if d in v]
    fig, axes = plt.subplots(1, len(dss), figsize=(7.0, 2.5))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, dss):
        for m in present(v, ds):
            rec = v[ds]["models"][m]
            c = rec.get("cost", {})
            # Adap-STWT's cost is per outer round, not per epoch.  Plotting it
            # on a per-epoch axis would make a unit error visually, so it is
            # left to the table where the unit can be stated.
            if ("train_s_per_epoch" not in c
                    or c.get("train_time_unit") != "epoch"):
                continue
            st = style(m)
            ax.scatter(c["train_s_per_epoch"], rec["overall"]["MAE"]["mean"],
                       s=70 if m == "HAWFormer" else 28,
                       color=st["color"], zorder=5 if m == "HAWFormer" else 3,
                       edgecolor="black" if m == "HAWFormer" else "none",
                       linewidth=0.7,
                       marker="*" if m == "HAWFormer" else "o", label=m)
            ax.annotate(m, (c["train_s_per_epoch"], rec["overall"]["MAE"]["mean"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=5.5,
                        color="black" if m == "HAWFormer" else "#495057")
        ax.set_xscale("log")
        ax.set_xlabel("training time (s per epoch, log scale)")
        ax.set_ylabel("MAE")
        ax.set_title(DSNAME[ds])
        ax.margins(x=0.22, y=0.12)
    p = out / "fig_cost.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", default="tables/paper_values.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    v = json.load(open(a.values))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    made = [fig_r2(v, out), fig_cost(v, out)]
    for ds in ("qdb19", "tdrive"):
        if ds in v:
            made.append(fig_horizon(v, ds, out))
    for p in made:
        print(f"  + {p}")
    print(f"wrote {len(made)} figures to {out}")


if __name__ == "__main__":
    main()
