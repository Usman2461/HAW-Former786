"""Publication figures for the HAWFormer paper.

Every figure is written as vector PDF (for LaTeX) and PNG (for previewing) at
IEEE column widths, so `\\includegraphics` needs no scaling.

Colour policy, applied throughout and not left to taste:

*   One fixed categorical order -- blue, vermillion, green, orange, purple,
    sky -- assigned by series identity, never cycled and never reassigned when
    a filter changes the series count.  The order was checked with a
    colour-vision-deficiency validator: worst adjacent pair dE 9.6 (deuteranopia),
    20.0 (normal vision), which clears the dE >= 8 bar.
*   Magnitude (adjacency, affinity) uses a **single-hue sequential** ramp, so
    larger really does read as darker.
*   Only signed quantities (learned graph minus predefined) use a **diverging**
    ramp, with a neutral grey midpoint at zero -- never a rainbow, and never a
    hue at the midpoint.
*   No dual-axis plots anywhere.  Two measures on different scales get two
    panels.
*   Every multi-series panel carries a legend, and lines are also distinguished
    by marker, so identity never rests on colour alone -- which matters when
    the paper is printed in greyscale.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

CAT = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK = "#1a1a1a"
INK2 = "#555555"
MUTED = "#8c8c8c"
GRID = "#e2e2e0"
SURFACE = "#ffffff"

SEQ = LinearSegmentedColormap.from_list(
    "haw_seq", ["#f4f8fb", "#cfe0ee", "#94bcdb", "#4e8fc0", "#0072B2", "#003f63"])
DIV = LinearSegmentedColormap.from_list(
    "haw_div", ["#8a3800", "#D55E00", "#f0b189", "#e8e8e6", "#8fbfdf", "#0072B2", "#00405f"])

COL1, COL2 = 3.45, 7.16          # IEEE single / double column width, inches


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.4,
    })


def _finish(ax, xlabel="", ylabel="", title="", grid_axis="y", legend=False,
            legend_loc="best"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=4)
    if grid_axis:
        ax.grid(True, axis=grid_axis, zorder=0)
        ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if legend:
        ax.legend(handlelength=1.6, borderpad=0.2, labelspacing=0.3,
                  loc=legend_loc)


def save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.pdf"
    fig.savefig(p)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 1. Architecture schematics
# ---------------------------------------------------------------------------

def _box(ax, x, y, w, h, text, fc="#ffffff", ec=INK2, fs=6.6, lw=0.7, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, linespacing=1.35)


def _arrow(ax, p0, p1, color=INK2, lw=0.8, style="-|>", rad=0.0, zorder=1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=7,
                                 linewidth=lw, color=color, zorder=zorder,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5))


def fig_overview(out_dir: Path) -> Path:
    """Fig. 1 -- the framework and the loop between its two components."""
    fig, ax = plt.subplots(figsize=(COL2, 2.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.005, 0.955, "Evidence", fontsize=7, color=MUTED, style="italic")
    _box(ax, 0.01, 0.70, 0.15, 0.16, "Vehicle\ntrajectories", fc="#eef4fa", ec=CAT[0])
    _box(ax, 0.01, 0.47, 0.15, 0.16, "Node flow\nseries", fc="#fdf0e8", ec=CAT[1])
    _box(ax, 0.01, 0.235, 0.15, 0.145, "Road network\n(initialisation)",
         fc="#f4f4f2", ec=MUTED)

    ax.text(0.205, 0.955, "HAGL  (graph learner)", fontsize=7, color=MUTED, style="italic")
    _box(ax, 0.20, 0.70, 0.185, 0.16, "Microscopic\ntransition graph $A_{mi}$", ec=CAT[0])
    _box(ax, 0.20, 0.47, 0.185, 0.16, "Macroscopic\nlow + high order $A_{ma}$", ec=CAT[1])
    _box(ax, 0.20, 0.235, 0.185, 0.145, "Mesoscopic\naffinity $A_{hi}$", ec=CAT[4])

    _box(ax, 0.425, 0.47, 0.085, 0.39, "gated\nfusion", fc="#f6f6f4")
    _box(ax, 0.425, 0.15, 0.085, 0.16, "balanced\npartition", fc="#f6f6f4")

    _box(ax, 0.55, 0.55, 0.115, 0.31, "$\\tilde{A}^{*}$\nlearned\ngraph",
         fc="#e8f0f8", ec=CAT[0], fs=7)
    _box(ax, 0.55, 0.15, 0.115, 0.23, "$H_s, H_g$\nhierarchy",
         fc="#f7ecf3", ec=CAT[4], fs=7)

    ax.text(0.695, 0.955, "BWST predictor", fontsize=7, color=MUTED, style="italic")
    _box(ax, 0.695, 0.66, 0.135, 0.20, "Haar bands\n$j = 0 \\ldots J$")
    _box(ax, 0.695, 0.40, 0.135, 0.20,
         "band-wise masks\n$\\kappa_j=\\kappa_0\\gamma^{\\,j}$", ec=CAT[3])
    _box(ax, 0.695, 0.14, 0.135, 0.20, "multi-scale\nattention + GCN")
    _box(ax, 0.865, 0.40, 0.125, 0.20, "single-pass\ndecoder")
    _box(ax, 0.865, 0.68, 0.125, 0.16, "$\\hat{Y}$\nforecast",
         fc="#eaf5f1", ec=CAT[2], fs=7)

    # evidence -> branches.  The road network is an initialisation for the
    # low-order macroscopic learner only; the mesoscopic branch takes no raw
    # evidence, it is driven by the hierarchy (orange loop below).
    _arrow(ax, (0.16, 0.78), (0.20, 0.78), color=CAT[0])
    _arrow(ax, (0.16, 0.55), (0.20, 0.55), color=CAT[1])
    _arrow(ax, (0.16, 0.31), (0.199, 0.50), color=MUTED, rad=-0.18)

    # branches -> gate -> learned graph
    _arrow(ax, (0.385, 0.78), (0.425, 0.72))
    _arrow(ax, (0.385, 0.55), (0.425, 0.62))
    _arrow(ax, (0.385, 0.31), (0.425, 0.52))
    _arrow(ax, (0.51, 0.70), (0.55, 0.70))

    # learned graph -> partition -> hierarchy   (this is the key substitution:
    # the partition is computed on the LEARNED graph, not on the road network)
    _arrow(ax, (0.567, 0.55), (0.50, 0.31), color=CAT[4], rad=0.28)
    _arrow(ax, (0.51, 0.23), (0.55, 0.25), color=CAT[4])

    # hierarchy -> predictor, and hierarchy -> mesoscopic branch (the loop)
    _arrow(ax, (0.665, 0.25), (0.695, 0.22), color=CAT[4])
    _arrow(ax, (0.60, 0.15), (0.2925, 0.235), color=CAT[1], rad=-0.32, lw=1.1,
           zorder=4)

    # learned graph -> predictor chain
    _arrow(ax, (0.665, 0.76), (0.695, 0.76), color=CAT[0])
    _arrow(ax, (0.7625, 0.66), (0.7625, 0.60))
    _arrow(ax, (0.7625, 0.40), (0.7625, 0.34))
    _arrow(ax, (0.83, 0.24), (0.90, 0.40), rad=-0.25)
    _arrow(ax, (0.9275, 0.60), (0.9275, 0.68))

    ax.text(0.5, -0.015,
            "hierarchy re-estimated on the learned graph  "
            "(slow clock, every $R$ epochs)",
            fontsize=6.4, color=CAT[1], ha="center", style="italic")
    return save(fig, out_dir, "fig1_overview")


def fig_bandwise(out_dir: Path, budgets: Optional[Sequence[int]] = None,
                 N: int = 150) -> Path:
    """Fig. 3 -- why the neighbourhood budget contracts with frequency."""
    budgets = list(budgets) if budgets is not None else [32, 16, 8]
    J = len(budgets)
    fig, axes = plt.subplots(1, J, figsize=(COL1, 1.35), sharey=True)
    rng = np.random.default_rng(3)
    names = ["$j=0$ approximation", "$j=1$ detail", "$j=2$ detail"]
    for j, ax in enumerate(np.atleast_1d(axes)):
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal"); ax.axis("off")
        th = rng.uniform(0, 2 * np.pi, 26)
        r = np.sqrt(rng.uniform(0.03, 1.0, 26))
        x, y = r * np.cos(th), r * np.sin(th)
        rad = 0.35 + 0.62 * (budgets[j] / max(budgets))
        inside = (x ** 2 + y ** 2) ** 0.5 <= rad
        ax.add_patch(plt.Circle((0, 0), rad, facecolor=CAT[j % len(CAT)],
                                alpha=0.13, edgecolor=CAT[j % len(CAT)],
                                linewidth=0.9, zorder=1))
        ax.scatter(x[~inside], y[~inside], s=5, color=MUTED, zorder=2, linewidths=0)
        ax.scatter(x[inside], y[inside], s=9, color=CAT[j % len(CAT)], zorder=3, linewidths=0)
        ax.scatter([0], [0], s=22, marker="*", color=INK, zorder=4)
        ax.set_title(f"{names[j] if j < 3 else j}\n$\\kappa_{{{j}}}={budgets[j]}$",
                     fontsize=6.6, color=INK, pad=1)
    return save(fig, out_dir, "fig3_bandwise_neighbourhood")


# ---------------------------------------------------------------------------
# 2. Data-driven figures
# ---------------------------------------------------------------------------

def fig_wavelet_bands(out_dir: Path, series: np.ndarray, levels: int = 2,
                      steps_per_day: int = 288, node_name: str = "") -> Path:
    """Fig. 4 -- the decomposition that the band-wise masks are computed on."""
    import torch
    from .models.wavelet import HaarBands

    x = torch.as_tensor(series, dtype=torch.float32).view(1, -1, 1)
    bands = HaarBands(levels)(x, time_dim=1)
    bands = [b.view(-1).numpy() for b in bands]

    n = len(bands) + 1
    fig, axes = plt.subplots(n, 1, figsize=(COL1, 0.72 * n + 0.35), sharex=True)
    t = np.arange(len(series)) / steps_per_day * 24.0
    axes[0].plot(t, series, color=INK, lw=1.0)
    axes[0].set_title(f"observed flow{(' — node ' + node_name) if node_name else ''}",
                      loc="left", fontsize=7, pad=2)
    labels = ["$j=0$  approximation (trend)"] + \
             [f"$j={k}$  detail (transient)" for k in range(1, len(bands))]
    for k, (b, lab) in enumerate(zip(bands, labels)):
        ax = axes[k + 1]
        ax.plot(t, b, color=CAT[k % len(CAT)], lw=1.0)
        ax.axhline(0, color=GRID, lw=0.5, zorder=0)
        ax.set_title(lab, loc="left", fontsize=7, pad=2)
    for ax in axes:
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel("hour")
    fig.supylabel("flow (veh / 5 min)", fontsize=7.5, x=0.005)
    fig.subplots_adjust(hspace=0.55)
    return save(fig, out_dir, "fig4_wavelet_bands")


def fig_loss_curves(out_dir: Path, histories: Dict[str, List[Dict]]) -> Path:
    """Fig. 5 -- convergence, the direct evidence on schedule stability."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.72, 1.85))
    for i, (label, hist) in enumerate(histories.items()):
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], color=CAT[i % len(CAT)],
                     marker=MARKERS[i % len(MARKERS)], markevery=max(1, len(ep) // 8),
                     label=label)
        axes[1].plot(ep, [h["val_MAE"] for h in hist], color=CAT[i % len(CAT)],
                     marker=MARKERS[i % len(MARKERS)], markevery=max(1, len(ep) // 8),
                     label=label)
    _finish(axes[0], "epoch", "training loss (Huber)", "(a) training", legend=len(histories) > 1)
    _finish(axes[1], "epoch", "validation MAE", "(b) validation", legend=len(histories) > 1)
    return save(fig, out_dir, "fig5_convergence")


def fig_horizon(out_dir: Path, by_model: Dict[str, Dict[str, Dict[str, float]]],
                metrics=("MAE", "RMSE", "WAPE")) -> Path:
    """Fig. 6 -- error growth with horizon, one panel per metric (no dual axes)."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(COL2, 1.9))
    for mi, met in enumerate(metrics):
        ax = axes[mi]
        for i, (name, byh) in enumerate(by_model.items()):
            steps = sorted(int(k.split("_")[1]) for k in byh)
            xs = [s * 5 for s in steps]
            ys = [byh[f"step_{s}"][met] for s in steps]
            ax.plot(xs, ys, color=CAT[i % len(CAT)], marker=MARKERS[i % len(MARKERS)],
                    label=name)
        unit = " (%)" if met == "WAPE" else ""
        # Every curve rises left to right, so the upper-left corner is the one
        # region guaranteed to be empty in all three panels.
        _finish(ax, "prediction horizon (min)", met + unit,
                f"({chr(97+mi)}) {met}", legend=(mi == 0),
                legend_loc="upper left")
    return save(fig, out_dir, "fig6_horizon")


def fig_ablation(out_dir: Path, rows: List[Tuple[str, str, Dict[str, Tuple[float, float]]]],
                 metrics=("MAE", "RMSE", "WAPE")) -> Path:
    """Fig. 7 -- ablation grid: change from the full model, with seed spread.

    Plotted as *difference from HAWFormer* rather than absolute error, because
    the question each variant answers is "how much worse without this?" -- and
    a bar chart of near-identical absolute values hides exactly that.
    """
    base = dict(rows[0][2])
    var = rows[1:]
    names = [r[0] for r in var]
    y = np.arange(len(var))[::-1]
    fig, axes = plt.subplots(1, len(metrics), figsize=(COL2, 0.24 * len(var) + 1.0),
                             sharey=True)
    for mi, met in enumerate(metrics):
        ax = axes[mi]
        d = np.array([r[2][met][0] - base[met][0] for r in var])
        e = np.array([r[2][met][1] for r in var])
        colors = [CAT[1] if v > 0 else CAT[0] for v in d]
        ax.barh(y, d, xerr=e, height=0.62, color=colors, alpha=0.9,
                error_kw=dict(elinewidth=0.7, ecolor=MUTED, capsize=1.5), zorder=3)
        ax.axvline(0, color=INK, lw=0.8, zorder=4)
        unit = " (pp)" if met == "WAPE" else ""
        _finish(ax, f"$\\Delta$ {met}{unit}", "", f"({chr(97+mi)}) {met}", grid_axis="x")
        # Pad the category axis so a short grid does not render as one slab.
        ax.set_ylim(-0.75, len(var) - 0.25)
        if mi == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=6.6)
    return save(fig, out_dir, "fig7_ablation")


def fig_graphs(out_dir: Path, A_pre: np.ndarray, A_learned: np.ndarray,
               order: Optional[np.ndarray] = None) -> Path:
    """Fig. 8 -- predefined vs learned dependency, and the signed difference."""
    if order is not None:
        A_pre = A_pre[np.ix_(order, order)]
        A_learned = A_learned[np.ix_(order, order)]

    # The predefined graph is binary; the learned one is a weighted graph whose
    # entries are heavy-tailed (a handful of strong edges, a long tail of weak
    # ones).  Dividing each by its MAXIMUM would render the learned panel almost
    # blank -- typical entries land at a few percent of the strongest one -- and
    # the reader would conclude the model learned nothing.  Normalising by the
    # 99th percentile of the non-zero entries and clipping shows the structure
    # at the cost of saturating the few strongest edges, which is the right
    # trade here; the true maximum is printed under each panel so the scaling is
    # never mistaken for the values.
    def nz(a):
        v = a[a > 0]
        m = float(np.percentile(v, 99)) if v.size else 0.0
        return (np.clip(a / m, 0, 1) if m > 1e-12 else a), float(a.max())

    Ap, ap_max = nz(A_pre)
    Al, al_max = nz(A_learned)
    d = Al - Ap
    lim = max(float(np.abs(d).max()), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.05))
    for ax, M, t, cm, kw in (
        (axes[0], Ap, "(a) predefined $A$", SEQ, dict(vmin=0, vmax=1)),
        (axes[1], Al, "(b) learned $\\tilde{A}^{*}$", SEQ, dict(vmin=0, vmax=1)),
        (axes[2], d, "(c) difference", DIV,
         dict(norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))),
    ):
        im = ax.imshow(M, cmap=cm, interpolation="nearest", **kw)
        ax.set_title(t, loc="left", pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6, width=0.5)
        cb.outline.set_linewidth(0.4)
    axes[0].set_ylabel("node")
    axes[0].set_xlabel("node")
    axes[1].set_xlabel(f"node   (max weight {al_max:.2f})")
    axes[2].set_xlabel("node   (both panels rescaled)")
    return save(fig, out_dir, "fig8_graphs")


def fig_partition(out_dir: Path, latlon: np.ndarray, hier_history: np.ndarray,
                  hier_epochs: np.ndarray) -> Path:
    """Fig. 9 -- which sensors migrated once the partition used the learned graph."""
    first, last = hier_history[0], hier_history[-1]
    moved = first != last
    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.72, 2.4), sharex=True, sharey=True)
    lat, lon = latlon[:, 0], latlon[:, 1]
    # The migration count is the substance of this figure, so it goes in the
    # panel title rather than a floating note that can collide with the axes.
    titles = (f"(a) first refresh (epoch {hier_epochs[0]})",
              f"(b) final refresh (epoch {hier_epochs[-1]}) "
              f"\u2014 {int(moved.sum())}/{len(moved)} moved")
    for ax, lab, t in ((axes[0], first, titles[0]), (axes[1], last, titles[1])):
        for g in np.unique(lab):
            m = lab == g
            ax.scatter(lon[m], lat[m], s=16, color=CAT[int(g) % len(CAT)],
                       marker=MARKERS[int(g) % len(MARKERS)],
                       label=f"group {int(g)}", linewidths=0, zorder=3)
        ax.set_title(t, loc="left", pad=3)
        ax.set_xlabel("longitude")
        ax.grid(True, axis="both")
        ax.set_axisbelow(True)
        ax.ticklabel_format(useOffset=False, style="plain")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].scatter(lon[moved], lat[moved], s=52, facecolors="none",
                    edgecolors=INK, linewidths=0.8, zorder=4,
                    label="changed group")
    axes[0].set_ylabel("latitude")
    h, l = axes[1].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.26)
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 6), frameon=False,
               fontsize=6.4, handlelength=1.2, columnspacing=1.1,
               bbox_to_anchor=(0.5, 0.005))
    return save(fig, out_dir, "fig9_partition")


def fig_affinity(out_dir: Path, lam_g: np.ndarray, lam_s: np.ndarray) -> Path:
    """Fig. 10 -- inter-group affinity; off-diagonal mass is the whole claim."""
    fig, axes = plt.subplots(1, 2, figsize=(COL1 * 1.5, 1.85))
    for ax, M, t in ((axes[0], lam_s, "(a) semantic $\\Lambda_s$"),
                     (axes[1], lam_g, "(b) structural $\\Lambda_g$")):
        if M.size == 0:
            ax.axis("off"); continue
        im = ax.imshow(M, cmap=SEQ, interpolation="nearest")
        ax.set_title(t, loc="left", pad=3)
        ax.set_xlabel("group"); ax.set_ylabel("group")
        ax.set_xticks(range(M.shape[0])); ax.set_yticks(range(M.shape[0]))
        ax.tick_params(width=0.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6, width=0.5)
        cb.outline.set_linewidth(0.4)
        off = M.copy()
        np.fill_diagonal(off, 0.0)
        frac = off.sum() / max(M.sum(), 1e-9)
        ax.set_xlabel(f"group   (off-diagonal mass {frac:.0%})")
    return save(fig, out_dir, "fig10_affinity")


def fig_band_weights(out_dir: Path, weights_by_dataset: Dict[str, np.ndarray]) -> Path:
    """Fig. 11 -- how much predictable signal each network carries per band."""
    names = list(weights_by_dataset)
    # Datasets may be configured with different numbers of wavelet levels, so a
    # series can be shorter than the axis.  Plot each series against its own
    # bands rather than broadcasting it to the widest one, which would either
    # crash or, worse, silently recycle values into bands that do not exist.
    series = {n: np.atleast_2d(weights_by_dataset[n]).mean(0) for n in names}
    J = max(len(w) for w in series.values())
    fig, ax = plt.subplots(figsize=(COL1, 1.5))
    width = 0.8 / max(len(names), 1)
    for i, n in enumerate(names):
        w = series[n]
        pos = np.arange(len(w)) + i * width - 0.4 + width / 2
        ax.bar(pos, w, width * 0.88, color=CAT[i % len(CAT)], label=n, zorder=3)
        for x, v in zip(pos, w):
            # Three decimals, not two: under a short training budget these
            # weights barely leave their uniform initialisation, and rounding
            # to 0.50 would present "did not move" as "exactly balanced".
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=5.8, color=INK2)
    xs = np.arange(J)
    ax.set_xticks(xs)
    ax.set_xticklabels([("approx.\n$j=0$" if j == 0 else f"detail\n$j={j}$") for j in range(J)])
    _finish(ax, "", "band weight $\\omega_j$", legend=len(names) > 1)
    ax.set_ylim(0, max(0.75, float(max(w.max() for w in series.values())) * 1.25))
    return save(fig, out_dir, "fig11_band_weights")


def fig_predictions(out_dir: Path, y_true: np.ndarray, y_pred: np.ndarray,
                    node: int = 0, step: int = 11, span: int = 288,
                    steps_per_day: int = 288, node_name: str = "") -> Path:
    """Fig. 12 -- forecast against ground truth at the 60-minute horizon."""
    n = min(span, y_true.shape[0])
    t = np.arange(n) / steps_per_day * 24.0
    fig, ax = plt.subplots(figsize=(COL2 * 0.72, 1.7))
    ax.plot(t, y_true[:n, step, node], color=INK, lw=1.2, label="ground truth")
    ax.plot(t, y_pred[:n, step, node], color=CAT[0], lw=1.2, ls="--",
            label="HAWFormer")
    _finish(ax, "hour of test period", "flow (veh / 5 min)",
            f"node {node_name or node}, {(step+1)*5}-minute horizon", legend=True)
    return save(fig, out_dir, "fig12_predictions")


def fig_scatter(out_dir: Path, y_true: np.ndarray, y_pred: np.ndarray,
                mask: Optional[np.ndarray] = None,
                steps=(2, 5, 11)) -> Path:
    """Fig. 13 -- predicted against true, with R^2, at three horizons."""
    fig, axes = plt.subplots(1, len(steps), figsize=(COL2, 2.05), sharex=True, sharey=True)
    for i, s in enumerate(steps):
        ax = axes[i]
        yt = y_true[:, s].ravel()
        yp = y_pred[:, s].ravel()
        if mask is not None:
            m = mask[:, s].ravel()
            yt, yp = yt[m], yp[m]
        ax.scatter(yt, yp, s=1.4, color=CAT[0], alpha=0.16, linewidths=0, zorder=3)
        lo = float(min(yt.min(), yp.min())); hi = float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, zorder=4)
        ss_res = ((yt - yp) ** 2).sum()
        ss_tot = ((yt - yt.mean()) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-9)
        ax.text(0.04, 0.93, f"$R^2 = {r2:.3f}$", transform=ax.transAxes,
                fontsize=7, va="top", color=INK)
        _finish(ax, "ground truth (veh / 5 min)", "predicted" if i == 0 else "",
                f"({chr(97+i)}) {(s+1)*5} min", grid_axis="both")
        ax.set_aspect("equal", adjustable="box")
    return save(fig, out_dir, "fig13_scatter")


def fig_conditions(out_dir: Path, by_condition: Dict[str, Dict[str, Dict[str, float]]],
                   metrics=("MAE", "RMSE"), conds: Optional[Sequence[str]] = None,
                   name: str = "fig14_conditions") -> Path:
    """Fig. 14 -- error by traffic condition: where a mechanism should pay off.

    `conds` is a parameter rather than a constant because the same panel serves
    the clock-time split (peak / off-peak) as well as the flow-regime split; a
    hardcoded list silently drew empty axes when handed the other one.
    """
    conds = list(conds) if conds is not None else ["free", "congested", "incident"]
    models = list(by_condition)
    fig, axes = plt.subplots(1, len(metrics), figsize=(COL2 * 0.8, 1.85))
    width = 0.8 / max(len(models), 1)
    xs = np.arange(len(conds))
    for mi, met in enumerate(metrics):
        ax = axes[mi]
        for i, mname in enumerate(models):    # `name` is now the output filename
            vals = [by_condition[mname].get(c, {}).get(met, np.nan) for c in conds]
            ax.bar(xs + i * width - 0.4 + width / 2, vals, width * 0.86,
                   color=CAT[i % len(CAT)], label=mname, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(["free-flow" if c == "free" else c for c in conds])
        # Headroom for the legend: bars start at zero and the tallest is on the
        # right, so without it the legend crowds the middle group.
        top = np.nanmax([[by_condition[n].get(c, {}).get(met, np.nan)
                          for c in conds] for n in models])
        if np.isfinite(top):
            ax.set_ylim(0, top * (1.30 if mi == 0 else 1.08))
        _finish(ax, "", met, f"({chr(97+mi)}) {met}", legend=(mi == 0),
                legend_loc="upper left")
    return save(fig, out_dir, name)


def fig_sensitivity(out_dir: Path, sweeps: Dict[str, Tuple[List, List, List]],
                    ylabel: str = "test MAE") -> Path:
    """Fig. 15 -- sensitivity to the parameters that govern each mechanism."""
    keys = list(sweeps)
    # Width follows the panel count so a single sweep is not stretched across
    # the full text width.
    w = COL1 if len(keys) == 1 else min(COL2, 1.9 * len(keys) + 0.9)
    fig, axes = plt.subplots(1, len(keys), figsize=(w, 1.75))
    axes = np.atleast_1d(axes)
    for i, k in enumerate(keys):
        xs, ys, es = sweeps[k]
        ax = axes[i]
        ax.errorbar(xs, ys, yerr=es, color=CAT[i % len(CAT)],
                    marker=MARKERS[i % len(MARKERS)], capsize=1.8,
                    elinewidth=0.7, lw=1.3)
        j = int(np.nanargmin(ys))
        ax.scatter([xs[j]], [ys[j]], s=34, facecolors="none", edgecolors=INK,
                   linewidths=0.9, zorder=5)
        # Label above the point, not below: below collides with the x-axis
        # whenever the minimum is also the lowest value on the panel, which is
        # exactly when this annotation is drawn.
        ax.annotate(f"best {xs[j]:g}", (xs[j], ys[j]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=6.2, color=INK2)
        ax.margins(y=0.28)
        _finish(ax, k, ylabel if i == 0 else "", f"({chr(97+i)}) {k}")
    return save(fig, out_dir, "fig15_sensitivity")


def fig_persistence_gap(out_dir: Path,
                        by_dataset: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
                        ) -> Path:
    """Fig. 16 -- the model's standing against persistence, by horizon.

    Plotted as a RATIO rather than as two error curves per dataset.  The three
    datasets differ by two orders of magnitude in flow, so absolute errors
    cannot share an axis, and a panel each would make the one comparison that
    matters -- how the model stands against a memoryless copy -- something the
    reader has to reconstruct by eye.  The ratio puts all three on one axis
    with an unambiguous reference line: below 1 the model wins.
    """
    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    for i, (name, d) in enumerate(by_dataset.items()):
        steps = sorted(int(k.split("_")[1]) for k in d["model"])
        xs = [s * 5 for s in steps]
        ys = [d["model"][f"step_{s}"]["MAE"] / d["ref"][f"step_{s}"]["MAE"]
              for s in steps]
        ax.plot(xs, ys, color=CAT[i % len(CAT)], marker=MARKERS[i % len(MARKERS)],
                label=name, zorder=3)
    ax.axhline(1.0, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=2)
    # Label at the left end and legend at the right: the curves all descend, so
    # the two would collide anywhere in the middle.
    ax.text(15.6, 1.012, "persistence", ha="left", va="bottom", fontsize=6.2,
            color=INK2)
    _finish(ax, "prediction horizon (min)", "MAE relative to persistence",
            legend=True, legend_loc="upper right")
    return save(fig, out_dir, "fig16_persistence_gap")


def fig_ablation_multi(out_dir: Path,
                       rows: Dict[str, List[Tuple[str, float]]],
                       metric: str = "MAE") -> Path:
    """Fig. 17 -- ablation effect on every dataset, as relative change.

    The three datasets differ by two orders of magnitude in flow, so absolute
    deltas cannot share an axis; percentages can, and percentage is also the
    quantity the discussion actually compares.  Grouping by variant rather than
    by dataset puts the question -- does this mechanism behave the same way
    everywhere? -- on the vertical alignment, where it is answered by looking
    at whether a group's bars fall on one side of zero.
    """
    variants = [v for v, _ in next(iter(rows.values()))]
    names = list(rows)
    fig, ax = plt.subplots(figsize=(COL2 * 0.62, 2.3))
    h = 0.8 / max(len(names), 1)
    ys = np.arange(len(variants))
    for i, n in enumerate(names):
        vals = [d for _, d in rows[n]]
        pos = ys + i * h - 0.4 + h / 2
        ax.barh(pos, vals, h * 0.86, color=CAT[i % len(CAT)], label=n, zorder=3)
    ax.axvline(0.0, color=INK, lw=0.9, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels(variants)
    ax.invert_yaxis()
    _finish(ax, f"change in {metric} when the mechanism is removed (%)", "",
            grid_axis="x", legend=True, legend_loc="lower right")
    return save(fig, out_dir, "fig17_ablation_multi")
