"""HAWFormer: full model assembly."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .bwst import BWSTBlock, SinglePassDecoder, band_budgets
from .hagl import HAGL
from .wavelet import HaarBands


@dataclass
class ModelConfig:
    num_nodes: int
    t_history: int = 12
    t_horizon: int = 12
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    sim_heads: int = 2
    wavelet_levels: int = 2
    cheb_k: int = 2
    dropout: float = 0.1
    time_dim: int = 16
    steps_per_day: int = 288
    lap_pe_dim: int = 8
    # neighbourhood budgets
    kappa0: int = 32
    gamma: float = 0.5
    residual_from_last: bool = False
    kappa_min: int = 4
    geo_topk: int = 12
    # graph learner
    topk: int = 20
    low_dim: int = 16
    high_dim: int = 32
    P: int = 6
    Q: int = 8
    # ablation switches
    use_micro: bool = True
    use_meso: bool = True
    use_high: bool = True
    use_gcn: bool = True
    use_dla: bool = True
    use_wavelet: bool = True
    identity_affinity: bool = False
    soft_membership: bool = True
    use_spectral_descriptor: bool = True
    static_hierarchy: bool = False
    membership_temperature: float = 1.0
    reg_covar: float = 1e-2
    # Recompute each BWST block's activations during the backward pass instead
    # of keeping them.  Mathematically exact -- it trades ~30% more compute for
    # roughly one block's worth of activation memory instead of n_layers' worth.
    # Off by default; the speed tuner turns it on when the card needs it.
    grad_checkpoint: bool = False
    # Recompute the Laplacian PE whenever the learned graph is frozen in.
    relap_on_freeze: bool = True


class DataEmbedding(nn.Module):
    """Value + structural + calendar + positional + local-context embedding.

    The structural encoding is computed from the *learned* graph, so a sensor's
    positional identity is defined by its role in the inferred dependency
    structure rather than by its place in the road topology.  It is refreshed
    whenever the graph is.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.value = nn.Linear(1, d)
        self.local = nn.Conv1d(1, d, kernel_size=3, padding=1)
        self.tod = nn.Embedding(cfg.steps_per_day + 1, cfg.time_dim)
        self.dow = nn.Embedding(7, cfg.time_dim)
        self.time_to_d = nn.Linear(2 * cfg.time_dim, d)
        self.lap = nn.Linear(cfg.lap_pe_dim, d)
        self.register_buffer("lap_pe",
                             torch.zeros(cfg.num_nodes, cfg.lap_pe_dim),
                             persistent=True)
        pe = torch.zeros(cfg.t_history, d)
        pos = torch.arange(cfg.t_history).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("tpe", pe, persistent=False)
        self.drop = nn.Dropout(cfg.dropout)

    @torch.no_grad()
    def set_lap_pe(self, A: torch.Tensor, k: int) -> None:
        """Laplacian eigenvector positional encoding of the current graph."""
        A = 0.5 * (A + A.t())
        A = A.detach().cpu().double().numpy()
        np.fill_diagonal(A, 0.0)
        deg = A.sum(1)
        dinv = np.zeros_like(deg)
        nz = deg > 1e-12
        dinv[nz] = deg[nz] ** -0.5
        L = np.eye(A.shape[0]) - (A * dinv[:, None]) * dinv[None, :]
        w, v = np.linalg.eigh(L)
        pe = v[:, 1:k + 1]
        if pe.shape[1] < k:                       # tiny graphs
            pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
        pe = pe / np.maximum(np.abs(pe).max(0, keepdims=True), 1e-9)
        self.lap_pe.copy_(torch.as_tensor(pe, dtype=self.lap_pe.dtype,
                                          device=self.lap_pe.device))

    def forward(self, x: torch.Tensor, t_idx: torch.Tensor):
        """x: [B,T,N,1];  t_idx: [B,T,2] -> ([B,T,N,d], time_emb [B,T,2*td])"""
        B, T, N, _ = x.shape
        v = self.value(x)
        loc = self.local(x.permute(0, 2, 3, 1).reshape(B * N, 1, T))
        loc = loc.view(B, N, -1, T).permute(0, 3, 1, 2)
        te = torch.cat([self.tod(t_idx[..., 0].clamp(0, self.tod.num_embeddings - 1)),
                        self.dow(t_idx[..., 1].clamp(0, 6))], dim=-1)   # [B,T,2td]
        cal = self.time_to_d(te).unsqueeze(2)
        lap = self.lap(self.lap_pe).unsqueeze(0).unsqueeze(0)
        h = v + loc + cal + lap + self.tpe[:T].view(1, T, 1, -1)
        return self.drop(h), te


class HAWFormer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.levels = cfg.wavelet_levels if cfg.use_wavelet else 0
        self.num_bands = self.levels + 1

        self.embed = DataEmbedding(cfg)
        self.haar = HaarBands(self.levels) if cfg.use_wavelet else None
        self.blocks = nn.ModuleList([
            BWSTBlock(
                d_model=cfg.d_model, t_history=cfg.t_history,
                num_bands=self.num_bands, heads=cfg.n_heads,
                sim_heads=cfg.sim_heads, cheb_k=cfg.cheb_k,
                dropout=cfg.dropout, time_dim=cfg.time_dim,
                use_gcn=cfg.use_gcn, use_dla=cfg.use_dla,
            ) for _ in range(cfg.n_layers)
        ])
        self.skip = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.d_model) for _ in range(cfg.n_layers)])
        self.decoder = SinglePassDecoder(
            cfg.d_model, cfg.t_history, cfg.t_horizon, cfg.n_heads, cfg.dropout)

        self.hagl = HAGL(
            num_nodes=cfg.num_nodes, in_steps=cfg.t_history,
            P=cfg.P, Q=cfg.Q, low_dim=cfg.low_dim, high_dim=cfg.high_dim,
            topk=cfg.topk, use_micro=cfg.use_micro, use_meso=cfg.use_meso,
            use_high=cfg.use_high, identity_affinity=cfg.identity_affinity,
        )

        N = cfg.num_nodes
        self.register_buffer("A_pre", torch.eye(N), persistent=True)
        self.register_buffer("A_mi", torch.zeros(N, N), persistent=True)
        self.register_buffer("H_s", torch.ones(N, 1), persistent=False)
        self.register_buffer("H_g", torch.ones(N, 1), persistent=False)
        self.register_buffer("geo_mask", torch.ones(N, N, dtype=torch.bool),
                             persistent=True)
        self.register_buffer("A_frozen", torch.eye(N), persistent=True)
        self.budgets = band_budgets(cfg.kappa0, cfg.gamma, self.num_bands,
                                    cfg.kappa_min, N)

    # -- external state ------------------------------------------------------
    def set_static_inputs(self, A_pre: np.ndarray, A_mi: Optional[np.ndarray],
                          latlon: Optional[np.ndarray]) -> None:
        dev = self.A_pre.device
        self.A_pre.copy_(torch.as_tensor(A_pre, dtype=torch.float32, device=dev))
        if A_mi is not None:
            self.A_mi.copy_(torch.as_tensor(A_mi, dtype=torch.float32, device=dev))
        self.geo_mask.copy_(self._build_geo_mask(A_pre, latlon))
        self.embed.set_lap_pe(self.A_pre, self.cfg.lap_pe_dim)
        self.A_frozen.copy_(self.A_pre)

    def _build_geo_mask(self, A_pre: np.ndarray,
                        latlon: Optional[np.ndarray]) -> torch.Tensor:
        """Static structural neighbourhood: connections between adjacent road
        segments do not change over an hour, so this mask is shared by all
        bands (Section V-C)."""
        N = A_pre.shape[0]
        k = int(min(max(self.cfg.geo_topk, 1), N))
        base = A_pre.copy()
        if latlon is not None and np.isfinite(latlon).all():
            from ..data.geohash import haversine
            d = haversine(latlon[:, 0][:, None], latlon[:, 1][:, None],
                          latlon[:, 0][None, :], latlon[:, 1][None, :])
            base = -d                                   # nearer = larger score
        idx = np.argpartition(-base, k - 1, axis=1)[:, :k]
        m = np.zeros((N, N), dtype=bool)
        m[np.arange(N)[:, None], idx] = True
        np.fill_diagonal(m, True)
        return torch.as_tensor(m, device=self.geo_mask.device)

    def set_hierarchy(self, H_s: np.ndarray, H_g: np.ndarray) -> None:
        dev = self.A_pre.device
        self.H_s = torch.as_tensor(H_s, dtype=torch.float32, device=dev)
        self.H_g = torch.as_tensor(H_g, dtype=torch.float32, device=dev)
        if self.hagl.meso is not None:
            self.hagl.meso.resize(H_s.shape[1], H_g.shape[1], dev)
            self.hagl.meso.to(dev)

    def freeze_graph(self, A: torch.Tensor) -> None:
        """Cache the sparsified graph used by the predictor between updates.

        Recomputing the Laplacian positional encoding here moves the predictor's
        *input* representation every time the graph is updated -- an eigenbasis
        that can rotate or change sign between refreshes, under a network that
        is mid-training.  ``relap_on_freeze=false`` pins the encoding to the
        predefined graph so the graph learner still feeds the attention and the
        Chebyshev branch, but stops perturbing the embedding.
        """
        self.A_frozen.copy_(A.detach())
        if self.cfg.relap_on_freeze:
            self.embed.set_lap_pe(A.detach(), self.cfg.lap_pe_dim)

    # -- graph -------------------------------------------------------------
    def learn_graph(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.hagl(
            x.squeeze(-1), self.A_pre,
            self.A_mi if self.cfg.use_micro else None,
            self.H_s, self.H_g,
        )
        out["A_sparse"] = self.hagl.sparsify(out["A_star"])
        return out

    # -- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor, t_idx: torch.Tensor,
                A: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """x: [B, T_h, N, 1] normalised;  t_idx: [B, T_h, 2]."""
        A = self.A_frozen if A is None else A
        h, time_emb = self.embed(x, t_idx)
        bands = self.haar(h, time_dim=1) if self.haar is not None else [h]

        acc = 0.0
        diag: Dict[str, torch.Tensor] = {}
        ckpt = self.cfg.grad_checkpoint and self.training and torch.is_grad_enabled()
        for i, blk in enumerate(self.blocks):
            if ckpt:
                # checkpoint() only passes tensors through, so the block's
                # diagnostics come back as a bare tensor and the band weights --
                # a function of the block's parameters alone -- are recomputed
                # here rather than carried out of the recomputed graph.
                def _run(*bt, _blk=blk):
                    hh, dd = _blk(list(bt), A, self.H_s, self.H_g,
                                  self.geo_mask, self.budgets, time_emb)
                    return hh, dd["mask_density_band0"]
                h, md = torch.utils.checkpoint.checkpoint(
                    _run, *bands, use_reentrant=False)
                d = {"mask_density_band0": md,
                     "band_weights": torch.softmax(blk.band_logits, 0).detach()}
            else:
                h, d = blk(bands, A, self.H_s, self.H_g,
                           self.geo_mask, self.budgets, time_emb)
            acc = acc + self.skip[i](h)
            bands = self.haar(h, time_dim=1) if self.haar is not None else [h]
            if i == len(self.blocks) - 1:
                diag = d
        y = self.decoder(acc)
        if self.cfg.residual_from_last:
            # Anchor the forecast on the last observed value and let the network
            # predict the CHANGE from it.  Without this the decoder has to
            # reconstruct the current level of every node from the encoder
            # state before it can say anything about the future, and the
            # horizon curves show it does that worse than simply copying:
            # at fifteen minutes the model is 34% behind persistence on
            # Qingdao.  The anchor makes persistence the zero of the output
            # space rather than something the model must learn to imitate.
            y = y + x[:, -1, :, 0].unsqueeze(1)
        return {"pred": y, **diag}


def huber_loss(pred: torch.Tensor, target: torch.Tensor,
               mask: Optional[torch.Tensor] = None,
               delta: float = 1.0) -> torch.Tensor:
    """Huber loss, robust to the outlying flow values that follow incidents."""
    err = pred - target
    a = err.abs()
    l = torch.where(a <= delta, 0.5 * err ** 2, delta * (a - 0.5 * delta))
    if mask is None:
        return l.mean()
    m = mask.float()
    return (l * m).sum() / m.sum().clamp_min(1.0)
