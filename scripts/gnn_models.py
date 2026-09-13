"""The four graph baselines from the Adap-STWT comparison, written from their papers.

  STGCN          Yu, Yin & Zhu, IJCAI 2018 -- ST-Conv blocks of
                 temporal gated conv (GLU) -> Chebyshev graph conv -> temporal
                 gated conv, then a temporal output layer.
  DCRNN          Li, Yu, Shahabi & Liu, ICLR 2018 -- encoder/decoder of GRU
                 cells whose matrix products are replaced by dual random-walk
                 diffusion convolution, trained with scheduled sampling.
  ASTGCN         Guo, Lin, Feng, Song & Wan, AAAI 2019 -- spatial and temporal
                 attention modulating a Chebyshev graph convolution, followed by
                 a temporal convolution, with a residual path.  Only the recent
                 component is used, which is the standard reduction when the
                 protocol feeds one hour of history and no daily/weekly windows.
  Graph WaveNet  Wu, Pan, Long, Jiang & Zhang, IJCAI 2019 -- stacked gated
                 dilated causal convolutions with a diffusion graph convolution
                 whose supports are the two transition matrices plus a learned
                 adaptive adjacency from node embeddings.

Every model takes normalised flow [B, T, N] and returns normalised [B, H, N],
so they share the harness, the windows and the metrics used everywhere else.
The graph supports are built once from the same adjacency the other models see.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# graph supports
# --------------------------------------------------------------------------

def _sym_norm_lap(adj: np.ndarray) -> np.ndarray:
    """Scaled symmetric normalised Laplacian, as STGCN and ASTGCN use."""
    a = np.asarray(adj, dtype=np.float64).copy()
    np.fill_diagonal(a, 0.0)
    d = a.sum(1)
    dinv = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    lap = np.eye(a.shape[0]) - (dinv[:, None] * a * dinv[None, :])
    ev = np.linalg.eigvalsh(lap)
    lmax = float(ev.max()) if np.isfinite(ev).all() and ev.max() > 0 else 2.0
    return (2.0 / lmax) * lap - np.eye(a.shape[0])


def cheb_supports(adj: np.ndarray, k: int) -> torch.Tensor:
    """[k, N, N] Chebyshev polynomials T_0..T_{k-1} of the scaled Laplacian."""
    lap = _sym_norm_lap(adj)
    n = lap.shape[0]
    out = [np.eye(n), lap]
    for i in range(2, k):
        out.append(2 * lap @ out[-1] - out[-2])
    return torch.tensor(np.stack(out[:k]), dtype=torch.float32)


def transition_supports(adj: np.ndarray) -> torch.Tensor:
    """[2, N, N] forward and backward random-walk matrices (DCRNN, GWNet)."""
    a = np.asarray(adj, dtype=np.float64).copy()
    a = a + np.eye(a.shape[0])
    f = a / np.maximum(a.sum(1, keepdims=True), 1e-12)
    b = a.T / np.maximum(a.T.sum(1, keepdims=True), 1e-12)
    return torch.tensor(np.stack([f, b]), dtype=torch.float32)


# --------------------------------------------------------------------------
# STGCN
# --------------------------------------------------------------------------

class _TemporalGatedConv(nn.Module):
    """Causal 1-D conv along time with a GLU, per STGCN eq. (5)."""

    def __init__(self, c_in: int, c_out: int, kt: int = 3):
        super().__init__()
        self.kt, self.c_out = kt, c_out
        self.conv = nn.Conv2d(c_in, 2 * c_out, (kt, 1))
        self.res = nn.Conv2d(c_in, c_out, (1, 1)) if c_in != c_out else nn.Identity()

    def forward(self, x):                       # [B, C, T, N]
        r = self.res(x)[:, :, self.kt - 1:]
        p, q = torch.chunk(self.conv(x), 2, dim=1)
        return (p + r) * torch.sigmoid(q)


class _ChebConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int):
        super().__init__()
        self.theta = nn.Parameter(torch.empty(k, c_in, c_out))
        self.b = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, x, sup):                  # x [B, C, T, N], sup [k, N, N]
        h = torch.einsum("knm,bctm->kbctn", sup, x)
        out = torch.einsum("kbctn,kcd->bdtn", h, self.theta)
        return out + self.b[None, :, None, None]


class STGCN(nn.Module):
    def __init__(self, num_nodes: int, horizon: int, t_in: int, sup: torch.Tensor,
                 channels=(1, 32, 64), kt: int = 3, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("sup", sup)
        c0, c1, c2 = channels
        self.b1_t1 = _TemporalGatedConv(c0, c1, kt)
        self.b1_g = _ChebConv(c1, c1, sup.shape[0])
        self.b1_t2 = _TemporalGatedConv(c1, c2, kt)
        self.b1_ln = nn.LayerNorm(num_nodes)
        self.b2_t1 = _TemporalGatedConv(c2, c1, kt)
        self.b2_g = _ChebConv(c1, c1, sup.shape[0])
        self.b2_t2 = _TemporalGatedConv(c1, c2, kt)
        self.b2_ln = nn.LayerNorm(num_nodes)
        self.drop = nn.Dropout(dropout)
        # four temporal convs of width kt consume 4*(kt-1) steps of history
        t_left = t_in - 4 * (kt - 1)
        if t_left < 1:
            raise ValueError(f"history {t_in} too short for kt={kt}")
        self.head_t = nn.Conv2d(c2, c2, (t_left, 1))
        self.head = nn.Conv2d(c2, horizon, (1, 1))
        self.horizon = horizon

    def forward(self, x):                       # [B, T, N]
        h = x.unsqueeze(1)                      # [B, 1, T, N]
        h = self.b1_t2(F.relu(self.b1_g(self.b1_t1(h), self.sup)))
        h = self.drop(self.b1_ln(h))
        h = self.b2_t2(F.relu(self.b2_g(self.b2_t1(h), self.sup)))
        h = self.drop(self.b2_ln(h))
        h = self.head_t(h)                      # [B, C, 1, N]
        return self.head(h).squeeze(2)          # [B, H, N]


# --------------------------------------------------------------------------
# DCRNN
# --------------------------------------------------------------------------

class _DiffusionConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, sup: torch.Tensor, order: int = 2):
        super().__init__()
        self.order = order
        self.register_buffer("sup", sup)
        n_sup = sup.shape[0] * order + 1
        self.lin = nn.Linear(c_in * n_sup, c_out)

    def forward(self, x):                       # [B, N, C]
        out = [x]
        for s in self.sup:
            h = x
            for _ in range(self.order):
                h = torch.einsum("nm,bmc->bnc", s, h)
                out.append(h)
        return self.lin(torch.cat(out, dim=-1))


class _DCGRUCell(nn.Module):
    def __init__(self, c_in: int, hidden: int, sup: torch.Tensor, order: int = 2):
        super().__init__()
        self.hidden = hidden
        self.gates = _DiffusionConv(c_in + hidden, 2 * hidden, sup, order)
        self.cand = _DiffusionConv(c_in + hidden, hidden, sup, order)

    def forward(self, x, h):                    # x [B,N,C], h [B,N,H]
        rz = torch.sigmoid(self.gates(torch.cat([x, h], -1)))
        r, z = torch.chunk(rz, 2, dim=-1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], -1)))
        return z * h + (1 - z) * c


class DCRNN(nn.Module):
    def __init__(self, num_nodes: int, horizon: int, sup: torch.Tensor,
                 hidden: int = 64, layers: int = 2, order: int = 2):
        super().__init__()
        self.N, self.H, self.hidden, self.layers = num_nodes, horizon, hidden, layers
        self.enc = nn.ModuleList([_DCGRUCell(1 if i == 0 else hidden, hidden, sup, order)
                                  for i in range(layers)])
        self.dec = nn.ModuleList([_DCGRUCell(1 if i == 0 else hidden, hidden, sup, order)
                                  for i in range(layers)])
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, y=None, teacher: float = 0.0):
        B, T, N = x.shape
        h = [torch.zeros(B, N, self.hidden, device=x.device) for _ in range(self.layers)]
        for t in range(T):
            inp = x[:, t].unsqueeze(-1)
            for i, cell in enumerate(self.enc):
                h[i] = cell(inp, h[i]); inp = h[i]
        go = torch.zeros(B, N, 1, device=x.device)
        preds = []
        for t in range(self.H):
            inp = go
            for i, cell in enumerate(self.dec):
                h[i] = cell(inp, h[i]); inp = h[i]
            go = self.out(inp)
            preds.append(go.squeeze(-1))
            if self.training and y is not None and teacher > 0 \
                    and float(torch.rand(1)) < teacher:
                go = y[:, t].unsqueeze(-1)
        return torch.stack(preds, dim=1)        # [B, H, N]


# --------------------------------------------------------------------------
# ASTGCN (recent component)
# --------------------------------------------------------------------------

class _SpatialAttention(nn.Module):
    def __init__(self, c: int, t: int, n: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.zeros(t)); self.W2 = nn.Parameter(torch.zeros(c, t))
        self.W3 = nn.Parameter(torch.zeros(c)); self.bs = nn.Parameter(torch.zeros(1, n, n))
        self.Vs = nn.Parameter(torch.zeros(n, n))
        for p in (self.W2, self.bs, self.Vs):
            nn.init.xavier_uniform_(p if p.dim() > 1 else p.unsqueeze(0))

    def forward(self, x):                       # [B, C, T, N]
        lhs = torch.einsum("bctn,t->bcn", x, self.W1)
        lhs = torch.einsum("bcn,ct->btn", lhs, self.W2)
        rhs = torch.einsum("c,bctn->btn", self.W3, x)
        s = torch.matmul(self.Vs, torch.sigmoid(torch.einsum("btn,btm->bnm", lhs, rhs)
                                                + self.bs))
        return torch.softmax(s, dim=-1)


class _TemporalAttention(nn.Module):
    def __init__(self, c: int, t: int, n: int):
        super().__init__()
        self.U1 = nn.Parameter(torch.zeros(n)); self.U2 = nn.Parameter(torch.zeros(c, n))
        self.U3 = nn.Parameter(torch.zeros(c)); self.be = nn.Parameter(torch.zeros(1, t, t))
        self.Ve = nn.Parameter(torch.zeros(t, t))
        nn.init.xavier_uniform_(self.U2); nn.init.xavier_uniform_(self.be)
        nn.init.xavier_uniform_(self.Ve)

    def forward(self, x):                       # [B, C, T, N]
        lhs = torch.einsum("bctn,n->bct", x, self.U1)
        lhs = torch.einsum("bct,cn->btn", lhs, self.U2)
        rhs = torch.einsum("c,bctn->btn", self.U3, x)
        e = torch.matmul(self.Ve, torch.sigmoid(torch.einsum("btn,bsn->bts", lhs, rhs)
                                                + self.be))
        return torch.softmax(e, dim=-1)


class _ASTGCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, t_in: int, n: int, sup: torch.Tensor,
                 t_stride: int = 1):
        super().__init__()
        self.register_buffer("sup", sup)
        self.sat = _SpatialAttention(c_in, t_in, n)
        self.tat = _TemporalAttention(c_in, t_in, n)
        self.theta = nn.Parameter(torch.empty(sup.shape[0], c_in, c_out))
        nn.init.xavier_uniform_(self.theta)
        self.tconv = nn.Conv2d(c_out, c_out, (3, 1), stride=(t_stride, 1), padding=(1, 0))
        self.res = nn.Conv2d(c_in, c_out, (1, 1), stride=(t_stride, 1))
        self.ln = nn.LayerNorm(c_out)

    def forward(self, x):                       # [B, C, T, N]
        e = self.tat(x)
        xt = torch.einsum("bctn,bts->bcsn", x, e)
        s = self.sat(xt)
        h = 0
        for k, sk in enumerate(self.sup):
            adj = sk.unsqueeze(0) * s                       # attention-modulated
            hk = torch.einsum("bnm,bctm->bctn", adj, xt)
            h = h + torch.einsum("bctn,cd->bdtn", hk, self.theta[k])
        h = F.relu(self.tconv(F.relu(h)) + self.res(x))
        return self.ln(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ASTGCN(nn.Module):
    def __init__(self, num_nodes: int, horizon: int, t_in: int, sup: torch.Tensor,
                 channels: int = 64, blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList()
        c_in = 1
        for _ in range(blocks):
            self.blocks.append(_ASTGCNBlock(c_in, channels, t_in, num_nodes, sup))
            c_in = channels
        self.head = nn.Conv2d(t_in, horizon, (1, channels))

    def forward(self, x):                       # [B, T, N]
        h = x.unsqueeze(1)
        for b in self.blocks:
            h = b(h)
        h = h.permute(0, 2, 3, 1)               # [B, T, N, C]
        return self.head(h).squeeze(-1)         # [B, H, N]


# --------------------------------------------------------------------------
# Graph WaveNet
# --------------------------------------------------------------------------

class _GWNetGCN(nn.Module):
    def __init__(self, c_in: int, c_out: int, n_sup: int, order: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.order, self.dropout = order, dropout
        self.mlp = nn.Conv2d(c_in * (order * n_sup + 1), c_out, (1, 1))

    def forward(self, x, sups):                 # x [B, C, N, T]
        out = [x]
        for a in sups:
            h = x
            for _ in range(self.order):
                h = torch.einsum("nm,bcmt->bcnt", a, h)
                out.append(h)
        h = self.mlp(torch.cat(out, dim=1))
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes: int, horizon: int, sup: torch.Tensor,
                 residual: int = 32, dilation: int = 32, skip: int = 256,
                 end: int = 512, blocks: int = 2, layers: int = 2,
                 emb: int = 10, dropout: float = 0.3):
        super().__init__()
        self.register_buffer("sup", sup)
        self.E1 = nn.Parameter(torch.randn(num_nodes, emb) * 0.1)
        self.E2 = nn.Parameter(torch.randn(emb, num_nodes) * 0.1)
        self.start = nn.Conv2d(1, residual, (1, 1))
        # No residual_convs: with the graph convolution enabled its output takes
        # that role, exactly as in the reference implementation.  Keeping them
        # would leave untrained weights in the parameter count the cost table
        # reports.
        self.filt, self.gate, self.skipc, self.gcn, self.bn = (
            nn.ModuleList() for _ in range(5))
        rf = 1
        for _ in range(blocks):
            d = 1
            for _ in range(layers):
                self.filt.append(nn.Conv2d(residual, dilation, (1, 2), dilation=d))
                self.gate.append(nn.Conv2d(residual, dilation, (1, 2), dilation=d))
                self.skipc.append(nn.Conv2d(dilation, skip, (1, 1)))
                self.gcn.append(_GWNetGCN(dilation, residual, sup.shape[0] + 1,
                                          dropout=dropout))
                self.bn.append(nn.BatchNorm2d(residual))
                rf += d; d *= 2
        self.receptive = rf
        self.e1 = nn.Conv2d(skip, end, (1, 1))
        self.e2 = nn.Conv2d(end, horizon, (1, 1))

    def forward(self, x):                       # [B, T, N]
        h = x.permute(0, 2, 1).unsqueeze(1)     # [B, 1, N, T]
        if h.shape[-1] < self.receptive:
            h = F.pad(h, (self.receptive - h.shape[-1], 0))
        h = self.start(h)
        adp = F.softmax(F.relu(torch.mm(self.E1, self.E2)), dim=1)
        sups = list(self.sup) + [adp]
        skip = 0
        for i in range(len(self.filt)):
            res = h
            f = torch.tanh(self.filt[i](h)); g = torch.sigmoid(self.gate[i](h))
            h = f * g
            s = self.skipc[i](h)
            skip = s if isinstance(skip, int) else skip[..., -s.shape[-1]:] + s
            h = self.gcn[i](h, sups)
            h = h + res[..., -h.shape[-1]:]
            h = self.bn[i](h)
        # Only the skip path feeds the head, so the final layer's graph conv and
        # batch norm receive no gradient.  That is the published architecture,
        # not an oversight here -- the reference implementation does the same.
        # Do not "fix" it: it would stop being Graph WaveNet.
        # The head is linear on the output side.  Targets here are z-scored, and
        # 59% of them are negative, so a ReLU on e2 would clamp the majority of
        # the label distribution to zero -- that is what drove the first run to
        # MAE 40 against everyone else's ~10.  The reference implementation ends
        # at end_conv_2 with no activation; so does this.
        h = self.e2(F.relu(self.e1(F.relu(skip))))
        return h[..., -1]                                     # [B, H, N]
