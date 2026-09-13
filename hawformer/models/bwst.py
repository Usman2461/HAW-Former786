"""BWST: Band-Wise Wavelet Spatio-Temporal block (Section V).

The central mechanism: the signal is decomposed into wavelet bands *before*
spatial relations are computed, and each band gets its own dynamic
neighbourhood, with a budget that contracts geometrically as frequency rises.

    kappa_j = max(kappa_min, ceil(kappa_0 * gamma^j)),   0 < gamma < 1

The approximation band attends over a wide neighbourhood because slow regional
trend really is coherent over districts; the detail bands are confined to tight
ones because a disturbance visible only in high-frequency content cannot have
propagated far within the observation window.  Setting gamma = 1 gives every
band the same budget and is the ablation that tests this claim.

Implementation note on masking.  The paper writes the masked attention as a
Hadamard product, ``softmax(A * M)``.  Taken literally that is wrong: a zeroed
logit still receives ``exp(0) = 1`` after the softmax, so masked-out neighbours
keep a share of the attention mass.  We implement the intended semantics --
additive ``-inf`` on masked positions before the softmax -- which is what makes
the neighbourhood budget an actual budget.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import HaarBands

NEG_INF = -1e9


def band_budgets(kappa0: int, gamma: float, num_bands: int,
                 kappa_min: int, num_nodes: int) -> List[int]:
    out = []
    for j in range(num_bands):
        k = max(kappa_min, math.ceil(kappa0 * (gamma ** j)))
        out.append(int(min(k, num_nodes)))
    return out


def topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean mask keeping the k largest entries of each row of [..., N, N]."""
    N = scores.shape[-1]
    k = int(min(max(k, 1), N))
    idx = scores.topk(k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter(-1, idx, True)


class ChannelAttention(nn.Module):
    """Balance instantaneous against periodic similarity evidence."""

    def __init__(self, channels: int = 2, hidden: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels)
        )

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        # stack: [B, T, C, N, N]
        desc = stack.mean(dim=(-2, -1))                    # [B, T, C]
        w = torch.softmax(self.mlp(desc), dim=-1)          # [B, T, C]
        return (stack * w[..., None, None]).sum(2)         # [B, T, N, N]


class DynamicLocalAwareness(nn.Module):
    """Per-band short-term pattern similarity and neighbourhood mask.

    The similarity is a learnable weighted cosine over M heads, so the
    effective observation window adapts per timestamp instead of being a fixed
    lag chosen in advance.  A periodicity term then reweights the similarity
    matrices of other timestamps by how similar their calendar position is,
    which recovers the fact that relations recur with the same period as the
    flow that carries them.
    """

    def __init__(self, t_history: int, d_model: int, heads: int = 2,
                 time_dim: int = 16):
        super().__init__()
        self.T = t_history
        self.heads = heads
        self.to_series = nn.Linear(d_model, 1)
        self.w = nn.Parameter(torch.ones(heads, t_history, t_history))
        self.time_proj = nn.Linear(2 * time_dim, time_dim)
        self.chan = ChannelAttention(2)

    def forward(self, xb: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """xb: [B, T, N, d];  time_emb: [B, T, 2*time_dim] -> [B, T, N, N]."""
        B, T, N, _ = xb.shape
        v = self.to_series(xb).squeeze(-1).permute(0, 2, 1)        # [B, N, T]

        w = self.w.unsqueeze(0)                                     # [1,M,T,T]
        vw = v[:, None, None, :, :] * w[:, :, :, None, :]           # [B,M,T,N,T]
        vw = F.normalize(vw, dim=-1, eps=1e-8)
        S = torch.einsum("bmtnk,bmtok->bmtno", vw, vw).mean(1)      # [B,T,N,N]

        te = self.time_proj(time_emb)                               # [B,T,td]
        te = F.normalize(te, dim=-1, eps=1e-8)
        s_tt = torch.bmm(te, te.transpose(1, 2))                    # [B,T,T]
        s_tt = torch.softmax(s_tt, dim=-1)
        P = torch.einsum("btu,bunm->btnm", s_tt, S)                 # [B,T,N,N]

        return self.chan(torch.stack([S, P], dim=2))


class MultiScaleAttention(nn.Module):
    """Node-level attention complemented by group-level keys and values.

    A node's query, key and value are each concatenated with the pooled state
    of the group it belongs to and projected back down, so relations *within* a
    region are strengthened and relations *between* regions are represented in
    the same attention matrix rather than in a separate mechanism.
    """

    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % heads == 0
        self.h = heads
        self.dk = d_model // heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.qc = nn.Linear(d_model, d_model)
        self.kc = nn.Linear(d_model, d_model)
        self.vc = nn.Linear(d_model, d_model)
        self.mq = nn.Linear(2 * d_model, d_model)
        self.mk = nn.Linear(2 * d_model, d_model)
        self.mv = nn.Linear(2 * d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        # Learned-graph prior on the attention logits.  Two reasons for it:
        # it lets the inferred dependency structure shape *which* neighbours
        # attention prefers, not only how features propagate; and it keeps the
        # graph learner differentiable through the predictor even when the
        # Chebyshev branch is switched off, so the `w/o GCN` ablation measures
        # the branch rather than silently severing HAGL from the loss.
        self.graph_bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _pool_scatter(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Pool to groups then scatter back to nodes. x: [B,T,N,d], H: [N,G]."""
        w = H / H.sum(0, keepdim=True).clamp_min(1e-6)      # column-normalised
        g = torch.einsum("btnd,ng->btgd", x, w)             # [B,T,G,d]
        back = H / H.sum(1, keepdim=True).clamp_min(1e-6)   # row-normalised
        return torch.einsum("btgd,ng->btnd", g, back)       # [B,T,N,d]

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        """[B,T,N,d] -> [B*T, heads, N, dk].

        Attention here runs independently per (window, timestamp) over nodes, so
        folding time into the batch dimension is exact and puts the tensors in
        the four-dimensional layout the fused attention kernels require.  The
        earlier [B, heads, T, N, N] layout forced the math backend, which
        materialises the whole score matrix and keeps it for the backward pass:
        twelve of those, at 55 MB each, is what pushed an 8 GB card into host
        memory and turned a 25 ms step into a five-second one.
        """
        B, T, N, D = t.shape
        return t.reshape(B * T, N, self.h, self.dk).transpose(1, 2)

    def forward(self, x: torch.Tensor, H: torch.Tensor,
                mask: Optional[torch.Tensor],
                A: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B,T,N,d];  H: [N,G];  mask: [B,T,N,N] or [N,N] bool;  A: [N,N]."""
        B, T, N, D = x.shape
        c = self._pool_scatter(x, H)
        q = self.mq(torch.cat([self.q(x), self.qc(c)], -1))
        k = self.mk(torch.cat([self.k(x), self.kc(c)], -1))
        v = self.mv(torch.cat([self.v(x), self.vc(c)], -1))

        qh, kh, vh = self._split(q), self._split(k), self._split(v)

        if A is None:
            # No learned-graph prior: the mask is a plain boolean that nothing
            # differentiates through, so the fused kernel can take it.  A static
            # [N,N] mask stays [1,1,N,N] and broadcasts; the per-timestamp mask
            # is already [B,T,N,N] and reshapes for free.
            am = None
            if mask is not None:
                am = (mask.view(1, 1, N, N) if mask.dim() == 2
                      else mask.reshape(B * T, 1, N, N))
            z = F.scaled_dot_product_attention(
                qh, kh, vh, attn_mask=am,
                dropout_p=float(self.drop.p) if self.training else 0.0)
        else:
            # The structural branch adds log1p(A) to the logits, and A is the
            # graph learner's output -- the gradient that reaches HAGL through
            # the predictor runs along this path.  Passing that as SDPA's
            # attn_mask makes it a tensor the fused backward has to differentiate
            # w.r.t., which it does not support: on this card it raised
            # "LSE is not correctly aligned (strideH)" rather than falling back,
            # and a version that merely fell back would have silently severed
            # HAGL from the loss.  So this branch stays explicit.  It is the
            # smaller cost of the two: the dynamic per-timestamp mask, which is
            # what actually makes the score matrix large, is on the branch above.
            att = (qh @ kh.transpose(-2, -1)) / math.sqrt(self.dk)
            att = att + self.graph_bias * torch.log1p(A.clamp_min(0.0))
            if mask is not None:
                m = (mask.view(1, 1, N, N) if mask.dim() == 2
                     else mask.reshape(B * T, 1, N, N))
                att = att.masked_fill(~m, NEG_INF)
            z = self.drop(torch.softmax(att, dim=-1)) @ vh
        z = z.transpose(1, 2).reshape(B, T, N, D)
        return self.out(z)


class ChebConv(nn.Module):
    """Bidirectional Chebyshev propagation over the learned graph.

    Attention selects neighbours by similarity; this propagates along the
    inferred dependency structure itself.  The two are complementary, and the
    ``w/o GCN`` ablation tests whether the second is redundant given the first.
    """

    def __init__(self, d_model: int, K: int = 2):
        super().__init__()
        self.K = K
        self.theta_f = nn.Parameter(torch.randn(K + 1, d_model, d_model) * 0.02)
        self.theta_b = nn.Parameter(torch.randn(K + 1, d_model, d_model) * 0.02)

    @staticmethod
    def _norm(A: torch.Tensor) -> torch.Tensor:
        d = A.sum(-1).clamp_min(1e-9)
        return A / d.unsqueeze(-1)

    def _prop(self, x: torch.Tensor, A: torch.Tensor,
              theta: torch.Tensor) -> torch.Tensor:
        out = x @ theta[0]
        if self.K >= 1:
            t_prev, t_cur = x, torch.einsum("nm,btmd->btnd", A, x)
            out = out + t_cur @ theta[1]
            for k in range(2, self.K + 1):
                t_next = 2 * torch.einsum("nm,btmd->btnd", A, t_cur) - t_prev
                out = out + t_next @ theta[k]
                t_prev, t_cur = t_cur, t_next
        return out

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        Af = self._norm(A)
        Ab = self._norm(A.t())
        return F.relu(self._prop(x, Af, self.theta_f)
                      + self._prop(x, Ab, self.theta_b))


class TemporalAttention(nn.Module):
    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(d_model, heads, dropout=dropout,
                                         batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        o, _ = self.att(h, h, h, need_weights=False)
        return o.view(B, N, T, D).permute(0, 2, 1, 3)


class BWSTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        t_history: int,
        num_bands: int,
        heads: int = 4,
        sim_heads: int = 2,
        cheb_k: int = 2,
        dropout: float = 0.1,
        time_dim: int = 16,
        use_gcn: bool = True,
        use_dla: bool = True,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.use_gcn = use_gcn
        self.use_dla = use_dla

        self.dla = nn.ModuleList(
            [DynamicLocalAwareness(t_history, d_model, sim_heads, time_dim)
             for _ in range(num_bands)]) if use_dla else None
        self.sms = nn.ModuleList(
            [MultiScaleAttention(d_model, heads, dropout) for _ in range(num_bands)])
        self.gms = nn.ModuleList(
            [MultiScaleAttention(d_model, heads, dropout) for _ in range(num_bands)])
        self.gcn = nn.ModuleList(
            [ChebConv(d_model, cheb_k) for _ in range(num_bands)]) if use_gcn else None

        fuse_in = (3 if use_gcn else 2) * d_model
        self.fuse = nn.ModuleList(
            [nn.Linear(fuse_in, d_model) for _ in range(num_bands)])
        self.band_logits = nn.Parameter(torch.zeros(num_bands))

        self.tsa = TemporalAttention(d_model, heads, dropout)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        bands: List[torch.Tensor],
        A: torch.Tensor,
        H_s: torch.Tensor,
        H_g: torch.Tensor,
        geo_mask: torch.Tensor,
        budgets: List[int],
        time_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        outs = []
        diag: Dict[str, torch.Tensor] = {}
        for j, xb in enumerate(bands):
            if self.use_dla:
                S = self.dla[j](xb, time_emb)                # [B,T,N,N]
                m = topk_mask(S, budgets[j])
            else:
                m = geo_mask
            sms = self.sms[j](xb, H_s, m)
            gms = self.gms[j](xb, H_g, geo_mask, A)   # structural branch sees A
            parts = [sms, gms]
            if self.use_gcn:
                parts.append(self.gcn[j](xb, A))
            outs.append(self.fuse[j](torch.cat(parts, -1)))
            if j == 0:
                diag["mask_density_band0"] = (
                    m.float().mean() if m.dtype == torch.bool else m.mean())

        omega = torch.softmax(self.band_logits, 0)
        diag["band_weights"] = omega.detach()
        spa = sum(omega[j] * outs[j] for j in range(self.num_bands))

        x_in = sum(bands)                                     # perfect reconstruction
        tem = self.tsa(x_in)
        h = self.merge(torch.cat([spa, tem], -1))
        h = self.norm1(x_in + self.drop(h))
        h = self.norm2(h + self.drop(self.ffn(h)))
        return h, diag


class SinglePassDecoder(nn.Module):
    """Emit all horizon steps in one forward pass.

    Autoregressive decoding accumulates error and puts sequential latency on
    the deployment path; a direct multi-step head avoids both.  The start token
    is the latter half of the encoder output, so generation is conditioned on
    recent context rather than starting from an uninformative state.
    """

    def __init__(self, d_model: int, t_history: int, t_horizon: int,
                 heads: int = 4, dropout: float = 0.1, token_len: int = 6):
        super().__init__()
        self.tp = t_horizon
        self.token_len = min(token_len, t_history)
        self.query = nn.Parameter(torch.randn(t_horizon, d_model) * 0.02)
        self.self_att = nn.MultiheadAttention(d_model, heads, dropout=dropout,
                                              batch_first=True)
        self.cross_att = nn.MultiheadAttention(d_model, heads, dropout=dropout,
                                               batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """memory: [B, T, N, d] -> [B, T_p, N]"""
        B, T, N, D = memory.shape
        mem = memory.permute(0, 2, 1, 3).reshape(B * N, T, D)
        tok = mem[:, -self.token_len:]
        q = self.query.unsqueeze(0).expand(B * N, -1, -1)
        dec = torch.cat([tok, q], dim=1)

        causal = torch.triu(
            torch.ones(dec.shape[1], dec.shape[1], device=dec.device, dtype=torch.bool),
            diagonal=1)
        h, _ = self.self_att(dec, dec, dec, attn_mask=causal, need_weights=False)
        dec = self.norm1(dec + h)
        h, _ = self.cross_att(dec, mem, mem, need_weights=False)
        dec = self.norm2(dec + h)
        dec = self.norm3(dec + self.conv(dec.transpose(1, 2)).transpose(1, 2))

        out = self.head(dec[:, -self.tp:]).squeeze(-1)        # [B*N, T_p]
        return out.view(B, N, self.tp).permute(0, 2, 1)
