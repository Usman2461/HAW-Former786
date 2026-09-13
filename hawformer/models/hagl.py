"""HAGL: Hierarchy-Aware Adaptive Graph Learning (Section IV).

Three sources of evidence, fused by a per-entry gate:

  macroscopic  attribute evidence -- a masked low-order edge-weight learner
               refines relations that already exist, a high-order attention
               learner discovers relations that do not
  microscopic  movement evidence  -- a fixed transition graph built offline
               from trajectory embeddings (see data/micro.py)
  mesoscopic   regional evidence  -- a cluster-affinity graph built from the
               current hierarchy with learnable inter-group affinities

The gate is a channel-wise softmax rather than a sigmoid: with three sources
the weights should lie on the simplex, and softmax keeps any one source from
dominating before the others have trained.  The three sources are informative
in different places -- trajectory evidence is strong where traffic is dense and
absent where it is sparse -- so the weights are per entry, not global.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sym_normalise(A: torch.Tensor) -> torch.Tensor:
    deg = A.sum(-1)
    dinv = torch.where(deg > 1e-12, deg.clamp_min(1e-12) ** -0.5,
                       torch.zeros_like(deg))
    return A * dinv.unsqueeze(-1) * dinv.unsqueeze(-2)


class LowOrderLearner(nn.Module):
    """Refine the weights of relations that already exist in the support.

    The mask is the point of this branch: it is confined to the support of the
    predefined adjacency, so it cannot invent edges.  Discovery is the job of
    the high-order branch, and keeping the two separate is what lets the
    ablation attribute a gain to one or the other.
    """

    def __init__(self, num_nodes: int, dim: int = 16):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, dim) * 0.05)
        self.E2 = nn.Parameter(torch.randn(num_nodes, dim) * 0.05)
        self.lam = nn.Parameter(torch.zeros(num_nodes))
        self.w_l = nn.Parameter(torch.tensor(0.0))
        self.gate = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        support = (A > 0).float()
        raw = self.E1 @ self.E2.t() - self.E2 @ self.E1.t()
        A_w = F.relu(raw + torch.diag(self.lam)) * support
        alpha = torch.sigmoid(self.gate(torch.stack([A, A_w])[None]))[0, 0]
        A1 = alpha * A + (1.0 - alpha) * A_w
        return F.relu(sym_normalise(A1) - self.w_l)


class HighOrderLearner(nn.Module):
    """Discover relations absent from the support, by attribute similarity.

    Attention is masked *away from* the low-order support so that this branch
    cannot simply relearn what the low-order branch already holds; it is
    pushed towards sensors that are functionally equivalent but physically
    disconnected -- two entrances to the same district, two ring segments
    carrying the same commuting pattern.
    """

    def __init__(self, in_steps: int, dim: int = 32):
        super().__init__()
        self.proj = nn.Conv1d(in_steps, dim, kernel_size=1)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, x: torch.Tensor, A_L: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N] -> node features [N, dim] averaged over the batch
        h = self.proj(x.mean(0).unsqueeze(0)).squeeze(0).t()      # [N, dim]
        q, k, v = self.q(h), self.k(h), self.v(h)
        att = (q @ k.t()) / (self.dim ** 0.5)
        att = att.masked_fill(A_L > 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = torch.nan_to_num(att, nan=0.0)
        z = att @ v                                                # [N, dim]
        A_H = F.relu(z @ z.t())
        return A_H / A_H.sum(-1, keepdim=True).clamp_min(1e-9)


class MesoAffinity(nn.Module):
    """Cluster-affinity graph with learnable inter-group relations.

    ``A_hi = norm( H_s softplus(L_s) H_s^T + H_g softplus(L_g) H_g^T )``

    Setting ``L = I`` and hardening ``H_s`` recovers a plain same-group
    indicator, which is the ``Lambda = I`` ablation.  Off-diagonal entries let
    a residential group and the corridor group that drains it exchange
    information without being merged into one.

    The product is evaluated in factored form, so nothing of size N x N is
    built from the group side; cost is O(N(P^2 + Q^2)).
    """

    def __init__(self, P: int, Q: int, identity_affinity: bool = False):
        super().__init__()
        self.identity_affinity = identity_affinity
        self.L_s = nn.Parameter(torch.zeros(P, P))
        self.L_g = nn.Parameter(torch.zeros(Q, Q))
        with torch.no_grad():
            self.L_s.copy_(torch.eye(P) * 2.0)
            self.L_g.copy_(torch.eye(Q) * 2.0)

    def resize(self, P: int, Q: int, device) -> None:
        """Group counts can change at a refresh; keep the shared scale."""
        if self.L_s.shape[0] != P:
            self.L_s = nn.Parameter(torch.eye(P, device=device) * 2.0)
        if self.L_g.shape[0] != Q:
            self.L_g = nn.Parameter(torch.eye(Q, device=device) * 2.0)

    def forward(self, H_s: torch.Tensor, H_g: torch.Tensor) -> torch.Tensor:
        if self.identity_affinity:
            Ls = torch.eye(H_s.shape[1], device=H_s.device)
            Lg = torch.eye(H_g.shape[1], device=H_g.device)
        else:
            Ls = F.softplus(0.5 * (self.L_s + self.L_s.t()))
            Lg = F.softplus(0.5 * (self.L_g + self.L_g.t()))
        A = (H_s @ Ls) @ H_s.t() + (H_g @ Lg) @ H_g.t()
        return A / A.sum(-1, keepdim=True).clamp_min(1e-9)

    def l1(self) -> torch.Tensor:
        if self.identity_affinity:
            return torch.zeros((), device=self.L_s.device)
        return F.softplus(self.L_s).abs().sum() + F.softplus(self.L_g).abs().sum()


class HAGL(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_steps: int,
        P: int,
        Q: int,
        low_dim: int = 16,
        high_dim: int = 32,
        topk: int = 20,
        use_micro: bool = True,
        use_meso: bool = True,
        use_high: bool = True,
        identity_affinity: bool = False,
    ):
        super().__init__()
        self.N = num_nodes
        self.topk = topk
        self.use_micro = use_micro
        self.use_meso = use_meso
        self.use_high = use_high

        self.low = LowOrderLearner(num_nodes, low_dim)
        self.high = HighOrderLearner(in_steps, high_dim) if use_high else None
        self.meso = MesoAffinity(P, Q, identity_affinity) if use_meso else None

        n_ch = 1 + int(use_micro) + int(use_meso)
        self.gate = nn.Conv2d(n_ch, n_ch, kernel_size=1)
        self.n_ch = n_ch

    def forward(
        self,
        x: torch.Tensor,          # [B, T, N] normalised flow
        A_pre: torch.Tensor,      # [N, N] predefined adjacency
        A_mi: Optional[torch.Tensor],
        H_s: Optional[torch.Tensor],
        H_g: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        A_L = self.low(A_pre)
        A_ma = A_L
        if self.high is not None:
            A_ma = F.relu(A_L + self.high(x, A_L))

        chans = [A_ma]
        if self.use_micro:
            chans.append(A_mi if A_mi is not None else torch.zeros_like(A_ma))
        A_hi = None
        if self.use_meso and H_s is not None and H_g is not None:
            A_hi = self.meso(H_s, H_g)
            chans.append(A_hi)

        stacked = torch.stack(chans)[None]                    # [1, C, N, N]
        beta = torch.softmax(self.gate(stacked), dim=1)[0]    # [C, N, N]
        A_star = (beta * torch.stack(chans)).sum(0)

        return {
            "A_star": A_star,
            "A_ma": A_ma,
            "A_hi": A_hi,
            "beta": beta.detach(),
        }

    def sparsify(self, A: torch.Tensor) -> torch.Tensor:
        k = int(min(max(self.topk, 1), A.shape[-1]))
        val, idx = torch.topk(A, k, dim=-1)
        out = torch.zeros_like(A)
        return out.scatter(-1, idx, val)

    def regularisation(self, A_star: torch.Tensor,
                       lam_a: float, lam_c: float) -> torch.Tensor:
        reg = lam_a * A_star.abs().sum()
        if self.meso is not None:
            reg = reg + lam_c * self.meso.l1()
        return reg
