"""Regional structure: semantic groups, structural groups, inter-group affinity.

This is the mesoscopic scale of Section IV-C.  Two decisions here are what make
the hierarchy *behavioural* rather than geographic:

*   The structural partition is computed on the **learned** dependency graph
    from the previous alternating stage, not on the predefined adjacency.  As
    the graph comes to connect two physically disjoint corridors carrying the
    same regime, the next refresh places them in the same part.

*   Semantic membership is **soft** (mixture responsibilities).  A sensor with
    an intermediate profile contributes to, and receives from, more than one
    group -- which matters most for the arterials that straddle districts, and
    those are the sensors a hard assignment handles worst.

Partitioning uses METIS when `pymetis` is installed and otherwise falls back to
a spectral + balanced-KMeans partitioner implemented here, so the code has no
hard dependency on a C library.  The fallback is deterministic given a seed and
produces parts within one node of equal size, which is the property that keeps
group-level pooling statistically stable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..utils import get_logger

LOG = get_logger()


@dataclass
class Hierarchy:
    H_s: np.ndarray      # [N, P] soft (or hard) semantic membership
    H_g: np.ndarray      # [N, Q] hard structural membership, one-hot
    P: int
    Q: int


# ---------------------------------------------------------------------------
# Semantic grouping
# ---------------------------------------------------------------------------

def semantic_descriptor(
    flow: np.ndarray,
    band_energy: Optional[np.ndarray] = None,
) -> np.ndarray:
    """[T, N] flow (+ optional [N, J+1] band energies) -> [N, F] descriptor."""
    z = np.stack([flow.mean(0), np.median(flow, 0), flow.std(0)], axis=-1)
    if band_energy is not None:
        z = np.concatenate([z, band_energy], axis=-1)
    mu, sd = z.mean(0, keepdims=True), z.std(0, keepdims=True)
    return (z - mu) / np.maximum(sd, 1e-8)


def semantic_groups(
    descriptor: np.ndarray,
    P: int,
    soft: bool = True,
    seed: int = 42,
    temperature: float = 1.0,
    reg_covar: float = 1e-2,
) -> np.ndarray:
    """Gaussian-mixture grouping. Returns [N, P] responsibilities.

    Two parameters exist because raw GMM responsibilities saturate.  With a
    small node set and a well-separated descriptor, a diagonal mixture becomes
    numerically certain -- off-argmax responsibilities of order 1e-14 -- and
    "soft" membership degenerates into hard assignment, which would make the
    graded-membership mechanism a no-op without anything in the logs saying so.

    `reg_covar` puts a floor under component variances, which is the standard
    remedy for overconfident near-singular components.  `temperature` > 1
    additionally flattens the responsibilities.  Both leave the ranking of
    components untouched; they only affect how much mass a boundary sensor
    keeps in its second-best group, which is exactly what this mechanism is
    supposed to model.  `saturation` is logged so the degenerate case is
    visible rather than silent.
    """
    from sklearn.mixture import GaussianMixture

    N = descriptor.shape[0]
    P = int(min(max(P, 1), N))
    gm = GaussianMixture(
        n_components=P, covariance_type="diag", random_state=seed,
        reg_covar=float(reg_covar), n_init=3, max_iter=200,
    ).fit(descriptor)

    logp = gm._estimate_weighted_log_prob(descriptor)          # [N, P]
    if temperature != 1.0:
        logp = logp / max(float(temperature), 1e-6)
    logp = logp - logp.max(1, keepdims=True)
    R = np.exp(logp)
    R = R / R.sum(1, keepdims=True).clip(1e-12)

    second = np.sort(R, axis=1)[:, -2] if P > 1 else np.zeros(N)
    LOG.info("semantic groups: P=%d, mean second-best responsibility %.4f "
             "(0 = fully hard)", P, float(second.mean()))

    if not soft:
        hard = np.zeros_like(R)
        hard[np.arange(N), R.argmax(1)] = 1.0
        R = hard
    # Drop groups that ended up empty, so pooling never divides by zero.
    keep = R.sum(0) > 1e-6
    if keep.sum() < R.shape[1]:
        LOG.info("dropping %d empty semantic groups", int((~keep).sum()))
        R = R[:, keep]
        R = R / R.sum(1, keepdims=True).clip(1e-12)
    return R.astype(np.float32)


def suggest_P(descriptor: np.ndarray, lo: int = 2, hi: int = 12, seed: int = 42) -> int:
    """Silhouette-based initialisation for the semantic group count."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    N = descriptor.shape[0]
    hi = min(hi, max(lo, N - 1))
    best, best_s = lo, -1.0
    for p in range(lo, hi + 1):
        lab = KMeans(n_clusters=p, n_init=5, random_state=seed).fit_predict(descriptor)
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(descriptor, lab)
        if s > best_s:
            best, best_s = p, s
    LOG.info("silhouette suggests P=%d (score %.3f)", best, best_s)
    return best


# ---------------------------------------------------------------------------
# Structural grouping
# ---------------------------------------------------------------------------

def _try_metis(adj: np.ndarray, Q: int) -> Optional[np.ndarray]:
    try:
        import pymetis
    except ImportError:
        return None
    N = adj.shape[0]
    xadj, adjncy, eweights = [0], [], []
    scale = 1000.0 / max(adj.max(), 1e-9)
    for i in range(N):
        nb = np.nonzero(adj[i])[0]
        nb = nb[nb != i]
        adjncy.extend(nb.tolist())
        eweights.extend(np.maximum((adj[i, nb] * scale).astype(int), 1).tolist())
        xadj.append(len(adjncy))
    if len(adjncy) == 0:
        return None
    _, parts = pymetis.part_graph(Q, xadj=xadj, adjncy=adjncy, eweights=eweights)
    return np.asarray(parts, dtype=np.int64)


def _spectral_balanced(adj: np.ndarray, Q: int, seed: int = 42) -> np.ndarray:
    """Spectral embedding + KMeans, then greedy rebalancing to equal sizes."""
    from sklearn.cluster import KMeans

    N = adj.shape[0]
    A = 0.5 * (adj + adj.T)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(1)
    dinv = np.zeros_like(deg)
    nz = deg > 1e-12
    dinv[nz] = deg[nz] ** -0.5
    L = np.eye(N) - (A * dinv[:, None]) * dinv[None, :]
    k = min(max(Q, 2), N - 1)
    w, v = np.linalg.eigh(L)
    emb = v[:, 1:k + 1]                       # skip the trivial eigenvector
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(nrm, 1e-12)

    lab = KMeans(n_clusters=Q, n_init=10, random_state=seed).fit_predict(emb)
    centers = np.stack([emb[lab == q].mean(0) if (lab == q).any() else emb.mean(0)
                        for q in range(Q)])

    # greedy rebalance: repeatedly move the node that loses least by moving
    cap = int(np.ceil(N / Q))
    for _ in range(4 * N):
        sizes = np.bincount(lab, minlength=Q)
        over = np.where(sizes > cap)[0]
        under = np.where(sizes < cap)[0]
        if over.size == 0 or under.size == 0:
            break
        src = over[np.argmax(sizes[over])]
        members = np.where(lab == src)[0]
        d_src = ((emb[members] - centers[src]) ** 2).sum(1)
        d_dst = ((emb[members][:, None, :] - centers[under][None]) ** 2).sum(-1)
        best_dst = d_dst.min(1)
        gain = best_dst - d_src
        i = int(np.argmin(gain))
        lab[members[i]] = under[int(np.argmin(d_dst[i]))]
    return lab.astype(np.int64)


def structural_groups(adj: np.ndarray, Q: int, seed: int = 42) -> np.ndarray:
    """Balanced partition of the learned graph. Returns one-hot [N, Q]."""
    N = adj.shape[0]
    Q = int(min(max(Q, 1), N))
    if Q == 1:
        return np.ones((N, 1), dtype=np.float32)
    lab = _try_metis(adj, Q)
    if lab is None:
        lab = _spectral_balanced(adj, Q, seed)
    H = np.zeros((N, Q), dtype=np.float32)
    H[np.arange(N), lab] = 1.0
    keep = H.sum(0) > 0
    if keep.sum() < Q:
        H = H[:, keep]
    return H


# ---------------------------------------------------------------------------
# Top-level refresh
# ---------------------------------------------------------------------------

def sparsify_topk(A: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest entries of each row; zero the rest."""
    N = A.shape[0]
    k = int(min(max(k, 1), N))
    idx = np.argpartition(-A, k - 1, axis=1)[:, :k]
    out = np.zeros_like(A)
    rows = np.arange(N)[:, None]
    out[rows, idx] = A[rows, idx]
    return out


def build_hierarchy(
    flow_train: np.ndarray,
    A_star: np.ndarray,
    P: int,
    Q: int,
    wavelet_levels: int = 2,
    use_spectral_descriptor: bool = True,
    soft_membership: bool = True,
    topk: int = 20,
    seed: int = 42,
    membership_temperature: float = 1.0,
    reg_covar: float = 1e-2,
) -> Hierarchy:
    """Recompute (H_s, H_g) from the current learned graph and training flow."""
    import torch
    from .wavelet import band_energies

    be = None
    if use_spectral_descriptor:
        be = band_energies(
            torch.as_tensor(flow_train, dtype=torch.float32), wavelet_levels
        ).cpu().numpy()
    desc = semantic_descriptor(flow_train, be)
    H_s = semantic_groups(desc, P, soft=soft_membership, seed=seed,
                          temperature=membership_temperature,
                          reg_covar=reg_covar)

    A = sparsify_topk(0.5 * (A_star + A_star.T), topk)
    H_g = structural_groups(A, Q, seed=seed)

    LOG.info("hierarchy refreshed: P=%d Q=%d (soft=%s, spectral_desc=%s)",
             H_s.shape[1], H_g.shape[1], soft_membership, use_spectral_descriptor)
    return Hierarchy(H_s=H_s, H_g=H_g, P=H_s.shape[1], Q=H_g.shape[1])
