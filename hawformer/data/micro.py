"""Microscopic branch: trajectories -> node transition graph.

Pipeline (Section IV-B of the paper):

  matched trajectories -> GeoHash tokens -> skip-gram embeddings
  -> per-node embedding -> pairwise similarity -> normalised A_mi

Two implementation notes worth stating plainly, because both depart from a
literal reading of the equations:

1.  The formulation writes the entry as the *Euclidean distance* between node
    embeddings.  Used directly, that gives large weights to unrelated nodes and
    near-zero weights to identical ones, which is the wrong sign for an
    adjacency.  We apply a Gaussian kernel to the distance,
    ``exp(-d^2 / sigma^2)``, with ``sigma`` set to the median pairwise
    distance.  This preserves the intended ordering and keeps the matrix
    scale-free across datasets.

2.  Embeddings are trained once on the training-split corpus and the resulting
    A_mi is held fixed.  A per-window rebuild -- closer to "real-time" -- is a
    straightforward extension, but it makes the fused graph sample-dependent,
    which in turn makes the balanced partition sample-dependent and breaks the
    single-hierarchy design.  The hook is `build_micro_graph(..., corpus_slice)`:
    pass a time-filtered corpus to rebuild for a period of interest.

Nodes whose cell does not occur in the corpus are out of vocabulary; their rows
and columns are left at zero so that *absence of evidence* is not read as
evidence of independence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils import get_logger

LOG = get_logger()


def read_corpus(path: str, limit: Optional[int] = None) -> List[List[str]]:
    sents: List[List[str]] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            toks = line.split()
            if len(toks) >= 2:
                sents.append(toks)
    return sents


def train_embeddings(
    corpus_path: str,
    dim: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 5,
    workers: int = 4,
    seed: int = 42,
    limit: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Train skip-gram embeddings over grid-token sequences."""
    try:
        from gensim.models import Word2Vec
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "gensim is required for the microscopic branch. "
            "pip install gensim, or set model.use_micro: false"
        ) from e

    sents = read_corpus(corpus_path, limit=limit)
    if not sents:
        LOG.warning("empty trajectory corpus; microscopic branch unavailable")
        return {}
    LOG.info("training skip-gram on %d sequences (dim=%d, window=%d)",
             len(sents), dim, window)
    model = Word2Vec(
        sentences=sents, vector_size=dim, window=window, min_count=min_count,
        sg=1, negative=10, epochs=epochs, workers=workers, seed=seed,
    )
    return {w: model.wv[w].astype(np.float64) for w in model.wv.index_to_key}


def node_embeddings(
    node_cells: Sequence[str],
    vectors: Dict[str, np.ndarray],
    dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map each node to its cell embedding. Returns (E [N,dim], in_vocab [N])."""
    N = len(node_cells)
    E = np.zeros((N, dim), dtype=np.float64)
    ok = np.zeros(N, dtype=bool)
    for i, c in enumerate(node_cells):
        v = vectors.get(str(c))
        if v is not None:
            E[i] = v
            ok[i] = True
    if ok.sum() < N:
        LOG.warning("%d/%d nodes are out of vocabulary in the corpus", N - int(ok.sum()), N)
    return E, ok


def build_micro_graph(
    node_cells: Sequence[str],
    corpus_path: str,
    dim: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 5,
    seed: int = 42,
    corpus_limit: Optional[int] = None,
    self_loops: bool = True,
) -> np.ndarray:
    """Return the symmetric normalised microscopic transition graph [N, N]."""
    N = len(node_cells)
    vectors = train_embeddings(
        corpus_path, dim=dim, window=window, min_count=min_count,
        epochs=epochs, seed=seed, limit=corpus_limit,
    )
    if not vectors:
        return np.zeros((N, N), dtype=np.float32)

    E, ok = node_embeddings(node_cells, vectors, dim)
    if ok.sum() < 2:
        LOG.warning("fewer than two in-vocabulary nodes; A_mi is empty")
        return np.zeros((N, N), dtype=np.float32)

    # pairwise Euclidean distance in embedding space
    sq = (E ** 2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * E @ E.T, 0.0)
    d = np.sqrt(d2)

    valid = np.outer(ok, ok)
    off = d[valid & ~np.eye(N, dtype=bool)]
    sigma = float(np.median(off)) if off.size else 1.0
    sigma = sigma if sigma > 1e-9 else 1.0

    S = np.exp(-d2 / (sigma ** 2))
    S[~valid] = 0.0
    np.fill_diagonal(S, 1.0 if self_loops else 0.0)
    S[~ok, :] = 0.0
    S[:, ~ok] = 0.0
    if self_loops:
        S[np.arange(N), np.arange(N)] = np.where(ok, 1.0, 0.0)

    A = sym_normalise(S)
    LOG.info("A_mi built: density %.3f, sigma %.3f", (A > 1e-6).mean(), sigma)
    return A.astype(np.float32)


def sym_normalise(A: np.ndarray) -> np.ndarray:
    """D^{-1/2} A D^{-1/2} with zero-degree nodes left at zero."""
    deg = A.sum(1)
    dinv = np.zeros_like(deg)
    nz = deg > 1e-12
    dinv[nz] = deg[nz] ** -0.5
    return (A * dinv[:, None]) * dinv[None, :]
