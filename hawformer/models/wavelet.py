"""Differentiable multi-level Haar decomposition along the time axis.

The decomposition sits *inside* the network -- it is applied to the embedded
sequence, not to the raw input -- so it has to be differentiable and to run on
the training device.  It is therefore implemented directly in PyTorch rather
than by calling out to a wavelet library.

Two properties are relied on elsewhere in the model:

*   **Perfect reconstruction.**  Each band is reconstructed to the original
    length by zeroing the coefficients of the other bands, so the bands sum
    exactly to the input.  Attention operates on aligned time indices, and the
    residual carries whatever the finite level count does not resolve.

*   **Compact support.**  Haar has the shortest support of any orthogonal
    wavelet, so a transient is confined to few coefficients.  That matters
    because the sub-band is about to be used to *define a neighbourhood*: a
    basis that smeared transients across time would smear them across space
    too.

Odd-length inputs are handled by symmetric padding at each level, and the
padding is removed on reconstruction, so any sequence length is accepted.
"""
from __future__ import annotations

from typing import List

import math
import torch
import torch.nn as nn


class HaarBands(nn.Module):
    """Split [B, T, ...] into ``levels + 1`` time-aligned bands.

    Output ordering is ``[approximation, detail_1, ..., detail_L]`` where
    ``detail_1`` is the *finest* (highest-frequency) band.  Index 0 is the
    approximation throughout the codebase, matching the paper's ``j = 0``.
    """

    def __init__(self, levels: int = 2):
        super().__init__()
        self.levels = int(levels)
        s = 1.0 / math.sqrt(2.0)
        self.register_buffer("lo", torch.tensor([s, s]), persistent=False)
        self.register_buffer("hi", torch.tensor([s, -s]), persistent=False)

    # -- single-level analysis / synthesis on the last axis ------------------
    def _dwt1(self, x: torch.Tensor):
        """x: [*, T] -> (approx [*, ceil(T/2)], detail, was_padded)"""
        T = x.shape[-1]
        pad = T % 2
        if pad:
            x = torch.cat([x, x[..., -1:]], dim=-1)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        s = 1.0 / math.sqrt(2.0)
        return (even + odd) * s, (even - odd) * s, pad

    def _idwt1(self, a: torch.Tensor, d: torch.Tensor, pad: int) -> torch.Tensor:
        s = 1.0 / math.sqrt(2.0)
        even = (a + d) * s
        odd = (a - d) * s
        out = torch.stack([even, odd], dim=-1).flatten(-2)
        if pad:
            out = out[..., :-1]
        return out

    def forward(self, x: torch.Tensor, time_dim: int = 1) -> List[torch.Tensor]:
        """Return ``levels + 1`` tensors with the same shape as ``x``."""
        xt = x.movedim(time_dim, -1)
        coeffs = []
        pads = []
        cur = xt
        for _ in range(self.levels):
            a, d, p = self._dwt1(cur)
            coeffs.append(d)
            pads.append(p)
            cur = a
        approx = cur

        bands = []
        # approximation band: keep approx, zero every detail
        rec = approx
        for lv in range(self.levels - 1, -1, -1):
            rec = self._idwt1(rec, torch.zeros_like(coeffs[lv]), pads[lv])
        bands.append(rec)

        # detail bands, finest first
        for k in range(self.levels):
            rec = torch.zeros_like(approx)
            for lv in range(self.levels - 1, -1, -1):
                d = coeffs[lv] if lv == k else torch.zeros_like(coeffs[lv])
                rec = self._idwt1(rec, d, pads[lv])
            bands.append(rec)

        return [b.movedim(-1, time_dim) for b in bands]

    @property
    def num_bands(self) -> int:
        return self.levels + 1


@torch.no_grad()
def band_energies(x: torch.Tensor, levels: int = 2) -> torch.Tensor:
    """Normalised per-series band energy, used as a clustering descriptor.

    ``x``: [T, N].  Returns [N, levels + 1] with rows summing to one.  Two
    sensors with the same mean and variance can differ sharply in how their
    energy is distributed across timescales -- an arterial with stop-and-go
    oscillation versus a smooth ring road -- and that difference is exactly
    what the semantic grouping should be able to see.
    """
    hb = HaarBands(levels).to(x.device)
    bands = hb(x.unsqueeze(0), time_dim=1)                 # each [1, T, N]
    e = torch.stack([(b ** 2).sum(dim=1).squeeze(0) for b in bands], dim=-1)
    return e / e.sum(-1, keepdim=True).clamp_min(1e-12)    # [N, J+1]
