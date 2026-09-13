"""Dependency-free GeoHash encoding / decoding.

Used to turn continuous GPS coordinates into discrete grid tokens, which is
what makes a trajectory a *sequence of words* that a skip-gram model can be
trained on (Section IV-B of the paper).

Precision guide (approximate cell size at mid latitudes):

    5 -> 4.9 km x 4.9 km
    6 -> 1.2 km x 0.61 km
    7 -> 153 m x 153 m
    8 -> 38 m x 19 m

For city-scale traffic nodes, precision 6-7 is usually the right range: small
enough that a cell corresponds to a recognisable place, large enough that the
vocabulary stays learnable from the available trajectories.
"""
from __future__ import annotations

from typing import Tuple

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_DECODE = {c: i for i, c in enumerate(_BASE32)}


def encode(lat: float, lon: float, precision: int = 7) -> str:
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    out = []
    bit = 0
    ch = 0
    even = True
    while len(out) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2.0
            if lon > mid:
                ch |= 1 << (4 - bit)
                lon_lo = mid
            else:
                lon_hi = mid
        else:
            mid = (lat_lo + lat_hi) / 2.0
            if lat > mid:
                ch |= 1 << (4 - bit)
                lat_lo = mid
            else:
                lat_hi = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            out.append(_BASE32[ch])
            bit = 0
            ch = 0
    return "".join(out)


def decode(gh: str) -> Tuple[float, float]:
    """Return the centre (lat, lon) of a geohash cell."""
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    even = True
    for c in gh:
        cd = _DECODE[c]
        for mask in (16, 8, 4, 2, 1):
            if even:
                mid = (lon_lo + lon_hi) / 2.0
                if cd & mask:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2.0
                if cd & mask:
                    lat_lo = mid
                else:
                    lat_hi = mid
            even = not even
    return (lat_lo + lat_hi) / 2.0, (lon_lo + lon_hi) / 2.0


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. Accepts scalars or numpy arrays."""
    import numpy as np

    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
