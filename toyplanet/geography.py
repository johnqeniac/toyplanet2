"""Surface geography: land fraction by latitude.

EARTH_LAND_10DEG gives the approximate fraction of land in each 10-degree
latitude band, south to north. These values are APPROXIMATE, written from
general knowledge of the zonal land distribution rather than computed from a
land mask; they reproduce Earth's ~29% land area and its main features
(Antarctica, the Southern Ocean gap, the northern land belt, the Arctic
Ocean). They should be replaced by values computed from a land-mask dataset.

land_fraction() averages the table over each model band, by area.
"""

from __future__ import annotations

import numpy as np

# 10-degree bins from 90S-80S up to 80N-90N (18 values). APPROXIMATE.
EARTH_LAND_10DEG = np.array([
    1.00, 0.75, 0.25,          # 90-60 S: Antarctica
    0.01, 0.03, 0.12,          # 60-30 S: Southern Ocean
    0.23, 0.22, 0.24,          # 30-0  S
    0.23, 0.25, 0.38,          # 0-30  N
    0.43, 0.45, 0.57,          # 30-60 N
    0.70, 0.45, 0.20,          # 60-90 N: land belt, then Arctic Ocean
])


def land_fraction(grid, profile: str) -> np.ndarray:
    """Land fraction per band for 'aquaplanet' (all ocean) or 'earth'."""
    if profile == "aquaplanet":
        return np.zeros(grid.n)
    if profile != "earth":
        raise ValueError(f"unknown land profile {profile!r}")
    # Area-weighted average over each band: sample uniformly in x = sin(lat).
    sub = 200
    frac = (np.arange(sub) + 0.5) / sub
    x = grid.x_edges[:-1, None] + frac[None, :] * grid.dx[:, None]
    lat = np.degrees(np.arcsin(x))
    idx = np.clip(((lat + 90.0) // 10.0).astype(int), 0, 17)
    return EARTH_LAND_10DEG[idx].mean(axis=1)
