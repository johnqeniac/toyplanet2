"""Diagnostics computed from model output. Pure functions of fields."""

from __future__ import annotations

import numpy as np


def ice_edge_north(x, T, T_f) -> float:
    """sin(latitude) of the northern ice edge: the poleward-most place in the
    northern hemisphere where T crosses T_f, interpolated linearly between band
    centres. Returns 1.0 if the hemisphere is ice-free and 0.0 if it is all ice."""
    x, T = np.asarray(x), np.asarray(T)
    north = x > 0
    xn, Tn = x[north], T[north]
    if np.all(Tn > T_f):
        return 1.0
    if np.all(Tn <= T_f):
        return 0.0
    i = np.where((Tn[:-1] > T_f) & (Tn[1:] <= T_f))[0][-1]
    return float(xn[i] + (T_f - Tn[i]) * (xn[i + 1] - xn[i]) / (Tn[i + 1] - Tn[i]))
