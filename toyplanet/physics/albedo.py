"""Surface albedo with a smooth ice transition.

    alpha(T) = alpha_ice + (alpha_free - alpha_ice) * (1 + tanh((T - T_f) / w)) / 2

Far below T_f the surface is ice-covered (alpha -> alpha_ice); far above it is
ice-free (alpha -> alpha_free); the transition has width about w.

Why a tanh and not a step: a step makes albedo jump when a band crosses T_f,
so the absorbed sunlight jumps, the temperature is pushed back across, and the
band oscillates from one time step to the next. A controller wrapped around
that sees a signal it can never settle. The smooth form also has a derivative
everywhere, which phase 2 needs to linearize the planet.

Why T_f is applied to absolute temperature (deg C), not to an anomaly: whether
ice forms depends on how cold it is, not on how much colder than some baseline.
Toy Planet I applied its threshold to the anomaly.
"""

from __future__ import annotations

import numpy as np


def albedo(T, cfg) -> np.ndarray:
    """Albedo for temperature T (deg C). cfg: AlbedoConfig."""
    s = np.tanh((np.asarray(T, dtype=float) - cfg.t_freeze) / cfg.t_width)
    return cfg.ice + (cfg.free - cfg.ice) * 0.5 * (1.0 + s)


def dalbedo_dT(T, cfg) -> np.ndarray:
    """d(alpha)/dT in K^-1. Negative when ice albedo exceeds ice-free albedo:
    warming removes ice and darkens the surface (the ice-albedo feedback)."""
    s = np.tanh((np.asarray(T, dtype=float) - cfg.t_freeze) / cfg.t_width)
    return (cfg.free - cfg.ice) * 0.5 * (1.0 - s**2) / cfg.t_width
