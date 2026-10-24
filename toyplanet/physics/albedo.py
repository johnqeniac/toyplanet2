"""Surface albedo with a smooth ice transition and a latitude-dependent ice-free value.

    alpha(T, x) = alpha_ice + (alpha_free(x) - alpha_ice) * (1 + tanh((T - T_f) / w)) / 2
    alpha_free(x) = alpha_free + a2 * P2(x),   P2(x) = (3 x^2 - 1) / 2,   x = sin(latitude)

Far below T_f the surface is ice-covered (alpha -> alpha_ice); far above it is
ice-free; the transition has width about w.

The P2 term makes ice-free surfaces reflect more where the sun is low (high
latitudes) than where it is high (tropics), mainly through clouds and the
reflectivity of water at grazing angles. With a2 > 0 the albedo rises from
alpha_free - a2/2 at the equator to alpha_free + a2 at the poles. This is the
form used by North & Coakley (1979).

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


def _p2(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (3.0 * x**2 - 1.0)


def albedo(T, cfg, x=0.0) -> np.ndarray:
    """Albedo for temperature T (deg C) at x = sin(latitude). cfg: AlbedoConfig."""
    free = cfg.free + cfg.free_p2 * _p2(x)
    s = np.tanh((np.asarray(T, dtype=float) - cfg.t_freeze) / cfg.t_width)
    return cfg.ice + (free - cfg.ice) * 0.5 * (1.0 + s)


def dalbedo_dT(T, cfg, x=0.0) -> np.ndarray:
    """d(alpha)/dT in K^-1. Negative when ice is brighter than ice-free surface:
    warming removes ice and darkens the surface (the ice-albedo feedback)."""
    free = cfg.free + cfg.free_p2 * _p2(x)
    s = np.tanh((np.asarray(T, dtype=float) - cfg.t_freeze) / cfg.t_width)
    return (free - cfg.ice) * 0.5 * (1.0 - s**2) / cfg.t_width
