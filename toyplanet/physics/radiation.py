"""Top-of-atmosphere radiation: absorbed sunlight and outgoing longwave.

Absorbed shortwave:   Q (1 - alpha)                    [W m^-2]
Outgoing longwave:    A + B T     (Budyko 1969)        [W m^-2], T in deg C

The linear longwave form replaces sigma*T^4 together with the greenhouse
effect and the water-vapour and lapse-rate feedbacks by a fit to the observed
relation between surface temperature and outgoing longwave. B is therefore the
local longwave feedback parameter: each kelvin of warming sends B more watts
per square metre to space. It is one of the parameters swept across the
planet family.
"""

from __future__ import annotations

import numpy as np


def absorbed_shortwave(Q, alpha) -> np.ndarray:
    """Sunlight absorbed per unit area: insolation Q times co-albedo (1 - alpha)."""
    return np.asarray(Q, dtype=float) * (1.0 - np.asarray(alpha, dtype=float))


def outgoing_longwave(T, cfg) -> np.ndarray:
    """Budyko outgoing longwave A + B*T. cfg: RadiationConfig; T in deg C."""
    return cfg.olr_a + cfg.olr_b * np.asarray(T, dtype=float)
