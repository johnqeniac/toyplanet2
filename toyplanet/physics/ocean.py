"""Two-layer ocean: heat exchange between the surface layer and the deep ocean.

    C   dT/dt   = (other fluxes) - gamma (T - T_d)
    C_d dT_d/dt =                  gamma (T - T_d)

The deep layer has no other heat source or sink, so at equilibrium its
temperature equals the time-mean surface temperature of the same band. Its
response time is about C_d / gamma (about 140 years with the defaults), which
is the slow mode of the Held et al. (2010) two-box model, applied per band.
"""

from __future__ import annotations

import numpy as np


def exchange_flux(T, T_d, cfg) -> np.ndarray:
    """Heat flux from surface layer to deep layer, W m^-2. cfg: OceanConfig."""
    return cfg.gamma * (np.asarray(T, dtype=float) - np.asarray(T_d, dtype=float))
