"""Actuators: how injection becomes forcing.

Phase 3 provides stratospheric aerosol injection (SAI) at a list of
latitudes. Each site k is one control channel with three parts:

1. Burden dynamics (part of the plant's physical state). Injection rate
   u_k (Tg S per year) passes through a conversion stage s_k (SO2 to sulfate,
   time constant tau_c; skipped for direct injection), then into the aerosol
   burden m_k (Tg S), which is removed with time constant tau_L:

       ds/dt = (u - s) / tau_c          dm/dt = s - m / tau_L

   At steady injection, m = u * tau_L.

2. Footprint. Stratospheric transport spreads each site's aerosol in
   latitude; the footprint psi_k is a Gaussian of width spread_deg about the
   injection latitude. (Transport barriers come in phase 6.)

3. Forcing. Shortwave forcing scales with sunlight: in band j at time t,

       F_j = -kappa * m_k * psi_k(j) Q_j(t) / < psi_k Q >

   where < > is the annual- and area-mean, so that one Tg S of burden gives
   exactly -kappa W m^-2 of global, annual-mean forcing. kappa = 0.39 W m^-2
   per Tg S is scaled from Pinatubo (3.5 W m^-2 from 9 Tg S). Forcing is
   linear in burden here; the sublinear response of large injections enters
   in phase 6 with the injection-density factor.

Units are chosen so the controller deals in Tg S per year, the unit used for
real injection scenarios.
"""

from __future__ import annotations

import numpy as np


class SAI:
    """Footprints, forcing pattern and burden dynamics for a set of injection sites."""

    def __init__(self, cfg_sai, grid, Q_year: np.ndarray):
        """Q_year: insolation table, shape (steps_per_year, n_bands)."""
        self.cfg = cfg_sai
        self.lat_sites = np.asarray(cfg_sai.sites_deg, dtype=float)
        self.k = len(self.lat_sites)
        lat = grid.lat[None, :]
        psi = np.exp(-0.5 * ((lat - self.lat_sites[:, None]) / cfg_sai.spread_deg) ** 2)   # (k, n)
        # Normalize so that, averaged over area and year, one Tg S gives kappa W m^-2.
        mean_psiQ = (psi[:, None, :] * Q_year[None, :, :]).mean(axis=1) @ grid.weights       # (k,)
        self._pattern = psi[:, None, :] * Q_year[None, :, :] / mean_psiQ[:, None, None]       # (k, steps, n)
        self.steps_per_year = Q_year.shape[0]

    def forcing(self, burden: np.ndarray, step: int) -> np.ndarray:
        """Forcing per band (W m^-2, negative = cooling) from burdens (Tg S, one per site)."""
        if not np.any(burden):
            return 0.0                       # no aerosol aloft: skip the arithmetic
        pat = self._pattern[:, step % self.steps_per_year, :]
        return -self.cfg.forcing_per_tg * (burden @ pat)

    def advance(self, s_conv, burden, u, dt_years):
        """One step of the burden dynamics. u: injection rate per site, Tg S / yr."""
        u = np.broadcast_to(np.asarray(u, dtype=float), (self.k,))
        if np.any(u < 0):
            raise ValueError("injection rates cannot be negative")
        tau_c = self.cfg.tau_conv_days / 365.2422
        if tau_c == 0.0:
            s_new = u.copy()
        else:
            s_new = s_conv + dt_years * (u - s_conv) / tau_c
        m_new = burden + dt_years * (s_conv - burden / self.cfg.tau_life_years)
        return s_new, m_new
