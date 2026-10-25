"""External forcing time series: natural variability and volcanic eruptions.

These are the "nature" inputs the plant receives in addition to greenhouse
forcing and actuators. Each function returns a whole time series up front,
one row per model step, so a run is fully determined by its config (including
the random seed) and can be reproduced exactly.

Natural variability: red noise (a first-order autoregressive process, AR(1)),
    F[k+1] = a F[k] + sigma sqrt(1 - a^2) w[k],   a = exp(-dt / tau),
with w[k] independent standard normal numbers. The series has standard
deviation sigma and memory time tau. Two components are provided: one shared
by the whole planet (global variability such as El Nino) and one independent
in each band (regional weather).

Volcanic pulse: a linear rise to the peak forcing over rise_years, then
exponential decay with time constant decay_years (the stratospheric aerosol
removal time), applied uniformly in latitude. Negative values cool.
"""

from __future__ import annotations

import numpy as np


def ar1(n_steps: int, dt_years: float, sigma: float, tau_days: float,
        rng: np.random.Generator, n_series: int = 1) -> np.ndarray:
    """Red-noise series, shape (n_steps, n_series), standard deviation sigma."""
    out = np.zeros((n_steps, n_series))
    if sigma == 0.0 or n_steps == 0:
        return out
    a = np.exp(-dt_years * 365.2422 / tau_days)
    kick = sigma * np.sqrt(1.0 - a * a)
    w = rng.standard_normal((n_steps, n_series))
    x = sigma * w[0]                       # start in the stationary distribution
    out[0] = x
    for k in range(1, n_steps):
        x = a * x + kick * w[k]
        out[k] = x
    return out


def volcanic_pulse(n_steps: int, dt_years: float, start_year: float, peak: float,
                   rise_years: float, decay_years: float) -> np.ndarray:
    """Global forcing from one eruption, W m^-2, shape (n_steps,)."""
    t = np.arange(n_steps) * dt_years - start_year     # years since eruption
    f = np.zeros(n_steps)
    rising = (t >= 0) & (t < rise_years)
    f[rising] = peak * t[rising] / rise_years
    after = t >= rise_years
    f[after] = peak * np.exp(-(t[after] - rise_years) / decay_years)
    return f


def build_external(cfg, n_steps: int, dt_years: float, n_bands: int) -> np.ndarray:
    """All external forcing for a run, shape (n_steps, n_bands), W m^-2."""
    v, vol = cfg.variability, cfg.volcanic
    rng = np.random.default_rng(v.seed)
    F = ar1(n_steps, dt_years, v.global_sigma, v.global_tau_days, rng)        # (n, 1): broadcasts
    F = F + ar1(n_steps, dt_years, v.band_sigma, v.band_tau_days, rng, n_bands)
    if vol.peak != 0.0:
        F = F + volcanic_pulse(n_steps, dt_years, vol.start_year, vol.peak,
                               vol.rise_years, vol.decay_years)[:, None]
    return F
