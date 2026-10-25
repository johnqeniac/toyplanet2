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


# --------------------------------------------------------------------------
# Historical eruptions
# --------------------------------------------------------------------------
# Stratospheric sulfur injections (Tg S) from eVolv2k, Toohey & Sigl (2017),
# Earth Syst. Sci. Data 9, 809, Table 2; Pinatubo from the same paper (9 +/- 2 Tg S).
HISTORICAL_ERUPTIONS = {
    # name: (year CE, sulfur injection Tg S)
    "Samalas":     (1257, 59.4),
    "Tambora":     (1815, 28.1),
    "Laki":        (1783, 20.8),   # high-latitude, partly tropospheric: shorter-lived than a tropical eruption
    "Huaynaputina": (1600, 19.0),
    "Pinatubo":    (1991, 9.0),
}

PINATUBO_SULFUR = 9.0      # Tg S
PINATUBO_FORCING = -3.5    # W m^-2, middle of the commonly cited -3 to -4 range (to be checked)
TAMBORA_SULFUR = 28.1      # Tg S; eVolv2k's aerosol model switches to a 2/3 power law near this size


def peak_forcing(sulfur_tg: float, estimate: str = "low") -> float:
    """Peak global forcing (W m^-2) for a sulfur injection, scaled from Pinatubo.

    Larger eruptions make larger particles, which reflect less per kilogram,
    so forcing grows more slowly than the injection. Two estimates bracket it:

      low:  F = F_pin (S / S_pin)^(2/3)                   sublinear throughout
      high: linear up to Tambora's size, then (S / S_Tambora)^(2/3) beyond

    The volcanic-record constraint uses the LOW estimate: it claims only that
    the planet survived a kick of at least this size.
    """
    r = sulfur_tg / PINATUBO_SULFUR
    if estimate == "low":
        return PINATUBO_FORCING * r ** (2.0 / 3.0)
    if estimate == "high":
        if sulfur_tg <= TAMBORA_SULFUR:
            return PINATUBO_FORCING * r
        F_tambora = PINATUBO_FORCING * TAMBORA_SULFUR / PINATUBO_SULFUR
        return F_tambora * (sulfur_tg / TAMBORA_SULFUR) ** (2.0 / 3.0)
    raise ValueError(f"estimate must be 'low' or 'high', got {estimate!r}")
