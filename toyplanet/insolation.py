"""Daily-mean insolation at the top of the atmosphere.

The quantity is Q(phi, t): solar energy reaching a horizontal unit area at
latitude phi, averaged over one day, in W m^-2. It depends only on orbital
geometry, so it can be computed exactly.

Geometry, in three steps:

1. Where the planet is on its orbit. Time t is measured in years from the
   northern vernal equinox. Kepler's equation gives the true anomaly nu (the
   angle from perihelion), and from it the Sun's ecliptic longitude
   lam = nu + lam_p, where lam_p is the Sun's longitude at perihelion, and the
   distance factor (a/r) = (1 + e cos nu) / (1 - e^2).

2. Solar declination: sin(delta) = sin(obliquity) * sin(lam).

3. Daily average over the rotation. The Sun is up while the hour angle h is
   within +/- h0, where cos(h0) = -tan(phi) tan(delta). Integrating the
   cosine of the solar zenith angle over that interval gives

       Q = (S0 / pi) (a/r)^2 [ h0 sin(phi) sin(delta)
                               + cos(phi) cos(delta) sin(h0) ]

   When |tan(phi) tan(delta)| > 1 the Sun never sets (h0 = pi, polar day) or
   never rises (h0 = 0, polar night). Clipping cos(h0) to [-1, 1] handles both
   with no special case.

Band averages: Q varies strongly with latitude near the poles, so the value at
a band's centre is not the band's average. band_mean() averages over points
spread evenly in x = sin(latitude) within each band, which is an average over
area because the grid's bands are equal-area.

Reference: Berger (1978), J. Atmos. Sci. 35, 2362, for the formulation.
Convention note: lam_p here is the Sun's geocentric longitude at perihelion
(about 283 deg for present-day Earth, i.e. early January). Berger's "varpi"
is the heliocentric version, lam_p - 180 deg.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Orbit:
    """Orbital and solar parameters. Defaults are present-day Earth."""

    solar_constant: float = 1361.0                    # W m^-2 at mean distance a
    obliquity_deg: float = 23.44
    eccentricity: float = 0.0167
    perihelion_solar_longitude_deg: float = 282.9     # Sun's longitude at perihelion

    def __post_init__(self):
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError(f"eccentricity must be in [0, 1), got {self.eccentricity}")


def _solve_kepler(mean_anomaly: np.ndarray, e: float) -> np.ndarray:
    """Eccentric anomaly E from Kepler's equation M = E - e sin E (Newton's method)."""
    E = np.array(mean_anomaly, dtype=float, copy=True)
    for _ in range(50):
        step = (E - e * np.sin(E) - mean_anomaly) / (1.0 - e * np.cos(E))
        E -= step
        if np.max(np.abs(step)) < 1e-14:
            return E
    raise RuntimeError("Kepler solver did not converge")


def solar_position(t_years, orbit: Orbit = Orbit()):
    """Sun's ecliptic longitude (radians) and distance factor (a/r)^2 at time t.

    t_years: years since the northern vernal equinox (scalar or array).
    """
    t = np.asarray(t_years, dtype=float)
    e = orbit.eccentricity
    lam_p = np.radians(orbit.perihelion_solar_longitude_deg)

    # At the equinox (t = 0) the Sun's longitude is 0, so its true anomaly is
    # -lam_p. Convert that to a mean anomaly once; after that the mean anomaly
    # advances uniformly by 2 pi per year.
    nu0 = -lam_p
    E0 = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(nu0 / 2.0),
                          np.sqrt(1.0 + e) * np.cos(nu0 / 2.0))
    M0 = E0 - e * np.sin(E0)

    # Reduce to [0, 2 pi): the orbit repeats every year, and an unreduced mean
    # anomaly grows without bound with t. After a century it is about 600 rad,
    # where double precision resolves only about 1e-13, so the Kepler solver's
    # convergence test could never be met.
    M = np.mod(M0 + 2.0 * np.pi * t, 2.0 * np.pi)
    E = _solve_kepler(M, e)
    nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0),
                          np.sqrt(1.0 - e) * np.cos(E / 2.0))
    lam = np.mod(nu + lam_p, 2.0 * np.pi)
    distance_factor_sq = ((1.0 + e * np.cos(nu)) / (1.0 - e**2)) ** 2
    return lam, distance_factor_sq


def daily_insolation(x, t_years, orbit: Orbit = Orbit()) -> np.ndarray:
    """Daily-mean top-of-atmosphere insolation, W m^-2.

    x: sine of latitude (array). t_years: time since vernal equinox (scalar or
    array). Arrays broadcast, so passing x with shape (n, 1) and t with shape
    (m,) returns an (n, m) table: latitude down, time across.
    """
    sin_phi = np.clip(np.asarray(x, dtype=float), -1.0, 1.0)
    cos_phi = np.sqrt(1.0 - sin_phi**2)

    lam, dist_sq = solar_position(t_years, orbit)
    sin_delta = np.sin(np.radians(orbit.obliquity_deg)) * np.sin(lam)
    cos_delta = np.sqrt(1.0 - sin_delta**2)

    # cos(h0) = -tan(phi) tan(delta), written to stay finite at the poles.
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_h0 = -(sin_phi * sin_delta) / (cos_phi * cos_delta)
    cos_h0 = np.where(np.isfinite(cos_h0), cos_h0, -np.sign(sin_phi * sin_delta))
    h0 = np.arccos(np.clip(cos_h0, -1.0, 1.0))

    return (orbit.solar_constant / np.pi) * dist_sq * (
        h0 * sin_phi * sin_delta + cos_phi * cos_delta * np.sin(h0)
    )


def band_mean(grid, t_years, orbit: Orbit = Orbit(), points_per_band: int = 16) -> np.ndarray:
    """Insolation averaged over each band of an equal-area grid.

    Returns shape (grid.n,) for scalar t, or (grid.n, len(t)) for array t.
    """
    # Midpoints of points_per_band equal sub-intervals of each band, in x.
    frac = (np.arange(points_per_band) + 0.5) / points_per_band
    x_sub = grid.x_edges[:-1, None] + frac[None, :] * grid.dx[:, None]   # (n, k)
    t = np.asarray(t_years, dtype=float)
    q = daily_insolation(x_sub[..., None], t.reshape(1, 1, -1), orbit)  # (n, k, m)
    q = q.mean(axis=1)
    return q[:, 0] if t.ndim == 0 else q
