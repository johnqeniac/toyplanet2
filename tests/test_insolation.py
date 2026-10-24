"""Tests for insolation against exact results of spherical geometry.

Every expected value below is derived analytically, not taken from the code.
"""

import numpy as np
import pytest

from toyplanet.grid import LatitudeGrid
from toyplanet.insolation import Orbit, band_mean, daily_insolation, solar_position

S0 = 1361.0
CIRCULAR = Orbit(eccentricity=0.0)
EQUINOX, SUMMER, AUTUMN, WINTER = 0.0, 0.25, 0.5, 0.75   # fractions of a year (circular orbit)


def test_equator_at_equinox_is_S0_over_pi():
    """Sun overhead at noon, 12 h day: daily mean = S0/pi."""
    q = daily_insolation(0.0, EQUINOX, CIRCULAR)
    assert np.isclose(q, S0 / np.pi, rtol=1e-12)


def test_polar_night_is_zero_and_polar_day_is_S0_sin_obliquity():
    """At the north pole the Sun circles at elevation delta all day.

    Northern summer solstice: delta = obliquity, so Q = S0 sin(obliquity).
    Northern winter solstice: the Sun never rises, so Q = 0.
    """
    eps = np.radians(23.44)
    assert np.isclose(daily_insolation(1.0, SUMMER, CIRCULAR), S0 * np.sin(eps), rtol=1e-12)
    assert daily_insolation(1.0, WINTER, CIRCULAR) == 0.0
    assert daily_insolation(-1.0, SUMMER, CIRCULAR) == 0.0


def test_whole_polar_cap_is_dark_in_winter():
    """Everywhere poleward of the Arctic Circle (90 - obliquity) gets no sun at winter solstice."""
    lats = np.radians(np.linspace(90 - 23.44 + 0.1, 90, 20))
    assert np.all(daily_insolation(np.sin(lats), WINTER, CIRCULAR) == 0.0)


def test_global_mean_is_S0_over_4_every_day():
    """A sphere intercepts pi R^2 of sunlight over 4 pi R^2 of area, on every
    day of the year, independent of obliquity. With eccentricity, scaled by (a/r)^2."""
    g = LatitudeGrid(18)
    for t in np.linspace(0, 1, 13):
        assert np.isclose(g.mean(band_mean(g, t, CIRCULAR)), S0 / 4, rtol=1e-4)
        _, dist_sq = solar_position(t, Orbit())
        assert np.isclose(g.mean(band_mean(g, t, Orbit())), S0 / 4 * dist_sq, rtol=1e-4)


def test_annual_mean_at_pole_is_S0_sin_obliquity_over_pi():
    """Known closed form for a circular orbit: annual-mean polar insolation = (S0/pi) sin(eps)."""
    t = (np.arange(3650) + 0.5) / 3650
    q = daily_insolation(1.0, t, CIRCULAR).mean()
    assert np.isclose(q, S0 / np.pi * np.sin(np.radians(23.44)), rtol=1e-4)


def test_annual_mean_is_hemispherically_symmetric_even_with_eccentricity():
    """Kepler's second law: the time spent near perihelion is shorter in exact
    proportion to the extra sunlight, so annual totals match between hemispheres."""
    orbit = Orbit(eccentricity=0.05)
    t = (np.arange(7300) + 0.5) / 7300
    x = np.array([0.3, 0.6, 0.9])
    north = daily_insolation(x[:, None], t, orbit).mean(axis=1)
    south = daily_insolation(-x[:, None], t, orbit).mean(axis=1)
    assert np.allclose(north, south, rtol=1e-4)


def test_insolation_is_never_negative():
    g = LatitudeGrid(36)
    t = np.linspace(0, 1, 365)
    assert band_mean(g, t).min() >= 0.0


def test_kepler_distance_extremes():
    """(a/r)^2 runs from 1/(1+e)^2 at aphelion to 1/(1-e)^2 at perihelion."""
    e = 0.0167
    _, d2 = solar_position(np.linspace(0, 1, 20001), Orbit(eccentricity=e))
    assert np.isclose(d2.max(), 1 / (1 - e) ** 2, rtol=1e-6)
    assert np.isclose(d2.min(), 1 / (1 + e) ** 2, rtol=1e-6)


def test_insolation_repeats_every_year_far_from_t_zero():
    """Runs last centuries. The orbit must repeat exactly each year, and the
    Kepler solver must still converge at large t (a bug found during spin-up)."""
    x = np.linspace(-0.95, 0.95, 7)[:, None]
    t = np.linspace(0, 1, 50)
    base = daily_insolation(x, t)
    for years in (1, 37, 500, 10000):
        assert np.allclose(daily_insolation(x, t + years), base, rtol=1e-9, atol=1e-9)


def test_rejects_unbound_orbit():
    with pytest.raises(ValueError):
        Orbit(eccentricity=1.0)
