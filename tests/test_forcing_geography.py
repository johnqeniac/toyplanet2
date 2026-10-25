"""Tests for external forcing (variability, volcanoes) and land geography."""

import numpy as np
import pytest

from toyplanet.config import from_dict
from toyplanet.forcing import ar1, build_external, volcanic_pulse
from toyplanet.geography import land_fraction
from toyplanet.grid import LatitudeGrid
from toyplanet.plant import Plant


def test_ar1_has_requested_std_and_memory():
    dt = 1 / 365
    x = ar1(400_000, dt, sigma=2.0, tau_days=30.0, rng=np.random.default_rng(0))[:, 0]
    assert np.isclose(x.std(), 2.0, rtol=0.03)
    lag = 30                                         # one memory time: autocorrelation 1/e
    r = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    assert np.isclose(r, np.exp(-1), atol=0.03)


def test_volcanic_pulse_shape():
    dt = 1 / 365
    f = volcanic_pulse(5 * 365, dt, start_year=1.0, peak=-3.5, rise_years=0.5, decay_years=1.0)
    t = np.arange(len(f)) * dt
    assert np.all(f[t < 1.0] == 0)                                  # nothing before the eruption
    assert np.isclose(f.min(), -3.5, atol=0.02)                     # reaches the peak
    k = np.searchsorted(t, 2.5)                                     # one decay time after the peak
    assert np.isclose(f[k], -3.5 / np.e, rtol=0.01)


def test_external_forcing_is_reproducible_from_seed():
    cfg = from_dict({"variability": {"global_sigma": 1.0, "seed": 7}})
    a = build_external(cfg, 1000, 1 / 365, 18)
    b = build_external(cfg, 1000, 1 / 365, 18)
    c = build_external(from_dict({"variability": {"global_sigma": 1.0, "seed": 8}}), 1000, 1 / 365, 18)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_plant_rejects_too_short_external_forcing():
    p = Plant(from_dict({"grid": {"n": 18}}))
    with pytest.raises(ValueError):
        p.run_years(p.initial_state(), 1, F_ext=np.zeros((10, 18)))


def test_earth_land_fraction_totals():
    g = LatitudeGrid(180)
    f = land_fraction(g, "earth")
    assert np.isclose(g.mean(f), 0.29, atol=0.01)                  # Earth: ~29% land
    assert f[g.lat > 0].mean() > 1.5 * f[g.lat < 0].mean()          # mostly in the north
    assert f[g.lat < -80].min() == 1.0                              # Antarctica
    assert np.all(land_fraction(g, "aquaplanet") == 0)


def test_land_makes_the_eruption_response_faster_and_larger():
    """Low heat capacity of land: a short pulse produces more cooling, sooner."""
    peaks = {}
    for profile in ("aquaplanet", "earth"):
        p = Plant(from_dict({"grid": {"n": 18}, "surface": {"land_profile": profile}}))
        s0 = p.spin_up(tol=1e-3)
        _, _, Tb, _ = p.run_years(s0, 6)
        cfg = from_dict({"grid": {"n": 18}, "surface": {"land_profile": profile},
                         "volcanic": {"peak": -3.5, "start_year": 1.0}})
        F = build_external(cfg, 6 * p.steps_per_year, p.dt, p.grid.n)
        _, _, Tv, _ = p.run_years(s0, 6, F_ext=F)
        d = (Tv - Tb) @ p.grid.weights
        peaks[profile] = (d.min(), d.argmin())
    assert peaks["earth"][0] < peaks["aquaplanet"][0]      # more cooling
    assert peaks["earth"][1] < peaks["aquaplanet"][1]      # earlier peak


def test_eruption_forcing_scaling():
    from toyplanet.forcing import HISTORICAL_ERUPTIONS, peak_forcing
    assert np.isclose(peak_forcing(9.0, "low"), -3.5) and np.isclose(peak_forcing(9.0, "high"), -3.5)
    for name, (_, sulfur) in HISTORICAL_ERUPTIONS.items():
        lo, hi = peak_forcing(sulfur, "low"), peak_forcing(sulfur, "high")
        assert hi <= lo <= 0                                      # high estimate is the larger cooling
    # sublinear: doubling the sulfur less than doubles the low-estimate forcing
    assert abs(peak_forcing(18.0)) < 2 * abs(peak_forcing(9.0))
