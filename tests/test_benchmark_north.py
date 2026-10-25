"""The North, Cahalan & Coakley (1981) ice-edge benchmark.

Three layers of check: (1) the model's grid and diffusion operator reproduce
the exact Legendre-series solution; (2) that solution reproduces the published
present-day ice edge; (3) the time-stepped plant, run with annual-mean
insolation, settles on the same ice edge by itself.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from toyplanet.benchmarks.north import NorthParams, grid_Q_of_xs, legendre_Q_of_xs
from toyplanet.config import from_dict
from toyplanet.diagnostics import ice_edge_north
from toyplanet.grid import LatitudeGrid
from toyplanet.plant import Plant

P = NorthParams()


@pytest.mark.parametrize("xs", [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98])
def test_model_grid_matches_exact_legendre_solution(xs):
    g = LatitudeGrid(180)
    assert np.isclose(grid_Q_of_xs(xs, g), legendre_Q_of_xs(xs), rtol=2e-4)


def test_published_present_day_ice_edge():
    """North et al. (1981): present ice edge at x_s = 0.95 for Q0 = 335 W m^-2."""
    xs = brentq(lambda x: legendre_Q_of_xs(x) - P.Q0, 0.7, 0.995)
    assert abs(xs - 0.95) < 0.01


def test_snowball_bifurcation_exists():
    """Q(x_s) has an interior minimum: below it no partly ice-covered climate exists."""
    xs = np.linspace(0.1, 0.95, 86)
    Q = np.array([legendre_Q_of_xs(x, nmax=200) for x in xs])
    i = np.argmin(Q)
    assert 0 < i < len(xs) - 1 and 0.3 < xs[i] < 0.7 and Q[i] < P.Q0


@pytest.mark.parametrize("frac", [1.00, 0.97])
def test_time_stepped_plant_finds_the_exact_ice_edge(frac):
    """Away from the polar cusp, where sensitivity to small differences diverges."""
    Q = frac * P.Q0
    cfg = from_dict({"grid": {"n": 90}, "insolation": {"seasonal": False},
                     "orbit": {"solar_constant": 4 * Q}, "albedo": {"t_width": 0.3}})
    p = Plant(cfg)
    s = p.spin_up(tol=1e-5)
    exact = brentq(lambda x: legendre_Q_of_xs(x) - Q, 0.6, 0.998)
    assert abs(ice_edge_north(p.grid.x, s.T, -10.0) - exact) < 0.003


def test_ice_edge_diagnostic():
    x = np.array([-0.75, -0.25, 0.25, 0.75])
    assert ice_edge_north(x, np.array([0, 20, 20, 0]), -10) == 1.0     # ice-free
    assert ice_edge_north(x, np.array([-30, -20, -20, -30]), -10) == 0.0
    assert np.isclose(ice_edge_north(x, np.array([-30, 10, 10, -30]), -10), 0.25 + 0.5 * 20 / 40)


def test_annual_mean_insolation_mode_is_constant_in_time():
    p = Plant(from_dict({"grid": {"n": 18}, "insolation": {"seasonal": False}}))
    assert np.allclose(p.insolation(0), p.insolation(150))
