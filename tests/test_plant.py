"""Tests for the assembled plant and its time stepping.

The two strongest checks use exact results: the energy budget must close on
every step, and with the ice feedback switched off (the equations are then
linear) the annual mean must match the steady solution of the same equations
solved directly, with no time stepping at all.
"""

import numpy as np
import pytest
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from toyplanet.config import from_dict
from toyplanet.plant import Plant

NO_ICE = {"albedo": {"ice": 0.30, "free": 0.30, "free_p2": 0.0}}   # constant albedo: linear planet


def make(overrides=None, **sections):
    d = {"grid": {"n": 18}}
    d.update(overrides or {})
    d.update(sections)
    return Plant(from_dict(d))


def test_energy_budget_closes_every_step():
    """Change in total heat content = dt x global-mean net radiation, to round-off."""
    p = make()
    s = p.initial_state(5.0)
    s = type(s)(s.T + np.linspace(-20, 20, p.grid.n), s.T_d, s.step)   # non-uniform start
    for _ in range(50):
        H0 = p.heat_content(s)
        expected = p.dt * p.grid.mean(p.net_radiation(s.T, s.step))
        s = p.step(s)
        assert abs(p.heat_content(s) - H0 - expected) < 1e-12 * max(1.0, abs(H0))


def test_linear_planet_matches_direct_steady_solution():
    """With constant albedo the annual mean satisfies
        (B I - D L) T = <Q>(1 - alpha) - A + F
    exactly, where <Q> is annual-mean insolation. Solve that directly and
    compare with the time-stepped, spun-up model."""
    p = make(NO_ICE, forcing={"ghg": 1.0})
    s = p.spin_up(tol=1e-5)
    _, _, T, _ = p.run_years(s, 1.0)
    Qbar = p._Q.mean(axis=0)          # annual-mean insolation the model actually sees
    c = p.cfg
    M = diags(np.full(p.grid.n, c.radiation.olr_b)) - c.transport.diffusivity * p.grid.diffusion_operator()
    T_direct = spsolve(M.tocsc(), Qbar * (1 - 0.30) - c.radiation.olr_a + c.forcing.ghg)
    assert np.abs(T.mean(axis=0) - T_direct).max() < 0.01


def test_global_response_to_uniform_forcing_is_F_over_B_without_ice():
    """Transport only redistributes heat, so in the linear planet the global
    mean must warm by exactly F/B at equilibrium."""
    base = make(NO_ICE)
    forced = make(NO_ICE, forcing={"ghg": 3.7})
    means = []
    for p in (base, forced):
        _, _, T, _ = p.run_years(p.spin_up(tol=1e-5), 1.0)
        means.append(p.grid.mean(T.mean(axis=0)))
    assert np.isclose(means[1] - means[0], 3.7 / base.cfg.radiation.olr_b, atol=0.01)


def test_circular_orbit_gives_hemispherically_symmetric_climate():
    p = make(orbit={"eccentricity": 0.0})
    _, _, T, _ = p.run_years(p.spin_up(), 1.0)
    ann = T.mean(axis=0)
    assert np.allclose(ann, ann[::-1], atol=0.02)


def test_spin_up_sets_deep_ocean_to_annual_mean_surface():
    p = make()
    s = p.spin_up(tol=1e-6)
    _, _, T, Td = p.run_years(s, 1)
    assert np.abs(Td.mean(axis=0) - T.mean(axis=0)).max() < 1e-5


def test_equator_warmer_than_poles_and_seasons_larger_at_high_latitude():
    p = make()
    _, _, T, _ = p.run_years(p.spin_up(), 1.0)
    ann, rng = T.mean(axis=0), T.max(axis=0) - T.min(axis=0)
    mid = p.grid.n // 2
    assert ann[mid] > ann[0] + 15 and ann[mid] > ann[-1] + 15
    assert rng[-1] > 2 * rng[mid]


def test_year_is_a_whole_number_of_steps():
    p = make()
    assert p.steps_per_year == 365 and np.isclose(p.dt * p.steps_per_year, 1.0)
    assert np.allclose(p.insolation(5), p.insolation(5 + 100 * p.steps_per_year))


def test_spin_up_converges_quickly_to_tight_tolerance():
    """Regression test: with a fractional year the spin-up stalled near 1e-4 K."""
    p = make(NO_ICE)
    p.spin_up(tol=1e-6, max_cycles=40)
