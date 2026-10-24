"""Tests for albedo and radiation, including the ice-albedo multiple equilibria
that any energy-balance model with these ingredients must show."""

import numpy as np
from scipy.optimize import brentq

from toyplanet.config import AlbedoConfig, RadiationConfig
from toyplanet.physics.albedo import albedo, dalbedo_dT
from toyplanet.physics.radiation import absorbed_shortwave, outgoing_longwave

ALB = AlbedoConfig()
RAD = RadiationConfig()


def test_albedo_limits_and_midpoint():
    assert np.isclose(albedo(-200.0, ALB), ALB.ice)
    assert np.isclose(albedo(+200.0, ALB), ALB.free)
    assert np.isclose(albedo(ALB.t_freeze, ALB), 0.5 * (ALB.ice + ALB.free))


def test_albedo_is_monotone_and_bounded():
    a = albedo(np.linspace(-100, 100, 2001), ALB)
    assert np.all(np.diff(a) <= 0)                       # warmer -> darker
    assert a.min() >= min(ALB.ice, ALB.free) and a.max() <= max(ALB.ice, ALB.free)


def test_albedo_derivative_matches_finite_difference():
    T = np.linspace(-30, 10, 41)
    h = 1e-5
    fd = (albedo(T + h, ALB) - albedo(T - h, ALB)) / (2 * h)
    assert np.allclose(dalbedo_dT(T, ALB), fd, atol=1e-8)


def test_olr_is_A_at_zero_and_slope_B():
    assert np.isclose(outgoing_longwave(0.0, RAD), RAD.olr_a)
    assert np.isclose(outgoing_longwave(1.0, RAD) - outgoing_longwave(0.0, RAD), RAD.olr_b)


def test_absorbed_shortwave():
    assert np.isclose(absorbed_shortwave(340.0, 0.3), 238.0)


def _global_imbalance(T, S0=1361.0):
    """0-D planet: sunlight absorbed minus longwave emitted, W m^-2."""
    return absorbed_shortwave(S0 / 4, albedo(T, ALB)) - outgoing_longwave(T, RAD)


def test_zero_d_planet_has_warm_unstable_and_snowball_equilibria():
    """With ice-albedo feedback the 0-D balance has three equilibria: a warm
    state, a snowball, and an unstable state between them. This is the
    bistability that claim tests (strong feedbacks, tipping) will exercise in
    the full model. A root is unstable when the imbalance INCREASES with T:
    a small warming then produces net heating and runs away."""
    T = np.linspace(-80, 40, 1201)
    f = _global_imbalance(T)
    brackets = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    roots = [brentq(_global_imbalance, T[i], T[i + 1]) for i in brackets]
    assert len(roots) == 3
    snowball, unstable, warm = roots
    slope = lambda t: (_global_imbalance(t + 1e-4) - _global_imbalance(t - 1e-4)) / 2e-4
    assert slope(snowball) < 0 and slope(warm) < 0 and slope(unstable) > 0
    assert 10.0 < warm < 20.0          # sanity check on the provisional defaults
    assert snowball < -30.0
