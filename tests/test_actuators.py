"""Tests for the SAI actuator: burden dynamics, footprint normalization, forcing."""

import numpy as np
import pytest

from toyplanet.config import ConfigError, from_dict, load, write_resolved
from toyplanet.plant import Plant

KAPPA = 0.39


def plant(**sai):
    return Plant(from_dict({"grid": {"n": 18}, "sai": sai}))


def test_one_tg_of_burden_gives_kappa_of_global_annual_mean_forcing():
    p = plant(sites_deg=[0.0, 30.0, -60.0])
    for k in range(3):
        burden = np.zeros(3); burden[k] = 1.0
        F = np.array([p.sai.forcing(burden, j) for j in range(p.steps_per_year)])
        assert np.isclose(F.mean(axis=0) @ p.grid.weights, -KAPPA, rtol=1e-10)


def test_footprint_centred_on_injection_latitude():
    p = plant(sites_deg=[30.0], spread_deg=15.0)
    F = np.array([p.sai.forcing(np.array([1.0]), j) for j in range(p.steps_per_year)]).mean(axis=0)
    assert abs(p.grid.lat[np.argmin(F)] - 30.0) < 15.0           # strongest cooling near 30 N
    assert F[p.grid.lat < -30].mean() > 0.5 * F[np.argmin(F)]    # weak in the far hemisphere


def test_steady_injection_gives_burden_u_times_lifetime():
    p = plant(tau_life_years=1.0)
    s = p.initial_state()
    s, *_ = p.run_years(s, 12, u_sai=[2.0])
    assert np.isclose(s.burden[0], 2.0 * 1.0, rtol=1e-3)


def test_burden_decays_with_lifetime_after_termination():
    p = plant(tau_life_years=1.0, tau_conv_days=0.0)
    s, *_ = p.run_years(p.initial_state(), 10, u_sai=[1.0])
    m0 = s.burden[0]
    s, *_ = p.run_years(s, 1, u_sai=[0.0])
    assert np.isclose(s.burden[0] / m0, np.exp(-1.0), rtol=5e-3)


def test_so2_conversion_delays_the_burden():
    fast, slow = plant(tau_conv_days=0.0), plant(tau_conv_days=60.0)
    m = []
    for p in (fast, slow):
        s = p.initial_state()
        for _ in range(30):                       # one month of injection
            s = p.step(s, u_sai=1.0)
        m.append(s.burden[0])
    assert m[1] < 0.8 * m[0]


def test_negative_injection_is_rejected():
    p = plant()
    with pytest.raises(ValueError):
        p.step(p.initial_state(), u_sai=-1.0)


def test_sites_list_in_config(tmp_path):
    cfg = from_dict({"sai": {"sites_deg": [-15, 15, 30]}})
    assert cfg.sai.sites_deg == (-15.0, 15.0, 30.0)
    assert load(write_resolved(cfg, tmp_path)) == cfg               # lists round-trip
    with pytest.raises(ConfigError, match="list of numbers"):
        from_dict({"sai": {"sites_deg": 15}})
    with pytest.raises(ConfigError, match="outside the allowed range"):
        from_dict({"sai": {"sites_deg": [0, 120]}})
