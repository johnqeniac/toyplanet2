"""Phase 3 gate: departure from linearity at test-scale doses, and recovery after termination.

Reduced grid (18 bands) for speed; the full-resolution numbers come from
scripts/phase3_dose_termination.py.
"""

import dataclasses

import numpy as np

from toyplanet.config import from_dict, load
from toyplanet.plant import Plant


def _setup():
    base = load("configs/stage1_earth.toml")
    base = dataclasses.replace(base, grid=dataclasses.replace(base.grid, n=18),
                               variability=dataclasses.replace(base.variability, global_sigma=0.0))
    p = Plant(base)
    return base, p, p.spin_up(tol=1e-4)


def test_response_is_proportional_at_test_scale_doses():
    base, p, s0 = _setup()
    years, spy = 10, p.steps_per_year
    rate = lambda F: F / (base.sai.forcing_per_tg * base.sai.tau_life_years)
    annual = lambda T: T.reshape(-1, spy, T.shape[1]).mean(axis=1) @ p.grid.weights
    _, _, T_ctl, _ = p.run_years(s0, years)
    per_watt = []
    for F in (0.01, 0.1, 0.3):
        _, _, T, _ = p.run_years(s0, years, u_sai=[rate(F)])
        per_watt.append(-(annual(T) - annual(T_ctl))[-2:].mean() / F)
    per_watt = np.array(per_watt)
    assert np.ptp(per_watt) / per_watt.mean() < 0.005          # within 0.5% across 30x in dose


def test_test_scale_injection_is_reversible():
    base, p, s0 = _setup()
    spy = p.steps_per_year
    u = 0.1 / (base.sai.forcing_per_tg * base.sai.tau_life_years)
    sched = np.zeros((20 * spy, 1)); sched[:10 * spy] = u
    _, _, T, _ = p.run_years(s0, 20, u_sai=sched)
    _, _, T_ctl, _ = p.run_years(s0, 20)
    d = (T - T_ctl) @ p.grid.weights
    at_stop = d[10 * spy - 1]
    assert d[-1] / at_stop < 0.1                                 # >90% recovered 10 yr after stopping
    annual = (T - T_ctl).reshape(20, spy, -1).mean(axis=1) @ p.grid.weights
    assert np.diff(annual)[9:].max() < 0.02                      # warming rate after stopping < 0.02 K/yr
