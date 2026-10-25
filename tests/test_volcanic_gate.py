"""Phase 2 gate: the response to a Pinatubo-like eruption is comparable to observation.

Observed (Soden et al. 2002, Science 296, 727): peak global cooling ~0.5 K
about 18 months after the June 1991 eruption (satellite lower-troposphere
record), then a slow return. Forcing peak taken as -3.5 W m^-2 (middle of the
commonly cited -3 to -4 range; to be checked against a forcing reconstruction).
Gate: peak cooling 0.3-0.6 K, reached 12-24 months after the eruption, with
most of it gone after 8 years.
"""

import numpy as np

from toyplanet.config import from_dict
from toyplanet.forcing import build_external
from toyplanet.plant import Plant


def test_pinatubo_like_response_within_observed_range():
    p = Plant(from_dict({}))
    s0 = p.spin_up(tol=1e-4)
    years = 10
    _, t, T_control, _ = p.run_years(s0, years)
    cfg = from_dict({"volcanic": {"peak": -3.5, "start_year": 1.0}})
    F = build_external(cfg, years * p.steps_per_year, p.dt, p.grid.n)
    _, _, T_volc, _ = p.run_years(s0, years, F_ext=F)
    d = (T_volc - T_control) @ p.grid.weights        # difference removes the seasonal cycle
    since = t - p.time(s0) - 1.0                      # years since eruption
    k = np.argmin(d)
    assert 0.3 < -d[k] < 0.6
    assert 1.0 < since[k] < 2.0
    assert -d[np.searchsorted(since, 8.0)] < 0.2 * -d[k]
