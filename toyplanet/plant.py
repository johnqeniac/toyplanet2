"""The plant: the planet's equations, assembled and stepped forward in time.

State per band: surface temperature T and deep-ocean temperature T_d (deg C).
Time t is in years since the northern vernal equinox.

Surface equation:
    C dT/dt = Q(1 - alpha(T)) - (A + B T) + F_ghg + F_act - gamma (T - T_d)
              + D d/dx((1 - x^2) dT/dx)

Deep equation:
    C_d dT_d/dt = gamma (T - T_d)

Time stepping (semi-implicit Euler). Diffusion is the stiff term: treated
explicitly, a step must be shorter than roughly (dx)^2 C / (2 D), which shrinks
as the grid is refined. So diffusion is taken at the NEW time and everything
else at the OLD time:

    (C/dt) (T_new - T) = R(T) + D L T_new
    =>  (C/dt I - D L) T_new = (C/dt) T + R(T)

The matrix on the left is the same every step, so it is factorized once at
start-up and each step is a cheap back-substitution. The other terms are slow
enough for explicit treatment: dt * (B + Q |d alpha/dT|) / C is about 1e-2
for a one-day step.

Energy accounting is exact. Diffusion moves heat between bands without
changing the total, and the surface-deep exchange uses the same old-time
temperatures on both sides, so the change in total heat content over a step
equals the global-mean net radiation at the old time, to round-off. The test
suite checks this every step.

The controller never imports this module. It sees only what sensors.py reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import factorized

from toyplanet.grid import LatitudeGrid
from toyplanet.insolation import band_mean
from toyplanet.physics.albedo import albedo
from toyplanet.actuators import SAI
from toyplanet.geography import land_fraction
from toyplanet.physics.radiation import absorbed_shortwave, outgoing_longwave

DAYS_PER_YEAR = 365.2422


@dataclass(frozen=True)
class State:
    T: np.ndarray       # surface temperature per band, deg C
    T_d: np.ndarray     # deep-ocean temperature per band, deg C
    step: int           # steps taken since the vernal equinox of year 0
    s_conv: np.ndarray | None = None    # SAI conversion stage per site, Tg S / yr
    burden: np.ndarray | None = None    # SAI aerosol burden per site, Tg S


class Plant:
    """The planet defined by a Config. Construct once; step many times.

    The year is divided into a WHOLE number of steps, steps_per_year =
    round(365.2422 / dt_days), so the time step is adjusted slightly (1 day
    becomes 0.99934 days). Two reasons:

    * Every year is then an exact seasonal cycle in step numbers, so an annual
      mean is an average over exactly one cycle. With a fractional number of
      steps, "the last 365 steps" misses part of a day of the cycle and leaves
      a seasonal remainder in every annual mean.
    * Insolation repeats every year, so it is computed once for each step of
      the year at start-up and looked up by (step mod steps_per_year), instead
      of re-solving Kepler's equation every step.

    Time is counted by the integer step, never by adding dt repeatedly, so no
    rounding error accumulates over long runs.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.grid = LatitudeGrid(cfg.grid.n)
        self.steps_per_year = max(1, int(round(DAYS_PER_YEAR / cfg.time.dt_days)))
        self.dt = 1.0 / self.steps_per_year                    # years
        n = self.grid.n
        # Each band is part land, part ocean, with one surface temperature.
        # Its heat capacity is the area-weighted mix; only the ocean part
        # exchanges heat with the deep ocean.
        self.land = land_fraction(self.grid, cfg.surface.land_profile)
        ocean = 1.0 - self.land
        self.C = self.land * cfg.surface.c_land + ocean * cfg.ocean.c_mixed
        self.gamma = ocean * cfg.ocean.gamma                   # W m^-2 K^-1, per band
        self.C_d = np.maximum(ocean, 1e-6) * cfg.ocean.c_deep  # deep layer under the ocean part
        L = self.grid.diffusion_operator()
        M = identity(n, format="csc").multiply(self.C[:, None] / self.dt) - cfg.transport.diffusivity * L
        self._solve = factorized(M.tocsc())                    # factorize once
        t_year = np.arange(self.steps_per_year) * self.dt
        self._Q = band_mean(self.grid, t_year, cfg.orbit).T.copy()   # (steps_per_year, n)
        if not cfg.insolation.seasonal:          # benchmark mode: annual mean every day
            self._Q[:] = self._Q.mean(axis=0)
        self.sai = SAI(cfg.sai, self.grid, self._Q)

    # -- the physics, as fluxes in W m^-2 ------------------------------------

    def time(self, s: State) -> float:
        """Years since the vernal equinox of year 0."""
        return s.step * self.dt

    def insolation(self, step: int) -> np.ndarray:
        return self._Q[step % self.steps_per_year]

    def net_radiation(self, T, step: int, F_act=0.0) -> np.ndarray:
        """Net downward energy flux at the top of the atmosphere, per band."""
        Q = self.insolation(step)
        return (absorbed_shortwave(Q, albedo(T, self.cfg.albedo, self.grid.x))
                - outgoing_longwave(T, self.cfg.radiation)
                + self.cfg.forcing.ghg + F_act)

    def heat_content(self, s: State) -> float:
        """Global-mean heat content relative to 0 deg C, W yr m^-2."""
        return self.grid.mean(self.C * s.T + self.C_d * s.T_d)

    # -- time stepping --------------------------------------------------------

    def sai_forcing(self, s: State) -> np.ndarray:
        """Forcing per band from the current aerosol burden, W m^-2."""
        return self.sai.forcing(self._burden(s), s.step)

    def _burden(self, s: State) -> np.ndarray:
        return np.zeros(self.sai.k) if s.burden is None else s.burden

    def step(self, s: State, F_act=0.0, u_sai=0.0) -> State:
        """Advance one time step.

        F_act: prescribed extra forcing per band, W m^-2.
        u_sai: SAI injection rate per site, Tg S per year (never negative).
        """
        burden = self._burden(s)
        s_conv = np.zeros(self.sai.k) if s.s_conv is None else s.s_conv
        F = F_act + self.sai.forcing(burden, s.step)
        exch = self.gamma * (s.T - s.T_d)
        R = self.net_radiation(s.T, s.step, F) - exch
        T_new = self._solve(self.C / self.dt * s.T + R)
        T_d_new = s.T_d + self.dt * exch / self.C_d
        s_new, m_new = self.sai.advance(s_conv, burden, u_sai, self.dt)
        return State(T_new, T_d_new, s.step + 1, s_new, m_new)

    def initial_state(self, T0: float = 15.0) -> State:
        n, k = self.grid.n, self.sai.k
        return State(np.full(n, T0), np.full(n, T0), 0, np.zeros(k), np.zeros(k))

    def run_years(self, s: State, years: int, F_act=0.0, F_ext=None, u_sai=None):
        """Step for a whole number of years. Returns the final state and a
        per-step record: times (m,), T (m, n), T_d (m, n).

        F_act: constant actuator forcing (scalar or per band).
        F_ext: optional prescribed forcing, one row per step, shape (m,) or
               (m, n), e.g. from forcing.build_external (variability, volcanoes).
        u_sai: optional SAI injection schedule, Tg S / yr: constant per site
               (shape (k,)) or one row per step (shape (m, k)).
        """
        steps = int(years) * self.steps_per_year
        times = np.empty(steps)
        Ts = np.empty((steps, self.grid.n))
        Tds = np.empty((steps, self.grid.n))
        if F_ext is not None:
            F_ext = np.asarray(F_ext, dtype=float)
            if F_ext.shape[0] < steps:
                raise ValueError(f"F_ext has {F_ext.shape[0]} rows; the run needs {steps}")
        u_sched = None if u_sai is None else np.asarray(u_sai, dtype=float)
        per_step = u_sched is not None and u_sched.ndim == 2
        for k in range(steps):
            F = F_act if F_ext is None else F_act + F_ext[k]
            u = 0.0 if u_sched is None else (u_sched[k] if per_step else u_sched)
            s = self.step(s, F, u)
            times[k], Ts[k], Tds[k] = self.time(s), s.T, s.T_d
        return s, times, Ts, Tds

    def spin_up(self, s: State | None = None, years_per_cycle: int = 5,
                tol: float = 1e-3, max_cycles: int = 200) -> State:
        """Bring the planet to its seasonally repeating equilibrium.

        Stepping through the deep ocean's ~140-year response time would take
        many centuries of model time. Instead: at equilibrium each band's deep
        temperature equals the annual-mean surface temperature above it (the
        deep layer has no other heat source). So run the surface for a few
        years, then shift T_d by the difference between the last year's mean
        surface and mean deep temperatures, and repeat until that shift is
        smaller than tol (K).

        The comparison is between ANNUAL MEANS of both layers. The deep layer
        carries a small seasonal ripple (a few thousandths of a kelvin), so
        setting its value at one moment equal to the surface annual mean would
        be slightly wrong at most points in the cycle.
        """
        s = self.initial_state() if s is None else s
        last = slice(-self.steps_per_year, None)
        for _ in range(max_cycles):
            s, _, Ts, Tds = self.run_years(s, years_per_cycle)
            shift = Ts[last].mean(axis=0) - Tds[last].mean(axis=0)
            s = State(s.T, s.T_d + shift, s.step, s.s_conv, s.burden)
            change = np.abs(shift).max()
            if change < tol:
                return s
        raise RuntimeError(f"spin-up did not converge in {max_cycles} cycles (last change {change:.2e} K)")
