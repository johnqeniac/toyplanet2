"""North-Coakley annual-mean energy-balance model: the ice-edge benchmark.

The model (North 1975; North & Coakley 1979; North, Cahalan & Coakley 1981):

    -D d/dx((1 - x^2) dT/dx) + A + B T = Q S(x) a(x, x_s)

    S(x)   = 1 + S2 P2(x)              annual-mean insolation shape, S2 = -0.477
    a(x)   = a0 + a2 P2(x) for |x| < x_s   (ice-free co-albedo)
           = b0                for |x| > x_s   (ice-covered co-albedo)
    T(x_s) = T_s = -10 C               the ice edge sits where T = -10 C

with the values of North, Cahalan & Coakley (1981, Rev. Geophys. 19, 91):
A = 203.3 W m^-2 and B = 2.09 W m^-2 K^-1 (their eq. 7), D = 0.649 W m^-2 K^-1
(section 2), S2 = -0.477 (eq. 3), ice-free co-albedo a0 = 0.681, a2 = -0.202
(eq. 18), ice co-albedo b0 = 0.38, and present-day Q = 335 W m^-2. With these
the present ice edge is at x_s = 0.954 (72.6 deg), matching their x_s = 0.95.

The classic result is the curve Q(x_s): the solar input needed to hold the ice
edge at x_s. It is computed by fixing x_s, which fixes the albedo and makes the
equation linear in T, then choosing Q so that T(x_s) = T_s. Fixing x_s rather
than Q gives every equilibrium, unstable ones included, which time stepping
cannot find.

Two independent methods are provided, and they must agree:

  legendre_Q_of_xs : the exact solution as a series in Legendre polynomials,
                     the method of North (1975). Uses no part of the model code.
  grid_Q_of_xs     : the same equation solved with the model's own grid and
                     diffusion operator.

Agreement validates the grid, the operator and its boundary condition against
the published analytic method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre as npleg
from scipy.special import eval_legendre
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

from toyplanet.grid import LatitudeGrid


@dataclass(frozen=True)
class NorthParams:
    A: float = 203.3
    B: float = 2.09
    D: float = 0.649
    S2: float = -0.477
    a0: float = 0.681
    a2: float = -0.202
    b0: float = 0.38
    T_s: float = -10.0
    Q0: float = 335.0        # present-day value, W m^-2 (S0 = 4 Q0 = 1340)


def _P2(x):
    return 0.5 * (3.0 * np.asarray(x) ** 2 - 1.0)


def _source(x, xs, p: NorthParams):
    """S(x) a(x, x_s): insolation shape times co-albedo, |x| measured from equator."""
    ax = np.abs(x)
    coalb = np.where(ax < xs, p.a0 + p.a2 * _P2(x), p.b0)
    return (1.0 + p.S2 * _P2(x)) * coalb


def legendre_Q_of_xs(xs: float, p: NorthParams = NorthParams(), nmax: int = 400) -> float:
    """Exact Q(x_s) from the Legendre-series solution (even terms only: symmetric planet).

    Each Legendre polynomial is an eigenfunction of the diffusion operator,
    d/dx((1-x^2) dP_n/dx) = -n(n+1) P_n, so the n-th coefficient of T is the
    n-th coefficient of the source divided by B + n(n+1) D.
    """
    nodes, weights = npleg.leggauss(nmax // 2 + 20)

    def integrate(lo, hi, f):          # Gauss-Legendre on [lo, hi]
        x = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        return 0.5 * (hi - lo) * np.dot(weights, f(x))

    total = 0.0
    for n in range(0, nmax + 1, 2):
        # H_n = (2n+1)/2 * integral over [-1, 1] = (2n+1) * integral over [0, 1] (even integrand);
        # split at x_s, where the co-albedo jumps, so each piece is a polynomial.
        f = lambda x, n=n: _source(x, xs, p) * eval_legendre(n, x)
        H = (2 * n + 1) * (integrate(0.0, xs, f) + integrate(xs, 1.0, f))
        total += H * eval_legendre(n, xs) / (p.B + n * (n + 1) * p.D)
    return (p.T_s + p.A / p.B) / total


def grid_Q_of_xs(xs: float, grid: LatitudeGrid, p: NorthParams = NorthParams(),
                 sub: int = 64) -> float:
    """Q(x_s) solved on the model grid with the model's diffusion operator.

    The source is averaged over each band (sub points per band), so the band
    containing the ice edge gets the correct partial coverage. T at x_s is
    interpolated linearly between band centres.
    """
    frac = (np.arange(sub) + 0.5) / sub
    x_sub = grid.x_edges[:-1, None] + frac[None, :] * grid.dx[:, None]
    src = _source(x_sub, xs, p).mean(axis=1)
    M = p.B * identity(grid.n, format="csc") - p.D * grid.diffusion_operator()
    u = spsolve(M.tocsc(), src)                      # temperature per unit Q, before -A/B
    u_at_xs = np.interp(xs, grid.x, u)
    return (p.T_s + p.A / p.B) / u_at_xs


def temperature_profile(Q: float, xs: float, grid: LatitudeGrid, p: NorthParams = NorthParams(),
                        sub: int = 64) -> np.ndarray:
    """Equilibrium T per band for given Q and ice edge (for plotting and checks)."""
    frac = (np.arange(sub) + 0.5) / sub
    x_sub = grid.x_edges[:-1, None] + frac[None, :] * grid.dx[:, None]
    src = _source(x_sub, xs, p).mean(axis=1)
    M = p.B * identity(grid.n, format="csc") - p.D * grid.diffusion_operator()
    return spsolve(M.tocsc(), Q * src - p.A)
