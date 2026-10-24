"""Latitude grid and the transport operator.

The planet is divided into N latitude bands of EQUAL AREA. The natural
coordinate for that is x = sin(latitude): on a sphere, the area between two
latitudes is proportional to the difference of their sines, so bands that are
equally spaced in x have equal area. Two consequences make the rest of the
code simpler:

  * the global mean of any field is the plain average over bands
    (every band carries the same weight 1/N);
  * the meridional diffusion operator takes the compact form

        (1/cos phi) d/dphi( cos phi dT/dphi ) = d/dx( (1 - x^2) dT/dx )

    which is the form used in the classic energy-balance-model literature
    (North 1975; North, Cahalan & Coakley 1981).

The factor (1 - x^2) = cos^2(phi) vanishes at the poles, so no heat flows
through them: the boundary condition comes from the geometry, with no special
case in the code.

Interface rule: the physics modules never assume latitude bands. They ask the
grid for weights, a mean, and a diffusion operator. A 2-D grid in phase 6 will
provide the same three things.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class LatitudeGrid:
    """N equal-area latitude bands from the south pole to the north pole.

    Attributes
    ----------
    n : int
        Number of bands.
    x_edges : ndarray, shape (n+1,)
        Band edges in x = sin(latitude), from -1 to +1.
    x : ndarray, shape (n,)
        Band centres in x (midpoints of the edges).
    lat : ndarray, shape (n,)
        Band centres in degrees latitude.
    weights : ndarray, shape (n,)
        Area fraction of each band; all equal to 1/n, summing to 1.
    """

    def __init__(self, n: int):
        if n < 2:
            raise ValueError(f"need at least 2 bands, got {n}")
        self.n = n
        self.x_edges = np.linspace(-1.0, 1.0, n + 1)
        self.x = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        self.lat = np.degrees(np.arcsin(self.x))
        self.dx = np.diff(self.x_edges)          # all equal to 2/n
        self.weights = self.dx / 2.0             # area fractions, sum to 1

    def mean(self, field: np.ndarray) -> float:
        """Area-weighted global mean of a field defined on the bands."""
        return float(np.dot(self.weights, field))

    def diffusion_operator(self) -> sp.csr_matrix:
        """Sparse matrix L such that (L @ T)[i] approximates

            d/dx( (1 - x^2) dT/dx )   at band i.

        Multiply by a diffusion coefficient D (W m^-2 K^-1) to get the
        transport tendency in W m^-2.

        Construction (finite volume): the flux across each interior edge is

            flux[e] = -(1 - x_e^2) * (T[i+1] - T[i]) / (x[i+1] - x[i])

        and a band's tendency is minus the difference of the fluxes on its two
        edges, divided by its width. Fluxes through the polar edges are zero
        because (1 - x^2) = 0 there.

        Why this form: whatever leaves one band through an edge enters its
        neighbour through the same edge, so the operator conserves energy
        exactly (to round-off). The test suite checks this.
        """
        n = self.n
        xe = self.x_edges[1:-1]                    # interior edges, n-1 of them
        conductance = (1.0 - xe**2) / np.diff(self.x)
        # Band i loses heat to band i+1 at rate conductance[i] * (T[i] - T[i+1]),
        # divided by band width dx[i] to turn a flux into a tendency.
        upper = conductance / self.dx[:-1]          # coupling of i to i+1
        lower = conductance / self.dx[1:]           # coupling of i+1 to i
        diag = np.zeros(n)
        diag[:-1] -= upper
        diag[1:] -= lower
        return sp.diags([lower, diag, upper], offsets=[-1, 0, 1], format="csr")
