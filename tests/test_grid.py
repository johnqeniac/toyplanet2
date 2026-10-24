"""Tests for the latitude grid and its diffusion operator.

Each test checks a property the physics depends on, not just that the code
runs: equal areas, exact energy conservation, no transport of a uniform field,
and convergence to the known analytic answer.
"""

import numpy as np
import pytest

from toyplanet.grid import LatitudeGrid


@pytest.mark.parametrize("n", [2, 9, 18, 90])
def test_bands_have_equal_area_summing_to_one(n):
    g = LatitudeGrid(n)
    assert np.allclose(g.weights, 1.0 / n)
    assert np.isclose(g.weights.sum(), 1.0)


def test_latitudes_are_symmetric_about_equator():
    g = LatitudeGrid(18)
    assert np.allclose(g.lat, -g.lat[::-1])


def test_uniform_field_is_not_transported():
    g = LatitudeGrid(18)
    T = np.full(g.n, 288.0)
    L = g.diffusion_operator()
    # Each row sums terms of size |L_ii| * T that cancel; the residue can be no
    # smaller than machine round-off (~1e-16) times that size, so the tolerance
    # is relative to it rather than an absolute number.
    scale = np.abs(L.diagonal()).max() * T.max()
    assert np.abs(L @ T).max() < 1e-13 * scale


def test_diffusion_conserves_energy_exactly():
    """Transport only moves heat between bands: the global mean tendency is zero."""
    g = LatitudeGrid(18)
    rng = np.random.default_rng(0)
    T = 250.0 + 50.0 * rng.random(g.n)
    tendency = g.diffusion_operator() @ T
    assert abs(g.mean(tendency)) < 1e-12 * np.abs(tendency).max()


def test_converges_to_legendre_eigenvalue():
    """The second Legendre polynomial P2(x) = (3x^2 - 1)/2 satisfies

        d/dx((1 - x^2) dP2/dx) = -6 P2

    exactly. The discrete operator should reproduce this, with an error that
    falls by about 4 each time the number of bands doubles (second-order
    accuracy). This is the standard analytic check for EBM transport.
    """
    errors = []
    for n in (18, 36, 72, 144):
        g = LatitudeGrid(n)
        p2 = 0.5 * (3.0 * g.x**2 - 1.0)
        err = np.abs(g.diffusion_operator() @ p2 + 6.0 * p2)
        errors.append(err.max())
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert errors[-1] < 1e-2
    assert np.all(ratios > 3.0), f"error ratios {ratios} show worse than 2nd order"


def test_rejects_too_few_bands():
    with pytest.raises(ValueError):
        LatitudeGrid(1)
