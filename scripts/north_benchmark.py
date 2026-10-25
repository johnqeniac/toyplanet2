"""Reproduce the North, Cahalan & Coakley (1981) ice-edge benchmark figure.

Plots the exact Q(x_s) curve (stable and unstable branches) and the ice edge
found by the time-stepped plant as the solar input is lowered and then raised,
showing the snowball bifurcation and the hysteresis loop.

Usage:  python scripts/north_benchmark.py [output.png]     (takes about a minute)
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toyplanet.benchmarks.north import NorthParams, legendre_Q_of_xs
from toyplanet.config import from_dict
from toyplanet.diagnostics import ice_edge_north
from toyplanet.plant import Plant, State


def main(out="docs/figures/north_benchmark.png"):
    P = NorthParams()

    def plant(frac):
        return Plant(from_dict({"grid": {"n": 60}, "insolation": {"seasonal": False},
                                "orbit": {"solar_constant": 4 * frac * P.Q0},
                                "albedo": {"t_width": 0.3}}))

    sweeps = {"down": np.round(np.arange(1.06, 0.84, -0.01), 3),
              "up": np.round(np.arange(0.86, 1.62, 0.04), 3)}
    results, s = {}, None
    for name, fracs in sweeps.items():
        out_pts = []
        for f in fracs:
            p = plant(f)
            start = None if s is None else State(s.T, s.T_d, 0)   # continue from previous climate
            s = p.spin_up(start, tol=1e-4)
            out_pts.append((f, ice_edge_north(p.grid.x, s.T, P.T_s)))
            print(f"{name:4s} Q/Q0 = {f:.2f}  ice edge {np.degrees(np.arcsin(out_pts[-1][1])):5.1f} deg")
        results[name] = np.array(out_pts)

    xs = np.linspace(0.02, 0.9995, 500)
    Q = np.array([legendre_Q_of_xs(x, P, 300) for x in xs]) / P.Q0
    stable = np.gradient(Q, xs) > 0
    lat = np.degrees(np.arcsin(xs))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(np.where(stable, Q, np.nan), lat, "k-", lw=2, label="exact solution, stable (dQ/dx$_s$ > 0)")
    ax.plot(np.where(~stable, Q, np.nan), lat, "k--", lw=1.3, label="exact solution, unstable")
    d, u = results["down"], results["up"]
    ax.plot(d[:, 0], np.degrees(np.arcsin(d[:, 1])), "o", color="#2b6cb0", ms=5,
            label="time-stepped plant, Q decreasing")
    ax.plot(u[:, 0], np.degrees(np.arcsin(u[:, 1])), "s", mfc="none", color="#c05621", ms=6,
            label="time-stepped plant, Q increasing")
    ax.axvline(1.0, color="0.6", lw=0.8)
    ax.text(1.003, 3, "present Q", fontsize=8, color="0.4")
    ax.set_xlabel("Solar input  Q / Q$_0$   (Q$_0$ = 335 W m$^{-2}$)")
    ax.set_ylabel("Ice-edge latitude (deg)")
    ax.set_title("North (1981) benchmark reproduced: ice edge vs solar input, with hysteresis", fontsize=11)
    ax.set_xlim(0.84, 1.62); ax.set_ylim(-2, 92); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
