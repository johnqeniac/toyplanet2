"""Phase 2 gate figure: response to a Pinatubo-like eruption, alone and with natural variability.

Usage:  python scripts/pinatubo_gate.py [output.png]      (about 30 seconds)
"""

import dataclasses
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toyplanet.config import load
from toyplanet.forcing import build_external, volcanic_pulse
from toyplanet.plant import Plant


def main(out="docs/figures/pinatubo_gate.png", peak=-3.5, years=10, seeds=8):
    base = load("configs/stage1_earth.toml")
    p = Plant(base)
    s0 = p.spin_up(tol=1e-4)
    n = years * p.steps_per_year
    gm = lambda T: T @ p.grid.weights
    _, t, T_control, _ = p.run_years(s0, years)
    t = t - p.time(s0) - 1.0                                   # years since eruption

    volc = dataclasses.replace(base.volcanic, peak=peak, start_year=1.0)
    quiet = dataclasses.replace(base, volcanic=volc,
                                variability=dataclasses.replace(base.variability, global_sigma=0.0))
    _, _, T, _ = p.run_years(s0, years, F_ext=build_external(quiet, n, p.dt, p.grid.n))
    clean = gm(T) - gm(T_control)

    members = []
    for seed in range(1, seeds + 1):
        noisy = dataclasses.replace(base, volcanic=volc,
                                    variability=dataclasses.replace(base.variability, seed=seed))
        _, _, T, _ = p.run_years(s0, years, F_ext=build_external(noisy, n, p.dt, p.grid.n))
        members.append(gm(T) - gm(T_control))

    k = np.argmin(clean)
    print(f"peak cooling {-clean[k]:.2f} K at {t[k] * 12:.1f} months after eruption")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True, gridspec_kw={"height_ratios": [1, 2.2]})
    a1.plot(t, volcanic_pulse(n, p.dt, 1.0, peak, volc.rise_years, volc.decay_years), color="#7b341e", lw=2)
    a1.set_ylabel("Forcing (W m$^{-2}$)"); a1.grid(alpha=0.3)
    a1.set_title("Pinatubo-like eruption in Toy Planet II (Earth-like land, calibrated)", fontsize=11)
    smooth = lambda x: np.convolve(x, np.ones(30) / 30, mode="same")
    for i, m in enumerate(members):
        a2.plot(t, smooth(m), color="#90a4ae", lw=0.8,
                label=f"with natural variability ({seeds} seeds, 30-day mean)" if i == 0 else None)
    a2.plot(t, clean, color="k", lw=2.2, label="eruption alone (difference from control)")
    a2.plot(1.5, -0.5, "D", color="#c05621", ms=8, label="observed: ~0.5 K at ~18 months (Soden et al. 2002)")
    a2.axhline(0, color="0.5", lw=0.8); a2.set_xlim(-1, years - 1)
    a2.set_xlabel("Years after eruption"); a2.set_ylabel("Global-mean temperature change (K)")
    a2.legend(fontsize=8, loc="lower right"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
