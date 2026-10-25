"""Phase 3 experiments: dose response and termination of a single SAI actuator.

Calibrated Earth-like planet, natural variability off (clean signal), one
equatorial injection site. Writes a two-panel figure and prints the numbers.

Usage:  python scripts/phase3_dose_termination.py [output.png]    (about 2 minutes)
"""

import dataclasses
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toyplanet.config import load
from toyplanet.plant import Plant


def main(out="docs/figures/phase3_dose_termination.png"):
    base = load("configs/stage1_earth.toml")
    base = dataclasses.replace(base, variability=dataclasses.replace(base.variability, global_sigma=0.0))
    p = Plant(base)
    s0 = p.spin_up(tol=1e-4)
    spy = p.steps_per_year
    rate_for = lambda F: F / (base.sai.forcing_per_tg * base.sai.tau_life_years)   # Tg S/yr for steady -F W m^-2
    gm = lambda T: T @ p.grid.weights
    annual = lambda T: T.reshape(-1, spy, T.shape[1]).mean(axis=1) @ p.grid.weights

    # Dose response: 15 years of steady injection, cooling over years 13-15
    years = 15
    _, _, T_ctl, _ = p.run_years(s0, years)
    doses = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
    cooling = []
    for F in doses:
        _, _, T, _ = p.run_years(s0, years, u_sai=[rate_for(F)])
        cooling.append(-(annual(T) - annual(T_ctl))[-3:].mean())
        print(f"dose -{F:5.2f} W/m2: cooling {cooling[-1] * 1000:8.2f} mK  ({cooling[-1] / F:.4f} K per W/m2)")
    cooling = np.array(cooling)

    # Termination: 10 years of injection, then stop (abrupt) or ramp down over 5 years
    total = 25
    _, _, T_ctl, _ = p.run_years(s0, total)
    cases = {}
    for label, F, ramp in [("test scale, -0.1 W/m2, abrupt stop", 0.1, 0),
                           ("deployment scale, -2 W/m2, abrupt stop", 2.0, 0),
                           ("deployment scale, -2 W/m2, 5-yr ramp-down", 2.0, 5)]:
        sched = np.zeros((total * spy, 1))
        sched[:10 * spy] = rate_for(F)
        if ramp:
            n = ramp * spy
            sched[10 * spy:10 * spy + n, 0] = rate_for(F) * np.linspace(1, 0, n)
        _, _, T, _ = p.run_years(s0, total, u_sai=sched)
        d = gm(T) - gm(T_ctl)
        max_rate = np.diff(annual(T) - annual(T_ctl))[9:].max()
        cases[label] = (d, max_rate)
        print(f"{label}: cooling at year 10 {-d[10 * spy - 1]:.3f} K; max warming rate after stopping {max_rate:.3f} K/yr")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.loglog(doses, cooling, "o", color="#2b6cb0", ms=7, label="model (years 13-15 of steady injection)")
    a1.loglog(doses, doses * cooling[0] / doses[0], "k--", lw=1, label="exact proportionality to smallest dose")
    a1.set_xlabel("Steady SAI forcing (W m$^{-2}$, magnitude)"); a1.set_ylabel("Global-mean cooling (K)")
    a1.set_title(f"Dose response: proportional within {100 * np.ptp(cooling / doses) / np.mean(cooling / doses):.1f}% "
                 "over 300x in dose", fontsize=10)
    a1.legend(fontsize=8); a1.grid(alpha=0.3, which="both")
    yr = np.arange(total * spy) / spy
    colors = ["#2b6cb0", "#c05621", "#2f855a"]
    for (label, (d, r)), c in zip(cases.items(), colors):
        a2.plot(yr, -d, color=c, lw=1.8, label=f"{label} (max {r:.3f} K/yr)")
    a2.axvline(10, color="0.5", lw=0.8, ls=":"); a2.text(10.2, 0.02, "injection stops / ramp begins", fontsize=8, color="0.4")
    a2.set_xlabel("Years from start of injection"); a2.set_ylabel("Global-mean cooling (K)")
    a2.set_title("Termination: recovery and warming rate after stopping", fontsize=10)
    a2.legend(fontsize=7.5, loc="center right"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
