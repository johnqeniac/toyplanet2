# Toy Planet II

A computational existence proof: across a family of planets whose physics
includes unknown feedbacks, a closed-loop actuator portfolio (stratospheric
aerosol at many injection sites, marine cloud brightening, cirrus thinning,
warming agents) drives tracked climate fields to a target, except for a
residual that is explicitly characterized.

It is also a test bench for claims about how intervention would work: a claim
is encoded as a planet variant or an actuator property and run through the
same experiments.

Design document: https://claude.ai/artifact/DMLLoH6CpcNcMfu9HqkTr4
Dated design snapshots are kept in `docs/`.

## Status

Phase 1 (seasonal energy-balance plant): in progress. Done: grid and transport operator (grid.py); daily-mean insolation (insolation.py); validated configuration (config.py, configs/earth_default.toml); albedo, radiation and ocean exchange (physics/); the assembled plant with semi-implicit time stepping and spin-up (plant.py). Remaining for phase 1: calibration against the validation targets.

## Layout (planned)

```
toyplanet/      the package: grid, insolation, physics, plant, actuators,
                sensors, controller, analysis, run.py
tests/          one test file per module, plus end-to-end runs
docs/           dated exports of the design document
runs/           run output, one time-stamped directory per run (not committed)
```

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest
```
