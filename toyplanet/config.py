"""Run configuration: one TOML file, checked against a schema before anything runs.

Design rules, each a lesson from Toy Planet I:

1. Every parameter lives here, once. Physics modules receive their values from
   a Config object; none of them hard-codes a number that a planet variant
   might change.
2. The schema is the set of dataclasses below. A key in the file that the
   schema does not name is an ERROR, reported with the closest valid name.
   (Toy Planet I's config had a "carbon_scenario" key that nothing read; the
   run went ahead and silently ignored it.)
3. Types and physical ranges are checked. All problems are collected and
   reported together, so one run of the loader shows everything to fix.
4. Keys left out of the file take the defaults below. The fully resolved
   configuration, defaults included, is written into each run's directory, so
   every result can be traced to the exact numbers that produced it.

Units are stated in each field's metadata. Temperatures are in degrees
Celsius, the convention of the Budyko longwave formula A + B*T.

Values marked "provisional" are starting points from the energy-balance-model
literature (North, Cahalan & Coakley 1981; Held et al. 2010) and will be set during
phase 1 validation.
"""

from __future__ import annotations

import dataclasses
import difflib
import sys
import typing
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.9 and 3.10: same parser, installed as the 'tomli' package
    import tomli as tomllib

from toyplanet.insolation import Orbit


def _p(default, units, lo=None, hi=None, doc=""):
    """A schema field: default value plus units, allowed range and description."""
    return field(default=default, metadata={"units": units, "min": lo, "max": hi, "doc": doc})


@dataclass(frozen=True)
class GridConfig:
    n: int = _p(36, "bands", 2, 1000, "number of equal-area latitude bands")


@dataclass(frozen=True)
class InsolationConfig:
    seasonal: bool = _p(True, "", doc="false: every day gets the annual-mean insolation (benchmarks)")


@dataclass(frozen=True)
class RadiationConfig:
    olr_a: float = _p(203.3, "W m^-2", 0, 400, "OLR at 0 C (Budyko A); provisional")
    olr_b: float = _p(2.09, "W m^-2 K^-1", 0.1, 5, "OLR slope (Budyko B), local feedback; provisional")


@dataclass(frozen=True)
class AlbedoConfig:
    ice: float = _p(0.62, "1", 0, 1, "albedo of ice-covered surface; provisional")
    free: float = _p(0.319, "1", 0, 1, "ice-free albedo, global-mean part (1 - a0, North et al. 1981 eq. 18)")
    free_p2: float = _p(0.202, "1", -0.5, 0.5,
                        "ice-free albedo P2(sin lat) coefficient (-a2, North et al. 1981 eq. 18): brighter at low sun")
    t_freeze: float = _p(-10.0, "C", -40, 10, "centre of the ice transition; provisional")
    t_width: float = _p(2.0, "K", 0.1, 20, "width of the smooth (tanh) transition")


@dataclass(frozen=True)
class TransportConfig:
    diffusivity: float = _p(0.649, "W m^-2 K^-1", 0, 10, "D in d/dx((1-x^2) D dT/dx); provisional")


@dataclass(frozen=True)
class OceanConfig:
    c_mixed: float = _p(8.0, "W yr m^-2 K^-1", 0.1, 100, "surface-layer heat capacity (about 60 m of water)")
    c_deep: float = _p(100.0, "W yr m^-2 K^-1", 1, 2000, "deep-layer heat capacity")
    gamma: float = _p(0.7, "W m^-2 K^-1", 0, 10, "surface-deep exchange coefficient")


@dataclass(frozen=True)
class ForcingConfig:
    ghg: float = _p(0.0, "W m^-2", -20, 20, "greenhouse forcing relative to the reference state")


@dataclass(frozen=True)
class TimeConfig:
    dt_days: float = _p(1.0, "days", 0.01, 30, "time step")
    spinup_years: float = _p(30.0, "years", 0, 10000, "spin-up to equilibrium before the run proper")
    run_years: float = _p(50.0, "years", 0, 10000, "length of the run after spin-up")


@dataclass(frozen=True)
class RunConfig:
    name: str = _p("default", "", doc="label used in the run directory name")
    output_root: str = _p("runs", "", doc="directory under which run directories are created")


@dataclass(frozen=True)
class Config:
    """The whole configuration. Each attribute is one [section] of the TOML file."""

    grid: GridConfig = field(default_factory=GridConfig)
    orbit: Orbit = field(default_factory=Orbit)
    insolation: InsolationConfig = field(default_factory=InsolationConfig)
    radiation: RadiationConfig = field(default_factory=RadiationConfig)
    albedo: AlbedoConfig = field(default_factory=AlbedoConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    ocean: OceanConfig = field(default_factory=OceanConfig)
    forcing: ForcingConfig = field(default_factory=ForcingConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    run: RunConfig = field(default_factory=RunConfig)


class ConfigError(ValueError):
    """Raised with every problem found in a config file, one per line."""


# --------------------------------------------------------------------------
# Loading and checking
# --------------------------------------------------------------------------

def load(path) -> Config:
    """Read a TOML file and return a checked Config. Raises ConfigError."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return from_dict(data)


def from_dict(data: dict) -> Config:
    """Build a Config from nested dicts, collecting every problem before raising."""
    problems: list[str] = []
    cfg = _build(Config, data, "", problems)
    if problems:
        raise ConfigError("invalid configuration:\n  " + "\n  ".join(problems))
    return cfg


def _build(cls, data, where, problems):
    if not isinstance(data, dict):
        problems.append(f"{where or 'top level'}: expected a table of settings, got {data!r}")
        return cls()
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}

    for key in data:
        if key not in fields:
            near = difflib.get_close_matches(key, fields, n=1)
            hint = f" (did you mean '{near[0]}'?)" if near else ""
            valid = ", ".join(fields)
            problems.append(f"{where}{key}: unknown setting{hint}; valid here: {valid}")

    values = {}
    for name, f in fields.items():
        if name not in data:
            continue
        value, label, kind = data[name], f"{where}{name}", hints[name]
        if dataclasses.is_dataclass(kind):
            values[name] = _build(kind, value, label + ".", problems)
        else:
            checked = _check_scalar(value, kind, f.metadata, label, problems)
            if checked is not None:
                values[name] = checked
    try:
        return cls(**values)
    except ValueError as err:          # a section's own __post_init__ check, e.g. Orbit
        problems.append(f"{where.rstrip('.') or 'top level'}: {err}")
        return cls()


def _check_scalar(value, kind, meta, label, problems):
    # bool is a subclass of int in Python, so 'true' would otherwise pass as 1.
    if isinstance(value, bool) and kind is not bool:
        problems.append(f"{label}: expected {kind.__name__}, got boolean {value}")
        return None
    if kind is float and isinstance(value, int):
        value = float(value)           # 'n = 2' where a float is wanted is fine
    if not isinstance(value, kind):
        problems.append(f"{label}: expected {kind.__name__}, got {type(value).__name__} {value!r}")
        return None
    lo, hi, units = meta.get("min"), meta.get("max"), meta.get("units", "")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        problems.append(f"{label}: {value} {units} is outside the allowed range [{lo}, {hi}]")
        return None
    return value


# --------------------------------------------------------------------------
# Writing the resolved configuration into a run directory
# --------------------------------------------------------------------------

def to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def to_toml(cfg: Config) -> str:
    """Resolved configuration as TOML text, with units as trailing comments.

    The schema is two levels deep (sections of scalars), so a small writer
    suffices; the standard library reads TOML but does not write it.
    """
    lines = ["# Resolved Toy Planet II configuration (defaults filled in).", ""]
    for section in dataclasses.fields(cfg):
        sub = getattr(cfg, section.name)
        lines.append(f"[{section.name}]")
        for f in dataclasses.fields(sub):
            v = getattr(sub, f.name)
            text = f'"{v}"' if isinstance(v, str) else repr(v).lower() if isinstance(v, bool) else repr(v)
            units = f.metadata.get("units", "")
            lines.append(f"{f.name} = {text}" + (f"    # {units}" if units else ""))
        lines.append("")
    return "\n".join(lines)


def write_resolved(cfg: Config, run_dir) -> Path:
    path = Path(run_dir) / "config_resolved.toml"
    path.write_text(to_toml(cfg))
    return path
