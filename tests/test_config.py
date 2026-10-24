"""Tests for config loading: the checks exist to stop bad runs before they start."""

import textwrap

import pytest

from toyplanet import config
from toyplanet.config import Config, ConfigError, from_dict, load, to_toml


def test_empty_config_gives_all_defaults():
    assert from_dict({}) == Config()


def test_example_file_loads(tmp_path):
    cfg = load("configs/earth_default.toml")
    assert cfg.grid.n == 36 and cfg.run.name == "earth_default"


def test_unknown_key_is_an_error_with_suggestion():
    """The Toy Planet I failure: a setting nothing reads must not pass silently."""
    with pytest.raises(ConfigError) as err:
        from_dict({"forcing": {"gh": 1.0}})
    assert "forcing.gh: unknown setting (did you mean 'ghg'?)" in str(err.value)


def test_unknown_section_is_an_error():
    with pytest.raises(ConfigError, match="carbon_scenario: unknown setting"):
        from_dict({"carbon_scenario": "rising"})


def test_all_problems_reported_at_once():
    with pytest.raises(ConfigError) as err:
        from_dict({"grid": {"n": 1}, "ocean": {"c_mixd": 8.0}, "albedo": {"ice": 1.5}})
    msg = str(err.value)
    assert "grid.n" in msg and "ocean.c_mixd" in msg and "albedo.ice" in msg


def test_wrong_type_rejected():
    with pytest.raises(ConfigError, match="grid.n: expected int, got str"):
        from_dict({"grid": {"n": "36"}})


def test_boolean_is_not_a_number():
    """Python treats True as the integer 1; the loader must not."""
    with pytest.raises(ConfigError, match="expected float, got boolean"):
        from_dict({"forcing": {"ghg": True}})


def test_integer_accepted_where_float_expected():
    assert from_dict({"forcing": {"ghg": 4}}).forcing.ghg == 4.0


def test_out_of_range_rejected_with_units():
    with pytest.raises(ConfigError, match=r"time.dt_days: 100.0 days is outside"):
        from_dict({"time": {"dt_days": 100.0}})


def test_section_own_validation_is_reported():
    """Orbit checks its eccentricity itself; the loader reports that as a config error."""
    with pytest.raises(ConfigError, match="orbit: eccentricity"):
        from_dict({"orbit": {"eccentricity": 1.2}})


def test_resolved_config_round_trips(tmp_path):
    """What is written into a run directory must reload to the identical config."""
    original = from_dict({"grid": {"n": 72}, "run": {"name": "trip"}, "forcing": {"ghg": 3.7}})
    path = config.write_resolved(original, tmp_path)
    assert load(path) == original
    assert "W m^-2" in to_toml(original)
