"""Guard against the Toy Planet I failure: an import that ran a simulation."""

import importlib


def test_import_has_no_side_effects(capsys):
    importlib.import_module("toyplanet")
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
