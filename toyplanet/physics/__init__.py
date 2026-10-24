"""Physics modules.

Each module provides pure functions: they take temperatures (and other state)
plus a config section, and return a flux in W m^-2 or a dimensionless factor.
No module keeps hidden state, reads files, or calls another physics module;
plant.py is the only place they are combined.
"""
