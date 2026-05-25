"""HEC-RAS automation harness (Stage 4).

The engineering layer that lets the system programmatically write HEC-RAS
geometry HDFs, launch runs, and parse results — all without manual RAS
Mapper interaction. Required by Phase B (Stage 5) onward; can be
developed in parallel with the breakline-model work in Stages 2-3.

  - inspect       : dump the structure of any HEC-RAS HDF for schema
                    reverse-engineering (Task 1)
  - write_geom    : breakline-replacement writer (Task 2)
  - run           (next): plan launcher (Task 3)
  - results       (next): results parser (Task 4)
"""

from hecras_mesh_ai.harness.inspect import dump_structure, walk_hdf
from hecras_mesh_ai.harness.write_geom import (
    Breakline,
    read_breaklines,
    replace_breaklines,
)

__all__ = [
    "Breakline",
    "dump_structure",
    "read_breaklines",
    "replace_breaklines",
    "walk_hdf",
]
