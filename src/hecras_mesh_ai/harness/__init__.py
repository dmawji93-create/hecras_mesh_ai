"""HEC-RAS automation harness (Stage 4).

The engineering layer that lets the system programmatically write HEC-RAS
geometry HDFs, launch runs, and parse results — all without manual RAS
Mapper interaction. Required by Phase B (Stage 5) onward; can be
developed in parallel with the breakline-model work in Stages 2-3.

  - inspect       : dump the structure of any HEC-RAS HDF for schema
                    reverse-engineering (Task 1)
  - write_geom    : breakline-replacement writer (Task 2)
  - launch        : plan launcher (Task 3)
  - results       : 2D unsteady results parser (Task 4)
"""

from hecras_mesh_ai.harness.inspect import dump_structure, walk_hdf
from hecras_mesh_ai.harness.launch import (
    RasInstall,
    RunResult,
    find_ras_install,
    run_plan,
)
from hecras_mesh_ai.harness.results import (
    list_2d_flow_areas,
    max_depth,
    max_face_velocity,
    max_water_surface,
)
from hecras_mesh_ai.harness.write_geom import (
    Breakline,
    read_breaklines,
    replace_breaklines,
)

__all__ = [
    "Breakline",
    "RasInstall",
    "RunResult",
    "dump_structure",
    "find_ras_install",
    "list_2d_flow_areas",
    "max_depth",
    "max_face_velocity",
    "max_water_surface",
    "read_breaklines",
    "replace_breaklines",
    "run_plan",
    "walk_hdf",
]
