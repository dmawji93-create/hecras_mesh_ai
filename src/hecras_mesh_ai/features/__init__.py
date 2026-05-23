"""DEM-derived feature channels for the breakline-detection model.

Each derivative is a thin, tested, pure-numpy function: 2D elevation array in,
2D feature array out, with explicit cell sizes. Higher-level CRS-aware stacking
lives outside this module (see Stage 1 Task 3).
"""

from hecras_mesh_ai.features.aspect import aspect_sincos
from hecras_mesh_ai.features.conditioning import patch_isolated_nan
from hecras_mesh_ai.features.plan_curvature import plan_curvature
from hecras_mesh_ai.features.profile_curvature import profile_curvature
from hecras_mesh_ai.features.slope import slope
from hecras_mesh_ai.features.stacker import FEATURE_CHANNELS, stack_dem_features

__all__ = [
    "FEATURE_CHANNELS",
    "aspect_sincos",
    "patch_isolated_nan",
    "plan_curvature",
    "profile_curvature",
    "slope",
    "stack_dem_features",
]
