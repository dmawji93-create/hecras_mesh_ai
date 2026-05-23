"""DEM-derived feature channels for the breakline-detection model.

Each derivative is a thin, tested, pure-numpy function: 2D elevation array in,
2D feature array out, with explicit cell sizes. Higher-level CRS-aware stacking
lives outside this module (see Stage 1 Task 3).
"""

from hecras_mesh_ai.features.slope import slope

__all__ = ["slope"]
