"""Label rasters for the breakline-detection model.

A "label" is what the supervised model learns to predict — the y in the (x, y)
training pair. For Phase A this is the binary breakline raster: 1 where a
pixel falls within a buffer around an expert breakline polyline, 0 elsewhere.

Phase B will add refinement-region labels here as a second module.
"""

from hecras_mesh_ai.labels.breaklines import rasterize_breaklines

__all__ = ["rasterize_breaklines"]
