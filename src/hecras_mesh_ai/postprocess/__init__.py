"""Post-processing: model probability map -> HEC-RAS-importable polylines.

The model output is a continuous (H, W) probability raster in [0, 1].
HEC-RAS modelers need vector polylines (a geopackage of LineStrings).
This module bridges the two:

  threshold -> binary mask
            -> skeletonize (1-pixel skeleton)
            -> connected components
            -> trace each component as an ordered pixel chain
            -> convert pixels to CRS coordinates via the affine transform
            -> Douglas-Peucker simplify
            -> pack with CRS as GeoDataFrame
"""

from hecras_mesh_ai.postprocess.breaklines import probability_to_polylines
from hecras_mesh_ai.postprocess.metrics import BufferedMetrics, buffered_iou_f1

__all__ = [
    "BufferedMetrics",
    "buffered_iou_f1",
    "probability_to_polylines",
]
