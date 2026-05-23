"""Rasterize breakline polylines into a binary label raster.

The supervised label for Stage 2's breakline detector. Each LineString in the
input GeoDataFrame is buffered by `buffer_width / 2` on each side (producing
a band of total width `buffer_width`), then burned onto the raster grid as
1-valued pixels. Everywhere else is 0.

`buffer_width` is a hyperparameter:
  - Wider band  =>  more positive-class pixels per breakline. Easier to learn
                    (less class imbalance) but coarser predictions; spatial
                    localization of the breakline degrades.
  - Narrower band =>  fewer positives. Sharper learning target but more class
                    imbalance — most pixels are 0, the model may collapse to
                    "always predict 0" without a class-weighted loss.

A reasonable starting default is ~3-5x the DEM resolution; that's enough to
guarantee at least 1 pixel of width along any line direction even on
diagonals, with a bit of tolerance for the model to learn "near a breakline."
This default is NOT baked in — caller supplies it.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio.features
from rasterio.transform import Affine


def rasterize_breaklines(
    breaklines: gpd.GeoDataFrame,
    *,
    out_shape: tuple[int, int],
    transform: Affine,
    target_crs,
    buffer_width: float,
) -> np.ndarray:
    """Rasterize a GeoDataFrame of breakline polylines into a binary mask.

    Parameters
    ----------
    breaklines
        GeoDataFrame whose `.geometry` column contains LineStrings or
        MultiLineStrings.
        - If `.crs` is None, it is assumed to already match `target_crs`
          (no reprojection). This is the common case for rashdf reads of
          HDFs that don't embed CRS (e.g. Muncie's g04, Bald Eagle's g02).
        - If `.crs` is set and differs from `target_crs`, the geometries
          are reprojected before rasterization.
    out_shape
        (height, width) of the output raster.
    transform
        Affine transform mapping raster (row, col) to (x, y) in
        `target_crs`. Typically taken from the feature stack via
        `da.rio.transform()`.
    target_crs
        The CRS the `transform` is expressed in. Accepts anything
        compatible with `gpd.GeoDataFrame.to_crs()` (a `pyproj.CRS`,
        an EPSG int, a WKT string, etc.).
    buffer_width
        Total width of the rasterized band, in the linear units of
        `target_crs` (typically feet or meters depending on the
        projection). Must be > 0. Each LineString becomes a band of
        this total width centered on the polyline.

    Returns
    -------
    np.ndarray of shape `out_shape`, dtype uint8, values in {0, 1}.
    """
    if buffer_width <= 0:
        raise ValueError(f"buffer_width must be > 0, got {buffer_width}")
    if len(out_shape) != 2:
        raise ValueError(f"out_shape must be a 2-tuple (H, W), got {out_shape}")

    if breaklines.crs is not None and breaklines.crs != target_crs:
        breaklines = breaklines.to_crs(target_crs)

    # shapely .buffer takes the perpendicular distance from the line.
    # Total band width = 2 * perpendicular_distance.
    half_width = buffer_width / 2.0
    buffered = breaklines.geometry.buffer(half_width)

    shapes = [(geom, 1) for geom in buffered if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)

    label = rasterio.features.rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
        all_touched=True,
    )
    return label
