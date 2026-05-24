"""Probability raster -> breakline polylines (geopackage-ready).

The full chain in one function: threshold, skeletonize, trace components,
project to CRS coordinates, simplify with Douglas-Peucker, return as a
GeoDataFrame.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import LineString
from skimage.morphology import skeletonize


def _trace_component(component_pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Walk one connected skeleton component into an ordered (row, col) chain.

    Strategy: start at an endpoint (a pixel with exactly 1 neighbor in the
    component), greedily walk to unvisited neighbors. Picks an arbitrary
    branch at junctions — adequate for line-like structures, gives a
    best-effort ordering for branched ones. For pure cycles (no endpoints),
    starts at an arbitrary pixel and walks until back to start.
    """
    if len(component_pixels) < 2:
        return list(component_pixels)

    def neighbors(rc):
        r, c = rc
        return [
            (r + dr, c + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and (r + dr, c + dc) in component_pixels
        ]

    # Prefer an endpoint as start; fall back to any pixel.
    endpoint = next(
        (p for p in component_pixels if len(neighbors(p)) == 1),
        next(iter(component_pixels)),
    )

    path: list[tuple[int, int]] = [endpoint]
    visited: set[tuple[int, int]] = {endpoint}
    current = endpoint
    while True:
        unvisited = [n for n in neighbors(current) if n not in visited]
        if not unvisited:
            break
        nxt = unvisited[0]  # arbitrary; OK for line-like shapes
        path.append(nxt)
        visited.add(nxt)
        current = nxt
    return path


def probability_to_polylines(
    prob: np.ndarray,
    *,
    transform: Affine,
    target_crs,
    threshold: float = 0.5,
    min_length_pixels: int = 10,
    simplify_tolerance: float = 0.0,
) -> gpd.GeoDataFrame:
    """Convert a model probability raster into breakline polylines.

    Parameters
    ----------
    prob
        2D float array of probabilities, shape (H, W), values in [0, 1].
    transform
        Affine transform from raster (row, col) to CRS (x, y). Typically
        taken from the feature stack DataArray's `.rio.transform()`.
    target_crs
        CRS for the output GeoDataFrame. Anything compatible with geopandas.
    threshold
        Probability cutoff for the binary mask. Higher -> fewer false
        positives but misses faint breaklines. Default 0.5.
    min_length_pixels
        Skeleton components shorter than this are dropped as noise.
        Default 10 — enough to skip isolated specks while keeping any
        real breakline.
    simplify_tolerance
        Douglas-Peucker tolerance in CRS linear units. 0 disables.
        Larger = fewer vertices, coarser polylines. Default 0.

    Returns
    -------
    GeoDataFrame with a `geometry` column of LineStrings in target_crs,
    plus a `length` column in CRS units. May be empty if no positives
    survived the threshold or all components were below min_length_pixels.
    """
    if prob.ndim != 2:
        raise ValueError(f"prob must be 2D, got shape {prob.shape}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if min_length_pixels < 1:
        raise ValueError(f"min_length_pixels must be >= 1, got {min_length_pixels}")
    if simplify_tolerance < 0:
        raise ValueError(f"simplify_tolerance must be >= 0, got {simplify_tolerance}")

    # 1. Threshold to binary mask.
    mask = (prob > threshold).astype(np.uint8)
    if mask.sum() == 0:
        return gpd.GeoDataFrame({"geometry": [], "length": []}, crs=target_crs)

    # 2. Skeletonize -> 1-pixel-wide skeleton.
    skeleton = skeletonize(mask).astype(np.uint8)

    # 3. Connected components, 8-connectivity.
    structure = ndimage.generate_binary_structure(2, 2)
    labeled, n_components = ndimage.label(skeleton, structure=structure)

    # 4. Trace each component into an ordered pixel chain.
    lines: list[LineString] = []
    for component_id in range(1, n_components + 1):
        rows, cols = np.where(labeled == component_id)
        if len(rows) < min_length_pixels:
            continue
        component_pixels = set(zip(rows.tolist(), cols.tolist(), strict=True))
        ordered = _trace_component(component_pixels)
        if len(ordered) < 2:
            continue

        # 5. Convert (row, col) -> CRS (x, y) at pixel centers.
        coords = [
            (
                transform.c + transform.a * (c + 0.5),
                transform.f + transform.e * (r + 0.5),
            )
            for (r, c) in ordered
        ]
        line = LineString(coords)

        # 6. Douglas-Peucker simplify.
        if simplify_tolerance > 0:
            line = line.simplify(simplify_tolerance, preserve_topology=False)
        if line.is_empty or line.length == 0:
            continue
        lines.append(line)

    if not lines:
        return gpd.GeoDataFrame({"geometry": [], "length": []}, crs=target_crs)

    gdf = gpd.GeoDataFrame(
        {"geometry": lines, "length": [line.length for line in lines]},
        crs=target_crs,
    )
    return gdf
