"""Slope from a regular DEM grid via Horn (1981).

Horn's formula is the de-facto standard for raster slope: it's what GRASS,
ArcGIS, gdaldem, and rasterio's slope all default to. It estimates the gradient
from a 3x3 neighborhood with weights (1, 2, 1) on the rows/columns parallel
to the differentiation direction:

    dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cellsize_x)
    dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cellsize_y)

where the 3x3 window is

    a b c
    d e f
    g h i

Slope is then arctan(sqrt((dz/dx)^2 + (dz/dy)^2)).
"""

from __future__ import annotations

import numpy as np

_Units = ("degrees", "radians", "percent")


def slope(
    z: np.ndarray,
    cellsize_x: float,
    cellsize_y: float,
    *,
    units: str = "degrees",
) -> np.ndarray:
    """Compute slope of a DEM via Horn (1981).

    Parameters
    ----------
    z
        2D elevation array, shape (H, W). May contain NaN.
    cellsize_x, cellsize_y
        Cell size in the projection's linear units (typically meters or feet).
        Must be positive.
    units
        "degrees" (default), "radians", or "percent".

    Returns
    -------
    Slope array of shape (H, W) and dtype float64. Border rows/cols are NaN
    (the 3x3 window cannot be centered there).
    """
    if z.ndim != 2:
        raise ValueError(f"z must be 2D, got shape {z.shape}")
    if cellsize_x <= 0 or cellsize_y <= 0:
        raise ValueError(f"cellsize must be positive, got ({cellsize_x}, {cellsize_y})")
    if units not in _Units:
        raise ValueError(f"units must be one of {_Units}, got {units!r}")

    z = np.asarray(z, dtype=np.float64)
    out = np.full_like(z, np.nan, dtype=np.float64)

    # 3x3 neighbourhood slices into the interior of the array.
    a = z[:-2, :-2]
    b = z[:-2, 1:-1]
    c = z[:-2, 2:]
    d = z[1:-1, :-2]
    f = z[1:-1, 2:]
    g = z[2:, :-2]
    h = z[2:, 1:-1]
    i = z[2:, 2:]

    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * cellsize_x)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * cellsize_y)

    slope_rad = np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy))
    out[1:-1, 1:-1] = slope_rad

    if units == "radians":
        return out
    if units == "degrees":
        return np.degrees(out)
    return np.tan(out) * 100.0  # percent
