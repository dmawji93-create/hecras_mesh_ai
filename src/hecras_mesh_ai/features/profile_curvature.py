"""Profile curvature from a regular DEM grid (Mitasova & Hofierka 1993).

Profile curvature is the curvature of the surface in the direction of steepest
descent — the vertical slice through the terrain along the slope line.
Hydraulically:

  - Positive  =>  upwardly convex profile  =>  slope steepens downhill
                  =>  flow accelerates.
  - Zero      =>  linear slope             =>  constant velocity gradient.
  - Negative  =>  upwardly concave profile =>  slope flattens downhill
                  =>  flow decelerates.

Where plan curvature discriminates ridges from valleys, profile curvature
discriminates accelerating from decelerating reaches. Together they describe
the geomorphic anatomy of where breaklines belong.

Formula (Mitasova & Hofierka 1993; sign convention matches ArcGIS):

    k_prof = -(p^2 r + 2 p q s + q^2 t) / ((p^2 + q^2) * (1 + p^2 + q^2)^(3/2))

The (1 + p^2 + q^2)^(3/2) factor is the surface metric — profile curvature
is defined on the curved surface, not on the planar projection. For shallow
slopes the factor approaches 1; for steep slopes it suppresses magnitude.

Convention insensitivity. Like plan curvature, profile curvature is invariant
under y-axis flip — q and s both change sign but appear in q^2 and p*q*s
which are quadratic in the flipped quantities. The row-direction breadcrumb
in aspect.py does NOT apply here.

At a flat point (p^2 + q^2 = 0) the slope line is undefined, so profile
curvature is undefined; we return 0 by convention. NaN in the input stencil
propagates to NaN in the output via the same explicit tracking used in
plan_curvature.
"""

from __future__ import annotations

import numpy as np


def profile_curvature(
    z: np.ndarray,
    cellsize_x: float,
    cellsize_y: float,
) -> np.ndarray:
    """Compute profile (slope-line) curvature of a DEM.

    Parameters
    ----------
    z
        2D elevation array, shape (H, W). May contain NaN.
    cellsize_x, cellsize_y
        Cell size in the projection's linear units. Must be positive.

    Returns
    -------
    Profile-curvature array of shape (H, W) and dtype float64. Border
    rows/cols are NaN. Flat interior cells are 0. NaN inputs propagate.
    """
    if z.ndim != 2:
        raise ValueError(f"z must be 2D, got shape {z.shape}")
    if cellsize_x <= 0 or cellsize_y <= 0:
        raise ValueError(f"cellsize must be positive, got ({cellsize_x}, {cellsize_y})")

    z = np.asarray(z, dtype=np.float64)
    out = np.full_like(z, np.nan, dtype=np.float64)

    # 3x3 neighborhood:
    #   z1 z2 z3
    #   z4 z5 z6
    #   z7 z8 z9
    z1 = z[:-2, :-2]
    z2 = z[:-2, 1:-1]
    z3 = z[:-2, 2:]
    z4 = z[1:-1, :-2]
    z5 = z[1:-1, 1:-1]
    z6 = z[1:-1, 2:]
    z7 = z[2:, :-2]
    z8 = z[2:, 1:-1]
    z9 = z[2:, 2:]

    dx = float(cellsize_x)
    dy = float(cellsize_y)

    p = (z6 - z4) / (2.0 * dx)
    q = (z8 - z2) / (2.0 * dy)
    r = (z6 - 2.0 * z5 + z4) / (dx * dx)
    t = (z8 - 2.0 * z5 + z2) / (dy * dy)
    s = (z9 - z7 - z3 + z1) / (4.0 * dx * dy)

    denom_sq = p * p + q * q
    surface_factor = (1.0 + denom_sq) ** 1.5
    numerator = -(p * p * r + 2.0 * p * q * s + q * q * t)

    nan_in_stencil = (
        np.isnan(z1)
        | np.isnan(z2)
        | np.isnan(z3)
        | np.isnan(z4)
        | np.isnan(z5)
        | np.isnan(z6)
        | np.isnan(z7)
        | np.isnan(z8)
        | np.isnan(z9)
    )
    flat = (denom_sq == 0) & ~nan_in_stencil

    with np.errstate(invalid="ignore", divide="ignore"):
        interior = numerator / (denom_sq * surface_factor)
    interior = np.where(flat, 0.0, interior)
    interior = np.where(nan_in_stencil, np.nan, interior)

    out[1:-1, 1:-1] = interior
    return out
