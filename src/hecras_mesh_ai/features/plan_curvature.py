"""Plan curvature from a regular DEM grid (Heerdegen & Beran 1982).

Plan curvature is the curvature of the contour line passing through a pixel
— the horizontal bend of the slice through the terrain at that elevation.
Hydraulically:

  - Positive  =>  contour curves outward (divergent)  =>  ridge-like; flow
                  disperses laterally.
  - Zero      =>  straight contour                   =>  planar flow.
  - Negative  =>  contour curves inward (convergent) =>  valley-like; flow
                  concentrates.

Breaklines align with both ridges and channel banks, so |plan curvature| is
a strong locator and its sign discriminates ridge from valley.

Formula (Heerdegen & Beran 1982; same as ArcGIS, SAGA, gdaldem "planc"):

    k_plan = -(q^2 r - 2 p q s + p^2 t) / (p^2 + q^2)^(3/2)

with p, q, r, t, s the first and second partial derivatives of z.

Convention insensitivity. Unlike aspect, plan curvature is invariant under
y-axis flip — q and s both change sign, but appear in q^2 and p*q*s terms
that are quadratic in the flipped quantities. Row-down vs row-up rasters
yield identical plan-curvature output. (See aspect.py for the row-direction
discussion; it does NOT apply here.)

At a flat point (p^2 + q^2 = 0) the contour-line curvature is mathematically
undefined; we return 0, by the same convention used for aspect's (0, 0) at
flat points. NaN in the input stencil propagates to NaN in the output.
"""

from __future__ import annotations

import numpy as np


def plan_curvature(
    z: np.ndarray,
    cellsize_x: float,
    cellsize_y: float,
) -> np.ndarray:
    """Compute plan (contour) curvature of a DEM.

    Parameters
    ----------
    z
        2D elevation array, shape (H, W). May contain NaN.
    cellsize_x, cellsize_y
        Cell size in the projection's linear units. Must be positive.

    Returns
    -------
    Plan-curvature array of shape (H, W) and dtype float64, in units of
    inverse length (1 / linear-unit-of-the-projection). Border rows/cols
    are NaN. Flat interior cells are 0. NaN inputs propagate as NaN.
    """
    if z.ndim != 2:
        raise ValueError(f"z must be 2D, got shape {z.shape}")
    if cellsize_x <= 0 or cellsize_y <= 0:
        raise ValueError(f"cellsize must be positive, got ({cellsize_x}, {cellsize_y})")

    z = np.asarray(z, dtype=np.float64)
    out = np.full_like(z, np.nan, dtype=np.float64)

    # 3x3 neighborhood named as in the docstring stencil:
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

    # First partial derivatives (central differences).
    p = (z6 - z4) / (2.0 * dx)
    q = (z8 - z2) / (2.0 * dy)

    # Second partial derivatives.
    r = (z6 - 2.0 * z5 + z4) / (dx * dx)
    t = (z8 - 2.0 * z5 + z2) / (dy * dy)
    # Mixed partial: standard 4-corner stencil.
    s = (z9 - z7 - z3 + z1) / (4.0 * dx * dy)

    denom_sq = p * p + q * q
    numerator = -(q * q * r - 2.0 * p * q * s + p * p * t)

    # NaN tracking. A "flat" check (denom == 0) alone is not enough to decide
    # the output, because a flat-on-its-axes pixel can have NaN entering only
    # through the cross-partial s — in which case p = q = 0 cleanly, denom = 0,
    # but the cell is actually next to nodata. We must check NaN explicitly.
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
        interior = numerator / (denom_sq**1.5)
    interior = np.where(flat, 0.0, interior)
    interior = np.where(nan_in_stencil, np.nan, interior)

    out[1:-1, 1:-1] = interior
    return out
