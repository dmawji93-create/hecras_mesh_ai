"""Aspect from a regular DEM grid, encoded as (sin, cos) for ML.

Aspect is the compass direction the terrain surface faces — the downslope
direction projected onto the horizontal plane. In GIS convention it's the
azimuth measured clockwise from north: 0 = N, 90 = E, 180 = S, 270 = W.

Aspect is a *circular* variable: 359 and 1 mean almost the same thing but
differ numerically by 358. Feeding aspect-in-degrees directly to a CNN forces
it to learn a wraparound it cannot represent. The standard ML treatment is to
encode aspect as a 2D unit vector — the sin and cos components on the unit
circle. At flat points the gradient magnitude is zero, so the unit vector
collapses to (0, 0), which is a valid value meaning "no preferred direction."
This avoids NaN propagation and gives the model a smooth, continuous signal.

Convention note: in a north-up raster, row index increases *southward*, so the
Horn dz/dy (computed by row differencing) is the change going *south* — the
opposite sign from change-going-north. The aspect math below flips that sign
back so the returned compass azimuth aligns with geographic north.
"""

from __future__ import annotations

import numpy as np


def aspect_sincos(
    z: np.ndarray,
    cellsize_x: float,
    cellsize_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute aspect as a (sin, cos) pair on the unit circle.

    Parameters
    ----------
    z
        2D elevation array, shape (H, W). May contain NaN.
    cellsize_x, cellsize_y
        Cell size in the projection's linear units. Must be positive.

    Returns
    -------
    (sin_aspect, cos_aspect)
        Two arrays of shape (H, W) and dtype float64. Both border rows/cols
        are NaN (the 3x3 window cannot be centered). At flat interior cells
        (zero gradient), both channels are 0 — meaning "no preferred direction."
        Otherwise, (sin, cos) is a unit vector pointing in the compass-azimuth
        downslope direction: cos = north component, sin = east component.
    """
    if z.ndim != 2:
        raise ValueError(f"z must be 2D, got shape {z.shape}")
    if cellsize_x <= 0 or cellsize_y <= 0:
        raise ValueError(f"cellsize must be positive, got ({cellsize_x}, {cellsize_y})")

    z = np.asarray(z, dtype=np.float64)
    sin_out = np.full_like(z, np.nan, dtype=np.float64)
    cos_out = np.full_like(z, np.nan, dtype=np.float64)

    a = z[:-2, :-2]
    b = z[:-2, 1:-1]
    c = z[:-2, 2:]
    d = z[1:-1, :-2]
    f = z[1:-1, 2:]
    g = z[2:, :-2]
    h = z[2:, 1:-1]
    i = z[2:, 2:]

    # Horn gradient in array coordinates: dzdx along columns (east),
    # dzdy along rows (south, because row index increases southward).
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * cellsize_x)
    dzdy_array = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * cellsize_y)

    # Geographic gradient: flip the y-axis so +y means north.
    # The downslope direction is the negative of the gradient.
    east_component_downslope = -dzdx
    north_component_downslope = dzdy_array  # double-flip cancels: -(-dzdy_north)

    magnitude = np.hypot(east_component_downslope, north_component_downslope)

    # Where magnitude > 0, unit vector = downslope / magnitude.
    # Where magnitude == 0 (flat), leave as 0 — meaning "no direction."
    # NaN inputs propagate naturally through arithmetic.
    sin_interior = np.zeros_like(magnitude)
    cos_interior = np.zeros_like(magnitude)
    nonflat = magnitude > 0
    sin_interior[nonflat] = east_component_downslope[nonflat] / magnitude[nonflat]
    cos_interior[nonflat] = north_component_downslope[nonflat] / magnitude[nonflat]

    sin_out[1:-1, 1:-1] = sin_interior
    cos_out[1:-1, 1:-1] = cos_interior
    return sin_out, cos_out
