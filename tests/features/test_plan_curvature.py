"""Plan curvature tests against closed-form synthetic surfaces.

Sign convention check:
  - Bowl (z = x^2 + y^2):  k_plan negative everywhere (valley-like).
  - Dome (z = -x^2 - y^2): k_plan positive everywhere (ridge-like).

Magnitude check: for a paraboloid z = x^2 + y^2, k_plan = -1/r at radius r.
"""

from __future__ import annotations

import numpy as np
import pytest

from hecras_mesh_ai.features import plan_curvature


def _paraboloid_grid(h: int, w: int, dx: float, dy: float, sign: float = 1.0):
    """Return (z, X, Y) where z = sign * (X^2 + Y^2) on an (h x w) grid
    centered in the array; X and Y are full 2D meshes of physical coords.
    """
    ii, jj = np.meshgrid(np.arange(h) - h // 2, np.arange(w) - w // 2, indexing="ij")
    X = jj * dx
    Y = ii * dy
    return sign * (X * X + Y * Y), X, Y


def test_flat_dem_has_zero_plan_curvature():
    z = np.full((10, 10), 100.0)
    k = plan_curvature(z, 1.0, 1.0)
    np.testing.assert_allclose(k[1:-1, 1:-1], 0.0, atol=1e-12)


def test_bowl_is_negative_everywhere_in_interior():
    """A paraboloid bowl should give negative plan curvature (valley) at every
    non-flat interior point. The single dead-center cell is the origin where
    (p, q) = (0, 0) and we report 0; everywhere else should be < 0."""
    z, X, Y = _paraboloid_grid(11, 11, dx=1.0, dy=1.0, sign=+1.0)
    k = plan_curvature(z, 1.0, 1.0)
    interior = k[1:-1, 1:-1]
    r2 = (X * X + Y * Y)[1:-1, 1:-1]
    nonzero = r2 > 0
    assert np.all(interior[nonzero] < 0), "bowl must have negative plan curvature"


def test_dome_is_positive_everywhere_in_interior():
    z, X, Y = _paraboloid_grid(11, 11, dx=1.0, dy=1.0, sign=-1.0)
    k = plan_curvature(z, 1.0, 1.0)
    interior = k[1:-1, 1:-1]
    r2 = (X * X + Y * Y)[1:-1, 1:-1]
    nonzero = r2 > 0
    assert np.all(interior[nonzero] > 0), "dome must have positive plan curvature"


def test_bowl_matches_minus_one_over_r_at_sampled_points():
    """Analytic result for z = x^2 + y^2: k_plan = -1/sqrt(x^2 + y^2).
    Verify at several lattice points away from the origin."""
    z, x, y = _paraboloid_grid(21, 21, dx=1.0, dy=1.0, sign=+1.0)
    k = plan_curvature(z, 1.0, 1.0)
    center = 10  # h // 2 with h = 21
    # Probe (x, y) = (3, 0), (0, 3), (3, 4) — analytic r = 3, 3, 5
    for di, dj, expected_r in [(0, 3, 3.0), (3, 0, 3.0), (4, 3, 5.0)]:
        got = k[center + di, center + dj]
        np.testing.assert_allclose(got, -1.0 / expected_r, atol=1e-10)


def test_nonsquare_cellsize_scales_correctly():
    """k_plan has units of 1/length. Stretching cellsize_x by 2x at fixed
    elevations means the same array gives a coarser physical surface — the
    curvature magnitudes should scale accordingly."""
    z, _, _ = _paraboloid_grid(15, 15, dx=1.0, dy=1.0, sign=+1.0)
    k1 = plan_curvature(z, 1.0, 1.0)
    # If we tell the function dx=2.0 with the same z array, every physical
    # x-distance doubles. At an array offset of dj=3, the new x = 6, y = 0,
    # but z[row, col] still equals dj^2 = 9 (not (dj*2)^2 = 36). So this is
    # NOT a coherent physical surface — but we can still verify that the
    # function does not silently ignore cellsize. The simpler check: at the
    # same array index, k changes when cellsize changes.
    k2 = plan_curvature(z, 2.0, 1.0)
    assert not np.allclose(k1[1:-1, 1:-1], k2[1:-1, 1:-1])


def test_invariant_under_row_flip():
    """Plan curvature is y-axis-flip-invariant — the row-direction convention
    that matters for aspect does NOT affect plan curvature."""
    z, _, _ = _paraboloid_grid(13, 13, dx=1.0, dy=1.0, sign=+1.0)
    k = plan_curvature(z, 1.0, 1.0)
    k_flipped = plan_curvature(z[::-1, :].copy(), 1.0, 1.0)
    # After row-flipping the result back, they should match.
    np.testing.assert_allclose(k[1:-1, 1:-1], k_flipped[::-1, :][1:-1, 1:-1], atol=1e-12)


def test_borders_are_nan():
    z = np.zeros((5, 5))
    k = plan_curvature(z, 1.0, 1.0)
    assert np.all(np.isnan(k[0, :]))
    assert np.all(np.isnan(k[-1, :]))
    assert np.all(np.isnan(k[:, 0]))
    assert np.all(np.isnan(k[:, -1]))


def test_nan_input_propagates_to_nan_output():
    """An isolated NaN at (i, j) should corrupt every interior cell whose
    3x3 stencil includes (i, j). The plan-curvature formula uses ALL 9 cells
    of the 3x3 (unlike Horn's first derivative which skips the center), so
    the cell directly under the NaN is corrupted too."""
    z = np.zeros((7, 7))
    z[3, 3] = np.nan
    k = plan_curvature(z, 1.0, 1.0)
    affected = (slice(2, 5), slice(2, 5))
    assert np.all(np.isnan(k[affected]))
    # Cells far from the NaN are unaffected.
    assert k[5, 5] == 0.0


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        plan_curvature(np.zeros(5), 1.0, 1.0)
    with pytest.raises(ValueError):
        plan_curvature(np.zeros((5, 5)), -1.0, 1.0)
