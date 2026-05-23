"""Profile curvature tests against closed-form synthetic surfaces.

Sign convention (matches ArcGIS):
  - Bowl  (z = x^2 + y^2):  k_prof negative (decelerating into the bottom).
  - Dome  (z = -x^2 - y^2): k_prof positive (accelerating off the top).

Closed-form magnitude: for z = x^2 + y^2 at radius r from center,
  k_prof = -2 / (1 + 4 r^2)^(3/2)
"""

from __future__ import annotations

import numpy as np
import pytest

from hecras_mesh_ai.features import profile_curvature


def _paraboloid_grid(h: int, w: int, dx: float, dy: float, sign: float = 1.0):
    ii, jj = np.meshgrid(np.arange(h) - h // 2, np.arange(w) - w // 2, indexing="ij")
    X = jj * dx
    Y = ii * dy
    return sign * (X * X + Y * Y), X, Y


def test_flat_dem_has_zero_profile_curvature():
    z = np.full((10, 10), 100.0)
    k = profile_curvature(z, 1.0, 1.0)
    np.testing.assert_allclose(k[1:-1, 1:-1], 0.0, atol=1e-12)


def test_bowl_is_negative_everywhere_in_interior():
    """Bowl decelerates flow as it approaches the bottom; k_prof < 0 off-center."""
    z, X, Y = _paraboloid_grid(11, 11, dx=1.0, dy=1.0, sign=+1.0)
    k = profile_curvature(z, 1.0, 1.0)
    interior = k[1:-1, 1:-1]
    r2 = (X * X + Y * Y)[1:-1, 1:-1]
    nonzero = r2 > 0
    assert np.all(interior[nonzero] < 0), "bowl must have negative profile curvature"


def test_dome_is_positive_everywhere_in_interior():
    """Dome accelerates flow as it descends; k_prof > 0 off-center."""
    z, X, Y = _paraboloid_grid(11, 11, dx=1.0, dy=1.0, sign=-1.0)
    k = profile_curvature(z, 1.0, 1.0)
    interior = k[1:-1, 1:-1]
    r2 = (X * X + Y * Y)[1:-1, 1:-1]
    nonzero = r2 > 0
    assert np.all(interior[nonzero] > 0), "dome must have positive profile curvature"


def test_bowl_matches_closed_form_at_sampled_points():
    """Analytic result for z = x^2 + y^2:
        k_prof = -2 / (1 + 4 r^2)^(3/2)
    Verify at several lattice points away from origin."""
    z, _, _ = _paraboloid_grid(21, 21, dx=1.0, dy=1.0, sign=+1.0)
    k = profile_curvature(z, 1.0, 1.0)
    center = 10
    for di, dj, expected_r in [(0, 3, 3.0), (3, 0, 3.0), (4, 3, 5.0)]:
        got = k[center + di, center + dj]
        expected = -2.0 / (1.0 + 4.0 * expected_r * expected_r) ** 1.5
        np.testing.assert_allclose(got, expected, atol=1e-10)


def test_invariant_under_row_flip():
    """Profile curvature, like plan curvature, is y-axis-flip-invariant."""
    z, _, _ = _paraboloid_grid(13, 13, dx=1.0, dy=1.0, sign=+1.0)
    k = profile_curvature(z, 1.0, 1.0)
    k_flipped = profile_curvature(z[::-1, :].copy(), 1.0, 1.0)
    np.testing.assert_allclose(k[1:-1, 1:-1], k_flipped[::-1, :][1:-1, 1:-1], atol=1e-12)


def test_borders_are_nan():
    z = np.zeros((5, 5))
    k = profile_curvature(z, 1.0, 1.0)
    assert np.all(np.isnan(k[0, :]))
    assert np.all(np.isnan(k[-1, :]))
    assert np.all(np.isnan(k[:, 0]))
    assert np.all(np.isnan(k[:, -1]))


def test_nan_input_propagates_to_nan_output():
    """Like plan curvature, profile curvature uses all 9 stencil cells, so an
    isolated NaN corrupts the full 3x3 surrounding region."""
    z = np.zeros((7, 7))
    z[3, 3] = np.nan
    k = profile_curvature(z, 1.0, 1.0)
    affected = (slice(2, 5), slice(2, 5))
    assert np.all(np.isnan(k[affected]))
    assert k[5, 5] == 0.0


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        profile_curvature(np.zeros(5), 1.0, 1.0)
    with pytest.raises(ValueError):
        profile_curvature(np.zeros((5, 5)), -1.0, 1.0)
