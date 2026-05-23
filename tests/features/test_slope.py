"""Slope tests against known-answer synthetic DEMs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hecras_mesh_ai.features import slope


def test_flat_dem_has_zero_slope():
    z = np.full((10, 10), 100.0)
    s = slope(z, cellsize_x=1.0, cellsize_y=1.0, units="degrees")
    interior = s[1:-1, 1:-1]
    np.testing.assert_allclose(interior, 0.0, atol=1e-12)


def test_tilted_plane_recovers_tilt_angle():
    # A plane that rises 1 unit per cell in the +x direction at cellsize 1
    # has tan(slope) = 1 => slope = 45 degrees.
    h, w = 6, 8
    z = np.tile(np.arange(w, dtype=np.float64), (h, 1))
    s = slope(z, cellsize_x=1.0, cellsize_y=1.0, units="degrees")
    interior = s[1:-1, 1:-1]
    np.testing.assert_allclose(interior, 45.0, atol=1e-12)


def test_tilted_plane_respects_cellsize():
    # Same elevations as the 45-degree case, but stretch cellsize in x to 2:
    # rise per unit distance halves => tan(slope) = 0.5 => slope = atan(0.5).
    h, w = 6, 8
    z = np.tile(np.arange(w, dtype=np.float64), (h, 1))
    s = slope(z, cellsize_x=2.0, cellsize_y=1.0, units="degrees")
    expected_deg = math.degrees(math.atan(0.5))
    np.testing.assert_allclose(s[1:-1, 1:-1], expected_deg, atol=1e-12)


def test_units_round_trip():
    # Verify the three unit modes are internally consistent on a plane.
    z = np.tile(np.arange(8, dtype=np.float64), (6, 1))
    s_rad = slope(z, 1.0, 1.0, units="radians")
    s_deg = slope(z, 1.0, 1.0, units="degrees")
    s_pct = slope(z, 1.0, 1.0, units="percent")
    interior = (slice(1, -1), slice(1, -1))
    np.testing.assert_allclose(np.degrees(s_rad[interior]), s_deg[interior], atol=1e-12)
    np.testing.assert_allclose(np.tan(s_rad[interior]) * 100.0, s_pct[interior], atol=1e-12)


def test_borders_are_nan():
    z = np.zeros((5, 5))
    s = slope(z, 1.0, 1.0)
    assert np.all(np.isnan(s[0, :]))
    assert np.all(np.isnan(s[-1, :]))
    assert np.all(np.isnan(s[:, 0]))
    assert np.all(np.isnan(s[:, -1]))


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        slope(np.zeros(5), 1.0, 1.0)  # 1D
    with pytest.raises(ValueError):
        slope(np.zeros((5, 5)), -1.0, 1.0)  # negative cellsize
    with pytest.raises(ValueError):
        slope(np.zeros((5, 5)), 1.0, 1.0, units="grad")  # unknown units
