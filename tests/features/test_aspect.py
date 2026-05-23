"""Aspect tests against known-direction synthetic DEMs.

Convention reminder: compass azimuth clockwise from north, encoded as
(sin, cos) of that angle.
  - North      (0 deg): (sin, cos) = ( 0,  1)
  - East      (90 deg): (sin, cos) = ( 1,  0)
  - South    (180 deg): (sin, cos) = ( 0, -1)
  - West     (270 deg): (sin, cos) = (-1,  0)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hecras_mesh_ai.features import aspect_sincos


def test_flat_dem_has_zero_vector_aspect():
    """At a flat surface, both sin and cos should be 0 — no direction."""
    z = np.full((10, 10), 100.0)
    sin_a, cos_a = aspect_sincos(z, 1.0, 1.0)
    interior = (slice(1, -1), slice(1, -1))
    np.testing.assert_allclose(sin_a[interior], 0.0, atol=1e-12)
    np.testing.assert_allclose(cos_a[interior], 0.0, atol=1e-12)


def test_plane_rising_eastward_faces_west():
    """If z increases as column index goes up, the slope faces west (270 deg).
    sin(270) = -1, cos(270) = 0."""
    h, w = 6, 8
    z = np.tile(np.arange(w, dtype=np.float64), (h, 1))
    sin_a, cos_a = aspect_sincos(z, 1.0, 1.0)
    interior = (slice(1, -1), slice(1, -1))
    np.testing.assert_allclose(sin_a[interior], -1.0, atol=1e-12)
    np.testing.assert_allclose(cos_a[interior], 0.0, atol=1e-12)


def test_plane_rising_southward_faces_north():
    """In array coords, +row = +south. If z increases with row index, the
    surface rises southward and faces north (0 deg). sin(0)=0, cos(0)=1."""
    h, w = 6, 8
    z = np.tile(np.arange(h, dtype=np.float64).reshape(-1, 1), (1, w))
    sin_a, cos_a = aspect_sincos(z, 1.0, 1.0)
    interior = (slice(1, -1), slice(1, -1))
    np.testing.assert_allclose(sin_a[interior], 0.0, atol=1e-12)
    np.testing.assert_allclose(cos_a[interior], 1.0, atol=1e-12)


def test_plane_rising_northward_faces_south():
    """If z DEcreases with row index (i.e. increases northward), aspect = 180.
    sin(180)=0, cos(180)=-1."""
    h, w = 6, 8
    z = np.tile(np.arange(h, 0, -1, dtype=np.float64).reshape(-1, 1), (1, w))
    sin_a, cos_a = aspect_sincos(z, 1.0, 1.0)
    interior = (slice(1, -1), slice(1, -1))
    np.testing.assert_allclose(sin_a[interior], 0.0, atol=1e-12)
    np.testing.assert_allclose(cos_a[interior], -1.0, atol=1e-12)


def test_diagonal_plane_recovers_45_degree_aspect():
    """Plane rising in +x and +y_array (east + south) should face NW (315).
    sin(315) = -sqrt(2)/2, cos(315) = +sqrt(2)/2."""
    h, w = 6, 8
    rows = np.arange(h, dtype=np.float64).reshape(-1, 1)
    cols = np.arange(w, dtype=np.float64).reshape(1, -1)
    z = rows + cols
    sin_a, cos_a = aspect_sincos(z, 1.0, 1.0)
    expected = math.sqrt(2.0) / 2.0
    interior = (slice(1, -1), slice(1, -1))
    np.testing.assert_allclose(sin_a[interior], -expected, atol=1e-12)
    np.testing.assert_allclose(cos_a[interior], expected, atol=1e-12)


def test_sin_squared_plus_cos_squared_is_zero_or_one():
    """Magnitude invariant: every interior pixel is either at the origin
    (flat) or on the unit circle (sloped). No 'wrong numbers'."""
    h, w = 12, 14
    rng = np.random.default_rng(0)
    z = rng.standard_normal((h, w)) * 5.0
    sin_a, cos_a = aspect_sincos(z, 2.5, 1.7)
    interior_mag_sq = (sin_a[1:-1, 1:-1] ** 2) + (cos_a[1:-1, 1:-1] ** 2)
    # Each interior value should be ~0 (flat) or ~1 (unit vector).
    is_zero = np.isclose(interior_mag_sq, 0.0, atol=1e-12)
    is_unit = np.isclose(interior_mag_sq, 1.0, atol=1e-12)
    assert np.all(is_zero | is_unit), "aspect (sin, cos) must lie on unit circle or at origin"


def test_borders_are_nan():
    z = np.zeros((5, 5))
    sin_a, cos_a = aspect_sincos(z, 1.0, 1.0)
    for arr in (sin_a, cos_a):
        assert np.all(np.isnan(arr[0, :]))
        assert np.all(np.isnan(arr[-1, :]))
        assert np.all(np.isnan(arr[:, 0]))
        assert np.all(np.isnan(arr[:, -1]))


def test_no_nan_in_interior_for_finite_input():
    """Critical guarantee: with finite input, no interior NaN is produced —
    flat points encode as (0, 0), not NaN. No runtime surprises."""
    z = np.zeros((20, 20))  # entirely flat
    sin_a, cos_a = aspect_sincos(z, 1.0, 1.0)
    interior = (slice(1, -1), slice(1, -1))
    assert not np.any(np.isnan(sin_a[interior]))
    assert not np.any(np.isnan(cos_a[interior]))


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        aspect_sincos(np.zeros(5), 1.0, 1.0)
    with pytest.raises(ValueError):
        aspect_sincos(np.zeros((5, 5)), 0.0, 1.0)
