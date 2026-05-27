"""Tests for the Thacker parabolic-bowl analytical solution.

These verify mathematical properties of the exact solution — not HEC-RAS
output. The properties tested are necessary conditions for correctness:
if any fail, the analytical solution has a bug.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

from hecras_mesh_ai.benchmark.thacker import (
    ThackerBowl,
    generate_initial_wse_raster,
    generate_thacker_terrain,
)

# Standard medium-scale bowl: a=5000 ft, D0=10 ft, A=0.5
BOWL = ThackerBowl(D0=10.0, a=5000.0, A=0.5)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_rejects_invalid_D0():
    with pytest.raises(ValueError, match="D0"):
        ThackerBowl(D0=-1, a=5000, A=0.5)


def test_rejects_invalid_A():
    with pytest.raises(ValueError, match="A must be in"):
        ThackerBowl(D0=10, a=5000, A=1.5)


# ---------------------------------------------------------------------------
# Bed profile
# ---------------------------------------------------------------------------


def test_bed_center_is_negative_D0():
    z = BOWL.bed_elevation(np.array([0.0]), np.array([0.0]))
    assert z[0] == pytest.approx(-BOWL.D0)


def test_bed_at_rim_is_zero():
    z = BOWL.bed_elevation(np.array([BOWL.a]), np.array([0.0]))
    assert z[0] == pytest.approx(0.0, abs=1e-10)


def test_bed_beyond_rim_is_positive():
    z = BOWL.bed_elevation(np.array([BOWL.a * 1.2]), np.array([0.0]))
    assert z[0] > 0


def test_bed_is_radially_symmetric():
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    r = BOWL.a * 0.6
    xs = r * np.cos(angles)
    ys = r * np.sin(angles)
    zs = BOWL.bed_elevation(xs, ys)
    np.testing.assert_allclose(zs, zs[0], atol=1e-10)


# ---------------------------------------------------------------------------
# Still-water case (A -> very small, t=0)
# ---------------------------------------------------------------------------


def test_still_water_depth_at_center():
    """At t=0, depth at the center should be approximately D0."""
    h = BOWL.depth(np.array([0.0]), np.array([0.0]), 0.0)
    # With A>0 the initial surface is tilted, so center depth won't be
    # exactly D0 — but should be in the right ballpark.
    assert h[0] > 0
    assert h[0] < BOWL.D0 * 3


def test_zero_amplitude_gives_still_water():
    """With A near zero, depth profile should match D0*(1 - r^2/a^2)."""
    bowl = ThackerBowl(D0=10.0, a=5000.0, A=1e-8)
    rs = np.linspace(0, bowl.a * 0.99, 50)
    xs = rs
    ys = np.zeros_like(rs)
    h = bowl.depth(xs, ys, 0.0)
    expected = bowl.D0 * (1.0 - rs**2 / bowl.a**2)
    np.testing.assert_allclose(h, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Periodicity
# ---------------------------------------------------------------------------


def test_period_is_positive():
    assert BOWL.period > 0


def test_solution_is_periodic():
    """Depth field at t and t+T should be identical."""
    xs = np.linspace(-BOWL.a, BOWL.a, 100)
    ys = np.zeros(100)
    t0 = BOWL.period * 0.37
    h0 = BOWL.depth(xs, ys, t0)
    h1 = BOWL.depth(xs, ys, t0 + BOWL.period)
    np.testing.assert_allclose(h1, h0, atol=1e-8)


def test_half_period_is_mirror_image():
    """At t + T/2 the surface tilt should be reversed in x."""
    xs = np.linspace(-BOWL.a * 0.8, BOWL.a * 0.8, 100)
    ys = np.zeros(100)
    h0 = BOWL.depth(xs, ys, 0.0)
    h_half = BOWL.depth(-xs, ys, BOWL.period / 2.0)
    np.testing.assert_allclose(h_half, h0, atol=1e-6)


# ---------------------------------------------------------------------------
# Volume conservation
# ---------------------------------------------------------------------------


def test_volume_conserved_across_times():
    """Total water volume must be constant at all times."""
    times = [0, BOWL.period * 0.25, BOWL.period * 0.5, BOWL.period * 0.75]
    volumes = [BOWL.volume(t, n_points=800) for t in times]
    np.testing.assert_allclose(volumes, volumes[0], rtol=5e-3)


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------


def test_velocity_is_zero_at_t0_and_half_period():
    """At t=0 and t=T/2, velocity is zero (surface at max displacement)."""
    assert BOWL.velocity_x(0.0) == pytest.approx(0.0, abs=1e-10)
    assert BOWL.velocity_x(BOWL.period / 2.0) == pytest.approx(0.0, abs=1e-10)


def test_velocity_y_is_always_zero():
    assert BOWL.velocity_y() == 0.0


def test_velocity_is_periodic():
    t0 = BOWL.period * 0.3
    assert BOWL.velocity_x(t0) == pytest.approx(BOWL.velocity_x(t0 + BOWL.period), abs=1e-10)


# ---------------------------------------------------------------------------
# Wetting/drying — shoreline moves
# ---------------------------------------------------------------------------


def test_shoreline_moves_between_times():
    """The shoreline position on the x-axis should differ at t=0 vs t=T/4."""
    sl_0 = BOWL.shoreline_x_extent(0.0)
    sl_quarter = BOWL.shoreline_x_extent(BOWL.period / 4.0)
    assert sl_0[0] != pytest.approx(sl_quarter[0], abs=10)


def test_dry_region_exists_outside_shoreline():
    """At any time, the far uphill side of the bowl must be dry."""
    # With A=0.5, water sloshes far past the rim on the downhill side.
    # Test the UPHILL side (negative x when water is tilted to positive x
    # at t=0) well beyond the rim.
    t = 0.0
    far_x = np.array([-BOWL.a * 1.2])
    far_y = np.array([0.0])
    assert BOWL.depth(far_x, far_y, t)[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Terrain GeoTIFF generation
# ---------------------------------------------------------------------------


def test_terrain_geotiff_creates_valid_raster(tmp_path):
    p = tmp_path / "terrain.tif"
    generate_thacker_terrain(BOWL, p, resolution=100.0)
    assert p.exists()
    with rasterio.open(p) as src:
        assert src.count == 1
        data = src.read(1)
        assert data.shape[0] > 100
        assert data.shape[1] > 100
        # Center pixel should be approximately -D0.
        center_r = data.shape[0] // 2
        center_c = data.shape[1] // 2
        assert data[center_r, center_c] == pytest.approx(-BOWL.D0, abs=1.0)


def test_initial_wse_geotiff_creates_valid_raster(tmp_path):
    p = tmp_path / "initial_wse.tif"
    generate_initial_wse_raster(BOWL, p, resolution=100.0)
    assert p.exists()
    with rasterio.open(p) as src:
        data = src.read(1)
        # Center should be wet — WSE above bed.
        center_r = data.shape[0] // 2
        center_c = data.shape[1] // 2
        assert data[center_r, center_c] > -BOWL.D0


# ---------------------------------------------------------------------------
# Print summary for manual inspection
# ---------------------------------------------------------------------------


def test_print_bowl_parameters(capsys):
    """Not a real test — just prints the bowl config for review."""
    print("\nThacker bowl parameters:")
    print(f"  D0 = {BOWL.D0} ft (max still-water depth)")
    print(f"  a  = {BOWL.a} ft (basin radius)")
    print(f"  A  = {BOWL.A} (oscillation amplitude)")
    print(f"  omega = {BOWL.omega:.6f} rad/s")
    print(f"  period = {BOWL.period:.1f} s = {BOWL.period/60:.1f} min")
    print(f"  velocity at T/4 = {BOWL.velocity_x(BOWL.period/4):.2f} ft/s")
    vol = BOWL.volume(0.0, n_points=800)
    print(f"  initial volume ~ {vol:.0f} ft^3 = {vol/43560:.0f} acre-ft")
