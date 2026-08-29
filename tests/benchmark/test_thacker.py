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
# Independent known values — hard-coded, derived by hand, NOT from the code.
# A mutated implementation (wrong omega, sign-flipped u, wrong c0) passes
# every self-consistency test above; these are the tests that catch it.
# ---------------------------------------------------------------------------
# For D0=10 ft, a=5000 ft, A=0.5, g=32.174 ft/s^2:
#   omega  = sqrt(2*32.174*10)/5000 = 5.07338e-3 rad/s
#   T      = 2*pi/omega             = 1238.46 s
#   u(T/4) = -A*a*omega             = -12.683 ft/s  (negative: toward -x)
#   c0     = -A^2*D0                = -2.5 ft
#   shoreline(0) = (-(1-A)*a, (1+A)*a) = (-2500, 7500) ft
#   volume = pi*D0*a^2/2            = 3.92699e8 ft^3


def test_period_matches_hand_computed_value():
    assert BOWL.period == pytest.approx(1238.46, abs=0.05)


def test_velocity_at_quarter_period_magnitude_and_sign():
    assert BOWL.velocity_x(BOWL.period / 4.0) == pytest.approx(-12.683, abs=0.005)


def test_c0_closed_form():
    assert BOWL._c0 == pytest.approx(-2.5, abs=1e-12)


def test_volume_matches_analytic_constant():
    """Against the analytic constant — not volumes[0] (which any c0 passes)."""
    v_analytic = np.pi * BOWL.D0 * BOWL.a**2 / 2.0
    assert BOWL.volume(0.0) == pytest.approx(v_analytic, rel=5e-3)
    assert BOWL.volume(BOWL.period * 0.31) == pytest.approx(v_analytic, rel=5e-3)


def test_shoreline_known_positions():
    lo, hi = BOWL.shoreline_x_extent(0.0)
    assert lo == pytest.approx(-2500.0, abs=1e-6)
    assert hi == pytest.approx(7500.0, abs=1e-6)
    lo, hi = BOWL.shoreline_x_extent(BOWL.period / 2.0)
    assert lo == pytest.approx(-7500.0, abs=1e-4)
    assert hi == pytest.approx(2500.0, abs=1e-4)


def test_shoreline_matches_depth_field():
    """Closed-form shoreline agrees with the depth field it summarizes."""
    t = BOWL.period * 0.17
    lo, hi = BOWL.shoreline_x_extent(t)
    eps = 1.0  # ft
    assert BOWL.depth(np.array([lo + eps]), np.array([0.0]), t)[0] > 0
    assert BOWL.depth(np.array([lo - eps]), np.array([0.0]), t)[0] == 0.0
    assert BOWL.depth(np.array([hi - eps]), np.array([0.0]), t)[0] > 0
    assert BOWL.depth(np.array([hi + eps]), np.array([0.0]), t)[0] == 0.0


def test_fields_satisfy_shallow_water_equations():
    """Finite-difference residuals of mass and x-momentum vanish at interior
    wet points: h_t + (h*u)_x = 0 and u_t + g*eta_x = 0 (v == 0, u_x == 0).

    This jointly pins omega, u (sign and magnitude), and c(t) — the single
    strongest necessary condition available for the exact solution.
    """
    xs = np.linspace(-1500.0, 3000.0, 7)  # well inside the wet disk at t0
    ys = np.zeros_like(xs)
    t0 = BOWL.period * 0.23
    dt = 0.05
    dx = 1.0

    u_mid = BOWL.velocity_x(t0)
    h_t = (BOWL.depth(xs, ys, t0 + dt) - BOWL.depth(xs, ys, t0 - dt)) / (2 * dt)
    hu_x = (BOWL.depth(xs + dx, ys, t0) - BOWL.depth(xs - dx, ys, t0)) * u_mid / (2 * dx)
    mass_residual = h_t + hu_x

    u_t = (BOWL.velocity_x(t0 + dt) - BOWL.velocity_x(t0 - dt)) / (2 * dt)
    eta_x = (BOWL.free_surface(xs + dx, ys, t0) - BOWL.free_surface(xs - dx, ys, t0)) / (2 * dx)
    mom_residual = u_t + BOWL.g * eta_x

    assert np.max(np.abs(mass_residual)) < 1e-6
    assert np.max(np.abs(mom_residual)) < 1e-6


def test_high_amplitude_bowl_is_consistent():
    """A=0.8 exercises the regime the old fixed 1.5a windows corrupted
    (c0 came out wrong by 0.215 ft at A=0.8 via the clipped quadrature)."""
    bowl = ThackerBowl(D0=10.0, a=5000.0, A=0.8)
    assert bowl._c0 == pytest.approx(-6.4, abs=1e-12)
    v_analytic = np.pi * bowl.D0 * bowl.a**2 / 2.0
    assert bowl.volume(0.0) == pytest.approx(v_analytic, rel=5e-3)
    lo, hi = bowl.shoreline_x_extent(0.0)
    assert lo == pytest.approx(-1000.0, abs=1e-6)
    assert hi == pytest.approx(9000.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Raster containment and registration
# ---------------------------------------------------------------------------


def test_wet_region_contained_within_recommended_extent():
    """The oscillating wet region must fit inside the raster at every time."""
    ext = BOWL.recommended_raster_extent
    edge = np.linspace(-ext, ext, 200)
    for t in [0.0, BOWL.period * 0.25, BOWL.period * 0.5]:
        for xs, ys in [
            (edge, np.full_like(edge, ext)),
            (edge, np.full_like(edge, -ext)),
            (np.full_like(edge, ext), edge),
            (np.full_like(edge, -ext), edge),
        ]:
            assert np.all(BOWL.depth(xs, ys, t) == 0.0)


def test_initial_wse_raster_edges_are_dry(tmp_path):
    """Boundary pixels of the IC raster must be dry (WSE == bed there).

    The old +/-1.3a extent clipped ~1.5% of the water volume with 3.6 ft
    of depth standing at the raster edge at t=0.
    """
    tp = tmp_path / "terrain.tif"
    wp = tmp_path / "wse.tif"
    generate_thacker_terrain(BOWL, tp, resolution=50.0)
    generate_initial_wse_raster(BOWL, wp, resolution=50.0)
    with rasterio.open(tp) as ts:
        bed = ts.read(1)
    with rasterio.open(wp) as ws:
        wse = ws.read(1)
    edges = np.concatenate([bed[0], bed[-1], bed[:, 0], bed[:, -1]])
    wedges = np.concatenate([wse[0], wse[-1], wse[:, 0], wse[:, -1]])
    np.testing.assert_allclose(wedges, edges, atol=1e-4)


def test_raster_registration_matches_geotransform(tmp_path):
    """Sampled values sit exactly at geotransform pixel centers — including
    for a resolution (30 ft) that does not divide the nominal extent. The
    old code drifted ~0.67 cell (~0.10 ft bed error) at the far edge."""
    p = tmp_path / "terrain30.tif"
    generate_thacker_terrain(BOWL, p, resolution=30.0)
    with rasterio.open(p) as src:
        assert src.transform.a == pytest.approx(30.0, abs=1e-9)
        assert -src.transform.e == pytest.approx(30.0, abs=1e-9)
        data = src.read(1)
        corners = [
            (0, 0),
            (0, data.shape[1] - 1),
            (data.shape[0] - 1, data.shape[1] - 1),
            (data.shape[0] // 2, data.shape[1] - 1),
        ]
        for row, col in corners:
            x, y = src.transform * (col + 0.5, row + 0.5)
            z_expected = BOWL.bed_elevation(np.array([x - 500_000.0]), np.array([y - 1_800_000.0]))[
                0
            ]
            assert data[row, col] == pytest.approx(z_expected, abs=5e-3)


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
