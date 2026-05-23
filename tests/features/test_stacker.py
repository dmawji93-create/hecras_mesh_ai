"""Tests for the CRS-aware DEM feature stacker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin

from hecras_mesh_ai.features import FEATURE_CHANNELS, stack_dem_features


def _write_synthetic_dem(
    path: Path,
    z: np.ndarray,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    crs: str = "EPSG:32633",
    nodata: float = -9999.0,
    row_up: bool = False,
) -> None:
    """Write a small synthetic DEM to a GeoTIFF.

    Standard layout (row_up=False): top-left at (0, h*dy), transform.e = -dy.
    Row-up layout (row_up=True): top-left at (0, 0), transform.e = +dy.
    """
    h, w = z.shape
    if not row_up:
        transform = from_origin(0, h * dy, dx, dy)  # transform.e = -dy
    else:
        # Non-identity origin so rasterio doesn't warn "NotGeoreferenced".
        transform = Affine(dx, 0, 1000.0, 0, dy, 500.0)  # transform.e = +dy (irregular)

    payload = np.where(np.isnan(z), nodata, z).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(payload, 1)


def test_stack_shape_dims_and_channel_names(tmp_path):
    z = np.ones((30, 40), dtype=np.float64)
    p = tmp_path / "flat.tif"
    _write_synthetic_dem(p, z)
    da = stack_dem_features(p)
    assert da.shape == (6, 30, 40)
    assert da.dims == ("band", "y", "x")
    assert list(da.band.values) == list(FEATURE_CHANNELS)


def test_stack_flat_dem_has_zero_derivatives_with_no_border_nan(tmp_path):
    """A flat DEM should yield derivatives that are zero everywhere — and
    crucially with NO NaN border, because the reflect-padding handles it."""
    z = np.full((20, 25), 100.0)
    p = tmp_path / "flat.tif"
    _write_synthetic_dem(p, z)
    da = stack_dem_features(p)

    np.testing.assert_allclose(da.sel(band="elevation").values, 100.0)
    for ch in ("slope", "aspect_sin", "aspect_cos", "plan_curvature", "profile_curvature"):
        arr = da.sel(band=ch).values
        assert not np.any(
            np.isnan(arr)
        ), f"channel {ch} has NaN — reflect-pad failed to clean the border"
        np.testing.assert_allclose(arr, 0.0, atol=1e-10)


def test_stack_preserves_source_crs(tmp_path):
    z = np.zeros((10, 10))
    p = tmp_path / "test.tif"
    _write_synthetic_dem(p, z, crs="EPSG:2271")  # NAD83 / Pennsylvania North ftUS
    da = stack_dem_features(p)
    assert da.rio.crs.to_epsg() == 2271


def test_stack_preserves_pixel_coordinates(tmp_path):
    """Pixel-center coordinates should reflect the source affine transform."""
    z = np.zeros((4, 5))
    p = tmp_path / "test.tif"
    _write_synthetic_dem(p, z, dx=10.0, dy=10.0)
    da = stack_dem_features(p)
    # from_origin(0, 40, 10, 10) -> top-left = (0, 40), pixel (0,0) center = (5, 35)
    np.testing.assert_allclose(da.x.values, [5.0, 15.0, 25.0, 35.0, 45.0])
    np.testing.assert_allclose(da.y.values, [35.0, 25.0, 15.0, 5.0])


def test_stack_patches_isolated_nan_in_elevation(tmp_path):
    z = np.ones((10, 10))
    z[5, 5] = np.nan
    p = tmp_path / "isolated_nan.tif"
    _write_synthetic_dem(p, z)
    da = stack_dem_features(p)
    el = da.sel(band="elevation").values
    assert not np.isnan(el[5, 5]), "isolated NaN must be patched"
    np.testing.assert_allclose(el[5, 5], 1.0)


def test_stack_preserves_larger_nan_region(tmp_path):
    z = np.ones((10, 10))
    z[4:7, 4:7] = np.nan  # 3x3 block, well above max_patch_size=1
    p = tmp_path / "block_nan.tif"
    _write_synthetic_dem(p, z)
    da = stack_dem_features(p)
    el = da.sel(band="elevation").values
    assert np.all(np.isnan(el[4:7, 4:7]))


def test_stack_propagates_nan_to_derivatives_at_block(tmp_path):
    """Larger NaN regions should be NaN in elevation AND in every derived
    channel — at minimum across the block itself."""
    z = np.ones((10, 10))
    z[4:7, 4:7] = np.nan
    p = tmp_path / "block_nan.tif"
    _write_synthetic_dem(p, z)
    da = stack_dem_features(p)
    for ch in FEATURE_CHANNELS:
        arr = da.sel(band=ch).values
        if ch == "elevation":
            assert np.all(np.isnan(arr[4:7, 4:7]))
        else:
            # Derivatives whose stencils overlap the NaN block must be NaN.
            # At minimum the block itself is NaN; the precise extent depends
            # on each derivative's stencil — we just check the block.
            assert np.all(np.isnan(arr[4:7, 4:7])), f"channel {ch} did not propagate NaN at block"


def test_row_up_dem_raises_with_helpful_message(tmp_path):
    z = np.zeros((10, 10))
    p = tmp_path / "row_up.tif"
    _write_synthetic_dem(p, z, row_up=True)
    with pytest.raises(ValueError, match="CONVENTION-TO-VERIFY"):
        stack_dem_features(p)


# ---------------------------------------------------------------------------
# Integration tests — real pilot DEMs. Skipped if data not present.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PILOT_BASE = (
    _REPO_ROOT
    / "data"
    / "raw"
    / "usace"
    / "RAS Samples"
    / "Example_Projects_7_0"
    / "2D Unsteady Flow Hydraulics"
)


def _find_pilot_dem(project_dir: str) -> Path | None:
    base = _PILOT_BASE / project_dir / "Terrain"
    if not base.exists():
        return None
    tifs = list(base.glob("*.tif"))
    return tifs[0] if tifs else None


_MUNCIE_DEM = _find_pilot_dem("Muncie")
_BALDEAGLE_DEM = _PILOT_BASE / "BaldEagleCrkMulti2D" / "Terrain" / "Terrain50.baldeagledem.tif"


@pytest.mark.skipif(_MUNCIE_DEM is None, reason="Muncie pilot DEM not present")
def test_stack_muncie_dem_integration():
    da = stack_dem_features(_MUNCIE_DEM)
    assert da.dims == ("band", "y", "x")
    assert da.shape[0] == 6
    assert list(da.band.values) == list(FEATURE_CHANNELS)
    # Sanity: no channel is entirely NaN.
    for ch in FEATURE_CHANNELS:
        arr = da.sel(band=ch).values
        assert not np.all(np.isnan(arr)), f"channel {ch} is all-NaN on Muncie"
    # Muncie has no embedded HDF CRS but the terrain TIFF does — EPSG:2965.
    assert da.rio.crs.to_epsg() == 2965


@pytest.mark.skipif(not _BALDEAGLE_DEM.exists(), reason="Bald Eagle pilot DEM not present")
def test_stack_baldeagle_dem_integration():
    da = stack_dem_features(_BALDEAGLE_DEM)
    assert da.shape[0] == 6
    assert list(da.band.values) == list(FEATURE_CHANNELS)
    for ch in FEATURE_CHANNELS:
        arr = da.sel(band=ch).values
        assert not np.all(np.isnan(arr)), f"channel {ch} is all-NaN on Bald Eagle"
    # Bald Eagle terrain TIFF CRS — EPSG:2271 (NAD83 / Pennsylvania North ftUS).
    assert da.rio.crs.to_epsg() == 2271
