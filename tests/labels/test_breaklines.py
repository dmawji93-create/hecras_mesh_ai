"""Tests for the breakline rasterizer."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiLineString

from hecras_mesh_ai.labels import rasterize_breaklines

# Standard 100x100 raster, 1-unit cells, top-left at (0, 100), 0..100 extent.
_TRANSFORM = from_origin(0, 100, 1, 1)
_CRS = "EPSG:32633"


def _gdf(*geoms, crs=_CRS):
    return gpd.GeoDataFrame({"geometry": list(geoms)}, crs=crs)


def test_empty_gdf_returns_zero_array_of_correct_shape_and_dtype():
    out = rasterize_breaklines(
        _gdf(),
        out_shape=(100, 100),
        transform=_TRANSFORM,
        target_crs=_CRS,
        buffer_width=4.0,
    )
    assert out.shape == (100, 100)
    assert out.dtype == np.uint8
    assert out.sum() == 0


def test_horizontal_line_creates_band_at_expected_rows():
    # Line at world-y=50; with transform.f=100, transform.e=-1,
    #   row = (50 - 100) / -1 = 50
    # buffer_width=4 -> band rows ~48..52 (5 rows wide with all_touched).
    line = LineString([(10, 50), (90, 50)])
    out = rasterize_breaklines(
        _gdf(line),
        out_shape=(100, 100),
        transform=_TRANSFORM,
        target_crs=_CRS,
        buffer_width=4.0,
    )
    rows_with_ones = np.where(out.any(axis=1))[0]
    assert rows_with_ones.size > 0
    assert rows_with_ones.min() >= 47
    assert rows_with_ones.max() <= 53
    # Nothing far outside the band.
    assert out[:45, :].sum() == 0
    assert out[55:, :].sum() == 0


def test_wider_buffer_produces_more_label_pixels():
    line = LineString([(10, 50), (90, 50)])
    narrow = rasterize_breaklines(
        _gdf(line),
        out_shape=(100, 100),
        transform=_TRANSFORM,
        target_crs=_CRS,
        buffer_width=2.0,
    )
    wide = rasterize_breaklines(
        _gdf(line),
        out_shape=(100, 100),
        transform=_TRANSFORM,
        target_crs=_CRS,
        buffer_width=10.0,
    )
    # Wide buffer should label substantially more pixels.
    assert wide.sum() > narrow.sum() * 2


def test_line_outside_raster_extent_yields_zero_output():
    # Far outside the (0..100, 0..100) world extent.
    line = LineString([(1000, 1000), (2000, 2000)])
    out = rasterize_breaklines(
        _gdf(line),
        out_shape=(100, 100),
        transform=_TRANSFORM,
        target_crs=_CRS,
        buffer_width=4.0,
    )
    assert out.sum() == 0


def test_crs_none_assumes_already_in_target_crs():
    """Common case: rashdf returns breaklines with crs=None for HDFs that
    don't embed CRS (e.g. Muncie g04). The function must not crash and
    must proceed without reprojection."""
    line = LineString([(10, 50), (90, 50)])
    gdf = gpd.GeoDataFrame({"geometry": [line]})  # crs is None
    assert gdf.crs is None
    out = rasterize_breaklines(
        gdf,
        out_shape=(100, 100),
        transform=_TRANSFORM,
        target_crs=_CRS,
        buffer_width=4.0,
    )
    assert out.sum() > 0


def test_reprojects_when_breaklines_in_different_crs():
    """Line in lat/lon (EPSG:4326), raster in projected meters (EPSG:32633).
    The reprojection must happen so the line falls inside the raster."""
    # WGS84 line near Dresden, Germany — UTM zone 33N covers it.
    line = LineString([(13.74, 51.00), (13.76, 51.00)])
    gdf = gpd.GeoDataFrame({"geometry": [line]}, crs="EPSG:4326")
    # Raster in UTM zone 33N, 100-m cells, covering the Dresden area.
    transform = from_origin(400000, 5660000, 100, 100)
    out = rasterize_breaklines(
        gdf,
        out_shape=(200, 200),
        transform=transform,
        target_crs="EPSG:32633",
        buffer_width=500.0,
    )
    assert out.sum() > 0


def test_multilinestring_is_rasterized():
    """rashdf occasionally returns MultiLineString; it must be handled."""
    mls = MultiLineString([[(10, 50), (40, 50)], [(60, 50), (90, 50)]])
    out = rasterize_breaklines(
        _gdf(mls),
        out_shape=(100, 100),
        transform=_TRANSFORM,
        target_crs=_CRS,
        buffer_width=4.0,
    )
    # Both segments produce ones.
    assert out[:, :50].sum() > 0
    assert out[:, 50:].sum() > 0
    # Gap between (40, 60) at row=50 should also be empty (lines don't overlap)
    # but we only check that the two halves are non-empty.


def test_two_crossing_lines_union_is_rasterized():
    h_line = LineString([(10, 50), (90, 50)])
    v_line = LineString([(50, 10), (50, 90)])
    out = rasterize_breaklines(
        _gdf(h_line, v_line),
        out_shape=(100, 100),
        transform=_TRANSFORM,
        target_crs=_CRS,
        buffer_width=4.0,
    )
    # Both rows (horizontal line) and both columns (vertical line) should
    # have label pixels.
    assert out[48:52, :].sum() > 0  # horizontal band
    assert out[:, 48:52].sum() > 0  # vertical band


def test_output_is_strictly_binary():
    line = LineString([(10, 50), (90, 50)])
    out = rasterize_breaklines(
        _gdf(line),
        out_shape=(100, 100),
        transform=_TRANSFORM,
        target_crs=_CRS,
        buffer_width=4.0,
    )
    unique_values = set(np.unique(out).tolist())
    assert unique_values <= {0, 1}, f"expected only {{0, 1}}, got {unique_values}"


def test_invalid_buffer_width_raises():
    with pytest.raises(ValueError):
        rasterize_breaklines(
            _gdf(),
            out_shape=(100, 100),
            transform=_TRANSFORM,
            target_crs=_CRS,
            buffer_width=0,
        )
    with pytest.raises(ValueError):
        rasterize_breaklines(
            _gdf(),
            out_shape=(100, 100),
            transform=_TRANSFORM,
            target_crs=_CRS,
            buffer_width=-1.0,
        )


def test_invalid_out_shape_raises():
    with pytest.raises(ValueError):
        rasterize_breaklines(
            _gdf(),
            out_shape=(100,),
            transform=_TRANSFORM,
            target_crs=_CRS,
            buffer_width=4.0,
        )


# ---------------------------------------------------------------------------
# Integration tests against real pilot HDFs + DEMs.
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
_MUNCIE_HDF = _PILOT_BASE / "Muncie" / "Muncie.g04.hdf"
_BALDEAGLE_HDF = _PILOT_BASE / "BaldEagleCrkMulti2D" / "BaldEagleDamBrk.g09.hdf"
_BALDEAGLE_DEM = _PILOT_BASE / "BaldEagleCrkMulti2D" / "Terrain" / "Terrain50.baldeagledem.tif"


def _find_muncie_dem() -> Path | None:
    base = _PILOT_BASE / "Muncie" / "Terrain"
    if not base.exists():
        return None
    tifs = list(base.glob("*.tif"))
    return tifs[0] if tifs else None


@pytest.mark.skipif(not _MUNCIE_HDF.exists(), reason="Muncie HDF not present")
def test_muncie_breaklines_rasterize_onto_feature_stack():
    from rashdf import RasGeomHdf

    from hecras_mesh_ai.features import stack_dem_features

    dem = _find_muncie_dem()
    if dem is None:
        pytest.skip("Muncie DEM not present")
    da = stack_dem_features(dem)

    g = RasGeomHdf(_MUNCIE_HDF)
    try:
        bls = g.breaklines()
    finally:
        g.close()

    labels = rasterize_breaklines(
        bls,
        out_shape=da.shape[1:],
        transform=da.rio.transform(),
        target_crs=da.rio.crs,
        buffer_width=20.0,  # 20 ft on Muncie terrain
    )
    assert labels.shape == tuple(da.shape[1:])
    assert labels.dtype == np.uint8
    assert labels.sum() > 0, "Muncie's 2 breaklines should produce positive label pixels"


@pytest.mark.skipif(
    not _BALDEAGLE_HDF.exists() or not _BALDEAGLE_DEM.exists(),
    reason="Bald Eagle HDF or DEM not present",
)
def test_baldeagle_g09_breaklines_rasterize_onto_feature_stack():
    from rashdf import RasGeomHdf

    from hecras_mesh_ai.features import stack_dem_features

    da = stack_dem_features(_BALDEAGLE_DEM)

    g = RasGeomHdf(_BALDEAGLE_HDF)
    try:
        bls = g.breaklines()
    finally:
        g.close()
    assert len(bls) == 4  # SayersDam, Lower, Middle, Upper

    labels = rasterize_breaklines(
        bls,
        out_shape=da.shape[1:],
        transform=da.rio.transform(),
        target_crs=da.rio.crs,
        buffer_width=50.0,  # 50 ft on Bald Eagle terrain (~36.5 ft cells)
    )
    assert labels.shape == tuple(da.shape[1:])
    assert labels.sum() > 0
