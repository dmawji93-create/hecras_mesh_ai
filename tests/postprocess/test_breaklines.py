"""Tests for the probability-to-polylines post-processing chain."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from hecras_mesh_ai.postprocess import probability_to_polylines

_TRANSFORM = from_origin(0, 100, 1, 1)  # 100x100 raster, 1-unit cells
_CRS = "EPSG:32633"


def _horizontal_line_prob(*, h=100, w=100, row=50, half_thick=2, value=0.9):
    """Probability raster with a horizontal band centered on `row`."""
    prob = np.zeros((h, w), dtype=np.float32)
    prob[row - half_thick : row + half_thick + 1, :] = value
    return prob


def test_empty_probability_yields_empty_geodataframe():
    prob = np.zeros((50, 50), dtype=np.float32)
    gdf = probability_to_polylines(prob, transform=_TRANSFORM, target_crs=_CRS)
    assert len(gdf) == 0
    assert gdf.crs.to_epsg() == 32633


def test_single_horizontal_band_yields_one_horizontal_line():
    prob = _horizontal_line_prob(row=50, half_thick=2, value=0.9)
    gdf = probability_to_polylines(prob, transform=_TRANSFORM, target_crs=_CRS, threshold=0.5)
    assert len(gdf) == 1
    line = gdf.geometry.iloc[0]
    coords = list(line.coords)
    ys = np.array([y for _, y in coords])
    # Skeleton of a 5-pixel-thick band centers near row 50, world y ~ 49.5.
    # End-cap artifacts at the column boundaries may push a few pixels off
    # by 1 row; we check the mean is near 49.5 and most pixels are within 1 row.
    assert abs(ys.mean() - 49.5) < 1.0, f"line not horizontal-centered at y~49.5: mean={ys.mean()}"
    assert (np.abs(ys - 49.5) <= 1.0).all(), f"some pixels too far from band center: ys={ys}"
    # The line should span essentially the full width.
    xs = sorted(x for x, _ in coords)
    assert xs[0] < 5 and xs[-1] > 95


def test_threshold_rejects_below_cutoff():
    prob = _horizontal_line_prob(value=0.3)
    gdf = probability_to_polylines(prob, transform=_TRANSFORM, target_crs=_CRS, threshold=0.5)
    assert len(gdf) == 0


def test_min_length_filters_short_components():
    """A 5-pixel-long band should be dropped at min_length_pixels=10."""
    prob = np.zeros((50, 50), dtype=np.float32)
    prob[20:23, 10:15] = 0.9  # 3-thick x 5-wide bar
    gdf = probability_to_polylines(
        prob,
        transform=_TRANSFORM,
        target_crs=_CRS,
        threshold=0.5,
        min_length_pixels=10,
    )
    assert len(gdf) == 0


def test_two_disjoint_bands_yield_two_lines():
    prob = np.zeros((100, 100), dtype=np.float32)
    prob[20:23, :] = 0.9  # band 1
    prob[70:73, :] = 0.9  # band 2
    gdf = probability_to_polylines(prob, transform=_TRANSFORM, target_crs=_CRS, threshold=0.5)
    assert len(gdf) == 2


def test_simplify_tolerance_reduces_vertex_count():
    """A straight line has 100 vertices after skeletonization; Douglas-Peucker
    at any tolerance should collapse it to ~2 endpoints."""
    prob = _horizontal_line_prob()
    gdf_full = probability_to_polylines(prob, transform=_TRANSFORM, target_crs=_CRS, threshold=0.5)
    gdf_simp = probability_to_polylines(
        prob,
        transform=_TRANSFORM,
        target_crs=_CRS,
        threshold=0.5,
        simplify_tolerance=2.0,
    )
    n_full = len(list(gdf_full.geometry.iloc[0].coords))
    n_simp = len(list(gdf_simp.geometry.iloc[0].coords))
    assert n_simp < n_full, f"simplify did not reduce vertices: full={n_full}, simp={n_simp}"
    assert n_simp <= 5  # straight line should collapse near-fully


def test_geodataframe_has_length_column_in_crs_units():
    prob = _horizontal_line_prob()
    gdf = probability_to_polylines(prob, transform=_TRANSFORM, target_crs=_CRS, threshold=0.5)
    assert "length" in gdf.columns
    assert gdf.iloc[0]["length"] == pytest.approx(gdf.geometry.iloc[0].length)
    # 100-unit-wide raster -> line length should be near 99 (excludes boundary half-pixels).
    assert 95 < gdf.iloc[0]["length"] <= 100


def test_invalid_input_raises():
    with pytest.raises(ValueError, match="2D"):
        probability_to_polylines(np.zeros(50), transform=_TRANSFORM, target_crs=_CRS)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        probability_to_polylines(
            np.zeros((10, 10)), transform=_TRANSFORM, target_crs=_CRS, threshold=1.5
        )
    with pytest.raises(ValueError, match=">= 1"):
        probability_to_polylines(
            np.zeros((10, 10)),
            transform=_TRANSFORM,
            target_crs=_CRS,
            min_length_pixels=0,
        )
    with pytest.raises(ValueError, match=">= 0"):
        probability_to_polylines(
            np.zeros((10, 10)),
            transform=_TRANSFORM,
            target_crs=_CRS,
            simplify_tolerance=-1.0,
        )
