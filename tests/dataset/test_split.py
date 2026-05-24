"""Tests for the spatial-holdout leakage check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from hecras_mesh_ai.dataset import RasterTileDataset, assert_no_spatial_overlap


def _write_pair(
    tmp_path: Path,
    name: str,
    *,
    origin_x: float = 0.0,
    origin_y_top: float = 100.0,
    h: int = 50,
    w: int = 50,
    cellsize: float = 1.0,
    crs: str = "EPSG:32633",
) -> tuple[Path, Path]:
    out = tmp_path / name
    out.mkdir()
    transform = from_origin(origin_x, origin_y_top, cellsize, cellsize)

    features_path = out / "features.tif"
    labels_path = out / "labels.tif"

    with rasterio.open(
        features_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=6,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(np.zeros((6, h, w), dtype="float32"))
    with rasterio.open(
        labels_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(np.zeros((h, w), dtype="uint8"), 1)
    return features_path, labels_path


def _open(features_path: Path, labels_path: Path) -> RasterTileDataset:
    return RasterTileDataset(features_path, labels_path)


def test_different_crs_passes_trivially(tmp_path):
    train = _open(*_write_pair(tmp_path, "train", crs="EPSG:32633"))
    val = _open(*_write_pair(tmp_path, "val", crs="EPSG:2271"))
    # Even though the numeric bounds overlap, different CRS = disjoint Earth.
    assert_no_spatial_overlap(train, val)


def test_same_crs_disjoint_horizontal_passes(tmp_path):
    train = _open(*_write_pair(tmp_path, "train", origin_x=0.0))
    # Val starts at x=100, well past train's x=50 right edge.
    val = _open(*_write_pair(tmp_path, "val", origin_x=100.0))
    assert_no_spatial_overlap(train, val)


def test_same_crs_disjoint_vertical_passes(tmp_path):
    # train: y in [950, 1000]
    train = _open(*_write_pair(tmp_path, "train", origin_x=1000.0, origin_y_top=1000.0))
    # val: y in [850, 900] -- well below train
    val = _open(*_write_pair(tmp_path, "val", origin_x=1000.0, origin_y_top=900.0))
    assert_no_spatial_overlap(train, val)


def test_same_crs_touching_at_edge_passes(tmp_path):
    """Boundary touching is not overlap — zero-area intersection is allowed.
    train x in [0, 50], val x in [50, 100] — they touch at x=50 exactly."""
    train = _open(*_write_pair(tmp_path, "train", origin_x=0.0))
    val = _open(*_write_pair(tmp_path, "val", origin_x=50.0))
    assert_no_spatial_overlap(train, val)


def test_same_crs_overlapping_raises(tmp_path):
    """train x in [0, 50], val x in [25, 75] — overlap in [25, 50]."""
    train = _open(*_write_pair(tmp_path, "train", origin_x=0.0))
    val = _open(*_write_pair(tmp_path, "val", origin_x=25.0))
    with pytest.raises(ValueError, match="overlap"):
        assert_no_spatial_overlap(train, val)


def test_same_crs_one_contained_in_other_raises(tmp_path):
    """Fully-contained val (smaller) inside train (larger) must be flagged."""
    train = _open(*_write_pair(tmp_path, "train", origin_x=0.0, h=100, w=100))
    val = _open(*_write_pair(tmp_path, "val", origin_x=25.0, origin_y_top=75.0, h=20, w=20))
    with pytest.raises(ValueError, match="overlap"):
        assert_no_spatial_overlap(train, val)


def test_same_crs_identical_bounds_raises(tmp_path):
    """Identical datasets are the maximal-overlap case."""
    train = _open(*_write_pair(tmp_path, "train"))
    val = _open(*_write_pair(tmp_path, "val"))
    with pytest.raises(ValueError, match="overlap"):
        assert_no_spatial_overlap(train, val)
