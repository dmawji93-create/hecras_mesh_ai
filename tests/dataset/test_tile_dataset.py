"""Tests for the tile dataset, sampler, and PyTorch iterable adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from hecras_mesh_ai.dataset import (
    IterableTileDataset,
    RandomTileSampler,
    RasterTileDataset,
)


def _write_synthetic_cache(
    tmp_path: Path,
    *,
    h: int = 100,
    w: int = 120,
    feature_count: int = 6,
    cellsize: float = 1.0,
    crs: str = "EPSG:32633",
) -> tuple[Path, Path]:
    """Write a synthetic features.tif + labels.tif pair shaped like a real cache."""
    out = tmp_path / "synthetic"
    out.mkdir()
    transform = from_origin(1000.0, 1000.0 + h * cellsize, cellsize, cellsize)

    features_data = (
        np.random.default_rng(0).standard_normal((feature_count, h, w)).astype("float32")
    )
    labels_data = (np.random.default_rng(1).random((h, w)) > 0.95).astype("uint8")

    features_path = out / "features.tif"
    labels_path = out / "labels.tif"

    with rasterio.open(
        features_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=feature_count,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(features_data)
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
        dst.write(labels_data, 1)

    return features_path, labels_path


# ---------------------------------------------------------------------------
# RasterTileDataset
# ---------------------------------------------------------------------------


def test_dataset_opens_and_exposes_metadata(tmp_path):
    features_path, labels_path = _write_synthetic_cache(tmp_path, h=100, w=120, feature_count=6)
    ds = RasterTileDataset(features_path, labels_path)

    assert ds.feature_count == 6
    assert ds.shape == (100, 120)
    assert ds.crs.to_epsg() == 32633
    assert ds.cellsize_x == 1.0
    assert ds.cellsize_y == 1.0
    left, bottom, right, top = ds.bounds
    assert left == 1000.0
    assert bottom == 1000.0
    assert right == 1120.0  # left + w * cellsize_x
    assert top == 1100.0  # bottom + h * cellsize_y


def test_dataset_raises_on_mismatched_crs(tmp_path):
    features_path, labels_path = _write_synthetic_cache(tmp_path)
    # Rewrite labels with a different CRS.
    with rasterio.open(labels_path) as src:
        meta = src.meta.copy()
        data = src.read(1)
    meta["crs"] = "EPSG:2271"
    with rasterio.open(labels_path, "w", **meta) as dst:
        dst.write(data, 1)
    with pytest.raises(ValueError, match="CRS"):
        RasterTileDataset(features_path, labels_path)


def test_dataset_raises_on_mismatched_shape(tmp_path):
    features_path, labels_path = _write_synthetic_cache(tmp_path)
    # Write a smaller labels file.
    with rasterio.open(features_path) as src:
        transform = src.transform
        crs = src.crs
    with rasterio.open(
        labels_path,
        "w",
        driver="GTiff",
        height=50,
        width=60,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(np.zeros((50, 60), dtype="uint8"), 1)
    with pytest.raises(ValueError, match="shape"):
        RasterTileDataset(features_path, labels_path)


def test_dataset_sample_returns_tensors_of_expected_shape_and_dtype(tmp_path):
    features_path, labels_path = _write_synthetic_cache(tmp_path, h=100, w=120, feature_count=6)
    ds = RasterTileDataset(features_path, labels_path)

    # A 10x10 CRS-unit bbox (= 10 pixels at cellsize=1) starting at (1010, 1010).
    bbox = (1010.0, 1010.0, 1020.0, 1020.0)
    features, labels = ds.sample(bbox)

    assert isinstance(features, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert features.shape == (6, 10, 10)
    assert labels.shape == (10, 10)
    assert features.dtype == torch.float32
    assert labels.dtype == torch.float32


def test_dataset_sample_raises_when_bbox_outside_bounds(tmp_path):
    features_path, labels_path = _write_synthetic_cache(tmp_path)
    ds = RasterTileDataset(features_path, labels_path)
    bbox_outside = (-100.0, -100.0, -50.0, -50.0)
    with pytest.raises(ValueError, match="bounds"):
        ds.sample(bbox_outside)


# ---------------------------------------------------------------------------
# RandomTileSampler
# ---------------------------------------------------------------------------


def test_sampler_yields_expected_count_and_bboxes_inside_bounds(tmp_path):
    features_path, labels_path = _write_synthetic_cache(tmp_path, h=100, w=120)
    ds = RasterTileDataset(features_path, labels_path)

    sampler = RandomTileSampler(ds, tile_size_pixels=16, samples_per_epoch=50, seed=42)
    bboxes = list(sampler)

    assert len(bboxes) == 50
    left, bottom, right, top = ds.bounds
    for minx, miny, maxx, maxy in bboxes:
        assert left <= minx and maxx <= right
        assert bottom <= miny and maxy <= top
        # Tile is 16 pixels at cellsize=1 -> 16 CRS units wide/tall.
        assert maxx - minx == pytest.approx(16.0)
        assert maxy - miny == pytest.approx(16.0)


def test_sampler_is_deterministic_given_seed(tmp_path):
    features_path, labels_path = _write_synthetic_cache(tmp_path)
    ds = RasterTileDataset(features_path, labels_path)

    s1 = RandomTileSampler(ds, tile_size_pixels=16, samples_per_epoch=10, seed=123)
    s2 = RandomTileSampler(ds, tile_size_pixels=16, samples_per_epoch=10, seed=123)
    assert list(s1) == list(s2)


def test_sampler_raises_when_tile_exceeds_dataset(tmp_path):
    features_path, labels_path = _write_synthetic_cache(tmp_path, h=20, w=20)
    ds = RasterTileDataset(features_path, labels_path)
    with pytest.raises(ValueError, match="exceeds"):
        RandomTileSampler(ds, tile_size_pixels=64, samples_per_epoch=1)


def test_sampler_invalid_args_raise(tmp_path):
    features_path, labels_path = _write_synthetic_cache(tmp_path)
    ds = RasterTileDataset(features_path, labels_path)
    with pytest.raises(ValueError):
        RandomTileSampler(ds, tile_size_pixels=0, samples_per_epoch=10)
    with pytest.raises(ValueError):
        RandomTileSampler(ds, tile_size_pixels=16, samples_per_epoch=0)


# ---------------------------------------------------------------------------
# IterableTileDataset (PyTorch adapter)
# ---------------------------------------------------------------------------


def test_iterable_dataset_yields_tensor_pairs(tmp_path):
    features_path, labels_path = _write_synthetic_cache(tmp_path, h=100, w=120, feature_count=6)
    ds = RasterTileDataset(features_path, labels_path)
    sampler = RandomTileSampler(ds, tile_size_pixels=16, samples_per_epoch=5, seed=0)
    it = IterableTileDataset(ds, sampler)

    pairs = list(it)
    assert len(pairs) == 5
    for features, labels in pairs:
        assert features.shape == (6, 16, 16)
        assert labels.shape == (16, 16)
        assert features.dtype == torch.float32
        assert labels.dtype == torch.float32


def test_iterable_dataset_compatible_with_torch_dataloader(tmp_path):
    """The whole point — confirm a vanilla DataLoader can drive batches."""
    features_path, labels_path = _write_synthetic_cache(tmp_path, h=100, w=120, feature_count=6)
    ds = RasterTileDataset(features_path, labels_path)
    sampler = RandomTileSampler(ds, tile_size_pixels=16, samples_per_epoch=8, seed=0)
    it = IterableTileDataset(ds, sampler)

    loader = torch.utils.data.DataLoader(it, batch_size=4)
    batches = list(loader)
    assert len(batches) == 2  # 8 samples / batch_size 4
    for features_batch, labels_batch in batches:
        assert features_batch.shape == (4, 6, 16, 16)
        assert labels_batch.shape == (4, 16, 16)
