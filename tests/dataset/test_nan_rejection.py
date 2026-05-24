"""Tests for IterableTileDataset NaN-rejection sampling."""

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


def _write_cache_with_nan_region(
    tmp_path: Path,
    *,
    h: int = 100,
    w: int = 100,
    nan_band_idx: int = 0,  # which feature band to corrupt
    nan_box: tuple[int, int, int, int] = (40, 40, 60, 60),  # rows, cols
) -> tuple[Path, Path]:
    """Synthetic cache where part of one feature band is NaN."""
    out = tmp_path / "synthetic"
    out.mkdir()
    transform = from_origin(1000.0, 1000.0 + h, 1.0, 1.0)

    features = np.random.default_rng(0).standard_normal((6, h, w)).astype("float32")
    r0, c0, r1, c1 = nan_box
    features[nan_band_idx, r0:r1, c0:c1] = np.nan
    labels = (np.random.default_rng(1).random((h, w)) > 0.95).astype("uint8")

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
        crs="EPSG:32633",
        transform=transform,
    ) as dst:
        dst.write(features)
    with rasterio.open(
        labels_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="uint8",
        crs="EPSG:32633",
        transform=transform,
    ) as dst:
        dst.write(labels, 1)
    return features_path, labels_path


def test_nan_rejection_yields_only_clean_tiles(tmp_path):
    features_path, labels_path = _write_cache_with_nan_region(tmp_path)
    ds = RasterTileDataset(features_path, labels_path)
    sampler = RandomTileSampler(ds, tile_size_pixels=8, samples_per_epoch=30, seed=42)
    it = IterableTileDataset(ds, sampler, skip_nan_tiles=True)

    pairs = list(it)
    assert len(pairs) == 30
    for features, _labels in pairs:
        assert not torch.isnan(features).any(), "NaN-rejection failed to skip a NaN tile"


def test_nan_rejection_disabled_yields_nan_tiles(tmp_path):
    """When skip_nan_tiles=False, the iterator yields whatever the sampler
    draws — including NaN-containing tiles. Demonstrates the bug we
    surfaced in Stage 2 overfit run."""
    features_path, labels_path = _write_cache_with_nan_region(tmp_path)
    ds = RasterTileDataset(features_path, labels_path)
    sampler = RandomTileSampler(ds, tile_size_pixels=8, samples_per_epoch=200, seed=42)
    it = IterableTileDataset(ds, sampler, skip_nan_tiles=False)

    pairs = list(it)
    any_nan = any(torch.isnan(features).any() for features, _ in pairs)
    assert any_nan, "synthetic test data should yield at least one NaN tile"


def test_nan_rejection_raises_when_no_clean_tile_found(tmp_path):
    """If the entire dataset is NaN, the rejection loop should give up
    and raise rather than spin forever."""
    out = tmp_path / "all_nan"
    out.mkdir()
    transform = from_origin(1000.0, 1100.0, 1.0, 1.0)
    features = np.full((6, 100, 100), np.nan, dtype="float32")
    labels = np.zeros((100, 100), dtype="uint8")
    fpath = out / "features.tif"
    lpath = out / "labels.tif"
    with rasterio.open(
        fpath,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=6,
        dtype="float32",
        crs="EPSG:32633",
        transform=transform,
    ) as dst:
        dst.write(features)
    with rasterio.open(
        lpath,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype="uint8",
        crs="EPSG:32633",
        transform=transform,
    ) as dst:
        dst.write(labels, 1)

    ds = RasterTileDataset(fpath, lpath)
    sampler = RandomTileSampler(ds, tile_size_pixels=8, samples_per_epoch=5, seed=42)
    it = IterableTileDataset(ds, sampler, skip_nan_tiles=True, max_attempts_per_tile=10)
    with pytest.raises(RuntimeError, match="NaN-free"):
        list(it)


def test_max_attempts_per_tile_validated(tmp_path):
    features_path, labels_path = _write_cache_with_nan_region(tmp_path)
    ds = RasterTileDataset(features_path, labels_path)
    sampler = RandomTileSampler(ds, tile_size_pixels=8, samples_per_epoch=5, seed=42)
    with pytest.raises(ValueError, match=">= 1"):
        IterableTileDataset(ds, sampler, max_attempts_per_tile=0)
