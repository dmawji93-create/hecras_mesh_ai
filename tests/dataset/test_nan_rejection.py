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
    with pytest.raises(RuntimeError, match="acceptable tile"):
        list(it)


def test_max_attempts_per_tile_validated(tmp_path):
    features_path, labels_path = _write_cache_with_nan_region(tmp_path)
    ds = RasterTileDataset(features_path, labels_path)
    sampler = RandomTileSampler(ds, tile_size_pixels=8, samples_per_epoch=5, seed=42)
    with pytest.raises(ValueError, match=">= 1"):
        IterableTileDataset(ds, sampler, max_attempts_per_tile=0)


# ---------------------------------------------------------------------------
# Positive-content biasing
# ---------------------------------------------------------------------------


def _write_cache_with_known_positives(
    tmp_path: Path,
    *,
    h: int = 100,
    w: int = 100,
    positive_box: tuple[int, int, int, int] = (40, 40, 70, 70),
) -> tuple[Path, Path]:
    """Synthetic cache where positives live in a known central rectangle.
    Most random tiles will have zero positives; tiles overlapping the
    rectangle will have many."""
    out = tmp_path / "with_positives"
    out.mkdir()
    transform = from_origin(1000.0, 1000.0 + h, 1.0, 1.0)
    features = np.random.default_rng(0).standard_normal((6, h, w)).astype("float32")
    labels = np.zeros((h, w), dtype="uint8")
    r0, c0, r1, c1 = positive_box
    labels[r0:r1, c0:c1] = 1

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


def test_positive_fraction_1_yields_only_positive_tiles(tmp_path):
    features_path, labels_path = _write_cache_with_known_positives(tmp_path)
    ds = RasterTileDataset(features_path, labels_path)
    sampler = RandomTileSampler(ds, tile_size_pixels=8, samples_per_epoch=20, seed=42)
    it = IterableTileDataset(ds, sampler, positive_fraction=1.0)
    pairs = list(it)
    assert len(pairs) == 20
    for _, labels in pairs:
        assert labels.sum() > 0, "positive_fraction=1.0 must yield only positive tiles"


def test_positive_fraction_0_equivalent_to_none(tmp_path):
    """positive_fraction=0 means 'never guarantee positive' which is the
    same as no biasing — representative sampling that yields whatever the
    sampler draws (mostly empty on this dataset)."""
    features_path, labels_path = _write_cache_with_known_positives(tmp_path)
    ds = RasterTileDataset(features_path, labels_path)
    sampler = RandomTileSampler(ds, tile_size_pixels=8, samples_per_epoch=20, seed=42)
    it = IterableTileDataset(ds, sampler, positive_fraction=0.0)
    pairs = list(it)
    n_pos = sum(1 for _, labels in pairs if labels.sum() > 0)
    # No bias = representative sampling on a mostly-empty dataset.
    assert n_pos < 0.5 * len(pairs)


def test_positive_fraction_half_yields_at_least_half_positive(tmp_path):
    """With positive_fraction=0.5, AT LEAST ~50% should be positive
    (positive-guaranteed slots contribute 50% baseline; the remaining
    no-constraint slots may add more if the natural distribution does).
    Hard lower bound: well above the no-bias baseline."""
    features_path, labels_path = _write_cache_with_known_positives(tmp_path)
    ds = RasterTileDataset(features_path, labels_path)

    # No-bias baseline first.
    s_base = RandomTileSampler(ds, tile_size_pixels=8, samples_per_epoch=60, seed=42)
    it_base = IterableTileDataset(ds, s_base, positive_fraction=None)
    baseline_pos = sum(1 for _, labels in it_base if labels.sum() > 0)

    s_bias = RandomTileSampler(ds, tile_size_pixels=8, samples_per_epoch=60, seed=42)
    it_bias = IterableTileDataset(ds, s_bias, positive_fraction=0.5, bias_seed=42)
    biased_pos = sum(1 for _, labels in it_bias if labels.sum() > 0)

    # Biasing must increase the positive count well above baseline. A
    # statistical lower bound for 0.5 bias is roughly half the
    # samples_per_epoch (60 * 0.5 = 30) plus whatever the natural rate
    # adds. Loosely: biased >= 25.
    assert biased_pos >= 25, f"biased positives {biased_pos} should be >= 25"
    assert (
        biased_pos > baseline_pos
    ), f"biasing did not increase positives: baseline={baseline_pos}, biased={biased_pos}"


def test_positive_fraction_none_disables_bias(tmp_path):
    """positive_fraction=None preserves the no-bias behavior."""
    features_path, labels_path = _write_cache_with_known_positives(tmp_path)
    ds = RasterTileDataset(features_path, labels_path)
    sampler = RandomTileSampler(ds, tile_size_pixels=8, samples_per_epoch=30, seed=42)
    it = IterableTileDataset(ds, sampler, positive_fraction=None)
    pairs = list(it)
    # With most of the raster empty and small tiles, mostly-empty is expected.
    n_pos = sum(1 for _, labels in pairs if labels.sum() > 0)
    n_empty = len(pairs) - n_pos
    assert n_empty > n_pos, "no-bias sampling should be representative (mostly empty)"


def test_invalid_positive_fraction_raises(tmp_path):
    features_path, labels_path = _write_cache_with_known_positives(tmp_path)
    ds = RasterTileDataset(features_path, labels_path)
    sampler = RandomTileSampler(ds, tile_size_pixels=8, samples_per_epoch=5, seed=42)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        IterableTileDataset(ds, sampler, positive_fraction=1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        IterableTileDataset(ds, sampler, positive_fraction=-0.1)
