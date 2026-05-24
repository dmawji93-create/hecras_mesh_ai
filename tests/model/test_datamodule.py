"""Tests for the Stage 2 BreaklinePilotDataModule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from hecras_mesh_ai.model import BreaklinePilotDataModule


def _write_synthetic_cache(
    out_dir: Path,
    name: str,
    *,
    h: int = 100,
    w: int = 120,
    feature_count: int = 6,
    cellsize: float = 1.0,
    crs: str = "EPSG:32633",
    origin_x: float = 0.0,
    origin_y_top: float = 100.0,
) -> tuple[Path, Path]:
    """Write a tiny synthetic features.tif + labels.tif pair."""
    out = out_dir / name
    out.mkdir()
    transform = from_origin(origin_x, origin_y_top, cellsize, cellsize)

    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    features = rng.standard_normal((feature_count, h, w)).astype("float32")
    labels = (rng.random((h, w)) > 0.97).astype("uint8")

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
        dst.write(features)
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
        dst.write(labels, 1)
    return features_path, labels_path


@pytest.fixture
def pilot_pair(tmp_path):
    """Two synthetic 'projects' with disjoint extents in the same CRS."""
    tf, tl = _write_synthetic_cache(tmp_path, "train", origin_x=0.0)
    vf, vl = _write_synthetic_cache(tmp_path, "val", origin_x=500.0)  # far east
    return (tf, tl, vf, vl)


def test_datamodule_setup_opens_both_pilots_and_passes_holdout(pilot_pair):
    tf, tl, vf, vl = pilot_pair
    dm = BreaklinePilotDataModule(
        train_features=tf,
        train_labels=tl,
        val_features=vf,
        val_labels=vl,
        tile_size_pixels=16,
        train_samples_per_epoch=10,
        val_samples_per_epoch=5,
        batch_size=2,
    )
    dm.setup()
    assert dm._train_ds is not None
    assert dm._val_ds is not None


def test_datamodule_raises_on_overlapping_pilots(tmp_path):
    """Spatial-holdout violation must surface at setup(), not at training."""
    tf, tl = _write_synthetic_cache(tmp_path, "train", origin_x=0.0)
    # Val overlaps train: same CRS, same origin -> identical bounds.
    vf, vl = _write_synthetic_cache(tmp_path, "val", origin_x=0.0)
    dm = BreaklinePilotDataModule(
        train_features=tf,
        train_labels=tl,
        val_features=vf,
        val_labels=vl,
        tile_size_pixels=16,
        train_samples_per_epoch=10,
        val_samples_per_epoch=5,
        batch_size=2,
    )
    with pytest.raises(ValueError, match="overlap"):
        dm.setup()


def test_train_dataloader_yields_batches_of_expected_shape(pilot_pair):
    tf, tl, vf, vl = pilot_pair
    dm = BreaklinePilotDataModule(
        train_features=tf,
        train_labels=tl,
        val_features=vf,
        val_labels=vl,
        tile_size_pixels=16,
        train_samples_per_epoch=10,
        val_samples_per_epoch=5,
        batch_size=4,
    )
    dm.setup()
    loader = dm.train_dataloader()
    batches = list(loader)
    # 10 samples / batch_size 4 = 2 full batches + 1 partial (size 2).
    assert len(batches) == 3
    for features, labels in batches[:2]:
        assert features.shape == (4, 6, 16, 16)
        assert labels.shape == (4, 16, 16)
        assert features.dtype == torch.float32
        assert labels.dtype == torch.float32


def test_val_dataloader_yields_smaller_batch_count(pilot_pair):
    tf, tl, vf, vl = pilot_pair
    dm = BreaklinePilotDataModule(
        train_features=tf,
        train_labels=tl,
        val_features=vf,
        val_labels=vl,
        tile_size_pixels=16,
        train_samples_per_epoch=10,
        val_samples_per_epoch=4,
        batch_size=2,
    )
    dm.setup()
    loader = dm.val_dataloader()
    batches = list(loader)
    assert len(batches) == 2
    for features, labels in batches:
        assert features.shape == (2, 6, 16, 16)
        assert labels.shape == (2, 16, 16)


def test_train_dataloader_is_deterministic_given_seed(pilot_pair):
    tf, tl, vf, vl = pilot_pair
    common = dict(
        train_features=tf,
        train_labels=tl,
        val_features=vf,
        val_labels=vl,
        tile_size_pixels=16,
        train_samples_per_epoch=5,
        val_samples_per_epoch=2,
        batch_size=1,
        train_seed=123,
    )
    dm1 = BreaklinePilotDataModule(**common)
    dm1.setup()
    dm2 = BreaklinePilotDataModule(**common)
    dm2.setup()
    b1 = [features for features, _ in dm1.train_dataloader()]
    b2 = [features for features, _ in dm2.train_dataloader()]
    for t1, t2 in zip(b1, b2, strict=True):
        torch.testing.assert_close(t1, t2)


def test_hparams_are_saved(pilot_pair):
    """save_hyperparameters() should capture all init kwargs."""
    tf, tl, vf, vl = pilot_pair
    dm = BreaklinePilotDataModule(
        train_features=tf,
        train_labels=tl,
        val_features=vf,
        val_labels=vl,
        tile_size_pixels=32,
        batch_size=4,
        train_seed=99,
    )
    assert dm.hparams.tile_size_pixels == 32
    assert dm.hparams.batch_size == 4
    assert dm.hparams.train_seed == 99
