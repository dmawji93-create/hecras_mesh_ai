"""Serve (features, labels) tiles from cached pilot GeoTIFFs.

Three classes, intentionally separate so each has one job:

  - RasterTileDataset   : opens a features.tif + labels.tif pair, exposes
                          bounds / crs / shape, reads a tile by bounding
                          box. Returns torch tensors.
  - RandomTileSampler   : yields random tile bounding boxes that lie
                          entirely inside a dataset's bounds. Tile size
                          is given in pixels; converted to CRS units via
                          the dataset's cellsize.
  - IterableTileDataset : torch.utils.data.IterableDataset adapter that
                          combines a raster dataset with a sampler so a
                          PyTorch DataLoader can drive training.

Separation lets us swap samplers (random vs grid vs center-on-known-feature)
without touching the raster reader, and lets non-PyTorch consumers (the
exploration notebook, sanity checks) use the lower layers directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window, from_bounds
from torch.utils.data import IterableDataset

BBox = tuple[float, float, float, float]  # (minx, miny, maxx, maxy) in CRS units


class RasterTileDataset:
    """Opens a cached features.tif + labels.tif pair and reads tiles.

    The two files must be pixel-aligned: identical CRS, transform, and
    shape. The labels file must be single-band uint8 with values in {0, 1};
    the features file must be multi-band float32 (typically 6 channels in
    FEATURE_CHANNELS order, but any band count is accepted).

    Open per call rather than holding a persistent handle — keeps
    PyTorch DataLoader multi-worker forking safe at the cost of a small
    per-tile open() overhead.
    """

    def __init__(self, features_path: Path | str, labels_path: Path | str):
        features_path = Path(features_path)
        labels_path = Path(labels_path)

        with rasterio.open(features_path) as f:
            self.feature_count = f.count
            self.transform = f.transform
            self.shape = (f.height, f.width)
            self.crs = f.crs
            self.bounds = tuple(f.bounds)  # (left, bottom, right, top)
        with rasterio.open(labels_path) as la:
            if la.crs != self.crs:
                raise ValueError(f"label CRS {la.crs} does not match feature CRS {self.crs}")
            if la.transform != self.transform:
                raise ValueError("label transform does not match feature transform")
            if (la.height, la.width) != self.shape:
                raise ValueError(
                    f"label shape {(la.height, la.width)} != feature shape {self.shape}"
                )
            if la.count != 1:
                raise ValueError(f"labels must be single-band, got {la.count}")

        self.features_path = features_path
        self.labels_path = labels_path

        # Cellsize in CRS units — positive magnitudes (transform.e is negative).
        self.cellsize_x = float(abs(self.transform.a))
        self.cellsize_y = float(abs(self.transform.e))

    def _bbox_inside_bounds(self, bbox: BBox) -> bool:
        left, bottom, right, top = self.bounds
        minx, miny, maxx, maxy = bbox
        return minx >= left and miny >= bottom and maxx <= right and maxy <= top

    def sample(self, bbox: BBox) -> tuple[torch.Tensor, torch.Tensor]:
        """Read a (features, labels) tile from the bbox.

        Returns
        -------
        features : torch.Tensor, shape (C, H, W), dtype float32
        labels   : torch.Tensor, shape (H, W),    dtype float32 in {0, 1}
                   (float for BCE-with-logits compatibility in Stage 2)
        """
        if not self._bbox_inside_bounds(bbox):
            raise ValueError(f"bbox {bbox} extends beyond dataset bounds {self.bounds}")
        window = from_bounds(*bbox, transform=self.transform)
        with rasterio.open(self.features_path) as f:
            features = f.read(window=window).astype(np.float32)
        with rasterio.open(self.labels_path) as la:
            labels = la.read(1, window=window).astype(np.float32)
        return torch.from_numpy(features), torch.from_numpy(labels)

    def sample_window(self, window: Window) -> tuple[torch.Tensor, torch.Tensor]:
        """Read by a rasterio Window directly — convenient for grid sampling."""
        with rasterio.open(self.features_path) as f:
            features = f.read(window=window).astype(np.float32)
        with rasterio.open(self.labels_path) as la:
            labels = la.read(1, window=window).astype(np.float32)
        return torch.from_numpy(features), torch.from_numpy(labels)


class RandomTileSampler:
    """Yields random tile bounding boxes that fit entirely inside a dataset.

    Tile size is specified in pixels and converted to CRS units via the
    dataset's cellsize. The sampler is deterministic given a seed —
    important for reproducible training.
    """

    def __init__(
        self,
        dataset: RasterTileDataset,
        *,
        tile_size_pixels: int = 256,
        samples_per_epoch: int = 1000,
        seed: int | None = None,
    ):
        if tile_size_pixels <= 0:
            raise ValueError(f"tile_size_pixels must be > 0, got {tile_size_pixels}")
        if samples_per_epoch <= 0:
            raise ValueError(f"samples_per_epoch must be > 0, got {samples_per_epoch}")
        self.dataset = dataset
        self.tile_size_pixels = tile_size_pixels
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)

        self.tile_w = tile_size_pixels * dataset.cellsize_x
        self.tile_h = tile_size_pixels * dataset.cellsize_y

        left, bottom, right, top = dataset.bounds
        if (right - left) < self.tile_w or (top - bottom) < self.tile_h:
            raise ValueError(
                f"tile size ({self.tile_w:.1f} x {self.tile_h:.1f} CRS units) "
                f"exceeds dataset bounds (width {right - left:.1f}, "
                f"height {top - bottom:.1f})"
            )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self) -> Iterator[BBox]:
        for _ in range(self.samples_per_epoch):
            yield self._random_bbox()

    def _random_bbox(self) -> BBox:
        left, bottom, right, top = self.dataset.bounds
        x = float(self.rng.uniform(left, right - self.tile_w))
        y = float(self.rng.uniform(bottom, top - self.tile_h))
        return (x, y, x + self.tile_w, y + self.tile_h)


class IterableTileDataset(IterableDataset):
    """PyTorch IterableDataset adapter — pairs a RasterTileDataset with a sampler.

    Yields (features, labels) tensor pairs. Wrap in a torch.utils.data.DataLoader
    to drive training.

    With num_workers > 0, each worker re-runs __iter__ and the underlying
    sampler's RNG is forked from a worker-specific seed (PyTorch's standard
    behavior). For deterministic training across worker counts, set
    samples_per_epoch on the sampler and let the DataLoader handle worker
    sharding via its own seed.
    """

    def __init__(
        self,
        raster_dataset: RasterTileDataset,
        sampler: RandomTileSampler,
    ):
        super().__init__()
        self.raster_dataset = raster_dataset
        self.sampler = sampler

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for bbox in self.sampler:
            yield self.raster_dataset.sample(bbox)
