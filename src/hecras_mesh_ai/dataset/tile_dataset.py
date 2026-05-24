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

        # Positive-pixel coordinates: loaded lazily on first access. The full
        # labels raster is small (single uint8 band) and the index is the
        # key to efficient positive-biased sampling on rare-positive datasets
        # — rejection sampling would burn 100+ attempts per positive tile.
        self._positive_pixel_rowcol: np.ndarray | None = None

    @property
    def positive_pixel_rowcol(self) -> np.ndarray:
        """Return all (row, col) integer coordinates where labels == 1.

        Loaded once on first access and cached. Shape (N, 2), dtype int64.
        Empty if no positives.
        """
        if self._positive_pixel_rowcol is None:
            with rasterio.open(self.labels_path) as la:
                labels = la.read(1)
            rows, cols = np.where(labels > 0)
            self._positive_pixel_rowcol = np.stack([rows, cols], axis=1).astype(np.int64)
        return self._positive_pixel_rowcol

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
            yield self.next_bbox()

    def next_bbox(self) -> BBox:
        """Draw one random bbox from the dataset's bounds.

        Public so that NaN-rejection consumers (IterableTileDataset) can
        call it directly in a retry loop without exhausting the
        __iter__ stream.
        """
        left, bottom, right, top = self.dataset.bounds
        x = float(self.rng.uniform(left, right - self.tile_w))
        y = float(self.rng.uniform(bottom, top - self.tile_h))
        return (x, y, x + self.tile_w, y + self.tile_h)

    def next_positive_centered_bbox(self) -> BBox:
        """Pick a random positive pixel and return a tile centered on it.

        For rare-positive datasets (breaklines are ~0.005% of pixels on
        Bald Eagle), rejection sampling burns 100+ attempts per positive
        tile. This direct method picks a positive (row, col) from the
        pre-computed index, converts to CRS-space coordinates, and
        returns the centered tile bbox (clamped to dataset bounds).
        """
        coords = self.dataset.positive_pixel_rowcol
        if coords.size == 0:
            raise RuntimeError(
                f"Dataset has no positive pixels — cannot center tiles. "
                f"Check labels file: {self.dataset.labels_path}"
            )

        # Pick one positive pixel.
        i = int(self.rng.integers(coords.shape[0]))
        row, col = coords[i]

        # Convert pixel (row, col) -> CRS-space (x, y) at pixel center.
        transform = self.dataset.transform
        cx = transform.c + transform.a * (col + 0.5)
        cy = transform.f + transform.e * (row + 0.5)

        # Center the tile on (cx, cy), clamp to dataset bounds so the
        # whole tile lies inside.
        left, bottom, right, top = self.dataset.bounds
        minx = max(left, min(cx - self.tile_w / 2, right - self.tile_w))
        miny = max(bottom, min(cy - self.tile_h / 2, top - self.tile_h))
        return (minx, miny, minx + self.tile_w, miny + self.tile_h)


class IterableTileDataset(IterableDataset):
    """PyTorch IterableDataset adapter — pairs a RasterTileDataset with a sampler.

    Yields (features, labels) tensor pairs. Wrap in a torch.utils.data.DataLoader
    to drive training.

    With num_workers > 0, each worker re-runs __iter__ and the underlying
    sampler's RNG is forked from a worker-specific seed (PyTorch's standard
    behavior). For deterministic training across worker counts, set
    samples_per_epoch on the sampler and let the DataLoader handle worker
    sharding via its own seed.

    NaN-tile rejection. Real DEMs have nodata regions (water bodies, off-survey
    areas, etc.) that the Stage 1 hybrid conditioner deliberately preserves as
    NaN — only isolated single-pixel holes get patched. If a random tile
    overlaps a preserved-NaN region, its features contain NaN and downstream
    training silently corrupts to NaN loss. The default `skip_nan_tiles=True`
    rejects any tile whose features contain any NaN and draws a replacement
    bbox; `max_attempts_per_tile` caps the retry loop.

    Positive-content biasing. The breakline-detection positive class is rare
    (~0.9% of pixels per the Stage 1 exit notebook). Random tiles mostly
    contain no breakline pixels, so the model can game BCE by predicting
    "0 everywhere" and never learns the positive class (Dice stays stuck at
    ~1). Pass `positive_fraction` in [0, 1] to **upweight** positive tiles:
    on each yield, with probability `positive_fraction`, the iterator
    insists on a tile with `labels.sum() > 0`; the rest of the time, any
    NaN-clean tile is accepted (which on these pilots will mostly be
    empty — matching the natural distribution). Default None disables
    biasing entirely — representative sampling, appropriate for validation.
    """

    def __init__(
        self,
        raster_dataset: RasterTileDataset,
        sampler: RandomTileSampler,
        *,
        skip_nan_tiles: bool = True,
        positive_fraction: float | None = None,
        max_attempts_per_tile: int = 50,
        bias_seed: int = 0,
    ):
        super().__init__()
        if max_attempts_per_tile < 1:
            raise ValueError(f"max_attempts_per_tile must be >= 1, got {max_attempts_per_tile}")
        if positive_fraction is not None and not 0.0 <= positive_fraction <= 1.0:
            raise ValueError(f"positive_fraction must be in [0, 1], got {positive_fraction}")
        self.raster_dataset = raster_dataset
        self.sampler = sampler
        self.skip_nan_tiles = skip_nan_tiles
        self.positive_fraction = positive_fraction
        self.max_attempts_per_tile = max_attempts_per_tile
        self._bias_rng = np.random.default_rng(bias_seed)

    def _tile_acceptable(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        must_be_positive: bool,
    ) -> bool:
        if self.skip_nan_tiles and torch.isnan(features).any():
            return False
        # Positive-required slot rejects empty tiles; no other constraint
        # on label content (no "must-be-empty" branch — see class docstring).
        return not (must_be_positive and labels.sum() == 0)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # Fast path: no filtering at all.
        if not self.skip_nan_tiles and self.positive_fraction is None:
            for bbox in self.sampler:
                yield self.raster_dataset.sample(bbox)
            return

        for _ in range(self.sampler.samples_per_epoch):
            # Coin-flip whether this slot must contain a breakline pixel.
            # Asymmetric semantics: "must_be_positive=True" hard-requires
            # labels.sum() > 0; "False" places no requirement (any non-NaN
            # tile is fine, including empties — which dominate the
            # natural distribution on these pilots). positive_fraction=None
            # is equivalent to always-False.
            must_be_positive = (
                self.positive_fraction is not None
                and self._bias_rng.random() < self.positive_fraction
            )

            for _attempt in range(self.max_attempts_per_tile):
                # Positive-centered draw when must_be_positive (efficient
                # on rare-positive datasets); random otherwise.
                bbox = (
                    self.sampler.next_positive_centered_bbox()
                    if must_be_positive
                    else self.sampler.next_bbox()
                )
                features, labels = self.raster_dataset.sample(bbox)
                if self._tile_acceptable(features, labels, must_be_positive):
                    yield features, labels
                    break
            else:
                raise RuntimeError(
                    f"Could not find an acceptable tile in "
                    f"{self.max_attempts_per_tile} attempts "
                    f"(skip_nan_tiles={self.skip_nan_tiles}, "
                    f"must_be_positive={must_be_positive}). The dataset may "
                    f"have very large NaN regions or too few positive pixels; "
                    f"investigate the source or raise max_attempts_per_tile."
                )
