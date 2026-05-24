"""LightningDataModule for the Stage 2 breakline-detection pilot.

Wraps the Stage 1 cached features.tif + labels.tif pairs (one per pilot
project) into a single object that Lightning's Trainer can drive. Holds
the train/val split decision, the spatial-holdout assertion, the sampler
configuration, and the DataLoader plumbing.

Train pilot: Bald Eagle g09 (4 named breaklines including SayersDam).
Val   pilot: Muncie         (2 breaklines: Road 1, HighGround 1).
"""

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from hecras_mesh_ai.dataset import (
    IterableTileDataset,
    RandomTileSampler,
    RasterTileDataset,
    assert_no_spatial_overlap,
)


class BreaklinePilotDataModule(pl.LightningDataModule):
    """Two cached pilots in, train/val (features, labels) batches out.

    Parameters
    ----------
    train_features, train_labels
        Paths to the cached train pilot's features.tif and labels.tif.
    val_features, val_labels
        Paths to the cached val pilot's features.tif and labels.tif.
    tile_size_pixels
        Side length of square training tiles in pixels. Default 256 —
        the standard segmentation tile size; balances spatial context
        against batch parallelism and fits cleanly on the RTX 3090.
    train_samples_per_epoch, val_samples_per_epoch
        How many random tiles each loader yields per epoch. "Epoch" here
        is a fixed number of samples, not a full pass over a finite set
        (the underlying dataset is a continuous-extent raster, not a
        discrete row count). Defaults: 1000 train, 200 val.
    batch_size
        Tiles per gradient step. Default 8 — comfortable headroom on the
        24 GB RTX 3090 with a ResNet-18 U-Net and 256x256 tiles.
    num_workers
        DataLoader subprocesses. Default 0 (single-process) — multi-worker
        DataLoaders on Windows require the script to be guarded by
        `if __name__ == "__main__":`. For the pilot, 0 is fine.
    train_seed, val_seed
        RNG seeds for the train and val samplers. Both default to fixed
        values for reproducibility; the val seed differs from the train
        seed so the two samplers don't produce coincidentally identical
        bbox sequences (they sample from disjoint datasets, but matching
        sequences would still be a confusing coincidence).
    """

    def __init__(
        self,
        train_features: Path | str,
        train_labels: Path | str,
        val_features: Path | str,
        val_labels: Path | str,
        *,
        tile_size_pixels: int = 256,
        train_samples_per_epoch: int = 1000,
        val_samples_per_epoch: int = 200,
        batch_size: int = 8,
        num_workers: int = 0,
        train_seed: int = 42,
        val_seed: int = 43,
    ):
        super().__init__()
        # save_hyperparameters captures every __init__ kwarg into self.hparams
        # and Lightning will log them automatically once W&B is wired in
        # Stage 2 Task 3.
        self.save_hyperparameters()
        self._train_features = Path(train_features)
        self._train_labels = Path(train_labels)
        self._val_features = Path(val_features)
        self._val_labels = Path(val_labels)
        self._train_ds: RasterTileDataset | None = None
        self._val_ds: RasterTileDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Open both cached pilots and assert spatial holdout.

        Called once per process by the Lightning Trainer (with stage in
        {"fit", "validate", "test", "predict"}). We open eagerly because
        the spatial-holdout check is cheap and we want any
        misconfiguration to surface before training begins.
        """
        if self._train_ds is None:
            self._train_ds = RasterTileDataset(self._train_features, self._train_labels)
        if self._val_ds is None:
            self._val_ds = RasterTileDataset(self._val_features, self._val_labels)
        assert_no_spatial_overlap(self._train_ds, self._val_ds)

    def _make_loader(
        self,
        dataset: RasterTileDataset,
        seed: int,
        samples_per_epoch: int,
    ) -> DataLoader:
        sampler = RandomTileSampler(
            dataset,
            tile_size_pixels=self.hparams.tile_size_pixels,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
        )
        iterable = IterableTileDataset(dataset, sampler)
        return DataLoader(
            iterable,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "call setup() before train_dataloader()"
        return self._make_loader(
            self._train_ds,
            seed=self.hparams.train_seed,
            samples_per_epoch=self.hparams.train_samples_per_epoch,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None, "call setup() before val_dataloader()"
        return self._make_loader(
            self._val_ds,
            seed=self.hparams.val_seed,
            samples_per_epoch=self.hparams.val_samples_per_epoch,
        )
