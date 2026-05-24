"""Train the Stage 2 breakline-detection U-Net on the pilot projects.

Two modes:

  --overfit
      Sanity check — train on a tiny fixed batch repeatedly. Loss must
      collapse to near-zero within ~20 epochs. Proves the gradient path
      is wired end-to-end through the real pilot data + GPU. The headline
      Stage 2 Task 4 deliverable.

  (default)
      Full pilot training run. samples_per_epoch * max_epochs gradient
      steps. ~30-60 min on RTX 3090 depending on epoch count.

Usage:
  uv run python scripts/train_pilot.py --overfit
  uv run python scripts/train_pilot.py --max-epochs 20

Logger:
  Lightning's CSVLogger by default — writes lightning_logs/<version>/
  with metrics.csv + hparams.yaml. Swap to WandbLogger by editing
  _make_logger().

Pilot paths default to the Stage 1 exit-notebook cache at
data/processed/stage1_exit/. If absent, run notebooks/03_stage1_exit_
features_and_labels.ipynb first to populate it (or call
hecras_mesh_ai.dataset.cache_pilot_project manually).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from hecras_mesh_ai.model import BreaklinePilotDataModule, BreaklineUNet

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "data" / "processed" / "stage1_exit"


def _existing(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected file not found: {path}\n"
            f"Run notebooks/03_stage1_exit_features_and_labels.ipynb first "
            f"to populate the cache, or pass explicit --*-features / "
            f"--*-labels paths."
        )
    return path


def _default_paths() -> dict[str, Path]:
    return {
        "train_features": DEFAULT_CACHE / "bald_eagle_g09" / "features.tif",
        "train_labels": DEFAULT_CACHE / "bald_eagle_g09" / "labels.tif",
        "val_features": DEFAULT_CACHE / "muncie" / "features.tif",
        "val_labels": DEFAULT_CACHE / "muncie" / "labels.tif",
    }


def _make_logger(overfit: bool) -> CSVLogger:
    name = "overfit_sanity" if overfit else "pilot"
    return CSVLogger(
        save_dir=str(REPO_ROOT / "lightning_logs"),
        name=name,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    defaults = _default_paths()
    p.add_argument("--train-features", type=Path, default=defaults["train_features"])
    p.add_argument("--train-labels", type=Path, default=defaults["train_labels"])
    p.add_argument("--val-features", type=Path, default=defaults["val_features"])
    p.add_argument("--val-labels", type=Path, default=defaults["val_labels"])
    p.add_argument(
        "--overfit",
        action="store_true",
        help="Run the overfit-on-tiny-batch sanity check. Loss must "
        "collapse to ~0 within max_epochs.",
    )
    p.add_argument(
        "--overfit-batches",
        type=int,
        default=4,
        help="Number of batches to overfit on (Lightning's overfit_batches). "
        "Only used with --overfit.",
    )
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument(
        "--train-samples-per-epoch",
        type=int,
        default=1000,
        help="Random tile samples per training epoch.",
    )
    p.add_argument(
        "--val-samples-per-epoch",
        type=int,
        default=200,
        help="Random tile samples per validation epoch.",
    )
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument(
        "--encoder-weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="ImageNet pretraining for the encoder. Use 'none' to " "train from scratch.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader subprocesses. Default 0 — Windows-safe.",
    )
    p.add_argument(
        "--train-positive-fraction",
        type=float,
        default=0.5,
        help="Fraction of training tiles biased to contain at least one "
        "breakline pixel. Counters the ~99/1 class imbalance. Default 0.5. "
        "Use 0 (no bias) for ablation runs.",
    )
    p.add_argument(
        "--val-positive-fraction",
        type=float,
        default=None,
        help="Same for validation. Default unset (representative natural "
        "distribution). Set to 0.5 if val loss is too noisy to compare.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    for p in (args.train_features, args.train_labels, args.val_features, args.val_labels):
        _existing(p)

    # Sensible defaults per mode.
    if args.max_epochs is None:
        args.max_epochs = 20 if args.overfit else 30

    # In overfit mode, sample more often per "epoch" so each epoch sees
    # the same fixed batches multiple times — drives loss down faster.
    if args.overfit:
        # Lightning's overfit_batches uses the first N training batches
        # and reuses them. We still need samples_per_epoch >= batch_size
        # * overfit_batches for that to work meaningfully.
        args.train_samples_per_epoch = max(
            args.train_samples_per_epoch,
            args.batch_size * args.overfit_batches * 4,
        )

    dm = BreaklinePilotDataModule(
        train_features=args.train_features,
        train_labels=args.train_labels,
        val_features=args.val_features,
        val_labels=args.val_labels,
        tile_size_pixels=args.tile_size,
        train_samples_per_epoch=args.train_samples_per_epoch,
        val_samples_per_epoch=args.val_samples_per_epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_seed=args.seed,
        val_seed=args.seed + 1,
        train_positive_fraction=args.train_positive_fraction,
        val_positive_fraction=args.val_positive_fraction,
    )

    model = BreaklineUNet(
        in_channels=6,
        encoder_name="resnet18",
        encoder_weights=None if args.encoder_weights == "none" else "imagenet",
        learning_rate=args.learning_rate,
        dice_weight=args.dice_weight,
    )

    logger = _make_logger(overfit=args.overfit)
    print(f"Logs -> {logger.log_dir}")

    callbacks = []
    if not args.overfit:
        # Two checkpoints saved: best-by-train-loss + the last epoch.
        # We monitor train/total_loss_epoch (not val) because the val loop
        # uses representative sampling on a sparse-positive dataset --
        # most val tiles are empty, so val loss is dominated by trivial
        # "predict 0" success and the val-loss-best checkpoint is the one
        # that learned to predict nothing. Discovered the hard way in the
        # first pilot run. Train loss is a more honest "is the model
        # learning?" signal under our biased sampling regime.
        callbacks.append(
            ModelCheckpoint(
                monitor="train/total_loss_epoch",
                mode="min",
                save_top_k=1,
                save_last=True,
                filename="best-{epoch:02d}-{train/total_loss_epoch:.4f}",
            )
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=5,
        overfit_batches=args.overfit_batches if args.overfit else 0,
        # Disable val loop in overfit mode (Lightning auto-disables it
        # for overfit_batches > 0; explicit for clarity).
        num_sanity_val_steps=0 if args.overfit else 2,
    )

    trainer.fit(model, datamodule=dm)

    print("Done.")
    print(f"  Logs:   {logger.log_dir}")
    if not args.overfit and trainer.checkpoint_callback is not None:
        print(f"  Best:   {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
