"""LightningModule wrapping a segmentation_models_pytorch U-Net.

The model:
  - Architecture: U-Net (Ronneberger et al. 2015) — encoder-decoder CNN
    with skip connections. The encoder progressively downsamples + deepens
    the feature map; the decoder mirrors it back up to input resolution;
    skip connections at each level marry the encoder's "what" with the
    decoder's "where" to produce pixel-precise predictions.
  - Encoder backbone: ResNet-18 with ImageNet pretrained weights. ~11M
    params, the lightest mainstream encoder. ImageNet pretraining gives
    the model a strong "what edges and textures look like" prior even
    though our 6-channel DEM input is nothing like RGB photos — smp
    handles the channel mismatch by reinitializing the first conv layer.
  - Output: 1 channel, raw logits (pre-sigmoid). The loss function and
    any post-processing apply sigmoid internally.

Lightning training loop (handled by Trainer, not us):
  for epoch:
    for batch in train_dataloader:
      logits = forward(features)
      loss = bce_dice_loss(logits, labels).total
      loss.backward(); optimizer.step()
      log("train/total_loss", loss); log("train/bce_loss"...); log("train/dice_loss"...)
    for batch in val_dataloader:
      logits = forward(features)
      ... log val losses

We write `forward`, `training_step`, `validation_step`, `configure_optimizers`.
Lightning runs the loop.
"""

from __future__ import annotations

import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from hecras_mesh_ai.model.loss import BCEDiceLoss

# The number of input channels in our feature stack — kept in sync with
# hecras_mesh_ai.features.FEATURE_CHANNELS.
DEFAULT_IN_CHANNELS = 6


class BreaklineUNet(pl.LightningModule):
    """Binary-segmentation U-Net for breakline detection.

    Parameters
    ----------
    in_channels
        Number of input feature channels. Default 6 (matches the Stage 1
        feature stack: elevation, slope, aspect_sin, aspect_cos,
        plan_curvature, profile_curvature).
    encoder_name
        smp encoder identifier. Default "resnet18" — the lightest
        mainstream choice, ~11M params, fastest training.
    encoder_weights
        Pretrained weights to initialize the encoder. Default "imagenet"
        — gives the model a strong "edges and textures" prior even
        though our input is not RGB. Use None to train from scratch
        (useful for ablations).
    learning_rate
        Adam learning rate. Default 1e-3 — standard first-try value.
    dice_weight, bce_pos_weight
        Loss-function hyperparameters. See BCEDiceLoss.
    """

    def __init__(
        self,
        *,
        in_channels: int = DEFAULT_IN_CHANNELS,
        encoder_name: str = "resnet18",
        encoder_weights: str | None = "imagenet",
        learning_rate: float = 1e-3,
        dice_weight: float = 1.0,
        bce_pos_weight: float | None = None,
    ):
        super().__init__()
        # Captures all kwargs to self.hparams; Lightning logs them
        # automatically once W&B is wired (Task 3).
        self.save_hyperparameters()

        self.net: nn.Module = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
        self.loss_fn = BCEDiceLoss(
            dice_weight=dice_weight,
            bce_pos_weight=bce_pos_weight,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        features
            Input batch, shape (B, C, H, W). C must equal `in_channels`.
            H and W must both be divisible by 32 (the U-Net's 5
            downsampling levels each halve resolution).

        Returns
        -------
        Raw logits, shape (B, 1, H, W). Apply sigmoid for probabilities.
        """
        return self.net(features)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], prefix: str) -> torch.Tensor:
        """Shared train/val step logic — forward, compute loss, log."""
        features, labels = batch
        logits = self(features)
        components = self.loss_fn(logits, labels)

        # on_step=True logs each batch (good for training curves);
        # on_epoch=True aggregates over the epoch (good for comparisons).
        # prog_bar=True surfaces the total loss in the Lightning progress bar.
        self.log(
            f"{prefix}/total_loss",
            components.total,
            on_step=(prefix == "train"),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{prefix}/bce_loss",
            components.bce,
            on_step=(prefix == "train"),
            on_epoch=True,
        )
        self.log(
            f"{prefix}/dice_loss",
            components.dice,
            on_step=(prefix == "train"),
            on_epoch=True,
        )
        return components.total

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
