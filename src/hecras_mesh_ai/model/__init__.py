"""PyTorch Lightning model + datamodule for breakline detection (Stage 2).

- datamodule : LightningDataModule wrapping the Stage 1 cached pilots.
- loss       (next) : BCE + Dice combined loss for class-imbalanced
                      binary segmentation.
- unet       (next) : LightningModule wrapping a segmentation_models_pytorch
                      U-Net with a ResNet-18 encoder (ImageNet pretrained).
"""

from hecras_mesh_ai.model.datamodule import BreaklinePilotDataModule
from hecras_mesh_ai.model.loss import BCEDiceLoss, LossComponents, dice_loss

__all__ = [
    "BCEDiceLoss",
    "BreaklinePilotDataModule",
    "LossComponents",
    "dice_loss",
]
