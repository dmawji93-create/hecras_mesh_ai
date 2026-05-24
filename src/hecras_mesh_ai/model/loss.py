"""Binary-segmentation loss for breakline detection: BCE + Dice combined.

Why combined.
  - BCE alone is per-pixel and smooth, but collapses under heavy class
    imbalance: with ~99% non-breakline pixels, "predict 0 everywhere" gets
    a near-perfect BCE and the model converges there.
  - Dice alone is naturally class-imbalance-robust (set-overlap metric;
    "predict 0 everywhere" yields the worst score, not the best), but its
    gradients are trickier at low overlap, especially early in training.
  - Sum the two and you get BCE's stable gradients early plus Dice's
    pressure to find the rare positives. De-facto standard for sparse-
    positive segmentation (the same shape as medical lesion detection).

The combined loss in math:

    L = BCE(logits, target) + dice_weight * DiceLoss(sigmoid(logits), target)

We operate on **logits** (raw model outputs, pre-sigmoid) for numerical
stability — torch's `binary_cross_entropy_with_logits` and our Dice both
fold the sigmoid in internally.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossComponents:
    """Holds the combined loss + individual components for separate logging."""

    total: torch.Tensor
    bce: torch.Tensor
    dice: torch.Tensor


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft Dice loss over a batch of binary segmentation predictions.

    Parameters
    ----------
    logits
        Model output, shape (B, 1, H, W) or (B, H, W). Real-valued
        (pre-sigmoid).
    target
        Ground truth binary mask, same broadcastable shape as logits,
        values in {0, 1}, float32 dtype.
    smooth
        Numerator + denominator smoothing constant. Larger -> stronger
        regularization toward "loss = 0 even when both sets are tiny."
        Default 1.0 is the textbook choice.

    Returns
    -------
    Scalar loss in [0, 1]. Lower = better overlap. 0 when prediction
    and target are identical binary masks.
    """
    probs = torch.sigmoid(logits)
    # Flatten (B, ..., H, W) -> (B, -1) for per-sample Dice, then mean.
    probs_flat = probs.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    intersection = (probs_flat * target_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss for binary segmentation.

    Parameters
    ----------
    dice_weight
        Multiplier on the Dice term. Default 1.0 (equal contribution).
    bce_pos_weight
        Per-pixel weight applied to the positive class in BCE. Larger
        values further penalize false negatives. Default None (no
        per-class weighting; Dice carries the imbalance handling).
    smooth
        Dice smoothing constant, passed through.
    """

    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        bce_pos_weight: float | None = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        if bce_pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(float(bce_pos_weight)))
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> LossComponents:
        """Compute the combined loss and its components.

        Parameters
        ----------
        logits
            Model output, real-valued (pre-sigmoid). Shape (B, 1, H, W)
            or (B, H, W). Squeezed to match `target` if needed.
        target
            Binary ground-truth mask, shape (B, H, W) or (B, 1, H, W),
            float32 dtype, values in {0, 1}.

        Returns
        -------
        LossComponents with `total`, `bce`, and `dice` scalar tensors.
        """
        if torch.isnan(target).any():
            raise ValueError("target contains NaN — labels must be finite {0, 1}")

        # Align shapes: squeeze a trailing channel dim of size 1 if present.
        if logits.shape != target.shape:
            if logits.dim() == target.dim() + 1 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            elif target.dim() == logits.dim() + 1 and target.shape[1] == 1:
                target = target.squeeze(1)
            else:
                raise ValueError(
                    f"logits shape {tuple(logits.shape)} and target shape "
                    f"{tuple(target.shape)} are not broadcast-compatible"
                )

        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self.pos_weight,
        )
        dice = dice_loss(logits, target, smooth=self.smooth)
        total = bce + self.dice_weight * dice
        return LossComponents(total=total, bce=bce, dice=dice)
