"""Buffer-based IoU and F1 for breakline evaluation.

Pixel-exact IoU/F1 are too strict for line-like targets: a 1-pixel
offset between predicted and true breakline yields IoU=0 even though
the model "found" the breakline. The standard fix is buffer-based:
a predicted pixel counts as a true positive if it falls within
`tolerance_pixels` of any ground-truth pixel, and vice versa.

This is the standard line-detection evaluation paradigm in road
extraction, vessel segmentation, and similar sparse-line problems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class BufferedMetrics:
    """Per-image buffered TP/FP/FN counts + derived precision/recall/F1/IoU."""

    pred_pixels: int  # total predicted positive pixels
    truth_pixels: int  # total ground-truth positive pixels
    pred_within_tolerance: int  # predictions within tolerance of any truth
    truth_within_tolerance: int  # truths within tolerance of any prediction
    tolerance_pixels: float

    @property
    def precision(self) -> float:
        """Fraction of predictions that fall within tolerance of truth."""
        if self.pred_pixels == 0:
            return 1.0 if self.truth_pixels == 0 else 0.0
        return self.pred_within_tolerance / self.pred_pixels

    @property
    def recall(self) -> float:
        """Fraction of truth covered within tolerance by any prediction."""
        if self.truth_pixels == 0:
            return 1.0 if self.pred_pixels == 0 else 0.0
        return self.truth_within_tolerance / self.truth_pixels

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

    @property
    def iou(self) -> float:
        """Buffered IoU: |pred∩buffer(truth)| / (pred + truth - intersection).

        Uses the *minimum* of pred_within_tolerance and truth_within_tolerance
        as the intersection — symmetric and bounded above by either.
        """
        intersection = min(self.pred_within_tolerance, self.truth_within_tolerance)
        union = self.pred_pixels + self.truth_pixels - intersection
        if union == 0:
            return 1.0
        return intersection / union


def buffered_iou_f1(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    tolerance_pixels: float = 3.0,
) -> BufferedMetrics:
    """Compute buffered IoU and F1 between a predicted binary mask and truth.

    Parameters
    ----------
    pred
        2D binary array, predicted breakline mask. Any nonzero value
        counts as positive.
    truth
        2D binary array, ground-truth breakline mask. Same shape as pred.
        Any nonzero value counts as positive.
    tolerance_pixels
        Buffer radius. A predicted pixel is "correct" if any truth pixel
        is within this many pixels (Euclidean distance). Likewise a
        truth pixel is "found" if any predicted pixel is within this
        distance. Default 3 — typical for breakline-band labels.

    Returns
    -------
    BufferedMetrics with the raw counts and derived precision / recall /
    F1 / IoU as properties.
    """
    if pred.shape != truth.shape:
        raise ValueError(f"pred shape {pred.shape} != truth shape {truth.shape}")
    if pred.ndim != 2:
        raise ValueError(f"pred and truth must be 2D, got shape {pred.shape}")
    if tolerance_pixels < 0:
        raise ValueError(f"tolerance_pixels must be >= 0, got {tolerance_pixels}")

    pred_bin = pred.astype(bool)
    truth_bin = truth.astype(bool)
    n_pred = int(pred_bin.sum())
    n_truth = int(truth_bin.sum())

    # Distance transform: at every pixel, the Euclidean distance to the
    # nearest True pixel of the inverse mask. We want distance to nearest
    # truth from each pred pixel, so compute the distance transform of
    # ~truth_bin (distances are zero inside truth, positive outside).
    if n_truth > 0:
        dist_to_truth = ndimage.distance_transform_edt(~truth_bin)
        pred_within = int(((dist_to_truth <= tolerance_pixels) & pred_bin).sum())
    else:
        pred_within = 0

    if n_pred > 0:
        dist_to_pred = ndimage.distance_transform_edt(~pred_bin)
        truth_within = int(((dist_to_pred <= tolerance_pixels) & truth_bin).sum())
    else:
        truth_within = 0

    return BufferedMetrics(
        pred_pixels=n_pred,
        truth_pixels=n_truth,
        pred_within_tolerance=pred_within,
        truth_within_tolerance=truth_within,
        tolerance_pixels=tolerance_pixels,
    )
