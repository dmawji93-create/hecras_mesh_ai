"""Tests for buffered IoU/F1 metrics."""

from __future__ import annotations

import numpy as np
import pytest

from hecras_mesh_ai.postprocess import buffered_iou_f1


def test_identical_masks_score_perfect():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, :] = 1
    m = buffered_iou_f1(mask, mask, tolerance_pixels=0)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.iou == 1.0


def test_disjoint_masks_score_zero_at_zero_tolerance():
    pred = np.zeros((50, 50), dtype=np.uint8)
    pred[10:12, :] = 1
    truth = np.zeros((50, 50), dtype=np.uint8)
    truth[40:42, :] = 1  # far away
    m = buffered_iou_f1(pred, truth, tolerance_pixels=0)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.iou == 0.0


def test_buffer_tolerance_rescues_near_miss():
    """Pred is a 1-pixel line at row 10; truth at row 12. With tolerance=0
    they share nothing; with tolerance=3 both are fully covered."""
    pred = np.zeros((30, 30), dtype=np.uint8)
    pred[10, :] = 1
    truth = np.zeros((30, 30), dtype=np.uint8)
    truth[12, :] = 1

    strict = buffered_iou_f1(pred, truth, tolerance_pixels=0)
    loose = buffered_iou_f1(pred, truth, tolerance_pixels=3)

    assert strict.f1 == 0.0
    assert loose.precision == 1.0
    assert loose.recall == 1.0
    assert loose.f1 == 1.0


def test_no_predictions_with_truth_present():
    pred = np.zeros((20, 20), dtype=np.uint8)
    truth = np.zeros((20, 20), dtype=np.uint8)
    truth[10, :] = 1
    m = buffered_iou_f1(pred, truth, tolerance_pixels=3)
    assert m.precision == 0.0  # no preds -> nothing within tolerance for truth
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_predictions_with_no_truth():
    pred = np.zeros((20, 20), dtype=np.uint8)
    pred[5, :] = 1
    truth = np.zeros((20, 20), dtype=np.uint8)
    m = buffered_iou_f1(pred, truth, tolerance_pixels=3)
    # No truth -> precision is 0 (no truth to match), recall is undefined-but-1.
    # Convention: precision = 0/n_pred = 0.
    assert m.precision == 0.0


def test_both_masks_empty():
    pred = np.zeros((20, 20), dtype=np.uint8)
    truth = np.zeros((20, 20), dtype=np.uint8)
    m = buffered_iou_f1(pred, truth, tolerance_pixels=3)
    # Convention: empty == empty -> perfect overlap.
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.iou == 1.0


def test_partial_overlap_intermediate_metrics():
    """Pred covers all of truth + extra: precision should drop, recall stay 1."""
    pred = np.zeros((20, 20), dtype=np.uint8)
    pred[5:8, :] = 1  # 3 rows x 20 cols = 60 pixels
    truth = np.zeros((20, 20), dtype=np.uint8)
    truth[6, :] = 1  # 20 pixels
    m = buffered_iou_f1(pred, truth, tolerance_pixels=0)
    # Pred pixels within tolerance of truth: all rows 5/6/7 are within 1 of row 6,
    # but with tolerance=0 only exact overlaps count. So only row 6 pred pixels
    # (20 of 60) are within tolerance.
    assert m.precision == pytest.approx(20 / 60)
    assert m.recall == 1.0  # every truth pixel is matched by the row-6 pred


def test_buffered_metrics_dataclass_fields_consistent():
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[5, :] = 1
    truth = np.zeros((10, 10), dtype=np.uint8)
    truth[5, :] = 1
    m = buffered_iou_f1(pred, truth, tolerance_pixels=0)
    assert m.pred_pixels == 10
    assert m.truth_pixels == 10
    assert m.pred_within_tolerance == 10
    assert m.truth_within_tolerance == 10
    assert m.tolerance_pixels == 0.0


def test_invalid_input_raises():
    with pytest.raises(ValueError, match="shape"):
        buffered_iou_f1(np.zeros((5, 5)), np.zeros((10, 10)))
    with pytest.raises(ValueError, match="2D"):
        buffered_iou_f1(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError, match=">= 0"):
        buffered_iou_f1(np.zeros((5, 5)), np.zeros((5, 5)), tolerance_pixels=-1)
