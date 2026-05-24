"""Tests for the BCE + Dice loss."""

from __future__ import annotations

import math

import pytest
import torch

from hecras_mesh_ai.model import BCEDiceLoss, LossComponents, dice_loss

# ---------------------------------------------------------------------------
# dice_loss
# ---------------------------------------------------------------------------


def test_dice_loss_zero_when_prediction_matches_target():
    """Perfect prediction (huge positive logits where target=1, huge
    negative where target=0) should drive Dice loss to ~0."""
    target = torch.zeros((2, 1, 8, 8))
    target[:, 0, 2:6, 2:6] = 1.0
    # Saturate the sigmoid: large +/- logits.
    logits = torch.where(target > 0.5, 50.0, -50.0)
    loss = dice_loss(logits, target)
    assert loss.item() < 1e-3


def test_dice_loss_high_when_prediction_is_opposite():
    """Opposite prediction should give Dice loss near 1."""
    target = torch.zeros((2, 1, 8, 8))
    target[:, 0, 2:6, 2:6] = 1.0
    logits = torch.where(target > 0.5, -50.0, 50.0)
    loss = dice_loss(logits, target)
    assert loss.item() > 0.98


def test_dice_loss_predict_zero_everywhere_is_much_worse_than_perfect():
    """The 'always predict 0' degenerate is what Dice penalizes. With
    smoothing the exact value depends on positive-pixel count and the
    smooth constant — what matters is that it's MUCH worse than a
    correct prediction. Exact value for this case:
        intersection = 0, union = 4 + 0 = 4, smooth = 1
        Dice score = (2*0 + 1) / (4 + 1) = 0.2 -> loss = 0.8
    """
    target = torch.zeros((1, 1, 8, 8))
    target[0, 0, 3:5, 3:5] = 1.0  # 4 positive pixels of 64

    perfect_logits = torch.where(target > 0.5, 50.0, -50.0)
    zero_logits = torch.full_like(target, -50.0)

    perfect_loss = dice_loss(perfect_logits, target).item()
    zero_loss = dice_loss(zero_logits, target).item()

    # Concrete spot-check of the formula above.
    assert zero_loss == pytest.approx(0.8, abs=1e-3)
    # And it's massively worse than the perfect case.
    assert zero_loss > perfect_loss + 0.5


def test_dice_loss_smoothing_avoids_zero_division():
    """All-zeros target + all-zeros prediction is a degenerate case that
    would 0/0 without smoothing. With smooth=1.0 it should be loss=0."""
    target = torch.zeros((1, 1, 4, 4))
    logits = torch.full_like(target, -50.0)
    loss = dice_loss(logits, target, smooth=1.0)
    # 2 * 0 + 1 / (0 + 0 + 1) = 1, so Dice = 1, loss = 0.
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# BCEDiceLoss
# ---------------------------------------------------------------------------


def test_bcedice_returns_three_scalar_components():
    """Sanity: outputs are scalars, components are individually
    accessible for separate logging."""
    target = torch.zeros((2, 1, 8, 8))
    target[:, 0, 2:6, 2:6] = 1.0
    logits = torch.randn_like(target)
    loss_fn = BCEDiceLoss()
    out = loss_fn(logits, target)
    assert isinstance(out, LossComponents)
    for t in (out.total, out.bce, out.dice):
        assert isinstance(t, torch.Tensor)
        assert t.dim() == 0


def test_bcedice_total_equals_bce_plus_weighted_dice():
    """The composition is what the docstring says."""
    target = torch.zeros((2, 1, 8, 8))
    target[:, 0, 2:6, 2:6] = 1.0
    logits = torch.randn_like(target)
    loss_fn = BCEDiceLoss(dice_weight=2.5)
    out = loss_fn(logits, target)
    expected_total = out.bce + 2.5 * out.dice
    torch.testing.assert_close(out.total, expected_total)


def test_bcedice_perfect_prediction_drives_total_to_near_zero():
    target = torch.zeros((2, 1, 8, 8))
    target[:, 0, 2:6, 2:6] = 1.0
    logits = torch.where(target > 0.5, 50.0, -50.0)
    loss_fn = BCEDiceLoss()
    out = loss_fn(logits, target)
    assert out.total.item() < 1e-3
    assert out.bce.item() < 1e-3
    assert out.dice.item() < 1e-3


def test_bcedice_handles_squeezed_logits():
    """The model may emit (B, H, W) instead of (B, 1, H, W). Both should
    pair with (B, H, W) targets without crashing."""
    target = torch.zeros((2, 8, 8))
    target[:, 2:6, 2:6] = 1.0
    logits = torch.where(target > 0.5, 50.0, -50.0)  # already (B, H, W)
    loss_fn = BCEDiceLoss()
    out = loss_fn(logits, target)
    assert out.total.item() < 1e-3


def test_bcedice_raises_on_nan_target():
    target = torch.zeros((2, 1, 4, 4))
    target[0, 0, 0, 0] = float("nan")
    logits = torch.randn_like(target)
    loss_fn = BCEDiceLoss()
    with pytest.raises(ValueError, match="NaN"):
        loss_fn(logits, target)


def test_bcedice_raises_on_incompatible_shapes():
    target = torch.zeros((2, 8, 8))
    logits = torch.zeros((2, 3, 8, 8))  # 3-channel logits — wrong
    loss_fn = BCEDiceLoss()
    with pytest.raises(ValueError, match="broadcast-compatible"):
        loss_fn(logits, target)


def test_bcedice_gradient_flows_through_total():
    """The loss must be differentiable end-to-end — a broken graph in
    Dice would silently zero out gradient updates on the rare class."""
    target = torch.zeros((1, 1, 4, 4))
    target[0, 0, 1:3, 1:3] = 1.0
    logits = torch.zeros_like(target, requires_grad=True)
    loss_fn = BCEDiceLoss()
    out = loss_fn(logits, target)
    out.total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert not (logits.grad == 0).all()


def test_bcedice_pos_weight_increases_bce_on_false_negatives():
    """When pos_weight > 1, missed positives should hurt more — the BCE
    component should be larger than the unweighted case for the same
    'predict all zeros' input."""
    target = torch.zeros((1, 1, 8, 8))
    target[0, 0, 2:6, 2:6] = 1.0
    logits = torch.full_like(target, -10.0)  # confidently predict 0 everywhere

    base = BCEDiceLoss().forward(logits, target)
    weighted = BCEDiceLoss(bce_pos_weight=5.0).forward(logits, target)
    assert weighted.bce.item() > base.bce.item() * 2  # 5x weight on positives


# ---------------------------------------------------------------------------
# Closed-form spot check — BCE
# ---------------------------------------------------------------------------


def test_bce_component_matches_closed_form_on_uniform_prediction():
    """Predict 0.5 everywhere; target = all 1s. BCE per pixel = -log(0.5)
    ≈ 0.6931. The combined loss should reflect this for the BCE part."""
    target = torch.ones((1, 1, 4, 4))
    logits = torch.zeros_like(target)  # sigmoid(0) = 0.5
    loss_fn = BCEDiceLoss()
    out = loss_fn(logits, target)
    assert out.bce.item() == pytest.approx(math.log(2), abs=1e-6)
