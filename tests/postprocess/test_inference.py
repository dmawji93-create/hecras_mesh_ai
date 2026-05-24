"""Tests for sliding-window inference."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from hecras_mesh_ai.postprocess import sliding_window_predict


class _ConstantModel(nn.Module):
    """Returns a constant logit value per pixel — useful for testing the
    stitching / overlap-averaging logic without any actual learning."""

    def __init__(self, value: float = 2.0):
        super().__init__()
        # A parameter so model.to(device) and eval() work normally.
        self.value = nn.Parameter(torch.tensor(value), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        return self.value.expand(B, 1, H, W)


def test_output_shape_matches_input_HW():
    model = _ConstantModel(value=10.0)  # sigmoid(10) ~ 1.0
    features = np.random.default_rng(0).standard_normal((6, 200, 300)).astype(np.float32)
    probs = sliding_window_predict(
        model, features, tile_size=64, overlap=16, batch_size=4, device="cpu"
    )
    assert probs.shape == (200, 300)
    assert probs.dtype == np.float32


def test_constant_model_yields_constant_probability_everywhere():
    """A model that outputs constant logits should give a uniform probability
    map. Tests that overlap-averaging doesn't introduce artifacts."""
    model = _ConstantModel(value=2.0)
    expected_prob = float(torch.sigmoid(torch.tensor(2.0)))
    features = np.zeros((6, 128, 192), dtype=np.float32)
    probs = sliding_window_predict(
        model, features, tile_size=64, overlap=16, batch_size=4, device="cpu"
    )
    np.testing.assert_allclose(probs, expected_prob, atol=1e-5)


def test_nan_input_preserved_in_output():
    """Pixels that were NaN in the input should be NaN in the output —
    the model output is meaningless there even if it ran."""
    model = _ConstantModel(value=2.0)
    features = np.zeros((6, 100, 100), dtype=np.float32)
    features[:, 20:30, 40:50] = np.nan
    probs = sliding_window_predict(
        model, features, tile_size=64, overlap=16, batch_size=2, device="cpu"
    )
    # Originally-NaN block must be NaN in output.
    assert np.all(np.isnan(probs[20:30, 40:50]))
    # Outside must NOT be NaN.
    assert not np.isnan(probs[0, 0])
    assert not np.isnan(probs[-1, -1])


def test_handles_input_exactly_one_tile():
    """When H == W == tile_size, no padding or stitching — degenerate case."""
    model = _ConstantModel(value=0.0)  # sigmoid(0) = 0.5
    features = np.zeros((6, 64, 64), dtype=np.float32)
    probs = sliding_window_predict(
        model, features, tile_size=64, overlap=16, batch_size=2, device="cpu"
    )
    assert probs.shape == (64, 64)
    np.testing.assert_allclose(probs, 0.5, atol=1e-5)


def test_handles_input_smaller_than_tile():
    """Pads up so a single tile covers the input. Output cropped back."""
    model = _ConstantModel(value=0.0)
    features = np.zeros((6, 40, 50), dtype=np.float32)
    probs = sliding_window_predict(
        model, features, tile_size=64, overlap=16, batch_size=1, device="cpu"
    )
    assert probs.shape == (40, 50)


def test_squeezed_model_output_handled():
    """Model that returns (B, H, W) instead of (B, 1, H, W) should still work."""

    class _SqueezedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.v = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        def forward(self, x):
            B, _, H, W = x.shape
            return self.v.expand(B, H, W)

    model = _SqueezedModel()
    features = np.zeros((6, 96, 96), dtype=np.float32)
    probs = sliding_window_predict(
        model, features, tile_size=64, overlap=16, batch_size=2, device="cpu"
    )
    assert probs.shape == (96, 96)
    np.testing.assert_allclose(probs, float(torch.sigmoid(torch.tensor(1.0))), atol=1e-5)


def test_invalid_input_raises():
    model = _ConstantModel()
    with pytest.raises(ValueError, match="C, H, W"):
        sliding_window_predict(model, np.zeros((100, 100)), device="cpu")
    with pytest.raises(ValueError, match="multiple of 32"):
        sliding_window_predict(
            model, np.zeros((6, 100, 100), dtype=np.float32), tile_size=50, device="cpu"
        )
    with pytest.raises(ValueError, match="overlap"):
        sliding_window_predict(
            model,
            np.zeros((6, 100, 100), dtype=np.float32),
            tile_size=64,
            overlap=64,
            device="cpu",
        )
