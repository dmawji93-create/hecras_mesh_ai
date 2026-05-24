"""Tests for the BreaklineUNet LightningModule.

Two ML gotchas that shape these tests:

  1. BatchNorm with batch_size=1 at a 1x1 bottleneck. The U-Net has 5
     downsampling levels — 32x32 input becomes 1x1 at the bottom.
     BatchNorm in train mode refuses to compute over 1 value per channel.
     Tests use either 64x64 input (bottleneck = 2x2) OR batch_size >= 2.

  2. self.log() outside a Trainer warns. Direct calls to training_step()
     in tests have no Trainer attached; Lightning warns. Harmless;
     filtered at module level below.
"""

from __future__ import annotations

import pytest
import torch

from hecras_mesh_ai.model import BreaklineUNet

# Silence the "trying to self.log() but trainer reference not registered"
# warning during direct training_step / validation_step calls in tests.
pytestmark = pytest.mark.filterwarnings("ignore:.*trying to .self.log..*:UserWarning")


@pytest.fixture
def model_cpu():
    """A small CPU model with no ImageNet download (encoder_weights=None
    keeps tests offline and fast)."""
    return BreaklineUNet(
        in_channels=6,
        encoder_name="resnet18",
        encoder_weights=None,
        learning_rate=1e-3,
    )


def test_forward_returns_logits_of_expected_shape(model_cpu):
    """The U-Net needs H, W divisible by 32 (5 downsampling levels)."""
    x = torch.randn(2, 6, 64, 64)
    logits = model_cpu(x)
    assert logits.shape == (2, 1, 64, 64)
    assert logits.dtype == torch.float32


def test_forward_works_at_canonical_256_tile_size(model_cpu):
    """The standard tile size for Stage 2 training is 256x256."""
    x = torch.randn(1, 6, 256, 256)
    logits = model_cpu(x)
    assert logits.shape == (1, 1, 256, 256)


def test_logits_are_unbounded_real_values(model_cpu):
    """Output is logits (pre-sigmoid) — should occasionally exceed
    [0, 1]. If they're always in [0, 1] the sigmoid leaked in somewhere."""
    x = torch.randn(4, 6, 32, 32) * 10  # big inputs -> big outputs
    logits = model_cpu(x)
    assert (logits.abs() > 1.0).any(), "logits look clipped to [0, 1]"


def test_training_step_returns_scalar_loss(model_cpu):
    """A single training_step must return a scalar tensor that
    can be backpropped through."""
    features = torch.randn(2, 6, 32, 32)
    labels = torch.zeros(2, 32, 32)
    labels[:, 8:24, 8:24] = 1.0
    loss = model_cpu.training_step((features, labels), batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_validation_step_returns_scalar_loss(model_cpu):
    features = torch.randn(2, 6, 32, 32)
    labels = torch.zeros(2, 32, 32)
    labels[:, 8:24, 8:24] = 1.0
    loss = model_cpu.validation_step((features, labels), batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_backward_updates_parameters(model_cpu):
    """One step of SGD should change the model parameters. Catches the
    silent failure mode where the loss is computed but disconnected
    from the model (e.g. .detach() somewhere by accident)."""
    optimizer = torch.optim.SGD(model_cpu.parameters(), lr=0.1)
    features = torch.randn(2, 6, 32, 32)
    labels = torch.zeros(2, 32, 32)
    labels[:, 8:24, 8:24] = 1.0

    before = {n: p.detach().clone() for n, p in model_cpu.named_parameters() if p.requires_grad}

    optimizer.zero_grad()
    loss = model_cpu.training_step((features, labels), batch_idx=0)
    loss.backward()
    optimizer.step()

    changed = 0
    for n, p in model_cpu.named_parameters():
        if not p.requires_grad:
            continue
        if not torch.equal(p, before[n]):
            changed += 1
    # The vast majority of parameters should have updated. Allow a few
    # to be unchanged (e.g. batch-norm running stats are not learnable
    # via gradient).
    total = sum(1 for p in model_cpu.parameters() if p.requires_grad)
    assert changed >= 0.9 * total, f"only {changed}/{total} params updated"


def test_overfit_one_batch_loss_collapses(model_cpu):
    """The single most important sanity check: given a fixed (features,
    labels) batch, repeated training_steps must drive the loss
    monotonically downward. If the loss doesn't collapse on one batch,
    the gradient path is broken — investigate before any real training.

    Uses batch_size=2 + 64x64 input so BatchNorm at the U-Net bottleneck
    (2x2 spatial) sees enough values per channel. The 'fixed batch'
    discipline is preserved — we train on the same 2-tile batch every
    step."""
    torch.manual_seed(0)
    features = torch.randn(2, 6, 64, 64)
    labels = torch.zeros(2, 64, 64)
    labels[:, 24:40, 24:40] = 1.0

    optimizer = torch.optim.Adam(model_cpu.parameters(), lr=1e-2)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        loss = model_cpu.training_step((features, labels), batch_idx=0)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert (
        losses[-1] < losses[0] * 0.3
    ), f"loss didn't collapse: start={losses[0]:.4f}, end={losses[-1]:.4f}"


def test_configure_optimizers_returns_adam_at_specified_lr():
    model = BreaklineUNet(encoder_weights=None, learning_rate=7e-4)
    opt = model.configure_optimizers()
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["lr"] == pytest.approx(7e-4)


def test_hparams_are_saved():
    model = BreaklineUNet(
        in_channels=4,
        encoder_name="resnet18",
        encoder_weights=None,
        learning_rate=5e-4,
        dice_weight=2.0,
    )
    assert model.hparams.in_channels == 4
    assert model.hparams.encoder_weights is None
    assert model.hparams.learning_rate == pytest.approx(5e-4)
    assert model.hparams.dice_weight == 2.0


def test_in_channels_propagates_to_first_conv(model_cpu):
    """The U-Net's first conv must accept our 6-channel input. If the
    smp constructor's in_channels argument silently failed to take
    effect, the model would crash here.

    Uses eval() to skip BatchNorm so batch_size=1 is safe."""
    model_cpu.eval()
    x = torch.randn(1, 6, 64, 64)
    _ = model_cpu(x)  # should not raise
    with pytest.raises(RuntimeError):
        # Wrong channel count should fail at the first conv.
        model_cpu(torch.randn(1, 3, 64, 64))
