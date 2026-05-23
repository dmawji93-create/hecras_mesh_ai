"""Tests for DEM conditioning — the isolated-NaN patcher."""

from __future__ import annotations

import numpy as np
import pytest

from hecras_mesh_ai.features.conditioning import patch_isolated_nan


def test_no_nan_input_returns_copy_unchanged():
    z = np.arange(25, dtype=np.float64).reshape(5, 5)
    out = patch_isolated_nan(z)
    np.testing.assert_array_equal(out, z)
    assert out is not z, "must not mutate the input"


def test_single_isolated_nan_is_patched_with_neighbor_mean():
    z = np.ones((5, 5))
    z[2, 2] = np.nan
    out = patch_isolated_nan(z)
    # All 8 neighbors equal 1, so the patch value should be 1.
    np.testing.assert_allclose(out[2, 2], 1.0)
    # Input unchanged.
    assert np.isnan(z[2, 2])


def test_isolated_nan_with_varied_neighbors_uses_mean():
    z = np.full((5, 5), np.nan)
    # Patch will be at (2, 2); surround it by known values.
    z[1, 1:4] = 10.0
    z[2, 1] = 20.0
    z[2, 3] = 30.0
    z[3, 1:4] = 40.0
    # The 8-neighbor mean = (10+10+10 + 20 + 30 + 40+40+40) / 8 = 200/8 = 25
    out = patch_isolated_nan(z)
    np.testing.assert_allclose(out[2, 2], 25.0)


def test_two_adjacent_nan_are_preserved_with_default_max_patch_size():
    """8-connectivity: two adjacent NaN form a size-2 component, not patched."""
    z = np.ones((5, 5))
    z[2, 2] = np.nan
    z[2, 3] = np.nan
    out = patch_isolated_nan(z)
    assert np.isnan(out[2, 2])
    assert np.isnan(out[2, 3])


def test_diagonal_nan_are_preserved_under_8_connectivity():
    """Two diagonal NaN -> one 2-cell component under 8-conn, preserved."""
    z = np.ones((5, 5))
    z[2, 2] = np.nan
    z[3, 3] = np.nan
    out = patch_isolated_nan(z, connectivity=2)
    assert np.isnan(out[2, 2])
    assert np.isnan(out[3, 3])


def test_diagonal_nan_are_patched_under_4_connectivity():
    """Two diagonal NaN -> two 1-cell components under 4-conn, both patched."""
    z = np.ones((5, 5))
    z[2, 2] = np.nan
    z[3, 3] = np.nan
    out = patch_isolated_nan(z, connectivity=1)
    assert not np.isnan(out[2, 2])
    assert not np.isnan(out[3, 3])


def test_3x3_nan_block_is_preserved():
    """A 9-cell NaN block is well above max_patch_size=1; must not be filled."""
    z = np.ones((7, 7))
    z[2:5, 2:5] = np.nan
    out = patch_isolated_nan(z)
    assert np.all(np.isnan(out[2:5, 2:5]))


def test_many_isolated_nan_each_patched_independently():
    z = np.ones((7, 7))
    positions = [(1, 1), (1, 5), (3, 3), (5, 1), (5, 5)]
    for r, c in positions:
        z[r, c] = np.nan
    out = patch_isolated_nan(z)
    for r, c in positions:
        assert not np.isnan(out[r, c]), f"isolated NaN at ({r},{c}) was not patched"
        np.testing.assert_allclose(out[r, c], 1.0)


def test_max_patch_size_zero_disables_patching():
    z = np.ones((5, 5))
    z[2, 2] = np.nan
    out = patch_isolated_nan(z, max_patch_size=0)
    assert np.isnan(out[2, 2])


def test_max_patch_size_two_patches_pair():
    """With max_patch_size=2, an adjacent pair counts as patchable."""
    z = np.ones((5, 5))
    z[2, 2] = np.nan
    z[2, 3] = np.nan
    out = patch_isolated_nan(z, max_patch_size=2)
    assert not np.isnan(out[2, 2])
    assert not np.isnan(out[2, 3])


def test_patch_at_corner_uses_available_neighbors():
    """A NaN at the corner has only 3 valid 8-neighbors. The patch should
    still succeed using those."""
    z = np.ones((5, 5))
    z[0, 0] = np.nan
    out = patch_isolated_nan(z)
    np.testing.assert_allclose(out[0, 0], 1.0)


def test_patch_uses_original_values_not_sequential_fills():
    """The mean for each patch is computed from the ORIGINAL z (not the
    already-being-mutated `out`), so the patch order cannot bias the result."""
    # Two isolated NaN with distinct neighborhoods, far enough apart that
    # they cannot influence each other.
    z = np.ones((9, 9))
    z[2, 2] = np.nan
    z[6, 6] = np.nan
    out = patch_isolated_nan(z)
    np.testing.assert_allclose(out[2, 2], 1.0)
    np.testing.assert_allclose(out[6, 6], 1.0)


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        patch_isolated_nan(np.zeros(5))
    with pytest.raises(ValueError):
        patch_isolated_nan(np.zeros((5, 5)), max_patch_size=-1)
    with pytest.raises(ValueError):
        patch_isolated_nan(np.zeros((5, 5)), connectivity=3)
