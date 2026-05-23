"""DEM conditioning — hybrid NaN handling before stencil-based feature derivation.

Real DEM nodata appears in two regimes:

  - Accidental small holes (corrupted pixels, interpolation artifacts at
    tile seams, isolated mis-classified water): worth patching so the
    downstream derivative stencils stay clean and the model gets the
    maximum usable area.
  - Intentional large voids (water bodies that were never measured,
    missing tiles, areas outside the survey footprint): NOT worth
    patching — synthesizing values where there is no measurement would
    inject fictitious training signal. These must stay NaN, and the
    Stage 1 Task 5 tile sampler is responsible for refusing tiles that
    overlap them.

The hybrid rule, configurable: patch a NaN connected component if its
size is at most `max_patch_size` pixels. Default 1 — only truly isolated
holes are filled. Larger regions pass through unchanged.

The fill rule for a patched pixel is the **mean of its valid 8-neighbors**,
computed from the *original* input (not in-place), so the patch order
cannot bias the result.

This module does NOT handle the natural stencil-edge NaN at the DEM border
— that is a separate concern handled by the feature stacker, which
reflect-pads the DEM by 1 pixel before derivation.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def patch_isolated_nan(
    z: np.ndarray,
    *,
    max_patch_size: int = 1,
    connectivity: int = 2,
) -> np.ndarray:
    """Fill small isolated NaN holes with the 8-neighbor mean.

    Parameters
    ----------
    z
        2D array. NaN regions are detected and selectively patched.
    max_patch_size
        Maximum connected-component size (in pixels) to fill. Default 1
        — only single isolated cells. Larger holes are preserved as NaN.
        Set to 0 to disable patching entirely.
    connectivity
        Connected-component rank passed to `scipy.ndimage.label`.
        1 = 4-connectivity (orthogonal neighbors only).
        2 = 8-connectivity (orthogonal + diagonal). Default 2 — the more
        conservative choice (two diagonally-adjacent NaN cells count as
        one 2-pixel region and stay NaN).

    Returns
    -------
    A new array with isolated NaN regions patched. The input is not modified.
    """
    if z.ndim != 2:
        raise ValueError(f"z must be 2D, got shape {z.shape}")
    if max_patch_size < 0:
        raise ValueError(f"max_patch_size must be >= 0, got {max_patch_size}")
    if connectivity not in (1, 2):
        raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")

    out = np.asarray(z, dtype=np.float64).copy()
    nan_mask = np.isnan(out)
    if not nan_mask.any() or max_patch_size == 0:
        return out

    structure = ndimage.generate_binary_structure(2, connectivity)
    labels, _ = ndimage.label(nan_mask, structure=structure)
    # bincount index 0 = background ("not NaN") count; indices >=1 are components.
    component_sizes = np.bincount(labels.ravel())

    patchable_label_ids = np.where(component_sizes <= max_patch_size)[0]
    patchable_label_ids = patchable_label_ids[patchable_label_ids > 0]
    if patchable_label_ids.size == 0:
        return out

    H, W = out.shape
    # Use the original z (not `out`) when reading neighbors, so patch order
    # cannot bias the mean for adjacent patches.
    z_original = np.asarray(z, dtype=np.float64)

    for lbl in patchable_label_ids:
        rows, cols = np.where(labels == lbl)
        for r, c in zip(rows, cols, strict=True):
            r0, r1 = max(r - 1, 0), min(r + 2, H)
            c0, c1 = max(c - 1, 0), min(c + 2, W)
            window = z_original[r0:r1, c0:c1]
            valid = window[~np.isnan(window)]
            if valid.size > 0:
                out[r, c] = float(valid.mean())
            # If somehow no valid neighbors (component size <= max_patch_size
            # but surrounded by other NaN — shouldn't happen for size=1 by
            # construction), leave as NaN.

    return out
