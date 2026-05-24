"""Sliding-window inference at full project extent.

The model trains on 256x256 tiles. At deployment we want a probability map
over the whole project (10s of millions of pixels). One forward pass is
infeasible; we tile, predict each tile, and stitch the outputs back with
overlap-averaging to reduce boundary artifacts.

Pipeline:
  feature stack (C, H, W)
    -> tile into overlapping windows
    -> batch through model (GPU)
    -> accumulate predicted logits + count per pixel
    -> divide -> mean logits per pixel
    -> sigmoid -> probabilities
"""

from __future__ import annotations

import numpy as np
import torch


def sliding_window_predict(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    tile_size: int = 256,
    overlap: int = 32,
    batch_size: int = 8,
    device: str = "cuda",
    nan_fill_value: float = 0.0,
) -> np.ndarray:
    """Predict a full-extent probability map by sliding-window tiling.

    Parameters
    ----------
    model
        A PyTorch model that maps (B, C, H, W) features -> (B, 1, H, W) or
        (B, H, W) logits. Will be put in eval mode for inference.
    features
        Full-project feature stack, shape (C, H, W), float32. May contain
        NaN — replaced with `nan_fill_value` before inference. Output
        probabilities at originally-NaN positions are forced back to NaN.
    tile_size
        Side of the square inference window. Must be divisible by 32
        for the standard 5-level U-Net.
    overlap
        Pixel overlap between adjacent tiles. Reduces boundary artifacts
        via overlap-averaging.
    batch_size
        How many tiles to forward at once. Limited by GPU memory.
    device
        Inference device. "cuda" by default; falls back to "cpu" if CUDA
        is unavailable.
    nan_fill_value
        Value used to replace NaN in features before forward pass. The
        output probability map preserves the original NaN positions.

    Returns
    -------
    probs : np.ndarray, shape (H, W), float32 in [0, 1] (or NaN where
            input was NaN).
    """
    if features.ndim != 3:
        raise ValueError(f"features must be (C, H, W), got shape {features.shape}")
    if tile_size <= 0 or tile_size % 32 != 0:
        raise ValueError(f"tile_size must be a positive multiple of 32, got {tile_size}")
    if not 0 <= overlap < tile_size:
        raise ValueError(f"overlap must be in [0, tile_size), got {overlap}")

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = model.to(device).eval()

    C, H, W = features.shape
    nan_mask = np.isnan(features).any(axis=0)  # (H, W) — True wherever any band is NaN
    feat_filled = np.where(nan_mask[None, :, :], nan_fill_value, features).astype(np.float32)

    stride = tile_size - overlap
    # Pad H, W so the input is at least tile_size and the last tile fits flush.
    pad_h = max(tile_size - H, 0)
    pad_w = max(tile_size - W, 0)
    if H + pad_h > tile_size:
        pad_h += (stride - (H + pad_h - tile_size) % stride) % stride
    if W + pad_w > tile_size:
        pad_w += (stride - (W + pad_w - tile_size) % stride) % stride
    feat_padded = np.pad(feat_filled, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    Hp, Wp = feat_padded.shape[1], feat_padded.shape[2]

    # Build the list of (row, col) origins. With the padding step above,
    # Hp >= tile_size and Wp >= tile_size are guaranteed even for small input.
    rows = list(range(0, max(Hp - tile_size + 1, 1), stride))
    cols = list(range(0, max(Wp - tile_size + 1, 1), stride))
    if not rows:
        rows = [0]
    if not cols:
        cols = [0]
    if rows[-1] + tile_size < Hp:
        rows.append(Hp - tile_size)
    if cols[-1] + tile_size < Wp:
        cols.append(Wp - tile_size)
    origins = [(r, c) for r in rows for c in cols]

    # Accumulators in padded space; we'll crop back at the end.
    accum = np.zeros((Hp, Wp), dtype=np.float32)
    count = np.zeros((Hp, Wp), dtype=np.float32)

    with torch.no_grad():
        for batch_start in range(0, len(origins), batch_size):
            batch_origins = origins[batch_start : batch_start + batch_size]
            batch_tiles = np.stack(
                [feat_padded[:, r : r + tile_size, c : c + tile_size] for (r, c) in batch_origins],
                axis=0,
            )
            x = torch.from_numpy(batch_tiles).to(device)
            logits = model(x)
            if logits.dim() == 4:
                logits = logits.squeeze(1)  # (B, H, W)
            probs = torch.sigmoid(logits).cpu().numpy()

            for (r, c), p in zip(batch_origins, probs, strict=True):
                accum[r : r + tile_size, c : c + tile_size] += p
                count[r : r + tile_size, c : c + tile_size] += 1.0

    mean_prob_padded = accum / np.maximum(count, 1.0)
    mean_prob = mean_prob_padded[:H, :W]

    # Restore NaN at originally-NaN positions.
    mean_prob = np.where(nan_mask, np.nan, mean_prob).astype(np.float32)
    return mean_prob
