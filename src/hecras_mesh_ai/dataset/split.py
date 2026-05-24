"""Spatial-holdout split + leakage checks for train/val tile datasets.

For our pilot, "train on Bald Eagle, val on Muncie" is the split — two
projects in different CRS (EPSG:2271 vs EPSG:2965), so spatial overlap is
trivially impossible. The leakage check is still useful: it documents the
guarantee, and it's the real machinery we need at Stage 3 when bulk-corpus
within-project splits arrive (same CRS, different tiles of the same
project).

Why the check matters at all: random train/val splitting on tiles cut from
a continuous landscape leaks information — val tiles overlap or border train
tiles by accident, and the model "memorizes neighborhoods" rather than
generalizing. The Stage 1 checkpoint requires "zero overlap" between train
and val tiles, so this check is the gate.
"""

from __future__ import annotations

from hecras_mesh_ai.dataset.tile_dataset import RasterTileDataset


def assert_no_spatial_overlap(
    train: RasterTileDataset,
    val: RasterTileDataset,
) -> None:
    """Raise ValueError if `train` and `val` bounding boxes overlap.

    Behavior:
      - Different CRS  =>  trivially no overlap. Returns silently.
      - Same CRS, disjoint bboxes  =>  returns silently.
      - Same CRS, touching at an edge  =>  returns silently (boundary
        touching is not overlap; zero-area intersection is allowed).
      - Same CRS, overlapping bboxes  =>  raises ValueError.

    For tile samplers, "non-overlapping bounding boxes" is the necessary
    condition for "no sampled tile from one set can equal a sampled tile
    from the other." It is sufficient when both samplers constrain tiles
    to lie entirely inside their respective dataset bounds — which our
    `RandomTileSampler` does.
    """
    if train.crs != val.crs:
        return  # different projected coordinate systems => disjoint Earth regions

    tl, tb, tr, tt = train.bounds
    vl, vb, vr, vt = val.bounds

    # Two boxes are disjoint iff one lies entirely to the left of, right
    # of, above, or below the other. Equivalently, they overlap iff none
    # of those conditions holds.
    x_disjoint = tr <= vl or vr <= tl
    y_disjoint = tt <= vb or vt <= tb
    if x_disjoint or y_disjoint:
        return

    raise ValueError(
        f"Spatial overlap between train ({train.bounds}) and val "
        f"({val.bounds}) datasets in CRS {train.crs}. "
        f"Random tile sampling from these would leak val pixels into the "
        f"train stream and vice versa."
    )
