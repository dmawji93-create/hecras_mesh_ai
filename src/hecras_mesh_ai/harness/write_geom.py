"""Breakline-replacement writer for HEC-RAS geometry HDFs.

Replaces only `/Geometry/2D Flow Area Break Lines/*` in an existing valid
geometry HDF. All other groups (cells, faces, sub-grid bathymetry, etc.)
are copied through unchanged — HEC-RAS will recompute the mesh on its
next "Generate Mesh" trigger using the new breaklines.

Layout (verified against Muncie.g04.hdf, see docs/hdf-schema/README.md):

    /Geometry/2D Flow Area Break Lines/
    ├── Attributes        compound (N_features,) — Name S32, spacings, etc.
    ├── Polyline Info     (N_features, 4) int32 — point/part start+count
    ├── Polyline Parts    (N_parts,    2) int32 — point start+count
    │                       NOTE: "Point Starting Index" here is RELATIVE
    │                       to each feature's own point block, not global.
    └── Polyline Points   (N_points,   2) float64 — X, Y in geometry CRS

Single-part polylines only (the common case). For multi-part lines we'd
emit one Polyline Parts row per part with the offset relative to the
feature's point block.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

BREAKLINES_PATH = "/Geometry/2D Flow Area Break Lines"

ATTRIBUTES_DTYPE = np.dtype(
    [
        ("Name", "S32"),
        ("Cell Spacing Near", "<f4"),
        ("Cell Spacing Far", "<f4"),
        ("Near Repeats", "<i4"),
        ("Protection Radius", "u1"),
    ]
)

_POLYLINE_INFO_COLUMNS = np.array(
    [b"Point Starting Index", b"Point Count", b"Part Starting Index", b"Part Count"],
    dtype="|S20",
)
_POLYLINE_PARTS_COLUMNS = np.array([b"Point Starting Index", b"Point Count"], dtype="|S20")
_POLYLINE_POINTS_COLUMNS = np.array([b"X", b"Y"], dtype="|S1")


@dataclass
class Breakline:
    """One breakline feature, single-part.

    `points` is (N, 2) float64 in the geometry's CRS — same CRS as the
    target HDF's `/Geometry/2D Flow Areas/Polygon Points`. Caller is
    responsible for CRS alignment; this writer does no reprojection.

    Default attribute values (0, 0, 0, 0) mean "use HEC-RAS's mesh-
    generation defaults for this breakline" — matches Muncie's stored
    breaklines, which all have zeros.
    """

    name: str
    points: np.ndarray
    cell_spacing_near: float = 0.0
    cell_spacing_far: float = 0.0
    near_repeats: int = 0
    protection_radius: int = 0

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(
                f"Breakline {self.name!r}: points must be shape (N, 2); " f"got {self.points.shape}"
            )
        if self.points.shape[0] < 2:
            raise ValueError(
                f"Breakline {self.name!r}: need at least 2 points; " f"got {self.points.shape[0]}"
            )
        if len(self.name.encode("utf-8")) > 32:
            raise ValueError(
                f"Breakline name {self.name!r} encodes to >32 bytes; "
                f"HEC-RAS S32 field will truncate."
            )


@dataclass
class _PackedBreaklines:
    """Internal: the four HDF arrays, fully packed."""

    attributes: np.ndarray
    polyline_info: np.ndarray
    polyline_parts: np.ndarray
    polyline_points: np.ndarray
    n_features: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_features = int(self.attributes.shape[0])


def _pack(breaklines: Sequence[Breakline]) -> _PackedBreaklines:
    """Lay out the breaklines into the four arrays HEC-RAS expects."""
    n_features = len(breaklines)

    attributes = np.zeros(n_features, dtype=ATTRIBUTES_DTYPE)
    polyline_info = np.zeros((n_features, 4), dtype=np.int32)
    polyline_parts = np.zeros((n_features, 2), dtype=np.int32)

    point_blocks: list[np.ndarray] = []
    global_point_cursor = 0
    for i, bl in enumerate(breaklines):
        n_pts = bl.points.shape[0]
        attributes[i] = (
            bl.name.encode("utf-8"),
            bl.cell_spacing_near,
            bl.cell_spacing_far,
            bl.near_repeats,
            bl.protection_radius,
        )
        # Single-part: each feature owns exactly one part row, at part index i.
        polyline_info[i] = (global_point_cursor, n_pts, i, 1)
        # Part offset is RELATIVE to the feature's point block — always 0 for
        # single-part polylines.
        polyline_parts[i] = (0, n_pts)
        point_blocks.append(bl.points)
        global_point_cursor += n_pts

    if point_blocks:
        polyline_points = np.concatenate(point_blocks, axis=0).astype(np.float64, copy=False)
    else:
        polyline_points = np.zeros((0, 2), dtype=np.float64)

    return _PackedBreaklines(
        attributes=attributes,
        polyline_info=polyline_info,
        polyline_parts=polyline_parts,
        polyline_points=polyline_points,
    )


def _write_packed(group: h5py.Group, packed: _PackedBreaklines) -> None:
    """Create the four datasets inside `group` with HEC-RAS-matching attrs."""
    n = packed.n_features
    # gzip + chunked to match HEC-RAS's own files. h5py rejects chunks for
    # zero-length datasets — fall back to contiguous in that edge case.
    if n > 0:
        attr_ds = group.create_dataset(
            "Attributes",
            data=packed.attributes,
            chunks=(n,),
            compression="gzip",
        )
        info_ds = group.create_dataset(
            "Polyline Info",
            data=packed.polyline_info,
            chunks=(n, 4),
            compression="gzip",
        )
        parts_ds = group.create_dataset(
            "Polyline Parts",
            data=packed.polyline_parts,
            chunks=(n, 2),
            compression="gzip",
        )
        n_pts = packed.polyline_points.shape[0]
        points_ds = group.create_dataset(
            "Polyline Points",
            data=packed.polyline_points,
            chunks=(n_pts, 2),
            compression="gzip",
        )
    else:
        attr_ds = group.create_dataset("Attributes", data=packed.attributes)
        info_ds = group.create_dataset("Polyline Info", data=packed.polyline_info)
        parts_ds = group.create_dataset("Polyline Parts", data=packed.polyline_parts)
        points_ds = group.create_dataset("Polyline Points", data=packed.polyline_points)

    info_ds.attrs["Column"] = _POLYLINE_INFO_COLUMNS
    info_ds.attrs["Feature Type"] = np.bytes_(b"Polyline")
    info_ds.attrs["Row"] = np.bytes_(b"Feature")

    parts_ds.attrs["Column"] = _POLYLINE_PARTS_COLUMNS
    parts_ds.attrs["Row"] = np.bytes_(b"Part")

    points_ds.attrs["Column"] = _POLYLINE_POINTS_COLUMNS
    points_ds.attrs["Row"] = np.bytes_(b"Points")

    # Attributes dataset carries no attrs in HEC-RAS's own files.
    del attr_ds  # silence unused-var; we keep the assignment for symmetry


def replace_breaklines(
    source_hdf_path: Path | str,
    target_hdf_path: Path | str,
    breaklines: Sequence[Breakline],
    *,
    overwrite: bool = False,
) -> Path:
    """Copy `source` to `target`, swapping in a new set of breaklines.

    Everything else in the HDF (Polygon, Cells, Faces, sub-grid bathymetry,
    Manning's n, etc.) is copied through verbatim. HEC-RAS recomputes the
    mesh on its next "Generate Mesh" trigger.

    Parameters
    ----------
    source_hdf_path
        Existing valid HEC-RAS geometry HDF.
    target_hdf_path
        Destination path. Must not exist unless `overwrite=True`.
    breaklines
        New breaklines to write. Empty sequence removes the breaklines
        group entirely (matches projects that have no breaklines).
    overwrite
        If True, replace an existing target file.

    Returns
    -------
    Path to the written file (same as `target_hdf_path`, as Path).
    """
    source = Path(source_hdf_path)
    target = Path(target_hdf_path)
    if not source.exists():
        raise FileNotFoundError(f"source HDF does not exist: {source}")
    in_place = source.resolve() == target.resolve()
    if target.exists() and not overwrite and not in_place:
        raise FileExistsError(f"target HDF exists: {target} (pass overwrite=True to replace)")
    target.parent.mkdir(parents=True, exist_ok=True)

    # Skip the copy when source and target are the same file — patch in place.
    if not in_place:
        if target.exists():
            target.unlink()
        shutil.copy2(source, target)

    packed = _pack(breaklines)
    with h5py.File(target, "r+") as f:
        if BREAKLINES_PATH in f:
            del f[BREAKLINES_PATH]
        if packed.n_features > 0:
            group = f.create_group(BREAKLINES_PATH)
            _write_packed(group, packed)
        # else: leave the group deleted — matches "no breaklines" projects

    return target


def read_breaklines(hdf_path: Path | str) -> list[Breakline]:
    """Read existing breaklines from an HDF — used for round-trip tests.

    Returns an empty list if the HDF has no breaklines group.
    """
    hdf_path = Path(hdf_path)
    with h5py.File(hdf_path, "r") as f:
        if BREAKLINES_PATH not in f:
            return []
        g = f[BREAKLINES_PATH]
        attrs = g["Attributes"][:]
        info = g["Polyline Info"][:]
        parts = g["Polyline Parts"][:]
        points = g["Polyline Points"][:]

    out: list[Breakline] = []
    for i in range(attrs.shape[0]):
        point_start = int(info[i, 0])
        point_count = int(info[i, 1])
        part_start = int(info[i, 2])
        part_count = int(info[i, 3])
        if part_count != 1:
            # Multi-part: concat parts in order, using each part's offset
            # relative to the feature's point block.
            feature_points = points[point_start : point_start + point_count]
            parts_pts = []
            for p in range(part_start, part_start + part_count):
                p_off = int(parts[p, 0])
                p_cnt = int(parts[p, 1])
                parts_pts.append(feature_points[p_off : p_off + p_cnt])
            bl_points = np.concatenate(parts_pts, axis=0)
        else:
            bl_points = points[point_start : point_start + point_count]

        name = attrs[i]["Name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        out.append(
            Breakline(
                name=name,
                points=bl_points.copy(),
                cell_spacing_near=float(attrs[i]["Cell Spacing Near"]),
                cell_spacing_far=float(attrs[i]["Cell Spacing Far"]),
                near_repeats=int(attrs[i]["Near Repeats"]),
                protection_radius=int(attrs[i]["Protection Radius"]),
            )
        )
    return out
