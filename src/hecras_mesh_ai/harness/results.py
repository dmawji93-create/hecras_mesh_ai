"""HEC-RAS 2D unsteady-results parser.

Reads per-cell and per-face maximum values from a plan results HDF
(`<project>.pNN.hdf`) and returns them as xarray DataArrays with the
cell-center / face-center coordinates aligned, so downstream code can
overlay results onto the same geospatial coordinate system the model
already uses.

Minimal scope (per Stage 4 Task 4 ADR): max water surface, max depth,
max face velocity. The 2D-flow-area cell counts in the results file
can differ from the geometry's input cell count because HEC-RAS
inserts cells around breaklines during mesh regeneration; we read
counts from the results file itself.

Verified against Muncie.p04.hdf, HEC-RAS 7.0. Schema paths:
    /Geometry/2D Flow Areas/<area>/Cells Center Coordinate
    /Geometry/2D Flow Areas/<area>/Cells Minimum Elevation
    /Geometry/2D Flow Areas/<area>/Faces FacePoint Indexes
    /Geometry/2D Flow Areas/<area>/FacePoints Coordinate
    /Results/Unsteady/Output/Output Blocks/Base Output/
        Summary Output/2D Flow Areas/<area>/Maximum Water Surface
        Summary Output/2D Flow Areas/<area>/Maximum Face Velocity

Maximum datasets are shape (2, N): row 0 = value, row 1 = time of max
(in days). Units live in attrs `Rows Variables` + `Units` / `Units per
row`.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import xarray as xr

_GEOMETRY_2D = "/Geometry/2D Flow Areas"
_RESULTS_2D = "/Results/Unsteady/Output/Output Blocks/Base Output/" "Summary Output/2D Flow Areas"


def list_2d_flow_areas(hdf_path: Path | str) -> list[str]:
    """Return the names of 2D flow areas defined in the geometry."""
    hdf_path = Path(hdf_path)
    with h5py.File(hdf_path, "r") as f:
        geom = f.get(_GEOMETRY_2D)
        if geom is None:
            return []
        # `Attributes` carries the area names in its 'Name' field.
        attrs = geom.get("Attributes")
        if attrs is None:
            return []
        names = []
        for row in attrs[:]:
            name = row["Name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8").rstrip("\x00 ")
            names.append(str(name))
        return names


def _resolve_area(f: h5py.File, area_name: str | None) -> str:
    """Resolve `area_name` to a concrete 2D-area subgroup name in the HDF.

    When `area_name` is None, returns the unique area or raises.
    """
    geom = f[_GEOMETRY_2D]
    candidates = [k for k in geom if isinstance(geom[k], h5py.Group)]
    if area_name is not None:
        if area_name not in candidates:
            raise ValueError(f"area {area_name!r} not found; available: {candidates!r}")
        return area_name
    if len(candidates) == 0:
        raise ValueError("no 2D flow areas in HDF")
    if len(candidates) > 1:
        raise ValueError(f"multiple 2D flow areas ({candidates!r}); pass area_name explicitly")
    return candidates[0]


def _read_max_dataset(
    f: h5py.File, area: str, dataset_name: str
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Read a (2, N) Maximum* dataset into (values, times, value_units, time_units)."""
    path = f"{_RESULTS_2D}/{area}/{dataset_name}"
    ds = f[path]
    arr = ds[:]
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(f"{path} shape {arr.shape} not (2, N); schema assumption broken")
    values = arr[0]
    times = arr[1]
    units_attr = ds.attrs.get("Units per row")
    if units_attr is None:
        units_attr = ds.attrs.get("Units")
    if isinstance(units_attr, np.ndarray):
        value_units = units_attr[0].decode("utf-8") if len(units_attr) else ""
        time_units = units_attr[1].decode("utf-8") if len(units_attr) > 1 else "days"
    else:
        value_units = (
            units_attr.decode("utf-8") if isinstance(units_attr, bytes) else str(units_attr or "")
        )
        time_units = "days"
    return values, times, value_units, time_units


def _cell_xy(f: h5py.File, area: str) -> np.ndarray:
    return f[f"{_GEOMETRY_2D}/{area}/Cells Center Coordinate"][:]


def _cell_min_elevation(f: h5py.File, area: str) -> np.ndarray:
    return f[f"{_GEOMETRY_2D}/{area}/Cells Minimum Elevation"][:]


def _face_midpoint_xy(f: h5py.File, area: str) -> np.ndarray:
    """Face midpoint = mean of the two facepoint XY coords."""
    fp_idx = f[f"{_GEOMETRY_2D}/{area}/Faces FacePoint Indexes"][:]
    fp_xy = f[f"{_GEOMETRY_2D}/{area}/FacePoints Coordinate"][:]
    return (fp_xy[fp_idx[:, 0]] + fp_xy[fp_idx[:, 1]]) / 2.0


def _inactive_cell_mask(z_min: np.ndarray) -> np.ndarray:
    """HEC-RAS appends "ghost" cells past the active mesh whose
    Cells Minimum Elevation is NaN and whose WSE results are 0. We
    treat z_min == NaN as the canonical inactive-cell signal."""
    return np.isnan(z_min)


def max_water_surface(hdf_path: Path | str, area_name: str | None = None) -> xr.DataArray:
    """Per-cell max water-surface elevation across the simulation.

    The DataArray has dim `cell` of size N_cells, with 1-D coords
    `x` and `y` (cell-center coordinates) and a `time_of_max` data
    variable in days. Units in attrs.

    Inactive ghost cells (those whose `Cells Minimum Elevation` is
    NaN) are masked to NaN in both the value and the time_of_max.
    """
    hdf_path = Path(hdf_path)
    with h5py.File(hdf_path, "r") as f:
        area = _resolve_area(f, area_name)
        values, times, units, time_units = _read_max_dataset(f, area, "Maximum Water Surface")
        xy = _cell_xy(f, area)
        z_min = _cell_min_elevation(f, area)
    inactive = _inactive_cell_mask(z_min)
    values = np.where(inactive, np.nan, values)
    times = np.where(inactive, np.nan, times)
    return xr.DataArray(
        values,
        dims=("cell",),
        coords={
            "x": ("cell", xy[:, 0]),
            "y": ("cell", xy[:, 1]),
            "time_of_max": ("cell", times),
        },
        attrs={
            "units": units,
            "time_units": time_units,
            "area_name": area,
            "long_name": "maximum water surface elevation",
            "source_hdf": str(hdf_path),
        },
        name="max_wse",
    )


def max_depth(hdf_path: Path | str, area_name: str | None = None) -> xr.DataArray:
    """Per-cell max depth = max_WSE - cell_min_elevation.

    Returns a DataArray on the same cell grid as `max_water_surface`,
    clipped at 0 (dry cells where WSE < min elevation). Inactive
    ghost cells (z_min == NaN) are returned as NaN.
    """
    hdf_path = Path(hdf_path)
    with h5py.File(hdf_path, "r") as f:
        area = _resolve_area(f, area_name)
        values, times, units, time_units = _read_max_dataset(f, area, "Maximum Water Surface")
        xy = _cell_xy(f, area)
        z_min = _cell_min_elevation(f, area)
    inactive = _inactive_cell_mask(z_min)
    # Compute depth on active cells only; leave inactive as NaN.
    with np.errstate(invalid="ignore"):
        depth = np.where(inactive, np.nan, np.maximum(values - z_min, 0.0))
    times = np.where(inactive, np.nan, times)
    return xr.DataArray(
        depth,
        dims=("cell",),
        coords={
            "x": ("cell", xy[:, 0]),
            "y": ("cell", xy[:, 1]),
            "time_of_max": ("cell", times),
        },
        attrs={
            "units": units,
            "time_units": time_units,
            "area_name": area,
            "long_name": "maximum depth (max_wse - cell_min_elevation, clipped at 0)",
            "source_hdf": str(hdf_path),
        },
        name="max_depth",
    )


def max_face_velocity(hdf_path: Path | str, area_name: str | None = None) -> xr.DataArray:
    """Per-face max velocity magnitude across the simulation.

    Stored on the face grid (N_faces, not N_cells). Velocity is signed
    in the dataset; the "maximum" here is HEC-RAS's max-absolute
    convention (the dataset shipped contains signed values whose max
    abs over time was recorded). Caller should `abs()` if magnitude
    matters.
    """
    hdf_path = Path(hdf_path)
    with h5py.File(hdf_path, "r") as f:
        area = _resolve_area(f, area_name)
        values, times, units, time_units = _read_max_dataset(f, area, "Maximum Face Velocity")
        xy = _face_midpoint_xy(f, area)
    return xr.DataArray(
        values,
        dims=("face",),
        coords={
            "x": ("face", xy[:, 0]),
            "y": ("face", xy[:, 1]),
            "time_of_max": ("face", times),
        },
        attrs={
            "units": units,
            "time_units": time_units,
            "area_name": area,
            "long_name": "maximum face velocity (signed, HEC-RAS convention)",
            "source_hdf": str(hdf_path),
        },
        name="max_face_velocity",
    )
