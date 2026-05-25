"""Tests for the HEC-RAS 2D unsteady-results parser."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

from hecras_mesh_ai.harness import (
    list_2d_flow_areas,
    max_depth,
    max_face_velocity,
    max_water_surface,
)
from hecras_mesh_ai.harness.results import _resolve_area

# ---------------------------------------------------------------------------
# Synthetic-HDF tests for the parser's logic
# ---------------------------------------------------------------------------

_AREA = "TestArea"
_GEOM = f"/Geometry/2D Flow Areas/{_AREA}"
_RES = "/Results/Unsteady/Output/Output Blocks/Base Output/" f"Summary Output/2D Flow Areas/{_AREA}"
_AREA_ATTRS_DTYPE = np.dtype([("Name", "S16"), ("Mann", "<f4")])


def _make_results_hdf(
    tmp_path: Path,
    *,
    n_cells: int = 4,
    n_faces: int = 5,
    wse_values: tuple[float, ...] = (10.0, 11.0, 12.0, 13.0),
    z_min: tuple[float, ...] = (5.0, 6.0, 15.0, 7.0),  # cell 2 stays dry
    face_velocity: tuple[float, ...] = (0.1, 0.2, -0.3, 0.5, 0.7),
) -> Path:
    """Build a near-minimal results HDF with one 2D flow area."""
    p = tmp_path / "synthetic.p01.hdf"
    with h5py.File(p, "w") as f:
        # Area attributes table.
        attrs = np.zeros(1, dtype=_AREA_ATTRS_DTYPE)
        attrs[0] = (_AREA.encode("utf-8"), 0.035)
        f.create_dataset("/Geometry/2D Flow Areas/Attributes", data=attrs)
        # Cell coords.
        cell_xy = np.array([[100.0 + i, 200.0 + i] for i in range(n_cells)], dtype=np.float64)
        f.create_dataset(f"{_GEOM}/Cells Center Coordinate", data=cell_xy)
        f.create_dataset(
            f"{_GEOM}/Cells Minimum Elevation",
            data=np.array(z_min, dtype=np.float32),
        )
        # Face geometry: 2 facepoints per face, both at known XY so face
        # midpoints are predictable.
        fp_xy = np.array([[i * 10.0, i * 20.0] for i in range(n_faces + 1)], dtype=np.float64)
        f.create_dataset(f"{_GEOM}/FacePoints Coordinate", data=fp_xy)
        face_pts = np.array([[i, i + 1] for i in range(n_faces)], dtype=np.int32)
        f.create_dataset(f"{_GEOM}/Faces FacePoint Indexes", data=face_pts)
        # Max water surface: row 0 = WSE, row 1 = time.
        wse = np.zeros((2, n_cells), dtype=np.float32)
        wse[0] = wse_values
        wse[1] = np.arange(n_cells) * 0.1
        wse_ds = f.create_dataset(f"{_RES}/Maximum Water Surface", data=wse)
        wse_ds.attrs["Rows Variables"] = np.array([b"WSEL", b"Time"], dtype="|S16")
        wse_ds.attrs["Units per row"] = np.array([b"ft", b"days"], dtype="|S16")
        # Max face velocity.
        fv = np.zeros((2, n_faces), dtype=np.float32)
        fv[0] = face_velocity
        fv[1] = np.arange(n_faces) * 0.2
        fv_ds = f.create_dataset(f"{_RES}/Maximum Face Velocity", data=fv)
        fv_ds.attrs["Units per row"] = np.array([b"ft/s", b"days"], dtype="|S16")
    return p


def test_list_2d_flow_areas_returns_names(tmp_path):
    p = _make_results_hdf(tmp_path)
    names = list_2d_flow_areas(p)
    assert names == [_AREA]


def test_list_2d_flow_areas_empty_when_no_geometry(tmp_path):
    p = tmp_path / "empty.hdf"
    with h5py.File(p, "w") as f:
        f.attrs["File Type"] = b"x"
    assert list_2d_flow_areas(p) == []


def test_max_water_surface_loads_values_coords_and_attrs(tmp_path):
    p = _make_results_hdf(tmp_path, wse_values=(10.0, 11.0, 12.0, 13.0))
    da = max_water_surface(p)
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("cell",)
    assert da.shape == (4,)
    np.testing.assert_array_equal(da.values, [10.0, 11.0, 12.0, 13.0])
    np.testing.assert_array_equal(da.coords["x"].values, [100.0, 101.0, 102.0, 103.0])
    assert da.attrs["units"] == "ft"
    assert da.attrs["time_units"] == "days"
    assert da.attrs["area_name"] == _AREA


def test_max_depth_subtracts_min_elevation_and_clips_at_zero(tmp_path):
    # cell 2 has z_min=15 but max WSE=12 -> negative; should clip to 0.
    p = _make_results_hdf(
        tmp_path,
        wse_values=(10.0, 11.0, 12.0, 13.0),
        z_min=(5.0, 6.0, 15.0, 7.0),
    )
    da = max_depth(p)
    np.testing.assert_array_equal(da.values, [5.0, 5.0, 0.0, 6.0])
    assert da.name == "max_depth"


def test_max_face_velocity_uses_face_midpoints(tmp_path):
    p = _make_results_hdf(tmp_path, face_velocity=(0.1, 0.2, -0.3, 0.5, 0.7))
    da = max_face_velocity(p)
    assert da.dims == ("face",)
    np.testing.assert_allclose(da.values, [0.1, 0.2, -0.3, 0.5, 0.7], rtol=1e-6)
    # Face 0 midpoint = mean of facepoints 0 (0,0) and 1 (10,20) = (5,10).
    assert da.coords["x"].values[0] == pytest.approx(5.0)
    assert da.coords["y"].values[0] == pytest.approx(10.0)


def test_resolve_area_rejects_unknown_name(tmp_path):
    p = _make_results_hdf(tmp_path)
    with h5py.File(p, "r") as f, pytest.raises(ValueError, match="not found"):
        _resolve_area(f, "Bogus")


def test_max_water_surface_handles_explicit_area_name(tmp_path):
    p = _make_results_hdf(tmp_path)
    da = max_water_surface(p, area_name=_AREA)
    assert da.attrs["area_name"] == _AREA


# ---------------------------------------------------------------------------
# Integration test against the real Muncie p04.hdf produced by Task 3
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MUNCIE_P04 = _REPO_ROOT / "data" / "_verification" / "Muncie_writer_roundtrip" / "Muncie.p04.hdf"


@pytest.mark.skipif(
    not _MUNCIE_P04.exists(),
    reason="Run scripts/verify_writer_muncie.py and Task 3 integration test first",
)
def test_integration_muncie_p04_max_water_surface():
    da = max_water_surface(_MUNCIE_P04)
    # 5765 cells in Muncie's regenerated mesh, ~374 of which are inactive.
    assert da.shape[0] > 5000
    finite = da.values[np.isfinite(da.values)]
    assert finite.min() > 900  # Muncie WSE range ~945-960 ft.
    assert finite.max() < 1000
    # Inactive ghost cells should be masked as NaN.
    assert np.isnan(da.values).sum() > 100
    assert da.attrs["units"] == "ft"


@pytest.mark.skipif(
    not _MUNCIE_P04.exists(),
    reason="Run scripts/verify_writer_muncie.py and Task 3 integration test first",
)
def test_integration_muncie_p04_max_depth_is_nonnegative_on_active_cells():
    da = max_depth(_MUNCIE_P04)
    active = ~np.isnan(da.values)
    assert (da.values[active] >= 0).all()
    # At least some cells should be wet (depth > 0).
    assert (da.values[active] > 0).sum() > 100


@pytest.mark.skipif(
    not _MUNCIE_P04.exists(),
    reason="Run scripts/verify_writer_muncie.py and Task 3 integration test first",
)
def test_integration_muncie_p04_max_face_velocity():
    da = max_face_velocity(_MUNCIE_P04)
    # ~11k faces in Muncie's regenerated mesh.
    assert da.shape[0] > 10000
    assert da.attrs["units"] == "ft/s"
    # Realistic 2D-floodplain velocities: anywhere from ~0 to ~10 ft/s.
    assert np.abs(da.values).max() < 50
