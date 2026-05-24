"""Tests for the pilot-project feature + label cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from hecras_mesh_ai.dataset import cache_pilot_project
from hecras_mesh_ai.features import FEATURE_CHANNELS


def _write_synthetic_dem(path: Path, z: np.ndarray, *, crs="EPSG:32633", nodata=-9999.0) -> None:
    h, w = z.shape
    transform = from_origin(0, h, 1, 1)
    payload = np.where(np.isnan(z), nodata, z).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(payload, 1)


# ---------------------------------------------------------------------------
# Synthetic project tests (no rashdf — the cache function needs an HDF).
# ---------------------------------------------------------------------------
# For the unit tests we use the real pilot HDFs even on the synthetic DEM
# case isn't feasible. So we keep these tests as integration tests against
# the real pilots, and rely on the underlying components (stack_dem_features,
# rasterize_breaklines) having their own synthetic-data tests.


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PILOT_BASE = (
    _REPO_ROOT
    / "data"
    / "raw"
    / "usace"
    / "RAS Samples"
    / "Example_Projects_7_0"
    / "2D Unsteady Flow Hydraulics"
)
_MUNCIE_HDF = _PILOT_BASE / "Muncie" / "Muncie.g04.hdf"
_BALDEAGLE_HDF = _PILOT_BASE / "BaldEagleCrkMulti2D" / "BaldEagleDamBrk.g09.hdf"
_BALDEAGLE_DEM = _PILOT_BASE / "BaldEagleCrkMulti2D" / "Terrain" / "Terrain50.baldeagledem.tif"


def _find_muncie_dem() -> Path | None:
    base = _PILOT_BASE / "Muncie" / "Terrain"
    if not base.exists():
        return None
    tifs = list(base.glob("*.tif"))
    return tifs[0] if tifs else None


@pytest.mark.skipif(
    not _MUNCIE_HDF.exists() or _find_muncie_dem() is None,
    reason="Muncie pilot data not present",
)
def test_cache_muncie_pilot_writes_aligned_geotiffs(tmp_path):
    dem = _find_muncie_dem()
    paths = cache_pilot_project(
        project_name="muncie",
        dem_path=dem,
        geometry_hdf_path=_MUNCIE_HDF,
        buffer_width=20.0,
        out_dir=tmp_path,
    )

    # Both files exist.
    assert paths.features.exists()
    assert paths.labels.exists()
    assert paths.project_name == "muncie"
    assert paths.feature_channels == FEATURE_CHANNELS

    # Features: 6 bands, float32, CRS + transform from source DEM.
    with rasterio.open(paths.features) as f:
        assert f.count == 6
        assert f.dtypes[0] == "float32"
        assert f.crs.to_epsg() == 2965  # Muncie terrain CRS
        feat_shape = (f.height, f.width)
        feat_transform = f.transform
        feat_crs = f.crs

    # Labels: 1 band, uint8, values in {0, 1}, same shape/CRS/transform.
    with rasterio.open(paths.labels) as la:
        assert la.count == 1
        assert la.dtypes[0] == "uint8"
        assert (la.height, la.width) == feat_shape
        assert la.transform == feat_transform
        assert la.crs == feat_crs
        label_arr = la.read(1)
        unique = set(np.unique(label_arr).tolist())
        assert unique <= {0, 1}, f"labels not strictly binary: {unique}"
        assert label_arr.sum() > 0, "Muncie's 2 breaklines should yield positives"


@pytest.mark.skipif(
    not _BALDEAGLE_HDF.exists() or not _BALDEAGLE_DEM.exists(),
    reason="Bald Eagle pilot data not present",
)
def test_cache_baldeagle_pilot_writes_aligned_geotiffs(tmp_path):
    paths = cache_pilot_project(
        project_name="bald_eagle_g09",
        dem_path=_BALDEAGLE_DEM,
        geometry_hdf_path=_BALDEAGLE_HDF,
        buffer_width=50.0,
        out_dir=tmp_path,
    )

    assert paths.features.exists()
    assert paths.labels.exists()
    assert paths.feature_channels == FEATURE_CHANNELS

    with rasterio.open(paths.features) as f:
        assert f.count == 6
        assert f.dtypes[0] == "float32"
        assert f.crs.to_epsg() == 2271  # Bald Eagle terrain CRS
        feat_shape = (f.height, f.width)
        feat_transform = f.transform
        feat_crs = f.crs

    with rasterio.open(paths.labels) as la:
        assert la.count == 1
        assert la.dtypes[0] == "uint8"
        assert (la.height, la.width) == feat_shape
        assert la.transform == feat_transform
        assert la.crs == feat_crs
        label_arr = la.read(1)
        unique = set(np.unique(label_arr).tolist())
        assert unique <= {0, 1}
        assert label_arr.sum() > 0


@pytest.mark.skipif(
    not _MUNCIE_HDF.exists() or _find_muncie_dem() is None,
    reason="Muncie pilot data not present",
)
def test_cached_features_roundtrip_matches_in_memory_stack(tmp_path):
    """The cached features.tif, when read back, must match what
    `stack_dem_features` produces in memory — value-for-value (within
    float32 precision)."""
    from hecras_mesh_ai.features import stack_dem_features

    dem = _find_muncie_dem()
    paths = cache_pilot_project(
        project_name="muncie",
        dem_path=dem,
        geometry_hdf_path=_MUNCIE_HDF,
        buffer_width=20.0,
        out_dir=tmp_path,
    )

    in_memory = stack_dem_features(dem)
    with rasterio.open(paths.features) as f:
        cached = f.read()  # shape (bands, H, W)

    assert cached.shape == in_memory.shape
    # float32 round-trip: values agree to ~1e-5 relative tolerance.
    np.testing.assert_allclose(cached, in_memory.values, rtol=1e-5, atol=1e-5, equal_nan=True)
