"""Cache a pilot project's features + labels to GeoTIFFs.

A pilot project = one DEM TIFF + one geometry HDF (rashdf-readable). For each,
we materialize the Stage 1 outputs to disk as GeoTIFFs so the dataset layer
can read tiles via rasterio windows during training — no recomputation per
epoch, no in-memory pressure on the 36-60 Mpx pilots.

  features.tif  : 6 bands, float32, channels in FEATURE_CHANNELS order.
                  CRS + transform from the source DEM.
  labels.tif    : 1 band, uint8, values in {0, 1}. Same CRS/transform/shape.

Both files share the source DEM's spatial layout pixel-for-pixel, so a
rasterio window over either reads aligned data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rashdf
import xarray as xr

from hecras_mesh_ai.features import FEATURE_CHANNELS, stack_dem_features
from hecras_mesh_ai.labels import rasterize_breaklines


@dataclass(frozen=True)
class CachedPaths:
    """File paths produced by `cache_pilot_project`."""

    project_name: str
    features: Path
    labels: Path
    feature_channels: tuple[str, ...] = FEATURE_CHANNELS


def cache_pilot_project(
    *,
    project_name: str,
    dem_path: Path | str,
    geometry_hdf_path: Path | str,
    buffer_width: float,
    out_dir: Path | str,
    patch_max_size: int = 1,
) -> CachedPaths:
    """Compute and cache features + labels for one pilot project.

    Parameters
    ----------
    project_name
        Subdirectory name under `out_dir`. Used as a stable identifier
        downstream (the dataset reads <out_dir>/<project_name>/features.tif).
    dem_path
        Path to the source DEM (rasterio-readable). Passed through to
        `stack_dem_features`.
    geometry_hdf_path
        Path to the geometry HDF whose breaklines become the label raster.
        Read via `rashdf.RasGeomHdf`.
    buffer_width
        Breakline label thickness in CRS linear units (typically feet).
        Passed through to `rasterize_breaklines`.
    out_dir
        Root directory under which `<project_name>/` is created. Typically
        `data/processed/pilot` (gitignored).
    patch_max_size
        Hybrid NaN-conditioning size limit, passed through to
        `stack_dem_features`. Default 1.

    Returns
    -------
    CachedPaths
        Records the two written file paths and the feature channel order.
    """
    out_dir = Path(out_dir) / project_name
    out_dir.mkdir(parents=True, exist_ok=True)

    features = stack_dem_features(
        dem_path,
        patch_max_size=patch_max_size,
    )

    g = rashdf.RasGeomHdf(Path(geometry_hdf_path))
    try:
        breaklines = g.breaklines()
    finally:
        g.close()

    labels = rasterize_breaklines(
        breaklines,
        out_shape=features.shape[1:],
        transform=features.rio.transform(),
        target_crs=features.rio.crs,
        buffer_width=buffer_width,
    )

    features_path = out_dir / "features.tif"
    labels_path = out_dir / "labels.tif"

    features.rio.to_raster(
        features_path,
        dtype="float32",
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    # Wrap the label ndarray as a DataArray so we can write it via rioxarray
    # and preserve CRS + transform identically to the features.
    label_da = xr.DataArray(
        labels,
        dims=("y", "x"),
        coords={"y": features.y, "x": features.x},
    )
    label_da.rio.write_crs(features.rio.crs, inplace=True)
    label_da.rio.write_transform(features.rio.transform(), inplace=True)
    label_da.rio.to_raster(
        labels_path,
        dtype="uint8",
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    return CachedPaths(
        project_name=project_name,
        features=features_path,
        labels=labels_path,
    )
