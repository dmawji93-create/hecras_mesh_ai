"""CRS-aware DEM feature stacker — the integration layer for Stage 1.

Given a path to a DEM, return a (6, H, W) `xarray.DataArray` carrying:

    elevation, slope, aspect_sin, aspect_cos, plan_curvature, profile_curvature

with the source CRS and affine transform attached via rioxarray. This is the
single function the breakline rasterizer (Task 4) and TorchGeo dataset
(Task 5) consume.

Pipeline:
  1. Open the DEM with rasterio; convert nodata -> NaN.
  2. Assert north-up row orientation (transform.e < 0) — the runtime check
     for the CONVENTION-TO-VERIFY breadcrumb in aspect.py. A row-up DEM
     would silently flip aspect 180 deg; we refuse it loudly here so the
     failure surfaces at ingest, not at training.
  3. Patch isolated single-pixel NaN holes (hybrid policy from
     conditioning.py). Larger NaN regions pass through unchanged.
  4. Reflect-pad the DEM by 1 pixel to handle the natural stencil
     boundary, so the output has the same spatial extent as the input
     with derivatives valid all the way to the border.
  5. Compute slope, aspect (sin/cos), plan curvature, profile curvature.
  6. Crop each derivative back to the original DEM shape and stack as
     (band, y, x) with the patched elevation as the first band.
  7. Wrap as xarray.DataArray; attach CRS and transform via rioxarray.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import rioxarray  # noqa: F401  -- registers the .rio accessor on xarray objects
import xarray as xr

from hecras_mesh_ai.features.aspect import aspect_sincos
from hecras_mesh_ai.features.conditioning import patch_isolated_nan
from hecras_mesh_ai.features.plan_curvature import plan_curvature
from hecras_mesh_ai.features.profile_curvature import profile_curvature
from hecras_mesh_ai.features.slope import slope

FEATURE_CHANNELS: tuple[str, ...] = (
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "plan_curvature",
    "profile_curvature",
)


def stack_dem_features(
    dem_path: Path | str,
    *,
    patch_max_size: int = 1,
    slope_units: str = "degrees",
) -> xr.DataArray:
    """Read a DEM and return all derived features stacked as (band, y, x).

    Parameters
    ----------
    dem_path
        Path to a rasterio-readable DEM (GeoTIFF, VRT, etc.).
    patch_max_size
        Maximum NaN connected-component size to fill via
        `patch_isolated_nan`. Default 1.
    slope_units
        Units for the slope channel: "degrees" (default), "radians",
        or "percent".

    Returns
    -------
    xarray.DataArray
        Shape (6, H, W) with dims ("band", "y", "x"). The `band` coordinate
        contains the FEATURE_CHANNELS names. CRS and transform are attached
        via the `.rio` accessor.

    Raises
    ------
    ValueError
        If the DEM is not in standard north-up row orientation
        (transform.e >= 0). See the CONVENTION-TO-VERIFY block in
        aspect.py for the diagnostic recipe.
    """
    dem_path = Path(dem_path)
    with rasterio.open(dem_path) as src:
        if src.transform.e >= 0:
            raise ValueError(
                f"DEM at {dem_path} has transform.e = {src.transform.e:.6f} >= 0, "
                "meaning row index increases northward. This module assumes the "
                "standard north-up layout (row index increases southward, "
                "transform.e < 0). See the CONVENTION-TO-VERIFY block in "
                "src/hecras_mesh_ai/features/aspect.py for the diagnostic "
                "recipe. Reproject the DEM to standard orientation first."
            )
        z = src.read(1, masked=True).astype(np.float64).filled(np.nan)
        transform = src.transform
        crs = src.crs

    cellsize_x = float(transform.a)
    cellsize_y = float(-transform.e)

    z_patched = patch_isolated_nan(z, max_patch_size=patch_max_size)
    z_padded = np.pad(z_patched, 1, mode="reflect")

    # Derivatives on the padded array, then cropped back to original shape.
    sl_full = slope(z_padded, cellsize_x, cellsize_y, units=slope_units)
    asin_full, acos_full = aspect_sincos(z_padded, cellsize_x, cellsize_y)
    pc_full = plan_curvature(z_padded, cellsize_x, cellsize_y)
    pfc_full = profile_curvature(z_padded, cellsize_x, cellsize_y)

    el = z_patched
    sl = sl_full[1:-1, 1:-1]
    asin = asin_full[1:-1, 1:-1]
    acos = acos_full[1:-1, 1:-1]
    pc = pc_full[1:-1, 1:-1]
    pfc = pfc_full[1:-1, 1:-1]

    stack = np.stack([el, sl, asin, acos, pc, pfc], axis=0)

    height, width = el.shape
    # Pixel-center coordinates in the source CRS.
    xs = transform.c + transform.a * (np.arange(width) + 0.5)
    ys = transform.f + transform.e * (np.arange(height) + 0.5)

    da = xr.DataArray(
        stack,
        dims=("band", "y", "x"),
        coords={
            "band": list(FEATURE_CHANNELS),
            "y": ys,
            "x": xs,
        },
        name="dem_features",
    )
    da.rio.write_crs(crs, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    return da
