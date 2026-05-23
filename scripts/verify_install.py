"""Quick post-install verification: import every declared ML/geospatial library
and report its version. Throwaway script — not part of the package."""

from importlib.metadata import version

import geopandas
import h5py
import lightning
import rashdf  # noqa: F401  (no __version__ attribute; pulled from metadata)
import rasterio
import rioxarray
import segmentation_models_pytorch as smp
import shapely
import skimage
import torch
import torchgeo
import wandb
import xarray

import hecras_mesh_ai

rows = [
    ("torch", f"{torch.__version__}  (cuda={torch.cuda.is_available()})"),
    ("lightning", lightning.__version__),
    ("segmentation-models-pytorch", smp.__version__),
    ("torchgeo", torchgeo.__version__),
    ("rasterio", rasterio.__version__),
    ("geopandas", geopandas.__version__),
    ("shapely", shapely.__version__),
    ("xarray", xarray.__version__),
    ("rioxarray", rioxarray.__version__),
    ("scikit-image", skimage.__version__),
    ("rashdf", version("rashdf")),
    ("h5py", h5py.__version__),
    ("wandb", wandb.__version__),
    ("hecras_mesh_ai", hecras_mesh_ai.__version__),
]

print(f"{'package':32s} version")
print("-" * 60)
for name, ver in rows:
    print(f"{name:32s} {ver}")
