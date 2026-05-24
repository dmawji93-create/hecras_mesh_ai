"""Dataset assembly: cache pilot features + labels to disk, serve tiles to the model.

Three layers, in commit order:
  - `cache`     : compute feature stack + label raster for a pilot project,
                  write both as GeoTIFFs under data/processed/<project_name>/.
  - `tile_dataset` (next) : open cached GeoTIFFs, serve (features, labels)
                  tiles for training and validation.
  - `split`     (next) : verify train/val spatial separation.
"""

from hecras_mesh_ai.dataset.cache import CachedPaths, cache_pilot_project

__all__ = ["CachedPaths", "cache_pilot_project"]
