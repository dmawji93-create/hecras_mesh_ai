# ADR 005: Tech Stack

**Status:** Accepted
**Date:** 2026-05-21

## Context

The project needs a tech stack that:

1. Fits the problem (geospatial semantic segmentation on multi-channel raster data, scaling to large corpora)
2. Builds durable, transferable ML engineering skills for the user (who is new to ML)
3. Has a healthy ecosystem and active maintenance

## Decision

**Python 3.11+** with `uv` for environment management. ML on **PyTorch + PyTorch Lightning + segmentation_models_pytorch + TorchGeo**. Geospatial I/O via the standard scientific Python geospatial stack. HEC-RAS HDF I/O via `rashdf` and `h5py`. Experiment tracking with Weights & Biases.

### Full stack

- **Language:** Python 3.11+
- **Env management:** `uv`
- **Core ML:** PyTorch, PyTorch Lightning
- **Model architectures:** `segmentation_models_pytorch` (smp)
- **Geospatial DL:** TorchGeo
- **Geospatial I/O:** rasterio, geopandas, xarray, rioxarray, shapely
- **Image processing:** scikit-image
- **HEC-RAS I/O:** `rashdf`, `h5py`
- **Experiment tracking:** Weights & Biases
- **Quality:** pytest, ruff, pre-commit
- **Config management (later):** Hydra

## Consequences

### Positive
- PyTorch is the dominant research and production ML framework; skills transfer everywhere.
- Lightning hides boilerplate while exposing the patterns underneath, ideal for learning.
- smp gives well-tested, swappable architectures (U-Net, U-Net++, DeepLabV3+, etc.) — focus on data and training, not re-implementing primitives.
- TorchGeo is purpose-built for geospatial DL: handles tile sampling, CRS, multi-band rasters, large-scale datasets.
- The geospatial stack (rasterio/geopandas/xarray) is the standard for Earth observation and GIS-adjacent ML work.
- Weights & Biases is essentially universal in industry; free tier is generous.
- `uv` is faster and more reliable than pip/conda and is becoming the modern default.

### Negative / risks
- Steeper initial learning curve than a pure scikit-learn / Keras approach.
- TorchGeo is still maturing — some rough edges and API churn expected.
- Multiple abstraction layers (Python → PyTorch → Lightning → smp → TorchGeo) means debugging can be tricky when something fails deep in the stack.
- `uv` is relatively new; small risk of ecosystem incompatibility for niche packages.

## Alternatives considered

- **TensorFlow / Keras:** Declining mindshare in research, less common in modern geospatial DL work.
- **Pure PyTorch without Lightning:** More foundational learning but slows the initial sprint significantly.
- **Foundation model fine-tune (Prithvi, SAM-geo) as the starting point:** Promising and on the table for Phase A.1, but not the default starting point because (a) backbones are large and slow, (b) ImageNet-pretrained smp encoders already work well for DEM-derived rasters, (c) starting simpler builds clearer understanding.
- **`conda` / `mamba`:** Mature but slower than `uv` and increasingly being replaced.

## Open questions

- Whether to introduce Hydra for config management from the start, or wait until config complexity demands it.
- When (and whether) to bring in a foundation model encoder for Phase A.1+ — depends on bulk corpus performance.
- Compute target (local vs cloud) — deferred until data scale is known.

## References

- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
- segmentation_models_pytorch: https://github.com/qubvel-org/segmentation_models.pytorch
- TorchGeo: https://torchgeo.readthedocs.io
- rashdf: https://github.com/fema-ffrd/rashdf
- uv: https://docs.astral.sh/uv/
