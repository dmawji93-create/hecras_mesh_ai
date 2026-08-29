# HEC-RAS Mesh AI

> Automated 2D mesh generation for HEC-RAS using deep learning.

Manual 2D mesh generation in HEC-RAS is slow, iterative, and bounded by expert intuition — and hand-built meshes are imperfect and unauditable. This project learns the workflow from expert-meshed projects, proposes high-quality meshes directly from terrain and ancillary geospatial data, and then refines them against a measurable, quantitative objective. The goal is meshes that are **faster, better, and more quantifiable** than the manual process.

## Approach

A staged plan:

- **Phase A** — Breakline detection from DEM + ancillary data.
- **Phase B** — A learned resolution field for complete, runnable static meshes (the Quick tier).
- **Phase C** — Adaptive refinement against a quantitative mesh-quality objective (the Optimal tier).
- **Phase D** — Productionization.

The strategic plan is in [`docs/roadmap.md`](docs/roadmap.md); the executable, checkpointed plan is in [`docs/build-plan/`](docs/build-plan/).

## Stack

PyTorch + Lightning + segmentation_models_pytorch + TorchGeo, on the standard Python geospatial stack (rasterio, geopandas, xarray). HEC-RAS HDF I/O via `rashdf` and `h5py`. Experiment tracking with Weights & Biases. See [`docs/decisions/005-tech-stack.md`](docs/decisions/005-tech-stack.md).

## Status

Phase A, early development. Phase A.0 Week 1 (plumbing) complete. Pilot dataset: HEC-RAS official examples (Muncie, Bald Eagle Dam Break).

## Getting started

```bash
uv sync --extra dev --extra ml
uv run pre-commit install
uv run pytest
```

The `ml` extra resolves `torch`/`torchvision` from the PyTorch CUDA 12.8 index and is Windows + CUDA specific — see [`docs/decisions/010-cuda-platform.md`](docs/decisions/010-cuda-platform.md).

## Development

See [`CLAUDE.md`](CLAUDE.md) for working conventions and [`docs/decisions/`](docs/decisions/) for the record of significant decisions.

## License

MIT — see [`LICENSE`](LICENSE).
