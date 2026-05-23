# HEC-RAS Mesh AI

> Automated 2D mesh generation for HEC-RAS using deep learning.

Manual 2D mesh generation in HEC-RAS is slow, iterative, and computationally expensive. Modelers spend hours placing breaklines, defining refinement regions, and tuning cell resolution — then must run a simulation to discover whether the mesh performs adequately. This project aims to learn that workflow from expert-meshed projects and propose high-quality meshes directly from terrain and ancillary geospatial data.

## Approach

A staged plan: predict breaklines first (a topographic computer-vision problem), then add a learned resolution field for complete static meshes, then close the loop with solution-based adaptive refinement.

- **Phase A** — Breakline detection from DEM + ancillary data
- **Phase B** — + Refinement region / resolution field for complete static mesh
- **Phase C** — Closed-loop mesh adaptation driven by simulation residuals

See [`docs/roadmap.md`](docs/roadmap.md) for detail.

## Stack

PyTorch + Lightning + segmentation_models_pytorch + TorchGeo, on top of the standard Python geospatial stack (rasterio, geopandas, xarray). HEC-RAS HDF I/O via `rashdf` and `h5py`. Experiment tracking with Weights & Biases.

## Status

Phase A in early development. Pilot dataset: HEC-RAS official example projects (Muncie, Bald Eagle Dam Break).

## Project structure

```
docs/                  # roadmap, decisions, domain primer
src/hecras_mesh_ai/    # package
notebooks/             # exploration
tests/                 # pytest
```

## Getting started

```bash
uv sync
uv run pre-commit install
uv run pytest
```

## Development

See [`CLAUDE.md`](CLAUDE.md) for working conventions and [`docs/decisions/`](docs/decisions/) for the record of significant decisions.

## License

TBD.
