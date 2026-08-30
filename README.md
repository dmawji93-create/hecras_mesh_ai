# HEC-RAS Mesh AI

> Automated 2D mesh generation for HEC-RAS using deep learning — with every mesh validated by physics.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue.svg)](pyproject.toml)
[![HEC-RAS 7.0](https://img.shields.io/badge/HEC--RAS-7.0-lightgrey.svg)](https://www.hec.usace.army.mil/software/hec-ras/)

Manual 2D mesh generation in HEC-RAS is slow, iterative, and bounded by expert intuition — and hand-built meshes ship with no quantitative statement of their adequacy. This project learns the expert workflow (breakline placement, refinement, resolution selection) from terrain and hydraulic principles, proposes meshes directly, and scores every mesh against a measurable error objective by running the simulation. The goal: meshes that are **faster, better, and quantifiable** — including a mesh-quality certificate no manual workflow can produce.

**Why doesn't this already exist?** CFD has had physics-validated meshing for decades; no riverine flood package ships it. The reasons are structural, and they carve out exactly the niche this tool occupies — see [`docs/research/why-no-physics-validated-meshing.md`](docs/research/why-no-physics-validated-meshing.md) (with a [one-page visual summary](docs/research/why-no-physics-validated-meshing-onepager.pdf)).

## How it works

A staged plan (strategic view in [`docs/roadmap.md`](docs/roadmap.md); executable, checkpointed plan in [`docs/build-plan/`](docs/build-plan/)):

- **Phase A** — Breakline detection from DEM + ancillary data. ✅ Complete: feature/label pipeline, U-Net pilot, and a closed-loop HEC-RAS automation harness (write geometry HDF → launch compute via COM → parse results), verified end-to-end against HEC-RAS 7.0.
- **Stage 6 (in progress)** — The mesh-quality measurement framework: analytical benchmarks (Thacker parabolic bowl — exact solution implemented and machine-precision verified), grid-refinement sequences, Richardson extrapolation, and a per-mesh error vector.
- **Phase B** — A learned resolution field for complete, runnable static meshes.
- **Phase C** — Refinement against the quantitative objective.

**Training data** comes from a certified-synthetic factory ([ADR 014](docs/decisions/014-certified-synthetic-data-strategy.md)): candidate breaklines are proposed cheaply from public data (USGS 3DEP terrain, levee/hydrography/transportation vectors) and **certified by simulation ablation** — a label earns its place only if removing it measurably degrades the solution. The proposer is never the judge.

## Stack

PyTorch + Lightning + segmentation_models_pytorch + TorchGeo, on the standard Python geospatial stack (rasterio, geopandas, xarray). HEC-RAS HDF I/O via `rashdf` and `h5py`; HEC-RAS control via COM (pywin32). See [ADR 005](docs/decisions/005-tech-stack.md).

## Getting started

```bash
uv sync --extra dev --extra ml --extra harness
uv run pre-commit install
uv run pytest -m "not slow"
```

Notes:

- The `ml` extra resolves `torch`/`torchvision` from the PyTorch CUDA 12.8 index and is Windows + CUDA specific — see [ADR 010](docs/decisions/010-cuda-platform.md).
- The `harness` extra (pywin32) is Windows-only and needed only for driving HEC-RAS.
- Tests marked `slow` launch a real local HEC-RAS 7.0 install; the default selection above skips them.
- **Pilot data is not included.** The pilots use the official HEC-RAS example projects (Muncie; Bald Eagle Dam Break), installable from HEC-RAS itself (*Help → Install Example Projects*); place them under `data/raw/usace/RAS Samples/`. Trained checkpoints and cached features are likewise gitignored and regenerable (`scripts/train_pilot.py`, notebook 03).

## Project documentation

| Where | What |
|---|---|
| [`docs/STATUS.md`](docs/STATUS.md) | Current project status — start here |
| [`docs/research/`](docs/research/) | Why this tool should exist: the meshing-gap research + market analysis |
| [`docs/build-plan/`](docs/build-plan/) | Executable, checkpointed stage plan (the operational source of truth) |
| [`docs/decisions/`](docs/decisions/) | Architecture Decision Records 001–014 |
| [`docs/hdf-schema/`](docs/hdf-schema/) | Reverse-engineered HEC-RAS geometry/results HDF schema notes |
| [`docs/hec-ras-primer.md`](docs/hec-ras-primer.md) | HEC-RAS mesh concepts for ML readers |
| [`CLAUDE.md`](CLAUDE.md) | Working conventions (AI-assisted development per ADR 008) |

## Status

Solo research project under active development; not yet a usable tool. Issues and discussion welcome. Findings, negative results, and defects are recorded honestly in the docs — see the audit notes in `docs/STATUS.md`.

## License

MIT — see [`LICENSE`](LICENSE).
