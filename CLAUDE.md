# HEC-RAS Mesh AI

Automated 2D computational mesh generation for HEC-RAS using deep learning. Replaces the iterative manual workflow of breakline placement, refinement region selection, and resolution tuning with a learned system that proposes meshes from terrain and ancillary data.

## Current status

**Phase A — Breakline Detection** (active).
Sprint: "make-one-work" pilot on the HEC-RAS Muncie example before scaling.

See `docs/roadmap.md` for the full A → B → C plan and current sprint detail.

## How I work with the user

The user has deep HEC-RAS / hydraulic engineering expertise but is **new to ML and PyTorch**. Teach as you go: explain *why*, not just *what*. Prefer incremental progress over dumping finished solutions. When a new concept appears (tensor, autograd, U-Net, IoU, Dice loss, DataLoader, etc.), give a short conceptual primer before using it. Assume strong general engineering ability and strong domain knowledge — just no ML background.

When proposing significant decisions (architecture, dependencies, sprint scope), surface the alternatives and tradeoffs, then recommend. Don't just execute silently on choices that have downstream implications.

## Tech stack

- **Language:** Python 3.11+, environment managed with `uv`
- **Core ML:** PyTorch + PyTorch Lightning
- **Model architectures:** `segmentation_models_pytorch` (smp) — U-Net, DeepLabV3+, etc. with swappable encoders
- **Geospatial DL:** TorchGeo (samplers, transforms, geo-aware datasets)
- **Geospatial I/O:** rasterio, geopandas, xarray + rioxarray, shapely
- **Image processing:** scikit-image (skeletonize, morphology, vectorize)
- **HEC-RAS I/O:** `rashdf` (read geometry/plan HDFs), `h5py` (low-level, eventual writes)
- **Experiment tracking:** Weights & Biases
- **Quality:** pytest, ruff, pre-commit
- **Configs (later):** Hydra

Full rationale: `docs/decisions/005-tech-stack.md`. Compute target deferred until data scale is understood — see `docs/decisions/004-training-data-source.md`.

## Workflow conventions

- **Notebooks** (`notebooks/`) — exploration only. One-off plots, data sanity checks, "what does this look like." Notebooks are throwaway by default; nothing imports from them.
- **Modules** (`src/hecras_mesh_ai/`) — keepers. Anything that gets imported or run more than twice lives here. Typed where it helps, tested where it matters.
- **Tests** (`tests/`) — pytest. Run via pre-commit.
- **ADRs** (`docs/decisions/`) — one markdown file per significant decision. If a decision is revisited, update the file's status and create a successor ADR — never silently rewrite history.
- **Roadmap** (`docs/roadmap.md`) — living document. Update as phases progress; note what changed and why.
- **Commits** — conventional commits format (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- **Branches** — `main` is protected; feature work happens on short-lived branches with PRs.

## Repository layout

```
hecras_mesh_ai/
├── CLAUDE.md                  # this file — auto-loaded by Claude Code
├── README.md                  # public-facing entry point
├── pyproject.toml             # project metadata + dependencies (uv)
├── .pre-commit-config.yaml
├── docs/
│   ├── roadmap.md             # phased delivery plan
│   ├── hec-ras-primer.md      # mesh concepts for ML folks
│   └── decisions/             # ADRs
├── src/hecras_mesh_ai/        # the package
├── notebooks/                 # exploration only
├── tests/                     # pytest
└── data/                      # gitignored; raw + processed datasets
```

## HEC-RAS domain context

If you're a Claude Code session without HEC-RAS background, read `docs/hec-ras-primer.md` first. Key terms you must understand before touching this codebase: *breakline*, *refinement region*, *computation point*, *2D flow area*, *sub-grid bathymetry*, *Voronoi mesh*, *cell face*.

## Reference links

- HEC-RAS 2D User's Manual: https://www.hec.usace.army.mil/confluence/rasdocs/r2dum/latest
- 2D Computational Mesh Development: https://www.hec.usace.army.mil/confluence/rasdocs/r2dum/6.0/development-of-a-2d-or-combined-1d-2d-model/development-of-the-2d-computational-mesh
- Sub-grid bathymetry theory: https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/theoretical-basis-for-one-dimensional-and-two-dimensional-hydrodynamic-calculations/2d-unsteady-flow-hydrodynamics/subgrid-bathymetry
- Mesh quality best practices: https://www.hec.usace.army.mil/confluence/rasdocs/h2sd/ras2dsed/6.0/hydraulic-best-practices-for-a-2d-sediment-model/mesh-quality
- rashdf: https://github.com/fema-ffrd/rashdf
- TorchGeo: https://torchgeo.readthedocs.io
- segmentation_models_pytorch: https://github.com/qubvel-org/segmentation_models.pytorch
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/

## When in doubt

Ask the user before making non-trivial decisions. Document them in `docs/decisions/` afterward. The cost of an extra prompt is small; the cost of silent drift is large.
