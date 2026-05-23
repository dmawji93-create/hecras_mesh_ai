# Project Status — `hecras_mesh_ai`

**As of:** 2026-05-22
**Phase:** A.0 "make-one-work" sprint (Week 1 complete)
**Repo:** `C:\dev\hecras_mesh_ai\AI Assisted Mesh Generation\`
**Branch:** `master` · 5 commits · no remote yet

---

## What this project is

Automated 2D computational mesh generation for HEC-RAS using deep learning. Replaces the manual workflow of breakline placement, refinement region selection, and resolution tuning with a learned system that proposes meshes from terrain + ancillary data. Staged delivery: **A** breaklines → **B** + resolution field for complete static meshes → **C** closed-loop adaptation against simulation runs.

## What's done

**Week 1 — Plumbing (complete).** Repository scaffolded with the `hecras_mesh_ai` package, `uv`-managed Python 3.12 environment, full ML/geospatial dependency stack installed, GPU enabled, pilot data in place, exploration notebook produced and verified end-to-end against the Muncie example. Pre-commit hooks (ruff, ruff-format, whitespace, large-file guard) active and tested. Five commits, all conventional-commit format. Project moved off OneDrive onto local SSD (`C:\dev\…`) after OneDrive caused hardlink and file-lock failures.

| Component | Status |
|---|---|
| Repo skeleton (`src/`, `tests/`, `notebooks/`, `docs/`, `data/`) | ✓ |
| `pyproject.toml` (full ML stack declared) | ✓ |
| `uv.lock` (reproducible env) | ✓ |
| Pre-commit hooks (ruff + format + safety) | ✓ |
| Smoke tests (`pytest`) | ✓ 2/2 passing |
| Pilot data (Muncie + Bald Eagle in `data/raw/usace/`) | ✓ |
| `notebooks/01_muncie_explore.ipynb` (read + plot verified) | ✓ |

## Current environment

| | |
|---|---|
| Python | 3.12.13 (uv-managed in `.venv/`) |
| Torch | `2.11.0+cu128` (CUDA enabled) |
| GPU | NVIDIA RTX 3090, 24 GB VRAM, driver 591.86 |
| ML stack | lightning 2.6.4 · segmentation-models-pytorch 0.5.0 · torchgeo 0.9.0 |
| Geospatial | rasterio 1.5.0 · geopandas 1.1.3 · shapely 2.1.2 · xarray 2026.4.0 · rioxarray 0.22.0 |
| HEC-RAS I/O | rashdf 0.12.0 · h5py 3.16.0 |
| Tracking | wandb 0.27.0 (not yet wired up) |

GPU verified with a 4096×4096 matmul on `cuda:0` (~200 MB VRAM used).

## What we learned from the Muncie read

- HEC-RAS Muncie geometry HDF (`Muncie.g04.hdf`) opens cleanly via `rashdf`.
- The HDF does *not* embed a CRS. The authoritative CRS lives on the terrain TIFF: **EPSG:2965 (NAD83 / Indiana East ftUS)**. Pipeline must source CRS from terrain when the HDF lacks it.
- Mesh: **5,391 cells**, 5,391 computation points, 1 perimeter polygon, **2 breaklines** ("Road 1" + "HighGround 1"), **0 refinement regions**.
- Terrain: 4,538 × 7,892 pixels at 5 ft resolution, elevation 898.9 → 1,013.2 ft (low-relief Indiana). Mesh sits comfortably inside terrain extent.
- Muncie is for plumbing only — 2 breakline labels is not a training signal.

## Decisions on record (ADRs in `docs/decisions/`)

001 Staged A → B → C delivery · 002 Mixed/general scope · 003 Supervised pretrain + performance fine-tune · 004 Hybrid FEMA + curated corpus · 005 Tech stack (PyTorch + Lightning + smp + TorchGeo) · 006 Pilot dataset (HEC-RAS official examples) · 007 Make-one-work sprint · 008 Claude Code in VS Code as system of record · 009 Notebooks for exploration, modules for keepers · **010 CUDA 12.8 via PyTorch index (RTX 3090), OneDrive operational notes**.

## Known open items

- **Bald Eagle Dam Break** not yet opened. Worth a quick check at the start of Week 2 to see if its newer HDF format embeds CRS properly (would let us prefer-HDF, fall-back-terrain).
- **Project folder name has spaces** (`AI Assisted Mesh Generation`). Working fine; flag if a CLI tool ever mishandles it.
- **No git remote yet.** Repo is local-only; cloud backup via private GitHub remote is planned but not done.
- **`refinement_regions()` returns an empty GeoDataFrame with no geometry column when empty.** Worked around in the notebook; possibly upstream rashdf bug worth filing later.

## Next: Week 2 — Features and labels

1. **DEM derivatives module** — slope, aspect, plan/profile curvature, TWI, flow accumulation. Each a tested function in `src/hecras_mesh_ai/features/`.
2. **Breakline rasterizer** — `breaklines` GeoDataFrame → binary label raster aligned to the DEM grid, configurable buffer width.
3. **TorchGeo-based dataset + spatial-holdout tile split** — wrap Muncie (and Bald Eagle) for sampling, with explicit spatial separation between train and val tiles.

After Week 2 we'll have features in / labels out, ready for Week 3 to train a U-Net.
