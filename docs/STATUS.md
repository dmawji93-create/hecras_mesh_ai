# Project Status — `hecras_mesh_ai`

**As of:** 2026-05-23
**Phase:** A.0 "make-one-work" sprint — Week 1 plumbing complete; **next: build-plan Stage 1**
**Repo:** `C:\dev\hecras_mesh_ai\`
**Branch:** `main` · 4 commits · no remote yet

---

## What this project is

Automated 2D computational mesh generation for HEC-RAS using deep learning. Replaces the slow, intuition-bound manual workflow of breakline placement, refinement region selection, and resolution tuning with a learned system that proposes meshes from terrain + ancillary data and refines them against a measurable, quantitative objective (ADR 011). Staged delivery: **A** breaklines → **B** + resolution field for complete static meshes → **C** closed-loop adaptation against a quantitative objective → **D** productionization. See `docs/roadmap.md` for the strategic view; `docs/build-plan/` is the operational source of truth.

## What's done

**Week 1 — Plumbing (complete).** Repository scaffolded with the `hecras_mesh_ai` package, `uv`-managed Python 3.12 environment, full ML/geospatial dependency stack installed, GPU enabled, pilot data in place, exploration notebook produced and verified end-to-end against the Muncie example. Pre-commit hooks (ruff, ruff-format, whitespace, large-file guard) active and tested. All commits use conventional-commit format. Project moved off OneDrive onto local SSD (`C:\dev\…`) after OneDrive caused hardlink and file-lock failures (ADR 010 operational notes).

**Post-Week-1 reconciliation (2026-05-23).** Added `docs/build-plan/` (10 stages) as the executable source of truth alongside the strategic roadmap; added ADR 011 (quantitative mesh-quality objective) and ADR 012 (refinement loop is numerical-methods-first); amended ADR 003 to reframe expert meshes as a prior, not the training target; rewrote `CLAUDE.md` and `README.md`; added `.gitattributes` (LF normalization) and `.gitignore` entry for `.claude/`.

| Component | Status |
|---|---|
| Repo skeleton (`src/`, `tests/`, `notebooks/`, `docs/`, `data/`, `scripts/`) | ✓ |
| `pyproject.toml` (full ML stack declared, `dev`/`ml` optional extras) | ✓ |
| `uv.lock` (reproducible env) | ✓ |
| Pre-commit hooks (ruff + format + safety) | ✓ |
| Smoke tests (`pytest`) | ✓ 2/2 passing |
| Pilot data (Muncie + Bald Eagle in `data/raw/usace/RAS Samples/Example_Projects_7_0/2D Unsteady Flow Hydraulics/`) | ✓ |
| `notebooks/01_muncie_explore.ipynb` (read + plot verified) | ✓ |
| Build-plan + ADRs 011-012 + ADR 003 amendment | ✓ (2026-05-23) |

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

## What we learned from the pilots

**Muncie (`Muncie.g04.hdf`):**
- Opens cleanly via `rashdf`. HDF does *not* embed a CRS — sourced from the terrain TIFF: **EPSG:2965 (NAD83 / Indiana East ftUS)**.
- Mesh: 5,391 cells, 1 perimeter, 2 breaklines (`Road 1`, `HighGround 1`), 0 refinement regions.
- Terrain: 4,538 × 7,892 pixels at 5 ft resolution. Low-relief (898.9 → 1,013.2 ft). Mesh sits inside terrain.
- Plumbing pilot only — 2 breakline labels is not a training signal.

**Bald Eagle Dam Break (Stage 1 Task 1, see `notebooks/02_baldeagle_explore.ipynb`):**
- Ships **12 alternate geometry definitions** of the same flow area (`.g01–.g13.hdf`, skipping `.g04/.g05/.g07`).
- **Canonical pilot pick:** `BaldEagleDamBrk.g09.hdf` — 18,066 cells, **4 semantically named breaklines** (`SayersDam`, `Lower`, `Middle`, `Upper`), the only geometry of the 12 with a CRS **embedded directly in the HDF** (EPSG:2271, NAD83 / Pennsylvania North ftUS).
- **Secondary pilot:** `BaldEagleDamBrk.g02.hdf` — 28,449 cells, 7 breaklines (`Lower`, `Middle`, `HighwayRoad`, plus four generic `Breakline 1–4` helpers). More label samples but noisier per ADR 003 (amended).
- **Zero refinement regions across all 12 Bald Eagle geometries.** The pilots produce no refinement-region labels — that's Stage 3 / bulk-corpus work.
- DEM (`Terrain50.baldeagledem.tif`): 6,902 × 8,643 pixels at ~36.5 ft resolution (despite the misleading "50" in the filename), covering ~63 mi × 50 mi. Both meshes fit comfortably inside.

**Cross-pilot finding:** CRS embedding in HEC-RAS HDFs is **variable** — sometimes present, usually not. The feature pipeline must implement **prefer-HDF-CRS, fall-back-terrain** as a first-class behavior.

## Decisions on record (ADRs in `docs/decisions/`)

001 Staged A → B → C delivery · 002 Mixed/general scope · 003 Supervised pretrain + performance fine-tune **(amended 2026-05-23 — expert meshes are a prior, not the target)** · 004 Hybrid FEMA + curated corpus · 005 Tech stack (PyTorch + Lightning + smp + TorchGeo) · 006 Pilot dataset (HEC-RAS official examples) · 007 Make-one-work sprint · 008 Claude Code in VS Code as system of record · 009 Notebooks for exploration, modules for keepers · 010 CUDA 12.8 via PyTorch index (RTX 3090), OneDrive operational notes · **011 Quantitative mesh-quality objective** · **012 Refinement loop is numerical-methods-first; ML is an optional optimization layer**.

## Known open items

- **Project folder name has spaces** in the data path (`…/RAS Samples/Example_Projects_7_0/2D Unsteady Flow Hydraulics/…`). Working fine for `rashdf` and `rasterio`; flag if a CLI tool ever mishandles it.
- **No git remote yet.** Repo is local-only; cloud backup via private GitHub remote is planned but not done.

## Recently closed

- **Bald Eagle Dam Break opens cleanly with rashdf** (Stage 1 Task 1, 2026-05-23). Canonical pilot = `g09.hdf`. See `notebooks/02_baldeagle_explore.ipynb`.
- **CRS embedding behavior characterized** (Stage 1 Task 1, 2026-05-23). Variable across HDFs; pipeline needs prefer-HDF-fall-back-terrain resolver. Both Muncie and 11/12 Bald Eagle geometries fall back to terrain; `g09` is the HDF-embedded case.
- **`rashdf.refinement_regions()` empty-frame bug** — no longer surfaces under 0.12.0; returns `len() == 0` cleanly on all 12 Bald Eagle geometries and Muncie.

## Next: Build-plan Stage 1 — Feature & Label Pipeline

Detailed in `docs/build-plan/01-feature-and-label-pipeline.md`. Summary:

1. **Open Bald Eagle Dam Break with rashdf** — confirm CRS handling, pick the canonical `.gNN.hdf`, inventory breaklines/refinement regions/cells.
2. **DEM-derivatives module** (`src/hecras_mesh_ai/features/`) — slope, aspect, plan/profile curvature, TWI, flow accumulation. Each a tested, typed function.
3. **Feature stacking** into a CRS-aware multi-channel array (xarray + rioxarray); verify CRS, resolution, extent alignment across all channels.
4. **Breakline rasterizer** — `breaklines` GeoDataFrame → binary label raster aligned to the DEM grid, configurable buffer width.
5. **TorchGeo dataset + spatial-holdout split** — wrap Muncie + Bald Eagle for sampling with explicit spatial separation between train and val tiles.
6. **Exploration notebook** visualizing every feature channel and the label raster overlaid on terrain.

Stage 1 exit criteria (must all pass before Stage 2): every DEM-derivative has passing unit tests; feature channels aligned (CRS / resolution / extent) verified visually and programmatically; breakline label raster pixel-aligned with feature stack; train/val tiles spatially separated with zero overlap (leakage check); pipeline runs end-to-end on both pilots; `pytest` green; pre-commit clean.
