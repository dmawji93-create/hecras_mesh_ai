# Project Status — `hecras_mesh_ai`

**As of:** 2026-05-23
**Phase:** A.0 "make-one-work" sprint — **build-plan Stage 1 complete**; next: Stage 2 (Breakline Model Pilot)
**Repo:** `C:\dev\hecras_mesh_ai\`
**Branch:** `main` · 20 commits · no remote yet

---

## What this project is

Automated 2D computational mesh generation for HEC-RAS using deep learning. Replaces the slow, intuition-bound manual workflow of breakline placement, refinement region selection, and resolution tuning with a learned system that proposes meshes from terrain + ancillary data and refines them against a measurable, quantitative objective (ADR 011). Staged delivery: **A** breaklines → **B** + resolution field for complete static meshes → **C** closed-loop adaptation against a quantitative objective → **D** productionization. See `docs/roadmap.md` for the strategic view; `docs/build-plan/` is the operational source of truth.

## What's done

**Week 1 — Plumbing (complete).** Repository scaffolded with the `hecras_mesh_ai` package, `uv`-managed Python 3.12 environment, full ML/geospatial dependency stack installed, GPU enabled, pilot data in place, exploration notebook produced and verified end-to-end against the Muncie example. Pre-commit hooks (ruff, ruff-format, whitespace, large-file guard) active and tested. All commits use conventional-commit format. Project moved off OneDrive onto local SSD (`C:\dev\…`) after OneDrive caused hardlink and file-lock failures (ADR 010 operational notes).

**Post-Week-1 reconciliation (2026-05-23).** Added `docs/build-plan/` (10 stages) as the executable source of truth alongside the strategic roadmap; added ADR 011 (quantitative mesh-quality objective) and ADR 012 (refinement loop is numerical-methods-first); amended ADR 003 to reframe expert meshes as a prior, not the training target; rewrote `CLAUDE.md` and `README.md`; added `.gitattributes` (LF normalization) and `.gitignore` entry for `.claude/`.

**Build-plan Stage 1 — Feature & Label Pipeline (complete, 2026-05-23).** All six tasks landed across ~15 commits:

| Task | Module | Tests |
|---|---|---|
| 1 — Bald Eagle exploration | `notebooks/02_baldeagle_explore.ipynb` | n/a |
| 2 — DEM derivatives (slope + aspect_sincos + plan/profile curvature) | `src/hecras_mesh_ai/features/{slope,aspect,plan_curvature,profile_curvature}.py` | 31 (synthetic + closed-form) |
| 3 — Feature stacker (CRS-aware, NaN-conditioning, row-direction runtime assertion) | `src/hecras_mesh_ai/features/{conditioning,stacker}.py` | 23 (synthetic + integration on both pilots) |
| 4 — Breakline rasterizer | `src/hecras_mesh_ai/labels/breaklines.py` | 13 (synthetic + integration) |
| 5 — Cache + RasterTileDataset + RandomTileSampler + IterableTileDataset + spatial-holdout | `src/hecras_mesh_ai/dataset/{cache,tile_dataset,split}.py` | 21 (synthetic + integration; full DataLoader path) |
| 6 — Exit notebook | `notebooks/03_stage1_exit_features_and_labels.ipynb` | runs end-to-end on both pilots |

**Stage 1 exit criteria — all met:**

- [x] Every DEM-derivative function has passing unit tests — 31 in `tests/features/`.
- [x] Feature channels confirmed aligned (CRS, resolution, extent) — programmatically (stacker integration tests against Muncie EPSG:2965 and Bald Eagle EPSG:2271) and visually (notebook 03 cells 8 + 9 + 16).
- [x] Breakline label raster aligns pixel-for-pixel with the feature stack — cache round-trip test + notebook 03 visual overlays.
- [x] Train/val tiles spatially separated with zero overlap — `assert_no_spatial_overlap` passes (different CRS, mathematically disjoint).
- [x] Pipeline runs end-to-end on both Muncie and Bald Eagle — integration tests + notebook 03.
- [x] `pytest` green; pre-commit clean — every commit ran the hook chain.

**Train/val split decision:** train on Bald Eagle g09 (4 named breaklines: SayersDam, Lower, Middle, Upper); val on Muncie (Road 1, HighGround 1). Per-project cross-CRS spatial holdout. Rationale: more training-set prior signal, cleaner engineered-structure + topographic-feature coverage, val noise is acceptable for Stage 2's deliberately-overfit-pilot phase.

| Component | Status |
|---|---|
| Repo skeleton (`src/`, `tests/`, `notebooks/`, `docs/`, `data/`, `scripts/`) | ✓ |
| `pyproject.toml` (full ML stack declared, `dev`/`ml` optional extras) | ✓ |
| `uv.lock` (reproducible env) | ✓ |
| Pre-commit hooks (ruff + format + safety + 6 MB notebook limit) | ✓ active at git-hook level |
| Pytest suite | ✓ ~75 tests; ~3 min including pilot integration |
| Pilot data (Muncie + Bald Eagle in `data/raw/usace/RAS Samples/...`) | ✓ |
| Stage 1 feature + label pipeline | ✓ complete |

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

## Assumptions to verify at test time

- **Raster row direction.** The aspect feature (and curvature features that build on the same gradient) assumes input DEMs are north-up rasters with row index increasing southward (rasterio `transform.e < 0`). Most modern DEM products satisfy this, but a legacy / converted DEM might not — symptom would be aspect rotated 180° on that source. The Stage 1 Task 3 feature-stacker is the right place to assert this at ingest. Diagnostic recipe in the `CONVENTION-TO-VERIFY` comment in `src/hecras_mesh_ai/features/aspect.py`.

## Recently closed

- **Bald Eagle Dam Break opens cleanly with rashdf** (Stage 1 Task 1, 2026-05-23). Canonical pilot = `g09.hdf`. See `notebooks/02_baldeagle_explore.ipynb`.
- **CRS embedding behavior characterized** (Stage 1 Task 1, 2026-05-23). Variable across HDFs; pipeline needs prefer-HDF-fall-back-terrain resolver. Both Muncie and 11/12 Bald Eagle geometries fall back to terrain; `g09` is the HDF-embedded case.
- **`rashdf.refinement_regions()` empty-frame bug** — no longer surfaces under 0.12.0; returns `len() == 0` cleanly on all 12 Bald Eagle geometries and Muncie.

## Next: Build-plan Stage 2 — Breakline Model (Pilot)

Detailed in `docs/build-plan/02-breakline-model-pilot.md`. Summary:

1. PyTorch Lightning `DataModule` wrapping the Stage 1 `IterableTileDataset`.
2. `LightningModule` with a small `segmentation_models_pytorch` U-Net (ResNet-18/34 encoder, ImageNet init).
3. BCE + Dice loss to handle the ~99/1 class imbalance the Stage 1 exit notebook confirmed.
4. W&B logging from the first run.
5. **Overfit sanity check** on a handful of tiles — loss must drive to near-zero. Validates the gradient path.
6. Full pilot training run on Bald Eagle g09; val on Muncie.
7. Post-processing chain: probability threshold → skeletonize → vectorize → simplify → smooth → polylines.
8. Buffer-based IoU / F1 against expert breaklines.

Stage 2 exit criteria: overfit-loss collapses, full pilot run completes + logs to W&B, end-to-end terrain → polyline path works, IoU/F1 computed against expert, visualized overlays, pytest + pre-commit clean. Overfit on the pilot is expected and is the point — Stage 3 is where generalization gets attacked.
