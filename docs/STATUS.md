# Project Status — `hecras_mesh_ai`

**As of:** 2026-08-24 *(doc dates are working-session dates; commit timestamps run a few days later on the machine clock)*
**Phase:** Phase A complete (Stages 1, 2, 4 + closing demo). **Stage 6 (mesh-quality framework) in progress.** Data strategy pivoted to the certified-synthetic factory (**ADR 014**); effective build order 1, 2, 4, 6 → 3 → 5. Next: wire the Thacker benchmark into a HEC-RAS project, then the Stage 6 grid-refinement runner.
**Repo:** `C:\dev\hecras_mesh_ai\`
**Branch:** `main` · remote: `github.com/dmawji93-create/hecras_mesh_ai` — **public, MIT-licensed** since 2026-08-24
**Tests:** 251 (250 pass + 1 platform-guard skip), incl. a live HEC-RAS 7.0 integration run

## 2026-08 resumption (after ~3 months idle)

- **Full system audit (2026-08-23).** Five parallel deep-dives over the data pipeline, model/postprocess, harness, benchmark, and docs. Core math verified sound everywhere (Horn slope/aspect, curvatures, loss, sliding-window stitching, HDF polyline encoding, ghost-cell mask, Thacker solution — several independently re-derived). ~14 major defects found, concentrated at integration boundaries; docs/plan staleness flagged throughout (largely repaired in this refresh).
- **Data strategy pivot — ADR 014.** NOAA ESIP credentials never arrived (13 weeks); client data is unavailable for training. The corpus is now *manufactured, not acquired*: rule/vector proposal tiers on public 3DEP terrain, certified by harness-run ablations scored against the Stage 6 error vector. Stage 3 rewritten accordingly and now depends on Stage 6.
- **"Why this tool should exist" research** (`docs/research/why-no-physics-validated-meshing.md` + one-pager): the five reasons no flood package ships physics-validated meshing, the market-demand analysis (calibration absorbs mesh error only where calibration data exists — the uncalibratable regime is the certificate's beachhead), and the HEC-RAS 2025 competitive assessment (verdict: net tailwind; arcs + metadata become the natural product surface).
- **Audit fixes landed (2026-08-24).** Thacker deployment defects (raster extent to (1+A)a, closed-form c0, exact registration, comparison protocol documented, independent-value + SWE-residual tests) — commit 6c548cb. Harness trust gaps (success requires fresh AND finished results via the `Solution` marker; COM timeout with orphan kill; in-place write opt-in; multi-part breakline refusal; completion-gated results parsing) — commit c7ff98a, verified against live HEC-RAS 7.0.
- **Open-sourced.** MIT license, repo public, GitHub secret scanning + push protection enabled; full-history secret sweep clean.
- **Bookkeeping correction:** the "Stage 5" label used below for the ML×harness wiring was a misnomer — build-plan Stage 5 is the resolution model. The wiring shipped as `scripts/phase_a_closing_demo.py` (commit 92c42ea, 2026-05-25): model → sliding-window inference → polylines → `replace_breaklines` → COM run → results parse, end-to-end on Muncie; the pipeline proved robust to bad model output (41 predicted breaklines land dry; max depth identical to baseline). An automated ML→harness integration test is still owed.
- Still open: degenerate CLI fallback (COM primary; now completion-guarded), raster row-direction assumption (unchanged), W&B not wired (decide at Stage 3), remaining audit minors (tracked in the owner's working notes, not in-repo), pre-commit ruff pinned older than dev ruff.
- **Step audit of this resumption batch (2026-08, two adversarial reviews):** all 12 audit fixes verified genuinely closed — closed forms independently re-derived; mutation experiments confirmed the new Thacker tests catch a wrong ω, a sign-flipped u, and a wrong c0. Follow-ups applied same day: the timeout kill now takes the full HEC-RAS process tree (`RasUnsteady.exe` et al., `taskkill /T`), `run_plan` clears stale results files before computing (makes freshness detection sound), lock-tolerant completion reads, a kill-baseline safety guard, CLI launch-failure capture, and doc reconciliations (roadmap body aligned to ADR 014; HEC-RAS 7.0 version story; kickoff/overview refresh). Accepted constraint: the PID-diff kill assumes one harness run per machine — the parallel factory must isolate runs (documented in `launch.py` and the Stage 3 notes).

*Counts and "next" statements below this line are historical (as of 2026-05-25), kept as the working record of Phase A.*

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
| Pytest suite | ✓ 206 tests; ~3 s for the fast suite, +30 s for the gated HEC-RAS integration test |
| Pilot data (Muncie + Bald Eagle in `data/raw/usace/RAS Samples/...`) | ✓ |
| Stage 1 feature + label pipeline | ✓ complete |
| Stage 2 breakline-detection U-Net pilot | ✓ complete |
| Stage 4 HEC-RAS engineering harness (writer + launcher + parser) | ✓ complete end-to-end against HEC-RAS 7.0 |

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

001 Staged A → B → C delivery · 002 Mixed/general scope · 003 Supervised pretrain + performance fine-tune **(amended 2026-05-23 — expert meshes are a prior, not the target)** · 004 Hybrid FEMA + curated corpus · 005 Tech stack (PyTorch + Lightning + smp + TorchGeo) · 006 Pilot dataset (HEC-RAS official examples) · 007 Make-one-work sprint · 008 Claude Code in VS Code as system of record · 009 Notebooks for exploration, modules for keepers · 010 CUDA 12.8 via PyTorch index (RTX 3090), OneDrive operational notes · **011 Quantitative mesh-quality objective** · **012 Refinement loop is numerical-methods-first; ML is an optional optimization layer** · **013 Bulk corpus access strategy (NOAA OWP primary, FEMA BLE secondary, MIP deferred)**.

## Known open items

- **NOAA ESIP credentials request in flight** (sent 2026-05-24 to the NOAA OWP team). Stage 3 (bulk-corpus harvesting via `s3://noaa-nws-owp-fim/ras2fim`) cannot start until creds arrive. Latency: days to weeks.
- **HEC-RAS CLI is undocumented for 7.0.** The launcher's CLI fallback runs `Ras.exe -c <prj> p<NN>` which exits 0 but doesn't actually compute. COM is the working primary; CLI fallback is degenerate. Not blocking — COM works — but a real CLI path would simplify batch contexts.
- **Project folder name has spaces** in the data path (`…/RAS Samples/Example_Projects_7_0/2D Unsteady Flow Hydraulics/…`). Working fine for `rashdf` and `rasterio`; flag if a CLI tool ever mishandles it.

## Assumptions to verify at test time

- **Raster row direction.** The aspect feature (and curvature features that build on the same gradient) assumes input DEMs are north-up rasters with row index increasing southward (rasterio `transform.e < 0`). Most modern DEM products satisfy this, but a legacy / converted DEM might not — symptom would be aspect rotated 180° on that source. The Stage 1 Task 3 feature-stacker is the right place to assert this at ingest. Diagnostic recipe in the `CONVENTION-TO-VERIFY` comment in `src/hecras_mesh_ai/features/aspect.py`.

## Recently closed

- **Bald Eagle Dam Break opens cleanly with rashdf** (Stage 1 Task 1, 2026-05-23). Canonical pilot = `g09.hdf`. See `notebooks/02_baldeagle_explore.ipynb`.
- **CRS embedding behavior characterized** (Stage 1 Task 1, 2026-05-23). Variable across HDFs; pipeline needs prefer-HDF-fall-back-terrain resolver. Both Muncie and 11/12 Bald Eagle geometries fall back to terrain; `g09` is the HDF-embedded case.
- **`rashdf.refinement_regions()` empty-frame bug** — no longer surfaces under 0.12.0; returns `len() == 0` cleanly on all 12 Bald Eagle geometries and Muncie.

**Build-plan Stage 2 — Breakline Model Pilot (complete, 2026-05-23).** All eight tasks landed across ~13 commits:

| Task | Module | Tests |
|---|---|---|
| 1 — Lightning DataModule | `src/hecras_mesh_ai/model/datamodule.py` | 6 |
| 2 — BCE + Dice loss | `src/hecras_mesh_ai/model/loss.py` | 13 |
| 1b — BreaklineUNet LightningModule (smp U-Net + ResNet-18 + ImageNet) | `src/hecras_mesh_ai/model/unet.py` | 10 |
| 3 — Training script with CSVLogger | `scripts/train_pilot.py` | (smoke) |
| 4 — Overfit sanity check (BCE 0.79→0.024, Dice 0.98→0.35) | commit `5c01a48` | runs on GPU |
| 5 — Full pilot training (30 epochs, ~17 min on RTX 3090) | `lightning_logs/pilot/` | live |
| 6 — Probability-to-polylines post-processing | `src/hecras_mesh_ai/postprocess/breaklines.py` | 8 |
| 7 — Buffered IoU/F1 metrics + sliding-window inference | `src/hecras_mesh_ai/postprocess/{metrics,inference}.py` | 16 |
| 8 — Results notebook | `notebooks/04_stage2_pilot_results.ipynb` | runs end-to-end |

**Stage 2 final pilot results** (deliberately-overfit per build plan):

- Train (Bald Eagle g09, model SAW): precision 0.21, **recall 0.79**, F1 0.33, IoU 0.20 against expert breaklines
- Val (Muncie, UNSEEN): precision/recall/F1/IoU = 0 — model fired 784 predicted positive pixels but none within 3 pixels of any actual Muncie breakline

Honest read: the model learned Bald Eagle's specific breakline morphology (~80% recall on the project it saw) but overfit hard to it, with zero transfer to Muncie. **This is exactly what the build plan predicted** — Stage 2's purpose was pipeline validation, not generalization; Stage 3 + bulk corpus is what makes the model actually learn what makes a breakline universally.

**Bugs caught (and fixed) during Stage 2 — all caught by the deliberately-overfit pilot doing its job:**

1. **Tile sampler accepted NaN-containing tiles** (Stage 1 design oversight) — fixed via `IterableTileDataset(skip_nan_tiles=True)` with explicit retry loop. Without this, the very first val pass crashed to NaN loss.
2. **Random sampling can't learn rare-positive classes** — at ~0.005% positive rate, BCE alone converges to "predict 0 everywhere." Fixed via positive-CENTERED sampling (`RasterTileDataset.positive_pixel_rowcol` + `RandomTileSampler.next_positive_centered_bbox`), gated by `positive_fraction`.
3. **Checkpoint monitor=val/total_loss selected a degenerate "predict zero" model** — val loss is dominated by trivial empty-tile success, so "best by val" picked the most-zero-predicting epoch. Fixed by monitoring `train/total_loss_epoch` and adding `save_last=True`. Saved checkpoint after first training run was loadable but predicted nothing on any input; second training run with the fix produced the real results above.

**Build-plan Stage 3 — Bulk corpus access strategy decided (2026-05-24).** Detailed in `docs/build-plan/03-breakline-model-scale.md` and **ADR 013**:

- **Phase 3A — NOAA OWP S3** (`s3://noaa-nws-owp-fim/ras2fim`). Primary. Bulk HEC-RAS models with terrain, geometry, and (in most cases) breaklines. Requires ESIP credentials. **Status: email sent 2026-05-24, awaiting reply.**
- **Phase 3B — FEMA BLE / InFRM corpus.** Secondary. Different access path (probably an InFRM-team request). Engaged only if 3A produces insufficient diversity.
- **Phase 3C — FEMA MIP corpus.** Deferred. Discovery probes confirmed no public REST endpoint; estBFE backend doesn't expose downloads; ScienceBase has no InFRM BLE collection; even NOAA OWP's `ras2fim` requires locally-downloaded models.

When NOAA credentials arrive: pause Stage 5 (if active) and build `src/hecras_mesh_ai/corpus/noaa_owp.py` — an `aws s3 sync` wrapper plus a `hecstac` STAC catalog inventory.

**Build-plan Stage 4 — HEC-RAS engineering harness (complete, 2026-05-25).** Built in parallel with the Stage 3 wait. All four tasks landed across 6 commits; full closed loop now works end-to-end against HEC-RAS 7.0:

| Task | Module | Tests |
|---|---|---|
| 1 — HDF inspector + schema notes | `src/hecras_mesh_ai/harness/inspect.py`, `docs/hdf-schema/` | 6 |
| 2 — Breakline-replacement writer | `src/hecras_mesh_ai/harness/write_geom.py` | 14 (incl. bit-for-bit round-trip on real Muncie HDF) |
| 3 — Plan launcher (COM primary, CLI fallback) | `src/hecras_mesh_ai/harness/launch.py` | 17 (16 unit + 1 slow integration via live HEC-RAS) |
| 4 — 2D unsteady results parser | `src/hecras_mesh_ai/harness/results.py` | 10 (incl. integration on real Muncie p04.hdf) |

**Stage 4 verification end-to-end:**

- `scripts/verify_writer_muncie.py` produces a HEC-RAS-openable round-trip copy of the Muncie project. Manually verified to open cleanly in HEC-RAS 7.0 with both breaklines (`Road 1`, `HighGround 1`) rendering identically to the pristine sample.
- `pytest -m slow tests/harness/test_launch.py` actually launches HEC-RAS via COM, computes Muncie's plan p04, and produces a 19 MB results HDF — all in ~30 seconds. Mesh regenerated (5765 cells vs 5391 input cells), all derived data populated.
- `max_water_surface()`, `max_depth()`, `max_face_velocity()` all parse the real results HDF, with active-cell masking (HEC-RAS appends ~374 ghost cells past the active mesh — caught by integration test, masked via `Cells Minimum Elevation == NaN`).

**The closed loop now works:** a single Python call sequence — `replace_breaklines()` → `run_plan()` → `max_water_surface()` — produces a quantitative simulation readout with no human in the loop. This is exactly the engineering substrate Stage 5 wires the ML model output through.

**New dependency:** `pywin32>=306` added under a Windows-only `harness` optional extra (`uv sync --extra harness`).

## Next: Build-plan Stage 5 — ML × Harness wiring

The natural next milestone. Take the Stage 2 model's predicted breakline polylines (from `probability_to_polylines`), pump them into `replace_breaklines()` → `run_plan()` → `max_water_surface()`, and close the full ML-to-simulation-to-score loop on a known plan. All pieces exist; this is glue + an end-to-end integration test.

(If NOAA credentials arrive before Stage 5 completes, pause Stage 5 and execute Phase 3A NOAA OWP downloader first — the bulk corpus unlocks the real generalization story.)
