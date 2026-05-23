# Roadmap

Living document. Update as phases complete or pivot. Significant pivots get a corresponding ADR in `docs/decisions/`.

---

## Phase A — Breakline Detection

**Goal:** Given a DEM + ancillary geospatial data over a 2D flow area, predict expert-quality breakline polylines that align cell faces with hydraulically critical linear features (ridges, levees, road embankments, channel banks).

**Deliverable:** A model + post-processing pipeline that emits a geopackage of breaklines importable into HEC-RAS RAS Mapper.

**Why first:** Pure topographic CV problem with strong signal from DEM derivatives. Labels are abundant (every FEMA study has expert breaklines). Sidesteps the unsolved engineering of programmatic geometry HDF writes — emit a geopackage and import via the existing UI.

### A.0 — Make-one-work sprint (4 weeks)

Single-project pilot on the HEC-RAS Muncie example, then Bald Eagle Dam Break. Goal: validate the entire pipeline end-to-end at small scale before investing in bulk data.

**Week 1 — Plumbing**
- src-layout repo (`hecras_mesh_ai`), `uv` environment, pytest + ruff + pre-commit
- Install full stack (PyTorch, Lightning, smp, TorchGeo, rasterio, geopandas, rioxarray, rashdf, h5py, wandb)
- Download Muncie + Bald Eagle example datasets
- Exploration notebook: open `Muncie.g04.hdf` via rashdf, plot mesh, breaklines, perimeter, refinement regions over the terrain. Verify we can read everything we need.

**Week 2 — Features + labels**
- DEM derivatives module: slope, aspect, plan/profile curvature, TWI, flow accumulation, hydro-conditioned ridges. Each as a tested function.
- Feature stacking into multi-channel xarray DataArrays
- Breakline rasterization with configurable buffer width
- TorchGeo-based train/val tile split with **spatial holdout** (not random — critical for geo data)

**Week 3 — Model**
- Lightning DataModule wrapping the TorchGeo dataset
- LightningModule with a small smp U-Net (ResNet-18/34 encoder, ImageNet init)
- Loss: BCE + Dice (handles severe class imbalance)
- Sanity-check training: intentionally overfit a handful of tiles
- W&B logging from day one

**Week 4 — Post-process + reflect**
- Probability threshold sweep
- Skeletonize → vectorize → simplify (Douglas-Peucker) → smooth → polylines
- Visual overlay of predicted vs expert breaklines
- Buffer-based IoU and F1 metrics
- Retrospective: what surprised us, what changes before scaling

### A.1 — Scale to bulk

- Inventory and download FEMA NFHL/MIP flood study HDFs at scale
- Build a deduplication and quality-filter pipeline (some FEMA studies are excellent, others are not)
- Curate a held-out validation set of ~10–20 high-quality projects across geographies and morphologies
- Retrain the Phase A model on the bulk corpus
- Evaluate on held-out projects

### A.2 — Hardening

- Robustness across DEM resolutions and CRS
- Sensitivity to ancillary data availability (graceful degradation when NLD or land cover is missing)
- Performance budgeting: end-to-end runtime, memory, GPU footprint
- API design: how a modeler invokes this on their own project

---

## Phase B — Resolution Field + Complete Static Mesh

**Goal:** Predict a continuous "nominal cell size" field across the domain, cluster into refinement region polygons, and combine with Phase A breaklines to produce a complete HEC-RAS geometry.

**New engineering work:**
- Geometry HDF5 writer (extending `rashdf` or building parallel writer)
- Automated end-to-end pipeline: terrain → features → breaklines → resolution field → refinement regions → computation points → write `.gNN.hdf` → ready to run in HEC-RAS
- Cell budget as a constraint (target total cell count)

**Modeling:**
- Multi-task head on the shared encoder from Phase A (breaklines + resolution as joint outputs)
- Labels from expert refinement regions in the bulk corpus
- Optional: feed Phase A breakline predictions as an input channel to the resolution head

---

## Phase C — Closed-Loop Adaptive Refinement

**Goal:** Mesh → run HEC-RAS → assess solution-based error indicators → refine → re-run. Optimize for accuracy under a runtime budget.

**Dependencies:** A working Phase B static mesh as the starting point. Without this, Phase C is starting from random.

**Approach options (decide closer to time):**
- Rule-based mesh adaptation driven by gradient / Froude / mass residual indicators
- RL policy fine-tuned on top of the Phase B static mesh model
- Bayesian optimization over refinement region parameters

**Reward / loss signal:** Composite of accuracy vs fine-mesh reference (or vs gauge data) and runtime / cell count. Exact form TBD.

---

## Phase D — Productionization

- Standalone CLI tool
- HEC-RAS plugin / RAS Mapper integration (long-term, depends on USACE plugin architecture)
- Validation suite on held-out projects with documented performance metrics
- Deployment: local desktop tool vs hosted service
- Documentation, examples, tutorials
