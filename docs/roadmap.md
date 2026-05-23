# Roadmap

Living document — the **strategic** view of the project. The **executable, checkpointed plan** lives in `docs/build-plan/`; that directory is the operational source of truth for day-to-day work. Significant pivots get a corresponding ADR in `docs/decisions/`.

**Status as of 2026-05-23:** Phase A.0 Week 1 (plumbing) complete. Next: build-plan **Stage 1 — Feature & Label Pipeline**.

**How the phases map to the build plan:**

| Roadmap phase | Build-plan stages |
|---|---|
| Phase A — Breakline Detection | Stages 1-3 |
| Phase B — Resolution Field + Static Mesh | Stages 4-5 |
| Phase C — Adaptive Refinement | Stages 6-8 |
| Phase D — Productionization & Future | build-plan `09-deferred-and-future.md` |

---

## Phase A — Breakline Detection

**Goal:** Given a DEM + ancillary geospatial data over a 2D flow area, predict expert-quality breakline polylines that align cell faces with hydraulically critical linear features (ridges, levees, road embankments, channel banks).

**Deliverable:** A model + post-processing pipeline that emits a geopackage of breaklines importable into HEC-RAS RAS Mapper.

**Why first:** Pure topographic CV problem with strong signal from DEM derivatives. Labels are abundant (every FEMA study has expert breaklines). Sidesteps the unsolved engineering of programmatic geometry HDF writes — emit a geopackage and import via the existing UI.

- **A.0 — Make-one-work sprint.** Single-project pilot on Muncie, then Bald Eagle Dam Break — validate the whole pipeline end-to-end at small scale before investing in bulk data. **Week 1 (plumbing) is complete.** Remaining A.0 work is detailed as build-plan **Stage 1** (feature & label pipeline) and **Stage 2** (breakline model pilot).
- **A.1 / A.2 — Scale and harden.** Bulk FEMA NFHL/MIP corpus, deduplication and quality filtering, a curated held-out validation set, retraining, robustness across DEM resolution/CRS, ancillary-data graceful degradation. Detailed as build-plan **Stage 3**.

Per ADR 003 (amended), expert meshes are used as a **prior / warm start**, not as the final target.

---

## Phase B — Resolution Field + Complete Static Mesh

**Goal:** Predict a continuous nominal-cell-size field across the domain, convert it into refinement region polygons, and combine with Phase A breaklines to produce a complete, runnable HEC-RAS geometry. Delivers the **Quick-tier product**: terrain in, expert-quality runnable mesh out, in seconds.

**Engineering:** The HEC-RAS automation harness — programmatic geometry-HDF5 writing, run launching, results parsing — is the enabling prerequisite. Detailed as build-plan **Stage 4** (harness) and **Stage 5** (resolution model + static-mesh assembly).

**Modeling:** A resolution-field head multi-task on the shared Phase A encoder. Resolution is hydraulically driven, so the head's inputs include a-priori-knowable hydraulic context — boundary conditions, the Manning's *n* field, structures — not terrain alone. Labels from expert refinement regions; a total cell-budget acts as a constraint.

---

## Phase C — Adaptive Refinement Against a Quantitative Objective

**Goal:** Take the Phase B warm-start mesh, run it, measure error against a quantitative objective, refine the high-error regions, and repeat until the user's tolerance vector is met — converging to optimal resolution per unit compute. Delivers the **Optimal-tier product**.

This phase is defined by two ADRs:

- **ADR 011 — Quantitative Mesh-Quality Objective.** Mesh quality is made measurable: a converged **reference solution** (v1: Richardson extrapolation; alternatives deferred), a **purpose-dependent error functional** (a vector of metrics — peak WSE, inundation extent, timing, velocity — each with a user-defined tolerance), and a **cost term**. Built as build-plan **Stage 6**.
- **ADR 012 — Numerical-methods-first.** The refinement loop is classical adaptive mesh refinement — a mature, solved method — built first with no ML (build-plan **Stage 7**). ML is an **optional optimization layer** added afterward (build-plan **Stage 8**) to reduce iteration count and enable richer, anisotropic, breakline-aware refinement actions.

The loop uses a **progressive tolerance schedule**: start coarse and wide, get a cheap approximate solution, refine, tighten, repeat until the declared purpose is met.

---

## Phase D — Productionization & Future

- Standalone CLI tool; HEC-RAS plugin / RAS Mapper integration (long-term, depends on USACE plugin architecture).
- Validation suite on held-out projects with documented performance metrics.
- Deployment: local desktop tool vs hosted service.
- Documentation, examples, tutorials.

**Deferred / off the critical path** (see `docs/build-plan/09-deferred-and-future.md`):

- **Terrain-to-hydraulics instant baseline tool** — a neural surrogate predicting approximate hydraulics from terrain + forcing, as a rapid screening tool. To be revisited only after the core mesh-generation product (Stages 1-8) is complete.
- **Alternative reference-solution methods** — ultra-fine single mesh, self-convergence / local stopping, physical-observation anchoring, analytical benchmark cases. v1 ships Richardson extrapolation only.
