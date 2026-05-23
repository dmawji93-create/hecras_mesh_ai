# ADR 007: "Make-One-Work" Sprint Approach

**Status:** Accepted
**Date:** 2026-05-21

## Context

The full Phase A pipeline has many moving parts: terrain ingestion, ancillary data fetching, feature engineering, label rasterization, spatial tiling, model definition, training loop, post-processing (skeletonize → vectorize → simplify), and evaluation. Building this at scale on bulk FEMA data from day one would mean discovering every I/O gotcha, every CRS mismatch, every memory issue under the worst possible conditions, with long iteration cycles.

The classic engineering principle: **make one thing work end-to-end before making many things work.**

## Decision

Phase A starts with a **4-week single-project end-to-end sprint** on the HEC-RAS Muncie example (and a secondary example, Bald Eagle Dam Break). The sprint runs the entire pipeline at small scale, deliberately accepting that the model will overfit. The deliverable is a working pipeline, not a useful model.

Only after the pipeline is validated do we scale to bulk FEMA data (Phase A.1).

## Sprint structure

- **Week 1 — Plumbing:** repo, env, data ingest, exploration notebook
- **Week 2 — Features and labels:** DEM derivatives, breakline rasterization, spatial tiling
- **Week 3 — Model:** Lightning DataModule, smp U-Net, training loop, W&B logging
- **Week 4 — Post-process and reflect:** vectorize, evaluate, retrospective

Detailed task lists in `docs/roadmap.md` (Phase A.0).

## Consequences

### Positive
- Surfaces every I/O and tooling gotcha early, when iteration is cheap.
- Gives the user (new to ML) a concrete, end-to-end mental model of how all the pieces fit before complexity ramps up.
- Produces a demoable deliverable in 4 weeks rather than 4 months.
- Builds the scaffolding (repo, modules, tests, logging) that everything else hangs on.
- Forces early decisions on data formats, naming conventions, and module boundaries when changing them is still cheap.

### Negative / risks
- Pipeline built for N=2 may need restructuring when scaled to N=100+ (deferred refactoring).
- Tempting to over-engineer for the bulk case during the pilot. Resist this — the sprint is about validation, not generality.
- Sprint timeline may slip due to learning curve; that is acceptable but should be tracked.

## References

- `docs/roadmap.md` Phase A.0
- `docs/decisions/006-pilot-dataset.md`
