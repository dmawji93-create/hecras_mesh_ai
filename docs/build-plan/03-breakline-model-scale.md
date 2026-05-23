# Stage 3 — Breakline Model (Scale)

**Type:** ML
**Status:** Not started
**Depends on:** Stage 2
**Maps to:** roadmap Phase A.1 / A.2

## Objective

Take the pilot pipeline and make it generalize: build a large, diverse, quality-filtered training corpus from FEMA flood studies, integrate ancillary data layers, retrain, and validate on held-out unseen projects. This stage delivers the first genuinely useful product — a breakline tool that works on terrain it has never seen.

## Scope

### In scope
- Bulk corpus acquisition from FEMA NFHL / MIP flood studies — download, inventory, deduplicate.
- Quality-filtering pipeline (mesh size, breakline density, run completion, etc.) — see ADR 004.
- Ancillary data integration: NHD streams, NLD levees, road centerlines, land cover — as additional feature channels.
- A hand-curated held-out validation set (~10-20 diverse, high-quality projects spanning rural riverine, leveed systems, urban, mountain, dam-breach).
- Retraining on the bulk corpus; robustness across DEM resolution and CRS.
- Graceful degradation when an ancillary layer is missing.
- Geopackage export of predicted breaklines, importable into RAS Mapper — this is the Quick-tier deliverable for breaklines.

### Out of scope (deferred)
- Resolution / refinement-region prediction (Stage 5).
- HEC-RAS runs and the automation harness (Stage 4).

## Tasks

1. Build the FEMA study acquisition + inventory pipeline.
2. Implement quality filters; manually spot-check their decisions.
3. Integrate ancillary data layers as feature channels.
4. Curate and document the held-out validation set.
5. Retrain on the bulk corpus; tune.
6. Evaluate on held-out projects; analyze failure modes.
7. Implement and test geopackage export.

## Checkpoint — exit criteria

- [ ] Bulk corpus assembled, deduplicated, quality-filtered, and documented.
- [ ] Held-out validation set curated and confirmed to have **zero overlap** with training data.
- [ ] Model meets an agreed metric bar on held-out unseen projects — *the bar itself must be agreed with the project owner before this stage starts.*
- [ ] Geopackage export produces files that import cleanly into RAS Mapper.
- [ ] Runtime and memory for inference on a full project are within an agreed budget.
- [ ] `pytest` green; pre-commit clean.

## Notes & risks

- FEMA studies are geographically and morphologically biased (mostly populated US, mostly riverine). The model inherits this — document it.
- Quality filtering is itself imperfect; some bad meshes will leak into training, some good ones will be rejected. Iterate.
- This stage gathers expert meshes by **reading** HDFs only — no HEC-RAS runs. The expensive run-based data generation begins at Stage 6.
- Agree the held-out metric bar up front; without it, "done" is undefined.
