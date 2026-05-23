# Stage 5 — Resolution Model + Complete Static Mesh

**Type:** ML + Engineering
**Status:** Not started
**Depends on:** Stage 3 and Stage 4
**Maps to:** roadmap Phase B

## Objective

Predict a continuous cell-size (resolution) field across the domain, convert it into refinement region polygons, and assemble a complete HEC-RAS geometry — breaklines plus refinement regions plus computation points — that runs. This stage delivers the **Quick-tier product**: terrain in, runnable expert-quality mesh out, in seconds.

## Scope

### In scope
- A resolution-field prediction head, multi-task on the shared encoder from Stages 2-3 (breaklines and resolution predicted jointly).
- Hydraulic-context inputs the resolution head needs and that are knowable a priori: boundary conditions, the Manning's *n* field, structures. (Per the project discussion — resolution is hydraulically driven, not purely topographic.)
- Refinement region polygon extraction from the predicted resolution field.
- A total cell-budget constraint.
- Assembly of a complete geometry via the Stage 4 writer.

### Out of scope (deferred)
- The adaptive refinement loop (Stage 7).
- Mesh quality scoring (Stage 6) — though Stage 6 can proceed in parallel.
- ML optimization of refinement (Stage 8).

## Tasks

1. Extend the model with a resolution-field head; confirm the shared encoder trains stably multi-task.
2. Add hydraulic-context input channels (BCs, Manning's *n*, structures).
3. Implement resolution-field → refinement-region polygon extraction.
4. Implement the cell-budget constraint.
5. Wire the full assembly: predictions → Stage 4 writer → `.gNN.hdf`.
6. Validate generated meshes run to completion in HEC-RAS.

## Checkpoint — exit criteria

- [ ] A fully ML-generated static mesh writes to `.gNN.hdf` and **runs to completion in HEC-RAS without errors**.
- [ ] The generated mesh respects the specified cell budget.
- [ ] The mesh is visually sane — refinement concentrates where expected, cells are non-degenerate.
- [ ] Generation time (terrain in → mesh out) is on the order of seconds.
- [ ] `pytest` green; pre-commit clean.

## Notes & risks

- The breakline and refinement-region outputs are coupled (a refinement region perimeter acts as a breakline; breaklines force local refinement). The joint multi-task model must respect HEC-RAS's mesh build order — see `docs/hec-ras-primer.md`.
- "Runs without error" is necessary but not sufficient for quality — quality is measured in Stage 6. Do not conflate "it runs" with "it is good."
- Hydraulic-context labels (expert refinement regions) are noisier than breakline labels — expert disagreement is real. This is expected; ADR 003 (amended) frames expert meshes as a prior, not ground truth.
