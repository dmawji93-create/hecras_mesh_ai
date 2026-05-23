# ADR 001: Staged A → B → C Delivery

**Status:** Accepted
**Date:** 2026-05-21

## Context

"Automate HEC-RAS 2D mesh generation with AI/ML" is not one problem — it is at least four nested ones:

1. Where to place breaklines (geometric / topographic problem)
2. Where to place refinement regions and what local cell sizes to use (hydraulically driven)
3. Predicting hydraulic behavior without running the model (surrogate modeling)
4. Closing the loop — mesh → run → assess → refine

Each sub-problem has different tractability, data requirements, and engineering effort. We need to pick where to start and how to sequence the rest.

## Decision

Stage delivery as **A → B → C** with the option to introduce a hydraulic surrogate later if Phase C reward shaping demands it:

- **Phase A:** Breakline detection (supervised CV from terrain + ancillary data).
- **Phase B:** Add a learned resolution field and refinement region generator; output complete static meshes.
- **Phase C:** Close the loop — solution-based adaptive refinement, fine-tuned against run performance.

Phase A and Phase B share an encoder backbone — the resolution head is grafted onto the Phase A model rather than starting from scratch.

## Consequences

### Positive
- Shippable v0.5 deliverable in weeks (breaklines as a geopackage), not months.
- Each phase de-risks the next: A validates data pipeline and feature engineering; B forces the HDF5 write engineering; C builds on a known-good static-mesh baseline.
- Maps cleanly onto the chosen training strategy (supervised pretrain in A and B → performance fine-tune in C).
- Phase A output is independently useful even if the project never reaches C.

### Negative / risks
- Phase A alone is a modest value-add to a working modeler (resolution decisions are where most iteration time lives).
- A breakline model that ignores hydraulics may miss flow-alignment breaklines that exist for non-topographic reasons.
- Tight coupling between breaklines and refinement region perimeters means the Phase B joint prediction is harder than either component alone.

## Alternatives considered

- **B directly:** Skip the A-only milestone, ship complete static meshes first. Rejected — too much engineering surface area at once (HDF writes + multi-task model + joint prediction), no early validation of feature pipeline.
- **A only:** Stop after breaklines. Rejected — ceiling on user value is too low to justify the project.
- **Closed-loop adaptive (C) from the start:** Rejected — each RL training step requires a full HEC-RAS run; infeasible without a static-mesh baseline to fine-tune from.
- **Hydraulic surrogate first:** Build a neural SWE solver, then use it to drive mesh decisions. Rejected — building a generalizable shallow-water surrogate is an open research problem and inverts the stated supervised-pretrain training strategy.

## References

- `docs/roadmap.md`
- `docs/decisions/003-training-strategy.md`
