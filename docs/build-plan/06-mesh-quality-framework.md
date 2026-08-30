# Stage 6 — Mesh Quality Measurement Framework

**Type:** Numerical methods
**Status:** In progress (since 2026-05-26 — begun ahead of Stage 5 as the only unblocked path, per the "parallel with Stage 5" dependency note. Done: Thacker analytical benchmark, Task-7 groundwork built first deliberately (5c16fe8; deployment defects fixed 2026-08-24, 6c548cb). Next: wire Thacker into a HEC-RAS project; Task 1 grid-refinement runner.)
**Depends on:** Stage 4. Can be developed in parallel with Stage 5.
**Maps to:** ADR 011

## Objective

Make mesh quality measurable. Build the framework that, given a candidate mesh and its run, computes a quantitative error vector against a reference solution — turning "is this mesh good?" from tacit judgement into a number. This framework is both an internal optimization signal and a **user-facing output of the tool**.

## Scope

### In scope
- **Reference solution by Richardson extrapolation** — run a grid-refinement sequence, extrapolate to zero cell size, compute the Grid Convergence Index. This is the v1 reference method.
- **Error functional as a vector of metrics**, each measured against the reference:
  - peak water-surface error (spatial RMSE / max, optionally area-of-interest weighted)
  - inundation-extent error (Critical Success Index / IoU)
  - timing error (time-to-peak, flood-front arrival)
  - velocity-field error
  - mass-conservation residual (sanity / disqualification)
- **User-defined tolerance bands** per metric — the `(metric, tolerance, weight)` structure.
- **Cost term** — cell count (training signal) and wall-clock runtime (reporting).
- A mesh-quality **report** emitted as a standard output of every generation run.

### Out of scope (deferred — see `09-deferred-and-future.md`)
- Alternative reference methods: ultra-fine single mesh, self-convergence / local stopping, physical-observation anchoring, analytical benchmark cases. **v1 uses Richardson extrapolation only.** These are noted for a future release, not deployed now.

## Tasks

1. Implement the grid-refinement sequence runner (via the Stage 4 harness).
2. Implement Richardson extrapolation and the Grid Convergence Index.
3. Implement each error metric.
4. Implement the tolerance-vector data structure and acceptance check.
5. Implement the cost term.
6. Implement the mesh-quality report output.
7. **Validate the framework on an analytical benchmark** (e.g. dam break with a known Ritter/Stoker solution) — the metrics must reproduce analytical truth within expected order.

## Checkpoint — exit criteria

- [ ] Given any candidate mesh, its run, and a Richardson reference, the full error vector is computed.
- [ ] The framework is validated on at least one analytical benchmark case — computed error agrees with known truth.
- [ ] The tolerance-vector acceptance check works (mesh passes/fails correctly against configured bands).
- [ ] A mesh-quality report is emitted in a documented, stable format.
- [ ] `pytest` green; pre-commit clean.

## Notes & risks

- This stage is where the **expensive run-based data generation begins** — grid-refinement sequences are sequences of HEC-RAS runs. This is the project's main compute cost. Plan the campaign deliberately.
- SWE solutions with shocks, hydraulic jumps, and wet/dry fronts may not show clean monotone grid convergence — the GCI can misbehave. Handle non-convergent cases explicitly rather than silently.
- Validating on an analytical case first is non-negotiable: if the metric framework is wrong, every downstream optimization optimizes a lie.
