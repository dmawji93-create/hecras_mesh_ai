# Stage 7 — Classical Adaptive Refinement Loop

**Type:** Numerical methods
**Status:** Not started
**Depends on:** Stage 4, Stage 5, Stage 6
**Maps to:** roadmap Phase C (classical baseline)

## Objective

Build the iterative refinement loop that takes a warm-start mesh, runs it, measures error, refines the high-error regions, and repeats until the user's tolerance vector is met. This is **textbook adaptive mesh refinement** applied to HEC-RAS — rule-based, no ML. It delivers the **Optimal-tier product** in its classical form, and its run trajectories become the training data for Stage 8.

## Scope

### In scope
- **A posteriori error indicators** — per-cell, from the current solution: gradient and curvature of water surface and velocity, the PDE residual, and Richardson-based local error estimates.
- **Cell flagging** against a tolerance schedule.
- **Refinement action** — decrease local target cell size / insert or tighten a refinement region.
- **Tolerance-schedule workflow** — start with wide tolerances and a coarse mesh, get a cheap approximate solution, refine, progressively tighten tolerances until the user's purpose is met. (The progressive workflow from the project discussion.)
- **Loop orchestration** — warm-start mesh → run → indicators → refine → run → … → tolerance vector satisfied.
- **Convergence safeguards** — maximum iteration count, divergence detection, detection of features the coarse mesh missed entirely.

### Out of scope (deferred)
- ML acceleration of the loop (Stage 8).

## Tasks

1. Implement each a posteriori error indicator.
2. Implement cell flagging against the tolerance schedule.
3. Implement the refinement action (re-mesh with locally reduced cell size).
4. Implement the progressive tolerance schedule.
5. Implement loop orchestration over the Stage 4 harness and Stage 6 framework.
6. Implement convergence safeguards and divergence handling.
7. Persist every loop trajectory (intermediate meshes + solutions + final converged mesh) — this is Stage 8's training data.

## Checkpoint — exit criteria

- [ ] Starting from a **deliberately coarse** mesh, the loop converges to within a target tolerance vector on a test case — demonstrated.
- [ ] The progressive tolerance schedule works (tolerances widen/tighten as configured).
- [ ] The loop terminates correctly — on success, on max-iterations, and on divergence.
- [ ] Loop trajectories are persisted in a documented format for Stage 8.
- [ ] `pytest` green; pre-commit clean.

## Notes & risks

- **Coarse-start failure mode:** a first mesh too coarse to capture a feature's topology will never be told to refine there — you cannot refine your way out of a feature you never resolved. The Stage 5 warm start mitigates this; the loop should also actively detect missed features.
- Each loop iteration is a full HEC-RAS run — iteration count directly drives cost. Minimizing it is the motivation for Stage 8.
- This is the guaranteed-working baseline for the Optimal tier. ML (Stage 8) is an optimization on top — the product is viable even if Stage 8 is never built.
