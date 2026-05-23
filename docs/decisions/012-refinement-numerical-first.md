# ADR 012: Refinement Loop is Numerical-Methods-First; ML is an Optional Optimization Layer

**Status:** Accepted
**Date:** 2026-05-23

## Context

Phase C closes the loop: mesh → run → assess → refine → re-run. A natural assumption is that "AI generates the mesh" implies machine learning drives every part of this loop, including the iterative refinement step.

That assumption is wrong, and getting it right materially de-risks the project. The iterative refinement step — solve on the current mesh, estimate per-cell error, refine where error is high, repeat — is **classical adaptive mesh refinement (AMR)**. A posteriori error estimation and h-refinement have been standard practice in computational fluid dynamics and finite-element analysis for 30-40 years. The algorithm is textbook; it provably converges under reasonable conditions. HEC-RAS does not do it automatically, but that is a missing *application*, not a missing *method*.

ML is therefore not a prerequisite for the refinement loop. It is an optional optimization layer added on top of a loop that already works.

## Decision

Build the adaptive refinement loop in two layers, in order:

1. **Classical layer (numerical methods, no ML — build-plan Stage 7).** A rule-based AMR loop: a posteriori error indicators (gradient/curvature of water surface and velocity, PDE residual, Richardson-based local estimates), cell flagging against a tolerance schedule, refinement actions, progressive tolerance tightening, convergence safeguards. This is the guaranteed-working baseline and is itself a shippable Optimal-tier product.

2. **ML optimization layer (optional — build-plan Stage 8).** Added only once the classical layer works. Two contributions: (a) **fewer iterations** — a model trained on classical refinement trajectories predicts the converged mesh directly, approximating the fixed point of the iteration; (b) **richer refinement actions** — anisotropic, flow-aligned, breakline- and structure-aware refinement beyond classical isotropic cell-splitting. ML proposes; the loop always verifies the result by running it.

ML in the refinement loop is **upside, not a dependency**.

## Consequences

### Positive
- **De-risks Phase C decisively.** It is not "invent a novel RL system or the project fails." There is a textbook fallback. The project remains viable even if the ML layer is never built.
- The classical loop is a complete, shippable product on its own.
- The ML layer's training data is the by-product of running the classical loop — a clean, non-circular dependency (Stage 8 depends on Stage 7's trajectories).
- Clarifies honestly where ML adds value (warm start, resolution prediction, iteration reduction, anisotropic actions) versus where it would be gratuitous.

### Negative / risks
- The classical loop still depends on the HEC-RAS automation harness (build-plan Stage 4) and is engineering-heavy — "solved method" does not mean "no work."
- Each AMR iteration is a full HEC-RAS run; the classical loop may be slow. That cost is the motivation for the Stage 8 ML layer.
- The ML layer may underdeliver if refinement trajectories are too chaotic to predict — mitigated because Stage 7 stands alone.

## Alternatives considered

- **ML-driven refinement from the start (e.g. an RL policy):** Rejected — expensive, cold-start, and with no classical baseline to compare against or fall back to. Classical AMR is the proven approach; ML should optimize it, not replace it untested.
- **Pure classical, never add ML:** A viable permanent fallback, but it leaves iteration-count savings and anisotropic/breakline-aware refinement on the table. Hence ML as an optional layer rather than excluded entirely.
- **Goal-oriented / adjoint-based error indicators:** Rigorous and would sharpen the classical layer, but HEC-RAS exposes no adjoint. An ML-approximated goal-oriented indicator is a possible future enhancement — deferred (see `docs/build-plan/09-deferred-and-future.md`).

## References

- `docs/decisions/001-staged-a-b-c-delivery.md` (Phase C)
- `docs/decisions/011-mesh-quality-objective.md` (the objective the loop drives toward)
- `docs/build-plan/07-classical-refinement-loop.md`, `08-ml-optimization-layer.md`
- `docs/hec-ras-primer.md` (error indicators, numerical diffusion, Courant coupling)
