# Stage 8 — ML Optimization Layer

**Type:** ML
**Status:** Not started
**Depends on:** Stage 7 (Stage 7 generates this stage's training data)
**Maps to:** roadmap Phase C (ML acceleration)

## Objective

Accelerate the classical refinement loop with machine learning. Two contributions: (1) reach the tolerance vector in **fewer iterations** by predicting the converged mesh directly, and (2) support **richer refinement actions** than classical isotropic cell-splitting. ML here is an optimization layer on a loop that already works — it is upside, not a dependency.

## Scope

### In scope
- **Iteration reduction.** Train a model on the classical refinement trajectories persisted in Stage 7 — mapping an early intermediate state (mesh + solution) to the final converged mesh. The model learns to approximate the fixed point of the refinement iteration and "jump ahead" several classical cycles.
- **Predict → verify → cleanup workflow.** The predicted jump mesh is always run and checked against the tolerance vector. If it overshoots or undershoots, fall back to classical refinement steps. ML proposes; the loop verifies.
- **Richer refinement actions.** A learned policy operating over an action space wider than isotropic splitting: anisotropic cells aligned with flow, breakline insertion, structure-aware refinement.

### Out of scope (deferred — see `09-deferred-and-future.md`)
- Goal-oriented / adjoint-based error indicators.
- Anything beyond accelerating and enriching the Stage 7 loop.

## Tasks

1. Assemble the training set from Stage 7's persisted trajectories.
2. Train the intermediate-state → converged-mesh model.
3. Implement the predict → verify → cleanup workflow with classical fallback.
4. Design and train the richer-action refinement policy.
5. Benchmark ML-augmented vs classical: iteration count, total compute, final accuracy.

## Checkpoint — exit criteria

- [ ] On held-out test cases, the ML-augmented loop reaches the **same tolerance vector** as the classical loop.
- [ ] It does so in **measurably fewer iterations / less total compute** — quantified against the Stage 7 baseline.
- [ ] **No loss of final accuracy** — the ML-accelerated result is within tolerance, verified, not just predicted.
- [ ] The classical fallback triggers correctly when a predicted jump misses.
- [ ] `pytest` green; pre-commit clean.

## Notes & risks

- The shortcut only works if converged meshes are reasonably predictable from early state. If refinement trajectories are chaotic, the model cannot reliably skip them — measure this before over-investing.
- ML never bypasses verification: every predicted mesh is run and checked. The workflow is "predict a big jump → verify → classical cleanup," not "ML replaces the loop."
- Stage 8 cannot start until Stage 7 has produced enough trajectories — the dependency is hard.
- If Stage 8 underdelivers, the Stage 7 classical loop remains a complete, shippable Optimal-tier product.
