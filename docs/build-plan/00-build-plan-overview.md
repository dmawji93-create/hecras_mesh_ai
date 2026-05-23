# Build Plan — Overview

This directory is the **executable, checkpointed plan** for building `hecras_mesh_ai`. Where `docs/roadmap.md` gives the strategic phases, this directory is the operational source of truth for execution. One file per stage; each stage has an explicit **checkpoint** that must pass before the next stage begins.

## How to use this directory

- Work stages **in order**. Do not start stage N+1 until stage N's checkpoint criteria are all met.
- Each stage file carries a `Status` field — keep it current (`Not started` / `In progress` / `Blocked` / `Complete`).
- When a checkpoint passes, record how it was verified (a commit hash, a notebook, a metrics screenshot) in that stage file before moving on.
- The plan is long-horizon. Crucially, it is structured so that **partial completion is still a usable product**: stopping after Stage 3 yields a breakline tool; after Stage 5, a static-mesh tool; after Stage 7, an adaptive-mesh tool. Every stage delivers verifiable value.

## Target deployment workflow

The product the stages assemble toward:

```
terrain  ──►  [ML warm-start mesh]  ──►  HEC-RAS run  ──►  [error vector vs Richardson reference]
                                                                      │
                            within user tolerance vector? ── yes ──►  done
                                                                      │
                                                                      no
                                                                      │
                                              [adaptive refinement of high-error regions]
                                                                      │
                                                                      └──►  re-run  ──►  (loop)
```

Two product tiers fall out of this:

- **Quick tier** — the ML warm-start mesh alone (Stage 5 output, or just breaklines from Stage 3). Terrain in, runnable expert-quality mesh out, in seconds. Imperfect but an excellent starting point. Many practitioners will be happy to stop here.
- **Optimal tier** — warm-start mesh plus the adaptive refinement loop driven by a user-defined tolerance vector (Stage 7, later accelerated by Stage 8). Slower, but converges to optimal resolution per unit compute for the user's declared purpose.

## What is ML and what is not

Per the project decision that the **ML component is the main focus and the differentiating value**, but with the honest caveat that the non-ML stages are load-bearing and must not be under-resourced:

| Stage | Type |
|---|---|
| 1 Feature & label pipeline | ML (data engineering) |
| 2 Breakline model — pilot | ML |
| 3 Breakline model — scale | ML |
| 4 HEC-RAS automation harness | Engineering |
| 5 Resolution model + static mesh | ML + Engineering |
| 6 Mesh quality measurement framework | Numerical methods |
| 7 Classical adaptive refinement loop | Numerical methods |
| 8 ML optimization layer | ML |

The real ML novelty lives in the warm-start (Stages 2-3) and resolution prediction (Stage 5). Stage 7 is textbook adaptive mesh refinement applied to a domain where it has not previously been applied — numerical methods, not research. Stage 8 is ML as an optimization layer on top of a working classical loop. **Stage 4 (the HEC-RAS automation harness) is unglamorous but it is a hard dependency for Stages 5-8 and for all reference-solution data generation — treat it as first-class engineering, not a side task.**

## Honest note on cost

Model training is not the bottleneck — the RTX 3090 handles that. The bottleneck and main cost center is **data generation**: gathering and quality-filtering the expert-mesh corpus (Stage 3) and, more expensively, computing Richardson-extrapolation reference solutions (Stage 6), each of which is a sequence of HEC-RAS runs. Budget for data generation as the main sustained effort of the project.

## Relationship to ADRs

- ADR 001 — staged A→B→C delivery
- ADR 003 (amended) — expert meshes are a prior/warm-start, not the target
- ADR 011 — quantitative mesh-quality objective (reference solution, error functional, cost term)
- ADR 012 — the refinement loop is numerical-methods-first; ML (Stage 8) is an optional optimization layer on a working classical loop (Stage 7)

## Stage list

1. `01-feature-and-label-pipeline.md`
2. `02-breakline-model-pilot.md`
3. `03-breakline-model-scale.md`
4. `04-hecras-automation-harness.md`
5. `05-resolution-model-static-mesh.md`
6. `06-mesh-quality-framework.md`
7. `07-classical-refinement-loop.md`
8. `08-ml-optimization-layer.md`
9. `09-deferred-and-future.md` — parking lot; not part of v1

## Claude Code kickoff

> Read `CLAUDE.md`, then every file in `docs/build-plan/` in order, then the ADRs referenced above. Confirm what you understand and flag anything ambiguous. Week 1 plumbing is already complete (see `STATUS.md`). Begin with **Stage 1 — Feature & Label Pipeline**. Do not proceed past a stage's checkpoint until every exit criterion is met and verified. Teach as you go; I am new to ML.
