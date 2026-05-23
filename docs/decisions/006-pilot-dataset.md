# ADR 006: Pilot Dataset — HEC-RAS Official Examples

**Status:** Accepted
**Date:** 2026-05-21

## Context

The "make-one-work" sprint (see ADR 007) needs a pilot dataset that lets us validate the entire pipeline end-to-end at small scale before investing in bulk FEMA data acquisition. Requirements:

- Has expert-placed breaklines and refinement regions as labels
- Small enough to iterate fast
- Well-documented so we can reproduce expected results
- Diverse enough across at least 2–3 examples to test generalization
- Freely available

## Decision

Use the **HEC-RAS official example projects** as the pilot dataset. Primary: **Muncie** (White River through Muncie, IN). Secondary: **Bald Eagle Dam Break** for a different morphology. Additional examples as needed.

## Why these specifically

- **Muncie** is the canonical 2D example used in essentially every HEC-RAS training course. It has:
  - A defined 2D flow area with multiple expert-placed breaklines
  - Refinement regions along the main channel
  - A bridge / hydraulic structure
  - Real-world terrain at usable resolution
  - It is the dataset that `rashdf` is documented against (their README example loads `Muncie.g04.hdf`)
- **Bald Eagle Dam Break** provides a different morphology — dam-breach wave propagation downstream through a constrained valley — which begins to exercise generalization.

## Consequences

### Positive
- Available immediately, no data acquisition lag.
- Well-known in the HEC-RAS community — easy to compare results against published tutorials.
- `rashdf`-compatible out of the box.
- Iteration cycle is fast (small mesh, fast load, fast plot).
- Reproducible: anyone can install HEC-RAS, get the same data, and verify our results.

### Negative / risks
- N=2 is not a training set; the pilot exists only to validate the pipeline.
- The model will completely overfit on these examples. That is intentional and expected — overfitting on the sanity-check tiles is part of the Week 3 plan.
- Examples are riverine; corpus diversity comes only at Phase A.1 (FEMA bulk).

## Open questions

- Which other HEC-RAS official examples to include in the pilot beyond Muncie and Bald Eagle Dam Break (BeaverLake, others)?

## References

- HEC-RAS example datasets ship with the application; available on the HEC website
- rashdf README and tutorials use Muncie: https://github.com/fema-ffrd/rashdf
- `docs/roadmap.md` Phase A.0
