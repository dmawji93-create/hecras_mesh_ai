# 09 — Deferred & Future Work

**Status:** Parking lot — explicitly **not part of v1**

This file holds ideas that are real and valued but deliberately off the critical path. Nothing here is started until Stages 1-8 are complete. The purpose of writing them down is so they are not forgotten — and so they are not allowed to creep into v1 scope.

---

## Terrain-to-hydraulics instant baseline tool (neural surrogate)

**Concept.** A model that emulates the HEC-RAS solver — taking terrain plus forcing and producing an approximate hydraulic result directly, with no mesh and no simulation. A "HAND, but learned and considerably better" rapid screening tool.

**Status:** To be built **after the entire mesh-generation product (Stages 1-8) is complete.** Come back to this then.

**Honest scope when revisited:**
- Input must be **terrain + forcing** (discharge, boundary conditions, roughness), not terrain alone — the same terrain floods differently under different forcing.
- Target **reduced outputs** — peak depth, inundation extent, time-to-peak. Not the full unsteady solution.
- Peak depth / inundation extent: feasible and useful. Velocity: rough at best. Full solver replacement: not realistic.
- It is a **fast approximate screening tool, not a certified result**, and must always be presented that way — a hallucinated flood map used for a real decision is a genuine harm.
- It is a **separate model with its own validation regime** — not part of the meshing pipeline.

**Synergy worth remembering:** it needs the same training data as the meshing tool (HEC-RAS runs pairing terrain, forcing, solution). A working surrogate could later feed the meshing tool — supplying cheap predicted hydraulics for Stage 5, or a cheap proxy reward for Stage 7/8 that reduces the number of expensive reference runs. Force multiplier, but only once the core product exists.

---

## Alternative reference-solution methods

Stage 6 ships with **Richardson extrapolation only**. These alternatives are noted for a future release:

- **Ultra-fine single mesh** — a single very-high-resolution run as the reference. Conceptually simple, very expensive (overnight to multi-day runtimes).
- **Self-convergence / local stopping** — define "converged" operationally as where further refinement stops moving the solution beyond tolerance; no global truth run required. Dovetails with the progressive tolerance workflow.
- **Physical-observation anchoring** — stream gauges, high-water marks, satellite (SAR) inundation extents. Physical truth, but sparse and event-specific.
- **Analytical benchmark cases** — exact solutions for idealized geometries (Ritter/Stoker dam break, Thacker parabolic bowl). Used in Stage 6 for *validating* the metric framework; could be extended.

Adding a method-selection option (let the user choose the reference basis) is a natural post-v1 feature.

---

## Other future items

- **HEC-RAS plugin / RAS Mapper integration** — native integration rather than a standalone tool. Depends on USACE plugin architecture.
- **Goal-oriented / adjoint-based error indicators** — rigorous dual-weighted-residual estimation that targets the specific output quantity the user cares about. Sophisticated; HEC-RAS does not expose an adjoint, so this likely means an ML-approximated indicator.
- **Deployment as a hosted service** vs local desktop tool — a product/infrastructure decision for after v1.
- **Domain-specific model variants** — if the general-purpose model underperforms on urban or coastal applications (see ADR 002), specialized variants may be warranted.
