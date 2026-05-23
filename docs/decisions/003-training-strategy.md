# ADR 003: Training Strategy — Supervised Pretrain + Performance Fine-Tune

**Status:** Accepted
**Date:** 2026-05-21

## Context

Two broad paradigms exist for training the mesh model:

- **Supervised mimicry of expert meshes** — learn from curated projects with expert-placed breaklines and refinement regions. Cheap to train once labels are available, but inherits expert idiosyncrasies and cannot exceed expert quality.
- **Closed-loop performance optimization** — generate a mesh, run HEC-RAS, score the result, refine. Captures real hydraulic performance but each training step is expensive (a full simulation), and is hard to start from a cold initialization.

Each has serious tradeoffs alone. Together they compose: supervised pretrain gets the model into a reasonable region of weight space cheaply, then performance fine-tuning refines toward what actually matters (accuracy and runtime, not just expert mimicry).

## Decision

Use a **two-stage training strategy**:

1. **Supervised pretrain** (Phases A and B): Train the model to predict expert breaklines, refinement regions, and resolution fields from a large corpus of curated and bulk HEC-RAS projects.
2. **Performance fine-tune** (Phase C): Starting from the pretrained model, fine-tune against simulation-based performance metrics. The exact objective is TBD but will balance solution accuracy against runtime / cell count.

## Consequences

### Positive
- Supervised pretrain bootstraps cheaply on existing labeled data, getting a working model into modelers' hands fast.
- Performance fine-tune lets the system eventually exceed expert mimicry by optimizing for hydraulic performance directly.
- Two-stage approach decouples the data-engineering work (Phase A.1 bulk corpus) from the compute-heavy work (Phase C fine-tuning).
- The pretrained model is a useful artifact in its own right, even if Phase C is never reached.

### Negative / risks
- Requires significant compute for Phase C: hundreds to thousands of HEC-RAS runs.
- Reward function design for Phase C is non-trivial — needs to capture accuracy, runtime, stability, and avoid degenerate solutions.
- Pretrain biases may be hard to overcome in fine-tune if too strong (e.g. always placing breaklines where experts do, even when the hydraulics don't warrant it).

## Alternatives considered

- **Supervised only:** Ship Phase A and B as a static-mesh tool, no Phase C. Simpler, cheaper, but ceiling is expert quality.
- **Performance optimization only:** Skip pretrain, train from scratch on simulation rewards. Rejected — infeasible from a cold start given HEC-RAS run cost.
- **Hydraulic surrogate as the reward signal:** Train a fast neural surrogate of the HEC-RAS solver, use it to score meshes during fine-tuning. Promising but defers Phase C until the surrogate is good enough. Keep on the table for Phase C implementation.

## References

- `docs/decisions/001-staged-a-b-c-delivery.md`
- `docs/roadmap.md` (Phase C section)
