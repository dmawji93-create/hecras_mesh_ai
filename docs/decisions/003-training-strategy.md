# ADR 003: Training Strategy — Supervised Pretrain + Performance Fine-Tune

**Status:** Accepted — amended 2026-05-23 (see Amendment below)
**Date:** 2026-05-21

---

## Amendment (2026-05-23)

The original decision below stands unchanged: training remains a two-stage process. What changes is the *framing and emphasis*, prompted by ADR 011 (Quantitative Mesh-Quality Objective).

The original ADR implicitly treated expert meshes as the thing to learn — supervised pretraining as the main course, performance fine-tuning as a later refinement. This undersells the project. Expert meshes are imperfect: bounded by intuition, inconsistent between modelers, slow to produce, and unauditable. A model that imitates them inherits a hard performance ceiling at roughly average-expert quality, and solves only the speed problem while leaving the quality problem untouched.

Revised framing:

- **Expert meshes are a prior, not a target.** Supervised pretraining (Stage 1) is *bootstrapping*. It places the model in a sane region of solution space cheaply and encodes hard-won knowledge — non-degenerate cell layouts, structure handling, domain conventions, the failure modes worth avoiding — that a from-scratch optimizer would otherwise rediscover painfully. This is analogous to AlphaGo learning from human game records: a strong, cheap starting point.

- **The quantitative objective is true north.** Stage 2 optimizes against the mesh-quality objective defined in ADR 011 — a measurable error functional against a converged reference solution, traded against compute cost. This is where the project's value actually lies, and it is what allows the model to *exceed* expert quality rather than asymptote to it. Analogous to AlphaZero surpassing AlphaGo by optimizing the true objective directly rather than imitating demonstrations.

Consequently, Stage 2 is not a Phase C afterthought; it is the destination. The objective it optimizes (ADR 011) is designed early, not deferred.

Disanalogy worth keeping in mind: unlike Go, evaluating the objective requires an expensive simulation, and the reward is an engineered functional rather than a free signal from nature. The optimization is AlphaZero-like in spirit but with costly rollouts and a designed reward — which is why ADR 011's compute cost and functional-validation risks are first-order concerns.

---

## Context

Two broad paradigms exist for training the mesh model:

- **Supervised mimicry of expert meshes** — learn from curated projects with expert-placed breaklines and refinement regions. Cheap to train once labels are available, but inherits expert idiosyncrasies and cannot exceed expert quality.
- **Closed-loop performance optimization** — generate a mesh, run HEC-RAS, score the result, refine. Captures real hydraulic performance but each training step is expensive (a full simulation), and is hard to start from a cold initialization.

Each has serious tradeoffs alone. Together they compose: supervised pretrain gets the model into a reasonable region of weight space cheaply, then performance fine-tuning refines toward what actually matters (accuracy and runtime, not just expert mimicry).

## Decision

Use a **two-stage training strategy**:

1. **Supervised pretrain** (Phases A and B): Train the model to predict expert breaklines, refinement regions, and resolution fields from a large corpus of curated and bulk HEC-RAS projects. *(Per the 2026-05-23 amendment: this stage is bootstrapping — it establishes a strong prior, not the final target.)*
2. **Performance fine-tune** (Phase C, and informing Phase B): Starting from the pretrained model, fine-tune against simulation-based performance metrics. *(Per the 2026-05-23 amendment: the objective is defined quantitatively in ADR 011 — reference solution, error functional, cost term. This stage is the destination, not an afterthought.)*

## Consequences

### Positive
- Supervised pretrain bootstraps cheaply on existing labeled data, getting a working model into modelers' hands fast.
- Performance fine-tune lets the system exceed expert mimicry by optimizing for hydraulic performance directly (objective defined in ADR 011).
- Two-stage approach decouples the data-engineering work (bulk corpus) from the compute-heavy work (fine-tuning).
- The pretrained model is a useful artifact in its own right, even before fine-tuning.

### Negative / risks
- Requires significant compute for Stage 2: grid-refinement reference runs plus the optimization loop itself (see ADR 011).
- Reward-function design for Stage 2 is non-trivial — see ADR 011's treatment of the error functional and its garbage-objective and reference-artifact risks.
- Pretrain biases may be hard to overcome in fine-tune if too strong (e.g. always placing breaklines where experts do, even when the quantitative objective does not warrant it). The amendment's "prior, not target" framing is partly a mitigation: pretraining should be weighted as a regulariser, not an anchor.

## Alternatives considered

- **Supervised only:** Ship Phases A and B as a static-mesh tool, no Stage 2. Simpler and cheaper, but the ceiling is expert quality — rejected per the 2026-05-23 amendment as failing to address the quality half of the value proposition.
- **Performance optimization only:** Skip pretrain, train from scratch on simulation rewards. Rejected — infeasible from a cold start given HEC-RAS run cost.
- **Hydraulic surrogate as the reward signal:** Train a fast neural surrogate of the HEC-RAS solver, use it to score meshes during fine-tuning. Promising as an accelerator; kept on the table for Stage 2 / Phase C implementation (see ADR 011 open questions).

## References

- `docs/decisions/001-staged-a-b-c-delivery.md`
- `docs/decisions/011-mesh-quality-objective.md` (defines the Stage 2 objective)
- `docs/roadmap.md` (Phase C section)
