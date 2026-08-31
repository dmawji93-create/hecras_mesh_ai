# ADR 014: Certified-Synthetic Data Strategy — physics as the label certifier

**Status:** Accepted
**Date:** 2026-08-24
**Supersedes:** ADR 013 (acquisition plan; see below). Re-concretizes ADR 004. Operationalizes the ADR 003 amendment.

## Context

Stage 3 requires a large, diverse corpus of labeled examples (breaklines,
refinement regions, resolution choices) to train a generalizing model. The
acquisition strategy in ADR 013 is dead or blocked:

- **NOAA OWP S3 (ADR 013 Phase 3A, primary): dead.** ESIP credentials were
  requested 2026-05-24 (via the NOAA OWP contacts). No response after
  13 weeks. Treated as a no.
- **Client data: unavailable.** Client work products are not available
  for model training. No internal corpus, ever.
- **FEMA BLE (ADR 013 Phase 3B): never initiated, now demoted.** The
  3B-email path remains worth sending opportunistically, but the project
  can no longer plan around externally-gated data.

Meanwhile, three assets are abundant and unencumbered:

1. **Terrain** — USGS 3DEP 1 m DEM (public domain, bulk-downloadable,
   no credentials), plus public ancillary vectors: USACE National Levee
   Database, NHDPlus HR hydrography, OSM/TIGER transportation, NLCD.
2. **Simulation** — HEC-RAS is free and the Stage 4 harness runs it
   closed-loop with no human (verified end-to-end 2026-08-23; Muncie
   computes in ~30 s). HEC-RAS 2025 adds headless Linux/Docker/CLI and a
   GPU solver (12–35× in alpha), making bulk runs cheaper still.
3. **A quantitative judge** — the Stage 6 mesh-quality framework (ADR 011)
   scores any mesh against a refined reference.

The scarce resource is **expert labels**. The project owner can hand-mesh
projects but only tens of them, they carry one expert's habits ("my human
error built in" — owner, 2026-08), and expert annotation does not scale.
The ADR 003 amendment (2026-05-23) already declared expert meshes **a
prior, not the training target**; this ADR makes that operational.

The key observation: a breakline is not an arbitrary annotation — it is a
**causal claim** ("conveyance is controlled here; a mesh that ignores this
line gets the wrong answer"). Causal claims are testable by simulation.
Labels can therefore be **certified** instead of trusted, converting the
data bottleneck into a compute problem, which is solvable.

## Decision

**Build the Stage 3 corpus as a certified-synthetic data factory: cheap
proposal generators produce candidate labels on real public terrain, and
the Stage 4 harness + Stage 6 error vector certify which candidates are
real. The proposer is never the judge.**

Four tiers:

- **Tier 0 — Real terrain, never synthetic terrain.** 3DEP 1 m DEMs,
  sampled by HUC12, stratified across physiographic settings (mountain
  valley, piedmont, coastal plain, urban, leveed corridor). Real terrain
  is free and unlimited; synthesizing it would buy a domain gap and
  nothing else.
- **Tier 1a — Programmatic candidate proposal.** Rules + public vectors
  propose candidate breaklines and refinement regions: NLD levee crests,
  road/rail embankments (centerline ∩ terrain-relief probe), channel
  banks/thalwegs (NHDPlus HR + curvature features from Stage 1), ridge
  detection, constriction/corridor heuristics.
- **Tier 1b — Synthetic structures on real terrain.** Parametric levees /
  road embankments spliced into real DEMs. Labels are correct **by
  construction** (the crest is where we put it); the hydraulic effect is
  real (the solver sees the barrier); geometry sweeps (height, crest
  width, skew) teach *why* structures matter, not just where they occur.
- **Tier 2 — Ablation certification (the bulk corpus).** Per domain:
  run a fine-mesh reference; run coarse-mesh variants with candidate
  labels ablated (greedy/grouped, not full factorial); score every
  variant with the Stage 6 error vector. **A candidate becomes a label
  only if removing it measurably degrades the solution.** Residual error
  with no candidate nearby marks a *missing* label to be mined. The same
  runs yield Phase B training data for free: (terrain, forcing) →
  (resolution field that met tolerance at minimum cost).
- **Tier 3 — Expert as auditor, not author.** The owner audits samples of
  certified labels, corrects systematic failures (corrections are the
  highest-information labels), and hand-meshes only a small gold set
  (~20–40 domains) for held-out evaluation and fine-tuning.

**Forcing:** plausibility and variety, not calibration. Regional-regression
peak flows (StreamStats-style) scaled to a small magnitude ensemble
(≈10-yr / 100-yr / 500-yr-scale) per domain; labels are certified only if
they matter **across** the ensemble — matching the one-mesh-serves-the-
ensemble regulatory practice.

**Scaling posture:** the binding constraint is terrain diversity ×
certified label quality — not parameter count, not GPU hours. The data
factory wants CPU cores / RAS instances (and later RAS 2025 GPU/headless);
H100-class training compute is the cheap step and is deferred until the
corpus exists.

## Consequences

### Positive

- **Removes the external gate entirely.** No credentials, no data-sharing
  agreements, no client data. Every input is public domain or generated
  in-house.
- **Launders both bias sources.** Rule-proposal bias and expert habit are
  both filtered by a judge that is independent of the proposer. Every
  surviving label carries measured evidence of hydraulic consequence —
  something even a real expert corpus could not provide.
- **Solves the refinement-region label famine.** The pilots contain zero
  refinement regions (ADR 006 reality). Tier 2 generates them from
  scratch by measuring where local refinement pays.
- **Unifies the critical path.** The factory's runner is Stage 4, its
  judge is Stage 6 — the same chain the flagship expert demo needs. One
  investment serves both.
- **Scales with money, not permission.** Hundreds of certified domains
  per month are plausible on one workstation + a few cloud instances
  (Muncie-scale runs are ~30 s–minutes).

### Negative / risks

- **Stage 6 becomes load-bearing for everything.** If the error vector is
  wrong, the factory certifies lies at scale. The build plan's rule —
  validate the metric framework on an analytical benchmark before use —
  is now non-negotiable (Thacker fixes from the 2026-08 audit first).
- **Harness trust fixes are prerequisites.** The audit found the launcher
  can launder a crashed run into a fake success and the writer corrupts
  multi-part breaklines. The factory must not certify labels against
  partial results. These fixes gate the factory build.
- **Proposal diversity ceiling.** Rules + vectors may miss expert moves
  (subtle alignment choices, judgment calls). Mitigations: Tier 3
  corrections, residual-mining, and honest documentation of what the
  proposal tiers cannot see.
- **Compute engineering is real work.** Parallel RAS instances, run
  timeouts, resumable orchestration, result caching — the factory is a
  system, not a script.
- **Reference-solution error is inherited.** A certified label is only as
  good as the fine reference; references must themselves pass a
  convergence check. Sub-grid convergence is nonstandard (see
  `docs/research/why-no-physics-validated-meshing.md`) — expect
  non-monotone, output-dependent behavior and handle it explicitly.
- **Corpus bias shifts, not disappears.** The corpus inherits US public
  data coverage and our rule vocabulary instead of FEMA study bias.
  Document it in the corpus datasheet.

### Neutral

- Expert meshes remain a prior (ADR 003 amendment unchanged); their role
  narrows to gold-set evaluation, fine-tuning, and audit.
- ADR 013's 3B-email remains worth sending; any BLE data that ever
  arrives becomes bonus held-out evaluation material, not a dependency.

## Alternatives considered

- **Alt A — Expert annotation at scale (owner hand-meshes everything).**
  Rejected: tens of examples maximum, one expert's habits baked in, and
  the owner's hours are the project's scarcest resource — better spent
  auditing a factory than feeding one.
- **Alt B — Rule-based labels without certification.** Cheap and fast.
  Rejected as sole source: trains the model to imitate our rules —
  the model can never exceed its teacher, and rule bias goes unmeasured.
  (Rules survive as Tier 1a *proposals*.)
- **Alt C — Generative-model synthetic meshes (train a generator, sample
  meshes).** Rejected: circular (needs the corpus we lack to train the
  generator) and unverifiable without the same physics judge — at which
  point the judge alone suffices.
- **Alt D — Keep waiting on NOAA / FEMA BLE.** Rejected: 13 weeks of
  silence, external gating incompatible with progress; demoted to
  opportunistic bonus.
- **Alt E — Commercial data partnership.** Rejected as before (ADR 013
  Alt B), and client-derived data is in any case unavailable for
  training.

## Open questions

- **Certification thresholds:** what error-vector delta earns a label its
  place? Must be set relative to the Stage 6 tolerance bands; expect
  iteration. Deferred to Stage 3 Task 5 design.
- **Ablation experiment design:** greedy vs grouped vs Shapley-style
  attribution for interacting candidates. Start greedy; revisit if
  interaction effects dominate.
- **Scope narrowing (ADR 002):** limited initial physiographic scope
  would cut the required diversity substantially. Deferred — revisit
  after the first 50 certified domains reveal transfer behavior.
- **Self-supervised terrain pretraining** (replacing ImageNet weights):
  strongly indicated by the 2026-08 audit's normalization finding;
  deserves its own ADR when designed.
- **Farm platform:** classic HEC-RAS COM (verified on 7.0; works today)
  vs 2025 headless/GPU (faster, beta). Start classic; migrate when
  2025's solver stabilizes.
- **Product surface:** RAS 2025's conceptual mesh (arcs + metadata) as
  the emission format for Phase B — separate ADR when Phase B starts.

## References

- ADR 002 (scope), ADR 003 + amendment (expert meshes are a prior),
  ADR 004 (superseded corpus framing), ADR 011 (quantitative objective),
  ADR 012 (numerical-methods-first), ADR 013 (superseded acquisition plan)
- `docs/build-plan/03-breakline-model-scale.md` (rewritten alongside this ADR)
- `docs/research/why-no-physics-validated-meshing.md` — market/positioning
  research; the "calibration absorbs mesh error only where calibration
  data exists" cornerstone; HEC-RAS 2025 competitive assessment
- 2026-08 system audit (owner's working notes; prerequisite findings
  summarized in `docs/STATUS.md`, 2026-08 resumption section) —
  harness-trust and training-methodology prerequisites
- USGS 3DEP, USACE National Levee Database, NHDPlus HR, NLCD, USGS
  StreamStats regional regression equations
- Prior art: AMBER (arXiv:2505.23663 — expert-imitating resolution
  fields), UM2N (arXiv:2407.00382), MeshGraphNets (arXiv:2010.03409) —
  none uses a physics certifier; that is this project's differentiator
