# Stage 3 — Certified-Synthetic Corpus & Breakline Model (Scale)

**Type:** ML + data engineering
**Status:** Not started
**Depends on:** Stage 2 (pilot pipeline), Stage 4 (harness), **Stage 6 (mesh-quality framework — the certifier)**, plus the 2026-08 audit prerequisite fixes (see Notes)
**Maps to:** roadmap Phase A.1 / A.2 (and generates Phase B training data as a by-product)

> **Revision note (2026-08-24).** This stage was rewritten per **ADR 014**.
> The original version planned bulk acquisition of expert-meshed FEMA/NOAA
> studies (ADR 013). NOAA credentials never arrived (requested 2026-05-24,
> no response), and client data is not available for training. The
> corpus is now **manufactured, not acquired**: candidate
> labels proposed cheaply on real public terrain, certified by simulation
> ablation against the Stage 6 error vector. Note the dependency change:
> Stage 3 now sits **after** Stage 6 in the effective build order
> (1, 2, 4, 6 → 3 → 5).

## Objective

Build the certified-synthetic data factory and use it to train the first
generalizing breakline model. The factory converts free public terrain +
free HEC-RAS runs into labels with measured hydraulic consequence; the
model trained on them must transfer to terrain it has never seen — the
failure mode the Stage 2 pilot deliberately exposed (train F1 0.33 /
unseen-project F1 0.0).

## Scope

### In scope
- **Tier 0 — terrain harvesting:** 3DEP 1 m DEM by HUC12, stratified
  across ≥5 physiographic settings; ancillary vector layers (NLD levees,
  NHDPlus HR, OSM/TIGER transportation, NLCD) cached in the Stage 1
  feature format.
- **Tier 1a — proposal engine:** rule + vector candidate breaklines and
  refinement regions.
- **Tier 1b — structure splicer:** parametric levees/embankments inserted
  into real terrain; labels correct by construction; geometry sweeps.
- **Forcing generator:** regional-regression peak flows, small magnitude
  ensemble per domain, automated boundary conditions.
- **Tier 2 — factory orchestrator:** fine reference run + coarse ablation
  variants via the Stage 4 harness; Stage 6 error vectors; certification
  rules; caching, resumability, parallel RAS instances.
- **Certification store + corpus datasheet:** per-label provenance
  (rule / spliced / expert), ablation evidence, forcing ensemble, terrain
  metadata, known biases.
- **Tier 3 — expert QA loop:** sampling protocol for owner audit,
  correction capture, gold-set curation (~20–40 domains incl. hand-meshed).
- **Training-methodology fixes** from the 2026-08 audit as an entry gate
  for retraining: input normalization (terrain-relative, not absolute
  elevation), fixed deterministic validation set, positive-sampling
  jitter, true fixed-batch overfit mode, cross-CRS-safe split check.
- **Retraining at scale** + evaluation on the gold set; failure-mode
  analysis; optional self-supervised terrain-pretraining spike.
- **Vector export** of predicted breaklines (geopackage/shapefile) that
  imports cleanly into RAS Mapper 6.x — and doubles as the HEC-RAS 2025
  conceptual-mesh arc import path.

### Out of scope (deferred)
- Resolution-field *model* (Stage 5) — though Tier 2 banks its training
  data (terrain, forcing → certified resolution field) as a by-product.
- Learned proposal models replacing Tier 1a rules (later iteration).
- RAS 2025 GPU/headless farm migration (start on 6.x COM; revisit when
  the 2025 solver stabilizes).
- Self-supervised pretraining as a full workstream (own ADR if the spike
  pays).

## Tasks

1. **Terrain harvester** — 3DEP + ancillary vector acquisition, stratified
   HUC12 sampler, Stage 1-format caching. Deliverable: ≥10 cached domains
   spanning ≥3 settings to seed the factory.
2. **Proposal engine (Tier 1a)** — candidate breaklines/refinement regions
   from rules + vectors; visual QA notebook.
3. **Structure splicer (Tier 1b)** — parametric embankment insertion with
   by-construction labels; geometry sweep configs.
4. **Forcing generator** — regression-based peak-flow ensemble, hydrograph
   shapes, automated BC writing through the harness.
5. **Factory orchestrator (Tier 2)** — reference + ablation scheduling,
   Stage 6 scoring, certification thresholds (design doc first: greedy vs
   grouped ablation, threshold vs tolerance bands), resumable state,
   parallel RAS instances.
6. **Certification store + datasheet** — schema, provenance, reproducible
   re-certification from stored artifacts.
7. **Expert QA loop (Tier 3)** — audit sampling protocol, correction
   ingestion, gold-set curation with documented zero overlap vs training.
8. **Training-methodology fixes** (audit items) — gate for Task 9.
9. **Retrain + evaluate** — scale training over certified corpus; report
   against the gold set; failure-mode analysis; compare vs Stage 2 pilot.
10. **Vector export** — geopackage/shapefile writer + RAS Mapper import
    verification (6.x now; conceptual-mesh import check when 2025 beta is
    installed).

## Checkpoint — exit criteria

*(Targets marked (p) are provisional — confirm with the project owner
before the stage starts, per the original Stage 3 discipline.)*

- [ ] Factory certifies **≥100 domains (p)** spanning **≥5 physiographic
      settings (p)**, of which **≥20 (p)** contain spliced structures.
- [ ] Every certified label carries stored ablation evidence; a sampled
      re-run reproduces certification decisions.
- [ ] Expert audit of a random certified sample meets an agreed
      acceptance rate **(p)**; systematic failures triaged into proposal
      or certification fixes.
- [ ] Gold set (~20–40 domains) curated with **zero overlap** with
      training data, verified by a cross-CRS-safe spatial check.
- [ ] Retrained model beats the Stage 2 pilot on the gold set against an
      **agreed metric bar — set before the stage starts**.
- [ ] Vector export imports cleanly into RAS Mapper (6.x).
- [ ] Corpus datasheet documents composition, provenance mix, and known
      biases.
- [ ] `pytest` green; pre-commit clean.

## Notes & risks

- **Prerequisite fixes (2026-08 audit) — do these first:** launcher
  completion validation (read the results-HDF `Solution` attribute; no
  fake successes), COM timeout, in-place-write opt-in guard, multi-part
  breakline handling, cross-CRS split check, empty-label-raster guard.
  The factory must never certify against a partial or wrong run.
- **Stage 6 correctness is load-bearing.** Validate the metric framework
  on the analytical benchmark (Thacker — with the audit's extent and
  protocol fixes) before the factory consumes it. If the judge is wrong,
  the factory manufactures lies at scale.
- **Reference solutions inherit sub-grid convergence weirdness** —
  output-dependent, possibly non-monotone (see research note). References
  must pass their own convergence check before serving as truth.
- **Ablation combinatorics:** prune aggressively (greedy/grouped);
  full-factorial is a compute bomb. Unstable coarse variants will waste
  runs — budget for a failure tail.
- **Proposal diversity ceiling:** rules + vectors won't see every expert
  move. Tier 3 corrections and residual mining are the safety net;
  document what the proposers can't express.
- **Bias moved, not removed:** the corpus reflects US public-data coverage
  and our rule vocabulary rather than FEMA-study demographics. The
  datasheet must say so plainly.
- **Compute farm engineering** (parallel COM instances, scheduling,
  retries) is real system work — scope it as such, don't bolt it on.
- The **3B-email** to InFRM (ADR 013) is still worth sending; anything
  that arrives becomes bonus held-out evaluation data, not a dependency.
