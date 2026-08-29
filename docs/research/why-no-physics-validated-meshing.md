# Why HEC-RAS Has No Physics-Validated Mesh Generation — and Why That Gap Justifies This Tool

**Type:** Research note (the "why this project should exist" document)
**Date:** 2026-08-23
**Status:** Living document. Feeds the ADR 014 context (data strategy), the Stage 6 design (mesh-quality framework), and the demo/pitch narrative.
**Method:** Three-track research sweep — (1) HEC's own documentation and roadmap statements, (2) the 2D flood-modeling software landscape and its academic literature, (3) the CFD mesh-adaptation literature for contrast. Primary sources cited throughout; confidence caveats at the end.

---

## The question

CFD has had physics-validated meshing for decades: feature-based adaptive refinement since the 1990s, adjoint/goal-oriented mesh adaptation commercialized in Fluent and STAR-CCM+, dedicated meshing vendors valuable enough to be acquired (Pointwise → Cadence, 2021). HEC-RAS — and every other practice-grade 2D flood package — has none of it. Mesh design remains a manual, intuition-bound craft. Why?

**Short answer:** it is not one reason but a stack of five, and they are worth understanding individually because the tool this project is building must live in the space those five reasons leave open — and it does.

---

## TL;DR — the five reasons, ranked by load-bearing weight

1. **The error budget is inverted, and calibration launders mesh error.** Terrain, roughness, and inflow uncertainty dominate discretization error, and Manning's n calibration silently absorbs mesh error — destroying the professional incentive that funded CFD's adaptation stack.
2. **Sub-grid bathymetry removed most of the need.** Since HEC-RAS 5.0, terrain fidelity lives in precomputed property tables, not the mesh. The mesh's only remaining job is resolving flow gradients — which weakens the classic CFD case for refinement and breaks the assumptions of classical error estimators.
3. **The objective structure is wrong for CFD-style adaptation.** The output is a distributed, event-dependent inundation map (not one scalar), produced by a transient whose domain is itself the unknown, in a regulatory practice that requires one fixed auditable mesh reused across an ensemble of design events. *(This is the "hydrograph" reason — see the practitioner's formulation below.)*
4. **HEC-RAS's solver architecture actively resists in-run adaptation.** Near-orthogonal conforming polygonal mesh, no hanging nodes, global implicit solve, and property tables invalidated by any mesh change.
5. **Error-estimation theory gaps plus institutional economics.** Wet/dry fronts break smooth a-posteriori estimators; and a small government team serving a reproducibility-first regulatory user base was never going to fund what aerospace money funded — on a 2D meshing tradition that is only ~10 years old.

None of these five opposes an **offline, learned, simulation-scored mesh generator** that emits a static, auditable mesh. That is the gap this project occupies.

---

## Reason 1 — The error budget is inverted, and calibration absorbs mesh error

In aerospace CFD, geometry and boundary conditions are known almost exactly (CAD geometry, measured freestream), so discretization error is the dominant *controllable* error — which is why decades of R&D went into adapting it away. Output-based adjoint adaptation (Pierce & Giles 2000; Venditti & Darmofal; Fidkowski & Darmofal's 2011 review) is the mature result.

In flood modeling the ranking flips. Bates (Annu. Rev. Fluid Mech., 2022) identifies terrain and boundary-condition error as the binding constraint on flood-model skill. And there is a well-documented, named phenomenon on top: **roughness calibration absorbs discretization error**.

- Horritt & Bates (2001, J. Hydrology): calibrated friction optima *shift with mesh resolution*; model performance saturated near 100 m for their reach.
- Yu & Lane (2006); Neal et al. (2012): roughness can be exploited to counterbalance lost topographic detail on coarse grids.
- Savage et al. (2016, WRR): performance loss on coarse grids is "predominantly caused by changes in flow pathways which lead to non-stationarity in optimal model parameters at different spatial resolutions"; refinement below ~50 m gave little skill gain at 10×+ cost.
- The UK Environment Agency's 2D benchmarking report (SC120002) argues that refining below ~2 m may not improve velocity predictions because DTM and boundary-condition uncertainty are of the same order — and fine grids destroy the ability to run uncertainty ensembles.

Once the calibration step launders mesh error into the friction field, mesh convergence stops being where accuracy is won. The incentive that paid for CFD's meshing industry never materializes.

**Implication for this tool (and the anticipated market objection):** the objection will be *"our calibration handles it."* The counter is this same literature: calibration-absorbed mesh error produces resolution-dependent, non-stationary parameters that fail outside the calibration events. A quantifiably adequate mesh removes a hidden error source rather than re-tuning around it — exactly the argument for the Stage 6 quality report as a user-facing output (ADR 011).

## Reason 2 — Sub-grid bathymetry removed most of the need

This is the architectural twist most non-RAS people miss. HEC's sub-grid theory page states the design premise outright: the free water surface is assumed **smoother than the bathymetry**, so a coarser grid can compute the spatial variability of the free surface while the fine terrain enters through mass conservation. Each cell carries an elevation–volume curve and each face carries elevation–area/wetted-perimeter/roughness curves precomputed from the full-resolution terrain (Casulli 2008/2009 lineage; HEC cites Casulli explicitly).

In classic CFD the mesh must resolve the geometry — that is precisely why bad meshes are fatal and adaptivity pays. In HEC-RAS, the terrain-resolution job was deliberately moved *out of the mesh* into the property tables. What remains for the mesh is resolving flow-field gradients (and aligning faces with flow-controlling features — hence breaklines). HEC's own cell-size guidance follows: large cells where the water surface is flat, small cells where WSE/velocity change rapidly, faces oriented along controlling terrain features.

The industry's chosen alternative to adaptive meshing was therefore **sub-grid physics on static grids**: TUFLOW's SGS (2020) claims cell-size sensitivity is "virtually eliminated" (vendor-sourced, but with published benchmarks: 20 m SGS cells within ~10% of a 1 m baseline where non-SGS was ~22% off). Sub-grid attacks the same error source as AMR — topographic under-resolution — without changing the mesh, preserving determinism and reviewability.

**Implication for this tool:** the mesh problem that *remains* after sub-grid — where do flow gradients need resolution, and where must faces align with controlling features — is exactly the problem an ML system trained on terrain + hydraulics and scored by simulation can solve. It is also unanswerable a priori by geometry alone, which is why CFD-style geometric mesh-quality metrics never took root here.

## Reason 3 — The objective structure is wrong for CFD-style adaptation (the hydrograph reason)

### The practitioner's formulation

> A senior HEC-RAS developer (personal communication, August 2026): the reason we don't have CFD-style mesh generation tools is that **in CFD the input flows are more easily modeled, whereas rainfall-runoff response hydrographs carry hydrologic variability and occur transiently over a time scale.**

This is the practitioner's statement of the mechanism, and the research strongly supports it. Unpacked:

- **In CFD, the forcing is well-posed.** A freestream velocity, an RPM, a mass-flow rate: deterministic, measured or specified, frequently steady. The solution settles into a state (or statistically steady state) that a mesh can be adapted *to*, and the whole commercial adjoint-adaptation stack assumes exactly this: a quasi-converged steady solve, one scalar output functional (lift, drag), adaptation iterating between solves.
- **In riverine hydraulics, the forcing is itself a model output with irreducible uncertainty.** A rainfall-runoff response hydrograph inherits hydrologic variability — storm spatial/temporal structure, antecedent moisture, loss rates — and the "design event" (1% AEP) is a statistical construct, not an observable state. The forcing then sweeps the system through a *transient range of hydraulic states over days*: the flood wave arrives, the wet/dry front advances across the floodplain, and recedes.

Two consequences follow, one for each half of his statement:

1. **Hydrologic variability** (the uncertainty half) means refining the mesh below the hydrologic error floor buys nothing — this is Reason 1 arriving through the upstream boundary condition.
2. **Transience over a time scale** (the dynamics half) means there is no single flow state to adapt a mesh to. The "optimal" solution-adapted mesh differs between the 10-yr and 500-yr events, and between hour 6 and hour 60 of the same event. The domain of interest — the inundated area — **is the answer**, not an input.

### The supporting structure around it

- **No single scalar functional:** flood outputs are maps (depth, extent, velocity, arrival time) per event. Unsteady adjoint adaptation requires marching the adjoint backward in time with the full forward state available (checkpointing ≈3× forward cost over 10^4–10^5 timesteps) — still research-grade even in aerospace.
- **The regulatory kill shot:** FEMA studies run the full exceedance ensemble (10%, 4%, 2%, 1%, 0.2% annual chance) through *one geometry*; no-rise/floodway certification requires exact reproduction of the duplicate effective model (state QA precision ~0.01 ft). A mesh that adapts to its inputs breaks the audit chain. **Static meshes are an implicit compliance feature.**
- **The scale-separation test:** AMR pays when a small moving front crosses a huge quiet domain. That is tsunami modeling — ocean-basin propagation at km cells, coastal inundation at ~10 m — and, tellingly, that is exactly where depth-averaged AMR *did* reach regulator-approved practice (GeoClaw, NTHMP-approved 2012). Riverine design-storm floods are slow and broadly wet at peak; there is little quiet far-field to coarsen. Kesserwani & Sharifian (2023) conclude directly: static non-uniform grids for gradual riverine floods; dynamic adaptivity pays mainly for rapidly propagating flows (dam break, tsunami). Even well-engineered dynamic adaptivity now returns only 1.2–4.5× on GPUs (LISFLOOD-FP 8.2 GPU-MWDG2, 2025) because brute-force uniform GPU grids became cheap.

## Reason 4 — HEC-RAS's architecture actively resists in-run adaptation

Documented architecture (HEC 2D User's Manual, Technical Reference):

- Mesh: unstructured convex polygons, ≤8 faces, generated Delaunay→Voronoi around user-seeded computation points; near-orthogonality is an *accuracy* property (the two-point flux approximation is second-order only on K-orthogonal meshes).
- No hanging nodes / nonconforming interfaces — quadtree-style local refinement has no representation in the data structure; resolution transitions use 5–8-sided transition cells.
- The solver is semi-implicit (Casulli lineage): a global Newton-like sparse solve over all cell water-surface elevations, designed to amortize large timesteps. Changing cell count/connectivity mid-run means rebuilding the matrix machinery the design exists to amortize.
- **Any mesh change invalidates the precomputed property tables** — runtime refinement would mean regenerating elevation–volume and face-conveyance curves from full-resolution terrain, plus conservatively remapping state, at every adaptation cycle, inside a days-long transient. There is no steady-state convergence loop to hide adaptation inside.
- The only runtime adaptivity HEC ever shipped is in **time**: Courant-based variable timestepping.

The HEC-RAS 2025 ("7.0"-generation) rewrite is confirmatory: its documentation concedes 6.x users had to "trick" the mesh generator; the new face-centric conceptual-mesh system with automated heuristic post-processing (SwapEdges, Smooth, Split/Merge) is a major *generation-side ergonomics* upgrade — and there is **no adaptive, physics-validated, or ML meshing anywhere on the public roadmap**. The 2025 solver bet is explicit + GPU (implicit returning ~2027), still static-mesh.

## Reason 5 — Theory gaps and institutional economics

- **Error estimation is structurally harder for SWE flood problems.** Wet/dry transitions are non-differentiable moving boundaries (loss of high-order convergence at the interface is documented); hydraulic jumps and bores reintroduce the shock-adjoint pathologies production CFD codes already ignore, but along a moving curve that *is* the model boundary; friction/bed source terms require well-balancedness that adaptation and remapping can violate. All of this is still actively published as open research (Poussel et al., IJNMF 2025; Wallwork et al. 2020 had to design their SWE goal-oriented estimator specifically around discontinuities).
- **Economics.** HEC-RAS is free government software; its credited development roster is ~25 named individuals across all modules and decades. Its regulatory user base rewards stability, auditability, and reproducibility over numerical innovation. Compare: aerospace/automotive money sustained dedicated meshing vendors for decades. Commercial flood vendors exist — and their flagship 2020 innovations were *static* quadtree nesting and sub-grid sampling: a revealed preference against dynamic adaptivity.
- **Youth.** 2D flow areas shipped with HEC-RAS 5.0 in 2015/2016. The RAS 2D meshing tradition is one decade old; CFD meshing had four.

---

## Landscape survey: nobody in riverine practice has it

Condensed from a ~17-package survey (see sources):

| Package | Meshing model | Runtime adaptivity |
|---|---|---|
| HEC-RAS 6.x / 2025 | Static Voronoi-like polygonal + breaklines/refinement regions | None (adaptive timestep only) |
| TUFLOW HPC + Quadtree | Static quadtree nesting, user-defined polygons; SGS sub-grid | None |
| MIKE 21 FM | Static flexible triangular mesh | None |
| TELEMAC-2D | Static unstructured FE/FV | None operational |
| SRH-2D, Delft3D FM, BASEMENT, Flood Modeller 2D, InfoWorks ICM, ANUGA, Iber, RiverFlow2D | Static (various) | None |
| LISFLOOD-FP 8.2 (research release) | Raster DG2 | **Dynamic multiwavelet adaptivity — research-grade, 1.2–4.5× over uniform GPU** |
| **GeoClaw** (tsunami practice) | Patch-based Berger–Oliger–Colella AMR | **Fully solution-adaptive; NTHMP-approved (2012)** |
| GeoFlood (2024, Boise State + USGS) | ForestClaw quadtree AMR over GeoClaw | Solution-adaptive; benchmarked against HEC-RAS; research-stage |

**Zero practice-grade riverine/urban 2D flood packages ship solution-adaptive meshing.** The one regulator-approved adaptive depth-averaged code (GeoClaw) lives in tsunami hazard — the regime with the scale separation riverine flooding lacks.

What regulators actually require:

- **FEMA** 2D recommended practices (Nov 2021 HUC8 doc): nominal 200-ft grid, manual breaklines/refinement regions, *iterative manual refinement* validated against gauges. No mesh-independence requirement.
- **UK EA** benchmarking: prescribed fixed resolutions; "adaptive" in its tables means timestep only.
- **Australia (ARR)** is the outlier: recommends ~5 elements laterally across a channel and cell-size *convergence testing* — the only practice guideline found that asks for it.

---

## The Stage 6 connection: a real literature gap

The research found **no published Richardson-extrapolation / grid-convergence-index treatment specific to sub-grid-bathymetry solvers.** And there is a mechanism to respect: under sub-grid, refining h simultaneously changes the terrain-lumping partition, so the *effective model changes with the mesh*. Consequences for the Stage 6 framework (ADR 011 already warns the GCI can misbehave; this is the cause):

- Convergence is output-dependent: storage/WSE converge almost immediately; face conveyance and velocity fields much later.
- Observed convergence can be non-monotone; non-convergent cases must be handled explicitly, not silently.
- A defensible convergence methodology for sub-grid solvers would be both an internal tool and a publishable contribution.

## ML meshing prior art (none of it in this lane)

- **M2N** (2022) and **UM2N** (NeurIPS 2024 Spotlight): learned mesh *movement* (r-adaptivity) networks; UM2N applied to SWE tsunami modeling in 2026 (~91% wave-peak error reduction vs coarse fixed mesh, ~32% runtime reduction) — research-stage.
- **E2N** (2022): neural error estimator replacing the expensive goal-oriented estimator, demonstrated on tidal SWE.
- **AMBER** (2025): supervised prediction of expert non-uniform mesh *resolution fields* — the closest published template to this project's Phase B, in generic form.
- **MeshGraphNets** (2021): GNN simulators with learned adaptive remeshing (foundational).
- **Commercial:** no AI-meshing product for flood modeling found (2023–2026 sweep). SimScale/BeyondMath target CFD setup; CONVERGE has (non-ML) runtime AMR in CFD. HEC-RAS 2025's conceptual mesh is rule-based automation, not learned, and not physics-scored.

## Positioning: why this tool fits the gap the five reasons leave open

Every force above pushes against **in-solver, in-run** adaptivity. None of them opposes this project's architecture:

1. **Offline, before the run.** Mesh design + refinement happens between simulations (the Stage 4 harness), not inside the solver — no property-table regeneration mid-run, no remapping, no solver changes.
2. **The output is still one static, auditable mesh** reusable across the design-event ensemble — fully inside the FEMA/no-rise reproducibility contract. We automate exactly the "manual adaptivity" practice already performs (breaklines, refinement regions, refine-and-recheck), which FEMA's own recommended practices describe as an iterative manual loop.
3. **Physics-validated, quantifiably.** The Stage 6 error-vector framework gives the mesh a measurable quality certificate — addressing the incentive problem (Reason 1) head-on instead of pretending practice will suddenly demand convergence studies: the certificate rides along with the mesh at near-zero user cost.
4. **The learned component targets the one question sub-grid leaves open** — where flow gradients will demand resolution and where faces must align with controlling features — which is a function of terrain + hydraulics that experts currently answer by intuition, and which simulation can score.

**Demo bar note:** the target demo audience (a senior HEC-RAS developer) has built these meshes by hand for decades. The demo must therefore not merely produce *a* mesh — it must produce a mesh that (a) an expert recognizes as well-formed (breaklines where an expert would put them), and (b) carries a quantitative quality report the manual workflow cannot produce (error vector vs. a refined reference, cost, tolerance verdict). The second half is the differentiator: it demonstrates something hand-meshing categorically does not output.

---

## Market demand: what the five reasons do and don't imply

The five reasons explain missing demand for **in-solver, accuracy-driven
adaptive meshing**. They do *not* imply missing demand for automated mesh
*production* — a different product with a different demand curve:

- **Revealed demand for mesh-production automation is strong.** HEC spent
  its scarce development budget making the 2025 rewrite's headline feature
  a meshing overhaul (users had to "trick" the 6.x generator); TUFLOW
  invested in quadtree + SGS to reduce mesh sensitivity; InfoWorks ICM
  markets automated mesh generation from GIS; FEMA BLE is mass-production
  2D modeling where meshing labor scales linearly.
- **The realistic value ordering** (refined with the project owner's
  practice experience — experts largely one-shot their meshes, so expert
  rework is *not* the pain): **(1) labor and speed**, (2) **skill
  transfer** — a validated tool lets a junior modeler's cheaper hours
  produce expert-shaped meshes, and cuts senior review burden, (3) the
  **quality certificate** as differentiator and trust-builder.
- **The cornerstone argument for the certificate** *(the load-bearing
  positioning insight)*: **calibration absorbs mesh error only where
  calibration data exists.** Dam breach, PMF, the 0.2% event, levee
  failure scenarios, climate-shifted hydrology — these are extrapolations
  with no observed event to calibrate against. There, the calibration
  crutch is gone, mesh error flows straight into the answer, and **nobody
  currently measures it**. That is precisely the highest-stakes regime in
  the profession. Add reviewers and litigation support — people who must
  *defend* a model rather than build one — and the certificate has real
  customers; they are simply not the median customer. Lead with labor;
  win with the certificate where calibration cannot go.
- **Honest concession:** pitched on accuracy alone to the median
  calibrated-study consultant, the certificate is a vitamin. The demand
  case must be led by labor/skill-transfer, with the uncalibratable
  regime and review/defense contexts as the certificate's beachhead.

## HEC-RAS 2025: competitive assessment (Aug 2026)

Does the 2025 rewrite's new meshing system kill the need? **No — net
tailwind.** Verified against the 2025 documentation, release notes
(through Beta 1, 2026-03-30), and roadmap:

| Tool leg | 2025 impact |
|---|---|
| Learned breakline/arc placement | Neutral → tailwind |
| Learned resolution field | Mildly narrowed; core intact |
| Simulation-validated quality certificate | Pure tailwind |

- The conceptual-mesh system (arcs + nodes + metadata; four cell systems;
  geometric post-processing recipes) automates mesh *emission* — but
  **100% of arc placement and all cell-size anchor values remain human
  decisions**. Verified absent from shipped features and roadmap: (a)
  automatic breakline/arc placement from terrain, (b) terrain- or
  hydraulics-driven cell sizing (Beta 1's sparse-size interpolation
  propagates *user guesses* geometrically), (c) any mesh validation,
  quality metric, or convergence automation, (d) any mention of ML/AI.
  HEC's own "Best Practices" and "Aligning Cells" sections are
  *Documentation Pending* — the craft knowledge this tool learns is
  exactly what HEC has deferred.
- **Product-surface implication:** a handful of arcs + metadata now fully
  determines a mesh — a compact, auditable representation that is what
  this project's models naturally predict. Shipped import path today:
  shapefile polylines into the conceptual mesh. The sparse-size
  interpolation makes the resolution model *easier* to deliver (predict
  anchors; let RAS interpolate). Open items to verify in the installed
  beta: C# API mesh-authoring scope; whether shapefile import carries
  metadata attributes.
- **Infrastructure implication:** 2025's headless Linux/Docker/CLI
  execution and GPU solver (12–35× in alpha) make the ADR 014 data
  factory and certification loops dramatically cheaper — capabilities HEC
  built for other reasons and is not itself using for validation. HEC's
  answer to mesh sensitivity is *cheaper manual experimentation* ("mesh
  experimentation will be far easier") — confirming the problem while
  leaving automated certification on the table.
- **Market timing:** 2025 is beta ("please don't" for production), missing
  1D/structures/dam breach until ~2027; FEMA acceptance applies to 6.6.
  The 6.x installed base — which the Stage 4 harness already targets — is
  the market for years, and the sanctioned 2025→6.6 mesh export
  ("mesh in 2025, solve in 6.x") lets an arcs-speaking tool serve both.

## Confidence and caveats

- FEMA's Nov 2023 2D hydraulics guidance PDF was inaccessible (HTTP 403); FEMA practice claims rest on the Nov 2021 HUC8 recommended-practices document (read in full) and secondary confirmations.
- TUFLOW SGS sensitivity figures and performance multipliers are vendor-sourced (with published benchmarks but limited independent verification).
- HEC-RAS 2025 documentation is beta-stage and in flux; statements about the final 7.0 feature set cannot yet be verified.
- No HEC-authored statement of *why* AMR is absent exists; Reason 4's structural argument is inference from documented architecture (labeled accordingly).
- The practitioner's formulation is a personal communication (August 2026), quoted from memory, not verbatim; the named attribution is withheld pending the speaker's OK.

## Sources (load-bearing)

**HEC primary:** 2D Modeling User's Manual (mesh development; advantages/capabilities; grid-size selection), Technical Reference Manual (Grid and Dual Grid; ELM-SWE solver; Subgrid Bathymetry; references listing the Casulli lineage), 2D Sediment Manual (Mesh Quality), HEC News Fall 2024 ("Future of HEC-RAS"), HEC-RAS 2025 Advanced Meshing docs. All at hec.usace.army.mil.

**Flood-modeling literature:** Bates, Annu. Rev. Fluid Mech. 54 (2022); Horritt & Bates, J. Hydrology 253 (2001); Savage et al., WRR 52 (2016); Casulli, IJNMF 60 (2009); Sehili, Lang & Lippert, Ocean Dynamics 64 (2014); Kesserwani & Sharifian, J. Hydrology (2023); LISFLOOD-FP 8.2, GMD 18 (2025); Poussel, Ersoy & Golay, IJNMF (2025); EA report SC120002 (2013); FEMA HUC8 2D Recommended Practices (2021); Berger, George, LeVeque & Mandli, GeoClaw (2011); GeoFlood v1.0.0 (2024).

**CFD adaptation:** Pierce & Giles, SIAM Review 42 (2000); Fidkowski & Darmofal, AIAA J 49 (2011); NASA CFD Vision 2030 Roadmap (2020); ANSYS Fluent adaption docs; Siemens STAR-CCM+ 2020.1 AMR release notes; Cadence–Pointwise acquisition (2021).

**ML meshing:** M2N (arXiv:2204.11188); UM2N (arXiv:2407.00382); E2N (arXiv:2207.11233); AMBER (arXiv:2505.23663); MeshGraphNets (arXiv:2010.03409); UM2N-tsunami application (arXiv:2603.06152).

Full URLs are preserved in the session research record; spot-check any claim against the primary source before external use.
