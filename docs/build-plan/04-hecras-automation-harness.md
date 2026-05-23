# Stage 4 — HEC-RAS Automation Harness

**Type:** Engineering
**Status:** Not started
**Depends on:** Week 1 plumbing. Can be developed in parallel with Stages 2-3. **Required before Stage 5.**
**Maps to:** roadmap Phase B prerequisite

## Objective

Build the engineering layer that lets the system *write* HEC-RAS geometry, *launch* simulations, and *parse* results — all programmatically, with no manual RAS Mapper interaction. Everything from Stage 5 onward depends on this. It is the single hardest pure-engineering piece of the project.

## Scope

### In scope
- **Geometry HDF5 writer** — write computation points, breaklines, refinement regions, and the 2D flow area perimeter into a valid `.gNN.hdf`. Extend `rashdf` where possible; fall back to direct `h5py` manipulation.
- **Run launcher** — programmatically execute a HEC-RAS plan (via the HECRASController COM interface or command-line invocation).
- **Results parser** — extract depth, velocity, and water-surface-elevation fields from the plan HDF (`.pNN.hdf`).
- **Round-trip verification** — read an existing geometry, write it back out, confirm equivalence.

### Out of scope (deferred)
- Any ML.
- Refinement logic (Stage 7).
- Mesh quality scoring (Stage 6).

## Tasks

1. Reverse-engineer the `.gNN.hdf` geometry schema for the elements we must write (use Muncie as the reference; compare rashdf reads against the raw HDF structure).
2. Implement the geometry writer element by element, testing each against a HEC-RAS open.
3. Implement the run launcher; confirm a plan executes headlessly.
4. Implement the results parser for depth/velocity/WSE.
5. Build the round-trip test harness.

## Checkpoint — exit criteria

- [ ] **Round trip:** read Muncie's geometry, write it back to a new `.gNN.hdf`, and confirm HEC-RAS opens it and the mesh is equivalent.
- [ ] A programmatically written geometry opens in HEC-RAS with **no errors or warnings**.
- [ ] A run can be launched headlessly and completes.
- [ ] Depth, velocity, and WSE fields are parsed from the result HDF and verified against RAS Mapper's own display of the same run.
- [ ] `pytest` green; pre-commit clean.

## Notes & risks

- **Highest-risk engineering in the project.** There is no first-party API for writing 2D mesh geometry — the HDF schema must be reverse-engineered. Budget accordingly and do not assume it is quick.
- The HECRASController COM interface is Windows-only; confirm it works on the target machine early.
- A malformed HDF can crash HEC-RAS silently or produce a subtly wrong mesh — the round-trip and "opens with no warnings" criteria are non-negotiable gates.
- Consider engaging with the `rashdf` maintainers (FEMA-FFRD) — write support may be of interest upstream.
