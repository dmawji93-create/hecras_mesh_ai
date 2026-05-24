# ADR 013: Bulk Corpus Access Strategy — NOAA OWP S3 primary, FEMA BLE secondary

**Status:** Proposed
**Date:** 2026-05-24

## Context

Stage 3 of the build plan (`docs/build-plan/03-breakline-model-scale.md`)
requires a bulk corpus of expert-meshed HEC-RAS 2D projects to train a
generalizing breakline detector. ADR 004 declared "Hybrid: FEMA bulk +
curated QC subset" as the source strategy in the abstract, but did not
identify concrete access endpoints. This ADR resolves that gap based on
a 2026-05-24 discovery pass.

### The 2026 landscape (findings)

**No centralized ML-ready corpus exists.** Every prior academic effort
curates its own training set from FEMA / USACE / state sources. There
is no public "HEC-RAS-2D ImageNet."

**Three viable public sources, ranked by access ease:**

1. **NOAA OWP `OWP_ras_models` (S3 bucket `s3://noaa-nws-owp-fim/ras2fim`).**
   Maintained by the National Water Model team to drive the `ras2fim`
   flood-inundation pipeline. Contains real HEC-RAS models already
   normalized to a known directory structure with HUC8-indexed catalog
   CSVs. Access via ESIP AWS credentials (contact Carson Pruitt /
   Fernando Salas at NOAA). **First-class programmatic access via
   standard AWS S3 tooling.**

2. **FEMA Base Level Engineering (BLE) via the estBFE Viewer**
   (`https://webapps.usgs.gov/infrm/estbfe/`). Per-watershed HEC-RAS
   submittals downloadable as zipped bundles via the interactive map
   UI. Standardized "RAS_Submittal" folder structure inside each
   bundle. Has the strongest breadth of US coverage but **no documented
   public REST API for bulk download** — the only documented access is
   the interactive JS viewer, plus the underlying USGS ScienceBase
   API which lists the datasets but requires per-watershed lookup.

3. **FEMA Mapping Information Platform (MIP)**
   (`https://hazards.fema.gov`). Requires authenticated access via
   FEMA's RAM Access Portal (RAP) single sign-on. The deeper studies
   (Technical Support Data Notebooks for individual flood studies)
   live here. Out of scope for v1 acquisition.

**Existing tools we can reuse (don't reinvent):**

- **`fema-ffrd/rashdf`** — already a project dependency (parses HEC-RAS
  HDF). Authoritative library, actively maintained.
- **`fema-ffrd/hecstac`** — catalogs HEC-RAS / HEC-HMS models as STAC
  (SpatioTemporal Asset Catalog) Items + Assets. Designed for exactly
  the kind of inventory layer Stage 3 Task 2 (quality filters) and
  Task 4 (held-out curation) need. Catalogs models you already have —
  not a downloader, but the natural inventory layer above whatever
  downloader we build.
- **`NOAA-OWP/ras2fim`** — flood-inundation pipeline that consumes
  HEC-RAS models. Source of the directory-structure convention used in
  the S3 bucket, so understanding it helps with parsing.
- **`NOAA-OWP/RRASSLER`** — "R-based HEC-RAS FAIR standardization." R,
  not Python, so not directly reusable, but the conventions encode
  community standards for catalog metadata.

### What we still don't know (acquisition-blocking unknowns)

- The **size and composition of `OWP_ras_models`**: how many distinct
  projects? What geographic distribution? What fraction are 2D vs 1D?
  What fraction have non-trivial expert breaklines vs default-mesh
  studies? Must be answered by getting credentials and looking.
- The **maximum breadth achievable via BLE**: estBFE coverage is
  growing; need to inventory currently-available studies. May require
  scraping the viewer's tile/REST backend if undocumented endpoints
  exist.
- **Licensing terms**: federal data is generally public domain, but
  ESIP-mediated NOAA data has its own access terms; BLE downloads come
  with FEMA's standard use terms.

## Decision

**Phased acquisition, primary-source first, with `hecstac` as the
inventory layer:**

### Phase 3A (immediate, weeks): NOAA OWP S3 as primary

1. Request ESIP AWS credentials for `s3://noaa-nws-owp-fim/ras2fim`.
2. Write a downloader (`src/hecras_mesh_ai/corpus/noaa_owp.py`) that
   syncs the `OWP_ras_models/` subtree to local cache under
   `data/raw/owp/`.
3. Inventory the contents with `hecstac` → STAC catalog at
   `data/catalogs/owp/`.
4. Filter to HEC-RAS 2D models with at least N breaklines using the
   Stage 3 Task 2 quality filter (TBD threshold). The Bald Eagle
   pilot's 4-breakline g09 sets a useful lower bound.
5. Train Stage 3 v0 on whatever this yields.

### Phase 3B (concurrent or follow-on, weeks-months): FEMA BLE augmentation

6. Probe estBFE Viewer's backend for any public ScienceBase / ArcGIS
   REST endpoints that list and link BLE submittal bundles. (If none,
   fall back to a documented Selenium-driving recipe that captures the
   viewer's network requests.)
7. Write a BLE-specific downloader (`src/hecras_mesh_ai/corpus/fema_ble.py`)
   per discovered access mechanism.
8. Catalog with `hecstac`, dedupe against Phase 3A via spatial
   intersection.
9. Retrain Stage 3 v1.

### Phase 3C (deferred): MIP, state DOTs

10. Only if 3A + 3B together fall short of the agreed-with-user
    held-out F1 target. Adds authentication-gated FEMA MIP and ad-hoc
    state sources.

### Tool reuse

- **`rashdf`** — parsing (already deps).
- **`hecstac`** — inventory + STAC catalog (add as dep).
- **Our code** — downloaders (per source), quality filters, training
  glue.

## Consequences

### Positive

- **Concrete starting point.** NOAA OWP S3 has actual HEC-RAS models
  in a known structure, accessible via standard `aws s3 sync`. No
  reverse-engineering or JS scraping.
- **Reuses community standards.** STAC catalogs via `hecstac` align
  with how the broader fema-ffrd / NOAA OWP ecosystem is moving;
  outputs we produce (curated inventories, model catalogs) are
  interoperable.
- **Phased = de-risked.** If Phase 3A yields enough breadth, we ship
  Stage 3 v0 without ever fighting the BLE scraping problem. If it
  doesn't, we have a fallback ranked by feasibility.
- **The unknowns are dischargeable cheaply.** The big "we don't know
  the corpus composition" question gets answered by emailing NOAA for
  credentials — days, not weeks.

### Negative / risks

- **NOAA S3 corpus may be biased or small.** OWP curates for the
  ras2fim pipeline's needs (flood-inundation libraries), which is
  related but not identical to our breakline-detection objective. The
  set may underrepresent dam-break, urban, or coastal cases.
- **BLE scraping may be brittle.** If estBFE truly has no public REST
  endpoint, Phase 3B reduces to either Selenium-style browser
  automation (fragile, blocked by FEMA UI changes) or per-watershed
  manual downloads (slow, doesn't scale to hundreds of studies).
- **ESIP credentials request is a human dependency.** Email-based
  access provisioning has variable latency. Plan A may sit blocked for
  days awaiting a response.
- **STAC introduces an abstraction layer.** Adds `hecstac` and
  `pystac` dependencies; learning curve for STAC metadata model.
  Tradeoff is worth it if we want our outputs to be reusable by the
  broader community; less worth it if we just want a private CSV
  inventory.

### Neutral

- **No MIP authentication setup in v1.** MIP holds the most
  comprehensive flood-study archive but its SSO-only access pattern
  makes it incompatible with reproducible automated download. Deferred
  to Phase 3C if Phase 3A+3B prove insufficient.

## Alternatives considered

- **Alt A — FEMA BLE first via Selenium scraping.** Drive the
  estBFE viewer with a headless browser, click through watersheds,
  download zipped submittals. **Rejected as primary** because Selenium
  flows break with any UI change, and FEMA's ToS may prohibit
  automated scraping. Reserved as a Phase 3B fallback if no REST
  endpoint is discoverable.
- **Alt B — Build a private corpus from scratch via paid HEC-RAS
  consulting partnerships.** Highest quality, fastest to control.
  **Rejected** — out of scope for the project's resourcing, and would
  trap the model behind non-public data unable to be released.
- **Alt C — Skip bulk corpus entirely; rely on hand-curation of 20-50
  high-quality models.** Smaller but cleaner. **Rejected** — Stage 3's
  whole point is to attack the generalization problem, which requires
  scale (typical CV models need 100s-1000s of distinct examples). A
  curated 20-50 corpus is the Stage 3 *held-out validation set* (Task 4),
  not the training set.
- **Alt D — Wait for fema-ffrd to publish a centralized FFRD
  training corpus.** The fema-ffrd organization is actively producing
  exactly this kind of standardized HEC-RAS data, but no public bulk
  release is announced for the project's timeline. **Rejected** as a
  blocker; revisit opportunistically.

## Open questions

- **Exact NOAA OWP corpus size and 2D-fraction** — answered by
  obtaining credentials and inventorying.
- **Whether estBFE viewer has any undocumented REST backend** —
  answered by browser-network-inspecting a single download.
- **Quality filter thresholds** (min breakline count, min mesh size,
  min run-completion status) — deferred to Stage 3 Task 2; will be
  data-driven once Phase 3A inventory is in hand.
- **Held-out validation set composition** — deferred to Stage 3 Task 4;
  must span the rural-riverine / leveed / urban / mountain / dam-breach
  categories per ADR 002.

## References

- ADR 002 — Mixed/general application scope
- ADR 004 — Hybrid FEMA + curated corpus (this ADR concretizes the
  abstract "hybrid" decision with named endpoints)
- `docs/build-plan/03-breakline-model-scale.md`
- [fema-ffrd / rashdf](https://github.com/fema-ffrd/rashdf)
- [fema-ffrd / hecstac](https://github.com/fema-ffrd/hecstac) — STAC catalog generator for HEC models
- [NOAA-OWP / ras2fim](https://github.com/NOAA-OWP/ras2fim) — source of OWP_ras_models conventions
- [NOAA-OWP / RRASSLER](https://github.com/NOAA-OWP/RRASSLER) — R-based HEC-RAS FAIR standardization (community standards reference)
- [estBFE Viewer](https://webapps.usgs.gov/infrm/estbfe/) — FEMA BLE access UI
- [BLE Data Download Reference Guide](https://webapps.usgs.gov/infrm/estbfe/download/ReferenceGuide.pdf)
