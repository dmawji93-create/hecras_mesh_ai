# ADR 004: Training Data Source — Hybrid FEMA Bulk + Curated QC Subset

**Status:** Accepted
**Date:** 2026-05-21

## Context

Phase A and B require a large, diverse corpus of HEC-RAS projects with expert-placed breaklines and refinement regions to use as supervised labels. Four candidate sources:

- **FEMA NFHL / MIP** flood studies — publicly available, vast scale, but variable quality (some studies have meticulous expert meshes; others use defaults or workarounds)
- **USACE / in-house projects** — typically higher quality, but limited access and licensing
- **Hybrid** — bulk from FEMA for scale, with a smaller curated subset for QC, validation, and bias correction
- **Self-curated from scratch** — full control over quality, but extremely slow to generate at meaningful scale

## Decision

**Hybrid: FEMA bulk + curated QC subset.**

- The training corpus is built from FEMA NFHL/MIP studies, deduplicated and filtered by quality heuristics (mesh size, breakline density, run completion, etc.).
- A **curated subset of ~10–20 high-quality, diverse projects** is set aside as a held-out validation set, hand-inspected for expert mesh quality, spanning rural riverine, leveed systems, urban, mountain, and dam-breach geometries.
- Quality filters on the bulk corpus are themselves treated as a learnable component — we may iterate on what counts as a "good enough" mesh for training inclusion.

## Consequences

### Positive
- Scale: enough labeled data to train a generalizing model.
- Validation rigor: a hand-curated held-out set provides a trustworthy benchmark.
- Bias mitigation: visibility into bulk corpus quality through the curated set lets us detect when the model is learning bad habits from low-quality studies.
- Public data: no licensing barriers; the corpus can be redistributed (subject to FEMA terms) which helps reproducibility.

### Negative / risks
- Bulk corpus quality is uneven; quality filters will miss some bad meshes and reject some good ones.
- Geographic and morphological bias of FEMA studies (heavily concentrated in populated US areas, mostly riverine) will be inherited.
- Curation of the held-out set is labor-intensive — needs domain expertise.
- Storage and bandwidth for downloading FEMA studies at scale is non-trivial.

## Open questions

- Concrete quality heuristics for filtering the bulk corpus.
- Whether to augment with USACE or consulting-firm projects later via partnerships.
- Compute target (deferred until data scale is quantified — see Phase A.1 milestones).

## References

- FEMA NFHL: https://hazards.fema.gov/femaportal/NFHL/
- FEMA MIP: https://hazards.fema.gov/femaportal/wps/portal/NFHLWMS
- `docs/roadmap.md` Phase A.1
