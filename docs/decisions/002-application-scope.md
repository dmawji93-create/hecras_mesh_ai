# ADR 002: Application Scope — Mixed / General-Purpose

**Status:** Accepted
**Date:** 2026-05-21

## Context

HEC-RAS 2D is used across very different application domains: FEMA-style riverine flood mapping, dam and levee breach analysis, urban stormwater, coastal storm surge, and sediment transport. Each has different mesh priorities:

- **Riverine flood mapping:** channel resolution, floodplain extent
- **Dam breach:** wavefront propagation, time-of-arrival
- **Urban:** street-grid alignment, building footprints, drainage structures
- **Coastal:** open boundary handling, tidal flats

A narrow scope yields a sharper model with less data; a general scope yields a more useful tool but demands a more diverse training corpus and risks underperforming relative to a specialist in any single domain.

## Decision

Target **mixed / general-purpose** scope. Train on a diverse corpus spanning at least riverine flood mapping, leveed/embankment-heavy systems, and dam-breach geometries. Defer explicit handling of urban storm drain coupling and coastal-specific concerns to a later phase if/when needed.

## Consequences

### Positive
- One tool covers the broad majority of HEC-RAS 2D applications.
- Forces the data pipeline to handle geographic and morphological diversity from day one — no painful rework when expanding scope later.
- The Phase A breakline problem is largely domain-agnostic: a ridge is a ridge, a levee is a levee, regardless of whether the model is a flood study or a breach.

### Negative / risks
- May underperform domain-specific specialists, especially for urban or coastal applications where features deviate from the rural-riverine norm.
- Training corpus must be carefully balanced — a corpus dominated by FEMA flood studies will yield a model that quietly assumes rural-riverine morphology.
- Some domain-specific tooling (e.g. urban drainage / sewer connections) will not be in scope at all.

## Alternatives considered

- **Riverine flood mapping only:** Simpler, but limiting. Most FEMA studies are this anyway, so the model would tend here regardless.
- **Dam/levee breach only:** Too narrow a market; would not justify the effort.
- **Urban/stormwater only:** Different problem profile (street grids, structures, sub-grid drainage). Defer.

## Open questions

- Concrete corpus balance targets — what % riverine vs leveed vs breach? Decide once we inventory available FEMA studies in Phase A.1.
- Whether to maintain domain-specific model variants in the long term, or rely on the general model being "good enough" everywhere.

## References

- `docs/roadmap.md`
- `docs/decisions/004-training-data-source.md`
