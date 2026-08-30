# Stage 1 — Feature & Label Pipeline

**Type:** ML (data engineering)
**Status:** Complete (2026-05-23; ~15 commits ending 4cea3ad — checkpoint evidence in `docs/STATUS.md`. Scope note: TWI/flow-accumulation/ridge-network ancillary features were deferred to Stage 3.)
**Depends on:** Week 1 plumbing (complete — see `STATUS.md`)
**Maps to:** roadmap Phase A.0 Week 2

## Objective

Turn raw HEC-RAS pilot projects into model-ready inputs and labels: a multi-channel feature tensor derived from terrain, a binary breakline label raster aligned to it, and a spatially-split tiling scheme. This stage produces the raw material the breakline model trains on — no model yet.

## Scope

### In scope
- DEM-derivatives module in `src/hecras_mesh_ai/features/` — slope, aspect, plan curvature, profile curvature, topographic wetness index, flow accumulation, hydro-conditioned ridge networks. Each a tested, typed function.
- Feature stacking into a multi-channel, CRS-aware array (xarray + rioxarray).
- Breakline rasterizer — `breaklines` GeoDataFrame to a binary label raster on the DEM grid, with configurable buffer width.
- TorchGeo-based dataset wrapping Muncie and Bald Eagle, with a **spatial-holdout** train/val tile split (geographically separated, never random).
- Works for both pilot projects (Muncie + Bald Eagle Dam Break).

### Out of scope (deferred)
- The model itself (Stage 2).
- Ancillary data layers — NHD, NLD, roads, land cover (Stage 3, with the bulk corpus).
- Bulk-corpus data acquisition (Stage 3).

## Tasks

1. Open Bald Eagle Dam Break with rashdf; confirm whether its HDF embeds a CRS (open item from `STATUS.md`).
2. Implement and unit-test each DEM-derivative function.
3. Implement feature stacking; verify CRS, resolution, and extent alignment across all channels.
4. Implement the breakline rasterizer with configurable buffer; verify alignment to the DEM grid.
5. Build the TorchGeo dataset and the spatial-holdout splitter.
6. Exploration notebook visualizing every feature channel and the label raster overlaid on terrain.

## Checkpoint — exit criteria

All must be true and verified before starting Stage 2:

- [ ] Every DEM-derivative function has passing unit tests.
- [ ] Feature channels are confirmed aligned (CRS, resolution, extent) — verified visually and programmatically.
- [ ] Breakline label raster aligns pixel-for-pixel with the feature stack.
- [ ] Train and val tiles are spatially separated with **zero overlap** — leakage check passes.
- [ ] The full pipeline runs end to end on both Muncie and Bald Eagle.
- [ ] `pytest` green; pre-commit clean.

## Notes & risks

- CRS handling is the known sharp edge (Muncie's HDF lacks a CRS — source it from terrain). Make CRS resolution explicit and tested.
- Buffer width for breakline rasterization is a hyperparameter — expose it, do not hard-code.
- Keep feature computation deterministic and cached; it will be re-run many times downstream.
