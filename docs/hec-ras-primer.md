# HEC-RAS 2D Mesh Primer

A short orientation for anyone joining this project who comes from the ML side and hasn't worked with HEC-RAS. If you're a modeler, skip this; if you're a Claude Code session reading this for the first time, read it.

## What HEC-RAS is

HEC-RAS (Hydrologic Engineering Center — River Analysis System) is the de-facto standard hydraulic modeling tool, developed and maintained by the US Army Corps of Engineers. It solves 1D, 2D, and coupled 1D/2D unsteady flow problems for rivers, floodplains, dam breaches, urban drainage, and similar.

This project concerns its **2D modeling** capability, which solves either the 2D Diffusion Wave equations (default, faster, more stable) or the full Shallow Water Equations on an unstructured mesh of polygonal cells.

## The 2D mesh

The 2D mesh is the spatial discretization the solver runs on. In HEC-RAS:

- Cells are **unstructured polygons** with 3 to 8 sides.
- The mesh is built as the **Voronoi dual** of a Delaunay triangulation of user-defined computation points.
- The user typically specifies a nominal cell size (e.g. 50 ft × 50 ft) for the whole domain, then refines locally with breaklines and refinement regions.

## Sub-grid bathymetry

HEC-RAS uses Casulli's sub-grid bathymetry approach (Casulli 2008). Each cell and cell face stores pre-computed hydraulic property tables — elevation-volume, elevation-wetted-perimeter, elevation-area, roughness — derived from the underlying high-resolution terrain. This means the cells **do not need to resolve every terrain feature**; they only need to be positioned and oriented well. The solver then accounts for sub-cell topography through the property tables.

**Implication for ML:** The fine terrain inside a cell is already accounted for. What matters is where cell *boundaries* (faces) fall — they must align with anything that blocks or redirects flow.

## Breaklines

**Polylines that force cell faces to align along them.** Used to enforce hydraulically meaningful boundaries:

- High ground (ridges) between channels and overbanks
- Levees, road embankments, railway embankments
- Channel banks
- Walls, weirs, hydraulic structures
- Flow-direction alignment in featureless floodplains (reduces numerical diffusion)

Breaklines are the **primary target of Phase A** of this project. They are stored as polylines in the geometry HDF5 file (`.gNN.hdf`).

## Refinement regions

**Polygons that locally change the cell size.** Used to:

- Increase resolution where hydraulics are complex (main channels, around structures, in urban canyons)
- Decrease resolution in homogeneous overbank areas to save compute
- Align cells with flow direction (often combined with a breakline down the centerline)

Refinement regions are the **primary target of Phase B** along with the broader resolution field.

## Mesh build order

When HEC-RAS regenerates a mesh, it processes inputs in this order:

1. Take computation points from the computation points layer
2. Insert refinement region perimeters as breaklines
3. Insert breakline points, **overriding** any computation points within a buffer around the breakline
4. Delaunay-triangulate the resulting point set
5. Compute the Voronoi dual to get cells

Any ML system that emits mesh artifacts must respect this order — for instance, predicting two breaklines too close to each other will produce point conflicts and degenerate cells.

## Mesh quality goals

A *good* mesh has cell faces aligned with hydraulically critical features, cells oriented with flow direction in channels, and resolution proportional to the gradient of the solution (fine where flow is complex, coarse where it's not). A *bad* mesh either misses critical features (water leaks across a road embankment that wasn't enforced) or wastes resolution in areas with smooth, slow flow.

## File formats

- **`.prj`** — project file (text)
- **`.gNN`** — geometry file, plain text (where NN is the geometry index, e.g. `g01`, `g04`)
- **`.gNN.hdf`** — geometry HDF5 (binary). Contains the mesh, breaklines, refinement regions, computation points, hydraulic property tables, etc. **This is the primary read target.**
- **`.pNN`** — plan file (text)
- **`.pNN.hdf`** — plan HDF5: input geometry + simulation results

`rashdf` (https://github.com/fema-ffrd/rashdf) provides convenient read access to both `.gNN.hdf` and `.pNN.hdf`, returning GeoDataFrames for mesh cells, faces, breaklines, refinement regions, etc. For Phase B writes we will need direct `h5py` manipulation since no first-party write API exists.

## Solvers and timesteps

- **Diffusion Wave (default)** — drops the momentum advection term. Faster, more stable, slightly less accurate. Fine for most flood-mapping applications.
- **SWE-ELM** — full Shallow Water Equations, Eulerian-Lagrangian advection.
- **SWE-EM** — full Shallow Water Equations, pure Eulerian advection. More momentum-conservative. Requires Courant < 1.

Larger cells and larger timesteps increase numerical diffusion. This is part of why mesh quality matters: a poorly aligned mesh wastes accuracy at the same cell count.

## References (official)

- HEC-RAS 2D User's Manual: https://www.hec.usace.army.mil/confluence/rasdocs/r2dum/latest
- Computational Mesh Development: https://www.hec.usace.army.mil/confluence/rasdocs/r2dum/6.0/development-of-a-2d-or-combined-1d-2d-model/development-of-the-2d-computational-mesh
- Sub-grid Bathymetry: https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/theoretical-basis-for-one-dimensional-and-two-dimensional-hydrodynamic-calculations/2d-unsteady-flow-hydrodynamics/subgrid-bathymetry
- 2D Flow Areas (RAS Mapper): https://www.hec.usace.army.mil/confluence/rasdocs/rmum/6.0/geometry-data/2d-flow-areas
- Mesh Quality Best Practices: https://www.hec.usace.army.mil/confluence/rasdocs/h2sd/ras2dsed/6.0/hydraulic-best-practices-for-a-2d-sediment-model/mesh-quality
