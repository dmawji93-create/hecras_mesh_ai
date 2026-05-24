# Stage 1 textbook brief — for ingestion by Claude chat

## How to use this file

Paste this entire document into a fresh Claude.ai conversation. Upload the three
PNG files from `stage1-images/` alongside (they're referenced by filename in
this brief). Then ask Claude to produce the textbook described in the **Your
task** section below.

---

## Your task

Produce a **textbook-style explanation** of Stage 1 of the `hecras_mesh_ai`
project, using the material in this brief as your source. Your output should
read like a chapter from a graduate-level "applied geospatial ML" text written
for a strong hydraulic engineer who is new to ML.

### Audience profile

- **Deep HEC-RAS / hydraulic-engineering expertise.** Don't explain breaklines,
  meshes, channels, or hydraulic structures. Do explain anything ML-specific.
- **New to ML and PyTorch.** When a term first appears — tensor, channel, U-Net,
  segmentation, loss, gradient, DataLoader, NaN propagation, class imbalance,
  spatial holdout, etc. — give a short conceptual primer the first time, then
  use it freely. Build the mental model incrementally.
- **Strong general engineering ability.** Treat the reader as a peer: they can
  read code, parse math, follow tradeoffs. Don't oversimplify.

### Output structure

Organize the chapter like this:

1. **Why Stage 1 exists** — the problem it solves and where it sits in the
   overall project.
2. **The end-to-end pipeline in one diagram** — text or ASCII art tracing
   `DEM (TIFF) + geometry HDF → feature stack → label raster → cached GeoTIFFs
   → tiles → (features, labels) batch ready for a U-Net`.
3. **Task-by-task walkthrough** — one section per task (1 through 6). For each:
    - The problem statement
    - Key concepts and ML primers (introduced as needed)
    - The math, where relevant, with LaTeX
    - The implementation sketch (signatures, key lines — not full files)
    - The design decisions and what alternatives were rejected
    - The tests and what they prove
    - Bugs caught + what they teach about ML-pipeline thinking
    - A small **exercise** the reader could try in isolation to build intuition
4. **The Stage 1 exit visuals** — reference each of the three attached PNGs;
   explain what they show and *why each visual is the proof of a specific
   checkpoint criterion*.
5. **What the reader has learned** — a short list of transferable concepts
   (not just "what we built" but "what mental models you now have").
6. **What's next (Stage 2 preview)** — one paragraph on what changes when the
   actual neural network arrives.

### Style guidance

- Use LaTeX for math: `$inline$` and `$$display$$`.
- Use fenced code blocks for code excerpts.
- Reference figures inline as `![caption](filename.png)` — the user will have
  uploaded the PNGs alongside.
- Don't paste long code dumps; show signatures and 3–10 key lines, then
  describe the rest.
- Be honest about tradeoffs and limitations — don't oversell.
- Where decisions were made in the session, frame them as "we considered X
  and Y, chose Y because Z" rather than "the right way is Y".

---

# Source material

The rest of this document is the raw material Stage 1 was built from and the
record of what was actually done. Use it to construct the textbook.

## Project context

`hecras_mesh_ai` automates the construction of 2D computational meshes for
HEC-RAS, the de-facto-standard hydraulic modeling tool. The manual workflow
(an expert places breaklines, draws refinement regions, picks cell sizes,
runs, judges by eye, iterates) is slow, inconsistent between modelers, and
inherits an expert ceiling. The project's value proposition is **faster,
better, and more quantifiable** meshes — replacing tacit expert judgment
with a learned system optimizing against a measurable quality objective.

Staged delivery:

- **Phase A — breakline detection** (where we are): predict expert-quality
  breakline polylines from terrain + ancillary geospatial data. A computer-
  vision problem: input = multi-channel raster, output = binary mask of
  "where do breaklines go," then post-processed to polylines.
- **Phase B — resolution field + complete static mesh**: learn local cell
  sizes, assemble a complete runnable mesh. The "Quick tier" deliverable.
- **Phase C — adaptive refinement against a quantitative objective**: run
  the mesh, measure error against a reference solution, refine, repeat.
  The "Optimal tier" deliverable.

The full ADR set is in `docs/decisions/`. Three are especially relevant to
Stage 1:

- **ADR 003 (amended 2026-05-23)** — Training strategy. Expert meshes are a
  *prior* (warm-start), not the training target. Imitating experts caps the
  model at expert quality; we want to exceed it via Phase C's quantitative
  objective. Stage 1 produces the labeled data for the supervised warm-start
  stage of training.
- **ADR 005** — Tech stack: PyTorch + Lightning + segmentation_models_pytorch
  + TorchGeo, on the standard Python geospatial stack (rasterio, geopandas,
  xarray + rioxarray, shapely). HEC-RAS HDF I/O via rashdf + h5py.
- **ADR 009** — Notebooks for exploration only; modules for keepers. Tested
  where it matters, typed where it helps.

## Stage 1 spec (from `docs/build-plan/01-feature-and-label-pipeline.md`)

**Objective.** Turn raw HEC-RAS pilot projects into model-ready inputs and
labels: a multi-channel feature tensor derived from terrain, a binary
breakline label raster aligned to it, and a spatially-split tiling scheme.
No model yet — just the raw material the model will train on.

**Pilots** (per ADR 006):
- **Muncie** (Indiana, White River). DEM at 5 ft / pixel, EPSG:2965
  (NAD83 / Indiana East ftUS). 5,391-cell mesh, 2 breaklines (`Road 1`,
  `HighGround 1`), 0 refinement regions. HDF does NOT embed CRS.
- **Bald Eagle Dam Break** (Pennsylvania). DEM at ~36.5 ft / pixel, EPSG:2271
  (NAD83 / Pennsylvania North ftUS). Ships 12 alternate geometry definitions
  (`.g01.hdf` through `.g13.hdf` with gaps). Picked `g09` as canonical —
  18,066 cells, 4 named breaklines (`SayersDam`, `Lower`, `Middle`, `Upper`),
  the only geometry that embeds CRS in the HDF.

**Exit criteria** (all required before Stage 2):
1. Every DEM-derivative function has passing unit tests.
2. Feature channels confirmed aligned (CRS, resolution, extent) — verified
   visually and programmatically.
3. Breakline label raster aligns pixel-for-pixel with the feature stack.
4. Train and val tiles spatially separated with zero overlap — leakage
   check passes.
5. Full pipeline runs end-to-end on both Muncie and Bald Eagle.
6. `pytest` green; pre-commit clean.

---

# Task-by-task record

## Task 1 — Bald Eagle exploration (`notebooks/02_baldeagle_explore.ipynb`)

**Goal.** Open the Bald Eagle Dam Break example and resolve two open items
from the Week-1 status: characterize CRS embedding behavior, and pick the
canonical geometry from the 12 available.

**Method.** Inventoried all 12 `BaldEagleDamBrk.g*.hdf` files programmatically
via `rashdf.RasGeomHdf`. Recorded per geometry: perimeter count, cell count,
breakline count, refinement-region count, whether CRS is embedded in the HDF.

**Findings.**
- All 12 cover the same study area (same perimeter bounds, ~22.5 mi × 15.4 mi).
- **Only `g09` embeds CRS in the HDF** (EPSG:2271). All others (and Muncie's
  g04) require CRS to be sourced from the terrain TIFF.
- **Zero refinement regions across all 12 geometries.** Phase A pilots produce
  no refinement-region labels — that's bulk-corpus work (Stage 3).
- Geometry breakline counts varied (`g01`: 1, `g02`: 7, `g09`: 4, others
  smaller).

**Decision: g09 canonical, g02 secondary.**

My initial recommendation was g02 ("richest labels" — 7 vs g09's 4). User
flipped to g09 (the recommendation in the chat options) after I surfaced that:
- g02's 7 breaklines include 4 generic `Breakline 1-4` flow-alignment helpers,
  semantically noisy.
- g09's 4 are *all* semantically named: `SayersDam` (the headline feature of
  the *dam-break* study), `Lower`/`Middle`/`Upper` (channel segments).
- Per ADR 003 (amended), expert meshes are a prior, not ground truth. Cleaner
  prior signal > more noisy prior signal.

**Pedagogical points:**
- The first contact with a new geospatial dataset is always inventory before
  decision. Don't trust file names; open and measure.
- "Most labels" and "best labels" are different optimization targets.

**Exercise.** Open any HEC-RAS official example with rashdf and produce an
inventory table by geometry. Note how often CRS is embedded.

---

## Task 2 — DEM derivatives

The Stage 1 spec named 7 candidate features (slope, aspect, plan curvature,
profile curvature, topographic wetness index, flow accumulation, hydro-
conditioned ridges). Picked the **minimal set** (slope + aspect + plan/profile
curvature) for the pilot — pure-numpy formulas, no external deps beyond
scipy. The remaining 3 (TWI, flow accumulation, ridges) need pit-filling
+ a flow-direction algorithm + a specialized library (`pysheds` or
`whitebox`); deferred until Stage 2 demands more signal.

**Concepts to introduce before code:**

- **Feature channel.** A 2D array of per-pixel values. The U-Net's input is a
  tensor of shape `(C, H, W)` = "C channels stacked over an H×W grid." Each
  channel hands the model a different physical property of the terrain at
  each pixel.
- **Why hand-engineered features?** A model with only the raw DEM has to
  *infer* slope/curvature internally from spatial context. Pre-computing
  them is free signal and convergence stability — especially useful at the
  pilot scale where we don't have millions of examples.

### 2a — Slope (`src/hecras_mesh_ai/features/slope.py`)

**Math: Horn (1981).** Standard GIS slope formula. For the 3×3 neighborhood

$$
\begin{matrix} a & b & c \\ d & e & f \\ g & h & i \end{matrix}
$$

centered on a pixel, with cellsize $(\Delta x, \Delta y)$:

$$\frac{\partial z}{\partial x} = \frac{(c + 2f + i) - (a + 2d + g)}{8 \Delta x}$$

$$\frac{\partial z}{\partial y} = \frac{(g + 2h + i) - (a + 2b + c)}{8 \Delta y}$$

$$\text{slope} = \arctan\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right)$$

Horn is what GRASS / ArcGIS / gdaldem / rasterio all default to — interoperable.
The alternative (Zevenbergen-Thorne 1987) is sharper at small features but
noisier in flat areas; Horn is the safer default.

**Why slope is the highest-signal feature for breaklines.** Breaklines align
with ridges, channel banks, levees, road embankments — every one of these is
a place where slope changes abruptly. A high-slope band against a low-slope
flat is exactly what a breakline should track.

**API.** Pure numpy: `slope(z, cellsize_x, cellsize_y, *, units="degrees") -> np.ndarray`.
Returns same shape as input; border rows/cols are NaN (the 3×3 window can't
center on the boundary). Units selectable: degrees / radians / percent.

**Tests** (`tests/features/test_slope.py`, 6 tests):
- Flat plane → slope = 0 everywhere (closed-form).
- Tilted plane (z = column index, dx=dy=1) → slope = 45° everywhere.
- Same plane with `cellsize_x=2` → slope = atan(0.5) ≈ 26.57° (verifies
  cellsize is respected).
- Three unit systems are internally consistent (radians/degrees/percent
  round-trip).
- Borders are NaN.
- Invalid input raises (1D array, negative cellsize, bad units).

**Pedagogical points:**
- Pure-numpy signatures keep math testable. The geospatial plumbing
  (rasterio, xarray, CRS) lives in a separate layer.
- "NaN at the border" is honest: we genuinely don't know the slope where
  the stencil can't fit. The naive alternative (extend by zeros) lies.

**Exercise.** Apply this slope function to a 100×100 sample of any DEM.
Plot the result. Pixels along ridges should glow.

### 2b — Aspect as (sin, cos) two-channel encoding (`features/aspect.py`)

**Aspect** is the compass direction the slope faces — the downslope direction
projected onto the horizontal plane. GIS convention: azimuth clockwise from
north. 0° = N, 90° = E, 180° = S, 270° = W.

**The problem: aspect is a circular variable.** A pixel at 359° and a pixel
at 1° face nearly the same direction but their numerical difference is 358°.
Feeding aspect-in-degrees to a CNN forces it to learn a wraparound it cannot
represent. The standard ML treatment for circular variables is to encode
them as a 2D unit vector on the circle: `(sin θ, cos θ)`. Two channels
instead of one. Smooth everywhere. Adjacent directions stay adjacent.

**At flat points, aspect is mathematically undefined.** The downslope
direction doesn't exist if there's no slope. Three options for what to
return:
1. NaN (honest but propagates through training and corrupts the loss).
2. A sentinel like -1 (lies about "no direction" being a specific direction).
3. `(0, 0)` (the origin of the unit disk, semantically "no preferred
   direction" — neither north nor east). This is what the sin/cos encoding
   naturally produces in the limit, and it's a valid value the model can
   learn from (magnitude=0 ↔ flat).

**Decision: option 3.** Zero magnitude *is* the meaning we want; there's no
ambiguity.

**Math.** Given Horn first derivatives $p = \partial z / \partial x$ and
$q_{\text{array}} = \partial z / \partial y_{\text{row}}$, the downslope
direction in geographic coordinates is $(-p, +q_{\text{array}})$ (the
geographic-y component flips sign because row index increases southward in
north-up rasters). Magnitude $m = \sqrt{p^2 + q_{\text{array}}^2}$. Then

$$\sin(\text{aspect}) = -p / m, \quad \cos(\text{aspect}) = q_{\text{array}} / m$$

at sloped points, and $(\sin, \cos) = (0, 0)$ at flat points.

**The CONVENTION-TO-VERIFY breadcrumb.** Row direction is the subtle bit. In
a north-up raster (rasterio `transform.e < 0`), row index increases
*southward*. If you feed in a legacy row-up DEM (transform.e > 0), the sign
flip is wrong and aspect comes out rotated 180°. Documented in three places:
the module docstring, STATUS.md, and a loud runtime assertion in the
stacker. Three-layer defense against a hard-to-debug regression.

**The bug we caught** (`fix: propagate NaN through aspect output`, commit
`c2446ff`). Original code:
```python
sin_interior = np.zeros_like(magnitude)
nonflat = magnitude > 0
sin_interior[nonflat] = east_component / magnitude[nonflat]
```
NaN comparisons return False. A nodata pixel → `magnitude = NaN` →
`nonflat = False` → `sin_interior` stays at its default zero. **Result:
nodata silently encoded as flat.** Exactly the wrong-number failure mode
the sin/cos encoding was designed to prevent.

Fix:
```python
with np.errstate(invalid="ignore", divide="ignore"):
    sin_interior = east_component / magnitude  # NaN propagates naturally
flat = magnitude == 0
sin_interior = np.where(flat, 0.0, sin_interior)  # genuine flats only
```

**Horn nuance the bug surfaced.** Horn's first-derivative stencil uses **8
of the 9** cells in the 3×3 window — it skips the *center*. So an isolated
NaN at cell `(i, j)` corrupts the 8 surrounding cells (their stencils
consume the NaN) but does NOT corrupt cell `(i, j)` itself (its 8
neighbors are clean, the center is unused). Documented in
`test_aspect.py::test_nan_input_propagates_to_nan_output_not_zero_vector`.

**9 tests** cover: flat → (0,0); each of N/S/E/W cardinal directions;
diagonal 45° face; unit-circle invariant on random DEM (every interior
pixel is either at origin or on the unit circle); NaN-only-on-borders;
explicit "no interior NaN for finite input"; the propagation test above;
invalid input.

**Pedagogical points:**
- Circular variables → multi-channel unit-vector encoding. Same trick is
  used for time-of-day, day-of-year, joint angles in robotics, etc.
- A "silent wrong number" in a feature pipeline is far more dangerous
  than a loud error. Test for the failure mode that hurts most.
- Horn's center-exclusion is documented robustness, not a bug. Different
  stencils have different NaN footprints.

**Exercise.** Construct a 5×5 DEM with one NaN pixel in the middle. Run
aspect_sincos. Verify by hand that the 8 surrounding cells are NaN and
the center cell is (0, 0).

### 2c — Plan curvature (`features/plan_curvature.py`)

**Plan curvature** is the curvature of the contour line through a pixel —
the bend of the horizontal slice through the terrain at that elevation.
Sign convention (matches ArcGIS / SAGA / gdaldem):
- **Positive** = divergent contour (ridge-like). Flow disperses laterally.
- **Negative** = convergent contour (valley-like). Flow concentrates.
- **Zero** = planar flow.

**Math (Heerdegen & Beran 1982).** With first partials $p, q$ (central
differences) and second partials $r = \partial^2 z / \partial x^2$,
$t = \partial^2 z / \partial y^2$, $s = \partial^2 z / \partial x \partial y$:

$$k_{\text{plan}} = -\frac{q^2 r - 2 p q s + p^2 t}{(p^2 + q^2)^{3/2}}$$

Closed-form check: for the paraboloid bowl $z = x^2 + y^2$, the formula
collapses to $k_{\text{plan}} = -1/r$ at radius $r$ from center. Negative
everywhere off the origin — valley-like. The test verifies this at three
sampled lattice points.

**Why both plan and profile curvature?** Plan tells you "ridge vs valley"
(transverse to flow). Profile tells you "accelerating vs decelerating"
(along flow). The pair gives the geomorphic anatomy of where breaklines
belong.

**The bug we caught** (`feat: add plan curvature`, commit `b42ad18`). Plan
curvature uses all 9 cells of the 3×3 (unlike Horn's first derivative
which skips the center). So NaN propagation needs all 9 checked. Naive
implementation:
```python
flat = denom_sq == 0
interior = np.where(flat, 0.0, numerator / denom_sq**1.5)
```
But: a "flat-on-its-axes" pixel can have `p = q = 0` cleanly while the
*cross*-partial `s` carries NaN from a corner of the stencil. Then
`denom_sq == 0` is True, `flat` is True, and `np.where` overrides the
NaN to 0. We've washed out the nodata signal.

Fix: explicit `nan_in_stencil` mask checking all 9 cells:
```python
nan_in_stencil = isnan(z1) | isnan(z2) | ... | isnan(z9)
flat = (denom_sq == 0) & ~nan_in_stencil
interior = np.where(flat, 0.0, interior)
interior = np.where(nan_in_stencil, np.nan, interior)
```

**y-flip invariance.** Plan curvature is invariant under y-axis flip
(both `q` and `s` change sign and they appear as `q²`, `pqs`). So the
row-direction CONVENTION-TO-VERIFY does NOT apply to curvature, only to
aspect. Different math, different NaN footprint, different conventional
sensitivities — each documented per module.

**8 tests** including flat → 0, bowl → negative, dome → positive,
closed-form match at sampled points, y-flip invariance, NaN propagation,
borders NaN, invalid input.

**Pedagogical points:**
- A "flat" check that conflates "denominator zero" with "no data" is a
  common foot-gun in numerical code. Always check both conditions.
- Different formulas using the same partial derivatives can have very
  different NaN/invariance properties. Test each independently.

### 2d — Profile curvature (`features/profile_curvature.py`)

**Profile curvature** is the curvature of the surface in the direction of
steepest descent — the vertical-section bend of the slope line.

- **Positive** = upwardly convex profile (∩). Slope steepens downhill →
  **flow accelerates**.
- **Negative** = upwardly concave profile (∪). Slope flattens downhill →
  **flow decelerates**.
- **Zero** = linear slope.

**Math (Mitasova & Hofierka 1993):**

$$k_{\text{prof}} = -\frac{p^2 r + 2 p q s + q^2 t}{(p^2 + q^2) (1 + p^2 + q^2)^{3/2}}$$

The $(1 + p^2 + q^2)^{3/2}$ factor is the surface metric — profile
curvature lives on the curved surface, not its planar projection. For
shallow slopes the factor → 1; for steep slopes it suppresses the
magnitude. Sign convention matches ArcGIS.

Closed-form check on the bowl: $k_{\text{prof}} = -2 / (1 + 4r^2)^{3/2}$.
Tested at three sampled points.

**8 tests** including the same NaN, y-flip, and closed-form checks as
plan curvature.

---

## Task 3 — CRS-aware feature stacker

This is where pure-numpy math gets wrapped in geospatial plumbing.

### 3a — Hybrid NaN conditioner (`features/conditioning.py`)

**Problem.** Real DEMs have nodata. PyTorch (and any CNN) crashes on NaN —
NaN in input → NaN in loss → NaN in gradient → broken training. Two
regimes of nodata coexist:
1. Accidental small holes (single corrupted pixels, interpolation artifacts
   at tile seams, isolated mis-classified water).
2. Intentional large voids (water bodies that were never measured, missing
   tiles, off-survey areas).

**The hybrid rule (user's design).** Patch a NaN connected component if its
size is ≤ `max_patch_size` (default 1 — only truly isolated pixels). Larger
regions stay NaN. The Stage 5 tile sampler will refuse tiles that overlap
NaN regions, so the model never sees them.

**Implementation.** `scipy.ndimage.label` with 8-connectivity (conservative —
two diagonally-adjacent NaN cells form a size-2 component and stay NaN).
Fill value is the **mean of valid 8-neighbors, computed from the *original*
input** so patch order can't bias the result.

**API.** `patch_isolated_nan(z, *, max_patch_size=1, connectivity=2) -> np.ndarray`.

**13 tests:** no-op on clean input, single-cell patch with varied
neighbors, adjacency preservation (8-conn and 4-conn variants), 3×3 block
preservation, many-isolated, max_patch_size=0 disables, max_patch_size=2
patches a pair, corner-pixel patching, order-independence, invalid input.

**Pedagogical points:**
- "Don't fabricate data" is a hard rule. The exception (1-pixel patch) is
  defensible because the alternative is throwing away a 3×3 stencil of
  good neighbors over a single missing value.
- A configurable threshold (`max_patch_size`) makes the policy *explicit*
  in the API. The caller can audit the choice. The model isn't subject to
  a hidden hand-tuned constant.

### 3b — The stacker (`features/stacker.py`)

**`stack_dem_features(dem_path, *, patch_max_size=1, slope_units="degrees") -> xr.DataArray`**

Pipeline:
1. Open the DEM with rasterio; convert nodata → NaN via `read(masked=True)`.
2. **Assert `transform.e < 0`** — the runtime check for the CONVENTION-TO-
   VERIFY breadcrumb. A row-up DEM raises `ValueError` with a message that
   points at the diagnostic recipe in aspect.py.
3. Apply `patch_isolated_nan`.
4. **Reflect-pad the DEM by 1 pixel** on all sides. This handles the
   natural stencil-edge: after deriving on the padded array and cropping
   back, the output has the same spatial extent as the input with valid
   derivatives all the way to the boundary.
5. Compute slope, aspect (sin/cos), plan curvature, profile curvature on
   the padded array.
6. Crop derivatives back to the original DEM shape.
7. Stack with the patched elevation as band 0 → shape `(6, H, W)`.
8. Wrap as `xarray.DataArray` with named bands, pixel-center coordinates,
   and CRS + transform attached via rioxarray's `.rio` accessor.

**6 feature channels** in canonical order:
```python
FEATURE_CHANNELS = (
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "plan_curvature",
    "profile_curvature",
)
```

**10 tests** including: shape/dims/channels, flat-DEM → all-zero derivatives
with no NaN border (proves reflect-pad works), CRS preservation, pixel-
center coordinate math against the transform, isolated-NaN patching,
3×3 NaN block preservation, NaN propagation to all channels, row-up
DEM raises with the CONVENTION breadcrumb in the error message, plus
integration tests against the real Muncie (EPSG:2965) and Bald Eagle
(EPSG:2271) DEMs — both run end-to-end.

**Pedagogical points:**
- A "loud runtime check at ingest" beats a "silent corruption at train
  time" every time.
- Reflect-padding before deriving + cropping after is the standard GIS
  trick to avoid border NaN.
- xarray + rioxarray gives you a "labeled tensor" with geospatial
  metadata — much better than passing parallel `(array, transform, crs)`
  tuples around.

---

## Task 4 — Breakline rasterizer (`labels/breaklines.py`)

This produces the **label (y)** for the supervised training pair. The feature
stack from Task 3 is the input (x); the rasterized breaklines are what the
model learns to predict.

### Concept: why breaklines need to be "thickened" in the label

Breaklines as expert-drawn polylines are 1D objects with zero physical
thickness. But the label raster is a grid of pixels. Three reasons we want
each breakline to occupy a multi-pixel-wide band, not just one pixel:

1. **Class imbalance.** A 1-pixel-thick breakline on a 5000×5000 raster is
   0.01% of all pixels. A model that just predicts "0 everywhere" scores
   99.99% accuracy. Even with a class-weighted loss, gradient signal from
   such sparse positives is weak. Thicker labels = more positive pixels =
   tractable loss landscape.

2. **Expert breaklines aren't pixel-perfect.** A different expert might
   draw the same conceptual breakline 5 ft to the side. Per ADR 003
   amended, the expert mesh is a *prior*, not ground truth. A thicker
   label is more honest about what we actually want the model to learn:
   "a breakline goes somewhere around here," not "at exactly this pixel."

3. **Model outputs are inherently smooth.** A U-Net's probability map
   can't be perfectly sharp — convolutions blur. Training with a 3–5
   pixel band matches the model's expressive capacity.

The thickness parameter (`buffer_width`) is the **central hyperparameter**
of the labeling step. NOT baked in — exposed at the API.

### Implementation

`rasterize_breaklines(breaklines, *, out_shape, transform, target_crs, buffer_width) -> np.ndarray`

Algorithm:
1. If `breaklines.crs is not None` and differs from `target_crs`, reproject.
   If `crs is None` (common for rashdf on HDFs without embedded CRS),
   assume already aligned — documented.
2. `shapely.buffer(buffer_width / 2)` on each LineString → polygon.
3. `rasterio.features.rasterize(shapes, ..., all_touched=True)` — any pixel
   the polygon touches becomes 1. `all_touched` matters for narrow buffers
   on diagonal alignments where center-only rasterization would leave gaps.

Output: `np.ndarray[uint8]` of shape `(H, W)`, values in `{0, 1}`.

**13 tests:** empty GDF, single-line band position math, buffer-width-
controls-thickness, out-of-extent yields zeros, crs=None proceeds without
reprojection, EPSG:4326 → EPSG:32633 reprojection, MultiLineString
handling, two crossing lines union, output is strictly binary, invalid
input, plus integration tests against the real Muncie (2 breaklines) and
Bald Eagle g09 (4 named breaklines) — both produce nonzero label rasters
aligned with the feature stack.

**Pedagogical points:**
- Terminology bridges matter. "Buffer width" is GIS-standard;
  hydraulically the parameter controls the "thickness" of the band
  perpendicular to the breakline. Both terms point at the same number;
  the docstring uses GIS convention with a clarifying phrase ("band of
  this total width centered on the polyline").
- The choice of buffer width is the first knob the modeler will want to
  tune. Exposing it as a function parameter (not a module-level constant)
  is what makes it tunable.

---

## Task 5 — Cache, dataset, sampler, split

This is the heaviest task in Stage 1 — three commits, three classes, the
spatial-holdout machinery, and the bridge into PyTorch.

### Concept primers

**Tiles, not whole projects.** The Bald Eagle feature stack is `(6, 6902, 8643)`
≈ 60 million pixels per channel × 6 channels × 4 bytes = ~1.4 GB. The 24 GB
RTX 3090 can't process that in one forward pass. Standard CNN training:
randomly sample **tiles** (e.g. 256×256 = 65k pixels = ~1.6 MB per sample)
from the project. Tens of thousands of tiles per training run. Each tile is
one training example.

**Spatial-holdout vs random-split.** The classical ML reflex is "shuffle
examples, split 80/20." Apply that to tiles cut from a continuous landscape
and most val tiles overlap or border train tiles by accident. The model
memorizes the neighborhood rather than learns the phenomenon. Validation
scores look great; production scores tank.

Spatial-holdout carves the *map* into train and val regions geographically.
For our pilot, the cleanest split is **per-project**: train Bald Eagle (PA),
val Muncie (IN). Different states, different CRS, mathematically disjoint.
A stronger generalization signal than within-project splits.

**The train/val direction debate.** I initially recommended train Muncie /
val Bald Eagle (it had less label data, but the small train set fits the
"deliberately overfit" pilot ethos). User pushed back: more training data
→ richer prior → train on the *larger* labeled set, val on the smaller.
Better logic. Agreed and reversed.

### 5a — Cache (`dataset/cache.py`)

`cache_pilot_project(*, project_name, dem_path, geometry_hdf_path, buffer_width, out_dir, patch_max_size=1) -> CachedPaths`

Composes Task 3 (stack_dem_features) and Task 4 (rasterize_breaklines),
writes both to GeoTIFFs under `<out_dir>/<project_name>/`:
- `features.tif` — 6 bands, float32, FEATURE_CHANNELS order, tiled 256×256
  with deflate compression.
- `labels.tif` — 1 band, uint8, values in {0, 1}, same tiled layout.

Both share the source DEM's CRS, transform, and shape exactly. The tiled
GeoTIFF layout matches the rasterio-window-read pattern the sampler uses
— window reads are cheap.

**Why GeoTIFFs rather than in-memory?** User picked GeoTIFFs: avoids the
Stage 3 refactor when bulk corpus arrives, follows TorchGeo's textbook
path. Trade-off: ~3-min cache time per project, then disk I/O each tile
read. Acceptable for N=2 pilot.

**3 integration tests** against the real pilots: cache Muncie (EPSG:2965),
cache Bald Eagle g09 (EPSG:2271), plus a round-trip test asserting the
cached features read back match the in-memory `stack_dem_features` output
to float32 precision (rtol=1e-5).

### 5b — Tile dataset + sampler (`dataset/tile_dataset.py`)

Three small classes, each with one job:

**`RasterTileDataset(features_path, labels_path)`** — opens both GeoTIFFs,
verifies alignment (same CRS, transform, shape), exposes `bounds`, `crs`,
`shape`, `cellsize_x/y`. Method `sample(bbox) -> (features_tensor, labels_tensor)`
reads via `rasterio.windows.from_bounds`. **Opens rasterio per call** —
not the persistent-handle pattern — for PyTorch DataLoader multi-worker
safety. The tiled GeoTIFF layout makes the per-call open cheap.

```python
def sample(self, bbox: BBox) -> tuple[torch.Tensor, torch.Tensor]:
    if not self._bbox_inside_bounds(bbox):
        raise ValueError(...)
    window = from_bounds(*bbox, transform=self.transform)
    with rasterio.open(self.features_path) as f:
        features = f.read(window=window).astype(np.float32)
    with rasterio.open(self.labels_path) as la:
        labels = la.read(1, window=window).astype(np.float32)
    return torch.from_numpy(features), torch.from_numpy(labels)
```

Labels return as **float32 in {0.0, 1.0}** (not int) — matches what
`BCEWithLogitsLoss` (Stage 2's expected loss function) wants.

**`RandomTileSampler(dataset, *, tile_size_pixels=256, samples_per_epoch=1000, seed=None)`**
— yields random tile bounding boxes that fit entirely inside the dataset
bounds. Tile size in pixels, converted to CRS units via the dataset's
cellsize. Deterministic given a seed (critical for reproducible training).

**`IterableTileDataset(raster_dataset, sampler)`** — a tiny
`torch.utils.data.IterableDataset` adapter. Yields `(features, labels)`
tensor pairs. Wrap in a vanilla `torch.utils.data.DataLoader` to drive
batches.

**11 synthetic tests** cover: metadata exposure, CRS/shape mismatch raises,
sample tensor shape/dtype, out-of-bounds bbox raises, sampler bbox-inside-
bounds invariant (sampled 50 times), sampler determinism by seed, oversized
tile raises, invalid args, IterableTileDataset round-trip, plus a
**DataLoader batch integration test** confirming the full PyTorch path
works — batch shape `(4, 6, 16, 16)` features + `(4, 16, 16)` labels.

### 5c — Spatial-holdout check (`dataset/split.py`)

`assert_no_spatial_overlap(train, val) -> None`

The gate for Stage 1 checkpoint criterion #4 ("zero overlap, leakage
check passes"):

- Different CRS → trivially passes (different projected coordinate systems
  map to disjoint regions of Earth).
- Same CRS, disjoint bboxes → passes.
- Same CRS, edge-touching (zero-area intersection) → passes.
- Same CRS, overlapping bboxes → raises `ValueError`.

For our pilot the cross-CRS path is the active one. The same-CRS machinery
is what we'll need at Stage 3 for within-project bulk-corpus splits.

**7 tests** cover all four cases plus identical-bounds and full-containment.

### What Stage 1 unlocks: the canonical Stage 2 training setup

```python
train_paths = cache_pilot_project(
    project_name="bald_eagle_g09",
    dem_path=BALDEAGLE_DEM, geometry_hdf_path=BALDEAGLE_HDF,
    buffer_width=50.0, out_dir=CACHE_DIR,
)
val_paths = cache_pilot_project(
    project_name="muncie",
    dem_path=MUNCIE_DEM, geometry_hdf_path=MUNCIE_HDF,
    buffer_width=20.0, out_dir=CACHE_DIR,
)

train_ds = RasterTileDataset(train_paths.features, train_paths.labels)
val_ds   = RasterTileDataset(val_paths.features, val_paths.labels)

assert_no_spatial_overlap(train_ds, val_ds)  # passes — different CRS

train_sampler = RandomTileSampler(train_ds, tile_size_pixels=256,
                                  samples_per_epoch=1000, seed=42)
train_loader = DataLoader(IterableTileDataset(train_ds, train_sampler),
                          batch_size=8)

for features, labels in train_loader:
    # features: (8, 6, 256, 256) float32
    # labels:   (8, 256, 256)    float32 in {0, 1}
    ...  # Stage 2 puts a U-Net here
```

---

## Task 6 — Exit notebook (`notebooks/03_stage1_exit_features_and_labels.ipynb`)

The end-to-end visual + programmatic verification of every Stage 1
checkpoint criterion. Three figures (attached as PNGs) cover the visual
verification.

### Figure 1 — `01-bald-eagle-full-project.png`

Bald Eagle g09 (TRAIN). Terrain hillshade with the binary label raster
overlaid in red and expert breaklines drawn as dashed black lines on
top. The red bands should trace the four named breaklines (SayersDam
across the dam itself, Lower/Middle/Upper along the channel system).

**Proves checkpoint #3** (label raster aligns pixel-for-pixel with the
feature stack) on the train pilot, by eye: the red rasterized bands
should sit *exactly* on the dashed expert polylines.

### Figure 2 — `02-muncie-full-project.png`

Muncie (VAL). Same overlay scheme — terrain + red label bands + dashed
breakline polylines. Muncie has only 2 breaklines (`Road 1` and
`HighGround 1`) so the figure is sparser. The red bands should sit on
the road embankment and the ridge.

**Proves checkpoint #3** on the val pilot.

### Figure 3 — `03-tile-level-features-and-label.png`

A 2×4 panel showing one random 256×256 tile from the train set that
contains breakline content. Panels 1-6 are the six feature channels
(elevation, slope, aspect_sin, aspect_cos, plan_curvature, profile_curvature)
each with its own colorbar. Panel 7 is the binary label mask. Panel 8 is
elevation with the label overlaid in red — the **alignment check at tile
resolution**.

**Proves checkpoint #2** (feature channels aligned in CRS / resolution /
extent) — same grid, same extent across all 6. **Proves checkpoint #3
at tile level** — the red label overlay in panel 8 sits exactly where
the breakline goes.

Notable: this tile has ~0.9% positive pixels (586 / 65536). That's the
**realistic breakline class imbalance** the Stage 2 loss function must
handle.

### A surprising finding from the DataLoader demo

The exit notebook's cell 18 pulls a vanilla DataLoader batch with
`seed=0` and prints per-sample positive counts. **All 4 samples in the
demo batch had 0 positive pixels.** A perfect proof-of-need for Stage
2's class-imbalanced loss: random sampling will frequently miss
breaklines entirely on a project where only ~1% of pixels are positive.
Stage 2's BCE + Dice combo is designed for exactly this — Dice loss
heavily weights the rare positive class.

---

## Bugs caught and what they teach

Summary table for the chapter's "lessons learned" section:

| Bug | Class | Lesson |
|---|---|---|
| Aspect silently encoded NaN as (0, 0) at flat-magnitude pixels | NaN-comparison-returns-False foot-gun | The "no preferred direction" output and the "no data" output should never be the same number. Test that explicitly. |
| Plan curvature `flat = (denom == 0)` washed NaN to 0 when cross-partial s carried it | Two conditions confused into one | When a default-value override fires on a derived predicate, audit whether the predicate also fires for "the data isn't there" cases. |
| Pre-commit hook missing at git-hook level after OneDrive→C:\dev migration | Tooling state isn't versioned | After environment moves, verify the things that live *outside* the repo (git hooks, env vars, cached credentials) survived. |
| f-string with `\"` escapes inside `{...}` brackets parses as illegal Python | Generator-script hygiene | Single quotes inside double-quoted f-strings, or vice versa. Don't escape. |

---

## Code-tour reference

Final module layout after Stage 1:

```
src/hecras_mesh_ai/
├── __init__.py
├── features/
│   ├── __init__.py              # public API
│   ├── slope.py                 # Horn 1981
│   ├── aspect.py                # sin/cos two-channel + CONVENTION-TO-VERIFY
│   ├── plan_curvature.py        # Heerdegen & Beran 1982
│   ├── profile_curvature.py     # Mitasova & Hofierka 1993
│   ├── conditioning.py          # patch_isolated_nan (hybrid NaN policy)
│   └── stacker.py               # stack_dem_features + CRS / transform / xarray
├── labels/
│   ├── __init__.py
│   └── breaklines.py            # rasterize_breaklines
└── dataset/
    ├── __init__.py
    ├── cache.py                 # cache_pilot_project
    ├── tile_dataset.py          # RasterTileDataset + RandomTileSampler + IterableTileDataset
    └── split.py                 # assert_no_spatial_overlap
```

`tests/` mirrors this layout. ~75 tests total, ~3 min suite runtime
(dominated by cache integration tests writing 60 Mpx GeoTIFFs).

---

## Glossary of ML terms introduced

(Use this as a checklist — make sure each is explained the first time it
appears in the chapter.)

- **Tensor / channel / (C, H, W)** — multi-dimensional array with named
  axes; CNNs consume `(C, H, W)` for single examples and `(B, C, H, W)`
  for batches.
- **Feature** vs **label** — input (x) vs target (y) in supervised
  learning. Features are what the model sees; labels are what it
  predicts.
- **Segmentation** — per-pixel classification. Output shape matches input
  shape; each pixel gets a class (binary here: breakline / not).
- **U-Net** — the standard encoder-decoder CNN architecture for segmentation.
  Stage 2 will introduce it properly.
- **Hyperparameter** — knobs the user sets before training (learning rate,
  batch size, buffer_width, etc.). The model learns parameters; the user
  picks hyperparameters.
- **Class imbalance** — when one class dominates the dataset (here, ~99%
  non-breakline pixels). Vanilla accuracy collapses to "predict majority."
- **BCE + Dice loss** — Binary Cross-Entropy (smooth per-pixel) plus Dice
  (set-overlap, heavily weights the rare class). Stage 2 will introduce.
- **NaN propagation** — IEEE 754 says NaN propagates through arithmetic
  but NaN comparisons return False. This combination is the source of
  several subtle bugs in feature pipelines.
- **Stencil** — the neighborhood pattern a finite-difference formula
  uses. Horn slope is a 3×3 stencil; specifically an 8-cell stencil
  because it doesn't use the center.
- **CRS / projection / EPSG code** — coordinate reference system; how
  raster pixels map to Earth locations.
- **Affine transform** — the 6-parameter matrix mapping pixel (row, col)
  to (x, y) in the CRS. `transform.e < 0` means row index increases
  southward (the convention).
- **Spatial holdout** — train/val split that carves the *map* into
  disjoint regions, not just the *examples*.
- **DataLoader** — PyTorch's batched-iteration adapter over a Dataset.
  Handles workers, shuffling, batch collation.
- **Iterable Dataset** — the streaming flavor of PyTorch dataset (no
  fixed length / indexing). Pairs with our random sampler.
- **Class-weighted loss** — penalizing missed positives more than missed
  negatives to overcome class imbalance.

---

## Suggested chapter exercises

(Embed these in the appropriate sections of the textbook.)

1. **Slope intuition.** Apply slope to a 100×100 sample of any DEM you
   have. Plot the result with a sequential colormap. Identify a feature
   you know is a ridge — does it appear as high slope?
2. **Aspect circularity.** Construct a 5×5 DEM with a known tilt
   direction. Run `aspect_sincos`. Compute `degrees(atan2(sin, cos))` and
   verify you recover the original aspect.
3. **NaN robustness.** Construct a 7×7 DEM with one isolated NaN pixel
   and one 3×3 NaN block. Run all four derivatives. Predict by hand which
   output pixels will be NaN. Verify.
4. **Plan vs profile curvature.** For the paraboloid bowl, derive both
   $k_{\text{plan}}$ and $k_{\text{prof}}$ analytically at radius $r=1$.
   Verify your derivation matches the formulas in the modules.
5. **Buffer width sensitivity.** Rasterize the same set of breaklines
   with `buffer_width = 5, 20, 50, 100`. Plot the resulting label masks
   side by side. Where does each become "too thin to learn from" vs "too
   thick to localize"?
6. **Tile sampling.** Open one of the cached pilots, draw 200 random
   256×256 tiles, count how many contain any breakline pixel. What
   fraction? What does that imply for training-batch composition in
   Stage 2?
7. **Spatial holdout.** Construct two overlapping bounding boxes in the
   same CRS. Call `assert_no_spatial_overlap`. Then make them disjoint
   and call it again. What's the error message?

---

## What to produce

A self-contained Markdown chapter that an engineering-strong, ML-new
hydraulic modeler could read end-to-end and come away understanding:

- What every Stage 1 module does and why it exists.
- The math behind each DEM derivative, with intuition.
- Why the (sin, cos) encoding is the right answer for aspect.
- Why the hybrid NaN policy is the right answer for nodata.
- What spatial holdout means and why it matters.
- How tiles + samplers + DataLoader compose into a training pipeline.
- The shape and dtype of every tensor that flows through the pipeline.
- The bugs we caught and the general principles they illustrate.

Length: as long as it needs to be. A typical chapter at this depth is
8,000–15,000 words. Include the three attached figures inline where they
illustrate the relevant section.

Begin.
