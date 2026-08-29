"""Thacker (1981) parabolic bowl benchmark — analytical shallow-water solution.

The Thacker bowl is a paraboloid-shaped basin where a tilted water surface
oscillates back and forth periodically. The depth, velocity, and shoreline
position have a closed-form exact solution, making it the standard
verification case for 2D shallow-water solvers with wetting/drying.

Reference:
    Thacker, W.C. (1981). "Some exact solutions to the nonlinear
    shallow-water wave equations." J. Fluid Mech., 107, 499-508.

Implementation follows Thacker's Case III (planar-surface oscillation in
a paraboloid), adapted to US customary units for HEC-RAS.

The solution is derived from first principles: the SWE on a paraboloid
admit an exact solution where the free surface is a rigid plane
oscillating back and forth. With the ansatz eta = c(t) + p(t)*x and
spatially-uniform velocity u = u(t), the SWE reduce to ODEs for p, u,
and c. The solution is:

    p(t)  = p0 * cos(wt)                   (surface tilt)
    u(t)  = -a^2 * p0 * w * sin(wt) / (2*D0)  (uniform velocity)
    c(t)  = c0 + a^2 * p0^2 * sin^2(wt) / (4*D0)  (mean level adjustment)

where w = sqrt(2*g*D0) / a and p0 = 2*A*D0/a parameterizes the
initial tilt amplitude.

c0 has a closed form. Requiring the total volume to equal the still-water
volume pi*D0*a^2/2 forces the wetted region to be a disk of radius
exactly `a`, translating rigidly with center x_c(t) = A*a*cos(wt), and

    c0 = -A^2 * D0.

(Completing the square in h = eta - z_b shows the wet region is always a
disk of radius R with R^2 = a^2*(c0 + D0)/D0 + a^2*A^2; volume
pi*D0*R^4/(2*a^2) equals the still-water volume iff R = a.)
Consequently the shoreline on the x-axis is exactly x_c(t) +/- a, and the
maximum shoreline excursion is (1 + A)*a — NOT the rim radius `a`. Any
raster meant to contain the benchmark must extend at least (1+A)*a from
the bowl center; `ThackerBowl.recommended_raster_extent` provides this
with margin.

HEC-RAS comparison protocol (IMPORTANT — read before using this as a
Stage 6 reference):

    The exact solution is FRICTIONLESS with full momentum. HEC-RAS
    cannot reproduce it exactly, so comparisons must follow a protocol:
    - Equation set: full Shallow Water Equations (SWE-ELM or SWE-EM).
      The default Diffusion Wave set has no local acceleration term and
      cannot slosh at all — the water simply settles. DWE runs of this
      benchmark are meaningless.
    - Manning's n: HEC-RAS requires n > 0. Use the smallest n the run
      tolerates (n = 0.001 is ~100x milder than n = 0.01 and viable for
      the first period away from the wet/dry fringe). Friction damping
      grows near the fringe where S_f ~ h^(-4/3) diverges.
    - Comparison window: the first oscillation period only. Expect and
      quantify amplitude decay from friction; do not attribute it to
      mesh error. The benchmark validates the ERROR-METRIC framework
      (does the computed error ordering behave as expected under grid
      refinement?), not HEC-RAS-vs-analytic agreement in the absolute.
    - Initial condition: t=0 is maximum tilt with u = 0 everywhere,
      matching HEC-RAS's zero-velocity WSE-raster initial condition.

Coordinate convention:
    - Origin (0, 0) at the center of the bowl (deepest point).
    - Bed elevation is negative at center, zero at the rim.
    - Positive x is the direction of initial surface tilt.
    - Free surface oscillates in the x-direction with period T.

Bed profile:
    z_b(x, y) = D0 * (r^2/a^2 - 1)    where r = sqrt(x^2 + y^2)

    z_b(0, 0) = -D0   (center, deepest)
    z_b(r=a)  =  0     (rim, at datum)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

G_FTPS2 = 32.174  # gravitational acceleration, ft/s^2


@dataclass(frozen=True)
class ThackerBowl:
    """Parameters and exact solution for the Thacker parabolic-bowl benchmark.

    Parameters
    ----------
    D0
        Maximum still-water depth at the bowl center [ft].
    a
        Basin radius — the distance from center where the still-water
        surface meets the rim (bed elevation = 0) [ft].
    A
        Oscillation amplitude parameter, dimensionless, 0 < A < 1.
        Larger = more violent sloshing; A = 0 is the still-water case.
        The surface tilt across the bowl is 2*A*D0 and the shoreline
        swings +/- A*a about the rim.
    g
        Gravitational acceleration [ft/s^2]. Default 32.174.
    """

    D0: float
    a: float
    A: float
    g: float = G_FTPS2

    # Derived constants, set in __post_init__ (not constructor arguments).
    _p0: float = field(init=False, repr=False, default=0.0)
    _c0: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        if self.D0 <= 0:
            raise ValueError(f"D0 must be positive; got {self.D0}")
        if self.a <= 0:
            raise ValueError(f"a must be positive; got {self.a}")
        if not 0 < self.A < 1:
            raise ValueError(f"A must be in (0, 1); got {self.A}")
        # p0 parameterizes the initial surface tilt.
        object.__setattr__(self, "_p0", 2.0 * self.A * self.D0 / self.a)
        # Volume conservation forces the wet disk radius to equal `a`,
        # which gives c0 in closed form (see module docstring).
        object.__setattr__(self, "_c0", -(self.A**2) * self.D0)

    @property
    def omega(self) -> float:
        """Angular frequency of the sloshing oscillation [rad/s]."""
        return np.sqrt(2 * self.g * self.D0) / self.a

    @property
    def period(self) -> float:
        """Oscillation period [s]."""
        return 2 * np.pi / self.omega

    @property
    def max_shoreline_extent(self) -> float:
        """Maximum distance the shoreline reaches from the bowl center [ft].

        The wet disk of radius `a` translates with center A*a*cos(wt),
        so the shoreline reaches (1 + A)*a at the extremes.
        """
        return (1.0 + self.A) * self.a

    @property
    def recommended_raster_extent(self) -> float:
        """Half-width for terrain/IC rasters: shoreline excursion + 10% margin."""
        return self.max_shoreline_extent * 1.1

    def _c(self, t: float) -> float:
        """Mean-level adjustment c(t) from the ODE solution."""
        return self._c0 + self.a**2 * self._p0**2 * np.sin(self.omega * t) ** 2 / (4.0 * self.D0)

    def _p(self, t: float) -> float:
        """Surface tilt p(t) = p0 * cos(wt)."""
        return self._p0 * np.cos(self.omega * t)

    def shoreline_center_x(self, t: float) -> float:
        """x-coordinate of the wet disk's center at time t [ft].

        x_c(t) = a^2 * p(t) / (2*D0) = A * a * cos(wt).
        """
        return self.A * self.a * np.cos(self.omega * t)

    def bed_elevation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bed (bottom) elevation at positions (x, y).

        z_b = D0 * (r^2/a^2 - 1).  Negative at center, zero at r=a.
        """
        r2 = x**2 + y**2
        return self.D0 * (r2 / self.a**2 - 1.0)

    def free_surface(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """Exact free-surface elevation eta(x, y, t).

        The surface is a plane at each instant: eta = c(t) + p(t)*x.
        Where the surface is below the bed, the water depth is zero
        (dry); this function returns the theoretical surface everywhere,
        callers clamp via `depth()`.
        """
        return self._c(t) + self._p(t) * np.asarray(x, dtype=np.float64)

    def depth(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """Exact water depth h(x, y, t) = max(0, eta - z_b)."""
        eta = self.free_surface(x, y, t)
        z_b = self.bed_elevation(x, y)
        return np.maximum(eta - z_b, 0.0)

    def wse(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """Exact water-surface elevation, NaN where dry.

        This is what HEC-RAS reports as "Water Surface" — the free
        surface where wet, undefined (NaN) where dry.
        """
        eta = self.free_surface(x, y, t)
        z_b = self.bed_elevation(x, y)
        wet = (eta - z_b) > 0
        return np.where(wet, eta, np.nan)

    def velocity_x(self, t: float) -> float:
        """Exact x-velocity at time t [ft/s].

        Spatially uniform (!) — the entire water body moves as a rigid
        mass. Derived from the ODE: u = a^2 * p'(t) / (2*D0), which
        equals -A * a * w * sin(wt).
        """
        return -(self.a**2) * self._p0 * self.omega * np.sin(self.omega * t) / (2.0 * self.D0)

    def velocity_y(self) -> float:
        """y-velocity is always zero (oscillation is in x only)."""
        return 0.0

    def shoreline_x_extent(self, t: float) -> tuple[float, float]:
        """x-coordinates of the shoreline on the x-axis (y=0) at time t.

        Closed form: the wet region is a disk of radius exactly `a`
        centered at x_c(t) = A*a*cos(wt), so the shoreline on y=0 is
        (x_c - a, x_c + a).
        """
        x_c = self.shoreline_center_x(t)
        return (x_c - self.a, x_c + self.a)

    def volume(self, t: float, *, n_points: int = 2000) -> float:
        """Numerically integrate the exact depth field to get total volume.

        Used for testing: volume must equal the still-water volume
        pi*D0*a^2/2 at all times. The integration window covers the full
        shoreline excursion (1+A)*a — a window clipped short of that
        silently underestimates the volume.
        """
        extent = self.max_shoreline_extent * 1.02
        xs = np.linspace(-extent, extent, n_points)
        ys = np.linspace(-extent, extent, n_points)
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        h = self.depth(X, Y, t)
        return float(np.sum(h) * dx * dy)

    def initial_wse_raster(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Initial water-surface elevation at t=0 for HEC-RAS IC setup.

        Returns the free surface where wet, bed elevation where dry
        (HEC-RAS needs a defined elevation everywhere for IC rasters).
        """
        eta = self.free_surface(x, y, 0.0)
        z_b = self.bed_elevation(x, y)
        wet = (eta - z_b) > 0
        return np.where(wet, eta, z_b)


def _make_grid(
    bowl: ThackerBowl,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> tuple[object, np.ndarray, np.ndarray, int, int]:
    """Build an exactly-registered sampling grid covering the benchmark.

    The half-width starts from `bowl.recommended_raster_extent` (full
    shoreline excursion + margin) and is rounded UP to a whole number of
    cells, then bounds are derived from the cell count so the pixel size
    is exactly `resolution`. Sample coordinates are pixel centers and
    match the geotransform exactly for any resolution — including ones
    that do not divide the nominal extent (the old code drifted up to
    ~0.7 cell at the far edge in that case).

    Returns (transform, local_x, local_y, n_rows, n_cols) where local_*
    are pixel-center coordinates relative to the bowl center.
    """
    half_cells = int(np.ceil(bowl.recommended_raster_extent / resolution))
    n_cols = n_rows = 2 * half_cells
    west = origin_x - half_cells * resolution
    north = origin_y + half_cells * resolution
    transform = from_origin(west, north, resolution, resolution)

    cols = np.arange(n_cols)
    rows = np.arange(n_rows)
    col_grid, row_grid = np.meshgrid(cols, rows)
    xs = west + (col_grid + 0.5) * resolution
    ys = north - (row_grid + 0.5) * resolution
    return transform, xs - origin_x, ys - origin_y, n_rows, n_cols


def _write_geotiff(
    data: np.ndarray,
    out_path: Path,
    transform: object,
    crs: CRS | str,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=CRS.from_string(crs) if isinstance(crs, str) else crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)
    return out_path


def generate_thacker_terrain(
    bowl: ThackerBowl,
    out_path: Path | str,
    *,
    resolution: float,
    crs: CRS | str = "EPSG:2965",
    origin_x: float = 500_000.0,
    origin_y: float = 1_800_000.0,
) -> Path:
    """Write the Thacker bowl bed elevation as a GeoTIFF.

    The raster is centered at (origin_x, origin_y) in the given CRS and
    extends past the full shoreline excursion (1+A)*a with a 10% margin
    (`bowl.recommended_raster_extent`), so the oscillating wet region is
    contained at every instant. Note this is substantially wider than
    the rim radius `a` — the water travels A*a beyond the rim.

    Parameters
    ----------
    bowl
        ThackerBowl instance defining the geometry.
    out_path
        Output GeoTIFF path.
    resolution
        Cell size in CRS linear units (ft for EPSG:2965). The written
        pixel size is exactly this value; the covered extent rounds up
        to a whole number of cells.
    crs
        Coordinate reference system. Default EPSG:2965 (Indiana East
        ftUS) to match Muncie for consistency.
    origin_x, origin_y
        CRS coordinates of the bowl center. Placed in the Muncie
        region so the same CRS works if you overlay them.

    Returns
    -------
    Path to the written GeoTIFF.
    """
    out_path = Path(out_path)
    transform, local_x, local_y, _, _ = _make_grid(bowl, resolution, origin_x, origin_y)
    z = bowl.bed_elevation(local_x, local_y).astype(np.float32)
    return _write_geotiff(z, out_path, transform, crs)


def generate_initial_wse_raster(
    bowl: ThackerBowl,
    out_path: Path | str,
    *,
    resolution: float,
    crs: CRS | str = "EPSG:2965",
    origin_x: float = 500_000.0,
    origin_y: float = 1_800_000.0,
) -> Path:
    """Write the t=0 water-surface elevation as a GeoTIFF for HEC-RAS IC.

    Same grid, extent, and registration guarantees as
    `generate_thacker_terrain` — the wet region is fully contained.
    """
    out_path = Path(out_path)
    transform, local_x, local_y, _, _ = _make_grid(bowl, resolution, origin_x, origin_y)
    wse = bowl.initial_wse_raster(local_x, local_y).astype(np.float32)
    return _write_geotiff(wse, out_path, transform, crs)
