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

c0 is determined by volume conservation: the total water volume must
equal the still-water volume pi*D0*a^2/2 at all times. Computed
numerically at initialization.

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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

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
    g
        Gravitational acceleration [ft/s^2]. Default 32.174.
    """

    D0: float
    a: float
    A: float
    g: float = G_FTPS2

    # Computed at init: c0 (mean-level constant for volume conservation)
    _p0: float = 0.0
    _c0: float = 0.0

    def __post_init__(self) -> None:
        if self.D0 <= 0:
            raise ValueError(f"D0 must be positive; got {self.D0}")
        if self.a <= 0:
            raise ValueError(f"a must be positive; got {self.a}")
        if not 0 < self.A < 1:
            raise ValueError(f"A must be in (0, 1); got {self.A}")
        # p0 parameterizes the initial surface tilt.
        object.__setattr__(self, "_p0", 2.0 * self.A * self.D0 / self.a)
        # c0 is determined by volume conservation at t=0.
        object.__setattr__(self, "_c0", self._solve_c0())

    def _solve_c0(self) -> float:
        """Find c(0) so that the initial volume matches still-water volume."""
        from scipy.optimize import brentq

        v_target = np.pi * self.D0 * self.a**2 / 2.0

        def _volume_error(c0: float) -> float:
            n = 800
            extent = self.a * 1.5
            xs = np.linspace(-extent, extent, n)
            ys = np.linspace(-extent, extent, n)
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            z_b = self.D0 * ((X**2 + Y**2) / self.a**2 - 1.0)
            eta = c0 + self._p0 * X
            h = np.maximum(eta - z_b, 0.0)
            return float(np.sum(h) * dx * dy) - v_target

        return float(brentq(_volume_error, -self.D0, self.D0, xtol=1e-10))

    @property
    def omega(self) -> float:
        """Angular frequency of the sloshing oscillation [rad/s]."""
        return np.sqrt(2 * self.g * self.D0) / self.a

    @property
    def period(self) -> float:
        """Oscillation period [s]."""
        return 2 * np.pi / self.omega

    def _c(self, t: float) -> float:
        """Mean-level adjustment c(t) from the ODE solution."""
        return self._c0 + self.a**2 * self._p0**2 * np.sin(self.omega * t) ** 2 / (4.0 * self.D0)

    def _p(self, t: float) -> float:
        """Surface tilt p(t) = p0 * cos(wt)."""
        return self._p0 * np.cos(self.omega * t)

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
        mass. Derived from the ODE: u = a^2 * p'(t) / (2*D0).
        """
        return -(self.a**2) * self._p0 * self.omega * np.sin(self.omega * t) / (2.0 * self.D0)

    def velocity_y(self) -> float:
        """y-velocity is always zero (oscillation is in x only)."""
        return 0.0

    def shoreline_x_extent(self, t: float) -> tuple[float, float]:
        """x-coordinates of the shoreline on the x-axis (y=0) at time t.

        Returns (x_min, x_max) of the wetted region along y=0.
        """
        xs = np.linspace(-self.a * 1.5, self.a * 1.5, 10000)
        h = self.depth(xs, np.zeros_like(xs), t)
        wet = h > 0
        if not wet.any():
            return (0.0, 0.0)
        wet_xs = xs[wet]
        return (float(wet_xs.min()), float(wet_xs.max()))

    def volume(self, t: float, *, n_points: int = 2000) -> float:
        """Numerically integrate the exact depth field to get total volume.

        Used for testing: volume must be constant across all times.
        """
        extent = self.a * 1.5
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

    The raster is centered at (origin_x, origin_y) in the given CRS,
    extending ±1.3a in both directions (enough margin beyond the rim
    for HEC-RAS to place boundary cells).

    Parameters
    ----------
    bowl
        ThackerBowl instance defining the geometry.
    out_path
        Output GeoTIFF path.
    resolution
        Cell size in CRS linear units (ft for EPSG:2965).
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
    extent = bowl.a * 1.3
    west = origin_x - extent
    east = origin_x + extent
    south = origin_y - extent
    north = origin_y + extent

    n_cols = int(np.ceil(2 * extent / resolution))
    n_rows = n_cols
    transform = from_bounds(west, south, east, north, n_cols, n_rows)

    cols = np.arange(n_cols)
    rows = np.arange(n_rows)
    col_grid, row_grid = np.meshgrid(cols, rows)
    xs = west + (col_grid + 0.5) * resolution
    ys = north - (row_grid + 0.5) * resolution
    local_x = xs - origin_x
    local_y = ys - origin_y
    z = bowl.bed_elevation(local_x, local_y).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=n_rows,
        width=n_cols,
        count=1,
        dtype="float32",
        crs=CRS.from_string(crs) if isinstance(crs, str) else crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(z, 1)

    return out_path


def generate_initial_wse_raster(
    bowl: ThackerBowl,
    out_path: Path | str,
    *,
    resolution: float,
    crs: CRS | str = "EPSG:2965",
    origin_x: float = 500_000.0,
    origin_y: float = 1_800_000.0,
) -> Path:
    """Write the t=0 water-surface elevation as a GeoTIFF for HEC-RAS IC."""
    out_path = Path(out_path)
    extent = bowl.a * 1.3
    west = origin_x - extent
    east = origin_x + extent
    south = origin_y - extent
    north = origin_y + extent

    n_cols = int(np.ceil(2 * extent / resolution))
    n_rows = n_cols
    transform = from_bounds(west, south, east, north, n_cols, n_rows)

    cols = np.arange(n_cols)
    rows = np.arange(n_rows)
    col_grid, row_grid = np.meshgrid(cols, rows)
    xs = west + (col_grid + 0.5) * resolution
    ys = north - (row_grid + 0.5) * resolution
    local_x = xs - origin_x
    local_y = ys - origin_y
    wse = bowl.initial_wse_raster(local_x, local_y).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=n_rows,
        width=n_cols,
        count=1,
        dtype="float32",
        crs=CRS.from_string(crs) if isinstance(crs, str) else crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(wse, 1)

    return out_path
