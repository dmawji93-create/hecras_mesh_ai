"""Analytical benchmark cases for mesh-quality framework verification.

Stage 6 requires comparing HEC-RAS simulation results against known truth.
Analytical benchmarks provide that truth — exact solutions to the shallow
water equations whose depth, velocity, and shoreline position can be
computed at any (x, y, t) from a formula.

  - thacker  : Thacker (1981) parabolic bowl with planar surface oscillation.
               Tests wetting/drying fronts and 2D wave propagation.
"""

from hecras_mesh_ai.benchmark.thacker import (
    ThackerBowl,
    generate_thacker_terrain,
)

__all__ = [
    "ThackerBowl",
    "generate_thacker_terrain",
]
