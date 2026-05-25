"""Phase A closing demo — run the full breakline-detection -> HEC-RAS loop.

Pipeline:
    Stage 2 model  ─sliding-window inference─>  probability raster
                   ─threshold+skeletonize+trace+simplify─>  predicted polylines
                   ─list[Breakline] wrap─>  Stage 4 writer
                   ─run_plan COM─>  HEC-RAS computes
                   ─max_water_surface─>  per-cell results

This is a *plumbing-validation* demo, not a model-quality demo: the Stage 2
checkpoint was trained on Bald Eagle and val F1 on Muncie was zero. We
don't expect the predicted breaklines to be hydraulically correct. We do
expect every Python -> HEC-RAS -> Python handoff to work end-to-end.

Output: data/_verification/Muncie_phase_a_demo/  + a printed comparison
against the baseline (no-prediction-applied) round-trip.

Usage:
    uv run python scripts/phase_a_closing_demo.py
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  registers .rio accessor

from hecras_mesh_ai.harness import (
    Breakline,
    find_ras_install,
    max_depth,
    max_water_surface,
    read_breaklines,
    replace_breaklines,
    run_plan,
)
from hecras_mesh_ai.model.unet import BreaklineUNet
from hecras_mesh_ai.postprocess.breaklines import probability_to_polylines
from hecras_mesh_ai.postprocess.inference import sliding_window_predict

REPO_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = REPO_ROOT / "lightning_logs" / "pilot" / "version_0" / "checkpoints" / "last.ckpt"
MUNCIE_FEATURES = REPO_ROOT / "data" / "processed" / "stage1_exit" / "muncie" / "features.tif"
SOURCE_PROJECT = (
    REPO_ROOT
    / "data"
    / "raw"
    / "usace"
    / "RAS Samples"
    / "Example_Projects_7_0"
    / "2D Unsteady Flow Hydraulics"
    / "Muncie"
)
TARGET_PROJECT = REPO_ROOT / "data" / "_verification" / "Muncie_phase_a_demo"

# Lower threshold than the default 0.5 - Stage 2 is known-mediocre on
# Muncie and the few positives it does fire are below 0.5. Tuned by
# inspecting prob.max() on a dry run.
PROB_THRESHOLD = 0.3
MIN_LENGTH_PIXELS = 8
SIMPLIFY_TOL_FT = 25.0  # Muncie CRS is EPSG:2965 ftUS


def _force_writeable(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _copy_project() -> Path:
    if TARGET_PROJECT.exists():
        shutil.rmtree(TARGET_PROJECT, onexc=_force_writeable)
    shutil.copytree(SOURCE_PROJECT, TARGET_PROJECT)
    return TARGET_PROJECT / "Muncie.g04.hdf"


def _load_model():
    import pathlib

    import torch

    # PyTorch 2.6 weights_only=True default rejects pathlib types saved
    # by Lightning's save_hyperparameters(). Whitelist them.
    torch.serialization.add_safe_globals(
        [
            pathlib.WindowsPath,
            pathlib.PosixPath,
            pathlib.PurePath,
            pathlib.PurePosixPath,
            pathlib.PureWindowsPath,
        ]
    )
    return BreaklineUNet.load_from_checkpoint(str(CHECKPOINT), map_location="cuda")


def _features_from_geotiff(path: Path):
    """Return (features ndarray (C,H,W), transform, crs)."""
    import xarray as xr

    da = xr.open_dataarray(path, engine="rasterio")
    features = da.values.astype(np.float32)
    transform = da.rio.transform()
    crs = da.rio.crs
    da.close()
    return features, transform, crs


def _gdf_to_breaklines(gdf) -> list[Breakline]:
    """Convert the polylines GeoDataFrame into Breakline records.

    Names are auto-generated since the model doesn't predict identifiers.
    """
    out: list[Breakline] = []
    for i, geom in enumerate(gdf.geometry):
        coords = np.asarray(geom.coords, dtype=np.float64)
        if coords.shape[0] < 2:
            continue
        out.append(Breakline(name=f"Predicted_{i:03d}", points=coords))
    return out


def main() -> int:
    if not CHECKPOINT.is_file():
        print(f"ERROR: missing checkpoint {CHECKPOINT}", file=sys.stderr)
        return 1
    if not MUNCIE_FEATURES.is_file():
        print(f"ERROR: missing features {MUNCIE_FEATURES}", file=sys.stderr)
        return 1

    print("-" * 70)
    print("Phase A closing demo — model -> writer -> HEC-RAS -> results")
    print("-" * 70)

    # 1. Load model + features.
    print("\n[1/6] Loading checkpoint + Muncie features...")
    model = _load_model()
    features, transform, crs = _features_from_geotiff(MUNCIE_FEATURES)
    print(f"      features {features.shape} dtype={features.dtype} crs={crs}")

    # 2. Sliding-window inference.
    print("\n[2/6] Running sliding-window inference...")
    probs = sliding_window_predict(model, features, tile_size=256, overlap=32, batch_size=16)
    finite = probs[np.isfinite(probs)]
    print(
        f"      prob map {probs.shape}: min={finite.min():.4f} "
        f"max={finite.max():.4f} mean={finite.mean():.4f}"
    )
    print(
        f"      positive fraction at threshold {PROB_THRESHOLD}: "
        f"{(probs > PROB_THRESHOLD).sum() / probs.size:.6%}"
    )

    # 3. Probability -> polylines.
    print("\n[3/6] Extracting polylines from probability map...")
    gdf = probability_to_polylines(
        probs,
        transform=transform,
        target_crs=crs,
        threshold=PROB_THRESHOLD,
        min_length_pixels=MIN_LENGTH_PIXELS,
        simplify_tolerance=SIMPLIFY_TOL_FT,
    )
    breaklines = _gdf_to_breaklines(gdf)
    print(f"      {len(breaklines)} predicted breaklines after simplification")
    for bl in breaklines[:5]:
        print(f"        - {bl.name}: {bl.points.shape[0]} pts")
    if len(breaklines) > 5:
        print(f"        ... and {len(breaklines) - 5} more")

    # 4. Copy project + replace breaklines.
    print("\n[4/6] Preparing verification project + writing breaklines...")
    target_hdf = _copy_project()
    original = read_breaklines(target_hdf)
    print(f"      original breaklines (will be replaced): {[b.name for b in original]}")
    replace_breaklines(target_hdf, target_hdf, breaklines, overwrite=True)
    print(f"      wrote {len(breaklines)} predicted breaklines to {target_hdf.name}")

    # 5. Run HEC-RAS.
    print("\n[5/6] Launching HEC-RAS via COM (Muncie plan p04)...")
    install = find_ras_install()
    if install is None:
        print("      ERROR: no HEC-RAS install found", file=sys.stderr)
        return 1
    print(f"      install: {install.version} at {install.install_dir}")
    project_prj = TARGET_PROJECT / "Muncie.prj"
    result = run_plan(project_prj, "04", install=install, prefer="com")
    if not result.success:
        print(f"      FAIL: {result.error}", file=sys.stderr)
        return 1
    print(f"      OK: {result.duration_seconds:.1f}s, results -> {result.results_hdf.name}")

    # 6. Parse and report.
    print("\n[6/6] Parsing results...")
    wse = max_water_surface(result.results_hdf)
    depth = max_depth(result.results_hdf)
    n_active = (~np.isnan(wse.values)).sum()
    wse_finite = wse.values[np.isfinite(wse.values)]
    depth_finite = depth.values[np.isfinite(depth.values)]
    print(f"      {n_active} active cells")
    print(f"      WSE   range {wse_finite.min():.1f} - {wse_finite.max():.1f} {wse.attrs['units']}")
    max_depth_idx = np.nanargmax(depth.values)
    max_depth_time = depth.coords["time_of_max"].values[max_depth_idx]
    print(
        f"      depth range {depth_finite.min():.2f} - {depth_finite.max():.2f} "
        f"{depth.attrs['units']}; max depth at t={max_depth_time:.3f} days"
    )

    # Comparison against baseline (the Task 3 round-trip Muncie copy).
    baseline_hdf = (
        REPO_ROOT / "data" / "_verification" / "Muncie_writer_roundtrip" / "Muncie.p04.hdf"
    )
    if baseline_hdf.is_file():
        baseline_depth = max_depth(baseline_hdf)
        bd_finite = baseline_depth.values[np.isfinite(baseline_depth.values)]
        print("\nComparison vs baseline (round-trip with expert breaklines):")
        print(f"      baseline max depth: {bd_finite.max():.2f} ft")
        print(f"      predicted max depth: {depth_finite.max():.2f} ft")
        print(f"      delta:               {depth_finite.max() - bd_finite.max():+.2f} ft")
        print(f"      baseline active cells: {(~np.isnan(baseline_depth.values)).sum()}")
        print(f"      predicted active cells: {n_active}")

    print("\n" + "-" * 70)
    print("Closed-loop demo complete. Phase A pipeline validated end-to-end.")
    print("-" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
