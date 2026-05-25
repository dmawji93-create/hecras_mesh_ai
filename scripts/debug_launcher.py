"""Probe HEC-RAS COM and CLI launchers to figure out what 7.0 actually exposes."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from hecras_mesh_ai.harness import find_ras_install
from hecras_mesh_ai.harness.launch import run_plan_com

REPO = Path(__file__).resolve().parents[1]
PRJ = REPO / "data" / "_verification" / "Muncie_writer_roundtrip" / "Muncie.prj"


def probe_com_methods(prog_id: str) -> None:
    """Enumerate what the HECRASController COM object actually offers."""
    print(f"--- Probing COM ProgID: {prog_id} ---")
    try:
        import win32com.client
    except ImportError as e:
        print(f"  pywin32 import failed: {e}")
        return
    try:
        hec = win32com.client.Dispatch(prog_id)
    except Exception as e:  # noqa: BLE001
        print(f"  Dispatch failed: {e}")
        return
    print(f"  Dispatch OK; type={type(hec).__name__}")
    # Late binding objects expose _oleobj_ but no direct dir() — try anyway.
    members = sorted(set(dir(hec)) - set(dir(object)))
    interesting = [m for m in members if not m.startswith("_")]
    print(f"  Members ({len(interesting)} top-level):")
    for m in interesting[:80]:
        print(f"    {m}")
    if len(interesting) > 80:
        print(f"    ... and {len(interesting) - 80} more")
    with contextlib.suppress(Exception):
        hec.QuitRas()


def main() -> int:
    install = find_ras_install()
    if install is None:
        print("No HEC-RAS install found.", file=sys.stderr)
        return 1
    print(f"Install: {install}")
    print(f"Project: {PRJ} (exists={PRJ.is_file()})")

    probe_com_methods(install.com_prog_id)

    print("\n--- Running COM path verbose ---")
    result_com = run_plan_com(install, PRJ, "04")
    print(f"  success={result_com.success}")
    print(f"  error={result_com.error}")
    print(f"  duration={result_com.duration_seconds:.1f}s")
    print(f"  results_hdf={result_com.results_hdf}")

    print("\n--- Files in project dir after COM attempt ---")
    files = sorted(PRJ.parent.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files[:10]:
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes, mtime={f.stat().st_mtime:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
