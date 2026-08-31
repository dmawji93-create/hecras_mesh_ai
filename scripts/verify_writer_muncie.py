"""Prepare a Muncie project copy whose geometry HDF has been round-tripped
through `replace_breaklines`, for manual verification that HEC-RAS 7.0 can
open it cleanly.

Run:
    uv run python scripts/verify_writer_muncie.py

Then open the resulting `Muncie.prj` in HEC-RAS 7.0 and confirm:
1. Geometry loads without errors or warnings.
2. RAS Mapper shows the same 2 breaklines (Road 1, HighGround 1) as the
   original.
3. "Generate Computation Points" / mesh regeneration produces the same
   cell count and topology.

The output dir is gitignored (data/_verification/).
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path

from hecras_mesh_ai.harness import read_breaklines, replace_breaklines


def _force_writeable(func, path, _exc):
    """rmtree onexc hook: clear read-only bit Windows sets on copied files."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


REPO_ROOT = Path(__file__).resolve().parents[1]
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
TARGET_PROJECT = REPO_ROOT / "data" / "_verification" / "Muncie_writer_roundtrip"

GEOMETRY_HDF_NAME = "Muncie.g04.hdf"


def main() -> int:
    if not SOURCE_PROJECT.exists():
        print(f"ERROR: source project not found at {SOURCE_PROJECT}", file=sys.stderr)
        return 1

    if TARGET_PROJECT.exists():
        print(f"Removing existing {TARGET_PROJECT}")
        shutil.rmtree(TARGET_PROJECT, onexc=_force_writeable)

    print(f"Copying project tree:\n  {SOURCE_PROJECT}\n  -> {TARGET_PROJECT}")
    shutil.copytree(SOURCE_PROJECT, TARGET_PROJECT)

    target_hdf = TARGET_PROJECT / GEOMETRY_HDF_NAME
    if not target_hdf.exists():
        print(f"ERROR: {target_hdf} missing after copy", file=sys.stderr)
        return 1

    print("\nReading original breaklines via writer's reader:")
    original = read_breaklines(target_hdf)
    for bl in original:
        print(f"  - {bl.name!r}: {bl.points.shape[0]} points")

    print(f"\nRound-tripping breaklines through replace_breaklines into:\n  {target_hdf}")
    replace_breaklines(target_hdf, target_hdf, original, overwrite=True)

    print("\nVerifying re-read matches:")
    roundtrip = read_breaklines(target_hdf)
    assert len(roundtrip) == len(original)
    for o, r in zip(original, roundtrip, strict=True):
        assert o.name == r.name
        assert o.points.shape == r.points.shape
        print(f"  - {o.name!r}: round-trip OK")

    print("\nDone. To verify in HEC-RAS:")
    print("  1. Open HEC-RAS 7.0")
    print(f"  2. File -> Open Project -> {TARGET_PROJECT / 'Muncie.prj'}")
    print("  3. Geometry -> 2D Flow Areas: confirm both breaklines visible")
    print("  4. Check 'Computation Messages' for any HDF / schema errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
