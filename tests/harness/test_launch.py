"""Tests for the HEC-RAS plan launcher."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from hecras_mesh_ai.harness import RasInstall, find_ras_install
from hecras_mesh_ai.harness.launch import (
    _build_install,
    _detect_success,
    _expected_results_hdf,
    run_plan,
    run_plan_cli,
)

# ---------------------------------------------------------------------------
# Pure-Python helpers (no HEC-RAS or COM needed)
# ---------------------------------------------------------------------------


def test_build_install_for_7_0_produces_correct_progid(tmp_path):
    inst = _build_install("7.0", tmp_path)
    assert inst.version == "7.0"
    assert inst.install_dir == tmp_path
    assert inst.ras_exe == tmp_path / "Ras.exe"
    assert inst.com_prog_id == "RAS70.HECRASController"


def test_build_install_for_6_3_produces_correct_progid(tmp_path):
    inst = _build_install("6.3", tmp_path)
    assert inst.com_prog_id == "RAS63.HECRASController"


def test_build_install_strips_dots_for_compound_version(tmp_path):
    inst = _build_install("6.4.1", tmp_path)
    assert inst.com_prog_id == "RAS641.HECRASController"


def test_find_ras_install_returns_none_when_no_roots_exist(tmp_path):
    fake_roots = (tmp_path / "nonexistent",)
    assert find_ras_install(install_roots=fake_roots) is None


def test_find_ras_install_picks_highest_version(tmp_path):
    """Stand up a synthetic HEC tree with 6.6 and 7.0 versions; expect 7.0."""
    root = tmp_path / "HEC-RAS"
    root.mkdir()
    for v in ("6.6", "7.0", "not-a-version"):
        d = root / v
        d.mkdir()
        if v != "not-a-version":
            (d / "Ras.exe").write_bytes(b"")
    install = find_ras_install(install_roots=(root,))
    assert install is not None
    assert install.version == "7.0"


def test_find_ras_install_honors_preferred_version(tmp_path):
    root = tmp_path / "HEC-RAS"
    root.mkdir()
    for v in ("6.6", "7.0"):
        d = root / v
        d.mkdir()
        (d / "Ras.exe").write_bytes(b"")
    install = find_ras_install(install_roots=(root,), preferred_version="6.6")
    assert install is not None
    assert install.version == "6.6"


def test_find_ras_install_returns_none_when_preferred_missing(tmp_path):
    root = tmp_path / "HEC-RAS"
    root.mkdir()
    d = root / "7.0"
    d.mkdir()
    (d / "Ras.exe").write_bytes(b"")
    assert find_ras_install(install_roots=(root,), preferred_version="9.9") is None


def test_find_ras_install_skips_versions_without_exe(tmp_path):
    """A version dir without Ras.exe must not be returned."""
    root = tmp_path / "HEC-RAS"
    root.mkdir()
    (root / "7.0").mkdir()  # no Ras.exe — must be skipped
    six = root / "6.6"
    six.mkdir()
    (six / "Ras.exe").write_bytes(b"")
    install = find_ras_install(install_roots=(root,))
    assert install is not None
    assert install.version == "6.6"


def test_expected_results_hdf_maps_plan_id():
    p = Path("/tmp/Foo.prj")
    assert _expected_results_hdf(p, "04") == Path("/tmp/Foo.p04.hdf")
    assert _expected_results_hdf(p, "01") == Path("/tmp/Foo.p01.hdf")


def test_detect_success_true_when_file_fresh(tmp_path):
    f = tmp_path / "results.p01.hdf"
    f.write_bytes(b"x")
    assert _detect_success(f, start_time=time.time() - 5)


def test_detect_success_false_when_file_missing(tmp_path):
    f = tmp_path / "results.p01.hdf"
    assert not _detect_success(f, start_time=time.time())


def test_detect_success_false_when_file_stale(tmp_path):
    f = tmp_path / "results.p01.hdf"
    f.write_bytes(b"x")
    # Backdate mtime to before start_time.
    old = time.time() - 1000
    os.utime(f, (old, old))
    assert not _detect_success(f, start_time=time.time())


# ---------------------------------------------------------------------------
# Orchestrator error paths (no HEC-RAS needed)
# ---------------------------------------------------------------------------


def test_run_plan_returns_failure_when_project_missing(tmp_path):
    result = run_plan(tmp_path / "nope.prj", "01")
    assert not result.success
    assert "not found" in result.error


def test_run_plan_rejects_unknown_prefer(tmp_path):
    prj = tmp_path / "x.prj"
    prj.write_bytes(b"")
    with pytest.raises(ValueError, match="prefer"):
        run_plan(prj, "01", prefer="grpc")


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows guard test")
def test_run_plan_refuses_non_windows(tmp_path):
    prj = tmp_path / "x.prj"
    prj.write_bytes(b"")
    result = run_plan(prj, "01")
    assert not result.success
    assert "Windows" in result.error


def test_run_plan_reports_when_no_install_found(tmp_path, monkeypatch):
    prj = tmp_path / "x.prj"
    prj.write_bytes(b"")
    # Force discovery to return None.
    monkeypatch.setattr(
        "hecras_mesh_ai.harness.launch.find_ras_install",
        lambda **_kw: None,
    )
    if sys.platform != "win32":
        # The Windows guard fires before install discovery.
        pytest.skip("Windows guard fires first on non-Windows")
    result = run_plan(prj, "01")
    assert not result.success
    assert "no HEC-RAS install" in result.error


# ---------------------------------------------------------------------------
# CLI fallback test that doesn't actually run HEC-RAS:
# we point ras_exe at a script that just exits non-zero so we can verify
# the failure path without launching the real GUI.
# ---------------------------------------------------------------------------


def _make_fake_ras_exe(tmp_path: Path, exit_code: int) -> Path:
    """Stand up a fake Ras.exe via cmd.exe so we can probe the CLI path."""
    if sys.platform != "win32":
        pytest.skip("CLI fallback test is Windows-only")
    # A .bat that simply exits with the given code.
    fake = tmp_path / "fake_ras.bat"
    fake.write_text(f"@echo off\r\nexit /b {exit_code}\r\n")
    return fake


@pytest.mark.skipif(sys.platform != "win32", reason="CLI test is Windows-only")
def test_run_plan_cli_reports_nonzero_exit(tmp_path):
    fake = _make_fake_ras_exe(tmp_path, exit_code=7)
    install = RasInstall(
        version="0.0",
        install_dir=tmp_path,
        ras_exe=fake,
        com_prog_id="RAS00.HECRASController",
    )
    prj = tmp_path / "demo.prj"
    prj.write_bytes(b"")
    result = run_plan_cli(install, prj, "01", timeout_seconds=10)
    assert not result.success
    assert "exit=7" in result.error


# ---------------------------------------------------------------------------
# Real HEC-RAS integration test (marked slow; opt-in).
# This actually attempts to compute Muncie's plan via the verified
# round-trip copy. Will not run unless explicitly invoked.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MUNCIE_PRJ = _REPO_ROOT / "data" / "_verification" / "Muncie_writer_roundtrip" / "Muncie.prj"


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "win32" or not _MUNCIE_PRJ.exists(),
    reason="Requires Windows and the prepared Muncie verification project",
)
def test_integration_compute_muncie_plan_via_com():
    """Actually launch HEC-RAS and run Muncie's plan. Expensive — gated."""
    install = find_ras_install()
    assert install is not None, "no HEC-RAS install found"
    # Muncie's default plan is p04 (matches geometry g04 in the project file).
    result = run_plan(_MUNCIE_PRJ, "04", install=install, prefer="com")
    assert result.success, f"compute failed: {result.error}\nstderr={result.stderr}"
    assert result.results_hdf is not None
    assert result.results_hdf.exists()
