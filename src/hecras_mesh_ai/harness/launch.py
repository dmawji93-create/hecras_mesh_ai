"""Programmatic HEC-RAS plan launcher.

Drives HEC-RAS to compute a plan from Python so the ML pipeline doesn't
need a human clicking the Compute button. Two backends:

  - COM (primary)   talks to HEC-RAS via the registered HECRASController
                    interface (e.g. RAS70.HECRASController). Stable,
                    well-trodden path used by ras-commander et al.
  - CLI (fallback)  invokes Ras.exe with best-effort arguments. The
                    HEC-RAS 7.0 CLI is essentially undocumented, so
                    this path is empirical and may need adjustment.

Success is detected by checking that the expected `.pNN.hdf` results
file exists and has an mtime newer than the run-start timestamp.

Requires the `harness` extra (pywin32) and a HEC-RAS install on
Windows. Non-Windows environments cannot use the COM backend; the CLI
backend also has no meaningful target there.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# pywin32 is Windows-only and lives behind the `harness` extra. Import
# lazily inside the COM functions so unit tests on other platforms /
# without the extra still work for everything except the COM path.

_DEFAULT_INSTALL_ROOTS = (
    Path(r"C:\Program Files (x86)\HEC\HEC-RAS"),
    Path(r"C:\Program Files\HEC\HEC-RAS"),
)


@dataclass(frozen=True)
class RasInstall:
    """Located HEC-RAS install."""

    version: str  # e.g. "7.0"
    install_dir: Path
    ras_exe: Path
    com_prog_id: str  # e.g. "RAS70.HECRASController"


@dataclass
class RunResult:
    """Outcome of a plan compute attempt."""

    success: bool
    backend: str  # "com" or "cli"
    results_hdf: Path | None
    duration_seconds: float
    stdout: str = ""
    stderr: str = ""
    error: str = ""


def find_ras_install(
    *,
    preferred_version: str | None = None,
    install_roots: tuple[Path, ...] = _DEFAULT_INSTALL_ROOTS,
) -> RasInstall | None:
    """Locate a HEC-RAS install on disk.

    Scans the standard Program Files locations for version directories
    (e.g. `6.6/`, `7.0/`). When `preferred_version` is given, returns
    that one if present; otherwise returns the highest-numbered version
    found, or None.
    """
    candidates: list[tuple[str, Path]] = []
    for root in install_roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if not re.fullmatch(r"\d+(\.\d+)*", child.name):
                continue
            exe = child / "Ras.exe"
            if exe.is_file():
                candidates.append((child.name, child))

    if not candidates:
        return None
    if preferred_version is not None:
        for ver, dir_ in candidates:
            if ver == preferred_version:
                return _build_install(ver, dir_)
        return None

    def _version_key(item: tuple[str, Path]) -> tuple[int, ...]:
        return tuple(int(p) for p in item[0].split("."))

    candidates.sort(key=_version_key, reverse=True)
    ver, dir_ = candidates[0]
    return _build_install(ver, dir_)


def _build_install(version: str, install_dir: Path) -> RasInstall:
    # HEC-RAS naming convention: ProgID = "RAS<major><minor>.HECRASController"
    # e.g. 7.0 -> RAS70, 6.3 -> RAS63, 6.41 -> RAS641 (no dot).
    digits = version.replace(".", "")
    return RasInstall(
        version=version,
        install_dir=install_dir,
        ras_exe=install_dir / "Ras.exe",
        com_prog_id=f"RAS{digits}.HECRASController",
    )


def _expected_results_hdf(project_prj: Path, plan_id: str) -> Path:
    """Map (project, plan_id) to its expected results HDF path.

    E.g. (Muncie.prj, "04") -> Muncie.p04.hdf in the same directory.
    """
    return project_prj.with_suffix(f".p{plan_id}.hdf")


def _detect_success(results_hdf: Path, start_time: float) -> bool:
    """Did the run produce a fresh results HDF after start_time?"""
    if not results_hdf.is_file():
        return False
    return results_hdf.stat().st_mtime >= start_time - 1.0


def run_plan_com(
    install: RasInstall,
    project_prj: Path,
    plan_id: str,
    *,
    show_window: bool = False,
) -> RunResult:
    """Launch a compute via the HECRASController COM interface.

    `plan_id` is the two-digit plan number ("04", not "p04"). HEC-RAS
    enumerates plans by display name internally; we look the plan up by
    its `.pNN` filename to map to the COM API's expected identifier.
    """
    start = time.monotonic()
    start_wall = time.time()
    expected_hdf = _expected_results_hdf(project_prj, plan_id)

    # pywin32 import is deferred so this module is importable on
    # non-Windows / no-extras environments.
    try:
        import pythoncom  # noqa: F401  (initializes COM threading)
        import win32com.client
    except ImportError as e:
        return RunResult(
            success=False,
            backend="com",
            results_hdf=None,
            duration_seconds=0.0,
            error=f"pywin32 not available (install with `uv sync --extra harness`): {e}",
        )

    try:
        hec = win32com.client.Dispatch(install.com_prog_id)
    except Exception as e:  # noqa: BLE001  (COM exceptions are varied)
        return RunResult(
            success=False,
            backend="com",
            results_hdf=None,
            duration_seconds=time.monotonic() - start,
            error=f"could not dispatch {install.com_prog_id!r}: {e}",
        )

    try:
        if show_window:
            # ShowRas() may not exist on every version; ignore failure.
            with contextlib.suppress(Exception):
                hec.ShowRas()

        hec.Project_Open(str(project_prj))

        # Find the plan file matching `plan_id`. HEC-RAS exposes the
        # list of plan filenames via Plan_Names; we match by the .pNN
        # suffix in the filename.
        plan_target = f".p{plan_id}"
        # Plan_Names returns (count, names_array) — older versions vary
        # in arg shape. Use late binding tolerantly.
        try:
            _count, plan_names = hec.Plan_Names(0, None, False)
        except Exception:  # noqa: BLE001
            # Some versions return just a list.
            plan_names = list(hec.Plan_Names())

        # plan_names may be a tuple/list of plan display names. We need
        # the file-suffix mapping; HEC-RAS exposes Plan_Information /
        # PlanFile if needed, but the simplest path is to set the plan
        # by file path via CurrentPlanFile (when supported).
        matched_name: str | None = None
        for nm in plan_names:
            # Best effort: plan display names usually contain the .pNN
            # number in PlanInformationValue, but we don't have that
            # mapping handy. Fall back to filename probing below.
            if plan_target in str(nm):
                matched_name = str(nm)
                break

        if matched_name is not None:
            hec.Plan_SetCurrent(matched_name)
        else:
            # Try the file-based setter (HEC-RAS 6.x+).
            plan_path = project_prj.with_suffix(f".p{plan_id}")
            try:
                hec.CurrentPlanFile = str(plan_path)
            except Exception as e:  # noqa: BLE001
                hec.QuitRas()
                return RunResult(
                    success=False,
                    backend="com",
                    results_hdf=None,
                    duration_seconds=time.monotonic() - start,
                    error=(
                        f"could not select plan {plan_id!r}: "
                        f"not in Plan_Names ({plan_names!r}) and "
                        f"CurrentPlanFile set failed: {e}"
                    ),
                )

        # Compute. Signature has varied across versions; try the common
        # ones in order.
        compute_ok = False
        compute_err: str = ""
        try:
            # 6.x signature: Compute_CurrentPlan(NMsg, Messages, Blocking)
            result = hec.Compute_CurrentPlan(0, None, True)
            compute_ok = bool(result)
        except Exception as e:  # noqa: BLE001
            compute_err = str(e)
            try:
                # Some versions: no args.
                hec.Compute_CurrentPlan()
                compute_ok = True
            except Exception as e2:  # noqa: BLE001
                compute_err = f"{compute_err}; fallback: {e2}"

        hec.QuitRas()

        success = compute_ok and _detect_success(expected_hdf, start_wall)
        return RunResult(
            success=success,
            backend="com",
            results_hdf=expected_hdf if success else None,
            duration_seconds=time.monotonic() - start,
            error="" if success else f"compute returned ok={compute_ok}; {compute_err}",
        )
    except Exception as e:  # noqa: BLE001
        with contextlib.suppress(Exception):
            hec.QuitRas()
        return RunResult(
            success=False,
            backend="com",
            results_hdf=None,
            duration_seconds=time.monotonic() - start,
            error=f"COM exception: {e}",
        )


def run_plan_cli(
    install: RasInstall,
    project_prj: Path,
    plan_id: str,
    *,
    timeout_seconds: float = 3600,
) -> RunResult:
    """Best-effort CLI launch via Ras.exe.

    HEC-RAS 7.0's CLI surface is undocumented. We try the most commonly
    cited invocation patterns (`-c` for compute) and detect success by
    the same fresh-`.pNN.hdf`-mtime test as the COM path. If this turns
    out to spawn the GUI instead of running headless, the caller's
    timeout will trip and we'll know to fall back to COM.
    """
    start = time.monotonic()
    start_wall = time.time()
    expected_hdf = _expected_results_hdf(project_prj, plan_id)

    cmd = [
        str(install.ras_exe),
        "-c",
        str(project_prj),
        f"p{plan_id}",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return RunResult(
            success=False,
            backend="cli",
            results_hdf=None,
            duration_seconds=time.monotonic() - start,
            stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
            stderr=(e.stderr or "") if isinstance(e.stderr, str) else "",
            error=f"timed out after {timeout_seconds}s (likely the CLI launched the GUI)",
        )

    success = proc.returncode == 0 and _detect_success(expected_hdf, start_wall)
    return RunResult(
        success=success,
        backend="cli",
        results_hdf=expected_hdf if success else None,
        duration_seconds=time.monotonic() - start,
        stdout=proc.stdout,
        stderr=proc.stderr,
        error=(
            ""
            if success
            else f"exit={proc.returncode}; results-hdf-fresh="
            f"{_detect_success(expected_hdf, start_wall)}"
        ),
    )


def run_plan(
    project_prj: Path | str,
    plan_id: str,
    *,
    install: RasInstall | None = None,
    prefer: str = "com",
    cli_timeout_seconds: float = 3600,
    show_window: bool = False,
) -> RunResult:
    """Orchestrator: run a HEC-RAS plan headlessly.

    Parameters
    ----------
    project_prj
        Path to the `.prj` file.
    plan_id
        Two-digit plan number ("04" for `<project>.p04`).
    install
        Pre-discovered HEC-RAS install. Auto-discovered if None.
    prefer
        "com" (default) or "cli". The other is tried as fallback if the
        first fails.
    cli_timeout_seconds
        Hard ceiling on CLI subprocess. The default (1 h) is intended
        to comfortably cover Muncie-scale runs.

    Returns RunResult — `success` reflects fresh-results-file detection,
    not just the backend's return code.
    """
    project_prj = Path(project_prj)
    if not project_prj.is_file():
        return RunResult(
            success=False,
            backend=prefer,
            results_hdf=None,
            duration_seconds=0.0,
            error=f"project file not found: {project_prj}",
        )

    if sys.platform != "win32":
        return RunResult(
            success=False,
            backend=prefer,
            results_hdf=None,
            duration_seconds=0.0,
            error="HEC-RAS launcher requires Windows",
        )

    if install is None:
        install = find_ras_install()
        if install is None:
            return RunResult(
                success=False,
                backend=prefer,
                results_hdf=None,
                duration_seconds=0.0,
                error="no HEC-RAS install found on this machine",
            )

    if prefer not in {"com", "cli"}:
        raise ValueError(f"prefer must be 'com' or 'cli'; got {prefer!r}")

    order = [prefer, "cli" if prefer == "com" else "com"]
    last: RunResult | None = None
    for backend in order:
        if backend == "com":
            last = run_plan_com(install, project_prj, plan_id, show_window=show_window)
        else:
            last = run_plan_cli(install, project_prj, plan_id, timeout_seconds=cli_timeout_seconds)
        if last.success:
            return last
    assert last is not None
    return last
