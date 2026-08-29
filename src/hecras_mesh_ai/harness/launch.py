"""Programmatic HEC-RAS plan launcher.

Drives HEC-RAS to compute a plan from Python so the ML pipeline doesn't
need a human clicking the Compute button. Two backends:

  - COM (primary)   talks to HEC-RAS via the registered HECRASController
                    interface (e.g. RAS70.HECRASController). Stable,
                    well-trodden path used by ras-commander et al.
  - CLI (fallback)  invokes Ras.exe with best-effort arguments. The
                    HEC-RAS 7.0 CLI is essentially undocumented, so
                    this path is empirical and may need adjustment.

Success requires BOTH of:
  1. the expected `.pNN.hdf` results file exists with an mtime newer
     than the overall run-start timestamp, AND
  2. the results HDF's `/Results/Unsteady/Summary@Solution` attribute
     records a finished run ("... Finished Successfully").

The second check is what makes the detection trustworthy: a compute
that crashes mid-solve leaves a fresh but PARTIAL results file, which
mtime alone would happily bless — and the CLI fallback would then
"succeed" against the wreckage of the COM attempt. The Solution
attribute is written only by a completed engine run.

The COM compute runs in a worker thread under a hard timeout; on
expiry, any Ras.exe processes spawned since the attempt started are
killed so a hung solve can't orphan the machine.

Requires the `harness` extra (pywin32) and a HEC-RAS install on
Windows. Non-Windows environments cannot use the COM backend; the CLI
backend also has no meaningful target there.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from hecras_mesh_ai.harness.results import run_completed

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
    """Did the run produce a fresh AND finished results HDF?

    Freshness (mtime >= start_time, 1 s filesystem slack) proves the
    file belongs to this run; the Solution completion marker proves the
    engine actually finished. Either alone can be fooled — a crashed
    solve leaves a fresh partial file, and a finished file from an
    earlier run is stale.
    """
    if not results_hdf.is_file():
        return False
    if results_hdf.stat().st_mtime < start_time - 1.0:
        return False
    return run_completed(results_hdf)


def _ras_pids() -> set[int]:
    """PIDs of currently-running Ras.exe processes (empty set off-Windows/on error)."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Ras.exe", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    pids: set[int] = set()
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) >= 2 and parts[0].lower() == "ras.exe":
            with contextlib.suppress(ValueError):
                pids.add(int(parts[1]))
    return pids


def _kill_ras_pids(pids: set[int]) -> list[int]:
    """Force-kill the given Ras.exe PIDs; returns those we attempted."""
    killed: list[int] = []
    for pid in sorted(pids):
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                capture_output=True,
                timeout=15,
                check=False,
            )
            killed.append(pid)
    return killed


def _run_plan_com_session(
    install: RasInstall,
    project_prj: Path,
    plan_id: str,
    expected_hdf: Path,
    *,
    show_window: bool,
    start: float,
    start_wall: float,
) -> RunResult:
    """The COM conversation itself. Runs inside the worker thread."""
    import win32com.client

    def _fail(msg: str) -> RunResult:
        return RunResult(
            success=False,
            backend="com",
            results_hdf=None,
            duration_seconds=time.monotonic() - start,
            error=msg,
        )

    try:
        hec = win32com.client.Dispatch(install.com_prog_id)
    except Exception as e:  # noqa: BLE001  (COM exceptions are varied)
        return _fail(f"could not dispatch {install.com_prog_id!r}: {e}")

    try:
        if show_window:
            # ShowRas() may not exist on every version; ignore failure.
            with contextlib.suppress(Exception):
                hec.ShowRas()

        hec.Project_Open(str(project_prj))

        # Plan_Names(PlanCount, PlanNames, OnlyBaseDir) — pywin32 late
        # binding turns output params into extra return values. Returns
        # [count, (titles_tuple), only_base_dir_echo].
        try:
            _count, plan_titles, _ = hec.Plan_Names(0, None, False)
        except Exception as e:  # noqa: BLE001
            return _fail(f"Plan_Names call failed: {e}")

        # Map plan display titles to their .pNN file paths via
        # Plan_GetFilename, find the one ending in .p<plan_id>.
        target_suffix = f".p{plan_id}".lower()
        matched_title: str | None = None
        title_to_filename: dict[str, str] = {}
        for title in plan_titles or ():
            try:
                raw = hec.Plan_GetFilename(title)
            except Exception:  # noqa: BLE001
                continue
            # pywin32 returns a tuple (filepath, echoed_title) for COM
            # methods with output params; some versions return just the
            # filepath string. Normalize.
            fname = str(raw[0]) if isinstance(raw, tuple) else str(raw)
            title_to_filename[title] = fname
            if fname.lower().endswith(target_suffix):
                matched_title = title
                break

        if matched_title is None:
            return _fail(
                f"no plan with filename suffix {target_suffix!r}; "
                f"available: {title_to_filename!r}"
            )

        # Plan_SetCurrent returns a success bool. If selection silently
        # failed, Compute_CurrentPlan would compute — and clobber the
        # results of — whichever plan happened to be current. Refuse.
        set_ret = hec.Plan_SetCurrent(matched_title)
        set_ok = bool(set_ret[0]) if isinstance(set_ret, tuple) else bool(set_ret)
        if not set_ok:
            return _fail(
                f"Plan_SetCurrent({matched_title!r}) returned False — "
                "refusing to compute whichever plan was already current"
            )

        # Compute_CurrentPlan(NMsg, Messages, Blocking). pywin32 returns
        # the function result; output params become extra return values.
        try:
            ret = hec.Compute_CurrentPlan(0, None, True)
        except Exception as e:  # noqa: BLE001
            return _fail(f"Compute_CurrentPlan raised: {e}")

        # `ret` may be a bool or a tuple of (bool, NMsg, Messages).
        if isinstance(ret, tuple):
            compute_ok = bool(ret[0])
            compute_err = f"messages={ret[2]!r}" if len(ret) >= 3 else ""
        else:
            compute_ok = bool(ret)
            compute_err = ""

        # Quit before success detection so the results file is fully
        # released — but never let a QuitRas hiccup convert a completed
        # compute into a reported failure (the finally re-tries anyway).
        with contextlib.suppress(Exception):
            hec.QuitRas()

        fresh_and_finished = _detect_success(expected_hdf, start_wall)
        success = compute_ok and fresh_and_finished
        return RunResult(
            success=success,
            backend="com",
            results_hdf=expected_hdf if success else None,
            duration_seconds=time.monotonic() - start,
            error=(
                ""
                if success
                else f"compute returned ok={compute_ok}; "
                f"results fresh+finished={fresh_and_finished}; {compute_err}"
            ),
        )
    except Exception as e:  # noqa: BLE001
        return _fail(f"COM exception: {e}")
    finally:
        with contextlib.suppress(Exception):
            hec.QuitRas()


def run_plan_com(
    install: RasInstall,
    project_prj: Path,
    plan_id: str,
    *,
    show_window: bool = False,
    timeout_seconds: float = 3600.0,
    start_wall: float | None = None,
) -> RunResult:
    """Launch a compute via the HECRASController COM interface.

    `plan_id` is the two-digit plan number ("04", not "p04"). HEC-RAS
    enumerates plans by display name internally; we look the plan up by
    its `.pNN` filename to map to the COM API's expected identifier.

    The COM session runs in a worker thread (its own COM apartment)
    bounded by `timeout_seconds`. `Compute_CurrentPlan(blocking=True)`
    has no timeout of its own — a non-converging solve or a modal
    dialog would otherwise block Python forever. On expiry, Ras.exe
    processes spawned since this attempt started are force-killed.

    `start_wall` lets the orchestrator share one run-start timestamp
    across backends for the freshness check; defaults to now.
    """
    start = time.monotonic()
    if start_wall is None:
        start_wall = time.time()
    expected_hdf = _expected_results_hdf(project_prj, plan_id)

    try:
        import pythoncom
        import win32com.client  # noqa: F401
    except ImportError as e:
        return RunResult(
            success=False,
            backend="com",
            results_hdf=None,
            duration_seconds=0.0,
            error=f"pywin32 not available (install with `uv sync --extra harness`): {e}",
        )

    pids_before = _ras_pids()
    box: list[RunResult] = []

    def _worker() -> None:
        pythoncom.CoInitialize()
        try:
            result = _run_plan_com_session(
                install,
                project_prj,
                plan_id,
                expected_hdf,
                show_window=show_window,
                start=start,
                start_wall=start_wall,
            )
            box.append(result)
        finally:
            pythoncom.CoUninitialize()

    thread = threading.Thread(target=_worker, daemon=True, name="hecras-com-compute")
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        killed = _kill_ras_pids(_ras_pids() - pids_before)
        return RunResult(
            success=False,
            backend="com",
            results_hdf=None,
            duration_seconds=time.monotonic() - start,
            error=(
                f"COM compute exceeded timeout_seconds={timeout_seconds}; "
                f"killed Ras.exe pids {killed or '(none found)'}"
            ),
        )
    if not box:
        return RunResult(
            success=False,
            backend="com",
            results_hdf=None,
            duration_seconds=time.monotonic() - start,
            error="COM worker terminated without producing a result",
        )
    return box[0]


def run_plan_cli(
    install: RasInstall,
    project_prj: Path,
    plan_id: str,
    *,
    timeout_seconds: float = 3600,
    start_wall: float | None = None,
) -> RunResult:
    """Best-effort CLI launch via Ras.exe.

    HEC-RAS 7.0's CLI surface is undocumented. We try the most commonly
    cited invocation patterns (`-c` for compute) and detect success by
    the same fresh-AND-finished results-HDF test as the COM path — the
    completion check is what stops the known-degenerate `-c` invocation
    (exit 0, computes nothing) from ever reporting a false success
    against a leftover or partial results file.
    """
    start = time.monotonic()
    if start_wall is None:
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

    fresh_and_finished = _detect_success(expected_hdf, start_wall)
    success = proc.returncode == 0 and fresh_and_finished
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
            else f"exit={proc.returncode}; results fresh+finished={fresh_and_finished}"
        ),
    )


def run_plan(
    project_prj: Path | str,
    plan_id: str,
    *,
    install: RasInstall | None = None,
    prefer: str = "com",
    com_timeout_seconds: float = 3600,
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
    com_timeout_seconds, cli_timeout_seconds
        Hard ceilings per backend. The defaults (1 h) comfortably cover
        Muncie-scale runs.

    Returns RunResult — `success` requires a results HDF that is both
    fresh (mtime after this call started; one shared timestamp across
    backends) and finished (the engine's Solution completion marker),
    never just a backend's return code. On total failure, `error`
    concatenates every attempted backend's error so the informative
    one isn't shadowed by the fallback's.
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

    start_wall = time.time()
    order = [prefer, "cli" if prefer == "com" else "com"]
    attempts: list[tuple[str, RunResult]] = []
    last: RunResult | None = None
    for backend in order:
        if backend == "com":
            last = run_plan_com(
                install,
                project_prj,
                plan_id,
                show_window=show_window,
                timeout_seconds=com_timeout_seconds,
                start_wall=start_wall,
            )
        else:
            last = run_plan_cli(
                install,
                project_prj,
                plan_id,
                timeout_seconds=cli_timeout_seconds,
                start_wall=start_wall,
            )
        attempts.append((backend, last))
        if last.success:
            return last
    assert last is not None
    if len(attempts) > 1:
        last.error = " | ".join(f"{b}: {r.error}" for b, r in attempts)
    return last
