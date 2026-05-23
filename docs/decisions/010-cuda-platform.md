# ADR 010: CUDA Platform — PyTorch cu128 via PyTorch Index

**Status:** Accepted
**Date:** 2026-05-21

## Context

ADR 005 selected PyTorch + Lightning as the core ML stack but left the compute
target deferred. The dev machine has an NVIDIA RTX 3090 (24 GB VRAM, Ampere,
compute capability 8.6) with driver 591.86 — recent enough to support up to
CUDA 13.1.

By default, `pip install torch` from PyPI ships a CPU-only wheel on Windows.
Without an explicit source override, we'd train on the CPU even with the 3090
sitting idle in the same machine, which is both slow and wasteful.

We need to wire torch (and torchvision, since they are version-locked) to a
CUDA-capable wheel.

## Decision

Pin `torch` and `torchvision` to the **PyTorch CUDA 12.8 index**
(`https://download.pytorch.org/whl/cu128`) via `[tool.uv.sources]` and
`[[tool.uv.index]]` in `pyproject.toml`.

This resolves torch to `2.11.0+cu128` (latest available cu128 build at time
of writing) and torchvision to `0.26.0` — both bundling their own CUDA 12.8
runtime libraries, so a separate CUDA toolkit (`nvcc`) install is not
required.

## Consequences

### Positive
- GPU training is available out of the box; no manual wheel surgery.
- CUDA 12.8 is well-matched to driver 591.86 (which supports up to 13.1) —
  comfortable headroom, modern features.
- The pinned source is declarative and reproducible via `uv.lock`.
- A separate CUDA toolkit install is unnecessary; the wheel ships its
  own bundled CUDA runtime.

### Negative / risks
- **Single-platform pyproject.** The `[tool.uv.sources]` block targets a
  Windows + CUDA installation. Cloning this repo on a CPU-only machine,
  on macOS, or on Linux without a compatible GPU will fail at `uv sync`
  unless the source override is removed or made conditional.
- **Version lag.** The PyTorch CUDA index lags PyPI by a release or two
  (we got 2.11.0+cu128 instead of 2.12.0). For now this is fine — the
  API surface we use is stable across these versions.
- **Wheel size.** The cu128 torch wheel is ~2.6 GB. First sync from a
  cold cache is a meaningful download.

### Neutral
- The cu128 wheel bundles CUDA runtime libraries; no global `nvcc`
  install needed. This also means future PyTorch updates re-bundle a
  fresh runtime, sidestepping system-wide CUDA toolkit version drift.

## Alternatives considered

- **cu126 or cu124.** Older PyTorch CUDA targets, both perfectly viable on
  this driver. Rejected because cu128 is the most modern stable option
  the driver fully supports.
- **CPU-only torch (status quo).** Rejected — leaves the 3090 idle.
- **Parallel `[cpu]` / `[cu]` extras for multi-platform support.** The
  "publish-ready" pattern: declare both CUDA and CPU torch sources behind
  separate optional-dependency extras (`uv sync --extra cu` vs
  `--extra cpu`). Deferred because there is currently only one developer
  on one machine; the added complexity isn't earned yet. **Revisit when:**
  (a) someone else needs to install this on a CPU-only machine, (b) we
  publish or deploy this anywhere, or (c) we want CI to test on CPU.
- **CUDA toolkit + system-managed CUDA libraries.** Rejected — the bundled
  CUDA runtime in the PyTorch wheels is simpler, more isolated, and
  doesn't depend on the user's system-wide toolkit state.

## Operational notes

**OneDrive gotcha (historical, resolved 2026-05-22).** The project
originally lived in a OneDrive-synced folder. uv attempted to use
hardlinks for cache efficiency, which failed on OneDrive placeholder
files (`os error 396`). A separate failure also occurred where OneDrive
briefly held a file handle on `.venv/Lib/site-packages/...dist-info`
during sync, blocking install cleanup. Both were workable with
`--link-mode=copy` + retries, but the cost grew with project size.

**Resolution:** moved the project to `C:\dev\hecras_mesh_ai\` (off
OneDrive) on 2026-05-22. `--link-mode=copy` is no longer required.
Cloud backup of source code will be handled by a private git remote
once one is set up; the regenerable `.venv/`, `data/`, and ML
artifacts (`lightning_logs/`, `wandb/`, checkpoints) never needed
backup in the first place.

## Verification

After applying this decision:

```
> uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
2.11.0+cu128 True

> uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
NVIDIA GeForce RTX 3090
```

A 4096×4096 matmul on `cuda:0` allocates ~200 MB VRAM and completes
instantaneously.

## References

- ADR 005: Tech Stack
- PyTorch Get Started: https://pytorch.org/get-started/locally/
- uv sources documentation: https://docs.astral.sh/uv/concepts/projects/dependencies/#index
- PyTorch CUDA 12.8 wheels: https://download.pytorch.org/whl/cu128
