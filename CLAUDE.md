# HEC-RAS Mesh AI

Automated 2D computational mesh generation for HEC-RAS using deep learning. Replaces the slow, iterative, intuition-bound manual workflow of breakline placement, refinement region selection, and resolution tuning with a learned system that proposes meshes from terrain and ancillary data — and then refines them against a measurable quantitative objective.

Owner: Dante Mawji. License: MIT.

## Current status

**Phase A — Breakline Detection.** Phase A.0 Week 1 (plumbing) is complete (see `docs/STATUS.md`). Next: build-plan **Stage 1 — Feature & Label Pipeline**.

## Two source-of-truth documents

- `docs/roadmap.md` — the **strategic** phased view (A → B → C → D).
- `docs/build-plan/` — the **executable, checkpointed plan**. One file per stage, each with hard exit criteria. This is the operational source of truth for day-to-day work; start at `00-build-plan-overview.md`.

## How I work with the user

The user has deep HEC-RAS / hydraulic engineering expertise but is **new to ML and PyTorch**. Teach as you go: explain *why*, not just *what*. When a new concept appears (tensor, autograd, U-Net, IoU, Dice loss, DataLoader, LightningModule, etc.), give a short conceptual primer before using it. Assume strong general engineering ability — just no ML background.

Walk through each step rather than dumping finished solutions. Surface significant decisions before making them: present alternatives and tradeoffs, recommend, let the user decide — then record the outcome as an ADR in `docs/decisions/` using `000-template.md`. Be candid; flag risks and uncertainty honestly.

## Tech stack

Per ADR 005 and ADR 010. Python 3.11-3.12, environment managed with `uv`.

- **Core ML:** PyTorch + PyTorch Lightning
- **Model architectures:** `segmentation_models_pytorch` (smp)
- **Geospatial DL:** TorchGeo
- **Geospatial I/O:** rasterio, geopandas, xarray + rioxarray, shapely
- **Image processing:** scikit-image
- **HEC-RAS I/O:** `rashdf` (read), `h5py` (low-level / Phase B writes)
- **Experiment tracking:** Weights & Biases
- **Quality:** pytest, ruff, pre-commit

Dependencies are split into `dev` and `ml` optional groups in `pyproject.toml`. Install with `uv sync --extra dev --extra ml`. `torch`/`torchvision` resolve from the PyTorch CUDA 12.8 index — see ADR 010; the `[tool.uv.sources]` block is Windows + CUDA specific.

## Workflow conventions

- **Notebooks** (`notebooks/`) — exploration only. Throwaway by default; nothing imports from them. (ADR 009)
- **Modules** (`src/hecras_mesh_ai/`) — keepers: anything imported or run more than twice.
- **Tests** (`tests/`) — pytest, run via pre-commit.
- **ADRs** (`docs/decisions/`) — one file per significant decision. Decisions are historical records: never silently rewrite one — mark it superseded, or amend it with a dated note (see ADR 003).
- **Roadmap & build plan** — living documents; update as work completes.
- **Commits** — conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- **Stage discipline** — do not start a build-plan stage until the previous stage's checkpoint criteria are all met and verified.

## Repository layout

```
hecras_mesh_ai/
├── CLAUDE.md                       # this file — auto-loaded by Claude Code
├── README.md
├── pyproject.toml
├── uv.lock
├── .pre-commit-config.yaml
├── .gitattributes
├── .gitignore
├── docs/
│   ├── CLAUDE_CODE_KICKOFF.md      # paste-in prompt to start a session
│   ├── STATUS.md                   # current project status
│   ├── roadmap.md                  # strategic phased plan
│   ├── hec-ras-primer.md           # mesh concepts for ML folks
│   ├── decisions/                  # ADRs 000-012
│   └── build-plan/                 # executable checkpointed stages 00-09
├── src/hecras_mesh_ai/             # the package
├── scripts/                        # helper scripts (e.g. verify_install.py)
├── notebooks/                      # exploration
├── tests/                          # pytest
└── data/                           # gitignored
```

## Decisions on record (ADRs)

001 staged A→B→C delivery · 002 mixed/general-purpose scope · 003 training strategy (amended — expert meshes are a prior, not a target) · 004 hybrid FEMA + curated corpus · 005 tech stack · 006 pilot dataset · 007 make-one-work sprint · 008 Claude Code as system of record · 009 notebooks vs modules · 010 CUDA cu128 platform · 011 quantitative mesh-quality objective · 012 refinement loop is numerical-methods-first.

## HEC-RAS domain context

If unfamiliar with HEC-RAS, read `docs/hec-ras-primer.md` first. Key terms: *breakline*, *refinement region*, *computation point*, *2D flow area*, *sub-grid bathymetry*, *Voronoi mesh*, *cell face*.

## When in doubt

Ask before making non-trivial decisions; document them as ADRs afterward. The cost of an extra prompt is small; the cost of silent drift is large.
