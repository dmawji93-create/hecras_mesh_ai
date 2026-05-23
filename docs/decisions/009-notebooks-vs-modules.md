# ADR 009: Notebooks for Exploration, Modules for Keepers

**Status:** Accepted
**Date:** 2026-05-21

## Context

Jupyter notebooks (`.ipynb`) are interactive documents that interleave code, output, and visualizations cell by cell. Excellent for exploration: looking at data, trying transformations, making plots. Python modules (`.py`) are regular code files: testable, importable, reusable. Excellent for production code.

The two extremes both fail:

- **All notebooks:** state becomes implicit and unreproducible, imports get tangled, code is hard to test, and the codebase becomes a graveyard of one-off `.ipynb` files no one can read.
- **All modules from day one:** exploration becomes slow and indirect; every "what does this raster look like" question requires writing a script and running it.

## Decision

**Notebooks for exploration. Modules for keepers.**

- **`notebooks/`** — one-off exploration: data sanity checks, ad-hoc plots, "I want to see what happens when I apply X to Y." Notebooks are throwaway by default; nothing imports from them.
- **`src/hecras_mesh_ai/`** — keepers: any code that will be imported, run more than twice, or run by anyone other than the author. Typed where it helps, tested where it matters.
- When a notebook contains useful logic, **promote it to a module** rather than importing from the notebook. Notebooks can then `from hecras_mesh_ai import …` to use the promoted code.

## Consequences

### Positive
- Fast exploration without sacrificing code quality where it counts.
- Clear mental model for the user (new to ML): notebooks are for "what does this look like," modules are for "how does this thing work that we use everywhere."
- Promotes the habit of refactoring exploration into reusable code rather than letting notebooks calcify.

### Negative / risks
- Requires discipline to actually promote notebook code into modules rather than letting useful logic stay locked in notebooks.
- Notebooks left over from exploration can clutter the repo. Mitigate by treating `notebooks/` as a scratchpad and pruning periodically.

## References

- "Don't import from notebooks" — a near-universal Python data-science convention.
- `docs/decisions/005-tech-stack.md`
