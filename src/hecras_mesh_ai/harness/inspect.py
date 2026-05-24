"""Recursive HDF5 structure dumper for HEC-RAS geometry schema work.

USACE has never published the HEC-RAS `.gNN.hdf` geometry schema; `rashdf`
itself was built by reverse-engineering. To implement a *writer* for valid
geometries we have to know which groups, datasets, and attributes exist
and what shape/dtype each has. This module dumps that information in a
human-readable Markdown table for any HEC-RAS HDF.

Usage:
    from hecras_mesh_ai.harness import dump_structure
    md = dump_structure("Muncie.g04.hdf", max_array_preview=4)
    Path("docs/hdf-schema/muncie-g04.md").write_text(md)

The output is intended to be committed alongside the writer code so that
future readers (human or AI) have a stable, grep-able reference for the
schema each writer module relies on.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class HdfNode:
    """One entry from a recursive HDF walk."""

    path: str
    kind: str  # "group" or "dataset"
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    n_attrs: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)


def walk_hdf(hdf_path: Path | str) -> Iterator[HdfNode]:
    """Yield every group and dataset in `hdf_path` in pre-order.

    The root group `/` is included. Attribute values are coerced to plain
    Python types for ease of inspection (bytes -> str, numpy scalars ->
    int/float, numpy arrays kept as ndarrays with shape/dtype).
    """
    hdf_path = Path(hdf_path)
    with h5py.File(hdf_path, "r") as f:

        def _attrs_to_dict(obj: h5py.HLObject) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for k, v in obj.attrs.items():
                if isinstance(v, bytes):
                    try:
                        out[k] = v.decode("utf-8")
                    except UnicodeDecodeError:
                        out[k] = repr(v)
                elif isinstance(v, np.ndarray) and v.dtype.kind in ("S", "O"):
                    out[k] = [x.decode("utf-8") if isinstance(x, bytes) else x for x in v.tolist()]
                elif isinstance(v, np.integer | np.floating):
                    out[k] = v.item()
                else:
                    out[k] = v
            return out

        def _emit(name: str, obj: h5py.HLObject) -> None:
            path = "/" + name if not name.startswith("/") else name
            if isinstance(obj, h5py.Dataset):
                node = HdfNode(
                    path=path,
                    kind="dataset",
                    shape=tuple(obj.shape),
                    dtype=str(obj.dtype),
                    n_attrs=len(obj.attrs),
                    attrs=_attrs_to_dict(obj),
                )
            else:
                node = HdfNode(
                    path=path,
                    kind="group",
                    n_attrs=len(obj.attrs),
                    attrs=_attrs_to_dict(obj),
                )
            collected.append(node)

        collected: list[HdfNode] = []
        # Root
        collected.append(
            HdfNode(path="/", kind="group", n_attrs=len(f.attrs), attrs=_attrs_to_dict(f))
        )
        # Recurse
        f.visititems(_emit)
        yield from collected


def _fmt_attrs(attrs: dict[str, Any], max_chars: int = 120) -> str:
    """Render an attribute dict to a single-line preview string."""
    if not attrs:
        return ""
    parts: list[str] = []
    for k, v in attrs.items():
        if isinstance(v, np.ndarray):
            preview = f"<ndarray shape={v.shape} dtype={v.dtype}>"
        elif isinstance(v, list | tuple) and len(v) > 6:
            preview = f"[{', '.join(repr(x) for x in v[:3])}, ..., {repr(v[-1])}] (len={len(v)})"
        elif isinstance(v, str) and len(v) > 60:
            preview = repr(v[:57] + "...")
        else:
            preview = repr(v)
        parts.append(f"{k}={preview}")
    rendered = "; ".join(parts)
    if len(rendered) > max_chars:
        rendered = rendered[: max_chars - 3] + "..."
    return rendered


def dump_structure(
    hdf_path: Path | str,
    *,
    title: str | None = None,
) -> str:
    """Return a Markdown dump of the HDF's full structure.

    The output is one row per group / dataset, in a table with columns:
    path, kind, shape, dtype, # attrs, attribute preview.

    Parameters
    ----------
    hdf_path
        Path to the .hdf file to inspect.
    title
        Optional title line for the Markdown document. Defaults to the
        file's name.
    """
    hdf_path = Path(hdf_path)
    title = title or f"HDF structure — `{hdf_path.name}`"

    nodes = list(walk_hdf(hdf_path))
    n_groups = sum(1 for n in nodes if n.kind == "group")
    n_datasets = sum(1 for n in nodes if n.kind == "dataset")

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Source:** `{hdf_path}`")
    lines.append(f"**File size:** {hdf_path.stat().st_size / 1024:.0f} KB")
    lines.append(f"**Groups:** {n_groups} · **Datasets:** {n_datasets}")
    lines.append("")
    lines.append(
        "Generated by `hecras_mesh_ai.harness.inspect.dump_structure`. "
        "One row per group or dataset in pre-order traversal. Attribute "
        "previews are truncated to ~120 chars."
    )
    lines.append("")
    lines.append("| path | kind | shape | dtype | attrs | attribute preview |")
    lines.append("|---|---|---|---|---:|---|")
    for n in nodes:
        shape = "" if n.shape is None else " × ".join(str(s) for s in n.shape) or "scalar"
        dtype = n.dtype or ""
        # Escape pipes inside attribute previews so they don't break the table.
        attr_preview = _fmt_attrs(n.attrs).replace("|", "\\|")
        path_md = f"`{n.path}`"
        lines.append(f"| {path_md} | {n.kind} | {shape} | {dtype} | {n.n_attrs} | {attr_preview} |")

    return "\n".join(lines) + "\n"
