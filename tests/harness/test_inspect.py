"""Tests for the HDF structure inspector."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from hecras_mesh_ai.harness import dump_structure, walk_hdf


def _make_synthetic_hdf(tmp_path: Path) -> Path:
    """Build a small HDF with a few groups, datasets, attributes — exercising
    each variant the real HEC-RAS HDFs use."""
    p = tmp_path / "synthetic.hdf"
    with h5py.File(p, "w") as f:
        f.attrs["File Type"] = b"Test"
        f.attrs["Version"] = 1.0
        g = f.create_group("Geometry")
        g.attrs["Description"] = b"a 2D flow area"
        sub = g.create_group("2D Flow Areas")
        sub.create_dataset("Cell Count", data=np.array([42], dtype=np.int32))
        sub.create_dataset(
            "Cell Coords",
            data=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        )
        # A dataset with attributes
        bls = g.create_dataset("Breaklines", data=np.array([1, 2, 3]))
        bls.attrs["Names"] = np.array([b"Road 1", b"HighGround 1"], dtype="S20")
    return p


def test_walk_emits_root_groups_and_datasets(tmp_path):
    p = _make_synthetic_hdf(tmp_path)
    nodes = list(walk_hdf(p))
    paths = [n.path for n in nodes]
    assert "/" in paths
    assert "/Geometry" in paths
    assert "/Geometry/2D Flow Areas" in paths
    assert "/Geometry/2D Flow Areas/Cell Count" in paths
    assert "/Geometry/2D Flow Areas/Cell Coords" in paths
    assert "/Geometry/Breaklines" in paths


def test_walk_classifies_groups_and_datasets(tmp_path):
    p = _make_synthetic_hdf(tmp_path)
    nodes = {n.path: n for n in walk_hdf(p)}
    assert nodes["/"].kind == "group"
    assert nodes["/Geometry"].kind == "group"
    assert nodes["/Geometry/2D Flow Areas/Cell Coords"].kind == "dataset"
    assert nodes["/Geometry/Breaklines"].kind == "dataset"


def test_walk_captures_dataset_shape_and_dtype(tmp_path):
    p = _make_synthetic_hdf(tmp_path)
    nodes = {n.path: n for n in walk_hdf(p)}
    coords = nodes["/Geometry/2D Flow Areas/Cell Coords"]
    assert coords.shape == (2, 2)
    assert "float64" in coords.dtype


def test_walk_decodes_bytes_attributes(tmp_path):
    p = _make_synthetic_hdf(tmp_path)
    nodes = {n.path: n for n in walk_hdf(p)}
    assert nodes["/"].attrs["File Type"] == "Test"
    assert nodes["/Geometry"].attrs["Description"] == "a 2D flow area"


def test_walk_decodes_bytes_ndarray_attributes(tmp_path):
    p = _make_synthetic_hdf(tmp_path)
    nodes = {n.path: n for n in walk_hdf(p)}
    names = nodes["/Geometry/Breaklines"].attrs["Names"]
    assert names == ["Road 1", "HighGround 1"]


def test_dump_structure_produces_markdown_with_expected_sections(tmp_path):
    p = _make_synthetic_hdf(tmp_path)
    md = dump_structure(p)
    assert md.startswith("# HDF structure")
    assert "**Groups:**" in md
    assert "**Datasets:**" in md
    assert "| path | kind | shape | dtype | attrs | attribute preview |" in md
    # Body rows include the synthetic paths.
    assert "`/Geometry/Breaklines`" in md
    assert "`/Geometry/2D Flow Areas/Cell Coords`" in md
    # Decoded attribute values appear.
    assert "Road 1" in md


def test_dump_structure_title_override(tmp_path):
    p = _make_synthetic_hdf(tmp_path)
    md = dump_structure(p, title="Custom Title")
    assert md.startswith("# Custom Title")


# ---------------------------------------------------------------------------
# Integration tests against real pilot HDFs
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PILOT_BASE = (
    _REPO_ROOT
    / "data"
    / "raw"
    / "usace"
    / "RAS Samples"
    / "Example_Projects_7_0"
    / "2D Unsteady Flow Hydraulics"
)
_MUNCIE_HDF = _PILOT_BASE / "Muncie" / "Muncie.g04.hdf"
_BALDEAGLE_HDF = _PILOT_BASE / "BaldEagleCrkMulti2D" / "BaldEagleDamBrk.g09.hdf"


@pytest.mark.skipif(not _MUNCIE_HDF.exists(), reason="Muncie HDF not present")
def test_walk_muncie_hdf_returns_substantial_structure():
    nodes = list(walk_hdf(_MUNCIE_HDF))
    # Sanity: a real HEC-RAS geometry has many groups + datasets even on
    # a small example like Muncie.
    n_groups = sum(1 for n in nodes if n.kind == "group")
    n_datasets = sum(1 for n in nodes if n.kind == "dataset")
    assert n_groups >= 5
    assert n_datasets >= 30
    # Some path we expect to exist in any HEC-RAS geometry HDF.
    paths = {n.path for n in nodes}
    assert any("Geometry" in p for p in paths)


@pytest.mark.skipif(not _MUNCIE_HDF.exists(), reason="Muncie HDF not present")
def test_dump_structure_muncie_writes_valid_markdown():
    md = dump_structure(_MUNCIE_HDF)
    # Spot-check that key sections render — should include a 2D Flow Areas group.
    assert "2D Flow Areas" in md
    # Markdown table header present.
    assert "| path | kind | shape | dtype | attrs | attribute preview |" in md


@pytest.mark.skipif(not _BALDEAGLE_HDF.exists(), reason="Bald Eagle HDF not present")
def test_walk_baldeagle_hdf_includes_embedded_projection():
    """g09 is the only Bald Eagle geometry with an HDF-embedded CRS.
    The Projection attribute / dataset should be visible in the dump."""
    nodes = list(walk_hdf(_BALDEAGLE_HDF))
    text = " ".join(n.path + " " + str(n.attrs) for n in nodes)
    assert "Projection" in text or "PROJCS" in text
