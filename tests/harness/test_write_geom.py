"""Tests for the breakline-replacement writer."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from hecras_mesh_ai.harness import (
    Breakline,
    read_breaklines,
    replace_breaklines,
)
from hecras_mesh_ai.harness.write_geom import (
    ATTRIBUTES_DTYPE,
    BREAKLINES_PATH,
    _pack,
)

# ---------------------------------------------------------------------------
# Pure-Python packing logic
# ---------------------------------------------------------------------------


def test_pack_single_breakline_layout():
    bl = Breakline(name="Test", points=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    packed = _pack([bl])
    assert packed.n_features == 1
    assert packed.attributes.dtype == ATTRIBUTES_DTYPE
    assert packed.attributes[0]["Name"] == b"Test"
    np.testing.assert_array_equal(packed.polyline_info, [[0, 3, 0, 1]])
    np.testing.assert_array_equal(packed.polyline_parts, [[0, 3]])
    np.testing.assert_array_equal(packed.polyline_points, [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])


def test_pack_multiple_breaklines_indices_chain_correctly():
    bls = [
        Breakline(name="A", points=np.array([[0.0, 0.0], [1.0, 1.0]])),
        Breakline(name="B", points=np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])),
        Breakline(name="C", points=np.array([[5.0, 5.0], [6.0, 6.0]])),
    ]
    packed = _pack(bls)
    # Polyline Info: global point starting indices accumulate.
    np.testing.assert_array_equal(
        packed.polyline_info,
        [
            [0, 2, 0, 1],
            [2, 3, 1, 1],
            [5, 2, 2, 1],
        ],
    )
    # Polyline Parts: each row's offset is RELATIVE to the feature's own
    # point block — always 0 for single-part.
    np.testing.assert_array_equal(
        packed.polyline_parts,
        [
            [0, 2],
            [0, 3],
            [0, 2],
        ],
    )
    assert packed.polyline_points.shape == (7, 2)


def test_pack_empty_breaklines_produces_zero_length_arrays():
    packed = _pack([])
    assert packed.n_features == 0
    assert packed.attributes.shape == (0,)
    assert packed.polyline_info.shape == (0, 4)
    assert packed.polyline_parts.shape == (0, 2)
    assert packed.polyline_points.shape == (0, 2)


# ---------------------------------------------------------------------------
# Breakline validation
# ---------------------------------------------------------------------------


def test_breakline_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        Breakline(name="x", points=np.array([0.0, 1.0, 2.0]))


def test_breakline_rejects_single_point():
    with pytest.raises(ValueError, match="at least 2 points"):
        Breakline(name="x", points=np.array([[0.0, 0.0]]))


def test_breakline_rejects_overlong_name():
    with pytest.raises(ValueError, match="32 bytes"):
        Breakline(name="x" * 33, points=np.array([[0.0, 0.0], [1.0, 1.0]]))


# ---------------------------------------------------------------------------
# End-to-end write against a synthetic source HDF
# ---------------------------------------------------------------------------


def _make_minimal_source_hdf(tmp_path: Path) -> Path:
    """A near-minimal valid-shape geometry HDF: just root attrs + Geometry
    + a placeholder breaklines group we'll overwrite."""
    p = tmp_path / "src.g01.hdf"
    with h5py.File(p, "w") as f:
        f.attrs["File Type"] = b"HEC-RAS Geometry"
        f.attrs["File Version"] = b"6.6"
        g = f.create_group("Geometry")
        g.attrs["Title"] = b"synthetic"
        # Throw a "preserved" sentinel dataset in so we can verify it survives.
        f.create_dataset(
            "/Geometry/2D Flow Areas/Preserved Marker",
            data=np.array([1, 2, 3], dtype=np.int32),
        )
        # Initial (different) breaklines to be overwritten.
        bgroup = f.create_group(BREAKLINES_PATH)
        bgroup.create_dataset("Attributes", data=np.zeros(1, dtype=ATTRIBUTES_DTYPE))
        bgroup.create_dataset("Polyline Info", data=np.zeros((1, 4), dtype=np.int32))
        bgroup.create_dataset("Polyline Parts", data=np.zeros((1, 2), dtype=np.int32))
        bgroup.create_dataset("Polyline Points", data=np.zeros((2, 2), dtype=np.float64))
    return p


def test_replace_breaklines_overwrites_target_and_preserves_other_groups(tmp_path):
    src = _make_minimal_source_hdf(tmp_path)
    tgt = tmp_path / "out.g01.hdf"

    bls = [
        Breakline(name="NewLine", points=np.array([[10.0, 10.0], [11.0, 11.0]])),
    ]
    out = replace_breaklines(src, tgt, bls)
    assert out == tgt
    assert tgt.exists()

    with h5py.File(tgt, "r") as f:
        # Other groups preserved.
        assert "/Geometry/2D Flow Areas/Preserved Marker" in f
        np.testing.assert_array_equal(f["/Geometry/2D Flow Areas/Preserved Marker"][:], [1, 2, 3])
        # Breaklines replaced.
        g = f[BREAKLINES_PATH]
        assert g["Attributes"][0]["Name"] == b"NewLine"
        np.testing.assert_array_equal(g["Polyline Info"][:], [[0, 2, 0, 1]])
        np.testing.assert_array_equal(g["Polyline Parts"][:], [[0, 2]])
        np.testing.assert_array_equal(g["Polyline Points"][:], [[10.0, 10.0], [11.0, 11.0]])
        # Attrs match HEC-RAS layout.
        assert g["Polyline Info"].attrs["Feature Type"] == b"Polyline"
        assert g["Polyline Points"].attrs["Row"] == b"Points"


def test_replace_breaklines_with_empty_list_removes_group(tmp_path):
    src = _make_minimal_source_hdf(tmp_path)
    tgt = tmp_path / "out.g01.hdf"
    replace_breaklines(src, tgt, [])
    with h5py.File(tgt, "r") as f:
        assert BREAKLINES_PATH not in f


def test_replace_breaklines_refuses_to_overwrite_existing_target(tmp_path):
    src = _make_minimal_source_hdf(tmp_path)
    tgt = tmp_path / "out.g01.hdf"
    tgt.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        replace_breaklines(src, tgt, [])


def test_replace_breaklines_in_place_requires_overwrite(tmp_path):
    """source==target mutates the ORIGINAL file — that needs explicit opt-in.

    Regression for the audit finding that flag-less in-place calls silently
    destroyed the original breaklines (contradicting the docstring)."""
    src = _make_minimal_source_hdf(tmp_path)
    bls = [Breakline(name="InPlace", points=np.array([[0.0, 0.0], [1.0, 1.0]]))]
    with pytest.raises(FileExistsError, match="in-place"):
        replace_breaklines(src, src, bls)
    # Original untouched — the placeholder breakline is still there.
    with h5py.File(src, "r") as f:
        assert f[BREAKLINES_PATH]["Attributes"][0]["Name"] == b""


def test_replace_breaklines_in_place_with_overwrite_patches_file(tmp_path):
    """With overwrite=True, source==target patches in place, no copy error."""
    src = _make_minimal_source_hdf(tmp_path)
    bls = [Breakline(name="InPlace", points=np.array([[0.0, 0.0], [1.0, 1.0]]))]
    out = replace_breaklines(src, src, bls, overwrite=True)
    assert out == src
    with h5py.File(src, "r") as f:
        assert f[BREAKLINES_PATH]["Attributes"][0]["Name"] == b"InPlace"
        # Preserved-marker sentinel must survive in-place patching.
        np.testing.assert_array_equal(f["/Geometry/2D Flow Areas/Preserved Marker"][:], [1, 2, 3])


def test_read_breaklines_refuses_multipart(tmp_path):
    """Multi-part breaklines must fail loudly, not silently weld parts
    together with a bridge segment (audit finding — corpus files can be
    multi-part even though both pilots are single-part)."""
    p = tmp_path / "multi.g01.hdf"
    with h5py.File(p, "w") as f:
        g = f.create_group(BREAKLINES_PATH)
        attrs = np.zeros(1, dtype=ATTRIBUTES_DTYPE)
        attrs[0]["Name"] = b"TwoPart"
        g.create_dataset("Attributes", data=attrs)
        # One feature, 5 points, 2 parts (3 + 2 points).
        g.create_dataset("Polyline Info", data=np.array([[0, 5, 0, 2]], dtype=np.int32))
        g.create_dataset("Polyline Parts", data=np.array([[0, 3], [3, 2]], dtype=np.int32))
        g.create_dataset("Polyline Points", data=np.arange(10, dtype=np.float64).reshape(5, 2))
    with pytest.raises(ValueError, match="multi-part"):
        read_breaklines(p)


def test_replace_breaklines_overwrites_when_flag_set(tmp_path):
    src = _make_minimal_source_hdf(tmp_path)
    tgt = tmp_path / "out.g01.hdf"
    tgt.write_bytes(b"existing")
    replace_breaklines(src, tgt, [], overwrite=True)
    assert tgt.exists()
    # Should be a valid HDF, not the leftover bytes.
    with h5py.File(tgt, "r") as f:
        assert "Geometry" in f


# ---------------------------------------------------------------------------
# Round-trip on real Muncie HDF — the key test
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MUNCIE_HDF = (
    _REPO_ROOT
    / "data"
    / "raw"
    / "usace"
    / "RAS Samples"
    / "Example_Projects_7_0"
    / "2D Unsteady Flow Hydraulics"
    / "Muncie"
    / "Muncie.g04.hdf"
)


@pytest.mark.skipif(not _MUNCIE_HDF.exists(), reason="Muncie HDF not present")
def test_round_trip_muncie_breaklines_preserves_data(tmp_path):
    """Read Muncie's breaklines, write them back to a new file, re-read,
    and verify exact equivalence. This is the primary correctness test —
    if it passes, our writer matches HEC-RAS's own layout."""
    original = read_breaklines(_MUNCIE_HDF)
    assert len(original) >= 1, "Muncie should have at least one breakline"

    out_hdf = tmp_path / "muncie_rewrite.g04.hdf"
    replace_breaklines(_MUNCIE_HDF, out_hdf, original)

    roundtrip = read_breaklines(out_hdf)
    assert len(roundtrip) == len(original)
    for o, r in zip(original, roundtrip, strict=True):
        assert o.name == r.name
        np.testing.assert_array_equal(o.points, r.points)
        assert o.cell_spacing_near == r.cell_spacing_near
        assert o.cell_spacing_far == r.cell_spacing_far
        assert o.near_repeats == r.near_repeats
        assert o.protection_radius == r.protection_radius


@pytest.mark.skipif(not _MUNCIE_HDF.exists(), reason="Muncie HDF not present")
def test_round_trip_preserves_hdf_layout_exactly(tmp_path):
    """Beyond Python-level equivalence, the HDF datasets themselves should
    match Muncie's bit-for-bit (modulo gzip-level differences)."""
    original = read_breaklines(_MUNCIE_HDF)
    out_hdf = tmp_path / "muncie_rewrite.g04.hdf"
    replace_breaklines(_MUNCIE_HDF, out_hdf, original)

    with h5py.File(_MUNCIE_HDF, "r") as orig, h5py.File(out_hdf, "r") as rewrite:
        for name in ("Attributes", "Polyline Info", "Polyline Parts", "Polyline Points"):
            o_data = orig[f"{BREAKLINES_PATH}/{name}"][:]
            r_data = rewrite[f"{BREAKLINES_PATH}/{name}"][:]
            np.testing.assert_array_equal(
                o_data, r_data, err_msg=f"dataset {name!r} differs after round-trip"
            )


@pytest.mark.skipif(not _MUNCIE_HDF.exists(), reason="Muncie HDF not present")
def test_replace_with_new_breaklines_on_muncie_preserves_unrelated_groups(tmp_path):
    """Sanity: replacing breaklines doesn't disturb other parts of the
    geometry HDF that HEC-RAS will need to read."""
    out_hdf = tmp_path / "muncie_new_breaklines.g04.hdf"
    new_bls = [
        Breakline(
            name="SyntheticTestLine",
            points=np.array([[409000.0, 1802000.0], [409100.0, 1802100.0], [409200.0, 1802200.0]]),
        ),
    ]
    replace_breaklines(_MUNCIE_HDF, out_hdf, new_bls, overwrite=False)

    with h5py.File(out_hdf, "r") as f:
        # 2D Flow Areas group preserved.
        assert "/Geometry/2D Flow Areas" in f
        assert "/Geometry/2D Flow Areas/Attributes" in f
        # File-level attrs preserved.
        assert "File Type" in f.attrs
        # New breakline written.
        g = f[BREAKLINES_PATH]
        assert g["Attributes"].shape == (1,)
        assert g["Attributes"][0]["Name"] == b"SyntheticTestLine"
