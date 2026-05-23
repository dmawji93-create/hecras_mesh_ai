"""Smoke test: confirm the package imports and reports a version."""

import hecras_mesh_ai


def test_package_imports() -> None:
    assert hecras_mesh_ai.__version__


def test_version_is_string() -> None:
    assert isinstance(hecras_mesh_ai.__version__, str)
