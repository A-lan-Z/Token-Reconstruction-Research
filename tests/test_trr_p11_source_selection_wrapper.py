"""Pure path-resolution tests for the P11 selector entrypoint."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.trr_p11 import run_source_selection


def test_relative_inputs_resolve_from_declared_repository_root(tmp_path: Path) -> None:
    root = tmp_path / "TRR-P11"
    sibling = tmp_path / "TRR-0010"
    root.mkdir()
    sibling.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text("{}")

    assert run_source_selection._resolve_path(
        Path("manifest.json"), root=root, description="manifest", require_exists=True
    ) == manifest.resolve()
    assert run_source_selection._resolve_path(
        Path("../TRR-0010"), root=root, description="TRR-0010", require_exists=True
    ) == sibling.resolve()


def test_missing_required_path_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="unavailable"):
        run_source_selection._resolve_path(
            Path("missing.json"), root=tmp_path, description="source inputs", require_exists=True
        )
