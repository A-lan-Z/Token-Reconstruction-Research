from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from scripts import trr0010_a1_a2_opened_fixture_equivalence as diagnostic


def test_resource_guard_uses_existing_ancestor_for_prospective_output(
    tmp_path: Path, monkeypatch
) -> None:
    prospective = tmp_path / "not-yet-created" / "opened-fixture"
    observed_paths: list[Path] = []

    monkeypatch.setattr(diagnostic, "_rss_bytes", lambda: 0)
    monkeypatch.setattr(diagnostic, "_host_available_bytes", lambda: 12 * 2**30)

    def disk_usage(path: Path) -> SimpleNamespace:
        observed_paths.append(Path(path))
        return SimpleNamespace(free=24 * 2**30)

    monkeypatch.setattr(diagnostic.shutil, "disk_usage", disk_usage)
    monkeypatch.setattr(
        diagnostic.legacy,
        "_resource_preflight",
        lambda *args, **kwargs: {"status": "PASS", "stage": kwargs["stage"]},
    )

    result = diagnostic._resource_guard(
        args=Namespace(output_root=prospective, max_seconds=30.0),
        device=torch.device("cpu"),
        started=time.perf_counter(),
        stage="before_load",
    )

    assert result["status"] == "PASS"
    assert observed_paths == [tmp_path]
    assert not prospective.exists()
