from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = ROOT / "experiments/TRR-0003/track_b/run_guard.py"


def _guard_module():
    spec = importlib.util.spec_from_file_location("trr0003_track_b_run_guard_test", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preflight(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "qualification_rule": {"minimum_margin_fraction": 0.25},
                "public_preparation_peak_envelope": {
                    "gpu_envelope_bytes": 600,
                    "host_envelope_bytes": 600,
                },
                "predicted_peak_envelope": {"affine_training_bytes": 500},
            }
        )
    )


def test_resource_preflight_checks_live_margin_and_exclusivity(tmp_path, monkeypatch) -> None:
    guard = _guard_module()
    preflight = tmp_path / "preflight.json"
    _preflight(preflight)
    monkeypatch.setattr(guard, "PREFLIGHT_PATH", preflight)
    monkeypatch.setattr(
        guard,
        "_live_gpu",
        lambda: {
            "name": "test",
            "total_bytes": 5000,
            "free_bytes": 1000,
            "used_bytes": 4000,
            "temperature_c": 40,
            "utilization_pct": 0,
            "compute_processes": [],
        },
    )
    monkeypatch.setattr(guard, "_available_host_bytes", lambda: 1000)
    result = guard.resource_preflight()
    assert result["status"] == "PASS"
    assert result["required_gpu_free_bytes"] == 800
    assert result["required_host_available_bytes"] == 800
    assert result["checks"] == {
        "gpu_margin_pass": True,
        "host_margin_pass": True,
        "thermal_pass": True,
        "exclusive_gpu_pass": True,
    }


def test_resource_preflight_fails_before_run_when_margin_is_insufficient(tmp_path, monkeypatch) -> None:
    guard = _guard_module()
    preflight = tmp_path / "preflight.json"
    _preflight(preflight)
    monkeypatch.setattr(guard, "PREFLIGHT_PATH", preflight)
    monkeypatch.setattr(
        guard,
        "_live_gpu",
        lambda: {
            "name": "test",
            "total_bytes": 5000,
            "free_bytes": 799,
            "used_bytes": 4201,
            "temperature_c": 40,
            "utilization_pct": 0,
            "compute_processes": [],
        },
    )
    monkeypatch.setattr(guard, "_available_host_bytes", lambda: 1000)
    with pytest.raises(RuntimeError, match="live resource margin failed"):
        guard.resource_preflight()


def test_main_reports_source_edit_and_returns_failure(monkeypatch, tmp_path) -> None:
    guard = _guard_module()
    snapshots = iter(
        [
            {"utc": "start", "git_commit": "a", "source_hashes": {"frozen": "one"}},
            {"utc": "end", "git_commit": "a", "source_hashes": {"frozen": "two"}},
        ]
    )
    monkeypatch.setattr(guard, "snapshot", lambda: next(snapshots))
    monkeypatch.setattr(guard, "resource_preflight", lambda: {"status": "PASS"})
    monkeypatch.setattr(guard.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    evidence = tmp_path / "guard.json"
    monkeypatch.setattr(guard, "sys", SimpleNamespace(argv=["run_guard.py", str(evidence), "true"]))
    assert guard.main() == 3
    payload = json.loads(evidence.read_text())
    assert payload["frozen_code_edit_during_run"] is True
    assert payload["source_hashes_unchanged"] is False
    assert payload["guard_passed"] is False
    assert payload["command_returncode"] == 0
