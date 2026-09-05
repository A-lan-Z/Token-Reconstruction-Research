from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = ROOT / "experiments/TRR-0003/footing/comparator_run_guard.py"


def _guard():
    spec = importlib.util.spec_from_file_location("trr0003_footing_comparator_guard_test", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preflight(path: Path, *, minimum_free: int = 8 * 1024**3) -> None:
    path.write_text(
        json.dumps(
            {
                "resource_envelope": {
                    "gpu_envelope_bytes": 6_500_000_000,
                    "host_envelope_bytes": 12_000_000_000,
                    "minimum_free_before_load_bytes": minimum_free,
                    "minimum_margin_fraction": 0.25,
                },
                "qualification_rule": {
                    "cell_id": "finance__public_base",
                    "candidate_budget": 256,
                    "record_batch_size": 4,
                    "require_remaining_free_fraction": 0.25,
                },
            }
        )
    )


def _gpu(free: int, processes: list[str] | None = None) -> dict:
    return {
        "name": "fixture",
        "total_bytes": 16 * 1024**3,
        "free_bytes": free,
        "used_bytes": 0,
        "temperature_c": 40,
        "utilization_pct": 0,
        "compute_processes": [] if processes is None else processes,
    }


def test_resource_preflight_requires_margin_and_eight_gib_before_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard = _guard()
    preflight = tmp_path / "preflight.json"
    _preflight(preflight)
    monkeypatch.setattr(guard, "PREFLIGHT_PATH", preflight)
    monkeypatch.setattr(guard, "_live_gpu", lambda: _gpu(12 * 1024**3))
    monkeypatch.setattr(guard, "_available_host_bytes", lambda: 20 * 1024**3)

    result = guard.resource_preflight()
    assert result["status"] == "PASS"
    assert result["minimum_free_before_load_bytes"] == 8 * 1024**3
    assert result["required_gpu_free_bytes"] == 8_666_666_667
    assert result["checks"] == {
        "gpu_margin_pass": True,
        "host_margin_pass": True,
        "thermal_pass": True,
        "exclusive_gpu_pass": True,
    }

    monkeypatch.setattr(guard, "_live_gpu", lambda: _gpu(7 * 1024**3))
    with pytest.raises(guard.GuardError, match="resource margin failed"):
        guard.resource_preflight()


def test_resource_preflight_blocks_nonexclusive_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard = _guard()
    preflight = tmp_path / "preflight.json"
    _preflight(preflight)
    monkeypatch.setattr(guard, "PREFLIGHT_PATH", preflight)
    monkeypatch.setattr(guard, "_live_gpu", lambda: _gpu(12 * 1024**3, ["123, python, 100 MiB"]))
    monkeypatch.setattr(guard, "_available_host_bytes", lambda: 20 * 1024**3)
    with pytest.raises(guard.GuardError, match="resource margin failed"):
        guard.resource_preflight()


def test_measured_peak_uses_largest_nested_cuda_peak(tmp_path: Path) -> None:
    guard = _guard()
    path = tmp_path / "evidence.json"
    path.write_text(
        json.dumps(
            {
                "timing": [
                    {"peak_memory": {"cuda_peak_allocated_bytes": 3, "cuda_peak_reserved_bytes": 5}},
                    {"peak_memory": {"cuda_peak_allocated_bytes": 7, "cuda_peak_reserved_bytes": 11}},
                ]
            }
        )
    )
    # The helper is intentionally tested without a repository-relative path;
    # measured_peak's path label is a presentation detail in real runs.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    try:
        result = guard.measured_peak(path)
    finally:
        monkeypatch.undo()
    assert result["cuda_peak_allocated_bytes"] == 7
    assert result["cuda_peak_reserved_bytes"] == 11
    assert result["observations"] == 2


def test_comparator_does_not_hardcode_zero_prefix_calls() -> None:
    source = (ROOT / "scripts/trr0003_footing_compare.py").read_text()
    assert "checked_cache_transitions" in source
    assert '"public_prefix_calls": prefix_calls' in source
    assert '"public_prefix_calls": 0' not in source
