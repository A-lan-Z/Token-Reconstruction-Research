from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

import trr0010_directional_fit_cli as cli


def test_cli_wires_existing_provider_and_writes_guard_receipt(monkeypatch, tmp_path: Path) -> None:
    lease_path = tmp_path / "lease.json"
    current_path = tmp_path / "current.json"
    expanded_path = tmp_path / "expanded.json"
    diagnostic_path = tmp_path / "diagnostic.json"
    lease_path.write_text(json.dumps({"status": "GRANTED"}))
    current_path.write_text(json.dumps({"arm": "current"}))
    expanded_path.write_text(json.dumps({"arm": "expanded"}))
    diagnostic_path.write_text(json.dumps({"synthetic": True}))
    output_root = tmp_path / "fit"
    provider = object()
    seen: dict[str, object] = {}

    def fake_lease(value):
        seen["lease_receipt"] = value
        return {
            "device": "cpu",
            "max_seconds": 10,
            "gpu_reserved_limit_bytes": 100,
            "gpu_free_floor_bytes": 1,
            "host_rss_limit_bytes": 100,
            "host_available_floor_bytes": 1,
            "disk_free_floor_bytes": 1,
        }

    monkeypatch.setattr(cli, "validate_exclusive_lease", fake_lease)
    monkeypatch.setattr(cli, "_load_callable", lambda _spec: provider)
    diagnostic_binding = {
        "schema": cli.DIAGNOSTIC_SCHEMA,
        "task_id": cli.DIAGNOSTIC_TASK_ID,
        "status": cli.DIAGNOSTIC_STATUS,
        "path": str(diagnostic_path),
        "bytes": diagnostic_path.stat().st_size,
        "sha256": "d" * 64,
        "fit_record_count": 64,
        "full_bank_endpoint_steps": [0, 13000],
        "selection_isolated": True,
        "banks": {"B0": {"record_count": 64}, "B1": {"record_count": 64}},
    }
    monkeypatch.setattr(cli, "_load_diagnostic_binding", lambda _path: diagnostic_binding)
    monkeypatch.setattr(cli, "resource_snapshot", lambda **kwargs: {"stage": "synthetic", "device": str(kwargs["device"]), "output_root": str(kwargs["output_root"])})
    monkeypatch.setattr(cli, "enforce_resource_guard", lambda *args, **kwargs: None)

    def fake_bind(value, **kwargs):
        seen["provider"] = value
        seen["binding_receipts"] = kwargs["binding_receipts"]
        seen["diagnostic_binding"] = kwargs["diagnostic_binding"]
        seen["device"] = kwargs["device"]
        return lambda **builder_kwargs: {"builder": builder_kwargs}

    monkeypatch.setattr(cli, "bind_concrete_provider", fake_bind)

    def fake_run(builder, **kwargs):
        seen["builder"] = builder
        seen["run_kwargs"] = kwargs
        kwargs["output_root"].mkdir(parents=True, exist_ok=True)
        (kwargs["output_root"] / "run_receipt.json").write_text("{}\n")
        return {"status": "FIT_COMPLETE"}

    monkeypatch.setattr(cli, "run_directional_arms", fake_run)
    args = argparse.Namespace(
        provider="synthetic:provider",
        binding_current=current_path,
        binding_expanded=expanded_path,
        diagnostic_binding=diagnostic_path,
        lease=lease_path,
        output_root=output_root,
        device="cpu",
        deadline_seconds=10.0,
    )
    result = cli.run_cli(args)
    assert result["status"] == "FIT_COMPLETE"
    assert seen["provider"] is provider
    assert seen["device"] == torch.device("cpu")
    assert seen["diagnostic_binding"] is diagnostic_binding
    assert seen["binding_receipts"] == {
        "current_directional": {"arm": "current"},
        "expanded_directional": {"arm": "expanded"},
    }
    receipt = json.loads((output_root / "resource_guard_receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["truth_opened"] is False
    assert [entry["stage"] for entry in receipt["checks"]] == ["before_provider", "after_fit"]


def test_approved_public_diagnostic_binding_metadata_smoke() -> None:
    path = Path("/tmp/trr-p09/experiments/TRR-P09/setup/fixed-diagnostic-binding-r1.json")
    if not path.is_file():
        pytest.skip("approved A2 diagnostic descriptor is not present in this environment")
    value = cli._load_diagnostic_binding(path)
    assert value["sha256"] == cli.DIAGNOSTIC_FILE_SHA256
    assert value["fit_record_count"] == 64
    assert value["global_row_mapping"].startswith("global_row")
    assert value["banks"]["B0"]["global_indices_sha256"] == cli.DIAGNOSTIC_BANK_DIGESTS["B0"]
    assert value["banks"]["B1"]["global_indices_sha256"] == cli.DIAGNOSTIC_BANK_DIGESTS["B1"]
    assert value["evaluation_truth_opened"] is False


def test_approved_public_diagnostic_binding_rejects_changed_bytes(tmp_path: Path) -> None:
    source = Path("/tmp/trr-p09/experiments/TRR-P09/setup/fixed-diagnostic-binding-r1.json")
    if not source.is_file():
        pytest.skip("approved A2 diagnostic descriptor is not present in this environment")
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(cli.ProductionCLIError, match="SHA-256"):
        cli._load_diagnostic_binding(changed)


def test_cli_single_arm_loads_only_selected_binding_and_uses_arm_runner(monkeypatch, tmp_path: Path) -> None:
    lease_path = tmp_path / "lease.json"
    current_path = tmp_path / "current.json"
    diagnostic_path = tmp_path / "diagnostic.json"
    lease_path.write_text(json.dumps({"status": "GRANTED"}))
    current_path.write_text(json.dumps({"arm": "current"}))
    diagnostic_path.write_text(json.dumps({"synthetic": True}))
    output_root = tmp_path / "fit-current"
    provider = object()
    seen: dict[str, object] = {}

    monkeypatch.setattr(cli, "validate_exclusive_lease", lambda _value: {
        "device": "cpu",
        "max_seconds": 10,
        "gpu_reserved_limit_bytes": 100,
        "gpu_free_floor_bytes": 1,
        "host_rss_limit_bytes": 100,
        "host_available_floor_bytes": 1,
        "disk_free_floor_bytes": 1,
    })
    monkeypatch.setattr(cli, "_load_callable", lambda _spec: provider)
    diagnostic_binding = {
        "schema": cli.DIAGNOSTIC_SCHEMA,
        "task_id": cli.DIAGNOSTIC_TASK_ID,
        "status": cli.DIAGNOSTIC_STATUS,
        "path": str(diagnostic_path),
        "bytes": diagnostic_path.stat().st_size,
        "sha256": "d" * 64,
        "fit_record_count": 64,
        "full_bank_endpoint_steps": [0, 13000],
        "selection_isolated": True,
        "banks": {"B0": {"record_count": 64}, "B1": {"record_count": 64}},
    }
    monkeypatch.setattr(cli, "_load_diagnostic_binding", lambda _path: diagnostic_binding)
    monkeypatch.setattr(cli, "resource_snapshot", lambda **kwargs: {"stage": "single", "device": str(kwargs["device"])})
    monkeypatch.setattr(cli, "enforce_resource_guard", lambda *args, **kwargs: None)

    def fake_bind(value, **kwargs):
        seen["provider"] = value
        seen["binding_receipts"] = kwargs["binding_receipts"]
        return lambda **builder_kwargs: {"builder": builder_kwargs}

    monkeypatch.setattr(cli, "bind_concrete_provider", fake_bind)

    def fake_run(builder, **kwargs):
        seen["builder"] = builder
        seen["run_kwargs"] = kwargs
        kwargs["output_root"].mkdir(parents=True, exist_ok=True)
        (kwargs["output_root"] / "run_receipt.json").write_text("{}\n")
        return {"status": "FIT_COMPLETE", "arm": "current_directional"}

    monkeypatch.setattr(cli, "run_directional_arm", fake_run)
    args = argparse.Namespace(
        provider="synthetic:provider",
        arm="current_directional",
        binding_current=current_path,
        binding_expanded=None,
        diagnostic_binding=diagnostic_path,
        lease=lease_path,
        output_root=output_root,
        device="cpu",
        deadline_seconds=7200.0,
    )
    result = cli.run_cli(args)
    assert result["arm"] == "current_directional"
    assert seen["provider"] is provider
    assert seen["binding_receipts"] == {"current_directional": {"arm": "current"}}
    assert seen["run_kwargs"]["output_root"] == output_root
