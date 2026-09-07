from __future__ import annotations

import json
from pathlib import Path

import pytest

import trr0010_production_provider as provider



def _file_record(root: Path, name: str, content: bytes = b"metadata-only") -> dict[str, object]:
    path = root / name
    path.write_bytes(content)
    import hashlib
    return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "status": "VERIFIED"}



def _binding(tmp_path: Path, *, arm_name: str, bank_role: str, diagnostic: dict[str, object]) -> dict:
    import hashlib
    artifacts = {}
    for role in ("contract", "bank_manifest", "schedule", "base_state", "public_embedding", "support_ids", "support_counts"):
        descriptor = _file_record(tmp_path, f"{arm_name}-{role}.bin")
        artifacts[role] = descriptor
    artifacts["contract"]["status"] = "FROZEN"
    artifacts["schedule"]["status"] = "FROZEN"
    artifacts["base_state"].update({"sha256": provider.EXPECTED_START_STATE_SHA256, "selected_step": 400, "method_id": "trr0007_residual_mlp512"})
    artifacts["fit_manifest"] = dict(artifacts["bank_manifest"])

    sources = {}
    for relative in provider.REQUIRED_A2_SOURCE_PATHS + provider.HELPER_SOURCE_PATHS:
        leaf = relative.replace("/", "_")
        content = (relative + "\n").encode()
        sources[relative] = _file_record(tmp_path, f"{arm_name}-{leaf}", content)
        sources[relative]["commit"] = "a" * 40
    resource = {
        "status": "PASS",
        "gpu_peak_reserved_bytes": 7600078848,
        "gpu_reserved_limit_bytes": 10737418240,
        "host_peak_rss_bytes": 4276609024,
        "host_rss_limit_bytes": 12884901888,
        "host_available_bytes": 18282762240,
        "host_available_floor_bytes": 8589934592,
        "wall_seconds": 26.84726572499494,
        "wall_limit_seconds": 7200.0,
        "production_caps": {
            "gpu_reserved_limit_bytes": 10737418240,
            "host_rss_limit_bytes": 12884901888,
            "host_available_floor_bytes": 8589934592,
            "wall_limit_seconds": 7200,
            "gpu_free_floor_bytes": 2147483648,
            "disk_free_floor_bytes": 21474836480,
            "output_bytes_limit": 5368709120,
        },
    }
    return {
        "schema": provider.PRODUCTION_SCHEMA,
        "task_id": "TRR-0010",
        "arm_name": arm_name,
        "bank_role": bank_role,
        "artifacts": artifacts,
        "fit_manifest": artifacts["fit_manifest"],
        "a2_sources": sources,
        "settings": {
            "steps": 13000,
            "batch_records": 8,
            "position_budget": 512,
            "sequence_tokens": 192,
            "hidden_size": 2048,
            "vocabulary_size": 128256,
            "base_learning_rate": 2e-4,
            "directional_learning_rate": 1e-4,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "optimizer": "AdamW",
            "optimizer_foreach": False,
            "optimizer_groups": ["decoder", "directional_delta"],
        },
        "training": {
            "steps": 13000,
            "validation_every": 1000,
            "validation_sequence_tokens": 128,
            "selection_metric": "domain_balanced_token_accuracy",
            "gradient_clip_norm": 1.0,
        },
        "schedule": {
            "steps": 13000,
            "seed": 4010,
            "semantic_sha256": "b" * 64,
            "exposure": {"draws_per_step": 512},
        },
        "control_schedule": {"sha256": artifacts["schedule"]["sha256"]},
        "validation_geometry": {"record_count": 384, "sequence_tokens": 128},
        "resource_qualification": resource,
    }


def _prepare_synthetic_bindings(receipts: dict[str, dict]) -> tuple[str, str]:
    """Use explicit test-side hash injection with a proper JSON contract."""
    import hashlib
    root = Path(next(iter(receipts.values()))["artifacts"]["contract"]["path"]).parent
    fixed = {}
    for receipt in receipts.values():
        role = receipt["bank_role"]
        schedule = receipt["schedule"]
        artifact = receipt["artifacts"]["schedule"]
        fixed[f"{role}_schedule"] = {
            "bytes": artifact["bytes"],
            "path": artifact["path"],
            "seed": schedule["seed"],
            "semantic_sha256": schedule["semantic_sha256"],
            "sha256": artifact["sha256"],
            "steps": schedule["steps"],
        }
    contract = {
        "schema": provider.SIGNED_STAGE3_CONTRACT_SCHEMA,
        "task_id": "TRR-0010",
        "status": "FROZEN_STAGE3_AGREED_PENDING_EXACT_HASH_COUNTERSIGNATURE",
        "fixed_control_configuration": fixed,
        "resource_caps": {"per_arm": {
            "cuda_reserved_bytes_directional": 10737418240,
            "host_rss_bytes": 12884901888,
            "host_available_floor_bytes": 8589934592,
            "max_seconds_including_preparation_diagnostics_checkpoint_export": 7200,
            "gpu_free_floor_bytes": 2147483648,
            "disk_free_floor_bytes": 21474836480,
            "output_bytes_limit": 5368709120,
        }},
        "directional_readout": {"support_by_bank": {
            "B0": {"post_bos_positions": 1, "support_count": 1, "support_digest": "a" * 64},
            "B1": {"post_bos_positions": 1, "support_count": 1, "support_digest": "b" * 64},
        }},
    }
    contract_path = root / "synthetic-stage3-contract.json"
    contract_bytes = json.dumps(contract, sort_keys=True, indent=2).encode() + b"\n"
    contract_path.write_bytes(contract_bytes)
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    counter = {
        "schema": "token-reconstruction.trr-p09-shared-stage3-contract-countersignature.v1",
        "task_id": "TRR-P09",
        "status": provider.SIGNED_STAGE3_COUNTERSIGNATURE_STATUS,
        "contract_copy": {"sha256": contract_sha, "source_sha256": contract_sha},
        "counter_signature": {"exact_contract_hash_agreed": True},
    }
    counter_path = root / "synthetic-stage3-countersignature.json"
    counter_bytes = json.dumps(counter, sort_keys=True, indent=2).encode() + b"\n"
    counter_path.write_bytes(counter_bytes)
    counter_sha = hashlib.sha256(counter_bytes).hexdigest()
    for receipt in receipts.values():
        receipt["artifacts"]["contract"] = {"path": str(contract_path), "bytes": len(contract_bytes), "sha256": contract_sha, "status": "FROZEN"}
        receipt["contract_countersignature"] = {"path": str(counter_path), "bytes": len(counter_bytes), "sha256": counter_sha, "status": "COUNTERSIGNED"}
    return contract_sha, counter_sha


def _diagnostic(tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "diagnostic.json"
    path.write_text("{}\n")
    return {
        "schema": "token-reconstruction.trr-p09-fixed-diagnostic-binding.v1",
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": "d" * 64,
        "fit_record_count": 64,
        "full_bank_endpoint_steps": [0, 13000],
        "selection_isolated": True,
        "banks": {
            "B0": {"global_indices_sha256": "e" * 64},
            "B1": {"global_indices_sha256": "f" * 64},
        },
    }



def test_configuration_dry_run_validates_both_explicit_banks_without_model(tmp_path: Path) -> None:
    diagnostic = _diagnostic(tmp_path)
    receipts = {
        "current_directional": _binding(tmp_path, arm_name="current_directional", bank_role="B0", diagnostic=diagnostic),
        "expanded_directional": _binding(tmp_path, arm_name="expanded_directional", bank_role="B1", diagnostic=diagnostic),
    }
    contract_sha, counter_sha = _prepare_synthetic_bindings(receipts)
    qualification_sha = receipts["current_directional"]["resource_qualification"].setdefault("qualification_receipt", {})
    guard_sha = receipts["current_directional"]["resource_qualification"].setdefault("resource_guard_receipt", {})
    # Synthetic evidence files are created below so the provider still checks
    # bytes/hashes instead of accepting placeholders.
    for receipt in receipts.values():
        resource = receipt["resource_qualification"]
        qpath = tmp_path / f"{receipt['arm_name']}-qualification.json"
        gpath = tmp_path / f"{receipt['arm_name']}-resource-guard.json"
        qpath.write_text('{"status":"QUALIFICATION_PASS"}\n')
        gpath.write_text('{"status":"PASS"}\n')
        resource["qualification_receipt"] = {"path": str(qpath), "bytes": qpath.stat().st_size, "sha256": __import__('hashlib').sha256(qpath.read_bytes()).hexdigest(), "status": "QUALIFICATION_PASS"}
        resource["resource_guard_receipt"] = {"path": str(gpath), "bytes": gpath.stat().st_size, "sha256": __import__('hashlib').sha256(gpath.read_bytes()).hexdigest(), "status": "PASS"}
    qualification_sha = receipts["current_directional"]["resource_qualification"]["qualification_receipt"]["sha256"]
    guard_sha = receipts["current_directional"]["resource_qualification"]["resource_guard_receipt"]["sha256"]
    result = provider.configuration_dry_run(binding_receipts=receipts, diagnostic_binding=diagnostic, expected_contract_sha256=contract_sha, expected_countersignature_sha256=counter_sha, expected_qualification_sha256=qualification_sha, expected_resource_guard_sha256=guard_sha)
    assert result["status"] == "PASS_CONFIGURATION_DRY_RUN"
    assert result["schedule_steps_count"] == 13000
    assert set(result["arms"]) == {"current_directional", "expanded_directional"}
    assert result["arms"]["current_directional"]["bank_role"] == "B0"
    assert result["arms"]["expanded_directional"]["bank_role"] == "B1"
    assert result["model_allocated"] is False
    assert result["updates"] is False
    assert result["truth_opened"] is False



def test_configuration_dry_run_rejects_role_aliasing(tmp_path: Path) -> None:
    diagnostic = _diagnostic(tmp_path)
    current = _binding(tmp_path, arm_name="current_directional", bank_role="B1", diagnostic=diagnostic)
    expanded = _binding(tmp_path, arm_name="expanded_directional", bank_role="B1", diagnostic=diagnostic)
    receipts = {"current_directional": current, "expanded_directional": expanded}
    contract_sha, counter_sha = _prepare_synthetic_bindings(receipts)
    with pytest.raises(provider.ProductionProviderError, match="does not declare B0"):
        provider.configuration_dry_run(
            binding_receipts=receipts,
            diagnostic_binding=diagnostic,
            expected_contract_sha256=contract_sha,
            expected_countersignature_sha256=counter_sha,
        )


def test_configuration_dry_run_rejects_signed_schedule_mismatch(tmp_path: Path) -> None:
    diagnostic = _diagnostic(tmp_path)
    current = _binding(tmp_path, arm_name="current_directional", bank_role="B0", diagnostic=diagnostic)
    receipts = {"current_directional": current}
    contract_sha, counter_sha = _prepare_synthetic_bindings(receipts)
    resource = current["resource_qualification"]
    qpath = tmp_path / "qualification.json"
    gpath = tmp_path / "resource_guard.json"
    qpath.write_text('{"status":"QUALIFICATION_PASS"}\n')
    gpath.write_text('{"status":"PASS"}\n')
    import hashlib
    resource["qualification_receipt"] = {"path": str(qpath), "bytes": qpath.stat().st_size, "sha256": hashlib.sha256(qpath.read_bytes()).hexdigest()}
    resource["resource_guard_receipt"] = {"path": str(gpath), "bytes": gpath.stat().st_size, "sha256": hashlib.sha256(gpath.read_bytes()).hexdigest()}
    qsha = resource["qualification_receipt"]["sha256"]
    gsha = resource["resource_guard_receipt"]["sha256"]
    result = provider.configuration_dry_run(binding_receipts=receipts, diagnostic_binding=diagnostic, arm_name="current_directional", expected_contract_sha256=contract_sha, expected_countersignature_sha256=counter_sha, expected_qualification_sha256=qsha, expected_resource_guard_sha256=gsha)
    assert result["status"] == "PASS_CONFIGURATION_DRY_RUN"
    current["schedule"]["semantic_sha256"] = "c" * 64
    with pytest.raises(provider.ProductionProviderError, match="differs from signed B0 schedule"):
        provider.configuration_dry_run(binding_receipts=receipts, diagnostic_binding=diagnostic, arm_name="current_directional", expected_contract_sha256=contract_sha, expected_countersignature_sha256=counter_sha, expected_qualification_sha256=qsha, expected_resource_guard_sha256=gsha)


def test_configuration_dry_run_rejects_placeholder_resource_qualification(tmp_path: Path) -> None:
    diagnostic = _diagnostic(tmp_path)
    current = _binding(tmp_path, arm_name="current_directional", bank_role="B0", diagnostic=diagnostic)
    receipts = {"current_directional": current}
    contract_sha, counter_sha = _prepare_synthetic_bindings(receipts)
    resource = current["resource_qualification"]
    qpath = tmp_path / "qualification.json"
    gpath = tmp_path / "resource_guard.json"
    qpath.write_text('{"status":"QUALIFICATION_PASS"}\n')
    gpath.write_text('{"status":"PASS"}\n')
    import hashlib
    qsha = hashlib.sha256(qpath.read_bytes()).hexdigest()
    gsha = hashlib.sha256(gpath.read_bytes()).hexdigest()
    resource["qualification_receipt"] = {"path": str(qpath), "bytes": qpath.stat().st_size, "sha256": qsha}
    resource["resource_guard_receipt"] = {"path": str(gpath), "bytes": gpath.stat().st_size, "sha256": gsha}
    resource["gpu_peak_reserved_bytes"] = 0
    with pytest.raises(provider.ProductionProviderError, match="gpu_peak_reserved_bytes.*positive"):
        provider.configuration_dry_run(binding_receipts=receipts, diagnostic_binding=diagnostic, arm_name="current_directional", expected_contract_sha256=contract_sha, expected_countersignature_sha256=counter_sha, expected_qualification_sha256=qsha, expected_resource_guard_sha256=gsha)
    resource["gpu_peak_reserved_bytes"] = 7600078848
    resource["gpu_reserved_limit_bytes"] = 10**12
    resource["production_caps"]["gpu_reserved_limit_bytes"] = 10**12
    with pytest.raises(provider.ProductionProviderError, match="resource cap gpu_reserved_limit_bytes differs"):
        provider.configuration_dry_run(binding_receipts=receipts, diagnostic_binding=diagnostic, arm_name="current_directional", expected_contract_sha256=contract_sha, expected_countersignature_sha256=counter_sha, expected_qualification_sha256=qsha, expected_resource_guard_sha256=gsha)
