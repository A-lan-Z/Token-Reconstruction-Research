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
        "gpu_peak_reserved_bytes": 1,
        "gpu_reserved_limit_bytes": 2,
        "host_peak_rss_bytes": 1,
        "host_rss_limit_bytes": 2,
        "host_available_bytes": 2,
        "host_available_floor_bytes": 1,
        "wall_seconds": 1.0,
        "wall_limit_seconds": 10.0,
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
    result = provider.configuration_dry_run(binding_receipts=receipts, diagnostic_binding=diagnostic)
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
    with pytest.raises(provider.ProductionProviderError, match="does not declare B0"):
        provider.configuration_dry_run(
            binding_receipts={"current_directional": current, "expanded_directional": expanded},
            diagnostic_binding=diagnostic,
        )
