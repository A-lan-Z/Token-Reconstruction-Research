from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import trr0005_export_prediction_contract as exporter
from token_reconstruction import trr0005_contract as contract


def _panel() -> dict:
    cells = {}
    for cell_id in contract.EXPECTED_CELL_IDS:
        style, condition = cell_id.split("__", 1)
        cells[cell_id] = {
            "cell_id": cell_id,
            "style": style,
            "condition": condition,
            "records": [
                {
                    "record_id": f"{style}-public-{index}",
                    "public_record_sha256": f"{index + 1:064x}",
                }
                for index in range(contract.RECORDS_PER_DOMAIN)
            ],
        }
    return {
        "schema": contract.PANEL_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_FRESH_CONFIRMATION_PANEL",
        "sequence_tokens": contract.SEQUENCE_TOKENS,
        "records_per_domain": contract.RECORDS_PER_DOMAIN,
        "cells": cells,
    }


def _registration() -> dict:
    return contract.build_registration(
        status="FROZEN_METHOD_REGISTRATION",
        code_commit="a" * 40,
        state_bindings={
            method_id: {"status": "FROZEN"}
            for method_id in contract.METHOD_IDS
        },
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _descriptor(
    *,
    repository_root: Path,
    artifact_path: Path,
    cell_id: str,
    method_id: str,
) -> dict:
    policy = contract.CANDIDATE_POLICIES[method_id]
    artifact_relative = artifact_path.resolve().relative_to(
        repository_root.resolve()
    ).as_posix()
    artifact_bytes = artifact_path.read_bytes()
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest()
    return {
        "schema": exporter.LEGACY_RECEIPT_SCHEMA,
        "timing_schema": exporter.LEGACY_RECEIPT_SCHEMA,
        "task_id": contract.TASK_ID,
        "cell_id": cell_id,
        "method_id": method_id,
        "canonical_method_id": method_id,
        "shape": [contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS],
        "observation_shape": [
            contract.RECORDS_PER_DOMAIN,
            contract.SEQUENCE_TOKENS,
            2048,
        ],
        "candidate_policy": policy,
        "candidate_arrays_present": False,
        "candidate_output": (
            "omitted_after_decision" if policy == "output_only" else None
        ),
        "warmup_runs_per_record": 1,
        "measured_runs_per_record": 1,
        "warmup_output_exact_match_measured": True,
        "measured_output_selected": True,
        "records": contract.RECORDS_PER_DOMAIN,
        "warmup_adapter_calls": contract.RECORDS_PER_DOMAIN,
        "measured_adapter_calls": contract.RECORDS_PER_DOMAIN,
        "adapter_calls_total": 2 * contract.RECORDS_PER_DOMAIN,
        "adapter_call_scope": "synthetic warmup plus measured calls",
        "warmup_seconds_sum": 1.25,
        "measured_seconds_sum": 2.5,
        "timed_interval_total_seconds": 3.75,
        "per_record_measured_seconds": [
            2.5 / contract.RECORDS_PER_DOMAIN
        ] * contract.RECORDS_PER_DOMAIN,
        "runtime_load_seconds": 0.125,
        "peak_memory": {
            "cuda_peak_allocated_bytes": 10,
            "cuda_peak_reserved_bytes": 20,
            "process_max_rss_bytes": 30,
        },
        "method_specific": {
            "calls": 2 * contract.RECORDS_PER_DOMAIN,
            "public_prefix_calls": 0,
            "candidate_simulations": 0,
            "numeric_marker": 12.5,
        },
        "active_tokens": contract.RECORDS_PER_DOMAIN * contract.SEQUENCE_TOKENS,
        "scored_tokens": contract.RECORDS_PER_DOMAIN * (contract.SEQUENCE_TOKENS - 1),
        "steady_interval": "synthetic interval",
        "synchronization": "host-only synchronization callback",
        "cold_costs_separate": True,
        "panel_sha256": "b" * 64,
        "selection_plan_sha256": "c" * 64,
        "observation_sha256": "d" * 64,
        "prediction_sha256": "e" * 64,
        "prediction_artifact": {
            "path": artifact_relative,
            "bytes": len(artifact_bytes),
            "sha256": artifact_sha,
        },
        "artifact_relative_to_root": artifact_relative,
    }


def _synthetic_source(tmp_path: Path) -> tuple[Path, dict, dict, dict]:
    source_root = tmp_path / "predictions_v1"
    entries: dict[str, dict] = {}
    receipt_bytes: dict[Path, bytes] = {}
    for cell_id in contract.EXPECTED_CELL_IDS:
        style, condition = cell_id.split("__", 1)
        for method_id in contract.METHOD_IDS:
            artifact_path = (
                source_root / style / condition / f"{method_id}.safetensors"
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(f"synthetic:{cell_id}:{method_id}\n".encode())
            descriptor = _descriptor(
                repository_root=tmp_path,
                artifact_path=artifact_path,
                cell_id=cell_id,
                method_id=method_id,
            )
            key = f"{cell_id}::{method_id}"
            entries[key] = descriptor
            receipt_path = artifact_path.with_name(f"{method_id}.run.json")
            _write_json(receipt_path, descriptor)
            receipt_bytes[receipt_path] = receipt_path.read_bytes()
    prediction_manifest = {
        "schema": exporter.PREDICTION_MANIFEST_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "PUBLIC_PREDICTIONS_COMPLETE_NO_TRUTH",
        "panel": {"path": "panel.json", "bytes": 1, "sha256": "f" * 64},
        "selection_plan": {
            "path": "selection_plan.json",
            "bytes": 1,
            "sha256": "0" * 64,
        },
        "registration": {
            "path": "registration.json",
            "bytes": 1,
            "sha256": "1" * 64,
        },
        "method_ids": list(contract.METHOD_IDS),
        "cells": list(contract.EXPECTED_CELL_IDS),
        "predictions": copy.deepcopy(entries),
        "truth_opened": False,
    }
    timing_manifest = {
        "schema": exporter.TIMING_MANIFEST_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "PUBLIC_TIMINGS_COMPLETE_NO_TRUTH",
        "panel_sha256": "f" * 64,
        "selection_plan_sha256": "0" * 64,
        "method_ids": list(contract.METHOD_IDS),
        "cells": list(contract.EXPECTED_CELL_IDS),
        "timings": copy.deepcopy(entries),
        "truth_opened": False,
    }
    _write_json(source_root / "predictions.json", prediction_manifest)
    _write_json(source_root / "timings.json", timing_manifest)
    _write_json(
        source_root / "run_evidence.json",
        {
            "schema": exporter.RUN_EVIDENCE_SCHEMA,
            "task_id": contract.TASK_ID,
            "status": "PUBLIC_PREDICTION_MATRIX_COMPLETE_NO_TRUTH",
            "git_commit": "a" * 40,
            "prediction_count": len(entries),
            "timing_count": len(entries),
            "truth_opened": False,
            "source_text_loaded": False,
            "target_labels_loaded": False,
            "future_activation_reads": False,
            "historical_numeric_marker": 123.456,
        },
    )
    return source_root, entries, prediction_manifest, receipt_bytes


def _discover_receipt_map(root: Path) -> dict[tuple[str, str], dict]:
    result = {}
    for path in sorted(root.rglob("*.run.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        result[(value["cell_id"], value["method_id"])] = value
    return result


def test_legacy_rows_fail_full_gate_and_exported_rows_pass(tmp_path: Path) -> None:
    source_root, source_entries, source_manifest, raw_receipts = _synthetic_source(
        tmp_path
    )
    output_root = tmp_path / "predictions_v2_contract_export"
    panel = _panel()
    registration = _registration()
    bad = {
        tuple(key.split("::", 1)): value for key, value in source_entries.items()
    }
    with pytest.raises(contract.ContractError, match="prediction identity changed"):
        contract.validate_complete_public_matrix(
            panel,
            registration,
            bad,
            timing_descriptors=bad,
        )

    run_evidence_before = (source_root / "run_evidence.json").read_bytes()
    result = exporter.export_prediction_contract(
        source_root,
        output_root,
        repository_root=tmp_path,
        execution_commit="a" * 40,
        exporter_path=Path(exporter.__file__),
    )
    assert result["status"] == "METADATA_ONLY_CONTRACT_EXPORT_NO_TRUTH"
    assert result["prediction_receipts"] == 32
    assert result["prediction_artifacts"] == 32

    corrected = _discover_receipt_map(output_root)
    assert set(corrected) == set(bad)
    for key, before in bad.items():
        after = corrected[key]
        assert after["schema"] == contract.PREDICTION_SCHEMA
        assert after["timing_schema"] == before["timing_schema"]
        exporter._assert_same_except(
            before,
            after,
            allowed_paths=exporter.ALLOWED_DESCRIPTOR_CHANGES,
        )
        destination_receipt = (
            output_root
            / key[0].split("__", 1)[0]
            / key[0].split("__", 1)[1]
            / f"{key[1]}.run.json"
        )
        destination_artifact = destination_receipt.with_suffix("").with_suffix(
            ".safetensors"
        )
        assert after["prediction_artifact"]["path"] == destination_artifact.relative_to(
            tmp_path
        ).as_posix()
        assert after["artifact_relative_to_root"] == after["prediction_artifact"]["path"]
        source_artifact = (
            source_root
            / key[0].split("__", 1)[0]
            / key[0].split("__", 1)[1]
            / f"{key[1]}.safetensors"
        )
        assert source_artifact.read_bytes() == destination_artifact.read_bytes()

    corrected_for_gate = {
        key: value for key, value in corrected.items()
    }
    gate = contract.validate_complete_public_matrix(
        panel,
        registration,
        corrected_for_gate,
        timing_descriptors=corrected_for_gate,
    )
    assert gate["prediction_artifacts"] == 32
    assert gate["timing_receipts"] == 32
    assert not (output_root / "run_evidence.json").exists()
    assert (source_root / "run_evidence.json").read_bytes() == run_evidence_before
    assert source_manifest["schema"] == exporter.PREDICTION_MANIFEST_SCHEMA
    assert json.loads((output_root / "predictions.json").read_text())["schema"] == exporter.PREDICTION_MANIFEST_SCHEMA
    assert json.loads((output_root / "timings.json").read_text())["schema"] == exporter.TIMING_MANIFEST_SCHEMA
    provenance = json.loads((output_root / "export_provenance.json").read_text())
    assert provenance["execution_commit"] == "a" * 40
    assert provenance["source"]["run_evidence_rewritten"] is False
    assert provenance["preserved_execution_record"]["raw_source_left_untouched"] is True
    assert provenance["allowed_descriptor_changes"] == sorted(
        exporter.ALLOWED_DESCRIPTOR_CHANGES
    )
    assert provenance["truth_opened"] is False
    assert provenance["binary_copy_policy"]["destination_bytes_and_sha256_verified"] is True
    assert len(provenance["destination"]["prediction_artifacts"]) == 32
    for path, original_bytes in raw_receipts.items():
        assert path.read_bytes() == original_bytes


def test_recursive_export_allowlist_rejects_timing_change() -> None:
    before = {"schema": "old", "nested": {"measured_seconds_sum": 1.25}}
    after = {"schema": "new", "nested": {"measured_seconds_sum": 1.5}}
    with pytest.raises(exporter.ExportError, match="unapproved value change"):
        exporter._assert_same_except(
            before,
            after,
            allowed_paths=frozenset({"schema"}),
        )
