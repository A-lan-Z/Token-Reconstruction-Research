"""Synthetic fail-closed tests for the TRR-0010 final adapters."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from scripts import trr0010_eval_capture as capture
from scripts import trr0010_eval_gate as gate
from scripts import trr0010_eval_register as register
from scripts import trr0010_prepare_truth as truth
from scripts import trr0010_select_public as selector


def _record(path: Path, root: Path) -> dict:
    return gate.file_record(path, root=root)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _design(tmp_path: Path, *, explicit: bool = True) -> Path:
    methods = {}
    for method_id in gate.METHOD_ORDER:
        state = tmp_path / "assets" / f"{method_id}.state"
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_bytes(method_id.encode())
        state_record = _record(state, tmp_path)
        required = gate.METHOD_RESOURCE_REQUIREMENTS[method_id]
        resources = {}
        for name in required:
            resource = tmp_path / "assets" / f"{method_id}.{name}"
            resource.write_bytes(f"{method_id}:{name}".encode())
            resources[name] = _record(resource, tmp_path)
        methods[method_id] = {
            "id": method_id,
            "role": gate.METHOD_ROLES[method_id],
            "frozen": True,
            "state_frozen": True,
            "state": state_record,
            "resources": resources,
            "loader": (
                {"interface": gate.A1_A2_LOADER_INTERFACE, "candidate_k": 256}
                if method_id == gate.A1_A2_METHOD_ID
                else {
                    "interface": gate.LOADER_INTERFACE,
                    "current_h_only": True,
                    "full_vocabulary": True,
                    "history_enabled": False,
                    "a2_enabled": False,
                }
            ),
        }
    code = tmp_path / "assets" / "selector.py"
    code.write_text("# selector\n", encoding="utf-8")
    final_b1 = tmp_path / "assets" / "final_b1.json"
    final_b1.write_text("{\"schema\":\"synthetic-exclusions\"}\n", encoding="utf-8")
    opaque = tmp_path / "assets" / "opaque.json"
    opaque.write_text("{\"schema\":\"synthetic-opaque\"}\n", encoding="utf-8")
    final = {
        "records_by_domain": dict(gate.RECORDS_BY_DOMAIN),
        "target_conditions": list(gate.TARGET_ORDER),
        "paired_sources_across_targets": True,
        "capture_geometry": {
            "batch_records": 8,
            "sequence_tokens": 192,
            "stored_sequence_tokens": 128,
            "retain_first_128": True,
        },
        "exclusion_bindings": {
            "final_b1": _record(final_b1, tmp_path),
            "approved_opaque_ledgers": [_record(opaque, tmp_path)],
        },
    }
    payload = {
        "schema": "token-reconstruction.trr0010-final-evaluation-design.v1",
        "task_id": "TRR-0010",
        "status": "FROZEN_TRR0010_FINAL_EVALUATION_DESIGN",
        "code_commit": "a" * 40,
        "contenders_frozen": explicit,
        "method_order": list(gate.METHOD_ORDER),
        "methods": methods,
        "decision_rules": {"status": "FROZEN", "rules": {"primary": {"margin": 0.05}}},
        "code_bindings": {"selector": _record(code, tmp_path)},
        "final_evaluation": final,
        "truth_opened": False,
        "source_text_written": False,
        "target_labels_loaded": False,
    }
    return _write(tmp_path / "design.json", payload)


def _row(style: str, index: int) -> dict:
    return {
        "record_id": f"{style}-{index}",
        "public_record_sha256": hashlib.sha256(f"source-{style}-{index}".encode()).hexdigest(),
        "dataset_key": style,
        "dataset_id": "synthetic",
        "split": "test",
        "revision": "r1",
        "row_index": index,
        "source_index": index,
        "full_token_count": 128,
        "post_bos_token_count": 127,
        "valid_tokens": 128,
        "final_sequence_sha256": hashlib.sha256(f"sequence-{style}-{index}".encode()).hexdigest(),
    }


def _selection(tmp_path: Path, count: int = 128) -> Path:
    rows = {style: [_row(style, i) for i in range(count)] for style in ("pile", "finance")}
    payload = {
        "schema": selector.SELECTION_SCHEMA,
        "task_id": "TRR-0010",
        "status": selector.SELECTION_STATUS,
        "records_by_domain": {"pile": count, "finance": count},
        "target_conditions": list(selector.TARGET_CONDITIONS),
        "paired_conditions": True,
        "selection_rule": {
            "record_ids_sha256": {style: selector._json_digest([r["record_id"] for r in rows[style]]) for style in rows},
            "records": rows,
        },
        "truth_opened": False,
        "truth_created": False,
        "source_text_or_target_labels": False,
    }
    return _write(tmp_path / "selection.json", payload)


def test_design_requires_explicit_six_contender_asset_freeze(tmp_path: Path) -> None:
    design = _design(tmp_path)
    record, payload = selector._validate_design(design, root=tmp_path)
    assert record["sha256"] == gate.sha256_file(design)
    assert set(payload["methods"]) == set(gate.METHOD_ORDER)

    draft = _design(tmp_path / "draft", explicit=False)
    with pytest.raises(selector.SelectionError, match="explicit all-contender"):
        selector._validate_design(draft, root=tmp_path / "draft")

    state_draft = _design(tmp_path / "state-draft")
    state_payload = json.loads(state_draft.read_text())
    first_method = gate.METHOD_ORDER[0]
    state_payload["methods"][first_method].pop("state_frozen")
    state_draft.write_text(json.dumps(state_payload), encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="contender/state freeze"):
        selector._validate_design(state_draft, root=tmp_path / "state-draft")


def test_trr10_selection_loader_rejects_wrong_schema_and_checks_order(tmp_path: Path) -> None:
    path = _selection(tmp_path, count=2)
    selection, record, rows, counts = selector.load_selection(path, repository_root=tmp_path, expected_counts={"pile": 2, "finance": 2})
    assert selection["schema"] == selector.SELECTION_SCHEMA
    assert record["bytes"] == path.stat().st_size
    assert counts == {"pile": 2, "finance": 2}
    assert [row["record_id"] for row in rows["pile"]] == ["pile-0", "pile-1"]

    bad = json.loads(path.read_text())
    bad["selection_rule"]["record_ids_sha256"]["pile"] = "0" * 64
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(selector.SelectionError, match="record-order digest"):
        selector.load_selection(path, repository_root=tmp_path, expected_counts={"pile": 2, "finance": 2})


def _selection_with_native_exclusion_union(tmp_path: Path) -> tuple[Path, dict, dict]:
    selection_path = _selection(tmp_path)
    final_b1 = tmp_path / "assets" / "final_b1.json"
    final_b1.parent.mkdir(parents=True, exist_ok=True)
    final_b1.write_text("{\"schema\":\"synthetic-final-b1\"}\n", encoding="utf-8")
    final_record = _record(final_b1, tmp_path)
    union_path = tmp_path / "assets" / "source_exclusions.json"
    union_payload = {
        "schema": selector.EXCLUSION_SCHEMA,
        "task_id": selector.TASK_ID,
        "status": selector.EXCLUSION_STATUS,
        "identity_only": True,
        "sources": [{**final_record, "available": True, "new_identity_count": 4}],
        "source_text_or_token_ids_written": False,
        "private_or_truth_payload_read": False,
        "truth_opened": False,
        "truth_created": False,
    }
    _write(union_path, union_payload)
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_payload["selection_exclusions"] = _record(union_path, tmp_path)
    selection_path.write_text(json.dumps(selection_payload, sort_keys=True) + "\n", encoding="utf-8")
    return selection_path, final_record, union_payload


def test_register_accepts_native_union_and_preserves_frozen_b1_provenance(tmp_path: Path) -> None:
    selection_path, final_record, union_payload = _selection_with_native_exclusion_union(tmp_path)
    _selection_value, _selection_record, metadata = register._load_selection(
        selection_path, root=tmp_path, final_b1=final_record
    )
    assert metadata["selection_exclusion_schema"] == selector.EXCLUSION_SCHEMA
    assert metadata["selection_exclusion_status"] == selector.EXCLUSION_STATUS
    assert metadata["selection_exclusion_sources"] == union_payload["sources"]
    assert metadata["selection_applied_final_b1"] == union_payload["sources"]


@pytest.mark.parametrize("tamper", ["missing", "changed"])
def test_register_rejects_native_union_without_exact_frozen_b1_source(tmp_path: Path, tamper: str) -> None:
    selection_path, final_record, union_payload = _selection_with_native_exclusion_union(tmp_path)
    if tamper == "missing":
        union_payload["sources"] = []
    else:
        union_payload["sources"] = [{**final_record, "available": True, "sha256": "0" * 64}]
    union_path = tmp_path / "assets" / "source_exclusions.json"
    _write(union_path, union_payload)
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_payload["selection_exclusions"] = _record(union_path, tmp_path)
    selection_path.write_text(json.dumps(selection_payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(register.RegisterError, match="does not apply the frozen final B1 ledger"):
        register._load_selection(selection_path, root=tmp_path, final_b1=final_record)


def _producer_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    selection_path = _selection(tmp_path, count=128)
    bridge = tmp_path / "producer" / "selection.json"
    bridge_record = capture.write_producer_selection_bridge(selection_path=selection_path, output_path=bridge, repository_root=tmp_path)
    producer = tmp_path / "producer"
    producer.mkdir(exist_ok=True)
    selection, _, _, _ = selector.load_selection(selection_path, repository_root=tmp_path)
    digests = selection["selection_rule"]["record_ids_sha256"]
    cells = []
    for cell_id in gate.CELL_ORDER:
        path = producer / f"{cell_id}.safetensors"
        path.write_bytes(cell_id.encode())
        observation_descriptor = _record(path, tmp_path)
        observation_descriptor.update({"shape": [128, 128, 2048], "stored_sequence_tokens": 128, "capture_sequence_tokens": 192, "capture_batch_records": 8})
        cells.append({
            "cell_id": cell_id,
            "records": 128,
            "record_ids_sha256": digests[cell_id.split("__", 1)[0]],
            "observation": observation_descriptor,
        })
    obs_payload = {
        "schema": "token-reconstruction.trr0009-public-observation-manifest.v1",
        "task_id": "TRR-0009",
        "status": "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH",
        "records_by_domain": dict(gate.RECORDS_BY_DOMAIN),
        "cell_order": list(gate.CELL_ORDER),
        "record_ids_sha256": digests,
        "cells": cells,
        "selection_plan": bridge_record,
        "truth_opened": False,
        "source_text_written": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
    }
    obs_path = _write(producer / "observations.json", obs_payload)
    obs_record = _record(obs_path, tmp_path)
    panel_payload = {
        "schema": "token-reconstruction.trr0009-public-source-panel.v1",
        "task_id": "TRR-0009",
        "status": "FROZEN_SOURCE_PANEL_NO_TRUTH",
        "records_by_domain": dict(gate.RECORDS_BY_DOMAIN),
        "cell_order": list(gate.CELL_ORDER),
        "record_ids_sha256": digests,
        "selection_plan": bridge_record,
        "observation_manifest": obs_record,
        "same_sources_across_targets": True,
        "public_material_only": True,
        "truth_opened": False,
    }
    panel_path = _write(producer / "panel.json", panel_payload)
    panel_record = _record(panel_path, tmp_path)
    capture_payload = {
        "schema": "token-reconstruction.trr0009-public-capture.v1",
        "task_id": "TRR-0009",
        "status": "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH",
        "records_by_domain": dict(gate.RECORDS_BY_DOMAIN),
        "selection_plan": bridge_record,
        "observations": obs_record,
        "panel": panel_record,
        "geometry": {"capture_batch_records": 8, "capture_sequence_tokens": 192, "stored_sequence_tokens": 128},
        "execution": {"producer_semantics": "public full forward B8x192; retain first 128 positions"},
        "conditions": {
            condition: {
                "cells": {
                    cell_id: {
                        "full_forward_retained_only_first_128": True,
                        "capture_batch_records": 8,
                        "capture_sequence_tokens": 192,
                        "stored_sequence_tokens": 128,
                        "observation": next(row["observation"] for row in cells if row["cell_id"] == cell_id),
                    }
                    for cell_id in gate.CELL_ORDER
                    if cell_id.endswith(f"__{condition}")
                }
            }
            for condition in gate.TARGET_ORDER
        },
        "truth_opened": False,
    }
    _write(producer / "capture.json", capture_payload)
    return selection_path, producer, bridge


def test_capture_repackages_trr9_metadata_under_trr10_schema(tmp_path: Path) -> None:
    selection_path, producer, bridge = _producer_fixture(tmp_path)
    result = capture.repackage_trr0009_capture(
        selection_path=selection_path,
        producer_root=producer,
        output_root=tmp_path / "experiments" / "TRR-0010" / "evaluation" / "capture",
        repository_root=tmp_path,
        producer_selection_path=bridge,
    )
    assert result["status"] == capture.CAPTURE_STATUS
    payload = json.loads(Path(result["capture"]["path"]).read_text())
    assert payload["task_id"] == "TRR-0010"
    assert payload["geometry"]["capture_batch_records"] == 8
    assert payload["truth_opened"] is False


def test_capture_rejects_changed_producer_observation_payload(tmp_path: Path) -> None:
    selection_path, producer, bridge = _producer_fixture(tmp_path)
    changed = producer / "finance__public_base.safetensors"
    changed.write_bytes(changed.read_bytes() + b"changed")
    with pytest.raises(capture.CaptureAdapterError, match="descriptor changed"):
        capture.repackage_trr0009_capture(
            selection_path=selection_path,
            producer_root=producer,
            output_root=tmp_path / "experiments" / "TRR-0010" / "evaluation" / "capture",
            repository_root=tmp_path,
            producer_selection_path=bridge,
        )


def test_registration_rejects_runtime_state_substitution(tmp_path: Path) -> None:
    design_path = _design(tmp_path)
    design = json.loads(design_path.read_text())
    actual = []
    for method_id in gate.METHOD_ORDER:
        row = design["methods"][method_id]
        actual.append({
            "id": method_id,
            "role": row["role"],
            "state": row["state"],
            "resources": row["resources"],
            "loader": row["loader"],
        })
    register._verify_method_rows_match_design(actual, design=design, root=tmp_path)
    replacement = tmp_path / "assets" / "substituted.state"
    replacement.write_bytes(b"substitution")
    actual[0]["state"] = _record(replacement, tmp_path)
    with pytest.raises(register.RegisterError, match="runtime/frozen state"):
        register._verify_method_rows_match_design(actual, design=design, root=tmp_path)


def test_truth_curator_calls_gate_before_materializer_and_validates_after(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    freeze = {
        "truth_opened": False,
        "registration": {"path": "registration", "bytes": 1, "sha256": "a" * 64},
        "run_manifest": {"path": "run", "bytes": 1, "sha256": "b" * 64},
        "contract_binding": {"path": "contract", "bytes": 1, "sha256": "c" * 64},
        "input_bindings": {},
        "observation_bindings": {cell: {"record_ids_sha256": "d" * 64} for cell in gate.CELL_ORDER},
    }
    monkeypatch.setattr(truth.gate, "validate_before_truth", lambda **kwargs: calls.append("gate") or freeze)

    def materializer(_freeze: Mapping[str, object], path: Path) -> None:
        calls.append("materializer")
        path.write_bytes(b"sealed-sidecar")

    def write_descriptor(path: Path, payload: Mapping[str, object], *, root: Path, description: str) -> dict:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return _record(path, root)

    monkeypatch.setattr(truth.register, "_write_create_only", write_descriptor)
    monkeypatch.setattr(truth.register, "validate_truth_descriptor", lambda *args, **kwargs: calls.append("validate") or {"truth_payload": kwargs.get("freeze", {}).get("truth_payload", {})})
    result = truth.prepare_truth_after_freeze(
        freeze_path=tmp_path / "freeze.json",
        truth_sidecar_path=tmp_path / "sealed.safetensors",
        descriptor_path=tmp_path / "experiments" / "TRR-0010" / "truth_descriptor.json",
        repository_root=tmp_path,
        materializer=materializer,
    )
    assert calls == ["gate", "materializer", "validate"]
    assert result["truth_opened"] is False


def test_truth_curator_never_materializes_after_failed_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called = False
    def fail_gate(**_kwargs):
        raise gate.GateError("incomplete public matrix")
    monkeypatch.setattr(truth.gate, "validate_before_truth", fail_gate)
    def materializer(_freeze: Mapping[str, object], _path: Path) -> None:
        nonlocal called
        called = True
    with pytest.raises(truth.TruthPreparationError, match="public freeze"):
        truth.prepare_truth_after_freeze(
            freeze_path=tmp_path / "freeze.json",
            truth_sidecar_path=tmp_path / "sealed.safetensors",
            descriptor_path=tmp_path / "experiments" / "TRR-0010" / "truth_descriptor.json",
            repository_root=tmp_path,
            materializer=materializer,
        )
    assert called is False
    assert not (tmp_path / "sealed.safetensors").exists()
