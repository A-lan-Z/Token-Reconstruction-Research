"""Synthetic fail-closed tests for the authorized TRR-0009 compatibility adapter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import trr0009_eval_gate as gate
from scripts import trr0009_eval_gate_compat as compat
from scripts import trr0009_eval_truth as truth


def _loader() -> dict[str, object]:
    return {
        "module": "token_reconstruction.trr0007_positionwise",
        "function": "load_positionwise_model_state",
        "interface": "trr0009.current_h.full_vocabulary.v1",
        "current_h_only": True,
        "full_vocabulary": True,
        "history_enabled": False,
        "a2_enabled": False,
        "kwargs": {
            "context_width": 128,
            "hidden_size": 2048,
            "method_id": "trr0007_residual_mlp512",
            "vocabulary_size": 128256,
        },
        "path_args": {},
        "tensor_args": {},
    }


def _row(path: Path, **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "continued_fixed_readout",
        "role": "continued_fixed_readout",
        "kind": "decoder",
        "state": {
            "path": str(path),
            "bytes": compat.FIXED_STATE_BYTES,
            "sha256": compat.FIXED_STATE_SHA256,
        },
        "loader": _loader(),
    }
    row.update(changes)
    return row


def _metadata(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": compat.FIXED_STATE_SCHEMA,
        "method_id": compat.FIXED_STATE_METHOD_ID,
        "inference_contract": compat.FIXED_INFERENCE_CONTRACT,
        "context_width": "128",
        "hidden_size": "2048",
        "vocabulary_size": "128256",
        "selected_step": "400",
    }
    value.update(changes)
    return value


@pytest.fixture
def fixed_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    state_path = tmp_path / "experiments/TRR-0009/training/run_v1/continued_fixed_readout/selected.safetensors"
    state_path.parent.mkdir(parents=True)
    state_path.write_bytes(b"synthetic state metadata fixture")
    monkeypatch.setattr(
        compat,
        "_record",
        lambda value, *, root, description: {
            "path": str(state_path.resolve()),
            "bytes": compat.FIXED_STATE_BYTES,
            "sha256": compat.FIXED_STATE_SHA256,
        },
    )
    monkeypatch.setattr(compat, "_state_metadata", lambda path: _metadata())
    return state_path


def test_exact_fixed_contract_accepts_omitted_redundant_flags(fixed_fixture: Path, tmp_path: Path) -> None:
    details = compat._validate_fixed_state(_row(fixed_fixture), root=tmp_path)
    assert details["inference_contract"] == compat.FIXED_INFERENCE_CONTRACT
    assert details["redundant_metadata_flags"] == {
        "current_H_only": "absent",
        "full_vocabulary_cross_entropy": "absent",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "token-reconstruction.other.v1"),
        ("method_id", "other_method"),
        ("inference_contract", "history_and_partial_vocabulary"),
        ("context_width", "127"),
        ("hidden_size", "1024"),
        ("vocabulary_size", "128255"),
        ("selected_step", "399"),
    ],
)
def test_metadata_identity_changes_fail_closed(
    fixed_fixture: Path,
    tmp_path: Path,
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compat, "_state_metadata", lambda path: _metadata(**{field: value}))
    with pytest.raises(compat.CompatibilityError, match="metadata"):
        compat._validate_fixed_state(_row(fixed_fixture), root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_H_only", "false"),
        ("full_vocabulary_cross_entropy", False),
        ("current_H_only", 1),
        ("full_vocabulary_cross_entropy", "yes"),
    ],
)
def test_present_redundant_flag_must_be_true(
    fixed_fixture: Path,
    tmp_path: Path,
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compat, "_state_metadata", lambda path: _metadata(**{field: value}))
    with pytest.raises(compat.CompatibilityError, match="contradicts"):
        compat._validate_fixed_state(_row(fixed_fixture), root=tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"id": "unchanged_anchor"},
        {"role": "unchanged_anchor"},
        {"loader": {"module": "wrong"}},
        {"kind": None},
        {"state": {"path": "/wrong/path", "bytes": compat.FIXED_STATE_BYTES, "sha256": compat.FIXED_STATE_SHA256}},
    ],
)
def test_method_loader_and_state_bindings_are_exact(
    fixed_fixture: Path,
    tmp_path: Path,
    change: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compat,
        "_record",
        lambda value, *, root, description: {
            "path": str(Path(value["path"]).resolve()),
            "bytes": value["bytes"],
            "sha256": value["sha256"],
        },
    )
    with pytest.raises(compat.CompatibilityError):
        compat._validate_fixed_state(_row(fixed_fixture, **change), root=tmp_path)


def test_state_hash_and_size_are_bound(fixed_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def wrong_record(value: object, *, root: Path, description: str) -> dict[str, object]:
        return {"path": str(fixed_fixture.resolve()), "bytes": 1, "sha256": "0" * 64}

    monkeypatch.setattr(compat, "_record", wrong_record)
    with pytest.raises(compat.CompatibilityError, match="bytes/hash"):
        compat._validate_fixed_state(_row(fixed_fixture), root=tmp_path)


def test_only_fixed_row_is_removed_before_original_semantics_check(
    fixed_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[object]] = []

    def original(*, registration: dict[str, object], root: Path) -> None:
        seen.append(registration["methods"])

    monkeypatch.setattr(compat, "_ORIGINAL_VALIDATE_STATE_SEMANTICS", original)
    registration = {
        "code_commit": compat.HISTORICAL_INFERENCE_COMMIT,
        "methods": [
            _row(fixed_fixture),
            {"id": "unchanged_anchor", "state": {"path": str(tmp_path / "anchor")}},
        ]
    }
    result = compat.validate_state_semantics_compat(registration=registration, root=tmp_path)
    assert result["status"] == compat.COMPAT_STATUS
    assert len(seen) == 1
    assert [row["id"] for row in seen[0]] == ["unchanged_anchor"]


def test_gate_patch_is_scoped_and_restored() -> None:
    original = gate._validate_state_semantics
    with compat._patched_gate_state_semantics():
        assert gate._validate_state_semantics is compat.validate_state_semantics_compat
    assert gate._validate_state_semantics is original


def test_registration_must_remain_historical(tmp_path: Path) -> None:
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps({"code_commit": "0" * 40}), encoding="utf-8")
    with pytest.raises(compat.CompatibilityError, match="historical inference commit"):
        compat._registration_context(registration_path, root=tmp_path)


def test_registered_fixed_state_metadata_smoke_is_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    registration = json.loads((root / "experiments/TRR-0009/evaluation/registration_v2.json").read_text())
    row = next(item for item in registration["methods"] if item["id"] == compat.FIXED_METHOD_ID)
    details = compat._validate_fixed_state(row, root=root)
    assert details["state"]["bytes"] == compat.FIXED_STATE_BYTES
    assert details["state"]["sha256"] == compat.FIXED_STATE_SHA256
    assert details["metadata_geometry"]["selected_step"] == "400"


def test_prepare_wrapper_rejects_changed_binding_before_truth_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_gate = truth.gate.validate_before_truth
    materialized = False

    def reject_changed_binding(**kwargs: object) -> dict[str, object]:
        raise compat.CompatibilityError("prediction binding changed")

    def forbidden_materialization(**kwargs: object) -> object:
        nonlocal materialized
        materialized = True
        raise AssertionError("truth materialization occurred before public validation")

    monkeypatch.setattr(compat, "validate_before_truth_compat", reject_changed_binding)
    monkeypatch.setattr(truth, "_materialize_truth", forbidden_materialization)
    with pytest.raises(compat.CompatibilityError, match="prediction binding changed"):
        compat.prepare_truth_compat(
            freeze_path=tmp_path / "freeze.json",
            registration_path=None,
            selection_path=tmp_path / "selection.json",
            truth_output=tmp_path / "truth.safetensors",
            truth_binding_path=tmp_path / "truth_binding.json",
            repository_root=tmp_path,
            execute=True,
        )
    assert materialized is False
    assert truth.gate.validate_before_truth is original_gate


def test_score_wrapper_rejects_changed_code_binding_before_truth_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_gate = truth.gate.validate_before_truth
    truth_read = False

    def reject_changed_code(**kwargs: object) -> dict[str, object]:
        raise compat.CompatibilityError("registered code binding changed")

    def forbidden_prediction_load(*args: object, **kwargs: object) -> object:
        nonlocal truth_read
        truth_read = True
        raise AssertionError("predictions/truth were read before public validation")

    monkeypatch.setattr(compat, "validate_before_truth_compat", reject_changed_code)
    monkeypatch.setattr(truth, "_load_predictions", forbidden_prediction_load)
    with pytest.raises(compat.CompatibilityError, match="registered code binding changed"):
        compat.score_truth_sidecar_compat(
            freeze_path=tmp_path / "freeze.json",
            truth_binding_path=tmp_path / "truth_binding.json",
            truth_sidecar_path=tmp_path / "truth.safetensors",
            frequency_reference_path=tmp_path / "frequency.json",
            output_path=tmp_path / "score.json",
            repository_root=tmp_path,
        )
    assert truth_read is False
    assert truth.gate.validate_before_truth is original_gate


def test_gate_patch_restores_after_exception() -> None:
    original = gate._validate_state_semantics
    with pytest.raises(RuntimeError):
        with compat._patched_gate_state_semantics():
            raise RuntimeError("synthetic gate failure")
    assert gate._validate_state_semantics is original


def test_truth_context_dispatches_require_current_head_keyword_without_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_gate = truth.gate.validate_before_truth
    seen: dict[str, object] = {}

    def reject_at_public_boundary(
        *, freeze_path: Path, repository_root: Path, require_current_head: bool = True
    ) -> dict[str, object]:
        seen.update(
            freeze_path=freeze_path,
            repository_root=repository_root,
            require_current_head=require_current_head,
        )
        raise compat.CompatibilityError("synthetic public binding failure")

    monkeypatch.setattr(compat, "validate_before_truth_compat", reject_at_public_boundary)
    with pytest.raises(compat.CompatibilityError, match="synthetic public binding failure"):
        with compat._patched_truth_gate():
            truth._load_public_context(
                freeze_path=tmp_path / "freeze.json", repository_root=tmp_path
            )
    assert seen["require_current_head"] is True
    assert truth.gate.validate_before_truth is original_gate


def _synthetic_receipt_bindings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    source_digests = {
        "compatibility adapter source": "a" * 64,
        "wrapped public gate source": "b" * 64,
        "wrapped truth boundary source": "c" * 64,
    }

    def source_record(path: Path, *, root: Path, description: str) -> dict[str, object]:
        return {"path": str(path.resolve()), "bytes": 1, "sha256": source_digests[description]}

    loader_record = {"path": str(tmp_path / "loader.json"), "bytes": 2, "sha256": "d" * 64}
    monkeypatch.setattr(compat, "_compatibility_source_record", source_record)
    monkeypatch.setattr(compat, "_loader_equivalence_record", lambda registration, *, root: dict(loader_record))
    registration_binding = {"path": str(tmp_path / "registration.json"), "bytes": 3, "sha256": "e" * 64}
    freeze = {"registration": registration_binding, "maintenance_validator_commit": "f" * 40}
    receipt: dict[str, object] = {
        "schema": compat.COMPAT_SCHEMA,
        "status": compat.COMPAT_STATUS,
        "historical_inference_commit": compat.HISTORICAL_INFERENCE_COMMIT,
        "maintenance_validator_commit": "f" * 40,
        "registration": registration_binding,
        "adapter_source": source_record(Path(compat.__file__), root=tmp_path, description="compatibility adapter source"),
        "wrapped_gate_source": source_record(Path(gate.__file__), root=tmp_path, description="wrapped public gate source"),
        "wrapped_truth_source": source_record(Path(truth.__file__), root=tmp_path, description="wrapped truth boundary source"),
        "loader_equivalence_receipt": loader_record,
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }
    registration = {"code_commit": compat.HISTORICAL_INFERENCE_COMMIT}
    return freeze, receipt, registration


def test_revalidation_pins_current_sources_and_loader_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, receipt, registration = _synthetic_receipt_bindings(monkeypatch, tmp_path)
    result = compat._validate_compatibility_receipt(
        freeze=freeze, receipt=receipt, registration=registration, root=tmp_path
    )
    assert result["loader_equivalence_receipt"]["sha256"] == "d" * 64


def test_revalidation_rejects_reloaded_registration_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, receipt, registration = _synthetic_receipt_bindings(monkeypatch, tmp_path)
    registration["code_commit"] = "0" * 40
    with pytest.raises(compat.CompatibilityError, match="reloaded registration"):
        compat._validate_compatibility_receipt(
            freeze=freeze, receipt=receipt, registration=registration, root=tmp_path
        )


@pytest.mark.parametrize(
    "field",
    ["adapter_source", "wrapped_gate_source", "wrapped_truth_source", "loader_equivalence_receipt"],
)
def test_revalidation_rejects_changed_source_or_loader_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    freeze, receipt, registration = _synthetic_receipt_bindings(monkeypatch, tmp_path)
    changed = dict(receipt[field])
    changed["bytes"] = int(changed["bytes"]) + 1
    receipt[field] = changed
    with pytest.raises(compat.CompatibilityError):
        compat._validate_compatibility_receipt(
            freeze=freeze, receipt=receipt, registration=registration, root=tmp_path
        )


@pytest.mark.parametrize(
    "field", ["truth_opened", "source_text_loaded", "target_labels_loaded", "candidate_arrays_persisted"]
)
def test_revalidation_requires_all_compatibility_flags_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    freeze, receipt, registration = _synthetic_receipt_bindings(monkeypatch, tmp_path)
    receipt[field] = True
    with pytest.raises(compat.CompatibilityError, match="flag is not false"):
        compat._validate_compatibility_receipt(
            freeze=freeze, receipt=receipt, registration=registration, root=tmp_path
        )


def test_revalidation_reruns_pinned_loader_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, receipt, registration = _synthetic_receipt_bindings(monkeypatch, tmp_path)
    calls: list[dict[str, object]] = []
    original = compat._loader_equivalence_record

    def counted(registration_value: dict[str, object], *, root: Path) -> dict[str, object]:
        calls.append(registration_value)
        return original(registration_value, root=root)

    monkeypatch.setattr(compat, "_loader_equivalence_record", counted)
    compat._validate_compatibility_receipt(
        freeze=freeze, receipt=receipt, registration=registration, root=tmp_path
    )
    assert calls == [registration]
