"""Pure-Python tests for the bounded P11 capture adapter."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.trr_p11 import public_capture as capture


ROOT = Path(__file__).resolve().parents[1]


def _subset() -> dict[str, object]:
    return {
        "records_per_domain": capture.A1_A2_RECORDS_PER_DOMAIN,
        "first_records_per_domain": True,
        "performance_based_drop": False,
        "domains": list(capture.DOMAIN_ORDER),
        "token_positions_per_cell": capture.A1_A2_RECORDS_PER_DOMAIN * capture.SCORED_POST_BOS_TOKENS,
    }


def test_full_and_compact_geometry_are_frozen_without_torch() -> None:
    assert capture.validate_full_capture_geometry(
        "pile__public_base", (256, 192, 2048)
    ) == (256, 192, 2048)
    compact = capture.validate_observation_geometry(
        "finance__public_lora_2601",
        activation_shape=(256, 128, 2048),
        activation_dtype="torch.bfloat16",
        attention_mask_shape=(256, 128),
        attention_mask_dtype="torch.uint8",
        position_ids_shape=(256, 128),
        position_ids_dtype="torch.int64",
    )
    assert compact["stored_sequence_tokens"] == 128
    assert compact["scored_post_bos_tokens"] == 127


def test_geometry_rejects_wrong_padding_and_mask() -> None:
    with pytest.raises(capture.CaptureAdapterError, match="full capture geometry"):
        capture.validate_full_capture_geometry("pile__public_base", (256, 128, 2048))
    with pytest.raises(capture.CaptureAdapterError, match="attention mask"):
        capture.validate_observation_geometry(
            "pile__public_base",
            activation_shape=(256, 128, 2048),
            activation_dtype="bfloat16",
            attention_mask_shape=(256, 128),
            attention_mask_dtype="uint8",
            position_ids_shape=(256, 128),
            position_ids_dtype="int64",
            attention_mask_all_one=False,
        )


def test_capture_requires_explicit_execute_before_heavy_imports(tmp_path: Path) -> None:
    with pytest.raises(capture.CaptureAdapterError, match="explicit execute"):
        capture.capture_public(
            selection_path=tmp_path / "selection.json",
            model_snapshot=tmp_path / "model",
            output_root=tmp_path / "evaluation" / "run",
            repository_root=tmp_path,
            execute=False,
        )


def test_missing_comparator_is_explicit_technical_blocker(tmp_path: Path) -> None:
    result = capture.inspect_comparator_availability(None, root=tmp_path)
    assert result["method"] == capture.OPTIONAL_COMPARATOR
    assert result["status"] == "BLOCKED_TECHNICAL"
    assert result["preregistered"] is True
    assert result["model_loaded"] is False
    assert result["truth_opened"] is False


def test_pending_comparator_preserves_preregistered_subset(tmp_path: Path) -> None:
    result = capture.inspect_comparator_availability(
        {
            "status": "PENDING_AGENT1_PACKAGE",
            "preregistered": True,
            "blocker_id": "AGENT1_PACKAGE_PENDING",
            "reason": "fit and restore are pending",
            "subset": _subset(),
        },
        root=tmp_path,
    )
    assert result["status"] == "BLOCKED_TECHNICAL"
    assert result["subset"]["records_per_domain"] == 128
    assert result["subset"]["first_records_per_domain"] is True


def test_ready_comparator_preflight_is_hash_only(tmp_path: Path) -> None:
    package = tmp_path / "comparator-package.bin"
    package.write_bytes(b"opaque package metadata fixture")
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    result = capture.inspect_comparator_availability(
        {
            "status": "READY",
            "preregistered": True,
            "subset": _subset(),
            "package": {"path": package.name, "sha256": digest},
        },
        root=tmp_path,
    )
    assert result["status"] == "AVAILABLE_METADATA_ONLY"
    assert result["package"]["sha256"] == digest
    assert result["model_loaded"] is False
    assert result["truth_opened"] is False


def test_comparator_hash_tamper_fails_closed(tmp_path: Path) -> None:
    package = tmp_path / "comparator-package.bin"
    package.write_bytes(b"opaque package metadata fixture")
    with pytest.raises(capture.ComparatorPreflightError, match="hash changed"):
        capture.inspect_comparator_availability(
            {
                "status": "READY",
                "preregistered": True,
                "subset": _subset(),
                "package": {"path": package.name, "sha256": "0" * 64},
            },
            root=tmp_path,
        )


def test_observation_manifest_rehashes_all_four_sanitized_cells(tmp_path: Path) -> None:
    observations: dict[str, dict[str, object]] = {}
    for cell in capture.CELL_ORDER:
        path = tmp_path / "evaluation" / f"{cell}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(cell.encode("ascii"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        observations[cell] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "shape": [256, 128, 2048],
            "activations_key": "activations",
            "attention_mask_key": "attention_mask",
            "position_ids_key": "position_ids",
        }
    payload = capture.build_observation_manifest(
        selection_record={"path": "selection.json", "bytes": 1, "sha256": "a" * 64},
        selection_payload={"status": "FROZEN_TRR-P11_SOURCE_SELECTION_NO_TRUTH"},
        observations=observations,
        record_ids_sha256={"pile": "b" * 64, "finance": "c" * 64},
        root=tmp_path,
    )
    assert payload["cell_order"] == list(capture.CELL_ORDER)
    assert len(payload["cells"]) == 4
    assert payload["truth_opened"] is False
    assert payload["source_pairing"]["same_record_ids_across_targets"] is True


def test_capture_public_injected_path_uses_p11_descriptors_and_trr6_keyword(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real adapter orchestration with fake heavy producers."""
    import json
    import sys
    import types

    root = tmp_path
    arrow_paths = {}
    for domain in capture.DOMAIN_ORDER:
        arrow = root / f"{domain}.arrow"
        arrow.write_bytes(f"{domain}-arrow".encode())
        arrow_paths[domain] = arrow
    tokenizer_dir = root / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_file = tokenizer_dir / "tokenizer.json"
    tokenizer_file.write_text("{}", encoding="utf-8")

    def record(path: Path) -> dict[str, object]:
        data = path.read_bytes()
        return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    inputs: dict[str, object] = {}
    for domain in capture.DOMAIN_ORDER:
        inputs[domain] = {
            "dataset_key": domain,
            **capture.selector._DATASET_META[domain],
            "arrow_files": [record(arrow_paths[domain])],
        }
    inputs["tokenizer"] = {"path": str(tokenizer_dir.resolve()), "files": {"tokenizer.json": record(tokenizer_file)}}
    rows = {
        domain: [{"record_id": f"{domain}-{index}"} for index in range(capture.RECORDS_PER_DOMAIN)]
        for domain in capture.DOMAIN_ORDER
    }
    selection = types.SimpleNamespace(
        payload={"public_sources_frozen": inputs},
        record={"path": str(root / "selection.json"), "bytes": 1, "sha256": "a" * 64},
        rows=rows,
    )
    monkeypatch.setattr(capture.selector, "load_selection", lambda path, root: selection)
    monkeypatch.setattr(capture.selector, "_normalize_source_inputs", lambda payload, root: inputs)
    monkeypatch.setattr(capture, "_materialize_selected", lambda context, trusted, datasets, tokenizer: {domain: [] for domain in capture.DOMAIN_ORDER})

    class FakeDevice:
        type = "cpu"

        def __str__(self) -> str:
            return "cpu"

    trusted = types.ModuleType("scripts.trr0005_produce_confirmation")
    trusted._device = lambda value: FakeDevice()
    trusted._load_tokenizer = lambda path: object()
    trusted._load_arrow_dataset = lambda paths: object()
    helper = types.ModuleType("scripts.trr0006_capture_public")
    call_log: list[dict[str, object]] = []

    def fake_capture_condition(**kwargs: object) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
        call_log.append(dict(kwargs))
        output_root = Path(str(kwargs["output_root"]))
        condition = str(kwargs["condition"])
        observations: dict[str, dict[str, object]] = {}
        for domain in capture.DOMAIN_ORDER:
            cell = f"{domain}__{condition}"
            path = output_root / "observations" / f"{cell}.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(cell.encode())
            record_order = [f"{domain}-{index}" for index in range(capture.RECORDS_PER_DOMAIN)]
            observations[cell] = {
                **record(path),
                "cell_id": cell,
                "shape": [256, 128, 2048],
                "stored_sequence_tokens": 128,
                "scored_post_bos_tokens": 127,
                "capture_batch_records": 8,
                "capture_sequence_tokens": 192,
                "activations_key": "activations",
                "attention_mask_key": "attention_mask",
                "position_ids_key": "position_ids",
                "tensor_sha256": {
                    "activations": "a" * 64,
                    "attention_mask": "b" * 64,
                    "position_ids": "c" * 64,
                },
                "record_order": record_order,
                "record_order_sha256": capture._canonical_digest(record_order),
            }
        return observations, {"condition": condition, "synthetic": True}

    helper._capture_condition = fake_capture_condition
    helper._batches = lambda records: {domain: object() for domain in capture.DOMAIN_ORDER}
    scripts_package = sys.modules["scripts"]
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "scripts.trr0005_produce_confirmation", trusted)
    monkeypatch.setitem(sys.modules, "scripts.trr0006_capture_public", helper)
    monkeypatch.setattr(scripts_package, "trr0005_produce_confirmation", trusted, raising=False)
    monkeypatch.setattr(scripts_package, "trr0006_capture_public", helper, raising=False)

    model = root / "model"
    model.mkdir()
    config = root / "lora.json"
    config.write_text("{}", encoding="utf-8")
    update = root / "lora.safetensors"
    update.write_bytes(b"lora")
    update_hash = hashlib.sha256(update.read_bytes()).hexdigest()
    monkeypatch.setattr(capture, "_load_target_lora_binding", lambda root: update_hash)
    monkeypatch.setattr(
        capture,
        "_observation_tensor_bindings",
        lambda path, normalized=False: {
            "activations": "a" * 64,
            "attention_mask": "b" * 64,
            "position_ids": "c" * 64,
        },
    )

    output = root / "experiments" / "TRR-P11" / "evaluation" / "capture"
    result = capture.capture_public(
        selection_path=root / "selection.json",
        model_snapshot=model,
        output_root=output,
        repository_root=root,
        lora_config=config,
        lora_update=update,
        device="cpu",
        execute=True,
    )
    assert result["status"] == capture.CAPTURE_STATUS
    assert [entry["condition"] for entry in call_log] == list(capture.TARGET_ORDER)
    assert all("lora_update" in entry and "lora_update_path" not in entry for entry in call_log)
    capture_record = json.loads(Path(result["capture"]["path"]).read_text(encoding="utf-8"))
    observation_manifest = json.loads(Path(result["observation_manifest"]["path"]).read_text(encoding="utf-8"))
    assert capture_record["truth_opened"] is False
    assert capture_record["execution"]["model_loaded_by_producer"] is True
    assert {item["cell_id"] for item in capture_record["cells"]} == set(capture.CELL_ORDER)
    assert all("tensor_sha256" in item["observation"] and "record_order" in item["observation"] for item in observation_manifest["cells"])
    first = observation_manifest["cells"][0]["observation"]
    assert first["record_order"][:2] == ["record/000", "record/001"]
    assert first["capture_record_order"][:2] == ["pile-0", "pile-1"]
    assert first["record_order_sha256"] == capture._canonical_digest(first["record_order"])
    assert first["capture_record_order_sha256"] == capture._canonical_digest(first["capture_record_order"])
    assert capture_record["cells"][0]["observation"]["record_order"] == first["record_order"]
