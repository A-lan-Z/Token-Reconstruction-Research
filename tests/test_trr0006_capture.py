from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from scripts import trr0006_capture_public as capture
from scripts import trr0006_prediction_contract as contract
from scripts import trr0006_select_public as selector
from scripts import trr0005_produce_confirmation as trusted


def _frozen_plan() -> dict[str, object]:
    return {
        "schema": "token-reconstruction.trr0006-decision-plan.v1",
        "task_id": "TRR-0006",
        "status": "FROZEN_BEFORE_NEW_SOURCE_SELECTION",
        "sample_size_frozen": True,
        "panel": {
            "records_per_domain": 8,
            "clip_tokens_including_bos": 128,
            "capture_batch_records": 8,
            "capture_tokens": 192,
            "selection_seed": 5005,
            "source_ranges_half_open": {"pile": [7000, 10000], "finance": [12000, 20000]},
        },
        "comparison": {"target_conditions": ["public_base", "public_lora_2601"]},
        "method_freeze_sha256": "a" * 64,
        "truth_gate": {"truth_status": "NOT_OPENED"},
    }


def _selection() -> dict[str, object]:
    records: dict[str, list[dict[str, object]]] = {}
    for style, start in (("pile", 7000), ("finance", 12000)):
        spec = capture.eligibility.SOURCE_PARTITIONS[style]
        rows = []
        for offset in range(8):
            index = start + offset
            record_id = trusted.source_record_id(
                str(spec["dataset_id"]), str(spec["split"]), str(spec["revision"]), index
            )
            rows.append(
                {
                    "record_id": record_id,
                    "public_record_sha256": f"{offset + (1 if style == 'pile' else 20):064x}",
                    "dataset_key": style,
                    "dataset_id": spec["dataset_id"],
                    "split": spec["split"],
                    "revision": spec["revision"],
                    "row_index": index,
                    "source_index": index,
                    "full_token_count": 128,
                    "post_bos_token_count": 127,
                    "valid_tokens": 128,
                    "final_sequence_sha256": f"{offset + (100 if style == 'pile' else 200):064x}",
                }
            )
        records[style] = rows
    return {
        "schema": capture.SELECTION_SCHEMA,
        "task_id": "TRR-0006",
        "status": capture.SELECTION_STATUS,
        "records_per_domain": 8,
        "method_freeze_sha256": "a" * 64,
        "source_ranges_half_open": {"pile": [7000, 10000], "finance": [12000, 20000]},
        "target_conditions": ["public_base", "public_lora_2601"],
        "paired_conditions": True,
        "selection_rule": {"records": records},
    }


def test_capture_validates_frozen_selection_and_source_pairing(tmp_path: Path) -> None:
    frozen_path = tmp_path / "frozen.json"
    selection_path = tmp_path / "selection.json"
    frozen_path.write_text(json.dumps(_frozen_plan()) + "\n", encoding="utf-8")
    selection_path.write_text(json.dumps(_selection()) + "\n", encoding="utf-8")

    frozen, selection, rows = capture._validate_frozen_selection(frozen_path, selection_path)

    assert frozen["records_per_domain"] == 8
    assert selection["status"] == capture.SELECTION_STATUS
    assert len(rows["pile"]) == len(rows["finance"]) == 8


def test_capture_rejects_selection_payload_field(tmp_path: Path) -> None:
    value = _selection()
    value["selection_rule"]["records"]["pile"][0]["token_ids"] = [1, 2]
    frozen_path = tmp_path / "frozen.json"
    selection_path = tmp_path / "selection.json"
    frozen_path.write_text(json.dumps(_frozen_plan()) + "\n", encoding="utf-8")
    selection_path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(capture.CaptureError, match="payload fields"):
        capture._validate_frozen_selection(frozen_path, selection_path)


def test_observation_artifact_has_only_runner_sidecars(tmp_path: Path) -> None:
    records = 8
    path = tmp_path / "observation.safetensors"
    batch = SimpleNamespace(
        attention_mask=torch.ones((records, capture.CAPTURE_SEQUENCE_TOKENS), dtype=torch.uint8),
        position_ids=torch.arange(capture.CAPTURE_SEQUENCE_TOKENS, dtype=torch.int64).repeat(records, 1),
    )
    descriptor = capture._save_observation(
        path,
        compact=torch.zeros((records, capture.SEQUENCE_TOKENS, capture.HIDDEN_SIZE), dtype=torch.bfloat16),
        batch=batch,
        cell_id="pile__public_base",
        records_per_domain=records,
        record_ids_sha256="a" * 64,
        selection_sha256="b" * 64,
    )

    assert descriptor["shape"] == [8, 128, 2048]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {"activations", "attention_mask", "position_ids"}
        assert handle.get_tensor("activations").dtype == torch.bfloat16


def test_observation_manifest_matches_runner_schema(tmp_path: Path) -> None:
    records = 8
    artifact = tmp_path / "observation.safetensors"
    batch = SimpleNamespace(
        attention_mask=torch.ones((records, capture.CAPTURE_SEQUENCE_TOKENS), dtype=torch.uint8),
        position_ids=torch.arange(capture.CAPTURE_SEQUENCE_TOKENS, dtype=torch.int64).repeat(records, 1),
    )
    descriptor = capture._save_observation(
        artifact,
        compact=torch.zeros((records, capture.SEQUENCE_TOKENS, capture.HIDDEN_SIZE), dtype=torch.bfloat16),
        batch=batch,
        cell_id="pile__public_base",
        records_per_domain=records,
        record_ids_sha256="a" * 64,
        selection_sha256="b" * 64,
    )
    observations = {
        cell_id: {
            **descriptor,
            "path": str(artifact),
            "producer_only_lora": cell_id.endswith("__public_lora_2601"),
        }
        for cell_id in capture.CELL_ORDER
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}\n", encoding="utf-8")
    manifest = capture._build_observation_manifest(
        output_root=tmp_path,
        selection_path=selection_path,
        selection_sha256="b" * 64,
        frozen={"records_per_domain": records, "method_freeze_sha256": "c" * 64},
        observations=observations,
        record_ids_sha256={"pile": "a" * 64, "finance": "d" * 64},
    )
    parsed = contract.validate_observation_manifest(
        manifest,
        registration={"records_per_domain": records},
        repository_root=tmp_path,
    )

    assert parsed["cell_order"] == list(capture.CELL_ORDER)
    assert parsed["record_ids_sha256"]["pile"] == "a" * 64


def test_capture_requires_explicit_execute_before_any_other_work() -> None:
    with pytest.raises(capture.CaptureError, match="--execute"):
        capture.capture_public(SimpleNamespace(execute=False))
