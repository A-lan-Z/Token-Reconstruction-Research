from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts import trr0005_produce_confirmation as producer
from token_reconstruction import trr0005_contract as contract
from token_reconstruction.trr0005_public_corpus import source_record_id


def _public_selection() -> dict:
    return {
        "schema": "token-reconstruction.trr0005-public-validation-selection.v1",
        "task_id": contract.TASK_ID,
        "status": "FROZEN_PUBLIC_VALIDATION_SELECTION",
        "selection_stage": "public_validation_before_fresh_evaluation",
        "truth_accessed": False,
        "distributions": {
            "original": {
                "selected_method_id": "original__joint_full_affine",
                "candidate_method_ids": [
                    "original__joint_full_affine",
                    "original__affine_trained_diagonal_attention128",
                ],
            },
            "enriched": {
                "selected_method_id": "enriched__joint_full_affine",
                "candidate_method_ids": [
                    "enriched__joint_full_affine",
                    "enriched__affine_trained_diagonal_attention128",
                ],
            },
        },
    }


def _freeze() -> dict:
    return {
        "schema": "synthetic-method-freeze.v1",
        "task_id": contract.TASK_ID,
        "status": "FROZEN_METHOD_PRESELECTION",
        "method_ids": list(contract.METHOD_IDS),
        "code_commit": "a" * 40,
        "decision_plan_sha256": "b" * 64,
        "state_bindings": {
            method_id: {"status": "FROZEN", "state_sha256": "c" * 64}
            for method_id in contract.METHOD_IDS
        },
        "public_validation_selection": _public_selection(),
        "truth_opened": False,
        "fresh_evaluation_started": False,
    }


def test_select_rejects_before_tokenizer_or_dataset_load(tmp_path, monkeypatch):
    freeze_path = tmp_path / "method_freeze.json"
    value = _freeze()
    value["status"] = "DECLARED"
    freeze_path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(
        producer,
        "_load_tokenizer",
        lambda _path: pytest.fail("tokenizer loaded before freeze rejection"),
    )
    args = Namespace(
        repository_root=tmp_path,
        method_freeze=freeze_path,
        decision_plan=None,
        public_validation_selection=None,
        tokenizer=tmp_path / "tokenizer",
        pile_arrow=[tmp_path / "pile.arrow"],
        finance_arrow=[tmp_path / "finance.arrow"],
        exclude_source=[],
        output=tmp_path / "selection.json",
    )
    with pytest.raises(producer.ProducerError, match="not in a frozen"):
        producer.select_public(args)


def test_reserved_row_guard_runs_before_dataset_subscript():
    class SpyDataset:
        def __init__(self):
            self.calls = []

        def __getitem__(self, index):
            self.calls.append(index)
            return {"text": "should never be read"}

    dataset = SpyDataset()
    with pytest.raises(producer.ProducerError, match="partition"):
        producer._read_reserved_row(dataset, style="pile", row_index=0)
    assert dataset.calls == []


def test_prior_normalized_and_tokenized_hashes_block_candidates():
    exclusions = producer.ExclusionSets(
        ids={style: set() for style in contract.STYLE_ORDER},
        hashes={style: set() for style in contract.STYLE_ORDER},
        indices={style: set() for style in contract.STYLE_ORDER},
        sources=[],
    )
    producer._scan_identity_metadata(
        {
            "record_id": "finance-public-old",
            "normalized_content_sha256": "a" * 64,
            "tokenized_record_sha256": "b" * 64,
            "token_ids_sha256": "c" * 64,
        },
        hint="prior-finance-panel",
        result=exclusions,
    )
    candidate = producer.FreshRecord(
        style="finance",
        dataset_key="finance",
        dataset_id="finance",
        split="train",
        revision="r",
        row_index=12000,
        record_id="finance-new",
        public_record_sha256="d" * 64,
        token_ids=(contract.BOS_TOKEN_ID,) + (1,) * 127,
        final_sequence_sha256="b" * 64,
    )
    assert "a" * 64 in exclusions.hashes["finance"]
    assert "b" * 64 in exclusions.hashes["finance"]
    assert "c" * 64 in exclusions.hashes["finance"]
    assert producer._blocked(candidate, exclusions) == "public_final_sequence_hash"


def test_panel_has_paired_contract_order_and_geometry(tmp_path):
    records = {}
    observations = {}
    for style in contract.STYLE_ORDER:
        spec = producer.SOURCE_PARTITIONS[style]
        rows = []
        for offset in range(contract.RECORDS_PER_DOMAIN):
            index = int(spec["holdout_reserve_start"]) + offset
            source_id = source_record_id(
                str(spec["dataset_id"]),
                str(spec["split"]),
                str(spec["revision"]),
                index,
            )
            rows.append(
                producer.FreshRecord(
                    style=style,
                    dataset_key=style,
                    dataset_id=str(spec["dataset_id"]),
                    split=str(spec["split"]),
                    revision=str(spec["revision"]),
                    row_index=index,
                    record_id=source_id,
                    public_record_sha256="d" * 64,
                    token_ids=(contract.BOS_TOKEN_ID,) + tuple(
                        (offset + token) % 1000 + 1 for token in range(127)
                    ),
                    final_sequence_sha256=f"{offset + 1:064x}",
                )
            )
        records[style] = rows
    plan_path = tmp_path / "selection_plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    plan = {
        "method_freeze_sha256": "e" * 64,
        "public_validation_selection": _public_selection(),
    }
    for cell_id in contract.EXPECTED_CELL_IDS:
        observations[cell_id] = {
            "path": str(tmp_path / f"{cell_id}.safetensors"),
            "bytes": 1,
            "sha256": "f" * 64,
            "shape": [128, 128, 2048],
        }
    panel = producer._build_panel(
        plan=plan,
        plan_path=plan_path,
        records=records,
        observations=observations,
    )
    assert tuple(panel["cells"]) == contract.EXPECTED_CELL_IDS
    assert panel["cells"]["pile__public_base"]["records"] == panel["cells"][
        "pile__public_lora_2601"
    ]["records"]
    assert panel["cells"]["finance__public_base"]["records"] == panel["cells"][
        "finance__public_lora_2601"
    ]["records"]
    assert panel["cells"]["pile__public_base"]["attention_mask"] == [[1] * 128] * 128
    assert panel["cells"]["pile__public_base"]["position_ids"][0] == list(range(128))


def test_truth_binding_rejects_wrong_sidecar_and_row_order(tmp_path):
    records = {}
    observations = {}
    for style in contract.STYLE_ORDER:
        spec = producer.SOURCE_PARTITIONS[style]
        rows = []
        for offset in range(contract.RECORDS_PER_DOMAIN):
            index = int(spec["holdout_reserve_start"]) + offset
            rows.append(
                producer.FreshRecord(
                    style=style,
                    dataset_key=style,
                    dataset_id=str(spec["dataset_id"]),
                    split=str(spec["split"]),
                    revision=str(spec["revision"]),
                    row_index=index,
                    record_id=source_record_id(
                        str(spec["dataset_id"]),
                        str(spec["split"]),
                        str(spec["revision"]),
                        index,
                    ),
                    public_record_sha256=f"{offset + 1:064x}",
                    token_ids=(contract.BOS_TOKEN_ID,) + (offset + 1,) * 127,
                    final_sequence_sha256=f"{offset + 1000:064x}",
                )
            )
        records[style] = rows
    plan = {
        "method_freeze_sha256": "e" * 64,
        "public_validation_selection": _public_selection(),
    }
    plan_path = tmp_path / "selection_plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    for cell_id in contract.EXPECTED_CELL_IDS:
        observations[cell_id] = {
            "path": str(tmp_path / f"{cell_id}.safetensors"),
            "bytes": 1,
            "sha256": "f" * 64,
            "shape": [128, 128, 2048],
        }
    panel = producer._build_panel(
        plan=plan,
        plan_path=plan_path,
        records=records,
        observations=observations,
    )
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(panel, sort_keys=True), encoding="utf-8")
    expected_ids = {
        style: [record.record_id for record in records[style]]
        for style in contract.STYLE_ORDER
    }
    tensors = {}
    labels = torch.arange(128 * 128, dtype=torch.int64).reshape(128, 128)
    mask = torch.ones((128, 128), dtype=torch.uint8)
    positions = torch.arange(128, dtype=torch.int64).repeat(128, 1)
    for cell_id in contract.EXPECTED_CELL_IDS:
        tensors[f"{cell_id}__token_ids"] = labels.clone()
        tensors[f"{cell_id}__attention_mask"] = mask.clone()
        tensors[f"{cell_id}__position_ids"] = positions.clone()

    def write_truth(path, metadata_ids):
        save_file(
            tensors,
            str(path),
            metadata={
                "schema": producer.TRUTH_SCHEMA,
                "task_id": contract.TASK_ID,
                "panel_sha256": producer._sha256_file(panel_path),
                "selection_plan_sha256": producer._sha256_file(plan_path),
                "method_freeze_sha256": plan["method_freeze_sha256"],
                "record_ids_sha256_pile": producer._json_sha256(metadata_ids["pile"]),
                "record_ids_sha256_finance": producer._json_sha256(metadata_ids["finance"]),
                "record_ids_pile": producer._canonical_json(metadata_ids["pile"]),
                "record_ids_finance": producer._canonical_json(metadata_ids["finance"]),
                "truth_opened": "false",
            },
        )
        return producer._file_record(path)

    truth_path = tmp_path / "truth.safetensors"
    truth_record = write_truth(truth_path, expected_ids)
    observation_hashes = {
        cell_id: observations[cell_id]["sha256"]
        for cell_id in contract.EXPECTED_CELL_IDS
    }
    manifest = {
        "schema": producer.TRUTH_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "PUBLIC_TRUTH_PREPARED_OUTSIDE_RECONSTRUCTION_ROOT",
        "truth_file": truth_record,
        "panel": producer._file_record(panel_path),
        "selection_plan": producer._file_record(plan_path),
        "method_freeze_sha256": plan["method_freeze_sha256"],
        "observation_sha256": observation_hashes,
        "record_ids_sha256": {
            style: producer._json_sha256(expected_ids[style])
            for style in contract.STYLE_ORDER
        },
        "record_ids": expected_ids,
        "cell_order": list(contract.EXPECTED_CELL_IDS),
        "truth_tensor_keys": sorted(tensors),
        "truth_opened": False,
        "reconstruction_root_contains_truth": False,
    }
    manifest_path = tmp_path / "truth.manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    validated = producer.validate_truth_binding(
        manifest_path,
        panel_path=panel_path,
        selection_plan_path=plan_path,
        truth_path=truth_path,
    )
    assert validated["row_order_validated"] is True

    wrong_manifest = dict(manifest)
    wrong_manifest["record_ids"] = {
        **expected_ids,
        "pile": list(reversed(expected_ids["pile"])),
    }
    wrong_manifest_path = tmp_path / "wrong-order.manifest.json"
    wrong_manifest_path.write_text(
        json.dumps(wrong_manifest, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(producer.ProducerError, match="ordered record IDs"):
        producer.validate_truth_binding(
            wrong_manifest_path,
            panel_path=panel_path,
            selection_plan_path=plan_path,
            truth_path=truth_path,
        )

    swapped_truth_path = tmp_path / "swapped-truth.safetensors"
    swapped_ids = {
        **expected_ids,
        "finance": list(reversed(expected_ids["finance"])),
    }
    swapped_record = write_truth(swapped_truth_path, swapped_ids)
    swapped_manifest = {
        **manifest,
        "truth_file": swapped_record,
    }
    swapped_manifest_path = tmp_path / "swapped-truth.manifest.json"
    swapped_manifest_path.write_text(
        json.dumps(swapped_manifest, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(producer.ProducerError, match="row order differs"):
        producer.validate_truth_binding(
            swapped_manifest_path,
            panel_path=panel_path,
            selection_plan_path=plan_path,
            truth_path=swapped_truth_path,
        )
