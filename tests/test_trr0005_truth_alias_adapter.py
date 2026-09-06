from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file as raw_save_file

from scripts import trr0005_produce_confirmation as producer
from scripts import trr0005_truth_alias_adapter as adapter
from token_reconstruction import trr0005_contract as contract


def _synthetic_records() -> dict[str, list[producer.FreshRecord]]:
    records: dict[str, list[producer.FreshRecord]] = {}
    for style in contract.STYLE_ORDER:
        rows: list[producer.FreshRecord] = []
        for row_index in range(contract.RECORDS_PER_DOMAIN):
            token_ids = (contract.BOS_TOKEN_ID,) + tuple(
                (row_index + offset) % 1000 + 1
                for offset in range(contract.SEQUENCE_TOKENS - 1)
            )
            rows.append(
                producer.FreshRecord(
                    style=style,
                    dataset_key=style,
                    dataset_id=f"synthetic-{style}",
                    split="train",
                    revision="synthetic-v1",
                    row_index=row_index,
                    record_id=f"{style}-synthetic-{row_index:03d}",
                    public_record_sha256=f"{row_index + 1:064x}",
                    token_ids=token_ids,
                    final_sequence_sha256=f"{row_index + 1000:064x}",
                )
            )
        records[style] = rows
    return records


def _synthetic_panel(
    *,
    plan_path: Path,
    records: dict[str, list[producer.FreshRecord]],
) -> dict:
    cells: dict[str, dict] = {}
    for cell_index, cell_id in enumerate(contract.EXPECTED_CELL_IDS):
        style, _condition = cell_id.split("__", 1)
        cells[cell_id] = {
            "records": [
                {"record_id": record.record_id}
                for record in records[style]
            ],
            "observation": {"sha256": f"{cell_index + 1:064x}"},
        }
    return {
        "selection_plan": {"sha256": producer._sha256_file(plan_path)},
        "cells": cells,
    }


def test_truth_cli_clones_shared_tensors_at_serialization_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The unmodified safetensors boundary rejects the exact alias pattern in
    # frozen prepare_truth: both condition keys point at one tensor object.
    aliased = torch.arange(16, dtype=torch.int64).reshape(4, 4)
    with pytest.raises(RuntimeError):
        raw_save_file(
            {"condition_a": aliased, "condition_b": aliased},
            str(tmp_path / "alias-rejected.safetensors"),
            metadata={"marker": "synthetic"},
        )

    root = tmp_path / "repo"
    root.mkdir()
    plan_path = root / "selection_plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    panel_path = root / "panel.json"
    records = _synthetic_records()
    panel_path.write_text(
        json.dumps(_synthetic_panel(plan_path=plan_path, records=records)) + "\n",
        encoding="utf-8",
    )
    freeze_path = root / "method_freeze.json"
    freeze_path.write_text("{}\n", encoding="utf-8")
    tokenizer_path = root / "tokenizer.json"
    pile_path = root / "pile.arrow"
    finance_path = root / "finance.arrow"
    for path in (tokenizer_path, pile_path, finance_path):
        path.write_bytes(b"synthetic placeholder")
    truth_path = tmp_path / "truth.safetensors"
    manifest_path = tmp_path / "truth.manifest.json"

    synthetic_plan = {"synthetic": True}
    monkeypatch.setattr(
        producer,
        "_validate_preselection",
        lambda *args, **kwargs: {"sha256": "a" * 64},
    )
    monkeypatch.setattr(
        producer,
        "_validate_selection_plan",
        lambda path, *, freeze: synthetic_plan,
    )
    monkeypatch.setattr(producer, "validate_panel_descriptor", lambda panel: {})
    monkeypatch.setattr(
        producer,
        "_validate_frozen_source_descriptors",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(producer, "_load_tokenizer", lambda path: object())
    monkeypatch.setattr(producer, "_load_arrow_dataset", lambda paths: object())

    def materialize(
        plan: object, *, style: str, dataset: object, tokenizer: object
    ) -> list[producer.FreshRecord]:
        return records[style]

    monkeypatch.setattr(producer, "_materialize_selected", materialize)

    observed: dict[str, object] = {}
    original_adapter_save = adapter.save_file_alias_safe
    original_raw_save = adapter._save_file

    def inspect_boundary(
        tensors: dict[str, torch.Tensor],
        filename: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> None:
        observed["input_keys"] = tuple(tensors)
        observed["input_tensors"] = {
            key: value.detach().clone() for key, value in tensors.items()
        }
        observed["input_metadata"] = dict(metadata or {})
        pile_base = tensors["pile__public_base__token_ids"]
        pile_lora = tensors["pile__public_lora_2601__token_ids"]
        observed["producer_passed_alias"] = (
            pile_base.data_ptr() == pile_lora.data_ptr()
        )
        original_adapter_save(tensors, filename, metadata=metadata)

    def inspect_serialized(
        tensors: dict[str, torch.Tensor],
        filename: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> None:
        storage = [value.untyped_storage().data_ptr() for value in tensors.values()]
        observed["serialized_unique_storage"] = len(storage) == len(set(storage))
        observed["serialized_keys"] = tuple(tensors)
        original_raw_save(tensors, filename, metadata=metadata)

    monkeypatch.setattr(adapter, "save_file_alias_safe", inspect_boundary)
    monkeypatch.setattr(adapter, "_save_file", inspect_serialized)

    argv = [
        "truth",
        "--repository-root",
        str(root),
        "--method-freeze",
        str(freeze_path),
        "--selection-plan",
        str(plan_path),
        "--panel",
        str(panel_path),
        "--tokenizer",
        str(tokenizer_path),
        "--pile-arrow",
        str(pile_path),
        "--finance-arrow",
        str(finance_path),
        "--truth-output",
        str(truth_path),
        "--truth-manifest",
        str(manifest_path),
    ]
    assert adapter.run_truth_cli(argv) == 0
    assert observed["producer_passed_alias"] is True
    assert observed["serialized_unique_storage"] is True

    with safe_open(truth_path, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(observed["serialized_keys"])
        for key, expected in observed["input_tensors"].items():
            actual = handle.get_tensor(key)
            assert actual.dtype == expected.dtype
            assert tuple(actual.shape) == tuple(expected.shape)
            assert torch.equal(actual, expected)
        assert dict(handle.metadata() or {}) == observed["input_metadata"]
