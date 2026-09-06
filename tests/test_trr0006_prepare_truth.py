"""Focused tests for the TRR-0006 private truth preparation boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest
from safetensors.torch import save_file
import torch

from scripts import trr0006_prepare_truth as truth


CELL_HASHES = {cell: "a" * 64 for cell in truth.CELL_ORDER}


def _record(record_id: str, offset: int) -> SimpleNamespace:
    token_ids = torch.arange(truth.SEQUENCE_TOKENS, dtype=torch.int64) + offset
    token_ids[0] = truth.BOS_TOKEN_ID
    return SimpleNamespace(record_id=record_id, token_ids=token_ids)


def _file_record(path: Path) -> dict[str, object]:
    return truth._file_record(path)


def test_build_truth_tensors_uses_two_domain_keys_and_clones_source_rows() -> None:
    source = {
        "pile": [_record("pile-0", 1), _record("pile-1", 2)],
        "finance": [_record("finance-0", 3), _record("finance-1", 4)],
    }

    tensors, digests = truth._build_truth_tensors(source, records_per_domain=2)

    assert set(tensors) == set(truth.TRUTH_TENSOR_KEYS)
    assert tensors["pile__token_ids"].shape == (2, truth.SEQUENCE_TOKENS)
    assert tensors["finance__token_ids"].dtype == torch.int64
    assert tensors["pile__token_ids"].data_ptr() != source["pile"][0].token_ids.data_ptr()
    assert tensors["finance__token_ids"].data_ptr() != source["finance"][0].token_ids.data_ptr()
    assert all(tensor[:, 0].eq(truth.BOS_TOKEN_ID).all().item() for tensor in tensors.values())
    assert digests == {
        "pile": truth._json_digest(["pile-0", "pile-1"]),
        "finance": truth._json_digest(["finance-0", "finance-1"]),
    }


def test_outside_destination_is_create_only_and_outside_repository(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()

    outside = truth._outside_destination(
        tmp_path / "private" / "truth.safetensors",
        root=root,
        description="truth sidecar",
    )
    assert outside == (tmp_path / "private" / "truth.safetensors").resolve()
    assert outside.parent.is_dir()

    with pytest.raises(truth.TruthPreparationError, match="outside the repository"):
        truth._outside_destination(
            root / "experiments/TRR-0006/truth.safetensors",
            root=root,
            description="truth sidecar",
        )


def test_load_truth_tensor_map_validates_two_domain_sidecar(tmp_path: Path) -> None:
    truth_path = tmp_path / "truth.safetensors"
    labels = torch.arange(2 * truth.SEQUENCE_TOKENS, dtype=torch.int64).reshape(2, truth.SEQUENCE_TOKENS)
    labels[:, 0] = truth.BOS_TOKEN_ID
    metadata = {
        "schema": truth.TRUTH_FILE_SCHEMA,
        "task_id": truth.TASK_ID,
        "decision_plan_sha256": "d" * 64,
        "source_selection_sha256": "s" * 64,
        "observation_sha256": json.dumps(CELL_HASHES, sort_keys=True, separators=(",", ":")),
        "truth_opened": "false",
    }
    save_file(
        {"pile__token_ids": labels, "finance__token_ids": labels.clone()},
        str(truth_path),
        metadata=metadata,
    )
    manifest_path = tmp_path / "truth.binding.json"
    manifest = {
        "schema": truth.TRUTH_SCHEMA,
        "task_id": truth.TASK_ID,
        "status": truth.TRUTH_STATUS,
        "truth_file": _file_record(truth_path),
        "decision_plan_sha256": "d" * 64,
        "source_selection_sha256": "s" * 64,
        "observation_sha256": CELL_HASHES,
        "records_per_domain": 2,
        "truth_tensor_keys": list(truth.TRUTH_TENSOR_KEYS),
        "reconstruction_root_contains_truth": False,
        "truth_opened": False,
    }
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    loaded = truth.load_truth_tensor_map(manifest_path)

    assert set(loaded) == set(truth.TRUTH_TENSOR_KEYS)
    assert torch.equal(loaded["pile__token_ids"], labels)
    assert torch.equal(loaded["finance__token_ids"], labels)
