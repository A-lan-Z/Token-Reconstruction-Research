"""Independent CPU integrity tests for the TRR-P01 freeze/score gate.

The fixture has the declared public geometry but synthetic finite activations.
It exercises the byte and identity contract without opening any real private
records.  The tests intentionally mutate one boundary at a time so a passing
freeze receipt cannot mask a later change in observations or predictions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from scripts.trr_p01 import freeze_score as fs
from scripts.trr_p01.common import (
    BOS_TOKEN_ID,
    CONFIG_SCHEMA,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    OBSERVATION_INDEX_SCHEMA,
    OBSERVATION_SCHEMA,
    PREDICTION_SCHEMA,
    SCORED_TOKENS,
    SEQUENCE_TOKENS,
    TASK_ID,
    VOCAB_SIZE,
    digest_tensor,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_fixture(tmp_path: Path) -> dict[str, Path | list[str]]:
    public = tmp_path / "public-arm"
    public.mkdir()
    record_ids = [f"p01-r{index:04d}" for index in range(1, 17)]

    # Keep values simple and finite while retaining the exact BF16 public-arm
    # geometry.  The freeze gate checks byte-level row digests independently.
    observations = torch.zeros((16, SEQUENCE_TOKENS, HIDDEN_SIZE), dtype=torch.bfloat16)
    observation_path = public / "observations.safetensors"
    save_file(
        {"activations": observations},
        observation_path,
        metadata={
            "schema": OBSERVATION_SCHEMA,
            "opaque_records": "true",
            "source_truth_included": "false",
        },
    )
    observation_digest = [digest_tensor(row) for row in observations]
    mask_digest = digest_tensor(torch.ones(SEQUENCE_TOKENS, dtype=torch.int64))
    position_digest = digest_tensor(torch.arange(SEQUENCE_TOKENS, dtype=torch.int64))
    observation_index = {
        "schema": OBSERVATION_INDEX_SCHEMA,
        "task_id": TASK_ID,
        "truth_opened": False,
        "source_material_included": False,
        "geometry": {
            "records": 16,
            "sequence_tokens": SEQUENCE_TOKENS,
            "scored_tokens": SCORED_TOKENS,
            "hidden_size": HIDDEN_SIZE,
        },
        "records": [
            {
                "record_id": record_id,
                "sequence_length": SEQUENCE_TOKENS,
                "mask_digest": mask_digest,
                "position_digest": position_digest,
                "observation_digest": row_digest,
            }
            for record_id, row_digest in zip(record_ids, observation_digest)
        ],
        "observation": {
            "path": "observations.safetensors",
            "bytes": observation_path.stat().st_size,
            "sha256": _sha256(observation_path),
        },
    }
    _write_json(public / "observation_index.json", observation_index)

    table_path = public / "table.bin"
    table_path.write_bytes(b"synthetic public table bytes")
    table_hash = _sha256(table_path)
    config = {
        "schema": CONFIG_SCHEMA,
        "task_id": TASK_ID,
        "truth_opened": False,
        "source_truth_included": False,
        "record_order": record_ids,
        "geometry": {
            "records": 16,
            "sequence_tokens": SEQUENCE_TOKENS,
            "scored_tokens": SCORED_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "cut_depth": 4,
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "hidden_size": HIDDEN_SIZE,
            "vocab_size": VOCAB_SIZE,
            "cut_depth": 4,
            "bos_token_id": BOS_TOKEN_ID,
        },
        "methods": ["boundary"],
        "metric_order": ["cosine"],
        "table": {
            "path": "table.bin",
            "bytes": table_path.stat().st_size,
            "sha256": table_hash,
        },
    }
    config_path = public / "sanitized_config.json"
    _write_json(config_path, config)
    config_hash = _sha256(config_path)

    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    predictions = torch.full(
        (16, SEQUENCE_TOKENS), BOS_TOKEN_ID, dtype=torch.int32
    )
    tensor_path = predictions_dir / "predictions.safetensors"
    save_file(
        {"boundary.cosine": predictions},
        tensor_path,
        metadata={
            "schema": PREDICTION_SCHEMA,
            "task_id": TASK_ID,
            "truth_opened": "false",
        },
    )
    evidence_hash = "e" * 64
    rows: list[dict[str, Any]] = []
    for index, record_id in enumerate(record_ids):
        rows.append(
            {
                "record_id": record_id,
                "method": "boundary",
                "metric": "cosine",
                "sequence_length": SEQUENCE_TOKENS,
                "prediction_tokens": [int(token) for token in predictions[index].tolist()],
                "mask_digest": mask_digest,
                "position_digest": position_digest,
                "observation_digest": observation_digest[index],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "cut_depth": 4,
                "vocab_size": VOCAB_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "config_sha256": config_hash,
                "evidence_sha256": evidence_hash,
                "table_sha256": table_hash,
                "truth_opened": False,
            }
        )
    jsonl_path = predictions_dir / "predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    truth_dir = tmp_path / "private"
    truth_dir.mkdir()
    truth = torch.full((16, SEQUENCE_TOKENS), BOS_TOKEN_ID, dtype=torch.int64)
    truth_path = truth_dir / "private_truth.safetensors"
    save_file({"input_ids": truth}, truth_path, metadata={})
    return {
        "public": public,
        "predictions": predictions_dir,
        "tensor": tensor_path,
        "jsonl": jsonl_path,
        "truth": truth_path,
        "receipt": tmp_path / "freeze-receipt.json",
        "record_ids": record_ids,
    }


def test_valid_freeze_validate_and_score_open_truth_only_after_gate(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    receipt = fs.freeze_predictions(
        fixture["public"], fixture["predictions"], fixture["receipt"]
    )
    assert receipt["truth_opened"] is False
    assert receipt["status"] == "FROZEN_AND_VERIFIED_BEFORE_TRUTH_OPEN"

    verified = fs.validate_frozen(
        fixture["public"], fixture["predictions"], fixture["receipt"]
    )
    assert verified["truth_opened"] is False
    scored = fs.score_frozen(
        fixture["public"],
        fixture["predictions"],
        fixture["receipt"],
        fixture["truth"],
    )
    assert scored["truth_opened_after_freeze_verification"] is True
    assert scored["arms"][0]["metrics"]["exact_record_rate"] == 1.0


@pytest.mark.parametrize(
    ("replacement", "message"),
    [("p01-r0002", "duplicate opaque IDs"), ("foreign-id", "foreign opaque ID")],
)
def test_freeze_rejects_duplicate_or_foreign_opaque_ids(
    tmp_path: Path, replacement: str, message: str
) -> None:
    fixture = _make_fixture(tmp_path)
    rows = [json.loads(line) for line in Path(fixture["jsonl"]).read_text().splitlines()]
    rows[0]["record_id"] = replacement
    with Path(fixture["jsonl"]).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with pytest.raises(fs.FreezeScoreError, match=message):
        fs.freeze_predictions(fixture["public"], fixture["predictions"], fixture["receipt"])


def test_validate_rejects_changed_observation_bytes(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    fs.freeze_predictions(fixture["public"], fixture["predictions"], fixture["receipt"])
    observation_path = Path(fixture["public"]) / "observations.safetensors"
    payload = bytearray(observation_path.read_bytes())
    payload[-1] ^= 1
    observation_path.write_bytes(payload)
    with pytest.raises(fs.FreezeScoreError, match="observation hash changed"):
        fs.validate_frozen(fixture["public"], fixture["predictions"], fixture["receipt"])


def test_freeze_rejects_wrong_prediction_shape_and_dtype(tmp_path: Path) -> None:
    for replacement, message in (
        (torch.zeros((15, SEQUENCE_TOKENS), dtype=torch.int32), "geometry changed"),
        (torch.zeros((16, SEQUENCE_TOKENS), dtype=torch.int64), "dtype changed"),
    ):
        case = tmp_path / ("shape" if replacement.shape[0] == 15 else "dtype")
        case.mkdir()
        fixture = _make_fixture(case)
        save_file(
            {"boundary.cosine": replacement},
            Path(fixture["tensor"]),
            metadata={
                "schema": PREDICTION_SCHEMA,
                "task_id": TASK_ID,
                "truth_opened": "false",
            },
        )
        with pytest.raises(fs.FreezeScoreError, match=message):
            fs.freeze_predictions(fixture["public"], fixture["predictions"], fixture["receipt"])


def test_freeze_rejects_prediction_pair_tampering(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    rows = [json.loads(line) for line in Path(fixture["jsonl"]).read_text().splitlines()]
    rows[0]["prediction_tokens"][1] = 1
    with Path(fixture["jsonl"]).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with pytest.raises(fs.FreezeScoreError, match="tensor and JSONL disagree"):
        fs.freeze_predictions(fixture["public"], fixture["predictions"], fixture["receipt"])


def test_score_rejects_tamper_before_private_truth_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _make_fixture(tmp_path)
    fs.freeze_predictions(fixture["public"], fixture["predictions"], fixture["receipt"])
    tensor_path = Path(fixture["tensor"])
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)
    called = False

    def should_not_open(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("private truth opened before public validation")

    monkeypatch.setattr(fs, "_load_truth", should_not_open)
    with pytest.raises(fs.FreezeScoreError, match="prediction tensor|invalid prediction"):
        fs.score_frozen(
            fixture["public"],
            fixture["predictions"],
            fixture["receipt"],
            fixture["truth"],
        )
    assert called is False


def test_validate_rejects_reordered_prediction_rows(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    fs.freeze_predictions(fixture["public"], fixture["predictions"], fixture["receipt"])
    rows = Path(fixture["jsonl"]).read_text(encoding="utf-8").splitlines()
    Path(fixture["jsonl"]).write_text(
        "\n".join([rows[1], rows[0], *rows[2:]]) + "\n", encoding="utf-8"
    )
    with pytest.raises(fs.FreezeScoreError, match="opaque ID order or coverage"):
        fs.validate_frozen(fixture["public"], fixture["predictions"], fixture["receipt"])


def test_freeze_rejects_wrong_prediction_metadata_and_row_identity(tmp_path: Path) -> None:
    cases = ("tensor-truth", "row-model", "row-observation")
    for case in cases:
        case_root = tmp_path / case
        case_root.mkdir()
        fixture = _make_fixture(case_root)
        if case == "tensor-truth":
            save_file(
                {"boundary.cosine": torch.full((16, SEQUENCE_TOKENS), BOS_TOKEN_ID, dtype=torch.int32)},
                Path(fixture["tensor"]),
                metadata={
                    "schema": PREDICTION_SCHEMA,
                    "task_id": TASK_ID,
                    "truth_opened": "true",
                },
            )
            expected = "metadata or truth state"
        else:
            rows = [json.loads(line) for line in Path(fixture["jsonl"]).read_text().splitlines()]
            if case == "row-model":
                rows[0]["model_revision"] = "wrong-revision"
                expected = "model identity"
            else:
                rows[0]["observation_digest"] = "0" * 64
                expected = "observation_digest changed"
            with Path(fixture["jsonl"]).open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        with pytest.raises(fs.FreezeScoreError, match=expected):
            fs.freeze_predictions(fixture["public"], fixture["predictions"], fixture["receipt"])
