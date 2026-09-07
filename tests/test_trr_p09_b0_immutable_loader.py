"""Synthetic tests for the metadata-bound monolithic B0 adapter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile

import pytest
import torch
from safetensors.torch import save_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.trr_p09.b0_immutable_loader import B0ImmutableLoader
from scripts.trr_p09.prepare_streamed_bank import BankContractError, file_record, sha256_file


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _binding_fixture(root: Path) -> Path:
    rows = [
        {"record_id": f"b0-{index:02d}", "slot": index}
        for index in range(8)
    ]
    mask = torch.zeros((8, 192), dtype=torch.uint8)
    mask[:, : 12] = 1
    positions = torch.where(
        mask.bool(),
        torch.arange(192, dtype=torch.int64).expand(8, -1),
        torch.zeros((8, 192), dtype=torch.int64),
    )
    tokens = torch.zeros((8, 192), dtype=torch.int32)
    tokens[:, 0] = 128000
    activations = torch.zeros((8, 192, 2048), dtype=torch.bfloat16)
    for index in range(8):
        activations[index, 0, 0] = index
    payload = root / "b0.safetensors"
    save_file(
        {
            "activations": activations,
            "attention_mask": mask,
            "position_ids": positions,
            "token_ids": tokens,
            "post_bos_selector_large": mask.clone(),
            "post_bos_selector_small": mask.clone(),
        },
        str(payload),
    )
    records = root / "records.json"
    _write_json(records, {"records": rows})
    published = root / "published.json"
    payload_digest = sha256_file(payload)
    ids = [row["record_id"] for row in rows]
    id_digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    _write_json(
        published,
        {
            "fit_record_count": 8,
            "record_ids_sha256": id_digest,
            "geometry": {"fit": [8, 192, 2048]},
        },
    )
    position_contract = root / "position.json"
    _write_json(
        position_contract,
        {
            "payload": {"sha256": payload_digest},
            "geometry": {"records": 8},
            "observed": {"active_arange": True, "inactive_position_values": [0]},
        },
    )
    corpus = root / "corpus.json"
    _write_json(corpus, {"public": True})

    def descriptor(path: Path) -> dict[str, object]:
        return file_record(path, label="fixture")

    binding = {
        "schema": "token-reconstruction.trr-p09-b0-immutable-loader-binding.v1",
        "task_id": "TRR-P09",
        "bank": {
            "expanded_row_origin": 0,
            "record_count": 8,
            "geometry": {
                "sequence_tokens": 192,
                "hidden_size": 2048,
                "loader_batch_records": 8,
                "shard_records": 8,
                "hidden_dtype": "torch.bfloat16",
            },
            "allowed_extra_tensor_keys": ["post_bos_selector_large", "post_bos_selector_small"],
        },
        "artifacts": {
            "payload": descriptor(payload),
            "records": descriptor(records),
            "published_manifest": descriptor(published),
            "position_contract": descriptor(position_contract),
            "corpus_plan": descriptor(corpus),
        },
        "record_ids": {"count": 8, "sha256": id_digest},
    }
    path = root / "binding.json"
    _write_json(path, binding)
    return path


def test_b0_loader_reads_only_requested_rows_and_preserves_repeats() -> None:
    with tempfile.TemporaryDirectory(prefix="trr-p09-b0-") as raw:
        binding = _binding_fixture(Path(raw))
        loader = B0ImmutableLoader(binding)
        batch = loader.get_records([7, 0, 7])
        assert batch.global_rows == (7, 0, 7)
        assert batch.record_ids == ("b0-07", "b0-00", "b0-07")
        assert batch.sequence_ids == batch.record_ids
        assert batch.activations.dtype == torch.bfloat16
        assert batch.activations[:, 0, 0].tolist() == [7, 0, 7]
        assert batch.attention_mask.dtype == torch.bool
        assert batch.position_ids[0, :12].tolist() == list(range(12))
        assert batch.position_ids[0, 12:].eq(0).all()


def test_b0_loader_rejects_changed_sidecar_after_binding() -> None:
    with tempfile.TemporaryDirectory(prefix="trr-p09-b0-") as raw:
        binding = _binding_fixture(Path(raw))
        loader = B0ImmutableLoader(binding)
        records = Path(raw) / "records.json"
        records.write_text(records.read_text(encoding="utf-8").replace("b0-00", "changed"), encoding="utf-8")
        with pytest.raises(BankContractError, match="immutable B0 artifact changed"):
            loader.get_records([0])

