"""Synthetic contract checks for the TRR-P09 streamed public bank.

The fixture is deliberately not a public source or model artifact.  It checks
that a current-bank prefix remains byte-equivalent when embedded in an
expanded bank, that the loader streams fixed 8-row batches, and that create
only publication refuses accidental replacement.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pytest
import torch

from scripts.trr_p09.prepare_streamed_bank import (
    BankContractError,
    BankGeometry,
    StreamedBankLoader,
    estimate_storage,
    file_record,
    sha256_file,
    validate_bank_manifest,
    verify_input_snapshot_bindings,
    write_fixture_bank,
)


def test_fixture_current_prefix_and_streamed_rows_are_equivalent() -> None:
    with tempfile.TemporaryDirectory(prefix="trr-p09-bank-") as raw:
        root = Path(raw) / "bank"
        manifest = write_fixture_bank(root, record_count=16, current_record_count=8)
        checked = validate_bank_manifest(root / "bank_manifest.json")
        assert checked["bank"]["current_prefix_record_count"] == 8
        assert manifest["bank"]["current_prefix_equivalence"]["verification"] == "FIXTURE_VERIFIED"

        batches = list(StreamedBankLoader(root / "bank_manifest.json").iter_batches())
        assert len(batches) == 2
        assert all(tuple(batch.activations.shape) == (8, 192, 2048) for batch in batches)
        assert all(batch.activations.dtype == torch.bfloat16 for batch in batches)
        assert batches[0].global_rows == tuple(range(8))
        assert batches[1].global_rows == tuple(range(8, 16))
        assert batches[0].record_ids == tuple(f"fixture-record-{i:04d}" for i in range(8))
        assert torch.equal(
            batches[0].position_ids.cpu(),
            torch.arange(192, dtype=torch.long).expand(8, -1),
        )
        assert batches[0].attention_mask.dtype == torch.bool

        scheduled = StreamedBankLoader(root / "bank_manifest.json").get_records([9, 0, 15, 9])
        assert scheduled.global_rows == (9, 0, 15, 9)
        assert scheduled.record_ids == (
            "fixture-record-0009",
            "fixture-record-0000",
            "fixture-record-0015",
            "fixture-record-0009",
        )
        assert torch.equal(scheduled.activations[0], scheduled.activations[3])


def test_fixture_output_is_create_only() -> None:
    with tempfile.TemporaryDirectory(prefix="trr-p09-bank-") as raw:
        root = Path(raw) / "bank"
        write_fixture_bank(root, record_count=8, current_record_count=8)
        with pytest.raises(BankContractError, match="empty/create-only"):
            write_fixture_bank(root, record_count=8, current_record_count=8)


def test_input_snapshot_records_are_exactly_hash_bound() -> None:
    with tempfile.TemporaryDirectory(prefix="trr-p09-bind-") as raw:
        path = Path(raw) / "public_snapshot_manifest.json"
        path.write_text("{\"public\":true}\n", encoding="utf-8")
        record = file_record(path, label="fixture snapshot")
        verified = verify_input_snapshot_bindings(
            {"public_model_snapshot_manifest": {"file": record}},
            required_roles=("public_model_snapshot_manifest",),
        )
        assert verified["public_model_snapshot_manifest"]["sha256"] == record["sha256"]
        path.write_text("{\"public\":false}\n", encoding="utf-8")
        with pytest.raises(BankContractError, match="changed"):
            verify_input_snapshot_bindings(
                {"public_model_snapshot_manifest": {"file": record}},
                required_roles=("public_model_snapshot_manifest",),
            )


def test_storage_preflight_keeps_bank_off_gpu() -> None:
    geometry = BankGeometry(shard_records=64)
    estimate = estimate_storage(record_count=12000, geometry=geometry)
    assert estimate["hidden_bytes"] == 12000 * 192 * 2048 * 2
    assert estimate["one_batch_hidden_bytes"] == 8 * 192 * 2048 * 2
    assert estimate["shard_count"] == 188
    assert estimate["include_hidden"] is True


def _fresh_fixture_root(raw: str) -> Path:
    root = Path(raw) / "bank"
    write_fixture_bank(root, record_count=16, current_record_count=8)
    return root


def test_integrity_gate_rejects_corrupted_payload_hash() -> None:
    with tempfile.TemporaryDirectory(prefix="trr-p09-integrity-") as raw:
        root = _fresh_fixture_root(raw)
        payload = root / "shards/shard-000000/payload.safetensors"
        value = bytearray(payload.read_bytes())
        value[-1] ^= 1
        payload.write_bytes(value)
        with pytest.raises(BankContractError, match="changed"):
            StreamedBankLoader(root / "bank_manifest.json")


def test_integrity_gate_rejects_corrupted_sidecar_metadata() -> None:
    with tempfile.TemporaryDirectory(prefix="trr-p09-integrity-") as raw:
        root = _fresh_fixture_root(raw)
        sidecar = root / "shards/shard-000000/shard.json"
        value = json.loads(sidecar.read_text(encoding="utf-8"))
        value["records"][0]["global_row"] = 99
        sidecar.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(BankContractError, match="changed"):
            StreamedBankLoader(root / "bank_manifest.json")


def test_integrity_gate_checks_global_row_after_sidecar_hash_binding() -> None:
    with tempfile.TemporaryDirectory(prefix="trr-p09-integrity-") as raw:
        root = _fresh_fixture_root(raw)
        sidecar = root / "shards/shard-000000/shard.json"
        value = json.loads(sidecar.read_text(encoding="utf-8"))
        value["records"][0]["global_row"] = 99
        sidecar.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_path = root / "bank_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        descriptor = manifest["sharding"]["shards"][0]["sidecar"]
        descriptor["bytes"] = sidecar.stat().st_size
        descriptor["sha256"] = sha256_file(sidecar)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(BankContractError, match="global_row"):
            StreamedBankLoader(manifest_path)


def test_integrity_gate_rejects_missing_complete_marker() -> None:
    with tempfile.TemporaryDirectory(prefix="trr-p09-integrity-") as raw:
        root = _fresh_fixture_root(raw)
        (root / "shards/shard-000000/COMPLETE").unlink()
        with pytest.raises(BankContractError, match="unavailable|absent|disappeared"):
            StreamedBankLoader(root / "bank_manifest.json")


@pytest.mark.parametrize("bad_start", [7, 9])
def test_integrity_gate_rejects_overlapping_or_gapped_ranges(bad_start: int) -> None:
    with tempfile.TemporaryDirectory(prefix="trr-p09-integrity-") as raw:
        root = _fresh_fixture_root(raw)
        manifest_path = root / "bank_manifest.json"
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        value["sharding"]["shards"][1]["row_range"]["start"] = bad_start
        value["sharding"]["shards"][1]["row_range"]["stop"] = bad_start + 8
        manifest_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(BankContractError, match="contiguous|row ranges"):
            StreamedBankLoader(manifest_path)


def test_loader_uses_stat_only_immutability_check_after_gate() -> None:
    with tempfile.TemporaryDirectory(prefix="trr-p09-integrity-") as raw:
        root = _fresh_fixture_root(raw)
        loader = StreamedBankLoader(root / "bank_manifest.json")
        payload = root / "shards/shard-000000/payload.safetensors"
        payload.write_bytes(payload.read_bytes())
        with pytest.raises(BankContractError, match="changed after integrity gate"):
            loader.get_records([0, 1, 2, 3])


def test_nonfixture_manifest_requires_and_verifies_input_snapshot_binding() -> None:
    with tempfile.TemporaryDirectory(prefix="trr-p09-integrity-") as raw:
        root = _fresh_fixture_root(raw)
        snapshot = Path(raw) / "public-snapshot-manifest.json"
        snapshot.write_text("{\"public_model\":true}\n", encoding="utf-8")
        manifest_path = root / "bank_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["fixture"] = False
        manifest["input_snapshots"] = {
            "public_model_snapshot_manifest": {"file": file_record(snapshot, label="fixture snapshot")}
        }
        manifest["input_binding"] = {
            "required_before_forward": True,
            "roles": ["public_model_snapshot_manifest"],
            "descriptor_field": "file",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        loader = StreamedBankLoader(manifest_path)
        assert loader.get_records([0, 1]).global_rows == (0, 1)
