"""CPU-only contract tests for the bounded STAGE1 capture adapter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import replace
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pytest
import torch
from safetensors.torch import save_file
from safetensors import safe_open

from scripts.trr_p09.prepare_streamed_bank import BankGeometry, CombinedStreamedBankLoader, file_record, write_fixture_bank
from scripts.trr_p09 import stage1_public_capture as capture


GIB = 2**30


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _descriptor(path: Path, *, root: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _input_fixture(tmp_path: Path, *, rows: int = 16, expanded_origin: int = 0) -> capture.InputManifest:
    assert rows % 8 == 0
    root = tmp_path / "inputs"
    root.mkdir()
    geometry = BankGeometry(shard_records=8)
    shards: list[dict[str, Any]] = []
    for shard_id, start in enumerate(range(0, rows, 8)):
        count = min(8, rows - start)
        token_ids = torch.full((count, 192), 128001, dtype=torch.int32)
        mask = torch.zeros((count, 192), dtype=torch.bool)
        for local in range(count):
            active = 128 if local == 0 else 192
            token_ids[local, 0] = 128000
            token_ids[local, 1:active] = torch.arange(1, active, dtype=torch.int32) + local
            mask[local, :active] = True
        positions = torch.zeros((count, 192), dtype=torch.int64)
        positions[:, :] = torch.where(mask, torch.arange(192, dtype=torch.int64).expand(count, -1), torch.zeros((count, 192), dtype=torch.int64))
        shard_dir = root / f"shard-{shard_id:06d}"
        shard_dir.mkdir()
        payload = shard_dir / "inputs.safetensors"
        save_file({"token_ids": token_ids, "attention_mask": mask, "position_ids": positions}, str(payload))
        records = []
        for local in range(count):
            records.append(
                {
                    "record_id": f"fixture-record-{start + local:04d}",
                    "parent_record_id": f"fixture-parent-{start + local:04d}",
                    "sequence_id": f"fixture-sequence-{start + local:04d}",
                    "source_record_sha256": _sha(f"source-{start + local}"),
                    "sequence_sha256": _sha(f"sequence-{start + local}"),
                    "active_token_count": 128 if local == 0 else 192,
                    "global_row": expanded_origin + start + local,
                }
            )
        sidecar = shard_dir / "input.json"
        sidecar.write_text(
            json.dumps(
                {
                    "schema": capture.INPUT_SHARD_SCHEMA,
                    "task_id": "TRR-P09",
                    "shard_id": shard_id,
                    "row_range": {"start": start, "stop": start + count, "count": count},
                    "records": records,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        shards.append(
            {
                "shard_id": shard_id,
                "row_range": {"start": start, "stop": start + count, "count": count},
                "payload": _descriptor(payload, root=root),
                "sidecar": _descriptor(sidecar, root=root),
            }
        )
    manifest_path = root / "input_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": capture.INPUT_MANIFEST_SCHEMA,
                "task_id": "TRR-P09",
                "status": "FROZEN_STAGE1_PUBLIC_INPUT_NO_TRUTH",
                "condition": "public_base",
                "truth_boundary": {"source_text_written": False, "truth_opened": False, "target_labels_loaded": False},
                "geometry": {"sequence_tokens": 192, "hidden_size": 2048, "loader_batch_records": 8, "shard_records": 8, "hidden_dtype": "torch.bfloat16"},
                "bank": {"record_count": rows, "expanded_row_origin": expanded_origin},
                "shards": shards,
                "qualification_batches": [{"shard_id": 0, "batch_index": 0}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return capture.load_input_manifest(manifest_path, expected_record_count=rows, require_stage1=False)


class _FakePrefix:
    cut_depth = 4

    def __init__(self) -> None:
        self.calls = 0

    def eval(self) -> "_FakePrefix":
        return self

    def forward_full(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        values = input_ids.to(dtype=torch.bfloat16).unsqueeze(-1)
        return values.expand(-1, -1, 2048).contiguous()


class _FailingPrefix(_FakePrefix):
    def forward_full(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise AssertionError("verified complete shards must be skipped before model forward")


def _snapshot(**kwargs: Any) -> dict[str, Any]:
    return {
        "gpu_free_bytes": 12 * GIB,
        "gpu_total_bytes": 16 * GIB,
        "gpu_reserved_bytes": 1 * GIB,
        "host_available_bytes": 20 * GIB,
        "process_rss_bytes": 1 * GIB,
        "disk_free_bytes": 100 * GIB,
        "retained_bytes": int(kwargs.get("retained_bytes", 0)),
    }


def _guard(tmp_path: Path) -> capture.ResourceGuard:
    return capture.ResourceGuard(
        device=torch.device("cpu"),
        output_root=tmp_path / "out",
        snapshot_fn=_snapshot,
    )


def test_cli_module_imports_and_help() -> None:
    with pytest.raises(SystemExit) as exc:
        capture._parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_qualify_repeats_b8_and_future_padding(tmp_path: Path) -> None:
    manifest = _input_fixture(tmp_path)
    prefix = _FakePrefix()
    result = capture.qualify_capture(
        prefix=prefix,
        input_manifest=manifest,
        device=torch.device("cpu"),
        guard=_guard(tmp_path),
    )
    assert result["status"] == "QUALIFICATION_PASS"
    assert result["batches"][0]["repeat_torch_equal"] is True
    assert result["batches"][0]["future_padding_active_torch_equal"] is True
    assert prefix.calls == 3  # original, repeat, future-padding variant


def test_compiler_capture_combined_b0_b1_loader_smoke(tmp_path: Path) -> None:
    """Exercise synthetic compiler output through capture and B0+B1 loading."""

    # _input_fixture is the small capture-input compiler fixture: its rows,
    # sidecars, active masks, and zero-padded positions pass the same parser
    # used by the production input schema.
    manifest = _input_fixture(tmp_path, rows=16, expanded_origin=64)
    parsed = capture.load_input_manifest(
        manifest.path, expected_record_count=16, require_stage1=False
    )
    assert parsed.expanded_row_origin == 64
    prefix_root = tmp_path / "b0"
    write_fixture_bank(prefix_root, record_count=64, current_record_count=64)
    addition_root = tmp_path / "b1"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    capture.qualify_capture(
        prefix=_FakePrefix(),
        input_manifest=parsed,
        device=torch.device("cpu"),
        guard=_guard(tmp_path),
    )
    capture.run_capture(
        prefix=_FakePrefix(),
        input_manifest=parsed,
        plan={"schema": "synthetic"},
        plan_record=file_record(plan_path, label="synthetic plan"),
        output_root=addition_root,
        device=torch.device("cpu"),
        guard=_guard(tmp_path),
    )
    combined = CombinedStreamedBankLoader(
        prefix_root / "bank_manifest.json",
        addition_root / "bank_manifest.json",
    )
    scheduled = combined.get_records([0, 63, 64, 71, 79, 64])
    assert scheduled.global_rows == (0, 63, 64, 71, 79, 64)
    assert scheduled.record_ids[0] == "fixture-record-0000"
    assert scheduled.record_ids[2] == "fixture-record-0000"
    # The first B1 row is the padded synthetic compiler representative.
    assert int(scheduled.attention_mask[2].sum().item()) == 128
    assert int(scheduled.position_ids[2, 127].item()) == 127
    assert int(scheduled.position_ids[2, 128].item()) == 0
    assert int(scheduled.position_ids[2, 191].item()) == 0


def test_capture_writes_immutable_shards_and_resumes_without_forward(tmp_path: Path) -> None:
    manifest = _input_fixture(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    plan_record = file_record(plan_path, label="synthetic plan")
    output = tmp_path / "out"
    first_prefix = _FakePrefix()
    first = capture.run_capture(
        prefix=first_prefix,
        input_manifest=manifest,
        plan={"schema": "synthetic"},
        plan_record=plan_record,
        output_root=output,
        device=torch.device("cpu"),
        guard=_guard(tmp_path),
    )
    assert first["status"] == "CAPTURE_COMPLETE_NO_TRUTH"
    assert first["shard_counts"] == {"created": 2, "skipped_existing_verified": 0}
    assert first_prefix.calls == 2  # one B8 forward per 8-record shard
    second = capture.run_capture(
        prefix=_FailingPrefix(),
        input_manifest=manifest,
        plan={"schema": "synthetic"},
        plan_record=plan_record,
        output_root=output,
        device=torch.device("cpu"),
        guard=_guard(tmp_path),
    )
    assert second["status"] == "CAPTURE_COMPLETE_NO_TRUTH"
    assert (output / "bank_manifest.json").is_file()
    for shard_id in range(2):
        assert (output / "shards" / f"shard-{shard_id:06d}" / "COMPLETE").read_bytes() == b"\n"


def test_prepared_full_payload_adapts_to_expanded_b1_rows(tmp_path: Path) -> None:
    root = tmp_path / "prepared"
    root.mkdir()
    rows = capture.TARGET_RECORDS
    tokens = torch.full((rows, 192), capture.PAD_TOKEN_ID, dtype=torch.int32)
    masks = torch.zeros((rows, 192), dtype=torch.uint8)
    positions = torch.zeros((rows, 192), dtype=torch.int64)
    tokens[:, 0] = capture.BOS_TOKEN_ID
    tokens[:, 1:192] = torch.arange(1, 192, dtype=torch.int32).expand(rows, -1)
    masks[:, :192] = 1
    positions[:, :] = torch.arange(192, dtype=torch.int64).expand(rows, -1)
    tokens[:capture.B0_ROWS, 128:] = capture.PAD_TOKEN_ID
    masks[:capture.B0_ROWS, 128:] = 0
    positions[:capture.B0_ROWS, 128:] = 0
    # Leave the first B1 row padded so the generated qualification includes a
    # future-padding case while the remainder supplies the longest row.
    tokens[capture.B0_ROWS, 128:] = capture.PAD_TOKEN_ID
    masks[capture.B0_ROWS, 128:] = 0
    positions[capture.B0_ROWS, 128:] = 0
    payload = root / "inputs.safetensors"
    save_file({"token_ids": tokens, "attention_mask": masks, "position_ids": positions}, str(payload))
    records = []
    for index in range(rows):
        records.append({
            "record_id": f"prepared-{index:05d}",
            "source_record_id": f"source-{index:05d}",
            "dataset_key": "pile" if index % 2 else "finance",
            "stratum": "pile_natural" if index % 2 else "finance_natural",
            "source_row_index": index,
            "rendered_sha256": f"{index + 1:064x}",
            "source_full_token_count": 193,
            "target_post_bos_token_count": 127 if index == capture.B0_ROWS else 191,
            "target_full_token_count": 129 if index < capture.B0_ROWS else 193,
            "sequence_h128_sha256": None,
            "global_row": index,
        })
    records_path = root / "records.json"
    records_path.write_text(json.dumps(records, sort_keys=True), encoding="utf-8")
    value = {
        "schema": capture.PREPARED_INPUT_SCHEMA,
        "task_id": "TRR-P09",
        "status": "CPU_INPUTS_COMPILED_NO_ACTIVATIONS",
        "plan": {"path": "experiments/TRR-P09/planning/stage1-public-bank-plan.json", "bytes": capture.SIGNED_STAGE1_PLAN_BYTES, "sha256": capture.SIGNED_STAGE1_PLAN_SHA256, "commit": "5bfed9ec6a7bb29a988ec0a4b1343b7745d8b81b"},
        "countersignature": {"path": "experiments/TRR-P09/setup/stage1-plan-countersignature-r1.json", "attested": True},
        "geometry": {"records": rows, "sequence_tokens": 192, "b0_prefix_records": capture.B0_ROWS, "input_dtype": "int32", "mask_dtype": "uint8", "position_dtype": "int64"},
        "artifacts": {
            "inputs": _descriptor(payload, root=root),
            "records": _descriptor(records_path, root=root),
        },
        "truth_boundary": {"public_fitting_labels_loaded": True, "source_text_persisted": False, "evaluation_truth_opened": False, "model_loaded": False, "activations_created": False},
    }
    manifest_path = root / "preparation_manifest.json"
    manifest_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    manifest = capture.load_input_manifest(manifest_path)
    assert manifest.expanded_row_origin == capture.B0_ROWS
    assert manifest.record_count == capture.NEW_RECORDS
    assert len(manifest.shards) == capture.TOTAL_SHARDS
    assert manifest.shards[0].source_start == capture.B0_ROWS
    assert manifest.shards[0].records[0]["global_row"] == capture.B0_ROWS
    assert manifest.shards[-1].records[-1]["global_row"] == capture.TARGET_RECORDS - 1


def test_watchdog_receipt_binds_real_wrapper_and_future_lease(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    input_path = tmp_path / "input.json"
    wrapper_path = tmp_path / "resource_watchdog.py"
    plan_path.write_text("{}", encoding="utf-8")
    input_path.write_text("{}", encoding="utf-8")
    wrapper_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    plan_record = file_record(plan_path, label="synthetic plan")
    input_record = file_record(input_path, label="synthetic input")
    wrapper_record = file_record(wrapper_path, label="synthetic watchdog")
    caps = capture.CaptureCaps()
    value = {
        "schema": capture.WATCHDOG_SCHEMA,
        "task_id": capture.TASK_ID,
        "status": "ARMED",
        "condition": capture.STAGE1_CONDITION,
        "mode": "qualify",
        "bindings": {
            "plan_sha256": plan_record["sha256"],
            "input_manifest_sha256": input_record["sha256"],
        },
        "command": [str(wrapper_path), "--child"],
        "executable": dict(wrapper_record, path=str(wrapper_path)),
        "lease": {"expires_utc": "2099-01-01T00:00:00Z"},
        "limits": {
            "wall_seconds_cap": caps.qualification_wall_seconds,
            "gpu_reserved_bytes_max": caps.max_reserved_gpu_bytes,
            "gpu_free_bytes_min": caps.min_gpu_free_before_bytes,
            "host_rss_bytes_max": caps.max_rss_bytes,
        },
        "post_child_race_safe": True,
    }
    receipt_path = tmp_path / "watchdog.json"
    receipt_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    checked, checked_record = capture._verify_watchdog_receipt(
        receipt_path, mode="qualify", plan_record=plan_record, input_record=input_record, caps=caps
    )
    assert checked["executable"]["sha256"] == wrapper_record["sha256"]
    assert checked_record["sha256"] == capture.sha256_file(receipt_path)
    value.pop("executable")
    receipt_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    with pytest.raises(capture.CaptureError, match="executable binding"):
        capture._verify_watchdog_receipt(
            receipt_path, mode="qualify", plan_record=plan_record, input_record=input_record, caps=caps
        )


def test_signed_plan_sha_override_cannot_weaken_binding() -> None:
    plan = Path("experiments/TRR-P09/planning/stage1-public-bank-plan.json")
    with pytest.raises(capture.CaptureError, match="signed plan SHA"):
        capture._verify_plan(plan, expected_sha256="0" * 64)


def test_capture_translates_new_rows_to_expanded_global_indices(tmp_path: Path) -> None:
    manifest = _input_fixture(tmp_path, expanded_origin=capture.B0_ROWS)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "out"
    capture.run_capture(
        prefix=_FakePrefix(),
        input_manifest=manifest,
        plan={"schema": "synthetic"},
        plan_record=file_record(plan_path, label="synthetic plan"),
        output_root=output,
        device=torch.device("cpu"),
        guard=_guard(tmp_path),
    )
    sidecar = json.loads((output / "shards" / "shard-000000" / "shard.json").read_text())
    assert sidecar["row_range"] == {"start": capture.B0_ROWS, "stop": capture.B0_ROWS + 8, "count": 8}
    bank = json.loads((output / "bank_manifest.json").read_text())
    assert bank["bank"]["global_row_range"] == {"start": capture.B0_ROWS, "stop": capture.B0_ROWS + 16, "count": 16}
    assert bank["bank"]["prefix_artifact"]["preserve_existing_prefix_rows"] is True


def test_resume_rejects_payload_input_mismatch_even_with_updated_payload_hash(tmp_path: Path) -> None:
    manifest = _input_fixture(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "out"
    capture.run_capture(
        prefix=_FakePrefix(),
        input_manifest=manifest,
        plan={"schema": "synthetic"},
        plan_record=file_record(plan_path, label="synthetic plan"),
        output_root=output,
        device=torch.device("cpu"),
        guard=_guard(tmp_path),
    )
    payload_path = output / "shards" / "shard-000000" / "payload.safetensors"
    with safe_open(str(payload_path), framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key).contiguous() for key in handle.keys()}
    tensors["token_ids"][0, 2] = tensors["token_ids"][0, 2] + 1
    save_file(tensors, str(payload_path))
    sidecar_path = output / "shards" / "shard-000000" / "shard.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["payload"]["bytes"] = payload_path.stat().st_size
    sidecar["payload"]["sha256"] = capture.sha256_file(payload_path)
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
    with pytest.raises(capture.CaptureError, match="token_ids differs"):
        capture.run_capture(
            prefix=_FailingPrefix(),
            input_manifest=manifest,
            plan={"schema": "synthetic"},
            plan_record=file_record(plan_path, label="synthetic plan"),
            output_root=output,
            device=torch.device("cpu"),
            guard=_guard(tmp_path),
        )


def test_qualification_selection_rejects_duplicate_and_missing_longest(tmp_path: Path) -> None:
    manifest = _input_fixture(tmp_path)
    with pytest.raises(capture.CaptureError, match="duplicated"):
        capture._validate_qualification_selection(
            manifest.shards,
            [{"shard_id": 0, "batch_index": 0}, {"shard_id": 0, "batch_index": 0}],
            require_stage1=True,
        )
    shortened = tuple({**row, "active_token_count": 128} for row in manifest.shards[0].records)
    shortened_shards = (replace(manifest.shards[0], records=shortened), *manifest.shards[1:])
    with pytest.raises(capture.CaptureError, match="longest"):
        capture._validate_qualification_selection(
            shortened_shards,
            [{"shard_id": 0, "batch_index": 0}],
            require_stage1=True,
        )


def test_resource_guard_fails_closed_on_gpu_floor(tmp_path: Path) -> None:
    def low_gpu(**kwargs: Any) -> dict[str, Any]:
        value = _snapshot(**kwargs)
        value["gpu_free_bytes"] = 1 * GIB
        return value

    guard = capture.ResourceGuard(device=torch.device("cpu"), output_root=tmp_path, snapshot_fn=low_gpu)
    with pytest.raises(capture.CaptureError, match="resource guard failed"):
        guard.check(phase="synthetic", prelaunch=False)
