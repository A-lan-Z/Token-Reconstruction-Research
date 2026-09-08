#!/usr/bin/env python3
"""Materialize only the signed B0 schedule from the surviving B0 mask.

This task adapter imports the exact P09 schedule factory and serializer. It
reads only the B0 attention-mask sidecar through safetensors and never opens
B0 activations, token IDs, model weights, public E, labels, or truth. B1 is
intentionally absent until the rebuilt B1 mask passes the frozen ledger.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from safetensors import safe_open

_TASK_ROOT = Path(__file__).resolve().parents[2]
_P09_ROOT = _TASK_ROOT.parent / "TRR-P10"
for _root in (_P09_ROOT, _P09_ROOT / "src"):
    if str(_root) in sys.path:
        sys.path.remove(str(_root))
    sys.path.insert(0, str(_root))

from scripts.trr_p09.fixed_control_caller import materialize_schedule_plan  # noqa: E402
from scripts.trr_p09.fixed_control_runner import tensor_digest  # noqa: E402
from scripts.trr_p09.prepare_common_schedules import (  # noqa: E402
    _file_record,
    _serialize_plan,
)

SEED = 4010
STEPS = 13000
RECORD_BATCH_SIZE = 8
POSITION_BUDGET = 512
SEQUENCE_TOKENS = 192
B0_RECORDS = 1200


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular file: {path}")
    return path


def prepare(*, input_payload: Path, b0_binding: Path, output_root: Path) -> dict[str, Any]:
    input_payload = _regular(input_payload, label="B0 payload")
    b0_binding = _regular(b0_binding, label="B0 binding")
    output_root = output_root.expanduser().resolve()
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError(f"B0 schedule output is create-only: {output_root}")
    output_root.mkdir(parents=True)

    with safe_open(str(input_payload), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if "attention_mask" not in keys:
            raise RuntimeError("B0 payload lacks attention_mask")
        mask = handle.get_tensor("attention_mask").contiguous()
    if tuple(mask.shape) != (B0_RECORDS, SEQUENCE_TOKENS):
        raise RuntimeError(f"B0 mask geometry changed: {tuple(mask.shape)}")
    if str(mask.dtype) not in {"torch.bool", "torch.uint8"}:
        raise RuntimeError(f"B0 mask dtype changed: {mask.dtype}")
    mask = mask.to(dtype=bool, device="cpu")
    if not bool(mask[:, 0].all().item()):
        raise RuntimeError("B0 mask lost BOS column")
    valid_positions = int(mask[:, 1:].sum().item())
    if valid_positions != 124371:
        raise RuntimeError(f"B0 valid-position count changed: {valid_positions}")

    plan = materialize_schedule_plan(
        mask,
        tuple(range(B0_RECORDS)),
        steps=STEPS,
        seed=SEED,
        record_batch_size=RECORD_BATCH_SIZE,
        position_budget=POSITION_BUDGET,
    )
    mask_info = {
        "file": _file_record(input_payload),
        "tensor": {
            "name": "attention_mask",
            "shape": [B0_RECORDS, SEQUENCE_TOKENS],
            "dtype": str(mask.dtype),
            "semantic_sha256": tensor_digest(mask),
            "post_bos_valid_positions": valid_positions,
            "read_scope": "attention_mask_only",
        },
    }
    summary = _serialize_plan(
        plan,
        output=output_root / f"schedule-b0-seed{SEED}.safetensors",
        bank="B0",
        mask_info=mask_info,
    )
    receipt = {
        "schema": "token-reconstruction.trr0012-b0-schedule-preparation-receipt.v1",
        "task_id": "TRR-0012",
        "status": "PASS_CPU_B0_SCHEDULE_PREPARED_NO_MODEL",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "p09_source_root": str(_P09_ROOT),
            "fixed_control_caller": _file_record(_P09_ROOT / "scripts/trr_p09/fixed_control_caller.py"),
            "fixed_control_runner": _file_record(_P09_ROOT / "scripts/trr_p09/fixed_control_runner.py"),
            "prepare_common_schedules": _file_record(_P09_ROOT / "scripts/trr_p09/prepare_common_schedules.py"),
            "factory": "scripts.trr_p09.fixed_control_caller.materialize_schedule_plan",
            "serializer": "scripts.trr_p09.prepare_common_schedules._serialize_plan",
        },
        "input": {
            "b0_binding": _file_record(b0_binding),
            "payload": _file_record(input_payload),
            "mask": mask_info["tensor"],
        },
        "schedule_contract": {
            "bank": "B0",
            "seed": SEED,
            "steps": STEPS,
            "record_batch_size": RECORD_BATCH_SIZE,
            "position_budget": POSITION_BUDGET,
            "sequence_tokens": SEQUENCE_TOKENS,
            "valid_positions": valid_positions,
            "schedule_semantic_sha256": plan.semantic_sha256,
            "expected_signed_schedule_semantic_sha256": "9aaad9c030f2f9b801f91f956c97f858966580edd21229cebb350dcde358f2c3",
            "expected_signed_mask_semantic_sha256": "1cae0090d1eb80479c0e0e18529efba3de508eba4ad497d680bcb478072ed0a0",
        },
        "output": summary,
        "truth_boundary": {
            "attention_mask_only": True,
            "activations_opened": False,
            "token_ids_opened": False,
            "model_loaded": False,
            "public_E_loaded": False,
            "fit_started": False,
            "evaluation_truth_opened": False,
            "B1_schedule_created": False,
        },
    }
    receipt_path = output_root / "schedule-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-payload", type=Path, required=True)
    parser.add_argument("--b0-binding", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(input_payload=args.input_payload, b0_binding=args.b0_binding, output_root=args.output_root)
    print(json.dumps({"status": result["status"], "output_root": str(args.output_root.resolve()), "schedule": result["output"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
