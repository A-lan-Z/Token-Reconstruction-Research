#!/usr/bin/env python3
"""Materialize the shared P09 CPU position schedules.

The schedule factory lives in :mod:`fixed_control_caller`; this adapter only
binds the already prepared public input mask to the two immutable bank views
and serializes the existing ``SchedulePlan`` fields.  It never loads H,
tokens, a model, or evaluation truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.trr_p09.fixed_control_caller import (  # noqa: E402
    SIGNED_POSITION_BUDGET,
    inherited_schedule_steps,
    materialize_schedule_plan,
)
from scripts.trr_p09.fixed_control_runner import (  # noqa: E402
    SchedulePlan,
    canonical_digest,
    tensor_digest,
)


SCHEMA = "token-reconstruction.trr-p09-common-schedules.v1"
SEED = 4010
STEPS = 13000
RECORD_BATCH_SIZE = 8
POSITION_BUDGET = 512
SEQUENCE_TOKENS = 192
EXPECTED_RECORDS = 12000
EXPECTED_B0_RECORDS = 1200


class SchedulePreparationError(RuntimeError):
    """Raised when the immutable schedule binding is incomplete."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise SchedulePreparationError(f"required file is unavailable: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_attention_mask(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise SchedulePreparationError(f"input safetensors file is unavailable: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if "attention_mask" not in keys:
            raise SchedulePreparationError("input payload lacks attention_mask")
        mask = handle.get_tensor("attention_mask").contiguous()
    if tuple(mask.shape) != (EXPECTED_RECORDS, SEQUENCE_TOKENS):
        raise SchedulePreparationError(f"attention mask geometry changed: {tuple(mask.shape)}")
    if mask.dtype not in (torch.bool, torch.uint8):
        raise SchedulePreparationError(f"attention mask dtype changed: {mask.dtype}")
    mask = mask.to(device="cpu", dtype=torch.bool)
    if not bool(mask[:, 0].all().item()):
        raise SchedulePreparationError("attention mask lost the BOS column")
    if int(mask[:, 1:].sum().item()) <= 0:
        raise SchedulePreparationError("attention mask has no post-BOS positions")
    return mask, {
        "file": _file_record(path),
        "tensor": {
            "name": "attention_mask",
            "shape": list(mask.shape),
            "dtype": str(mask.dtype),
            "semantic_sha256": tensor_digest(mask),
            "post_bos_valid_positions": int(mask[:, 1:].sum().item()),
        },
        "read_scope": "attention_mask_only; token_ids, positions, H, model, and truth were not read",
    }


def _serialize_plan(plan: SchedulePlan, *, output: Path, bank: str, mask_info: dict[str, Any]) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise SchedulePreparationError(f"schedule output is create-only: {output}")
    plan.validate(
        record_batch_size=RECORD_BATCH_SIZE,
        position_budget=POSITION_BUDGET,
        sequence_tokens=SEQUENCE_TOKENS,
    )
    batch = torch.tensor(
        [step.batch_global_rows for step in plan.steps], dtype=torch.int32
    ).contiguous()
    draw_record = torch.tensor(
        [step.draw_record_slots for step in plan.steps], dtype=torch.int16
    ).contiguous()
    draw_position = torch.tensor(
        [step.draw_position_slots for step in plan.steps], dtype=torch.int16
    ).contiguous()
    replacement = torch.tensor(
        [step.used_replacement for step in plan.steps], dtype=torch.uint8
    ).contiguous()
    expected_shapes = {
        "batch_record_indices": (STEPS, RECORD_BATCH_SIZE),
        "draw_record_slots": (STEPS, POSITION_BUDGET),
        "draw_position_slots": (STEPS, POSITION_BUDGET),
        "used_replacement": (STEPS,),
    }
    tensors = {
        "batch_record_indices": batch,
        "draw_record_slots": draw_record,
        "draw_position_slots": draw_position,
        "used_replacement": replacement,
    }
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected_shapes[name]:
            raise SchedulePreparationError(f"{bank} {name} shape changed: {tuple(tensor.shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": SCHEMA,
        "bank": bank,
        "seed": str(SEED),
        "steps": str(STEPS),
        "record_batch_size": str(RECORD_BATCH_SIZE),
        "position_budget": str(POSITION_BUDGET),
        "sequence_tokens": str(SEQUENCE_TOKENS),
        "schedule_semantic_sha256": plan.semantic_sha256,
        "valid_mask_semantic_sha256": str(mask_info["tensor"]["semantic_sha256"]),
        "factory": "scripts.trr_p09.fixed_control_caller.inherited_schedule_steps",
        "replacement_dtype": "torch.uint8; false/true",
    }
    save_file(tensors, str(output), metadata=metadata)
    file_record = _file_record(output)
    return {
        "bank": bank,
        "file": file_record,
        "tensor_shapes": {name: list(shape) for name, shape in expected_shapes.items()},
        "tensor_dtypes": {name: str(tensor.dtype) for name, tensor in tensors.items()},
        "seed": SEED,
        "steps": STEPS,
        "record_batch_size": RECORD_BATCH_SIZE,
        "position_budget": POSITION_BUDGET,
        "sequence_tokens": SEQUENCE_TOKENS,
        "record_count": int(batch.max().item()) + 1 if batch.numel() else 0,
        "schedule_semantic_sha256": plan.semantic_sha256,
        "exposure": plan.exposure_summary(),
        "valid_mask": mask_info["tensor"],
        "create_only": True,
    }


def prepare(*, input_path: Path, output_root: Path) -> dict[str, Any]:
    mask, mask_info = _load_attention_mask(input_path)
    if output_root.exists() and not output_root.is_dir():
        raise SchedulePreparationError(f"output root is not a directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    plans: list[dict[str, Any]] = []
    for bank, record_count in (("B0", EXPECTED_B0_RECORDS), ("B1", EXPECTED_RECORDS)):
        bank_mask = mask[:record_count].contiguous()
        rows = tuple(range(record_count))
        plan = materialize_schedule_plan(
            bank_mask,
            rows,
            steps=STEPS,
            seed=SEED,
            record_batch_size=RECORD_BATCH_SIZE,
            position_budget=POSITION_BUDGET,
        )
        plans.append(
            _serialize_plan(
                plan,
                output=output_root / f"schedule-{bank.lower()}-seed{SEED}.safetensors",
                bank=bank,
                mask_info={
                    **mask_info,
                    "tensor": {
                        **mask_info["tensor"],
                        "shape": [record_count, SEQUENCE_TOKENS],
                        "bank_records": record_count,
                        "semantic_sha256": tensor_digest(bank_mask),
                        "post_bos_valid_positions": int(bank_mask[:, 1:].sum().item()),
                    },
                },
            )
        )
        del plan, bank_mask
    receipt = {
        "schema": "token-reconstruction.trr-p09-common-schedule-receipt.v1",
        "task_id": "TRR-P09",
        "status": "PASS_CPU_COMMON_SCHEDULES_PREPARED",
        "source_commit": _git_commit(),
        "source_script": _file_record(Path(__file__)),
        "command_scope": "CPU-only schedule preparation; no model, H, activation bank, fit, or evaluation truth",
        "input": mask_info,
        "factory": {
            "module": "scripts.trr_p09.fixed_control_caller",
            "function": "materialize_schedule_plan",
            "step_factory": "inherited_schedule_steps",
            "semantic_digest_contract": "SchedulePlan.semantic_sha256",
        },
        "schedule_contract": {
            "seed": SEED,
            "steps": STEPS,
            "record_batch_size": RECORD_BATCH_SIZE,
            "position_budget": POSITION_BUDGET,
            "sequence_tokens": SEQUENCE_TOKENS,
            "same_schedule_seed_for_b0_b1": True,
            "replacement_rng": "existing CPU torch.Generator/randperm/randint factory",
        },
        "banks": plans,
        "truth_boundary": {
            "model_loaded": False,
            "activation_bank_opened": False,
            "fit_started": False,
            "evaluation_truth_opened": False,
        },
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    receipt_path = output_root / "schedule-receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise SchedulePreparationError(f"receipt is create-only: {receipt_path}")
    # Keep the receipt free of a self-hash cycle. The caller binds this final
    # file record in task-local evidence after the run.
    receipt["receipt_path"] = str(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = prepare(input_path=args.input, output_root=args.output_root)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"prepare_common_schedules: ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
