#!/usr/bin/env python3
"""Build the small TRR-P12 qualification-only target source bundle.

The source is the already-opened TRR-0004 public fitting activation artifact.
Only its token IDs and attention masks are read.  No source text, activations,
P03 material, fresh evaluation rows, or target weights are selected or copied.
The output root is create-only and is intended to remain under ignored
``outputs/`` storage.  ``--full128-r2`` is a qualification compatibility
fixture: it takes the first eight already-opened public fitting rows whose
source mask has at least 128 active positions, then repeats those rows to the
fixed trainer shapes.  The length filter is not a scientific source rule.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


TASK_ID = "TRR-P12"
FIXTURE_SCHEMA = "token-reconstruction.trr-p12-qualification-fixture.v1"
SOURCE_SCHEMA = "token-reconstruction.trr-p12-target-source-bundle.v1"
SOURCE_TASK_ID = "TRR-0004"
BOS_TOKEN_ID = 128000
VOCAB_SIZE = 128256
SOURCE_ROWS = 1200
SOURCE_SEQUENCE_LENGTH = 192
SEQUENCE_LENGTH = 128
TRAIN_ROWS = 256
VALIDATION_ROWS = 64
VALIDATION_SEED_ROWS = 8
SOURCE_SHA256 = "d1c78fcf1acc91b57d51355ee11f267bf4c12f1bc7d5160164b3b6ea11b45344"
SOURCE_RECORDS_SHA256 = "34e10ddc7c502b30822730b84c6f3a48b6285888b8ef92fc9e1aef65ecca29df"
SOURCE_ROLE = "public fit activations and labels; nested selectors included"

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
WORKTREE_ROOT = SCRIPT_ROOT
DEFAULT_SOURCE = WORKTREE_ROOT.parent / "TRR-0004" / "outputs" / "TRR-0004" / "public_activation_v2" / "train_large_cut4.safetensors"
DEFAULT_STATUS = WORKTREE_ROOT.parent / "TRR-0004" / "experiments" / "TRR-0004" / "status.json"
DEFAULT_RECORDS = WORKTREE_ROOT.parent / "TRR-0004" / "outputs" / "TRR-0004" / "public_activation_v2" / "train_large_records.json"
DEFAULT_OUTPUT = WORKTREE_ROOT / "outputs" / "TRR-P12" / "qualification-fixture-r1"
DEFAULT_OUTPUT_R2 = WORKTREE_ROOT / "outputs" / "TRR-P12" / "qualification-fixture-r2"


class FixtureError(RuntimeError):
    """Raised when a public fixture binding cannot be verified."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_binding(path: Path, *, label: str, expected_sha256: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FixtureError(f"{label} must be a regular file: {path}")
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise FixtureError(f"{label} SHA-256 changed")
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": actual,
        "readonly": True,
    }


def json_file(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FixtureError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixtureError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise FixtureError(f"{label} must contain an object")
    return value


def resolve_path(value: str, *, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def verify_p04_public_fit(source: Path, status_path: Path, records_path: Path) -> dict[str, Any]:
    source_binding = file_binding(source, label="P04 train_large source", expected_sha256=SOURCE_SHA256)
    records_binding = file_binding(
        records_path,
        label="P04 train_large records",
        expected_sha256=SOURCE_RECORDS_SHA256,
    )
    status = json_file(status_path, label="P04 status")
    if status.get("task_id") != SOURCE_TASK_ID:
        raise FixtureError("source status is not the TRR-0004 record")
    preparation = status.get("public_activation_preparation")
    if not isinstance(preparation, dict) or preparation.get("status") != "COMPLETE_NO_CONFIRMATION":
        raise FixtureError("P04 public activation preparation is not complete")
    successful = preparation.get("successful_run")
    outputs = successful.get("outputs") if isinstance(successful, dict) else None
    train_large = outputs.get("train_large") if isinstance(outputs, dict) else None
    if not isinstance(train_large, dict):
        raise FixtureError("P04 successful public train_large output is missing")
    if train_large.get("sha256") != SOURCE_SHA256 or int(train_large.get("bytes", -1)) != source.stat().st_size:
        raise FixtureError("P04 status does not bind the expected train_large artifact")
    if train_large.get("role") != SOURCE_ROLE:
        raise FixtureError("P04 train_large role is not the registered public fitting role")
    recorded_path = Path(str(train_large.get("path", ""))).resolve()
    if recorded_path != source.resolve():
        raise FixtureError("P04 status points at a different train_large path")
    access = status.get("access_and_separation")
    if not isinstance(access, dict):
        raise FixtureError("P04 access and separation record is missing")
    required_false = (
        "public_activation_evaluator_private_truth_accessed",
        "public_activation_target_weights_accessed",
        "trr0003_panel_reserved_for_new_heldout_evaluation",
        "trr0004_source_truth_accessed",
        "trr_p01_worktree_or_outputs_touched",
    )
    if any(access.get(key) is not False for key in required_false):
        raise FixtureError("P04 access record does not establish public fitting separation")
    binding = preparation.get("source_binding")
    if not isinstance(binding, dict):
        raise FixtureError("P04 source binding is missing")
    expected_binding = {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "model_revision": "9213176726f574b556790deb65791e0c5aa438b6",
        "model_weights_sha256": "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f",
        "dataset_arrow_sha256": "f45103036ed651f4c06d0a3c3e0fb7d53acb3074ed5c8e804a69c1efc1cea794",
        "dataset_info_sha256": "25fae40b04375187311d88f1c4cb9f7c262fae1cc3212354c3075b6ad5db4f67",
        "split_plan_sha256": "29ca79db0f29fd5ec1e3769b5427ac1496b59ef7f88e7e921e41c13b1e07eb2e",
        "cut_depth": 4,
    }
    if any(binding.get(key) != value for key, value in expected_binding.items()):
        raise FixtureError("P04 public source binding differs from the registered values")
    return {
        "source_artifact": source_binding,
        "source_records": records_binding,
        "p04_status": file_binding(status_path, label="P04 status"),
        "source_task_id": SOURCE_TASK_ID,
        "source_role": SOURCE_ROLE,
        "preparation_status": preparation["status"],
        "public_fitting_data": True,
        "fresh_evaluation_data": False,
        "p03_data": False,
        "access_and_separation": {key: access[key] for key in required_false},
        "source_binding": expected_binding,
    }


def validate_pair(ids: torch.Tensor, mask: torch.Tensor, *, label: str, rows: int) -> dict[str, int]:
    if tuple(ids.shape) != (rows, SEQUENCE_LENGTH):
        raise FixtureError(f"{label} IDs have the wrong shape")
    if tuple(mask.shape) != (rows, SEQUENCE_LENGTH):
        raise FixtureError(f"{label} masks have the wrong shape")
    if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise FixtureError(f"{label} IDs are not integer tensors")
    if mask.dtype not in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise FixtureError(f"{label} masks are not boolean/integer tensors")
    ids_long = ids.to(dtype=torch.long)
    mask_bool = mask.to(dtype=torch.bool)
    if bool(ids_long.lt(0).any().item()) or bool(ids_long.ge(VOCAB_SIZE).any().item()):
        raise FixtureError(f"{label} IDs contain an out-of-vocabulary value")
    if bool(ids_long[:, 0].ne(BOS_TOKEN_ID).any().item()):
        raise FixtureError(f"{label} does not retain the fixed BOS token")
    mask_values = mask.to(dtype=torch.int64)
    if bool(mask_values.lt(0).any().item()) or bool(mask_values.gt(1).any().item()):
        raise FixtureError(f"{label} mask is not binary")
    if bool((~mask_bool[:, 0]).any().item()):
        raise FixtureError(f"{label} masks BOS")
    if bool((~mask_bool[:, :-1] & mask_bool[:, 1:]).any().item()):
        raise FixtureError(f"{label} is not right padded")
    active = mask_bool.sum(dim=1)
    if bool(active.lt(2).any().item()):
        raise FixtureError(f"{label} contains a sequence shorter than two active tokens")
    return {
        "rows": rows,
        "sequence_length": SEQUENCE_LENGTH,
        "active_tokens_min": int(active.min().item()),
        "active_tokens_max": int(active.max().item()),
    }


def load_public_token_fields(source: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    try:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            expected_keys = {
                "activations",
                "attention_mask",
                "position_ids",
                "post_bos_selector_large",
                "post_bos_selector_small",
                "token_ids",
            }
            if keys != expected_keys:
                raise FixtureError("P04 train_large keys differ from the registered public artifact")
            token_ids = handle.get_tensor("token_ids")
            attention_mask = handle.get_tensor("attention_mask")
    except FixtureError:
        raise
    except Exception as exc:
        raise FixtureError("cannot read the P04 public token fields") from exc
    if tuple(token_ids.shape) != (SOURCE_ROWS, SOURCE_SEQUENCE_LENGTH):
        raise FixtureError("P04 token_ids geometry differs from the registered train_large artifact")
    if tuple(attention_mask.shape) != (SOURCE_ROWS, SOURCE_SEQUENCE_LENGTH):
        raise FixtureError("P04 attention_mask geometry differs from the registered train_large artifact")
    if token_ids.dtype != torch.int32 or attention_mask.dtype != torch.uint8:
        raise FixtureError("P04 token field dtypes differ from the registered public artifact")
    return token_ids, attention_mask, {
        "source_keys_read": ["attention_mask", "token_ids"],
        "source_keys_not_read": [
            "activations",
            "position_ids",
            "post_bos_selector_large",
            "post_bos_selector_small",
        ],
        "source_geometry": [SOURCE_ROWS, SOURCE_SEQUENCE_LENGTH],
        "source_token_dtype": str(token_ids.dtype),
        "source_mask_dtype": str(attention_mask.dtype),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FixtureError(f"refusing to overwrite artifact: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ordinal_digest(indices: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in indices.tolist():
        digest.update(f"{int(value)}\n".encode("ascii"))
    return digest.hexdigest()


def build_fixture(
    *,
    source: Path,
    status_path: Path,
    records_path: Path,
    output_root: Path,
    full128_r2: bool = False,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FixtureError(f"output root already exists; create-only policy refuses it: {output_root}")
    provenance = verify_p04_public_fit(source, status_path, records_path)
    token_ids, attention_mask, source_fields = load_public_token_fields(source)
    if full128_r2:
        source_active_lengths = attention_mask.to(dtype=torch.bool).sum(dim=1)
        eligible = torch.nonzero(source_active_lengths.ge(SEQUENCE_LENGTH), as_tuple=False).flatten()
        if int(eligible.numel()) < VALIDATION_SEED_ROWS:
            raise FixtureError("P04 public fitting artifact has fewer than eight rows with 128 active tokens")
        selected = eligible[:VALIDATION_SEED_ROWS]
        selected_ids = token_ids.index_select(0, selected)[:, :SEQUENCE_LENGTH].contiguous()
        selected_mask = attention_mask.index_select(0, selected)[:, :SEQUENCE_LENGTH].contiguous()
        train_ids = selected_ids.repeat((TRAIN_ROWS // VALIDATION_SEED_ROWS, 1)).contiguous()
        train_mask = selected_mask.repeat((TRAIN_ROWS // VALIDATION_SEED_ROWS, 1)).contiguous()
        validation_ids = selected_ids.repeat((VALIDATION_ROWS // VALIDATION_SEED_ROWS, 1)).contiguous()
        validation_mask = selected_mask.repeat((VALIDATION_ROWS // VALIDATION_SEED_ROWS, 1)).contiguous()
        selection_metadata = {
            "mode": "qualification_full128_r2",
            "source_rows_train": "repeat the selected eight rows 32 times to 256",
            "source_rows_validation_seed": "the same selected eight rows, repeated eight times to 64",
            "validation_rows": "repeat the first eight eligible public fitting rows eight times to 64 rows",
            "eligibility_rule": "source attention_mask sum >= 128; choose first eight in existing P04 order",
            "selection_reason": "qualification geometry compatibility only; no outcome or source-content criterion",
            "selected_source_row_ordinal_digest": _ordinal_digest(selected),
            "eligible_min_active_tokens": SEQUENCE_LENGTH,
            "fresh_row_selection": False,
            "fresh_evaluation_data": False,
            "public_fitting_fixture": True,
            "source_text_read": False,
        }
    else:
        train_ids = token_ids[:TRAIN_ROWS, :SEQUENCE_LENGTH].contiguous()
        train_mask = attention_mask[:TRAIN_ROWS, :SEQUENCE_LENGTH].contiguous()
        validation_ids = token_ids[:VALIDATION_SEED_ROWS, :SEQUENCE_LENGTH].repeat((8, 1)).contiguous()
        validation_mask = attention_mask[:VALIDATION_SEED_ROWS, :SEQUENCE_LENGTH].repeat((8, 1)).contiguous()
        selection_metadata = {
            "source_rows_train": [0, TRAIN_ROWS],
            "source_rows_validation_seed": [0, VALIDATION_SEED_ROWS],
            "validation_rows": "repeat the first 8 already-opened public fitting rows eight times to 64 rows",
            "trim_rule": "source token_ids and attention_mask[:, :128]",
            "fresh_row_selection": False,
            "fresh_evaluation_data": False,
            "public_fitting_fixture": True,
            "source_text_read": False,
        }
    train_stats = validate_pair(train_ids, train_mask, label="train", rows=TRAIN_ROWS)
    validation_stats = validate_pair(validation_ids, validation_mask, label="validation", rows=VALIDATION_ROWS)

    output_root.mkdir(parents=True, exist_ok=False)
    train_path = output_root / "train.safetensors"
    validation_path = output_root / "validation.safetensors"
    save_file(
        {"train_input_ids": train_ids, "train_attention_mask": train_mask},
        str(train_path),
        metadata={
            "schema": SOURCE_SCHEMA,
            "task_id": TASK_ID,
            "split": "train",
            "purpose": "qualification_only",
        },
    )
    save_file(
        {"validation_input_ids": validation_ids, "validation_attention_mask": validation_mask},
        str(validation_path),
        metadata={
            "schema": SOURCE_SCHEMA,
            "task_id": TASK_ID,
            "split": "validation",
            "purpose": "qualification_only",
        },
    )
    train_binding = file_binding(train_path, label="fixture train tensor")
    validation_binding = file_binding(validation_path, label="fixture validation tensor")
    source_bundle = {
        "schema": SOURCE_SCHEMA,
        "task_id": TASK_ID,
        "status": "QUALIFICATION_FIXTURE_ONLY",
        "created_utc": utc_now(),
        "train_tensor": {
            "path": train_path.name,
            "bytes": train_binding["bytes"],
            "sha256": train_binding["sha256"],
        },
        "validation_tensor": {
            "path": validation_path.name,
            "bytes": validation_binding["bytes"],
            "sha256": validation_binding["sha256"],
        },
        "contract": {
            "sequence_length": SEQUENCE_LENGTH,
            "vocab_size": VOCAB_SIZE,
            "bos_token_id": BOS_TOKEN_ID,
            "train_shape": [TRAIN_ROWS, SEQUENCE_LENGTH],
            "validation_shape": [VALIDATION_ROWS, SEQUENCE_LENGTH],
            "validation_batching_contract": "trainer evaluates fixed validation rows in batch size 2",
        },
        "selection": selection_metadata,
        "provenance": provenance,
        "source_fields": source_fields,
        "access_policy": {
            "target_weights": "not accessed",
            "evaluator_private_truth": "not accessed",
            "decoder_reconstruction_input": "not an input; target evaluator-only",
            "p03_or_fresh_evaluation_data": "not used",
        },
        "tensor_statistics": {
            "train": train_stats,
            "validation": validation_stats,
            "ids_dtype": str(train_ids.dtype),
            "mask_dtype": str(train_mask.dtype),
        },
    }
    source_bundle_path = output_root / "source-bundle.json"
    write_json(source_bundle_path, source_bundle)
    source_bundle_binding = file_binding(source_bundle_path, label="source bundle")
    manifest = {
        "schema": FIXTURE_SCHEMA,
        "task_id": TASK_ID,
        "status": "CREATED_PUBLIC_FITTING_QUALIFICATION_FIXTURE",
        "created_utc": source_bundle["created_utc"],
        "source_bundle": source_bundle_binding,
        "train_tensor": train_binding,
        "validation_tensor": validation_binding,
        "provenance": provenance,
        "selection": source_bundle["selection"],
        "contract": source_bundle["contract"],
        "source_fields": source_fields,
        "access_policy": source_bundle["access_policy"],
        "tensor_statistics": source_bundle["tensor_statistics"],
        "create_only": True,
        "ignored_output_root": True,
    }
    manifest_path = output_root / "manifest.json"
    write_json(manifest_path, manifest)
    return {
        "status": manifest["status"],
        "output_root": str(output_root.resolve()),
        "source_bundle": source_bundle_binding,
        "manifest": file_binding(manifest_path, label="fixture manifest"),
        "train_tensor": train_binding,
        "validation_tensor": validation_binding,
        "source_artifact": provenance["source_artifact"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--full128-r2",
        action="store_true",
        help="select the first eight public fitting rows with >=128 active source positions and repeat them",
    )
    args = parser.parse_args()
    source = resolve_path(str(args.source), root=WORKTREE_ROOT)
    status_path = resolve_path(str(args.status), root=WORKTREE_ROOT)
    records_path = resolve_path(str(args.records), root=WORKTREE_ROOT)
    output_value = DEFAULT_OUTPUT_R2 if args.full128_r2 and args.output_root == DEFAULT_OUTPUT else args.output_root
    output_root = resolve_path(str(output_value), root=WORKTREE_ROOT)
    try:
        result = build_fixture(
            source=source,
            status_path=status_path,
            records_path=records_path,
            output_root=output_root,
            full128_r2=args.full128_r2,
        )
    except (FixtureError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
