#!/usr/bin/env python3
"""Bind the frozen 64-row diagnostics for the P09 B0 and B1 banks.

This is a read-only public-input audit.  It re-applies the signed
``TRR-0010|fixed-diagnostic|4010|{record_id}`` rule to the already prepared
record sidecar, verifies the selected rows against the preparation manifest,
and binds the actual input mask, position, and sequence digests.  It never
opens activations, a model, source plaintext, or evaluation truth, and it
does not rewrite either bank manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from safetensors import safe_open
import torch


TASK_ID = "TRR-P09"
DIAGNOSTIC_SEED = 4010
DIAGNOSTIC_PREFIX = "TRR-0010|fixed-diagnostic|4010|"
STRATUM_ORDER = (
    "alpaca_natural",
    "pile_natural",
    "finance_natural",
    "pile_controlled",
    "finance_controlled",
)
DIAGNOSTIC_QUOTAS = {
    "alpaca_natural": 32,
    "pile_natural": 16,
    "finance_natural": 10,
    "pile_controlled": 3,
    "finance_controlled": 3,
}
B0_ROWS = 1200
TOTAL_ROWS = 12000
SEQUENCE_TOKENS = 192


class DiagnosticBindingError(RuntimeError):
    """Raised when the immutable public inputs do not satisfy the rule."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = canonical_bytes({"shape": list(tensor.shape), "dtype": str(tensor.dtype)})
    return sha256_bytes(header + tensor.view(torch.uint8).numpy().tobytes(order="C"))


def h128_digest(value: torch.Tensor) -> str | None:
    tensor = value.detach().cpu().contiguous().reshape(-1)
    if tensor.numel() < 128:
        return None
    ids = tensor[:128].to(dtype=torch.int32).contiguous()
    return sha256_bytes(ids.numpy().tobytes(order="C"))


def require_file(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise DiagnosticBindingError(f"{label} is not a regular file: {path}")
    return path


def file_record(path: Path, *, label: str) -> dict[str, Any]:
    path = require_file(path, label=label)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(require_file(path, label=label).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosticBindingError(f"cannot read {label}: {path}") from exc


def _manifest_artifact(
    preparation_manifest: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    artifacts = preparation_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get(name), Mapping):
        raise DiagnosticBindingError(f"preparation manifest lacks artifacts.{name}")
    value = artifacts[name]
    if not isinstance(value.get("path"), str) or not isinstance(value.get("sha256"), str):
        raise DiagnosticBindingError(f"preparation artifact {name} is incomplete")
    return value


def verify_declared_artifact(path: Path, declared: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = file_record(path, label=label)
    if int(declared.get("bytes", -1)) != actual["bytes"]:
        raise DiagnosticBindingError(f"{label} byte count differs from preparation manifest")
    if str(declared.get("sha256")) != actual["sha256"]:
        raise DiagnosticBindingError(f"{label} SHA-256 differs from preparation manifest")
    return actual


def _row_rank(row: Mapping[str, Any]) -> str:
    try:
        record_id = str(row["record_id"])
    except KeyError as exc:
        raise DiagnosticBindingError("record sidecar row lacks record_id") from exc
    if not record_id:
        raise DiagnosticBindingError("record sidecar contains an empty record_id")
    return sha256_bytes((DIAGNOSTIC_PREFIX + record_id).encode("utf-8"))


def choose_rows(rows: Sequence[Mapping[str, Any]], *, limit: int) -> dict[str, Any]:
    if limit < B0_ROWS or limit > len(rows):
        raise DiagnosticBindingError(f"diagnostic bank limit is invalid: {limit}")
    seen_ids: set[str] = set()
    by_stratum: dict[str, list[tuple[str, int, Mapping[str, Any]]]] = {
        stratum: [] for stratum in STRATUM_ORDER
    }
    for index, row in enumerate(rows[:limit]):
        if not isinstance(row, Mapping):
            raise DiagnosticBindingError(f"record row {index} is not an object")
        if int(row.get("global_row", -1)) != index:
            raise DiagnosticBindingError(f"record row {index} has a mismatched global_row")
        record_id = str(row.get("record_id", ""))
        if record_id in seen_ids:
            raise DiagnosticBindingError(f"duplicate record_id in bank: {record_id}")
        seen_ids.add(record_id)
        stratum = str(row.get("stratum", ""))
        if stratum not in by_stratum:
            raise DiagnosticBindingError(f"unknown diagnostic stratum at row {index}: {stratum}")
        by_stratum[stratum].append((_row_rank(row), index, row))

    selected: list[tuple[str, int, Mapping[str, Any]]] = []
    for stratum in STRATUM_ORDER:
        ranked = sorted(by_stratum[stratum], key=lambda item: (item[0], item[1]))
        quota = DIAGNOSTIC_QUOTAS[stratum]
        if len(ranked) < quota:
            raise DiagnosticBindingError(f"diagnostic stratum {stratum} is short")
        selected.extend(ranked[:quota])
    if len(selected) != 64:
        raise DiagnosticBindingError(f"diagnostic selection has {len(selected)} rows")
    sorted_indices = sorted(item[1] for item in selected)
    return {
        "selection_order": selected,
        "global_indices": sorted_indices,
        "global_indices_sha256": sha256_bytes(canonical_bytes(sorted_indices)),
        "stratum_quotas": dict(DIAGNOSTIC_QUOTAS),
    }


def _mask_is_prefix(mask: torch.Tensor) -> int:
    values = [int(value) for value in mask.tolist()]
    first_zero = next((index for index, value in enumerate(values) if value == 0), len(values))
    if any(value != 1 for value in values[:first_zero]) or any(value != 0 for value in values[first_zero:]):
        raise DiagnosticBindingError("attention_mask is not a contiguous active prefix")
    return first_zero


def bind_bank_rows(
    rows: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
    *,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    bank: str,
) -> dict[str, Any]:
    descriptors: list[dict[str, Any]] = []
    for rank, index, row in selected["selection_order"]:
        mask = attention_mask[index].contiguous()
        positions = position_ids[index].contiguous()
        tokens = token_ids[index].contiguous()
        active = _mask_is_prefix(mask)
        if active != int(row.get("target_full_token_count", -1)):
            raise DiagnosticBindingError(f"{bank} row {index} active length differs from sidecar")
        if active <= 0 or active > SEQUENCE_TOKENS:
            raise DiagnosticBindingError(f"{bank} row {index} has invalid active length")
        expected_positions = torch.zeros_like(positions)
        expected_positions[:active] = torch.arange(active, dtype=positions.dtype)
        if not torch.equal(positions, expected_positions):
            raise DiagnosticBindingError(f"{bank} row {index} position IDs differ from the bound convention")
        input_h128 = h128_digest(tokens[:active])
        # The preparation sidecar's sequence_h128_sha256 is a source-sequence
        # identity for natural rows.  A natural row may be clipped below 128
        # tokens for the fitting bank, so its source H128 is intentionally not
        # the digest of the shorter emitted tensor.  Once the emitted row is
        # at least 128 tokens, clipping preserves the first 128 IDs and the
        # two namespaces must agree.  Controlled rows store the constructed
        # sequence identity and obey the same check whenever H128 exists.
        source_h128 = row.get("sequence_h128_sha256")
        if input_h128 is not None and input_h128 != source_h128:
            raise DiagnosticBindingError(f"{bank} row {index} input H128 digest differs from sidecar")
        descriptors.append(
            {
                "global_row": int(index),
                "record_id": str(row["record_id"]),
                "source_record_id": str(row.get("source_record_id", row["record_id"])),
                "stratum": str(row["stratum"]),
                "synthetic": bool(row.get("synthetic", False)),
                "target_full_token_count": int(active),
                "target_post_bos_token_count": int(row.get("target_post_bos_token_count", active - 1)),
                "rank_sha256": rank,
                "sequence_sha256": tensor_digest(tokens[:active]),
                "source_h128_sha256": source_h128,
                "input_sequence_h128_sha256": input_h128,
                "attention_mask_sha256": tensor_digest(mask),
                "position_ids_sha256": tensor_digest(positions),
            }
        )
    descriptor_digest = sha256_bytes(canonical_bytes(descriptors))
    return {
        "bank": bank,
        "record_count": len(descriptors),
        "global_indices": list(selected["global_indices"]),
        "global_indices_sha256": str(selected["global_indices_sha256"]),
        "selection_order": "stratum_order_then_rank_sha256_then_global_row",
        "rows": descriptors,
        "rows_sha256": descriptor_digest,
        "mask_digest_namespace": "tensor_digest(shape,dtype,bytes) over uint8 attention_mask[192]",
        "sequence_digest_namespace": "tensor_digest(shape,dtype,bytes) over active int32 token_ids",
        "source_h128_namespace": "preparation sidecar source-sequence H128 identity; may remain non-null when the emitted fitting row is clipped below 128",
        "input_h128_namespace": "SHA-256 over first 128 active int32 token IDs, little-endian C-order; null below 128 emitted tokens",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records_path = require_file(args.records, label="records sidecar")
    inputs_path = require_file(args.inputs, label="public input tensor file")
    preparation_path = require_file(args.preparation_manifest, label="preparation manifest")
    bank_manifest_path = require_file(args.bank_manifest, label="qualified bank manifest")
    capture_receipt_path = require_file(args.capture_receipt, label="capture receipt")
    output_path = args.output.expanduser().resolve()
    if output_path.exists() or output_path.is_symlink():
        raise DiagnosticBindingError(f"output is create-only and already exists: {output_path}")

    preparation = load_json(preparation_path, label="preparation manifest")
    if preparation.get("schema") != "token-reconstruction.trr-p09-stage1-public-inputs.v1":
        raise DiagnosticBindingError("preparation manifest schema changed")
    if preparation.get("status") != "CPU_INPUTS_COMPILED_NO_ACTIVATIONS":
        raise DiagnosticBindingError("preparation manifest is not the public CPU input artifact")
    declared_records = _manifest_artifact(preparation, "records")
    declared_inputs = _manifest_artifact(preparation, "inputs")
    records_file = verify_declared_artifact(records_path, declared_records, label="records sidecar")
    inputs_file = verify_declared_artifact(inputs_path, declared_inputs, label="public input tensors")
    bank_file = file_record(bank_manifest_path, label="qualified bank manifest")
    bank_manifest = load_json(bank_manifest_path, label="qualified bank manifest")
    if bank_manifest.get("status") != "STAGE1_PUBLIC_BASE_CAPTURE_COMPLETE_NO_TRUTH":
        raise DiagnosticBindingError("qualified bank manifest status changed")
    capture_file = file_record(capture_receipt_path, label="capture receipt")
    capture_receipt = load_json(capture_receipt_path, label="capture receipt")
    if capture_receipt.get("status") != "CAPTURE_COMPLETE_NO_TRUTH":
        raise DiagnosticBindingError("capture receipt status changed")
    capture_geometry = capture_receipt.get("geometry")
    if not isinstance(capture_geometry, Mapping) or int(capture_geometry.get("records", -1)) != 10800 or int(capture_geometry.get("sequence_tokens", -1)) != SEQUENCE_TOKENS:
        raise DiagnosticBindingError("capture receipt geometry is not the signed 10,800-row T=192 capture")
    geometry = bank_manifest.get("geometry")
    if not isinstance(geometry, Mapping) or int(geometry.get("sequence_tokens", -1)) != SEQUENCE_TOKENS:
        raise DiagnosticBindingError("qualified bank geometry is not T=192")
    exposure = bank_manifest.get("exposure_manifest")
    if not isinstance(exposure, Mapping) or int(exposure.get("loader_batch_count", -1)) != 1350:
        raise DiagnosticBindingError("qualified bank capture exposure is not the expected 1,350 batches")
    if int(exposure.get("loader_batch_count")) * 8 != 10800:
        raise DiagnosticBindingError("capture batch count does not cover the 10,800 new rows")

    records = load_json(records_path, label="records sidecar")
    if not isinstance(records, list) or len(records) != TOTAL_ROWS:
        raise DiagnosticBindingError("prepared records must contain exactly 12,000 rows")
    for index, row in enumerate(records):
        if not isinstance(row, Mapping) or int(row.get("global_row", -1)) != index:
            raise DiagnosticBindingError(f"prepared records are not globally ordered at row {index}")

    with safe_open(str(inputs_path), framework="pt", device="cpu") as handle:
        expected_keys = {"token_ids", "attention_mask", "position_ids"}
        if set(handle.keys()) != expected_keys:
            raise DiagnosticBindingError(f"public input keys changed: {sorted(handle.keys())}")
        token_ids = handle.get_tensor("token_ids").contiguous()
        attention_mask = handle.get_tensor("attention_mask").contiguous()
        position_ids = handle.get_tensor("position_ids").contiguous()
    expected_shape = (TOTAL_ROWS, SEQUENCE_TOKENS)
    if tuple(token_ids.shape) != expected_shape or tuple(attention_mask.shape) != expected_shape or tuple(position_ids.shape) != expected_shape:
        raise DiagnosticBindingError("public input tensor geometry changed")
    if token_ids.dtype != torch.int32 or attention_mask.dtype != torch.uint8 or position_ids.dtype != torch.int64:
        raise DiagnosticBindingError("public input tensor dtypes changed")

    current = choose_rows(records, limit=B0_ROWS)
    expanded = choose_rows(records, limit=TOTAL_ROWS)
    declared_diag = preparation.get("diagnostic")
    if not isinstance(declared_diag, Mapping):
        raise DiagnosticBindingError("preparation manifest lacks diagnostic metadata")
    for bank, selected in (("B0", current), ("B1", expanded)):
        declared = declared_diag.get("current_bank" if bank == "B0" else "expanded_bank")
        if not isinstance(declared, Mapping):
            raise DiagnosticBindingError(f"preparation manifest lacks {bank} diagnostic")
        if list(declared.get("indices", [])) != list(selected["global_indices"]):
            raise DiagnosticBindingError(f"{bank} diagnostic indices differ from prepared manifest")
        if declared.get("indices_sha256") != selected["global_indices_sha256"]:
            raise DiagnosticBindingError(f"{bank} diagnostic index digest differs from prepared manifest")
    current_binding = bind_bank_rows(records, current, token_ids=token_ids, attention_mask=attention_mask, position_ids=position_ids, bank="B0")
    expanded_binding = bind_bank_rows(records, expanded, token_ids=token_ids, attention_mask=attention_mask, position_ids=position_ids, bank="B1")
    output = {
        "schema": "token-reconstruction.trr-p09-fixed-diagnostic-binding.v1",
        "task_id": TASK_ID,
        "status": "PASS_FIXED_DIAGNOSTIC64_BOUND_PUBLIC_INPUTS_ONLY",
        "rule": {
            "seed": DIAGNOSTIC_SEED,
            "prefix": DIAGNOSTIC_PREFIX,
            "stratum_order": list(STRATUM_ORDER),
            "stratum_quotas": dict(DIAGNOSTIC_QUOTAS),
            "selection": "ascending rank SHA-256 within each stratum; no model score or evaluation answer",
            "shared_across": "fixed_and_directional_arms_within_each_bank_only",
        },
        "source_bindings": {
            "preparation_manifest": {**file_record(preparation_path, label="preparation manifest"), "schema": preparation.get("schema")},
            "records": records_file,
            "inputs": inputs_file,
            "qualified_bank_manifest": {**bank_file, "status": bank_manifest.get("status"), "geometry": geometry},
            "capture_receipt": {**capture_file, "status": capture_receipt.get("status"), "geometry": capture_geometry, "shard_counts": capture_receipt.get("shard_counts")},
            "preparation_declared_diagnostic": {
                "current_indices_sha256": preparation["diagnostic"]["current_bank"]["indices_sha256"],
                "expanded_indices_sha256": preparation["diagnostic"]["expanded_bank"]["indices_sha256"],
            },
        },
        "banks": {"B0": current_binding, "B1": expanded_binding},
        "capture_exposure": {
            "qualified_capture_forward_batch_count": int(exposure["loader_batch_count"]),
            "complete_records_per_forward_batch": 8,
            "captured_new_records": int(capture_geometry["records"]),
            "sequence_tokens": int(capture_geometry["sequence_tokens"]),
            "basis": "qualified bank exposure_manifest.loader_batch_count; 10,800 new rows / 8 complete B8x192 rows per forward",
            "bank_manifest_public_forward_count_field": exposure.get("public_forward_count"),
        },
        "verification": {
            "record_count": len(records),
            "b0_prefix_records": B0_ROWS,
            "sequence_tokens": SEQUENCE_TOKENS,
            "public_fitting_token_ids_read": True,
            "attention_masks_read": True,
            "position_ids_read": True,
            "model_loaded": False,
            "activations_read": False,
            "source_plaintext_read": False,
            "evaluation_truth_opened": False,
            "published_bank_manifests_modified": False,
        },
        "reproduction": {
            "command": [
                "PYTHONPATH=.:src:scripts",
                "python3",
                "scripts/trr_p09/bind_fixed_diagnostic.py",
                "--records",
                str(records_path),
                "--inputs",
                str(inputs_path),
                "--preparation-manifest",
                str(preparation_path),
                "--bank-manifest",
                str(bank_manifest_path),
                "--capture-receipt",
                str(capture_receipt_path),
                "--output",
                str(output_path),
            ],
            "code_scope": "CPU metadata and public fitting token/mask/position inputs only",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(output_path), "B0": current_binding["global_indices_sha256"], "B1": expanded_binding["global_indices_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticBindingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
