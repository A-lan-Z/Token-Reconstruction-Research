#!/usr/bin/env python3
"""Recover the common public B0 frequency map from the exact fitting payload.

The historical compact support file is a documented runtime artifact and is
not required for this derivation.  This bounded recovery opens only
``token_ids`` and ``attention_mask`` from the hash-bound B0 fitting payload,
derives the dense post-BOS frequency vector, and writes a fresh compact
safetensors file containing the dense vector plus its sorted positive support.
The old path and expected identity remain in the receipt for provenance; the
old file is never used as a source or fallback.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SCHEMA = "token-reconstruction.trr-p11-common-frequency-recovery.v1"
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
VOCAB_SIZE = 128256


class RecoveryError(RuntimeError):
    """Raised when the public B0 recovery contract is violated."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise RecoveryError(f"{label} must be an existing regular file: {candidate}")
    return candidate.resolve()


def file_record(path: Path, *, label: str, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    actual_path = regular_file(path, label=label)
    record = {"path": str(actual_path), "bytes": int(actual_path.stat().st_size), "sha256": sha256_file(actual_path)}
    if expected is not None:
        for key in ("bytes", "sha256"):
            if expected.get(key) is not None and record[key] != expected[key]:
                raise RecoveryError(f"{label} {key} changed: expected {expected[key]}, got {record[key]}")
    return record


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def prefixed_tensor_digest(value: torch.Tensor, *, prefix: bytes) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(prefix)
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def support_digest(ids: torch.Tensor, counts: torch.Tensor) -> str:
    digest = hashlib.sha256(b"trr0010-support-v1\0")
    digest.update(prefixed_tensor_digest(ids, prefix=b"trr0010-support-ids\0").encode("ascii"))
    digest.update(prefixed_tensor_digest(counts, prefix=b"trr0010-support-counts\0").encode("ascii"))
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"cannot parse config: {path}") from exc
    if not isinstance(value, Mapping):
        raise RecoveryError("config must be a JSON object")
    return value


def _resolve(config_path: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RecoveryError(f"{label} path is absent")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path


def _expected_reference(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("common_frequency_reference")
    if not isinstance(value, Mapping):
        raise RecoveryError("common_frequency_reference is absent")
    for key in ("support_digest_trr0010", "support_ids_tensor_sha256", "support_counts_tensor_sha256", "frequency_vector_tensor_sha256"):
        if not isinstance(value.get(key), str) or len(value[key]) != 64:
            raise RecoveryError(f"common reference expected identity is absent: {key}")
    return value


def derive(config_path: Path, output_path: Path, receipt_path: Path) -> dict[str, Any]:
    config_path = regular_file(config_path, label="recovery config")
    config = _json(config_path)
    if config.get("schema") != "token-reconstruction.trr-p11-public-support-histogram.v1":
        raise RecoveryError("unsupported support histogram contract")
    banks = config.get("banks")
    if not isinstance(banks, Mapping) or not isinstance(banks.get("B0"), Mapping):
        raise RecoveryError("B0 binding is absent")
    b0_spec = banks["B0"]
    b0_path = _resolve(config_path, b0_spec.get("path"), label="B0 payload")
    source = file_record(b0_path, label="B0 payload", expected=b0_spec)
    reference = _expected_reference(config)
    historical_path = _resolve(config_path, reference.get("path"), label="historical common reference")
    historical_exists = historical_path.is_file() and not historical_path.is_symlink()
    expected_records = int(b0_spec.get("records", 0))
    expected_width = int(b0_spec.get("width", 0))
    expected_post_bos = int(b0_spec.get("post_bos_positions", -1))
    if expected_records <= 0 or expected_width <= 0 or expected_post_bos < 0:
        raise RecoveryError("B0 geometry binding is incomplete")
    try:
        with safe_open(str(b0_path), framework="pt", device="cpu") as handle:
            keys = sorted(handle.keys())
            for required in ("token_ids", "attention_mask"):
                if required not in keys:
                    raise RecoveryError(f"B0 payload lacks {required}")
            token_ids = handle.get_tensor("token_ids").detach().cpu().contiguous()
            attention_mask = handle.get_tensor("attention_mask").detach().cpu().contiguous()
    except RecoveryError:
        raise
    except Exception as exc:
        raise RecoveryError("cannot read B0 token/mask tensors") from exc
    if tuple(token_ids.shape) != (expected_records, expected_width) or tuple(attention_mask.shape) != tuple(token_ids.shape):
        raise RecoveryError("B0 token/mask geometry differs from contract")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise RecoveryError("B0 token_ids dtype must be int32 or int64")
    if attention_mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int32, torch.int64):
        raise RecoveryError("B0 attention_mask dtype is not integer-like")
    if token_ids[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise RecoveryError("B0 rows do not begin with the declared BOS")
    if attention_mask.lt(0).any().item() or attention_mask.gt(1).any().item():
        raise RecoveryError("B0 attention_mask is not binary")
    mask = attention_mask.to(dtype=torch.bool)
    active = mask.sum(dim=1)
    if active.le(1).any().item():
        raise RecoveryError("B0 contains a row without a post-BOS token")
    for index in range(expected_records):
        row_active = int(active[index].item())
        if not mask[index, :row_active].all().item() or mask[index, row_active:].any().item():
            raise RecoveryError(f"B0 mask is not a contiguous prefix at row {index}")
        if not token_ids[index, row_active:].eq(PAD_TOKEN_ID).all().item():
            raise RecoveryError(f"B0 padding token identity changed at row {index}")
        if token_ids[index, 1:row_active].eq(PAD_TOKEN_ID).any().item():
            raise RecoveryError(f"B0 active token region contains padding at row {index}")
    post_bos = int(mask[:, 1:].sum().item())
    if post_bos != expected_post_bos:
        raise RecoveryError(f"B0 post-BOS positions {post_bos} != {expected_post_bos}")
    labels = token_ids[:, 1:][mask[:, 1:]].to(dtype=torch.int64).contiguous()
    frequency_counts = torch.bincount(labels, minlength=VOCAB_SIZE).to(dtype=torch.int64).contiguous()
    support_ids = torch.nonzero(frequency_counts > 0, as_tuple=False).flatten().to(dtype=torch.int64).contiguous()
    support_counts = frequency_counts.index_select(0, support_ids).contiguous()
    identities = {
        "support_digest_trr0010": support_digest(support_ids, support_counts),
        "support_ids_tensor_sha256": tensor_sha256(support_ids),
        "support_counts_tensor_sha256": tensor_sha256(support_counts),
        "frequency_vector_tensor_sha256": tensor_sha256(frequency_counts),
    }
    for key, actual in identities.items():
        if actual != reference[key]:
            raise RecoveryError(f"derived common frequency identity differs: {key}")
    if int(support_ids.numel()) != int(reference.get("support_count", -1)):
        raise RecoveryError("derived support count differs from binding")
    if int(frequency_counts.sum().item()) != int(reference.get("positive_occurrences", -1)):
        raise RecoveryError("derived positive occurrence count differs from binding")
    output_path = Path(output_path).expanduser()
    receipt_path = Path(receipt_path).expanduser()
    if output_path.exists() or output_path.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
        raise RecoveryError("recovery output and receipt are create-only")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "frequency_counts": frequency_counts,
            "support_ids": support_ids,
            "support_counts": support_counts,
        },
        str(output_path),
    )
    output = file_record(output_path, label="recovered common frequency payload")
    try:
        with safe_open(str(output_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"frequency_counts", "support_ids", "support_counts"}:
                raise RecoveryError("recovered support payload tensor keys changed")
            reread_frequency = handle.get_tensor("frequency_counts").detach().cpu().contiguous()
            reread_ids = handle.get_tensor("support_ids").detach().cpu().contiguous()
            reread_counts = handle.get_tensor("support_counts").detach().cpu().contiguous()
    except RecoveryError:
        raise
    except Exception as exc:
        raise RecoveryError("recovered support payload cannot be reopened") from exc
    if not torch.equal(reread_frequency, frequency_counts) or not torch.equal(reread_ids, support_ids) or not torch.equal(reread_counts, support_counts):
        raise RecoveryError("recovered support payload tensor values changed after save")
    receipt = {
        "schema": SCHEMA,
        "task_id": "TRR-P11",
        "status": "PASS_PUBLIC_B0_FREQUENCY_DERIVED",
        "created_utc": utc_now(),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "source": {
            "file": source,
            "role": "exact hash-bound B0 public fitting payload",
            "accessed_tensors": ["token_ids", "attention_mask"],
            "excluded_tensors": ["position_ids", "activations", "hidden_states"],
            "records": expected_records,
            "width": expected_width,
            "post_bos_positions": post_bos,
            "labels": "token_ids[:,1:][attention_mask[:,1:]]",
        },
        "historical_reference": {
            "path": str(historical_path),
            "expected_bytes": reference.get("bytes"),
            "expected_sha256": reference.get("sha256"),
            "status": "PRESENT_NOT_USED" if historical_exists else "MISSING_AT_RECOVERY_NOT_USED",
            "opened": False,
        },
        "derived": {
            "file": output,
            "tensor_keys": ["frequency_counts", "support_ids", "support_counts"],
            "tensor_dtypes": {
                "frequency_counts": str(frequency_counts.dtype),
                "support_ids": str(support_ids.dtype),
                "support_counts": str(support_counts.dtype),
            },
            "tensor_shapes": {
                "frequency_counts": list(frequency_counts.shape),
                "support_ids": list(support_ids.shape),
                "support_counts": list(support_counts.shape),
            },
            "support_count": int(support_ids.numel()),
            "positive_occurrences": int(frequency_counts.sum().item()),
            **identities,
        },
        "truth_boundary": {
            "public_token_ids_read": True,
            "public_attention_masks_read": True,
            "source_text_read": False,
            "hidden_states_read": False,
            "model_opened": False,
            "evaluation_truth_opened": False,
            "predictions_read": False,
        },
    }
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing to recover without explicit --execute")
    start = time.monotonic()
    try:
        receipt = derive(args.config, args.output, args.receipt)
    except RecoveryError as exc:
        raise SystemExit(f"common frequency recovery failed: {exc}") from exc
    receipt["command"] = [str(value) for value in sys.argv]
    receipt["completed_utc"] = utc_now()
    receipt["wall_seconds"] = time.monotonic() - start
    receipt["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    receipt_path = Path(args.receipt).expanduser()
    if receipt_path.exists() or receipt_path.is_symlink():
        raise SystemExit(f"recovery receipt must be create-only: {receipt_path}")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: receipt[key] for key in ("status", "derived", "historical_reference", "wall_seconds", "peak_rss_kib")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
