#!/usr/bin/env python3
"""Build the TRR-0004 affine-fit manifest from prepared public tensors.

The public prefix preparation emits separate padded Alpaca fit and validation
artifacts.  This small adapter keeps the fit artifact intact and pads the
existing public Pile24 validation slice to the same width before concatenating
it with Alpaca validation.  The resulting manifest is consumed by
``trr0004_historical_affine_ce.py``; it is preparation metadata, not a model
fit or an evaluation scorer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from token_reconstruction.historical_affine_ce import FIT_DATA_SCHEMA, file_sha256


TASK_ID = "TRR-0004"
ADAPTER_SCHEMA = "token-reconstruction.trr0004-affine-fit-manifest-adapter.v1"
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256


class AffineManifestAdapterError(RuntimeError):
    """Raised when the public affine-fit adapter cannot bind its inputs."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _regular_file(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise AffineManifestAdapterError(f"{label} must be a regular file: {path}")
    return path


def _file_record(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_file(path, label=label)
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": file_sha256(path)}


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    path = _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AffineManifestAdapterError(f"cannot parse {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise AffineManifestAdapterError(f"{label} must contain an object")
    return value


def _tensor(path: Path, key: str, *, label: str) -> torch.Tensor:
    path = _regular_file(path, label=label)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if key not in handle.keys():
                raise AffineManifestAdapterError(f"{label} has no {key!r} tensor")
            return handle.get_tensor(key).contiguous()
    except AffineManifestAdapterError:
        raise
    except Exception as exc:  # pragma: no cover - backend-specific
        raise AffineManifestAdapterError(f"cannot load {label}: {path}") from exc


def _artifact_entry(path: Path, key: str, value: torch.Tensor, *, label: str) -> dict[str, Any]:
    record = _file_record(path, label=label)
    record.update(
        {
            "tensor_key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    )
    return record


def _resolve_evidence_output(
    evidence: Mapping[str, Any], section: str, *, preparation_root: Path
) -> Path:
    outputs = evidence.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get(section), Mapping):
        raise AffineManifestAdapterError(f"preparation evidence has no {section} output")
    raw = outputs[section].get("path")
    if not isinstance(raw, str) or not raw:
        raise AffineManifestAdapterError(f"preparation evidence {section} path is missing")
    path = Path(raw)
    if not path.is_absolute():
        path = preparation_root / path
    path = _regular_file(path, label=f"prepared {section} artifact")
    expected = outputs[section]
    current = _file_record(path, label=f"prepared {section} artifact")
    for key in ("bytes", "sha256"):
        if expected.get(key) != current[key]:
            raise AffineManifestAdapterError(f"prepared {section} artifact binding changed")
    return path


def _resolve_record_output(
    outputs: Mapping[str, Any], section: str, *, preparation_root: Path, fallback: Path
) -> Path:
    """Resolve a preparation record path using the receipt's path semantics."""

    entry = outputs.get(section)
    raw = entry.get("path") if isinstance(entry, Mapping) else None
    path = Path(raw) if isinstance(raw, str) and raw else fallback
    if not path.is_absolute():
        path = preparation_root / path
    return _regular_file(path, label=f"prepared {section}")


def _verify_receipt_file(
    path: Path, expected: Any, *, label: str
) -> Path:
    """Bind one externally prepared file to the bytes recorded in its receipt."""

    if not isinstance(expected, Mapping):
        raise AffineManifestAdapterError(f"{label} receipt entry is missing")
    current = _file_record(path, label=label)
    if expected.get("bytes") != current["bytes"] or expected.get("sha256") != current["sha256"]:
        raise AffineManifestAdapterError(f"{label} binding changed")
    return path


def _check_right_padded_mask(mask: torch.Tensor, labels: torch.Tensor, *, label: str) -> None:
    """Check binary right-padding and ensure padding cannot become supervision."""

    integer_dtypes = (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    if mask.dtype == torch.bool:
        active = mask
    elif mask.dtype in integer_dtypes:
        if mask.lt(0).any().item() or mask.gt(1).any().item():
            raise AffineManifestAdapterError(f"{label} integer values must be exactly 0 or 1")
        active = mask.to(torch.bool)
    else:
        raise AffineManifestAdapterError(f"{label} must be boolean or integer")
    if not active[:, 0].all().item():
        raise AffineManifestAdapterError(f"{label} rows must begin active")
    if (active[:, 1:] & ~active[:, :-1]).any().item():
        raise AffineManifestAdapterError(f"{label} must use contiguous right-padding")
    if active.sum(dim=1).le(1).any().item():
        raise AffineManifestAdapterError(f"{label} rows have no post-BOS position")
    if labels[~active].ne(PAD_TOKEN_ID).any().item():
        raise AffineManifestAdapterError(f"{label} padding labels must use PAD_TOKEN_ID")


def _sanitize_records(
    path: Path, *, label: str, style: str | None = None, full_count: int | None = None
) -> list[dict[str, Any]]:
    payload = _load_json(path, label=label)
    values = payload.get("records")
    if not isinstance(values, list) or not values:
        raise AffineManifestAdapterError(f"{label} has no records")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, Mapping) or not isinstance(value.get("record_id"), str):
            raise AffineManifestAdapterError(f"{label} record {index} has no record_id")
        record_id = str(value["record_id"])
        if not record_id or record_id in seen:
            raise AffineManifestAdapterError(f"{label} record IDs are empty or duplicated")
        seen.add(record_id)
        # The adapter carries only registration metadata.  In particular, it
        # drops any accidental source/token fields before writing the new manifest.
        allowed = (
            "record_id",
            "row_index",
            "source_row",
            "public_record_sha256",
            "rendered_sha256",
            "rendered_char_count",
            "full_token_count",
            "post_bos_token_count",
            "active_token_count",
            "padded_length",
            "style",
            "group",
        )
        row = {key: value[key] for key in allowed if key in value}
        if style is not None:
            row["style"] = style
        if full_count is not None:
            row["full_token_count"] = int(full_count)
            row["post_bos_token_count"] = int(full_count) - 1
        if row.get("full_token_count") is None and row.get("post_bos_token_count") is not None:
            row["full_token_count"] = int(row["post_bos_token_count"]) + 1
        if row.get("post_bos_token_count") is None or int(row["post_bos_token_count"]) <= 0:
            raise AffineManifestAdapterError(f"{label} record {index} has no positive post-BOS count")
        result.append(row)
    return result


def _relative(path: Path, *, root: Path) -> str:
    return os.path.relpath(path, root)


def _resource(path: Path, key: str, value: torch.Tensor, *, root: Path, label: str) -> dict[str, Any]:
    record = _artifact_entry(path, key, value, label=label)
    record["path"] = _relative(path, root=root)
    return record


def _check_tensor(value: torch.Tensor, *, shape: tuple[int, ...], dtype: torch.dtype, label: str) -> None:
    if tuple(value.shape) != shape or value.dtype != dtype:
        raise AffineManifestAdapterError(
            f"{label} geometry/dtype changed: {tuple(value.shape)} {value.dtype}"
        )
    if value.dtype.is_floating_point and not torch.isfinite(value).all().item():
        raise AffineManifestAdapterError(f"{label} contains non-finite values")


def build_manifest(args: argparse.Namespace) -> int:
    repository_root = args.repository_root.expanduser().resolve()
    preparation_root = args.preparation_root.expanduser().resolve()
    pile_root = args.pile_root.expanduser().resolve()
    output_manifest = args.output_manifest.expanduser().resolve()
    if output_manifest.exists() or output_manifest.is_symlink():
        raise AffineManifestAdapterError(f"output manifest is create-only: {output_manifest}")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    validation_artifact = (
        args.validation_artifact.expanduser().resolve()
        if args.validation_artifact is not None
        else output_manifest.parent / "validation_mixed_cut4.safetensors"
    )
    if validation_artifact.exists() or validation_artifact.is_symlink():
        raise AffineManifestAdapterError(f"validation artifact is create-only: {validation_artifact}")

    evidence_path = preparation_root / "preparation_evidence.json"
    evidence = _load_json(evidence_path, label="public activation preparation evidence")
    if evidence.get("schema") != "token-reconstruction.trr0004-public-activation-preparation.v1":
        raise AffineManifestAdapterError("public activation preparation schema changed")
    if evidence.get("status") != "PUBLIC_ACTIVATION_PREPARATION_COMPLETE_NO_CONFIRMATION":
        raise AffineManifestAdapterError("public activation preparation is not complete")
    contract = evidence.get("access_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("evaluator_private_truth_accessed") is not False
        or contract.get("target_weights_accessed") is not False
        or contract.get("confirmation_records_generated") is not False
    ):
        raise AffineManifestAdapterError("public preparation access contract changed")
    train_artifact = _resolve_evidence_output(evidence, "train_large", preparation_root=preparation_root)
    alpaca_artifact = _resolve_evidence_output(
        evidence, "validation_alpaca", preparation_root=preparation_root
    )
    outputs = evidence["outputs"]
    train_records_path = _regular_file(
        preparation_root / "train_large_records.json", label="prepared fit records"
    )
    alpaca_records_path = _regular_file(
        preparation_root / "validation_alpaca_records.json", label="prepared Alpaca validation records"
    )
    # Resolve the recorded files if the preparation was invoked with an
    # alternate output root or path layout.
    if isinstance(outputs, Mapping):
        train_records_path = _resolve_record_output(
            outputs, "train_records", preparation_root=preparation_root, fallback=train_records_path
        )
        alpaca_records_path = _resolve_record_output(
            outputs,
            "validation_records",
            preparation_root=preparation_root,
            fallback=alpaca_records_path,
        )

    train_x = _tensor(train_artifact, "activations", label="prepared fit activations")
    train_y = _tensor(train_artifact, "token_ids", label="prepared fit labels")
    train_mask = _tensor(train_artifact, "attention_mask", label="prepared fit attention mask")
    alpaca_x = _tensor(alpaca_artifact, "activations", label="prepared Alpaca validation activations")
    alpaca_y = _tensor(alpaca_artifact, "token_ids", label="prepared Alpaca validation labels")
    alpaca_mask = _tensor(alpaca_artifact, "attention_mask", label="prepared Alpaca validation mask")
    if tuple(train_x.shape) != (1200, 192, HIDDEN_SIZE):
        raise AffineManifestAdapterError("registered public fit geometry changed")
    if tuple(alpaca_x.shape) != (24, 192, HIDDEN_SIZE):
        raise AffineManifestAdapterError("registered Alpaca validation geometry changed")
    if tuple(train_y.shape) != tuple(train_x.shape[:2]) or tuple(train_mask.shape) != tuple(train_x.shape[:2]):
        raise AffineManifestAdapterError("prepared fit combined artifact geometry changed")
    if tuple(alpaca_y.shape) != tuple(alpaca_x.shape[:2]) or tuple(alpaca_mask.shape) != tuple(alpaca_x.shape[:2]):
        raise AffineManifestAdapterError("prepared Alpaca validation artifact geometry changed")
    if train_x.shape[1] != alpaca_x.shape[1]:
        raise AffineManifestAdapterError("prepared fit and Alpaca validation widths differ")
    _check_tensor(train_x, shape=tuple(train_x.shape), dtype=torch.bfloat16, label="prepared fit activations")
    _check_tensor(alpaca_x, shape=tuple(alpaca_x.shape), dtype=torch.bfloat16, label="prepared Alpaca activations")
    if train_y.dtype not in (torch.int32, torch.int64) or alpaca_y.dtype not in (torch.int32, torch.int64):
        raise AffineManifestAdapterError("prepared public labels must be integer")
    accepted_mask_dtypes = (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    if train_mask.dtype not in accepted_mask_dtypes or alpaca_mask.dtype not in accepted_mask_dtypes:
        raise AffineManifestAdapterError("prepared public masks must be boolean or integer")
    if train_y[:, 0].ne(BOS_TOKEN_ID).any().item() or alpaca_y[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise AffineManifestAdapterError("prepared public rows lost BOS")
    _check_right_padded_mask(train_mask, train_y, label="prepared fit mask")
    _check_right_padded_mask(alpaca_mask, alpaca_y, label="prepared Alpaca validation mask")

    pile_evidence_path = pile_root / "validation_slice_evidence.json"
    pile_evidence = _load_json(pile_evidence_path, label="public Pile validation evidence")
    if pile_evidence.get("truth_role") != "public auxiliary validation only; no evaluator-private labels":
        raise AffineManifestAdapterError("Pile validation truth role is not public auxiliary")
    pile_contract = pile_evidence.get("disjointness")
    if not isinstance(pile_contract, Mapping) or pile_contract.get("checked_before_validation_label_access") is not True:
        raise AffineManifestAdapterError("Pile validation disjointness receipt is not verified")
    pile_artifact = _verify_receipt_file(
        pile_root / "public_validation_observations.safetensors",
        pile_evidence.get("outputs", {}).get("observations") if isinstance(pile_evidence.get("outputs"), Mapping) else None,
        label="public Pile validation activations",
    )
    pile_truth_artifact = _verify_receipt_file(
        pile_root / "public_validation_truth.safetensors",
        pile_evidence.get("outputs", {}).get("truth") if isinstance(pile_evidence.get("outputs"), Mapping) else None,
        label="public Pile validation labels",
    )
    pile_records_path = _verify_receipt_file(
        pile_root / "public_validation_records.json",
        pile_evidence.get("outputs", {}).get("records") if isinstance(pile_evidence.get("outputs"), Mapping) else None,
        label="public Pile validation records",
    )
    pile_x = _tensor(pile_artifact, "activations", label="public Pile validation activations")
    pile_y = _tensor(pile_truth_artifact, "token_ids", label="public Pile validation labels")
    if pile_x.ndim != 3 or pile_x.shape[1] <= 1 or pile_x.shape[2] != HIDDEN_SIZE:
        raise AffineManifestAdapterError("public Pile validation geometry changed")
    if tuple(pile_y.shape) != tuple(pile_x.shape[:2]) or pile_y.dtype not in (torch.int32, torch.int64):
        raise AffineManifestAdapterError("public Pile validation labels geometry changed")
    if pile_y[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise AffineManifestAdapterError("public Pile validation rows lost BOS")
    if not torch.isfinite(pile_x.float()).all().item():
        raise AffineManifestAdapterError("public Pile validation activations are non-finite")
    if pile_x.shape[0] != 24 or pile_x.shape[1] != 40:
        raise AffineManifestAdapterError("the registered public Pile24 validation geometry changed")

    width = int(train_x.shape[1])
    if int(pile_x.shape[1]) > width:
        raise AffineManifestAdapterError("public Pile validation exceeds prepared padded width")
    mixed_x = torch.zeros((int(alpaca_x.shape[0]) + int(pile_x.shape[0]), width, HIDDEN_SIZE), dtype=torch.bfloat16)
    mixed_y = torch.full(
        (mixed_x.shape[0], width), PAD_TOKEN_ID, dtype=torch.int32
    )
    mixed_mask = torch.zeros((mixed_x.shape[0], width), dtype=torch.uint8)
    mixed_x[: alpaca_x.shape[0]] = alpaca_x.to(dtype=torch.bfloat16)
    mixed_y[: alpaca_y.shape[0]] = alpaca_y.to(dtype=torch.int32)
    mixed_mask[: alpaca_mask.shape[0]] = alpaca_mask.to(dtype=torch.uint8)
    pile_start = int(alpaca_x.shape[0])
    mixed_x[pile_start:, : pile_x.shape[1]] = pile_x.to(dtype=torch.bfloat16)
    mixed_y[pile_start:, : pile_y.shape[1]] = pile_y.to(dtype=torch.int32)
    mixed_mask[pile_start:, : pile_y.shape[1]] = 1
    if not torch.isfinite(mixed_x.float()).all().item():
        raise AffineManifestAdapterError("mixed validation activations are non-finite")
    fit_records = _sanitize_records(train_records_path, label="prepared fit records")
    alpaca_records = _sanitize_records(alpaca_records_path, label="prepared Alpaca validation records", style="alpaca")
    pile_records = _sanitize_records(
        pile_records_path,
        label="public Pile validation records",
        style="pile",
        full_count=int(pile_x.shape[1]),
    )
    validation_records = alpaca_records + pile_records
    fit_ids = {str(row["record_id"]) for row in fit_records}
    validation_ids = {str(row["record_id"]) for row in validation_records}
    if fit_ids & validation_ids:
        raise AffineManifestAdapterError("public fit and mixed validation records overlap")
    if len(fit_records) != int(train_x.shape[0]) or len(alpaca_records) != int(alpaca_x.shape[0]):
        raise AffineManifestAdapterError("prepared record count does not match tensor rows")
    for index, row in enumerate(fit_records):
        expected = int(train_mask[index].sum().item()) - 1
        if int(row["post_bos_token_count"]) != expected:
            raise AffineManifestAdapterError("prepared fit record length disagrees with mask")
    for index, row in enumerate(alpaca_records):
        expected = int(alpaca_mask[index].sum().item()) - 1
        if int(row["post_bos_token_count"]) != expected:
            raise AffineManifestAdapterError("prepared Alpaca record length disagrees with mask")
    if len(validation_records) != int(mixed_x.shape[0]):
        raise AffineManifestAdapterError("mixed validation record count changed")

    records_dir = output_manifest.parent
    fit_records_out = records_dir / "affine_fit_records.json"
    validation_records_out = records_dir / "affine_validation_records.json"
    for path in (fit_records_out, validation_records_out):
        if path.exists() or path.is_symlink():
            raise AffineManifestAdapterError(f"adapter record manifest is create-only: {path}")

    embedding_path = _regular_file(args.embedding_table, label="public normalized embedding table")
    embedding = _tensor(embedding_path, "embeddings", label="public normalized embedding table")
    if tuple(embedding.shape) != (VOCAB_SIZE, HIDDEN_SIZE) or not embedding.dtype.is_floating_point:
        raise AffineManifestAdapterError("public normalized embedding table geometry changed")
    if not torch.isfinite(embedding.float()).all().item():
        raise AffineManifestAdapterError("public normalized embedding table is non-finite")
    norms = torch.linalg.vector_norm(embedding.float(), dim=-1)
    if not torch.isclose(norms, torch.ones_like(norms), atol=2e-4, rtol=2e-4).all().item():
        raise AffineManifestAdapterError("public embedding table is not normalized")

    validation_artifact.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"activations": mixed_x, "token_ids": mixed_y, "attention_mask": mixed_mask},
        str(validation_artifact),
    )

    fit_records_out.write_text(
        json.dumps({"records": fit_records}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validation_records_out.write_text(
        json.dumps({"records": validation_records}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def resource(path: Path, key: str, value: torch.Tensor, label: str) -> dict[str, Any]:
        return _resource(path, key, value, root=records_dir, label=label)

    def json_resource(path: Path, label: str) -> dict[str, Any]:
        record = _file_record(path, label=label)
        record["path"] = _relative(path, root=records_dir)
        return record

    manifest = {
        "schema": FIT_DATA_SCHEMA,
        "task_id": TASK_ID,
        "layout": "padded_records",
        "bos_token_id": BOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "embedding_table_normalized": True,
        "alignment": {
            "mode": "current_token",
            "observation_index": "i",
            "label_index": "i",
            "bos_position": 0,
            "scored_positions": "post_bos",
        },
        "resources": {
            "fit_records": json_resource(fit_records_out, "adapter fit records"),
            "validation_records": json_resource(validation_records_out, "adapter validation records"),
            "fit_observations": resource(train_artifact, "activations", train_x, "prepared fit activations"),
            "fit_truth": resource(train_artifact, "token_ids", train_y, "prepared fit labels"),
            "fit_valid_mask": resource(train_artifact, "attention_mask", train_mask, "prepared fit attention mask"),
            "validation_observations": resource(validation_artifact, "activations", mixed_x, "mixed validation activations"),
            "validation_truth": resource(validation_artifact, "token_ids", mixed_y, "mixed validation labels"),
            "validation_valid_mask": resource(validation_artifact, "attention_mask", mixed_mask, "mixed validation mask"),
            "embedding_table": resource(embedding_path, "embeddings", embedding, "public normalized embedding table"),
        },
        "source": {
            "adapter_schema": ADAPTER_SCHEMA,
            "adapter_script": _file_record(Path(__file__).resolve(), label="adapter script"),
            "git_commit": _git_commit(repository_root),
            "preparation_evidence": _file_record(evidence_path, label="public preparation evidence"),
            "pile_validation_evidence": _file_record(pile_evidence_path, label="public Pile validation evidence"),
            "public_embedding_table": _file_record(embedding_path, label="public normalized embedding table"),
            "styles": ["alpaca", "pile"],
            "validation_records": int(len(validation_records)),
            "validation_geometry": list(mixed_x.shape),
        },
        "public_only": True,
        "private_evaluator_truth_accessed": False,
        "target_weights_accessed": False,
        "adapter_generated_at_utc": _utc_now(),
    }
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "adapter_manifest_complete",
                "manifest": str(output_manifest),
                "validation_artifact": str(validation_artifact),
                "fit_records": len(fit_records),
                "validation_records": len(validation_records),
                "validation_styles": ["alpaca", "pile"],
                "validation_shape": list(mixed_x.shape),
            },
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--pile-root", type=Path, required=True)
    parser.add_argument("--embedding-table", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--validation-artifact", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return build_manifest(args)


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (AffineManifestAdapterError, OSError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
