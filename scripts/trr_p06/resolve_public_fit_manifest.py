#!/usr/bin/env python3
"""Resolve and verify the immutable published P05 public-fit manifest.

The P06 fit runner consumes the published P05 manifest schema.  This adapter
makes every referenced path absolute and verifies the manifest-declared bytes
and SHA-256 digests before a later fit may load any public tensors.  Hashing is
streamed; this command never opens a safetensors tensor and never reads fresh
panel, evaluator, target, or holdout data.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any


TASK_ID = "TRR-P06"
RESOLUTION_SCHEMA = "token-reconstruction.trr-p06-resolved-public-fit-manifest.v1"
SUPPORTED_SOURCE_SCHEMA = "token-reconstruction.trr0005-public-fit-data.v1"
REQUIRED_RESOURCE_NAMES = (
    "embedding_table",
    "fit_observations",
    "fit_records",
    "fit_truth",
    "fit_valid_mask",
    "validation_observations",
    "validation_records",
    "validation_truth",
    "validation_valid_mask",
)


class ResolutionError(RuntimeError):
    """Raised when an immutable published resource cannot be verified."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, expected_sha256: str | None = None, expected_bytes: int | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ResolutionError(f"resource is not a regular file: {path}")
    actual_bytes = path.stat().st_size
    if expected_bytes is not None and int(expected_bytes) != actual_bytes:
        raise ResolutionError(f"resource byte count changed: {path}")
    actual_sha256 = _sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != str(expected_sha256):
        raise ResolutionError(f"resource hash changed: {path}")
    result = {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha256}
    if expected_sha256 is not None:
        result["expected_sha256"] = str(expected_sha256)
    if expected_bytes is not None:
        result["expected_bytes"] = int(expected_bytes)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResolutionError(f"invalid JSON manifest: {path}") from exc
    if not isinstance(value, Mapping):
        raise ResolutionError(f"manifest is not an object: {path}")
    return dict(value)


def _resolve_declared_path(manifest_path: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ResolutionError("manifest resource has no path")
    path = Path(raw)
    return (path if path.is_absolute() else manifest_path.parent / path).expanduser().absolute()


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
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def _resource_bindings(manifest_path: Path, manifest: Mapping[str, Any]) -> list[tuple[str, dict[str, Any], Path]]:
    resources = manifest.get("resources")
    if not isinstance(resources, Mapping):
        raise ResolutionError("published manifest has no resources object")
    missing = [name for name in REQUIRED_RESOURCE_NAMES if name not in resources]
    if missing:
        raise ResolutionError(f"published manifest is missing resources: {missing}")
    bindings: list[tuple[str, dict[str, Any], Path]] = []
    for name in REQUIRED_RESOURCE_NAMES:
        descriptor = resources[name]
        if not isinstance(descriptor, Mapping):
            raise ResolutionError(f"resource descriptor is malformed: {name}")
        path = _resolve_declared_path(manifest_path, descriptor.get("path"))
        bindings.append((f"resources.{name}", dict(descriptor), path))
    source = manifest.get("source")
    if isinstance(source, Mapping):
        for name, descriptor in source.items():
            if not isinstance(descriptor, Mapping) or "path" not in descriptor or "sha256" not in descriptor:
                continue
            path = _resolve_declared_path(manifest_path, descriptor.get("path"))
            bindings.append((f"source.{name}", dict(descriptor), path))
    return bindings


def resolve_manifest(args: argparse.Namespace) -> dict[str, Any]:
    started_utc = _utc_now()
    started_clock = time.perf_counter()
    canonical = Path(args.canonical_manifest).expanduser().absolute()
    root = Path(args.repository_root).expanduser().resolve()
    if canonical.is_symlink() or not canonical.is_file():
        raise ResolutionError(f"canonical manifest is unavailable: {canonical}")
    source_manifest = _file_record(canonical)
    manifest = _load_json(canonical)
    if manifest.get("schema") != SUPPORTED_SOURCE_SCHEMA:
        raise ResolutionError(f"unsupported published manifest schema: {manifest.get('schema')}")
    if manifest.get("task_id") != "TRR-0005":
        raise ResolutionError("published manifest task ID changed")

    resolved = json.loads(json.dumps(manifest))
    verified: list[dict[str, Any]] = []
    by_path: dict[Path, dict[str, Any]] = {}
    for logical_name, descriptor, path in _resource_bindings(canonical, manifest):
        record = by_path.get(path)
        if record is None:
            record = _file_record(
                path,
                expected_sha256=str(descriptor["sha256"]),
                expected_bytes=(None if "bytes" not in descriptor else int(descriptor["bytes"])),
            )
            by_path[path] = record
        elif str(descriptor.get("sha256")) != record["sha256"]:
            raise ResolutionError(f"duplicate resource has a different hash: {logical_name}")
        verified.append(
            {
                "logical_name": logical_name,
                "path": str(path),
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "manifest_sha256": str(descriptor["sha256"]),
                "manifest_bytes": descriptor.get("bytes"),
                "verified": True,
            }
        )
        if logical_name.startswith("resources."):
            resolved["resources"][logical_name.split(".", 1)[1]]["path"] = str(path)
        elif logical_name.startswith("source."):
            resolved["source"][logical_name.split(".", 1)[1]]["path"] = str(path)

    if len(verified) < len(REQUIRED_RESOURCE_NAMES):
        raise ResolutionError("not all public fit/validation resources were verified")
    resolved["p06_resolution"] = {
        "schema": RESOLUTION_SCHEMA,
        "task_id": TASK_ID,
        "status": "VERIFIED_EXTERNAL_PUBLIC_RESOURCES",
        "source_manifest": source_manifest,
        "canonical_manifest_schema": manifest.get("schema"),
        "canonical_manifest_task_id": manifest.get("task_id"),
        "resolved_at_utc": _utc_now(),
        "verified_bindings": verified,
        "unique_files_verified": len(by_path),
        "full_file_sha256": True,
        "safetensors_opened": False,
        "tensor_payload_loaded": False,
        "source_rows_or_plaintext_read": False,
        "fresh_truth_or_answers_read": False,
        "target_update_payload_opened": False,
        "repository_root": str(root),
        "code_commit_before_resolution": _git_commit(root),
        "command": list(sys.argv),
        "peak_rss_bytes": _rss_bytes(),
    }
    output = Path(args.output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ResolutionError(f"resolved manifest output is create-only: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(resolved, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    output_record = _file_record(output)
    # Add a separate immutable completion sidecar only when requested.  The
    # manifest itself remains the implementation input; this receipt does not
    # duplicate tensor data or change the canonical P05 manifest.
    if args.receipt is not None:
        receipt_path = Path(args.receipt).expanduser().absolute()
        if receipt_path.exists() or receipt_path.is_symlink():
            raise ResolutionError(f"resolution receipt output is create-only: {receipt_path}")
        receipt = {
            "schema": RESOLUTION_SCHEMA + ".receipt",
            "task_id": TASK_ID,
            "status": "PASS_ALL_DECLARED_PUBLIC_RESOURCES",
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started_clock,
            "canonical_manifest": source_manifest,
            "resolved_manifest": output_record,
            "unique_files_verified": len(by_path),
            "logical_bindings_verified": len(verified),
            "verified": verified,
            "safetensors_opened": False,
            "tensor_payload_loaded": False,
            "source_rows_or_plaintext_read": False,
            "fresh_truth_or_answers_read": False,
            "command": list(sys.argv),
            "repository_root": str(root),
            "code_commit_before_resolution": _git_commit(root),
            "peak_rss_bytes": _rss_bytes(),
        }
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        receipt_record = _file_record(receipt_path)
    else:
        receipt_record = None
    return {
        "task_id": TASK_ID,
        "status": "VERIFIED_EXTERNAL_PUBLIC_RESOURCES",
        "resolved_manifest": output_record,
        "receipt": receipt_record,
        "unique_files_verified": len(by_path),
        "logical_bindings_verified": len(verified),
        "safetensors_opened": False,
        "tensor_payload_loaded": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(resolve_manifest(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ResolutionError, ValueError) as exc:
        raise SystemExit(f"TRR-P06 manifest resolution error: {exc}")
