"""Immutable output receipts and the fail-closed truth-opening gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping


FREEZE_SCHEMA = "token-reconstruction.freeze-receipt.v1"


class FreezeError(RuntimeError):
    """Raised when frozen state is absent, mutable, or inconsistent."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FreezeError(f"{description} must be a regular file: {path}")


def _relative(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise FreezeError(f"path is outside repository root: {path}") from exc


def _safe_relative(value: Any, *, description: str) -> str:
    """Validate a receipt path before joining it to a trusted directory.

    Receipt paths are data written by a previous process. Treating them as
    ordinary strings lets ``..`` escape the frozen root during verification,
    which could make an unrelated file appear to satisfy the receipt.
    """

    if not isinstance(value, str) or not value:
        raise FreezeError(f"{description} path is absent")
    if "\\" in value:
        raise FreezeError(f"{description} path must use POSIX separators")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in ("", ".", "..") for part in candidate.parts)
    ):
        raise FreezeError(f"{description} path is unsafe: {value}")
    normalized = candidate.as_posix()
    if normalized != value:
        raise FreezeError(f"{description} path is not normalized: {value}")
    return normalized


def _path_under(path: Path, root: Path, *, description: str) -> Path:
    """Resolve a regular path and require that it remains below ``root``."""

    if path.is_symlink():
        raise FreezeError(f"{description} must not be a symbolic link: {path}")
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise FreezeError(f"{description} path escaped its root: {path}") from exc
    return resolved


def _prohibited_relative(relative: str) -> bool:
    lowered = relative.casefold()
    lowered_parts = tuple(part.casefold() for part in PurePosixPath(relative).parts)
    return any(fragment in lowered for fragment in ("truth", "oracle")) or any(
        part == "evaluator_private" or part.startswith("target_lora")
        for part in lowered_parts
    )


def freeze_payload(
    *,
    repository_root: Path,
    frozen_root: Path,
    plan_path: Path,
    preregistration_commit: str,
    created_utc: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical receipt content without writing or changing permissions."""

    if len(preregistration_commit) != 40:
        raise FreezeError("full preregistration commit is required")
    _regular_file(plan_path, "plan")
    if frozen_root.is_symlink() or not frozen_root.is_dir():
        raise FreezeError("frozen root must be a regular directory")

    entries: list[dict[str, Any]] = []
    for path in sorted(frozen_root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        _regular_file(path, "frozen artifact")
        relative = _relative(path, repository_root)
        if _prohibited_relative(relative):
            raise FreezeError(f"prohibited private artifact in frozen bundle: {relative}")
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    if not entries:
        raise FreezeError("frozen bundle is empty")

    payload = {
        "schema": FREEZE_SCHEMA,
        "created_utc": created_utc,
        "preregistration_commit": preregistration_commit,
        "plan": {
            "path": _relative(plan_path, repository_root),
            "bytes": plan_path.stat().st_size,
            "sha256": sha256_path(plan_path),
        },
        "frozen_root": _relative(frozen_root, repository_root),
        "entries": entries,
        "metadata": dict(metadata or {}),
    }
    json.dumps(payload, sort_keys=True, allow_nan=False)
    return payload


def create_freeze_receipt(
    *,
    repository_root: Path,
    frozen_root: Path,
    plan_path: Path,
    receipt_path: Path,
    preregistration_commit: str,
    created_utc: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a receipt create-only, then make every frozen input read-only."""

    if receipt_path.exists() or receipt_path.is_symlink():
        raise FreezeError(f"receipt already exists: {receipt_path}")
    payload = freeze_payload(
        repository_root=repository_root,
        frozen_root=frozen_root,
        plan_path=plan_path,
        preregistration_commit=preregistration_commit,
        created_utc=created_utc,
        metadata=metadata,
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(receipt_path, flags, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    for entry in payload["entries"]:
        artifact = repository_root / entry["path"]
        artifact.chmod(0o444)
    return payload


def verify_freeze_receipt(
    receipt_path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Fail closed unless every frozen byte and the plan match the receipt."""

    _regular_file(receipt_path, "freeze receipt")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeError("freeze receipt is invalid JSON") from exc
    if payload.get("schema") != FREEZE_SCHEMA:
        raise FreezeError("freeze receipt schema changed")
    if not isinstance(payload.get("entries"), list) or not payload["entries"]:
        raise FreezeError("freeze receipt has no entries")

    plan = payload.get("plan")
    if not isinstance(plan, dict):
        raise FreezeError("freeze receipt plan record is absent")
    plan_relative = _safe_relative(plan.get("path"), description="frozen plan")
    plan_path = _path_under(
        repository_root / plan_relative,
        repository_root,
        description="frozen plan",
    )
    _regular_file(plan_path, "frozen plan")
    if plan_path.stat().st_size != plan.get("bytes") or sha256_path(plan_path) != plan.get(
        "sha256"
    ):
        raise FreezeError("frozen plan hash or size changed")

    frozen_root_relative = _safe_relative(
        payload.get("frozen_root"), description="frozen root"
    )
    frozen_root = _path_under(
        repository_root / frozen_root_relative,
        repository_root,
        description="frozen root",
    )
    if frozen_root.is_symlink() or not frozen_root.is_dir():
        raise FreezeError("frozen root must be a regular directory")

    observed_paths: set[str] = set()
    for entry in payload["entries"]:
        if not isinstance(entry, dict):
            raise FreezeError("freeze receipt entry is malformed")
        relative = _safe_relative(entry.get("path"), description="frozen artifact")
        if relative in observed_paths:
            raise FreezeError("freeze receipt path is absent or duplicated")
        if _prohibited_relative(relative):
            raise FreezeError(f"prohibited private artifact in frozen bundle: {relative}")
        observed_paths.add(relative)
        artifact = _path_under(
            repository_root / relative,
            frozen_root,
            description="frozen artifact",
        )
        _regular_file(artifact, "frozen artifact")
        if artifact.stat().st_size != entry.get("bytes"):
            raise FreezeError(f"frozen artifact size changed: {relative}")
        if sha256_path(artifact) != entry.get("sha256"):
            raise FreezeError(f"frozen artifact hash changed: {relative}")

    actual_paths: set[str] = set()
    for path in sorted(frozen_root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        _regular_file(path, "frozen artifact")
        relative = path.resolve().relative_to(repository_root.resolve()).as_posix()
        if _prohibited_relative(relative):
            raise FreezeError(f"prohibited private artifact in frozen bundle: {relative}")
        actual_paths.add(relative)
    if actual_paths != observed_paths:
        missing = sorted(observed_paths - actual_paths)
        extra = sorted(actual_paths - observed_paths)
        raise FreezeError(
            "frozen artifact set changed: "
            f"missing={missing!r} extra={extra!r}"
        )
    return payload


def require_truth_open_allowed(
    *,
    receipt_path: Path,
    repository_root: Path,
    truth_path: Path,
) -> dict[str, Any]:
    """Verify frozen state before a caller may read a private truth file."""

    payload = verify_freeze_receipt(receipt_path, repository_root=repository_root)
    _regular_file(truth_path, "truth sidecar")
    frozen_root = (repository_root / payload["frozen_root"]).resolve()
    try:
        truth_path.resolve().relative_to(frozen_root)
    except ValueError:
        return payload
    raise FreezeError("truth sidecar must remain outside the frozen reconstruction root")
