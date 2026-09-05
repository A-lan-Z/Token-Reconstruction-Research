#!/usr/bin/env python3
"""Create a create-only digest manifest for a pinned public model snapshot.

The manifest contains no model data.  It records the absolute snapshot path
and hashes every regular file visible below it, allowing the Track A runner to
verify the public resources before loading them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from token_reconstruction.footing import MODEL_ID, MODEL_REVISION
from token_reconstruction.experiment_runtime import utc_now


SCHEMA = "token-reconstruction.public-model-resource-manifest.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    *,
    model_path: Path,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    root = model_path.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"model snapshot must be a regular directory: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if not relative or relative.startswith("/") or ".." in relative.split("/"):
            raise RuntimeError(f"unsafe model resource path: {relative}")
        rows.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise RuntimeError("model snapshot has no regular files")
    if not any(row["path"].casefold().endswith(".safetensors") for row in rows):
        raise RuntimeError("model snapshot has no safetensors weights")
    if not any(row["path"].casefold().endswith("config.json") for row in rows):
        raise RuntimeError("model snapshot has no config.json")
    return {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "model": {"id": model_id, "revision": revision},
        "snapshot_path": str(root),
        "files": rows,
    }


def write_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    args = parser.parse_args()
    payload = build_manifest(
        model_path=args.model_path,
        model_id=args.model_id,
        revision=args.revision,
    )
    write_create_only(args.output, payload)
    print(
        json.dumps(
            {
                "status": "CREATED",
                "output": str(args.output.resolve()),
                "files": len(payload["files"]),
                "snapshot_path": payload["snapshot_path"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
