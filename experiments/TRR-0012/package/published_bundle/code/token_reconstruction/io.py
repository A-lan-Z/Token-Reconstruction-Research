"""Create-only serialization for permitted boundary observations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save
import torch

from .access import BoundaryObservation, OBSERVATION_SCHEMA


class ObservationIOError(RuntimeError):
    """Raised when an observation cannot be stored or verified safely."""


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_observation(
    observation: BoundaryObservation, path: str | os.PathLike[str]
) -> str:
    """Write a new observation atomically enough to reject reuse or overwrites."""

    observation.validate()
    destination = Path(path)
    if destination.is_symlink():
        raise ObservationIOError("refusing to write through a symbolic link")
    destination.parent.mkdir(parents=True, exist_ok=True)

    tensors = {
        "activation": observation.activation.detach().to(device="cpu").contiguous(),
        "attention_mask": observation.attention_mask.detach()
        .to(device="cpu", dtype=torch.uint8)
        .contiguous(),
        "position_ids": observation.position_ids.detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous(),
    }
    metadata = {
        "schema": OBSERVATION_SCHEMA,
        "cut_depth": str(observation.cut_depth),
        "source_id": observation.source_id,
        "metadata_json": json.dumps(
            dict(observation.metadata), sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    }
    payload = save(tensors, metadata=metadata)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    file_descriptor: int | None = None
    created = False
    try:
        file_descriptor = os.open(destination, flags, 0o600)
        created = True
        with os.fdopen(file_descriptor, "wb") as handle:
            file_descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ObservationIOError(f"observation already exists: {destination}") from exc
    except OSError as exc:
        if created:
            destination.unlink(missing_ok=True)
        raise ObservationIOError(f"failed to create observation: {destination}") from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
    return sha256_file(destination)


def load_observation(path: str | os.PathLike[str]) -> BoundaryObservation:
    """Load and validate exactly the fields allowed by the observation schema."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ObservationIOError("observation must be a regular, existing file")
    try:
        with safe_open(source, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != {"activation", "attention_mask", "position_ids"}:
                raise ObservationIOError(f"unexpected tensor fields: {sorted(keys)}")
            raw_metadata: dict[str, Any] = handle.metadata() or {}
            if set(raw_metadata) != {
                "schema",
                "cut_depth",
                "source_id",
                "metadata_json",
            }:
                raise ObservationIOError("observation metadata fields changed")
            if raw_metadata["schema"] != OBSERVATION_SCHEMA:
                raise ObservationIOError("unsupported observation schema")
            observation = BoundaryObservation(
                activation=handle.get_tensor("activation"),
                attention_mask=handle.get_tensor("attention_mask").to(torch.long),
                position_ids=handle.get_tensor("position_ids").to(torch.long),
                cut_depth=int(raw_metadata["cut_depth"]),
                source_id=raw_metadata["source_id"],
                metadata=json.loads(raw_metadata["metadata_json"]),
            )
    except ObservationIOError:
        raise
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ObservationIOError(f"invalid observation file: {source}") from exc
    observation.validate()
    return observation
