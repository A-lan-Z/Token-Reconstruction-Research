"""Fail-closed public inputs for the reconstruction process."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path


class SeparationError(ValueError):
    """Raised when a reconstruction input could expose evaluator-private state."""


_FORBIDDEN_PATH_FRAGMENTS = (
    "truth",
    "oracle",
    "evaluator_private",
    "target_lora",
    "source_token",
)


def _validate_public_path(path: Path, *, kind: str) -> None:
    lowered = "/".join(path.parts).casefold()
    if any(fragment in lowered for fragment in _FORBIDDEN_PATH_FRAGMENTS):
        raise SeparationError(f"{kind} path contains an evaluator-private fragment")
    if path.is_symlink():
        raise SeparationError(f"{kind} path may not be a symbolic link")


@dataclass(frozen=True)
class ReconstructionInputs:
    """The complete reconstruction interface; deliberately contains no truth."""

    observation_index: Path
    inverse_directory: Path
    plan_path: Path
    output_directory: Path
    model_id: str
    model_revision: str

    def validate(self) -> None:
        for path, kind in (
            (self.observation_index, "observation index"),
            (self.inverse_directory, "inverse directory"),
            (self.plan_path, "plan"),
            (self.output_directory, "output directory"),
        ):
            _validate_public_path(path, kind=kind)
        if not self.observation_index.is_file():
            raise SeparationError("observation index must exist")
        if not self.inverse_directory.is_dir():
            raise SeparationError("inverse directory must exist")
        if not self.plan_path.is_file():
            raise SeparationError("plan must exist")
        if self.output_directory.exists():
            raise SeparationError("output directory must be create-only")
        if not self.model_id or len(self.model_revision) != 40:
            raise SeparationError("pinned public model identity is required")


def reconstruction_input_fields() -> tuple[str, ...]:
    return tuple(field.name for field in fields(ReconstructionInputs))
