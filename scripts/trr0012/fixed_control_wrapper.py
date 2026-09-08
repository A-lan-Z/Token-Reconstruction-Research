"""TRR-0012 task wrapper for the documented P09 fixed-readout runner.

This module owns task identity, durable output paths, and the two new model
identities.  The numerical implementation remains in the published P09
caller/runner imported below.  ``build_run_plan`` is metadata-only and is the
safe entry point before the resource window.  ``run_fixed_arm`` accepts
already-restored loaders and tensors from the bank owner and then delegates
one arm to the original runner with the exact fixed recipe.  The former
object-level entry point is disabled; executable fits go through the native
CLI launcher in ``run_fixed_pair.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Prefer a task-local vendor when present. Until then, import the exact
# read-only P09 source from the persistent sibling worktree.
_P09_SOURCE_ROOT = _REPOSITORY_ROOT.parent / "TRR-P10"
_TASK_HAS_P09 = (_REPOSITORY_ROOT / "scripts" / "trr_p09").is_dir() and (
    _REPOSITORY_ROOT / "src" / "token_reconstruction" / "trr_p09_fixed_control_adapter.py"
).is_file()
_PREFERRED_IMPORT_ROOTS = (
    (_REPOSITORY_ROOT / "src", _REPOSITORY_ROOT)
    if _TASK_HAS_P09
    else (_P09_SOURCE_ROOT / "src", _P09_SOURCE_ROOT, _REPOSITORY_ROOT / "src", _REPOSITORY_ROOT)
)
for _root in reversed(_PREFERRED_IMPORT_ROOTS):
    if _root.is_dir() and str(_root) in sys.path:
        sys.path.remove(str(_root))
    if _root.is_dir():
        sys.path.insert(0, str(_root))

from scripts.trr_p09 import fixed_control_caller as _P09_CALLER  # noqa: E402
from scripts.trr_p09 import fixed_control_runner as _P09_RUNNER  # noqa: E402
from scripts.trr_p09.fixed_control_caller import signed_p09_checkpoint_grid  # noqa: E402


TASK_ID = "TRR-0012"
SOURCE_COMMIT = "26e08098135c47a5607aff915e8025b57a747e20"
SOURCE_FILE_SHA256 = {
    "scripts/trr_p09/fixed_control_cli.py": "707848724cff8bdaadbbd287cf61b2eafebf32bd9f40add7645e25bf2800234e",
    "scripts/trr_p09/fixed_control_caller.py": "13410a03b825467479001fa97112ed05c9784b9ad64dd46880bcd8076de53f6b",
    "scripts/trr_p09/fixed_control_runner.py": "fde3db3c9655afca1678f3d2d0641fba5d4130a73fdb77cc2287a38b1741a98e",
    "scripts/trr_p09/b0_immutable_loader.py": "915c53781c24d997eccc7311553c2d80ff0d7e14688d22bb64c41db2f5e43f70",
    "scripts/trr_p09/prepare_streamed_bank.py": "e76a27f94328648f8fe3da6c6fae48fffd27006b213f9b70b976bcb5f4077c4b",
    "scripts/trr_p09/public_validation_loader.py": "3705ff2a58d34512c7f30d9eb2657206fca3621eb9f6fb685b82c12ee008cd0f",
    "src/token_reconstruction/trr_p09_fixed_control_adapter.py": "aa6a8b21fcc03d55583e69940b722bc9a1e4f4caa05e888f7dab8bf02d3e1a41",
    "src/token_reconstruction/trr0007_positionwise.py": "89fcd036a57407c8c49294aa5fd15ccef46f02b6fabcc35435d097f1213de485",
}
STARTING_STATE_SHA256 = "5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14"
PUBLIC_EMBEDDING_SHA256 = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"
SEED = 4010
STEPS = 13000
RECORD_BATCH_SIZE = 8
POSITION_BUDGET = 512
TRAIN_SEQUENCE_TOKENS = 192
VALIDATION_SEQUENCE_TOKENS = 128
VALIDATION_EVERY_METADATA = 1000
HIDDEN_SIZE = 2048
VOCABULARY_SIZE = 128256
LEARNING_RATE = 2.0e-4
WEIGHT_DECAY = 0.0
GRADIENT_CLIP_NORM = 1.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TRR0012WrapperError(ValueError):
    """Raised when the task wrapper contract is incomplete or changed."""


@dataclass(frozen=True)
class ArtifactBinding:
    """Hash-bound path descriptor supplied by the restored-asset owner."""

    label: str
    path: str
    bytes: int
    sha256: str

    def validate(self) -> None:
        if not self.label or not self.path:
            raise TRR0012WrapperError("artifact label and path are required")
        if isinstance(self.bytes, bool) or self.bytes <= 0:
            raise TRR0012WrapperError(f"{self.label} byte count is invalid")
        if _SHA256.fullmatch(self.sha256) is None:
            raise TRR0012WrapperError(f"{self.label} SHA-256 is invalid")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    bank: str
    model_id: str
    expected_valid_positions: int
    expected_records: int


ARM_SPECS: Mapping[str, ArmSpec] = {
    "current_fixed_replication_1": ArmSpec(
        arm="current_fixed_replication_1",
        bank="B0",
        model_id="TRR-0012/current_fixed_replication_1",
        expected_valid_positions=124371,
        expected_records=1200,
    ),
    "expanded_fixed_replication_1": ArmSpec(
        arm="expanded_fixed_replication_1",
        bank="B1",
        model_id="TRR-0012/expanded_fixed_replication_1",
        expected_valid_positions=1243710,
        expected_records=12000,
    ),
}


def _persistent(path: str | Path, *, label: str, output_root: Path | None = None) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise TRR0012WrapperError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if any(part.lower() == "tmp" for part in resolved.parts):
        raise TRR0012WrapperError(f"{label} may not use temporary storage: {resolved}")
    if output_root is not None:
        root = Path(output_root).expanduser().resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise TRR0012WrapperError(f"{label} must be below durable output root {root}") from exc
    return resolved


def _artifact(value: ArtifactBinding | Mapping[str, Any], *, label: str) -> ArtifactBinding:
    if isinstance(value, ArtifactBinding):
        result = value
    elif isinstance(value, Mapping):
        try:
            result = ArtifactBinding(
                label=str(value.get("label", label)),
                path=str(value["path"]),
                bytes=int(value["bytes"]),
                sha256=str(value["sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TRR0012WrapperError(f"{label} binding is malformed") from exc
    else:
        raise TRR0012WrapperError(f"{label} binding is malformed")
    result.validate()
    # Plan construction also rejects temporary references; byte existence is
    # checked separately immediately before an explicit fit.
    _persistent(result.path, label=label)
    return result


def _artifact_map(artifacts: Mapping[str, ArtifactBinding | Mapping[str, Any]]) -> dict[str, ArtifactBinding]:
    required = ("starting_state", "public_embedding", "fit_bank", "schedule", "validation_manifest", "validation_rows")
    if set(artifacts) != set(required):
        missing = sorted(set(required) - set(artifacts))
        extra = sorted(set(artifacts) - set(required))
        raise TRR0012WrapperError(f"artifact set differs; missing={missing}, extra={extra}")
    return {name: _artifact(artifacts[name], label=name) for name in required}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(artifacts: Mapping[str, ArtifactBinding | Mapping[str, Any]]) -> dict[str, Any]:
    """Verify all restored bytes before a caller constructs a fit."""

    checked: dict[str, Any] = {}
    for name, binding in _artifact_map(artifacts).items():
        path = _persistent(binding.path, label=name)
        if not path.is_file():
            raise TRR0012WrapperError(f"{name} is unavailable: {path}")
        actual_bytes = int(path.stat().st_size)
        actual_sha = _sha256_file(path)
        if actual_bytes != binding.bytes or actual_sha != binding.sha256:
            raise TRR0012WrapperError(
                f"{name} changed: bytes {actual_bytes}/{binding.bytes}, SHA-256 {actual_sha}/{binding.sha256}"
            )
        checked[name] = {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha}
    return checked


def build_run_plan(
    *,
    arm: str,
    actual_valid_positions: int,
    artifacts: Mapping[str, ArtifactBinding | Mapping[str, Any]],
    output_root: str | Path = _REPOSITORY_ROOT / "outputs" / TASK_ID,
) -> dict[str, Any]:
    """Build a metadata-only fixed-arm plan; no tensor or model is opened."""

    try:
        spec = ARM_SPECS[arm]
    except KeyError as exc:
        raise TRR0012WrapperError(f"unknown arm: {arm}") from exc
    if isinstance(actual_valid_positions, bool) or actual_valid_positions != spec.expected_valid_positions:
        raise TRR0012WrapperError(
            f"{spec.bank} valid-position count must be {spec.expected_valid_positions}"
        )
    output = _persistent(output_root, label="output root")
    output.mkdir(parents=True, exist_ok=True)
    artifact_map = _artifact_map(artifacts)
    # P09 appends the common terminal update to the bank-derived milestones.
    # This keeps B0 on the same 13,000-update budget as B1 even though B0's
    # position-derived N is only 12,000.
    grid = tuple(sorted(set(int(step) for step in signed_p09_checkpoint_grid(actual_valid_positions)) | {STEPS}))
    if grid != (0, 1000, 2000, 4000, 8000, 12000, 13000):
        raise TRR0012WrapperError(f"signed P09 grid changed: {grid}")
    if artifact_map["starting_state"].sha256 != STARTING_STATE_SHA256:
        raise TRR0012WrapperError("starting state does not bind the exact 5cada... common start")
    if artifact_map["public_embedding"].sha256 != PUBLIC_EMBEDDING_SHA256:
        raise TRR0012WrapperError("public embedding does not bind the signed fixed E")
    return {
        "schema": "token-reconstruction.trr0012-fixed-control-run-plan.v1",
        "task_id": TASK_ID,
        "status": "PLAN_ONLY_PENDING_RESOURCE_RELEASE",
        "source_commit": SOURCE_COMMIT,
        "source_bindings": source_provenance(),
        "arm": spec.arm,
        "bank": spec.bank,
        "model_id": spec.model_id,
        "expected_records": spec.expected_records,
        "actual_valid_positions": actual_valid_positions,
        "output_root": str(output),
        "artifacts": {name: binding.as_dict() for name, binding in artifact_map.items()},
        "training": {
            "seed": SEED,
            "steps": STEPS,
            "record_batch_size": RECORD_BATCH_SIZE,
            "position_budget": POSITION_BUDGET,
            "train_sequence_tokens": TRAIN_SEQUENCE_TOKENS,
            "validation_sequence_tokens": VALIDATION_SEQUENCE_TOKENS,
            "validation_every_metadata": VALIDATION_EVERY_METADATA,
            "checkpoint_grid": list(grid),
            "selection_metric": "domain_balanced_token_accuracy",
            "selection_rule": "earliest strict maximum over the seven grid validations, including step 0",
            "optimizer": "AdamW foreach=False",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "scheduler": "CosineAnnealingLR(T_max=13000)",
            "readout": "fixed_public_E; decoder-only trainable parameters; compute_base_logits=true",
            "loss": "torch.nn.functional.cross_entropy",
        },
        "truth_boundary": {
            "public_fit_support_only": True,
            "private_evaluation_opened": False,
            "P03_holdout_touched": False,
        },
    }


def write_run_plan(plan: Mapping[str, Any], path: str | Path) -> Path:
    """Write a create-only, durable plan receipt."""

    output = _persistent(path, label="run plan")
    if output.exists():
        raise TRR0012WrapperError(f"run plan is create-only: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def source_provenance() -> dict[str, Any]:
    """Bind the files actually imported by this wrapper to frozen hashes."""

    caller_path = Path(_P09_CALLER.__file__).resolve()
    source_root = caller_path.parents[2]
    checked: dict[str, Any] = {}
    for relative, expected in SOURCE_FILE_SHA256.items():
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise TRR0012WrapperError(f"frozen P09 source is unavailable: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise TRR0012WrapperError(f"frozen P09 source changed: {relative}")
        checked[relative] = {"path": str(path), "sha256": actual}
    imported = {
        "scripts/trr_p09/fixed_control_caller.py": str(caller_path),
        "scripts/trr_p09/fixed_control_runner.py": str(Path(_P09_RUNNER.__file__).resolve()),
    }
    for relative, path in imported.items():
        if Path(path).resolve() != source_root / relative:
            raise TRR0012WrapperError(f"imported P09 module path differs: {relative}")
    return {"commit": SOURCE_COMMIT, "files": checked, "imported": imported}


def run_fixed_arm(
    *,
    plan: Mapping[str, Any],
    decoder: Any,
    embedding: Any,
    fit_source: Any,
    schedule_steps: Iterable[Any],
    schedule_semantic_sha256: str,
    schedule_exposure: Mapping[str, Any],
    validation_callback: Callable[..., Mapping[str, Any]],
    checkpoint_callback: Callable[..., Mapping[str, Any] | None] | None,
    deadline_seconds: float | None = None,
    resource_guard_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Reject the former object-level entry point.

    The old signature accepted arbitrary in-memory decoder, embedding, bank,
    schedule, and validation objects.  Keeping it callable would allow a fit
    to bypass the native P09 CLI's hash gates, external watchdog, and durable
    checkpoint contract.  ``run_fixed_pair.py`` is the only executable fit
    adapter for this task.
    """

    raise TRR0012WrapperError(
        "run_fixed_arm is disabled; execute scripts/trr0012/run_fixed_pair.py "
        "so the native P09 CLI owns asset loading and checkpoint selection"
    )


if __name__ == "__main__":
    raise SystemExit("Use build_run_plan from a task-owned launcher; no implicit fit is permitted.")
