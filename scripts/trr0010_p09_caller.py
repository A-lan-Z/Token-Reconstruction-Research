"""Fail-closed production caller helpers for TRR-0010 and TRR-P09.

The A2 runner owns streamed-bank loading, schedule iteration, validation,
checkpoint selection, and resource guards.  This module only binds immutable
handoff metadata, constructs the directional hook/optimizer, and provides
create-only model-state/export callbacks.  It intentionally has no training
loop, sampler, public-data loader, or truth access.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from trr0010_model import (
    DEFAULT_DIRECTIONAL_LEARNING_RATE,
    DirectionalTokenReadout,
    build_directional_from_base,
    export_effective_embedding,
    save_directional_state,
)
from token_reconstruction.trr0007_positionwise import (
    RESIDUAL_MLP_METHOD_ID,
    ResidualMLPPositionwiseDecoder,
    save_positionwise_state,
)
from trr0010_p09_integration import validate_p09_module


TASK_ID = "TRR-0010"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_A2_SOURCE_PATHS = (
    "src/token_reconstruction/trr_p09_fixed_control_adapter.py",
    "scripts/trr_p09/fixed_control_runner.py",
    "scripts/trr_p09/prepare_streamed_bank.py",
)


class P09CallerError(ValueError):
    """Raised when production caller bindings are incomplete or inconsistent."""


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise P09CallerError(f"{label} must be a lowercase SHA-256")
    return value


def _commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise P09CallerError(f"{label} must be a hexadecimal commit identity")
    return value


@dataclass(frozen=True)
class FrozenArtifact:
    """Hash-only binding supplied by the final A2 handoff."""

    label: str
    path: str
    bytes: int
    sha256: str
    status: str

    def validate(self, *, required_status: str) -> None:
        if not self.label or not self.path:
            raise P09CallerError("artifact label and path are required")
        if self.bytes <= 0:
            raise P09CallerError(f"{self.label} byte count must be positive")
        _sha(self.sha256, f"{self.label}.sha256")
        if self.status != required_status:
            raise P09CallerError(
                f"{self.label} status must be {required_status!r}, got {self.status!r}"
            )


@dataclass(frozen=True)
class A2SourceBinding:
    path: str
    commit: str
    sha256: str

    def validate(self) -> None:
        if not self.path:
            raise P09CallerError("A2 source path is required")
        _commit(self.commit, f"A2 source {self.path}.commit")
        _sha(self.sha256, f"A2 source {self.path}.sha256")


@dataclass(frozen=True)
class ResourceQualification:
    """Measured numeric guard receipt for the largest directional cell."""

    status: str
    gpu_peak_reserved_bytes: int
    gpu_reserved_limit_bytes: int
    host_peak_rss_bytes: int
    host_rss_limit_bytes: int
    host_available_bytes: int
    host_available_floor_bytes: int
    wall_seconds: float
    wall_limit_seconds: float

    def validate(self) -> None:
        if self.status != "PASS":
            raise P09CallerError("resource qualification is not PASS")
        integer_fields = (
            "gpu_peak_reserved_bytes",
            "gpu_reserved_limit_bytes",
            "host_peak_rss_bytes",
            "host_rss_limit_bytes",
            "host_available_bytes",
            "host_available_floor_bytes",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise P09CallerError(f"resource field {field} is invalid")
        for field in ("wall_seconds", "wall_limit_seconds"):
            value = float(getattr(self, field))
            if not torch.isfinite(torch.tensor(value)).item() or value < 0.0:
                raise P09CallerError(f"resource field {field} is invalid")
        if self.gpu_peak_reserved_bytes > self.gpu_reserved_limit_bytes:
            raise P09CallerError("GPU reserved peak exceeds its frozen limit")
        if self.host_peak_rss_bytes > self.host_rss_limit_bytes:
            raise P09CallerError("host RSS peak exceeds its frozen limit")
        if self.host_available_bytes < self.host_available_floor_bytes:
            raise P09CallerError("host available memory is below its frozen floor")
        if self.wall_seconds > self.wall_limit_seconds:
            raise P09CallerError("qualifier wall time exceeds its frozen limit")


@dataclass(frozen=True)
class ProductionBindings:
    """Final immutable handoff required before constructing a run."""

    contract: FrozenArtifact
    bank_manifest: FrozenArtifact
    schedule: FrozenArtifact
    resource_qualification: ResourceQualification
    a2_sources: tuple[A2SourceBinding, ...]

    def validate(self) -> None:
        self.contract.validate(required_status="FROZEN")
        self.bank_manifest.validate(required_status="VERIFIED")
        self.schedule.validate(required_status="FROZEN")
        self.resource_qualification.validate()
        by_path = {source.path: source for source in self.a2_sources}
        missing = [path for path in REQUIRED_A2_SOURCE_PATHS if path not in by_path]
        if missing:
            raise P09CallerError("A2 source bindings are missing: " + ", ".join(missing))
        if len(by_path) != len(self.a2_sources):
            raise P09CallerError("A2 source bindings contain duplicate paths")
        for source in self.a2_sources:
            source.validate()

    def metadata(self) -> dict[str, Any]:
        self.validate()
        return {
            "contract": self.contract.__dict__.copy(),
            "bank_manifest": self.bank_manifest.__dict__.copy(),
            "schedule": self.schedule.__dict__.copy(),
            "resource_qualification": self.resource_qualification.__dict__.copy(),
            "a2_sources": [source.__dict__.copy() for source in self.a2_sources],
        }

    def source_binding_digest(self) -> str:
        """Digest the imported A2 path/commit/content bindings."""

        self.validate()
        payload = [source.__dict__ for source in self.a2_sources]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _resolve_bound_path(path: str | Path, *, root: Path | None) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute() and root is not None:
        resolved = root / resolved
    return resolved.resolve()


def _verify_file(path: Path, *, expected_bytes: int, expected_sha256: str, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise P09CallerError(f"{label} is not a regular file: {path}")
    actual_bytes = int(path.stat().st_size)
    actual_sha256 = _file_sha256(path)
    if actual_bytes != int(expected_bytes) or actual_sha256 != expected_sha256:
        raise P09CallerError(f"{label} bytes or SHA-256 differs from its frozen binding")
    return {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha256}


def verify_production_bindings(
    bindings: ProductionBindings,
    protocol: Any,
    *,
    source_paths: Mapping[str, str | Path],
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Verify all frozen artifact/source bytes before constructing a run."""

    bindings.validate()
    validate_p09_module(protocol)
    if not isinstance(getattr(protocol, "__file__", None), str):
        raise P09CallerError("A2 protocol module must expose its imported __file__")
    source_by_path = {source.path: source for source in bindings.a2_sources}
    if set(source_paths) != set(REQUIRED_A2_SOURCE_PATHS):
        raise P09CallerError("source_paths must cover exactly the three required A2 files")
    verified_artifacts = []
    for artifact, expected_status in (
        (bindings.contract, "FROZEN"),
        (bindings.bank_manifest, "VERIFIED"),
        (bindings.schedule, "FROZEN"),
    ):
        artifact.validate(required_status=expected_status)
        actual = _verify_file(
            _resolve_bound_path(artifact.path, root=artifact_root),
            expected_bytes=artifact.bytes,
            expected_sha256=artifact.sha256,
            label=artifact.label,
        )
        verified_artifacts.append({"binding": artifact.__dict__.copy(), "actual": actual})
    verified_sources = []
    protocol_path = Path(protocol.__file__).expanduser().resolve()
    for path in REQUIRED_A2_SOURCE_PATHS:
        source = source_by_path[path]
        source_path = _resolve_bound_path(source_paths[path], root=artifact_root)
        if path == REQUIRED_A2_SOURCE_PATHS[0] and source_path != protocol_path:
            raise P09CallerError("imported A2 protocol __file__ differs from its source binding")
        actual = _verify_file(
            source_path,
            expected_bytes=source_path.stat().st_size,
            expected_sha256=source.sha256,
            label=f"A2 source {path}",
        )
        verified_sources.append({"binding": source.__dict__.copy(), "actual": actual})
    receipt = {"artifacts": verified_artifacts, "a2_sources": verified_sources}
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    receipt["verification_sha256"] = hashlib.sha256(encoded).hexdigest()
    return receipt


@dataclass
class DirectionalRuntime:
    decoder: nn.Module
    hook: DirectionalTokenReadout
    optimizer: torch.optim.AdamW
    bindings: ProductionBindings
    binding_receipt: Mapping[str, Any]
    base_learning_rate: float
    weight_decay: float


def _module_device(module: nn.Module) -> torch.device:
    devices = {parameter.device for parameter in module.parameters()}
    if len(devices) != 1:
        raise P09CallerError("decoder parameters must occupy exactly one device")
    return next(iter(devices))


def _validate_hook_device(decoder: nn.Module, hook: DirectionalTokenReadout) -> torch.device:
    if hook.base is not decoder:
        raise P09CallerError("directional hook.base must be the shared decoder object")
    device = _module_device(decoder)
    hook_devices = {parameter.device for parameter in hook.parameters()}
    if hook_devices != {device}:
        raise P09CallerError("directional hook parameters are not on the decoder device")
    return device


def validate_optimizer_configuration(
    optimizer: torch.optim.Optimizer,
    decoder: nn.Module,
    hook: DirectionalTokenReadout,
    *,
    expected_weight_decay: float,
    expected_base_learning_rate: float | None = None,
    expected_current_learning_rates: tuple[float, float] | None = None,
) -> None:
    """Fail closed on optimizer policy, schedule values, and ownership.

    ``expected_base_learning_rate`` is supplied only while constructing the
    optimizer.  The shared runner applies its frozen cosine schedule between
    checkpoints, so callback validation must accept finite scheduled values,
    including the terminal zero, instead of comparing every callback against
    the construction-time rates.  A caller that has an independently bound
    expected schedule point may pass ``expected_current_learning_rates``.
    """

    if optimizer.defaults.get("foreach") is not False:
        raise P09CallerError("AdamW foreach must be explicitly False for bounded scratch")
    expected_weight_decay = float(expected_weight_decay)
    if not torch.isfinite(torch.tensor(expected_weight_decay)).item() or expected_weight_decay < 0.0:
        raise P09CallerError("expected weight decay must be finite and non-negative")
    default_weight_decay = float(optimizer.defaults.get("weight_decay", float("nan")))
    if not torch.isfinite(torch.tensor(default_weight_decay)).item() or default_weight_decay != expected_weight_decay:
        raise P09CallerError("optimizer default weight decay differs from the frozen caller config")
    names = tuple(str(group.get("name", "")) for group in optimizer.param_groups)
    if names != ("decoder", "directional_delta"):
        raise P09CallerError(f"optimizer groups differ: expected decoder/directional_delta, got {names}")

    expected_lrs: tuple[float, float] | None = None
    if expected_base_learning_rate is not None:
        expected_base_learning_rate = float(expected_base_learning_rate)
        if not torch.isfinite(torch.tensor(expected_base_learning_rate)).item() or expected_base_learning_rate <= 0.0:
            raise P09CallerError("expected base learning rate must be finite and positive")
        expected_directional_learning_rate = float(DEFAULT_DIRECTIONAL_LEARNING_RATE)
        if not torch.isfinite(torch.tensor(expected_directional_learning_rate)).item() or expected_directional_learning_rate <= 0.0:
            raise P09CallerError("default directional learning rate must be finite and positive")
        expected_lrs = (expected_base_learning_rate, expected_directional_learning_rate)
    if expected_current_learning_rates is not None:
        if len(expected_current_learning_rates) != 2:
            raise P09CallerError("expected current learning rates must contain two values")
        current = tuple(float(value) for value in expected_current_learning_rates)
        if any(not torch.isfinite(torch.tensor(value)).item() or value < 0.0 for value in current):
            raise P09CallerError("expected current learning rates must be finite and non-negative")
        expected_lrs = current

    for index, group in enumerate(optimizer.param_groups):
        lr = float(group.get("lr", float("nan")))
        group_weight_decay = float(group.get("weight_decay", float("nan")))
        group_foreach = group.get("foreach", optimizer.defaults.get("foreach"))
        if not torch.isfinite(torch.tensor(lr)).item() or lr < 0.0:
            raise P09CallerError(f"optimizer group {index} learning rate must be finite and non-negative")
        if expected_lrs is not None and lr != expected_lrs[index]:
            raise P09CallerError(f"optimizer group {index} learning rate differs from its frozen value")
        if not torch.isfinite(torch.tensor(group_weight_decay)).item() or group_weight_decay != expected_weight_decay:
            raise P09CallerError(f"optimizer group {index} weight decay differs from its frozen value")
        if group_foreach is not False:
            raise P09CallerError(f"optimizer group {index} foreach must be False")

    expected = {id(parameter) for parameter in decoder.parameters() if parameter.requires_grad}
    expected.add(id(hook.delta_rows))
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
        if isinstance(parameter, nn.Parameter) and parameter.requires_grad
    ]
    actual = {id(parameter) for parameter in optimizer_parameters}
    if len(actual) != len(optimizer_parameters):
        raise P09CallerError("optimizer repeats a trainable parameter")
    if actual != expected:
        raise P09CallerError("optimizer parameter ownership differs from decoder plus Delta")
    _validate_hook_device(decoder, hook)


def prepare_directional_runtime(
    *,
    protocol: Any,
    base_decoder: nn.Module,
    support_ids: torch.Tensor | Sequence[int],
    support_counts: torch.Tensor | Sequence[int],
    public_embedding: torch.Tensor,
    bindings: ProductionBindings,
    source_paths: Mapping[str, str | Path],
    base_learning_rate: float,
    weight_decay: float = 0.0,
    artifact_root: Path | None = None,
) -> DirectionalRuntime:
    """Construct one directional arm after all final handoff guards pass."""

    binding_receipt = verify_production_bindings(
        bindings,
        protocol,
        source_paths=source_paths,
        artifact_root=artifact_root,
    )
    if base_learning_rate <= 0.0 or weight_decay < 0.0:
        raise P09CallerError("optimizer rates are invalid")
    device = _module_device(base_decoder)
    if public_embedding.device != device:
        raise P09CallerError("public embedding must already be staged on decoder device")
    hook = build_directional_from_base(base_decoder, support_ids, support_counts)
    # The constructor creates Delta and non-persistent buffers on CPU even
    # when the shared decoder is already on CUDA; move the complete hook
    # before binding E-derived statistics.
    hook.to(device)
    hook.bind_embedding_statistics(public_embedding)
    _validate_hook_device(base_decoder, hook)
    groups = hook.optimizer_param_groups(base_decoder, base_learning_rate=base_learning_rate)
    optimizer = torch.optim.AdamW(groups, weight_decay=float(weight_decay), foreach=False)
    validate_optimizer_configuration(
        optimizer,
        base_decoder,
        hook,
        expected_weight_decay=weight_decay,
        expected_base_learning_rate=base_learning_rate,
    )
    return DirectionalRuntime(
        base_decoder,
        hook,
        optimizer,
        bindings,
        binding_receipt,
        float(base_learning_rate),
        float(weight_decay),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256(b"trr0010-caller-state-v1\0")
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def make_serialization_only_checkpoint_callback(
    *,
    output_root: Path,
    runtime: DirectionalRuntime,
    base_state: Mapping[str, Any],
    fit_manifest: Mapping[str, Any],
) -> Any:
    """Return the A2 callback that writes compact model checkpoints only.

    The A2 callback does not receive an optimizer, so this receipt explicitly
    records that optimizer state is external. A caller may not resume updates
    from these files without a separately bound optimizer-state artifact.
    """

    runtime.bindings.validate()
    if not isinstance(base_state.get("sha256"), str) or not isinstance(fit_manifest.get("sha256"), str):
        raise P09CallerError("base_state and fit_manifest must carry SHA-256 bindings")
    root = Path(output_root)

    def callback(point: Mapping[str, Any], decoder: nn.Module, hook: DirectionalTokenReadout) -> Mapping[str, Any]:
        if decoder is not runtime.decoder or hook is not runtime.hook:
            raise P09CallerError("checkpoint callback received a different model or hook")
        _validate_hook_device(decoder, hook)
        validate_optimizer_configuration(
            runtime.optimizer,
            decoder,
            hook,
            expected_weight_decay=runtime.weight_decay,
        )
        try:
            step = int(point["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise P09CallerError("checkpoint point lacks an integer step") from exc
        if step < 0:
            raise P09CallerError("checkpoint step cannot be negative")
        before = _state_digest(hook)
        path = root / f"checkpoint_step_{step:06d}.safetensors"
        result = save_directional_state(
            path,
            hook,
            selected_step=step,
            base_state=base_state,
            fit_manifest=fit_manifest,
            metadata={
                "serialization_only": True,
                "optimizer_state_external": True,
                "optimizer_foreach": False,
                "p09_contract_sha256": runtime.bindings.contract.sha256,
                "p09_bank_sha256": runtime.bindings.bank_manifest.sha256,
                "p09_schedule_sha256": runtime.bindings.schedule.sha256,
                "p09_a2_source_binding_digest": runtime.bindings.source_binding_digest(),
                "runner_point_state_sha256": str(point.get("state_sha256", "")),
            },
        )
        after = _state_digest(hook)
        if before != after:
            raise P09CallerError("checkpoint serialization mutated model state")
        return {
            "checkpoint": result,
            "serialization_only": True,
            "optimizer_state_external": True,
            "resume_requires_separate_optimizer_artifact": True,
        }

    return callback


def restore_selected_and_export(
    *,
    checkpoint_path: Path,
    expected_checkpoint: Mapping[str, Any],
    export_path: Path,
    runtime: DirectionalRuntime,
    public_embedding: torch.Tensor,
    selected_step: int,
) -> dict[str, Any]:
    """Verify/restore one selected state, then export its coherent effective E."""

    if selected_step < 0:
        raise P09CallerError("selected step cannot be negative")
    _validate_hook_device(runtime.decoder, runtime.hook)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise P09CallerError("selected checkpoint must be a regular file")
    bound_path = Path(str(expected_checkpoint.get("path", ""))).expanduser().resolve()
    if bound_path != checkpoint_path:
        raise P09CallerError("selected checkpoint path differs from its binding")
    try:
        expected_bytes = int(expected_checkpoint["bytes"])
        expected_sha256 = _sha(expected_checkpoint["sha256"], "selected checkpoint.sha256")
    except (KeyError, TypeError, ValueError) as exc:
        raise P09CallerError("selected checkpoint binding is incomplete") from exc
    actual_checkpoint = _verify_file(
        checkpoint_path,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
        label="selected checkpoint",
    )
    try:
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
    except Exception as exc:
        raise P09CallerError("selected checkpoint metadata cannot be read") from exc
    expected_metadata = {
        "selected_step": str(selected_step),
        "p09_contract_sha256": runtime.bindings.contract.sha256,
        "p09_bank_sha256": runtime.bindings.bank_manifest.sha256,
        "p09_schedule_sha256": runtime.bindings.schedule.sha256,
        "p09_a2_source_binding_digest": runtime.bindings.source_binding_digest(),
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise P09CallerError(f"selected checkpoint metadata binding differs for {key}")
    state = load_file(str(checkpoint_path), device="cpu")
    try:
        runtime.hook.load_state_dict(state, strict=True)
    except Exception as exc:
        raise P09CallerError("selected checkpoint state differs from the runtime hook") from exc
    if public_embedding.device != _module_device(runtime.decoder):
        raise P09CallerError("public embedding must be on the decoder device")
    exported = export_effective_embedding(
        Path(export_path),
        runtime.hook,
        public_embedding,
        metadata={
            "selected_step": selected_step,
            "selected_checkpoint_sha256": _file_sha256(checkpoint_path),
            "p09_contract_sha256": runtime.bindings.contract.sha256,
            "p09_bank_sha256": runtime.bindings.bank_manifest.sha256,
            "p09_schedule_sha256": runtime.bindings.schedule.sha256,
            "p09_a2_source_binding_digest": runtime.bindings.source_binding_digest(),
            "optimizer_state_external": True,
        },
    )
    return {
        "selected_step": selected_step,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": actual_checkpoint["bytes"],
        "checkpoint_sha256": actual_checkpoint["sha256"],
        "checkpoint_state_digest": _state_digest(runtime.hook),
        "export": exported,
        "resume_requires_separate_optimizer_artifact": True,
    }


def export_selected_base_decoder_state(
    *,
    path: Path,
    runtime: DirectionalRuntime,
    selected_receipt: Mapping[str, Any],
    selected_step: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export the selected decoder base in the established positionwise format.

    The directional checkpoint remains the restart/training artifact.  This
    create-only export deliberately contains only ``runtime.decoder`` so the
    evaluator can load ``state/base_decoder_state`` with the published
    TRR-0007 loader, while ``effective_readout_w`` remains the separate
    deployment readout artifact.
    """

    if not isinstance(runtime.decoder, ResidualMLPPositionwiseDecoder):
        raise P09CallerError("selected base export requires the residual MLP512 decoder")
    if selected_step < 0:
        raise P09CallerError("selected step cannot be negative")
    try:
        receipt_step = int(selected_receipt["selected_step"])
        checkpoint_sha256 = _sha(
            selected_receipt["checkpoint_sha256"],
            "selected receipt checkpoint_sha256",
        )
        checkpoint_path = str(selected_receipt["checkpoint_path"])
    except (KeyError, TypeError, ValueError) as exc:
        raise P09CallerError("selected restore receipt is incomplete") from exc
    if receipt_step != selected_step or not checkpoint_path:
        raise P09CallerError("selected base export does not match the restored checkpoint")
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    try:
        expected_checkpoint_bytes = int(selected_receipt["checkpoint_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise P09CallerError("selected restore receipt lacks checkpoint_bytes") from exc
    _verify_file(
        checkpoint_file,
        expected_bytes=expected_checkpoint_bytes,
        expected_sha256=checkpoint_sha256,
        label="selected restore checkpoint for base export",
    )
    export_metadata: dict[str, Any] = {
        "base_state_role": "BASE_ONLY_SELECTED_DECODER",
        "directional_readout_excluded": True,
        "selected_directional_checkpoint_sha256": checkpoint_sha256,
        "selected_directional_checkpoint_path": checkpoint_path,
        "p09_contract_sha256": runtime.bindings.contract.sha256,
        "p09_bank_sha256": runtime.bindings.bank_manifest.sha256,
        "p09_schedule_sha256": runtime.bindings.schedule.sha256,
        "p09_a2_source_binding_digest": runtime.bindings.source_binding_digest(),
    }
    if metadata:
        export_metadata.update({str(key): value for key, value in metadata.items()})
    return save_positionwise_state(
        Path(path),
        runtime.decoder,
        method_id=RESIDUAL_MLP_METHOD_ID,
        selected_step=selected_step,
        initialization="restored selected directional checkpoint base parameters",
        distribution="TRR-0010 directional continuation; base-only deployment state",
        bottleneck_size=runtime.decoder.bottleneck_size,
        metadata=export_metadata,
    )


__all__ = [
    "A2SourceBinding",
    "DirectionalRuntime",
    "FrozenArtifact",
    "P09CallerError",
    "ProductionBindings",
    "ResourceQualification",
    "make_serialization_only_checkpoint_callback",
    "prepare_directional_runtime",
    "restore_selected_and_export",
    "export_selected_base_decoder_state",
    "validate_optimizer_configuration",
    "verify_production_bindings",
]
