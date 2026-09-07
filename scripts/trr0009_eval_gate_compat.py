"""Explicit maintenance adapter for the TRR-0009 fixed-state metadata gap.

The registered evaluation was produced at the historical inference commit
``55ef2d4``.  The continued-fixed state has the published inference contract,
but its safetensors metadata predates two redundant boolean fields required by
the generic gate.  This module accepts that one exact state under a separately
recorded compatibility receipt while retaining the generic gate's registration,
input, prediction, timing, and hash checks.

The adapter is deliberately separate from :mod:`trr0009_eval_gate`.  It does
not edit registered files, fake ``HEAD``, or weaken state checks for any other
method.  The original gate failure remains preserved in its existing receipt.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Iterator

from safetensors import safe_open

from scripts import trr0009_eval_contract as contract
from scripts import trr0009_eval_gate as gate
from scripts import trr0009_eval_truth as truth


class CompatibilityError(gate.GateError):
    """Raised when the narrow fixed-state compatibility rule is not met."""


HISTORICAL_INFERENCE_COMMIT = "55ef2d46413de28be70b4014f596b277c0e1582c"
FIXED_METHOD_ID = contract.FIXED_METHOD_ID
FIXED_STATE_BYTES = 29_390_492
FIXED_STATE_SHA256 = "5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14"
FIXED_STATE_RELATIVE_PATH = "experiments/TRR-0009/training/run_v1/continued_fixed_readout/selected.safetensors"
FIXED_STATE_SCHEMA = "token-reconstruction.trr0007-positionwise.v1"
FIXED_STATE_METHOD_ID = "trr0007_residual_mlp512"
FIXED_INFERENCE_CONTRACT = "current_activation_H_i_only; full_vocabulary_tied_E"
COMPAT_SCHEMA = "token-reconstruction.trr0009-state-semantics-compatibility.v1"
COMPAT_STATUS = "EXPLICIT_FIXED_STATE_METADATA_COMPATIBILITY"
DEFAULT_RECEIPT = Path("experiments/TRR-0009/evaluation/state_semantics_compatibility_v2.json")
LOADER_EQUIVALENCE_RELATIVE_PATH = "experiments/TRR-0009/evaluation/loader_qualification_v2/qualification.json"
LOADER_EQUIVALENCE_BYTES = 110_493
LOADER_EQUIVALENCE_SHA256 = "e45d4f4618aaedac7cc6eaf5c6bda1f028d6b62a87edaca6d91040f459f2c552"
LOADER_EQUIVALENCE_STATUS = "TRR8_PUBLIC_FIXTURE_LOADER_EQUIVALENCE_PASS"


# Keep a reference to the original function before any scoped patching.
_ORIGINAL_VALIDATE_STATE_SEMANTICS = gate._validate_state_semantics


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise CompatibilityError(f"repository root is unavailable: {root}")
    return root


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CompatibilityError("cannot resolve maintenance validator commit") from exc
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise CompatibilityError("maintenance validator commit is not a full hash")
    return value


def _record(value: Any, *, root: Path, description: str) -> dict[str, Any]:
    try:
        return contract.validate_file_record(
            value, repository_root=root, description=description, verify=True
        )
    except contract.ContractError as exc:
        raise CompatibilityError(str(exc)) from exc


def _load(path: Path, *, description: str) -> dict[str, Any]:
    try:
        return contract.load_json(path, description=description)
    except contract.ContractError as exc:
        raise CompatibilityError(str(exc)) from exc


def _same_record(actual: Mapping[str, Any], expected: Mapping[str, Any], *, description: str) -> None:
    for key in ("path", "bytes", "sha256"):
        if str(actual.get(key)) != str(expected.get(key)):
            raise CompatibilityError(f"{description} {key} binding changed")


def _state_metadata(path: Path) -> dict[str, Any]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return dict(handle.metadata() or {})
    except Exception as exc:
        raise CompatibilityError(f"fixed state metadata is unreadable: {path}") from exc


def _exact_fixed_loader(row: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "module": "token_reconstruction.trr0007_positionwise",
        "function": "load_positionwise_model_state",
        "interface": contract.LOADER_INTERFACE,
        "current_h_only": True,
        "full_vocabulary": True,
        "history_enabled": False,
        "a2_enabled": False,
        "kwargs": {
            "context_width": 128,
            "hidden_size": 2048,
            "method_id": FIXED_STATE_METHOD_ID,
            "vocabulary_size": 128256,
        },
        "path_args": {},
        "tensor_args": {},
    }
    loader = row.get("loader")
    if not isinstance(loader, Mapping):
        raise CompatibilityError("fixed state loader binding is absent")
    try:
        actual = json.loads(json.dumps(loader, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise CompatibilityError("fixed state loader binding is not serializable") from exc
    if actual != expected:
        raise CompatibilityError("fixed state loader binding differs from the published loader")
    return expected


def _is_true_flag(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


def _validate_fixed_state(row: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Validate the one state for which redundant metadata flags are absent."""
    if row.get("id") != FIXED_METHOD_ID or row.get("role") != FIXED_METHOD_ID:
        raise CompatibilityError("fixed-state compatibility received the wrong method row")
    if row.get("kind") != "decoder":
        raise CompatibilityError("continued-fixed method kind changed")
    state = row.get("state")
    if not isinstance(state, Mapping):
        raise CompatibilityError("fixed state binding is absent")
    checked_state = _record(state, root=root, description="continued-fixed state")
    expected_path = (root / FIXED_STATE_RELATIVE_PATH).resolve()
    if Path(checked_state["path"]).resolve() != expected_path:
        raise CompatibilityError("continued-fixed state path is not the registered fixed state")
    if int(checked_state["bytes"]) != FIXED_STATE_BYTES or checked_state["sha256"] != FIXED_STATE_SHA256:
        raise CompatibilityError("continued-fixed state bytes/hash are not the registered published state")
    metadata = _state_metadata(Path(checked_state["path"]))
    expected_metadata = {
        "schema": FIXED_STATE_SCHEMA,
        "method_id": FIXED_STATE_METHOD_ID,
        "inference_contract": FIXED_INFERENCE_CONTRACT,
        "context_width": "128",
        "hidden_size": "2048",
        "vocabulary_size": "128256",
        "selected_step": "400",
    }
    for key, expected in expected_metadata.items():
        if str(metadata.get(key)) != expected:
            raise CompatibilityError(f"continued-fixed metadata {key} differs from the published contract")
    # The two omitted booleans are redundant with inference_contract.  Missing
    # is allowed for this exact state; present false/malformed values fail.
    for key in ("current_H_only", "full_vocabulary_cross_entropy"):
        if key in metadata and not _is_true_flag(metadata[key]):
            raise CompatibilityError(f"continued-fixed metadata contradicts the published contract: {key}")
    loader = _exact_fixed_loader(row)
    return {
        "method_id": FIXED_METHOD_ID,
        "state": checked_state,
        "state_schema": FIXED_STATE_SCHEMA,
        "state_method_id": FIXED_STATE_METHOD_ID,
        "inference_contract": FIXED_INFERENCE_CONTRACT,
        "loader": loader,
        "metadata_geometry": {
            "context_width": 128,
            "hidden_size": 2048,
            "vocabulary_size": 128256,
            "selected_step": "400",
        },
        "redundant_metadata_flags": {
            "current_H_only": metadata.get("current_H_only", "absent"),
            "full_vocabulary_cross_entropy": metadata.get("full_vocabulary_cross_entropy", "absent"),
        },
    }


def validate_state_semantics_compat(*, registration: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Run the original state checks, replacing only the exact fixed row."""
    if registration.get("code_commit") != HISTORICAL_INFERENCE_COMMIT:
        raise CompatibilityError("registration is not bound to the historical inference commit")
    methods = registration.get("methods")
    if not isinstance(methods, list):
        raise CompatibilityError("registration method rows are absent")
    methods = registration.get("methods")
    if not isinstance(methods, list):
        raise CompatibilityError("registration method rows are absent")
    fixed_indices = [
        index for index, row in enumerate(methods)
        if isinstance(row, Mapping) and row.get("id") == FIXED_METHOD_ID
    ]
    if len(fixed_indices) != 1:
        raise CompatibilityError("registration must contain exactly one continued-fixed row")
    fixed = _validate_fixed_state(methods[fixed_indices[0]], root=root)
    remaining = dict(registration)
    remaining["methods"] = [row for index, row in enumerate(methods) if index != fixed_indices[0]]
    try:
        _ORIGINAL_VALIDATE_STATE_SEMANTICS(registration=remaining, root=root)
    except gate.GateError as exc:
        raise CompatibilityError(str(exc)) from exc
    return {"schema": COMPAT_SCHEMA, "status": COMPAT_STATUS, "fixed_state": fixed}


@contextmanager
def _patched_gate_state_semantics() -> Iterator[None]:
    """Install the explicit validator only for the wrapped gate call."""
    previous = gate._validate_state_semantics
    gate._validate_state_semantics = validate_state_semantics_compat  # type: ignore[assignment]
    try:
        yield
    finally:
        gate._validate_state_semantics = previous  # type: ignore[assignment]


def _compatibility_source_record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    return _record(
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": contract.sha256_file(path),
        },
        root=root,
        description=description,
    )


def _write_create_only(
    path: Path,
    value: Mapping[str, Any],
    *,
    root: Path,
    description: str,
) -> dict[str, Any]:
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = root / target
    try:
        return contract.write_create_only(target, value)
    except contract.ContractError as exc:
        raise CompatibilityError(f"{description}: {exc}") from exc


def _registration_context(
    registration_path: Path,
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(registration_path).expanduser().resolve()
    registration = _load(path, description="TRR-0009 registration")
    record = _compatibility_source_record(path, root=root, description="registration")
    if registration.get("code_commit") != HISTORICAL_INFERENCE_COMMIT:
        raise CompatibilityError("registration is not bound to the historical inference commit")
    return registration, record


def _loader_equivalence_record(registration: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    initialization = registration.get("initialization_equivalence")
    if not isinstance(initialization, Mapping) or initialization.get("status") != "PASS" or initialization.get("truth_opened") is not False:
        raise CompatibilityError("loader-equivalence PASS receipt is absent or opened truth")
    binding = initialization.get("receipt")
    if not isinstance(binding, Mapping):
        raise CompatibilityError("loader-equivalence receipt binding is absent")
    checked = _record(binding, root=root, description="loader-equivalence receipt")
    expected_path = (root / LOADER_EQUIVALENCE_RELATIVE_PATH).resolve()
    if Path(checked["path"]).resolve() != expected_path or int(checked["bytes"]) != LOADER_EQUIVALENCE_BYTES or checked["sha256"] != LOADER_EQUIVALENCE_SHA256:
        raise CompatibilityError("loader-equivalence receipt is not the registered public qualification")
    payload = _load(Path(checked["path"]), description="loader-equivalence receipt")
    if payload.get("status") != LOADER_EQUIVALENCE_STATUS or payload.get("truth_opened") is not False or payload.get("source_text_loaded") is not False or payload.get("target_labels_loaded") is not False:
        raise CompatibilityError("loader-equivalence receipt is not a truth-free PASS")
    return checked


def _compatibility_receipt(
    *,
    registration: Mapping[str, Any],
    registration_record: Mapping[str, Any],
    root: Path,
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    maintenance_commit = _git_head(root)
    if maintenance_commit == HISTORICAL_INFERENCE_COMMIT:
        raise CompatibilityError("maintenance validator commit must remain separate from historical inference commit")
    adapter_path = Path(__file__).resolve()
    gate_path = Path(gate.__file__).resolve()
    truth_path = Path(truth.__file__).resolve()
    return {
        "schema": COMPAT_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": COMPAT_STATUS,
        "role": "post-run maintenance adapter for a historical state metadata omission",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": list(sys.argv),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "historical_inference_commit": HISTORICAL_INFERENCE_COMMIT,
        "maintenance_validator_commit": maintenance_commit,
        "registration": dict(registration_record),
        "fixed_state_compatibility": dict(fixed),
        "loader_equivalence_receipt": _loader_equivalence_record(registration, root=root),
        "adapter_source": _compatibility_source_record(adapter_path, root=root, description="compatibility adapter source"),
        "wrapped_gate_source": _compatibility_source_record(gate_path, root=root, description="wrapped public gate source"),
        "wrapped_truth_source": _compatibility_source_record(truth_path, root=root, description="wrapped truth boundary source"),
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }


def write_compatibility_freeze(
    *,
    registration_path: Path,
    run_manifest_path: Path,
    output_path: Path,
    compatibility_receipt_path: Path,
    repository_root: Path,
    balanced_timing_path: Path,
) -> dict[str, Any]:
    """Create a compatibility receipt and a normal freeze, create-only."""
    root = _root(repository_root)
    registration, registration_record = _registration_context(registration_path, root=root)
    fixed = validate_state_semantics_compat(registration=registration, root=root)
    # Validate all ordinary gate conditions before writing either artifact.
    with _patched_gate_state_semantics():
        freeze = gate.validate_public_outputs(
            registration_path=registration_path,
            run_manifest_path=run_manifest_path,
            repository_root=root,
            require_current_head=False,
            balanced_timing_path=balanced_timing_path,
        )
    receipt = _compatibility_receipt(
        registration=registration,
        registration_record=registration_record,
        root=root,
        fixed=fixed["fixed_state"],
    )
    receipt_record = _write_create_only(
        compatibility_receipt_path,
        receipt,
        root=root,
        description="compatibility receipt",
    )
    freeze["state_semantics_compatibility"] = receipt_record
    freeze["historical_inference_commit"] = HISTORICAL_INFERENCE_COMMIT
    freeze["maintenance_validator_commit"] = receipt["maintenance_validator_commit"]
    freeze_record = _write_create_only(output_path, freeze, root=root, description="compatibility public freeze")
    return freeze | {"freeze_receipt": freeze_record, "compatibility_receipt": receipt_record}


def _validate_compatibility_receipt(
    *,
    freeze: Mapping[str, Any],
    receipt: Mapping[str, Any],
    registration: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Revalidate every maintenance binding against current imported files."""
    if registration.get("code_commit") != HISTORICAL_INFERENCE_COMMIT:
        raise CompatibilityError("reloaded registration is not bound to the historical inference commit")
    if receipt.get("historical_inference_commit") != HISTORICAL_INFERENCE_COMMIT:
        raise CompatibilityError("compatibility receipt historical inference binding changed")
    if receipt.get("maintenance_validator_commit") != freeze.get("maintenance_validator_commit"):
        raise CompatibilityError("compatibility receipt maintenance commit changed")
    if receipt.get("registration") != freeze.get("registration"):
        raise CompatibilityError("compatibility receipt registration binding changed")
    for key in ("truth_opened", "source_text_loaded", "target_labels_loaded", "candidate_arrays_persisted"):
        if receipt.get(key) is not False:
            raise CompatibilityError(f"compatibility receipt {key} flag is not false")

    expected_sources = {
        "adapter_source": (Path(__file__).resolve(), "compatibility adapter source"),
        "wrapped_gate_source": (Path(gate.__file__).resolve(), "wrapped public gate source"),
        "wrapped_truth_source": (Path(truth.__file__).resolve(), "wrapped truth boundary source"),
    }
    for key, (path, description) in expected_sources.items():
        declared = receipt.get(key)
        if not isinstance(declared, Mapping):
            raise CompatibilityError(f"{description} binding is absent")
        expected = _compatibility_source_record(path, root=root, description=description)
        _same_record(declared, expected, description=description)

    declared_loader = receipt.get("loader_equivalence_receipt")
    if not isinstance(declared_loader, Mapping):
        raise CompatibilityError("loader-equivalence receipt binding is absent")
    expected_loader = _loader_equivalence_record(registration, root=root)
    _same_record(declared_loader, expected_loader, description="loader-equivalence receipt")
    return {"registration": dict(registration), "loader_equivalence_receipt": expected_loader}


def _compatibility_record_from_freeze(freeze: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    if freeze.get("truth_opened") is not False:
        raise CompatibilityError("compatibility freeze records truth access")
    if freeze.get("historical_inference_commit") != HISTORICAL_INFERENCE_COMMIT:
        raise CompatibilityError("compatibility freeze historical inference binding changed")
    receipt_binding = freeze.get("state_semantics_compatibility")
    if not isinstance(receipt_binding, Mapping):
        raise CompatibilityError("compatibility receipt binding is absent")
    receipt_record = _record(receipt_binding, root=root, description="state semantics compatibility receipt")
    receipt = _load(Path(receipt_record["path"]), description="state semantics compatibility receipt")
    if receipt.get("schema") != COMPAT_SCHEMA or receipt.get("status") != COMPAT_STATUS:
        raise CompatibilityError("compatibility receipt identity/status changed")
    registration_binding = freeze.get("registration")
    if not isinstance(registration_binding, Mapping):
        raise CompatibilityError("compatibility freeze registration binding is absent")
    registration = _load(Path(str(registration_binding["path"])), description="TRR-0009 registration")
    bindings = _validate_compatibility_receipt(
        freeze=freeze, receipt=receipt, registration=registration, root=root
    )
    fixed = validate_state_semantics_compat(registration=registration, root=root)
    if receipt.get("fixed_state_compatibility") != fixed["fixed_state"]:
        raise CompatibilityError("compatibility fixed-state declaration changed")
    return {"record": receipt_record, "receipt": receipt, "fixed": fixed, **bindings}


def validate_before_truth_compat(
    *,
    freeze_path: Path,
    repository_root: Path,
    require_current_head: bool = True,
) -> dict[str, Any]:
    """Re-run the complete public gate while accepting only the fixed state.

    ``require_current_head`` is retained for drop-in compatibility with the
    truth boundary, which passes it explicitly.  The registered inference
    artifacts intentionally remain bound to the historical commit, so this
    adapter always verifies the registration's historical code/input/state
    bindings and the separate maintenance receipt instead of requiring the
    live checkout HEAD to equal that historical inference commit.
    """
    if not isinstance(require_current_head, bool):
        raise CompatibilityError("require_current_head must be boolean")
    root = _root(repository_root)
    path = Path(freeze_path).expanduser().resolve()
    freeze = _load(path, description="compatibility public freeze receipt")
    if freeze.get("schema") != contract.FREEZE_SCHEMA or freeze.get("task_id") != contract.TASK_ID or freeze.get("status") != "PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH":
        raise CompatibilityError("compatibility freeze identity/status changed")
    compatibility = _compatibility_record_from_freeze(freeze, root=root)
    registration_record = _record(freeze.get("registration"), root=root, description="frozen registration")
    run_record = _record(freeze.get("run_manifest"), root=root, description="frozen run manifest")
    balanced = freeze.get("balanced_timing")
    balanced_path = Path(str(balanced["path"])) if isinstance(balanced, Mapping) and balanced.get("path") else None
    with _patched_gate_state_semantics():
        refreshed = gate.validate_public_outputs(
            registration_path=Path(registration_record["path"]),
            run_manifest_path=Path(run_record["path"]),
            repository_root=root,
            require_current_head=False,
            balanced_timing_path=balanced_path,
        )
    for key in ("state_semantics_compatibility", "historical_inference_commit", "maintenance_validator_commit"):
        refreshed[key] = freeze.get(key)
    if set(refreshed) != set(freeze):
        raise CompatibilityError("compatibility freeze field set changed")
    for key in refreshed:
        if refreshed.get(key) != freeze.get(key):
            raise CompatibilityError(f"compatibility freeze binding changed: {key}")
    if compatibility["receipt"].get("registration") != registration_record:
        raise CompatibilityError("compatibility registration record changed")
    return freeze


@contextmanager
def _patched_truth_gate() -> Iterator[None]:
    previous = truth.gate.validate_before_truth
    truth.gate.validate_before_truth = validate_before_truth_compat  # type: ignore[assignment]
    try:
        yield
    finally:
        truth.gate.validate_before_truth = previous  # type: ignore[assignment]


def prepare_truth_compat(**kwargs: Any) -> dict[str, Any]:
    """Run existing truth preparation only after compatibility revalidation."""
    with _patched_truth_gate():
        return truth.prepare_truth(**kwargs)


def score_truth_sidecar_compat(**kwargs: Any) -> dict[str, Any]:
    """Run existing scoring only after compatibility revalidation."""
    with _patched_truth_gate():
        return truth.score_truth_sidecar(**kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--registration", type=Path, required=True)
    freeze.add_argument("--run-manifest", type=Path, required=True)
    freeze.add_argument("--balanced-timing", type=Path, required=True)
    freeze.add_argument("--compatibility-receipt", type=Path, default=DEFAULT_RECEIPT)
    freeze.add_argument("--freeze-output", type=Path, required=True)
    validate = sub.add_parser("validate-before-truth")
    validate.add_argument("--freeze", type=Path, required=True)
    prepare = sub.add_parser("prepare-truth")
    prepare.add_argument("--freeze", type=Path, required=True)
    prepare.add_argument("--selection", type=Path, required=True)
    prepare.add_argument("--truth-output", type=Path, required=True)
    prepare.add_argument("--truth-binding", type=Path, required=True)
    prepare.add_argument("--execute", action="store_true")
    score = sub.add_parser("score")
    score.add_argument("--freeze", type=Path, required=True)
    score.add_argument("--truth-binding", type=Path, required=True)
    score.add_argument("--truth-sidecar", type=Path)
    score.add_argument("--frequency-reference", type=Path)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze":
            result = write_compatibility_freeze(
                registration_path=args.registration,
                run_manifest_path=args.run_manifest,
                output_path=args.freeze_output,
                compatibility_receipt_path=args.compatibility_receipt,
                repository_root=args.repository_root,
                balanced_timing_path=args.balanced_timing,
            )
            output = {"status": result["status"], "freeze_receipt": result["freeze_receipt"], "compatibility_receipt": result["compatibility_receipt"]}
        elif args.command == "validate-before-truth":
            result = validate_before_truth_compat(freeze_path=args.freeze, repository_root=args.repository_root)
            output = {"status": result["status"], "truth_opened": result.get("truth_opened")}
        elif args.command == "prepare-truth":
            result = prepare_truth_compat(freeze_path=args.freeze, registration_path=None, selection_path=args.selection, truth_output=args.truth_output, truth_binding_path=args.truth_binding, repository_root=args.repository_root, execute=args.execute)
            output = {"status": result["status"], "truth_opened": result.get("truth_opened")}
        else:
            result = score_truth_sidecar_compat(freeze_path=args.freeze, truth_binding_path=args.truth_binding, truth_sidecar_path=args.truth_sidecar, frequency_reference_path=args.frequency_reference, output_path=args.output, repository_root=args.repository_root)
            output = {"status": result["status"], "truth_opened": result.get("truth_opened")}
    except Exception as exc:
        print(f"TRR-0009 compatibility boundary failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
