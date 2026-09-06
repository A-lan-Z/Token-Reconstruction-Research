"""TRR-0007 source-free evaluation contract.

This module owns the immutable interface between public observation capture,
the variable-method prediction runner, the pre-truth gate, and the scorer.
It deliberately contains no truth loader.  Structural validation does not
stat or hash assets; the runner/gate call the same validators with asset
verification enabled before any truth binding can be used.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch
from safetensors import safe_open


TASK_ID = "TRR-0007"
PLAN_SCHEMA = "token-reconstruction.trr0007-evaluation-plan.v1"
REGISTRATION_SCHEMA = "token-reconstruction.trr0007-frozen-evaluation-registration.v1"
OBSERVATION_SCHEMA = "token-reconstruction.trr0007-public-observation-manifest.v1"
PREDICTION_SCHEMA = "token-reconstruction.trr0007-prediction.v1"
TIMING_SCHEMA = "token-reconstruction.trr0007-prediction-timing.v1"
RUN_SCHEMA = "token-reconstruction.trr0007-prediction-run.v1"
FREEZE_SCHEMA = "token-reconstruction.trr0007-freeze.v1"
SCORE_SCHEMA = "token-reconstruction.trr0007-score.v1"
METHOD_FREEZE_SCHEMA = "token-reconstruction.trr0007-method-freeze.v1"
METHOD_FREEZE_STATUS = "FROZEN_EVALUATION_DESIGN_BEFORE_SOURCE_SELECTION"
SOURCE_SELECTION_SCHEMA = "token-reconstruction.trr0007-source-selection.v1"
SOURCE_SELECTION_STATUS = "FROZEN_TRR0007_SOURCE_SELECTION_NO_TRUTH"
CAPTURE_SCHEMA = "token-reconstruction.trr0007-public-capture.v1"
CAPTURE_STATUS = "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH"
FREQUENCY_REFERENCE_SCHEMA = "token-reconstruction.trr0005-frequency-references.v1"
FREQUENCY_REFERENCE_PATH = "experiments/TRR-0005/frequency_references_v1.json"

CELL_ORDER = (
    "pile__public_base",
    "pile__public_lora_2601",
    "finance__public_base",
    "finance__public_lora_2601",
)
DOMAIN_ORDER = ("pile", "finance")
TARGET_ORDER = ("public_base", "public_lora_2601")
BASE_CELL_ORDER = ("pile__public_base", "finance__public_base")

REFERENCE_METHOD_ID = "trr6__enriched_trained_diagonal_attention128"
STUDENT_METHOD_IDS = (
    "current_enriched__trained_diagonal",
    "current_enriched__residual_mlp512",
    "improved_public_bank__trained_diagonal",
    "improved_public_bank__residual_mlp512",
)
ANCHOR_METHOD_ID = "bounded_a1_a2_k256_p0"
METHOD_ORDER = (REFERENCE_METHOD_ID, *STUDENT_METHOD_IDS, ANCHOR_METHOD_ID)
STUDENT_METHOD_MODEL_IDS = {
    "current_enriched__trained_diagonal": "trr0007_current_positionwise",
    "current_enriched__residual_mlp512": "trr0007_residual_mlp512",
    "improved_public_bank__trained_diagonal": "trr0007_current_positionwise",
    "improved_public_bank__residual_mlp512": "trr0007_residual_mlp512",
}
STUDENT_SUPPORT = {
    "current_enriched__trained_diagonal": "current_enriched",
    "current_enriched__residual_mlp512": "current_enriched",
    "improved_public_bank__trained_diagonal": "improved_public_bank",
    "improved_public_bank__residual_mlp512": "improved_public_bank",
}
STUDENT_CAPACITY = {
    "current_enriched__trained_diagonal": "trained_diagonal",
    "current_enriched__residual_mlp512": "residual_mlp512",
    "improved_public_bank__trained_diagonal": "trained_diagonal",
    "improved_public_bank__residual_mlp512": "residual_mlp512",
}

REFERENCE_STATE_PATH = (
    "experiments/TRR-0005/joint_fit_v1/enriched/"
    "affine_trained_diagonal_attention128/selected.safetensors"
)
REFERENCE_STATE_SHA256 = "696eb9fc951e85356a06575faf18a2011616692a086bdac3b2fa368e69d599a2"
REFERENCE_STATE_BYTES = 20990652
PUBLIC_E_PATH = (
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/"
    "TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors"
)
PUBLIC_E_SHA256 = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"
PUBLIC_E_BYTES = 1050673488

CAPTURE_BATCH_RECORDS = 8
CAPTURE_SEQUENCE_TOKENS = 192
STORED_SEQUENCE_TOKENS = 128
SCORED_SEQUENCE_TOKENS = 128
SCORED_POST_BOS_TOKENS = 127
RECORDS_PER_DOMAIN = 128
ANCHOR_RECORDS_PER_DOMAIN = 32
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
BOS_TOKEN_ID = 128000
INVALID_TOKEN_ID = -1
FIT_POST_BOS_POSITIONS = 124371
FIT_TRAINING_DRAWS = 1_536_000
A2_PROPOSAL_K = 512
A2_K = 256

MIN_FREE_GPU_BYTES = 8 * 2**30
MAX_RESERVED_GPU_BYTES = 6 * 2**30
MAX_RSS_BYTES = 16 * 2**30
MIN_HOST_AVAILABLE_BYTES = 10 * 2**30
MAX_SECONDS = 1800

NUMERICAL_SETTINGS = {
    "activation_input_dtype": "torch.bfloat16",
    "staged_activation_dtype": "torch.float32",
    "staged_mask_dtype": "torch.bool",
    "decoder_compute_dtype": "torch.float32",
    "embedding_dtype": "torch.float32",
    "autocast": False,
    "cuda_matmul_allow_tf32": False,
    "cuda_cudnn_allow_tf32": True,
    "float32_matmul_precision": "highest",
    "cpu_intraop_threads": 8,
    "cpu_interop_threads": 32,
}
CODE_BINDING_SPECS = (
    ("evaluation_contract", "scripts/trr0007_eval_contract.py"),
    ("evaluation_register", "scripts/trr0007_eval_register.py"),
    ("evaluation_selector", "scripts/trr0007_eval_select.py"),
    ("final_bank_ledger", "scripts/trr0007_bank_ledger.py"),
    ("evaluation_capture", "scripts/trr0007_eval_capture.py"),
    ("evaluation_truth", "scripts/trr0007_eval_truth.py"),
    ("evaluation_runner", "scripts/trr0007_eval_runner.py"),
    ("evaluation_gate", "scripts/trr0007_eval_gate.py"),
    ("evaluation_scorer", "scripts/trr0007_score.py"),
    ("retained_decoder_numerics", "src/token_reconstruction/trr0005_joint_decoder.py"),
    ("positionwise_numerics", "src/token_reconstruction/trr0007_positionwise.py"),
    ("a1_a2_adapter_numerics", "scripts/trr0003_footing_compare.py"),
)
LOADER_ALLOWLIST = {
    ("token_reconstruction.trr0005_joint_decoder", "load_decoder_state"),
    ("token_reconstruction.trr0007_positionwise", "load_positionwise_model_state"),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class ContractError(ValueError):
    """Raised when a TRR-0007 binding or artifact is incomplete."""


def sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"asset is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_digest(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError("value cannot be canonically encoded") from exc
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{description} must be a JSON object")
    return value


def resolve_path(value: Any, *, repository_root: Path, description: str, require_file: bool = True) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{description} path is absent")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(repository_root).expanduser().resolve() / path
    path = path.resolve()
    if path.is_symlink():
        raise ContractError(f"{description} is a symlink: {path}")
    if require_file and not path.is_file():
        raise ContractError(f"{description} is unavailable: {path}")
    return path


def validate_file_record(
    value: Mapping[str, Any],
    *,
    repository_root: Path,
    description: str,
    verify: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{description} binding is malformed")
    path = resolve_path(value.get("path"), repository_root=repository_root, description=description)
    raw_bytes = value.get("bytes")
    digest = value.get("sha256")
    if isinstance(raw_bytes, bool) or not isinstance(raw_bytes, int) or raw_bytes <= 0:
        raise ContractError(f"{description} byte count is invalid")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ContractError(f"{description} SHA-256 is invalid")
    record = {"path": str(path), "bytes": int(raw_bytes), "sha256": digest}
    if verify:
        actual = {"bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
        if actual != {"bytes": record["bytes"], "sha256": record["sha256"]}:
            raise ContractError(f"{description} binding does not match {path}")
    return record


def _method_state_descriptor(binding: Any, *, method_id: str) -> Mapping[str, Any]:
    """Extract one selected-state descriptor from a frozen method binding."""

    if isinstance(binding, Mapping):
        for key in ("state", "selected_state", "state_binding"):
            value = binding.get(key)
            if isinstance(value, Mapping):
                return value
        value = binding.get("method_state")
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], Mapping):
            return value[0]
        if all(key in binding for key in ("path", "bytes", "sha256")):
            return binding
    raise ContractError(f"method freeze state descriptor is absent: {method_id}")


def load_method_freeze(
    path: Path,
    *,
    repository_root: Path,
    verify_assets: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Load and verify the immutable TRR-0007 method/state preselection ledger.

    The ledger is the authority for selected student state paths and hashes. A
    caller may not replace it with a bare digest or a separately supplied state
    path.  The four TRR-0007 students must be present; extra reference metadata
    is allowed so the ledger can retain its broader provenance.
    """

    resolved = Path(path).expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ContractError(f"method freeze is unavailable: {resolved}")
    record = validate_file_record(
        {
            "path": str(resolved),
            "bytes": int(resolved.stat().st_size),
            "sha256": sha256_file(resolved),
        },
        repository_root=repository_root,
        description="TRR-0007 method freeze",
        verify=verify_assets,
    )
    payload = load_json(resolved, description="TRR-0007 method freeze")
    if payload.get("schema") != METHOD_FREEZE_SCHEMA or payload.get("task_id") != TASK_ID:
        raise ContractError("TRR-0007 method freeze identity changed")
    if payload.get("status") != METHOD_FREEZE_STATUS:
        raise ContractError("TRR-0007 method freeze is not frozen")
    for key in (
        "truth_opened",
        "fresh_evaluation_started",
        "source_accessed",
        "target_loaded",
        "target_labels_loaded",
        "private_or_truth_payload_read",
    ):
        if payload.get(key) is True:
            raise ContractError(f"method freeze records forbidden access: {key}")
    declared_ids = payload.get("method_ids")
    if not isinstance(declared_ids, list) or not all(isinstance(value, str) for value in declared_ids):
        raise ContractError("method freeze method_ids are absent")
    if not set(STUDENT_METHOD_IDS).issubset(set(declared_ids)):
        raise ContractError("method freeze does not contain all four TRR-0007 students")
    bindings = payload.get("state_bindings")
    if not isinstance(bindings, Mapping):
        methods = payload.get("methods")
        if isinstance(methods, list):
            bindings = {
                str(row.get("id")): row
                for row in methods
                if isinstance(row, Mapping) and isinstance(row.get("id"), str)
            }
    if not isinstance(bindings, Mapping):
        raise ContractError("method freeze state_bindings are absent")
    states: dict[str, dict[str, Any]] = {}
    for method_id in STUDENT_METHOD_IDS:
        binding = bindings.get(method_id)
        descriptor = _method_state_descriptor(binding, method_id=method_id)
        actual = validate_file_record(
            descriptor,
            repository_root=repository_root,
            description=f"{method_id} frozen state",
            verify=verify_assets,
        )
        declared_hash = None
        if isinstance(binding, Mapping):
            declared_hash = binding.get("state_sha256")
        if declared_hash is None:
            declared_hash = descriptor.get("state_sha256")
        if declared_hash is not None and declared_hash != actual["sha256"]:
            raise ContractError(f"{method_id} frozen state SHA-256 differs from file")
        states[method_id] = actual
    return record, payload, states


def _require_hash(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ContractError(f"{description} must be a lowercase SHA-256")
    return value


def _require_commit(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise ContractError(f"{description} must be a full commit hash")
    return value


def _require_int(value: Any, *, description: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{description} must be an integer >= {minimum}")
    return int(value)


def _structural_file_record(value: Any, *, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{description} binding is malformed")
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise ContractError(f"{description} path is absent")
    raw_bytes = value.get("bytes")
    if isinstance(raw_bytes, bool) or not isinstance(raw_bytes, int) or raw_bytes <= 0:
        raise ContractError(f"{description} byte count is invalid")
    return {
        "path": str(value["path"]),
        "bytes": int(raw_bytes),
        "sha256": _require_hash(value.get("sha256"), description=f"{description} SHA-256"),
    }


def _validate_code_bindings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(CODE_BINDING_SPECS):
        raise ContractError("code bindings are incomplete")
    result: list[dict[str, Any]] = []
    for index, (role, expected_path) in enumerate(CODE_BINDING_SPECS):
        row = value[index]
        if not isinstance(row, Mapping) or row.get("role") != role or row.get("path") != expected_path:
            raise ContractError(f"code binding {role} is missing or reordered")
        _structural_file_record(row, description=f"code binding {role}")
        result.append(dict(row))
    return result


def _validate_loader(value: Any, *, description: str, expected: tuple[str, str] | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{description} loader is absent")
    module = value.get("module")
    function = value.get("function")
    if (module, function) not in LOADER_ALLOWLIST:
        raise ContractError(f"{description} loader is not allowlisted")
    if expected is not None and (module, function) != expected:
        raise ContractError(f"{description} loader changed")
    kwargs = value.get("kwargs", {})
    if not isinstance(kwargs, Mapping):
        raise ContractError(f"{description} loader kwargs are malformed")
    return {"module": str(module), "function": str(function), "kwargs": dict(kwargs)}


def _validate_decoder_kwargs(loader: Mapping[str, Any], *, method_id: str) -> None:
    kwargs = loader["kwargs"]
    if kwargs.get("method_id") != method_id:
        raise ContractError(f"{method_id} loader method_id differs")
    expected = {
        "hidden_size": HIDDEN_SIZE,
        "vocabulary_size": VOCAB_SIZE,
        "context_width": STORED_SEQUENCE_TOKENS,
    }
    for key, value in expected.items():
        if kwargs.get(key) != value:
            raise ContractError(f"{method_id} loader geometry changed: {key}")


def _validate_state_binding(
    value: Any,
    *,
    description: str,
    expected_path: str | None = None,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    row = _structural_file_record(value, description=description)
    if expected_path is not None and row["path"] != expected_path:
        raise ContractError(f"{description} path changed")
    if expected_sha256 is not None and row["sha256"] != expected_sha256:
        raise ContractError(f"{description} SHA-256 changed")
    if expected_bytes is not None and row["bytes"] != expected_bytes:
        raise ContractError(f"{description} byte count changed")
    return row


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("task_id") != TASK_ID:
        raise ContractError("evaluation plan identity changed")
    if plan.get("status") not in {
        "DRAFT_PENDING_ROOT_REVIEW",
        "FROZEN_EVALUATION_DESIGN_BEFORE_SOURCE_SELECTION",
    }:
        raise ContractError("evaluation plan has an invalid status")
    parent = plan.get("parent_lineage")
    if not isinstance(parent, Mapping):
        raise ContractError("parent lineage is absent")
    if parent.get("trr0005_fitting_draws_per_arm") != FIT_TRAINING_DRAWS:
        raise ContractError("TRR-0005 fitting draw denominator changed")
    if parent.get("trr0005_fitting_post_bos_positions") != FIT_POST_BOS_POSITIONS:
        raise ContractError("TRR-0005 fitting position denominator changed")
    if any(str(key).startswith("trr0006_fitting_") for key in parent):
        raise ContractError("TRR-0006 fitting fields are misattributed")
    panel = plan.get("panel")
    if not isinstance(panel, Mapping):
        raise ContractError("evaluation panel is absent")
    if panel.get("records_per_domain") != RECORDS_PER_DOMAIN:
        raise ContractError("evaluation panel record count changed")
    if panel.get("domains") != list(DOMAIN_ORDER) or panel.get("target_conditions") != list(TARGET_ORDER):
        raise ContractError("evaluation panel domains or targets changed")
    if panel.get("cell_order") != list(CELL_ORDER):
        raise ContractError("evaluation panel cell order changed")
    if panel.get("capture_sequence_tokens") != CAPTURE_SEQUENCE_TOKENS:
        raise ContractError("capture sequence length changed")
    if panel.get("stored_sequence_tokens") != STORED_SEQUENCE_TOKENS:
        raise ContractError("stored sequence length changed")
    if panel.get("scored_post_bos_tokens_per_record") != SCORED_POST_BOS_TOKENS:
        raise ContractError("post-BOS denominator changed")
    if panel.get("capture_batch_records") != CAPTURE_BATCH_RECORDS:
        raise ContractError("capture batch size changed")
    if panel.get("same_record_ids_across_targets") is not True:
        raise ContractError("target pairing was not declared")
    selection = panel.get("selection")
    if not isinstance(selection, Mapping):
        raise ContractError("selection recipe is absent")
    if selection.get("seed") != 5005 or selection.get("requested_records") != RECORDS_PER_DOMAIN:
        raise ContractError("selection seed or requested count changed")
    ranges = selection.get("source_ranges_half_open")
    if ranges != {"pile": [7000, 10000], "finance": [12000, 20000]}:
        raise ContractError("source ranges changed")
    exclusion = selection.get("exclusions")
    if not isinstance(exclusion, list) or len(exclusion) < 5:
        raise ContractError("fitting/development exclusion list is incomplete")
    anchor = plan.get("a1_a2_anchor")
    if not isinstance(anchor, Mapping):
        raise ContractError("bounded A1+A2 anchor is absent")
    if anchor.get("method_id") != ANCHOR_METHOD_ID or anchor.get("records_per_domain") != ANCHOR_RECORDS_PER_DOMAIN:
        raise ContractError("anchor identity or size changed")
    if anchor.get("target_conditions") != ["public_base"]:
        raise ContractError("anchor target scope changed")
    denominator = anchor.get("denominator")
    if not isinstance(denominator, Mapping):
        raise ContractError("anchor denominator is absent")
    if denominator.get("exact_records_per_domain") != ANCHOR_RECORDS_PER_DOMAIN:
        raise ContractError("anchor exact denominator changed")
    if denominator.get("token_positions_per_domain") != ANCHOR_RECORDS_PER_DOMAIN * SCORED_POST_BOS_TOKENS:
        raise ContractError("anchor token denominator changed")
    if denominator.get("total_token_positions") != 2 * ANCHOR_RECORDS_PER_DOMAIN * SCORED_POST_BOS_TOKENS:
        raise ContractError("anchor total denominator changed")
    if denominator.get("exact_definition") != "all 127 post-BOS token positions must recover for an exact record; BOS is a fixed known diagnostic and is not counted in exact recovery":
        raise ContractError("anchor exact-record definition changed")
    if "public_base" not in str(anchor.get("selection", "")):
        raise ContractError("anchor selection no longer identifies public_base")
    if "float32 CPU" not in str(anchor.get("numerical_port_caveat", "")):
        raise ContractError("A2 CPU numerical-port caveat is absent")
    quality = plan.get("quality_and_cost_decision")
    if not isinstance(quality, Mapping):
        raise ContractError("quality/cost decision is absent")
    for key in ("useful_point_estimate", "margin_evidence", "support_vs_capacity"):
        if not isinstance(quality.get(key), str) or not quality[key]:
            raise ContractError(f"quality decision field is absent: {key}")
    uncertainty = plan.get("uncertainty_and_multiplicity")
    if not isinstance(uncertainty, Mapping) or uncertainty.get("bootstrap_draws") != 50000:
        raise ContractError("uncertainty/multiplicity rule changed")
    if uncertainty.get("token_tail_alpha") != "0.05/64" or uncertainty.get("exact_tail_alpha") != "0.05/128":
        raise ContractError("multiplicity tail allocation changed")
    primary_family = str(uncertainty.get("primary_family", "")).lower()
    required_edges = (
        "support at trained-diagonal capacity",
        "support at residual capacity",
        "capacity on current enriched bank",
        "capacity on improved public bank",
    )
    if "direct factorial edges" not in primary_family or "64 directional bounds" not in primary_family or any(edge not in primary_family for edge in required_edges):
        raise ContractError("primary contrast family is not the four direct factorial edges")
    if "interaction" not in str(uncertainty.get("secondary_descriptive", "")).lower() or "outside" not in str(uncertainty.get("secondary_descriptive", "")).lower():
        raise ContractError("secondary interaction scope is not descriptive")
    gate = plan.get("truth_gate")
    if not isinstance(gate, Mapping) or gate.get("truth_opened") is not False:
        raise ContractError("truth gate is absent or open")
    binding_step = str(gate.get("binding_step", ""))
    if "trr0007_eval_truth.py" not in binding_step or "sidecar" not in binding_step or "once" not in binding_step:
        raise ContractError("truth binding step is not explicit")
    flags = plan.get("status_flags")
    flag_names = (
        "source_selection_started",
        "observation_capture_started",
        "truth_created",
        "truth_opened",
    )
    if not isinstance(flags, Mapping) or any(not isinstance(flags.get(k), bool) for k in flag_names):
        raise ContractError("plan status flags are incomplete")
    if flags.get("truth_created") is True or flags.get("truth_opened") is True:
        raise ContractError("plan status flags indicate forbidden truth progress")
    if flags.get("source_selection_started") is not False or flags.get("observation_capture_started") is not False:
        raise ContractError("execution progress belongs in immutable receipts, not the design plan")
    return dict(plan)


def validate_registration(registration: Mapping[str, Any]) -> dict[str, Any]:
    """Validate frozen matrix shape and method decisions without opening assets."""

    if registration.get("schema") != REGISTRATION_SCHEMA or registration.get("task_id") != TASK_ID:
        raise ContractError("evaluation registration identity changed")
    if registration.get("status") != "FROZEN_EVALUATION_REGISTRATION":
        raise ContractError("evaluation registration is not frozen")
    if registration.get("records_per_domain") != RECORDS_PER_DOMAIN:
        raise ContractError("registration record count changed")
    if registration.get("anchor_records_per_domain") != ANCHOR_RECORDS_PER_DOMAIN:
        raise ContractError("registration anchor count changed")
    if registration.get("cell_order") != list(CELL_ORDER):
        raise ContractError("registration cell order changed")
    if registration.get("method_ids") != list(METHOD_ORDER):
        raise ContractError("registration method order changed")
    _require_hash(registration.get("plan_sha256"), description="plan")
    _structural_file_record(registration.get("plan"), description="evaluation plan")
    _require_commit(registration.get("code_commit"), description="registration code commit")
    if registration.get("truth_opened") is not False or registration.get("truth_created") is not False:
        raise ContractError("registration truth flags are open")
    if registration.get("candidate_arrays_persisted") is not False:
        raise ContractError("candidate arrays must not be persisted")
    geometry = registration.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ContractError("registration geometry is absent")
    expected_geometry = {
        "capture_batch_records": CAPTURE_BATCH_RECORDS,
        "capture_sequence_tokens": CAPTURE_SEQUENCE_TOKENS,
        "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
        "scored_sequence_tokens": SCORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        "hidden_size": HIDDEN_SIZE,
        "vocabulary_size": VOCAB_SIZE,
        "chunk_records": CAPTURE_BATCH_RECORDS,
    }
    for key, value in expected_geometry.items():
        if geometry.get(key) != value:
            raise ContractError(f"registration geometry changed: {key}")
    methods = registration.get("methods")
    if not isinstance(methods, list) or [row.get("id") for row in methods if isinstance(row, Mapping)] != list(METHOD_ORDER):
        raise ContractError("method rows are incomplete or reordered")
    for row in methods:
        if not isinstance(row, Mapping):
            raise ContractError("method row is malformed")
        method_id = row.get("id")
        if method_id == REFERENCE_METHOD_ID:
            if row.get("role") != "reference" or row.get("support") != "current_enriched" or row.get("capacity") != "trained_diagonal":
                raise ContractError("retained reference role changed")
            if row.get("cells") != list(CELL_ORDER) or row.get("records_per_cell") != RECORDS_PER_DOMAIN:
                raise ContractError("retained reference coverage changed")
            _validate_state_binding(
                row.get("state"),
                description="retained reference state",
                expected_path=REFERENCE_STATE_PATH,
                expected_sha256=REFERENCE_STATE_SHA256,
                expected_bytes=REFERENCE_STATE_BYTES,
            )
            loader = _validate_loader(
                row.get("loader"),
                description="retained reference",
                expected=("token_reconstruction.trr0005_joint_decoder", "load_decoder_state"),
            )
            _validate_decoder_kwargs(loader, method_id="affine_trained_diagonal_attention128")
            if row.get("candidate_policy") != "forbidden":
                raise ContractError("reference candidate policy changed")
        elif method_id in STUDENT_METHOD_IDS:
            if row.get("role") != "student" or row.get("support") != STUDENT_SUPPORT[method_id] or row.get("capacity") != STUDENT_CAPACITY[method_id]:
                raise ContractError(f"student factors changed: {method_id}")
            if row.get("cells") != list(CELL_ORDER) or row.get("records_per_cell") != RECORDS_PER_DOMAIN:
                raise ContractError(f"student coverage changed: {method_id}")
            state_binding = _validate_state_binding(row.get("state"), description=f"{method_id} state")
            if row.get("method_freeze_sha256") != registration.get("method_freeze", {}).get("sha256"):
                raise ContractError(f"{method_id} method-freeze state binding changed")
            selected_state_hashes = registration.get("method_freeze_state_sha256")
            if not isinstance(selected_state_hashes, Mapping) or selected_state_hashes.get(method_id) != state_binding["sha256"]:
                raise ContractError(f"{method_id} selected state differs from the method-freeze ledger")
            loader = _validate_loader(
                row.get("loader"),
                description=method_id,
                expected=("token_reconstruction.trr0007_positionwise", "load_positionwise_model_state"),
            )
            _validate_decoder_kwargs(loader, method_id=STUDENT_METHOD_MODEL_IDS[method_id])
            if row.get("candidate_policy") != "forbidden":
                raise ContractError(f"student candidate policy changed: {method_id}")
        elif method_id == ANCHOR_METHOD_ID:
            if row.get("role") != "anchor" or row.get("kind") != "a1_a2":
                raise ContractError("anchor role changed")
            if row.get("cells") != list(BASE_CELL_ORDER) or row.get("records_per_cell") != ANCHOR_RECORDS_PER_DOMAIN:
                raise ContractError("anchor coverage changed")
            if row.get("proposal_budget") != A2_PROPOSAL_K or row.get("candidate_budget") != A2_K:
                raise ContractError("A1+A2 budgets changed")
            if row.get("candidate_policy") != "output_only":
                raise ContractError("A1+A2 output policy changed")
            adapter = row.get("adapter")
            if not isinstance(adapter, Mapping) or adapter.get("kind") != "legacy_trr0003_a1_a2_p0":
                raise ContractError("A1+A2 adapter identity changed")
            if adapter.get("selection_policy") != "fixed_k256_direct_cosine":
                raise ContractError("A1+A2 selection policy changed")
            if adapter.get("proposal_max_k") != A2_PROPOSAL_K or adapter.get("proposal_chunk") != 256:
                raise ContractError("A1+A2 proposal contract changed")
        else:
            raise ContractError(f"unknown method row: {method_id}")
    runtime = registration.get("runtime_assets")
    if not isinstance(runtime, Mapping):
        raise ContractError("runtime assets are absent")
    embedding = runtime.get("normalized_public_E")
    _structural_file_record(embedding, description="normalized public E")
    if embedding.get("sha256") != PUBLIC_E_SHA256 or embedding.get("bytes") != PUBLIC_E_BYTES:
        raise ContractError("normalized public E binding changed")
    if embedding.get("shape") != [VOCAB_SIZE, HIDDEN_SIZE] or embedding.get("dtype") != "torch.float32":
        raise ContractError("normalized public E geometry changed")
    a2_assets = runtime.get("a1_a2")
    if not isinstance(a2_assets, Mapping):
        raise ContractError("A1+A2 runtime assets are absent")
    for key in ("public_model_snapshot", "lens", "reference"):
        value = a2_assets.get(key)
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str) or not value["path"]:
            raise ContractError(f"A1+A2 asset is absent: {key}")
    observation = registration.get("observation_manifest")
    _structural_file_record(observation, description="observation manifest")
    capture_receipt = registration.get("capture_receipt")
    _structural_file_record(capture_receipt, description="capture receipt")
    method_freeze = registration.get("method_freeze")
    _structural_file_record(method_freeze, description="method freeze")
    state_hashes = registration.get("method_freeze_state_sha256")
    if not isinstance(state_hashes, Mapping) or set(state_hashes) != set(STUDENT_METHOD_IDS):
        raise ContractError("method-freeze selected-state hashes are incomplete")
    for method_id in STUDENT_METHOD_IDS:
        _require_hash(state_hashes.get(method_id), description=f"{method_id} method-freeze state hash")
    frequency_reference = registration.get("frequency_reference")
    frequency_record = _structural_file_record(frequency_reference, description="public frequency reference")
    if Path(frequency_record["path"]).name != Path(FREQUENCY_REFERENCE_PATH).name:
        raise ContractError("public frequency reference path changed")
    final_bank = registration.get("final_bank_ledgers")
    if not isinstance(final_bank, Mapping) or final_bank.get("schema") != "token-reconstruction.trr0007-final-bank-ledger.v1" or final_bank.get("task_id") != TASK_ID or final_bank.get("status") != "METADATA_BANK_READY_FOR_REAL_P0_CAPTURE":
        raise ContractError("final v5 bank ledger binding is absent")
    final_files = final_bank.get("files")
    if not isinstance(final_files, Mapping) or set(final_files) != {"exclusion_manifest", "selected_parent_rows", "corpus_plan"}:
        raise ContractError("final v5 bank ledger files are incomplete")
    for key in ("exclusion_manifest", "selected_parent_rows", "corpus_plan"):
        _structural_file_record(final_files.get(key), description=f"final v5 bank {key}")
    if final_bank.get("exclusion_set_counts") != {"record_ids": 2449, "source_row_keys": 1848, "opaque_sequence_or_reservation_digests": 4073}:
        raise ContractError("final v5 bank exclusion counts changed")
    if final_bank.get("selected_parent_rows") != {"rows": 120, "rows_by_domain": {"controlled_pile_context": 60, "controlled_finance_context": 60}} or final_bank.get("source_and_sequence_ledgers_verified") is not True:
        raise ContractError("final v5 bank parent/sequence ledger binding changed")
    source_selection = registration.get("source_selection")
    _structural_file_record(source_selection, description="source selection")
    exclusions = registration.get("exclusion_manifest")
    _structural_file_record(exclusions, description="exclusion manifest")
    timing = registration.get("timing_contract")
    if not isinstance(timing, Mapping) or timing.get("warmup_runs_per_record") != 1 or timing.get("measured_runs_per_record") != 1:
        raise ContractError("timing contract must be one warmup and one measured run")
    if timing.get("repeat_integrity") != "Require warmup and measured predicted IDs to match exactly":
        raise ContractError("timing repeat-integrity rule changed")
    guard = registration.get("resource_guard")
    if not isinstance(guard, Mapping):
        raise ContractError("resource guard is absent")
    for key in ("minimum_free_gpu_bytes", "maximum_reserved_gpu_bytes", "maximum_rss_bytes", "minimum_host_available_bytes", "maximum_seconds"):
        _require_int(guard.get(key), description=f"resource guard {key}")
    settings = registration.get("numerical_settings")
    if not isinstance(settings, Mapping) or dict(settings) != NUMERICAL_SETTINGS:
        raise ContractError("numerical settings changed")
    _validate_code_bindings(registration.get("code_bindings"))
    output_root = registration.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise ContractError("output root is absent")
    if registration.get("source_text_or_target_labels") is not False:
        raise ContractError("public registration contains source/target labels")
    return dict(registration)


def load_registration(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    value = load_json(resolved, description="TRR-0007 evaluation registration")
    value["_path"] = str(resolved)
    value["registration_sha256"] = sha256_file(resolved)
    return validate_registration(value)


def _observation_binding(value: Any, *, cell_id: str, records: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"observation binding is absent: {cell_id}")
    shape = value.get("shape")
    if shape != [records, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE]:
        raise ContractError(f"observation shape changed: {cell_id}")
    for key, expected in (
        ("stored_sequence_tokens", STORED_SEQUENCE_TOKENS),
        ("scored_sequence_tokens", SCORED_SEQUENCE_TOKENS),
        ("scored_post_bos_tokens", SCORED_POST_BOS_TOKENS),
        ("capture_batch_records", CAPTURE_BATCH_RECORDS),
        ("capture_sequence_tokens", CAPTURE_SEQUENCE_TOKENS),
        ("activations_key", "activations"),
        ("attention_mask_key", "attention_mask"),
        ("position_ids_key", "position_ids"),
    ):
        # The trusted producer's existing public descriptors omit the
        # redundant 128-token scored-width field.  Preserve strict checking
        # when present and backfill this fixed geometry for parsed consumers.
        actual = value.get(key, expected) if key == "scored_sequence_tokens" else value.get(key)
        if actual != expected:
            raise ContractError(f"observation geometry/key changed: {cell_id}/{key}")
    value = dict(value)
    value.setdefault("scored_sequence_tokens", SCORED_SEQUENCE_TOKENS)
    if value.get("public_full_forward") is not True:
        raise ContractError(f"public full-forward provenance missing: {cell_id}")
    expected_lora = cell_id.endswith("__public_lora_2601")
    if value.get("producer_only_lora") is not expected_lora:
        raise ContractError(f"LoRA provenance changed: {cell_id}")
    for forbidden in ("token_ids", "source_text", "target_labels", "truth", "labels", "source"):
        if forbidden in value:
            raise ContractError(f"observation contains forbidden field {forbidden}: {cell_id}")
    return dict(value)


def validate_observation_manifest(
    manifest: Mapping[str, Any],
    *,
    registration: Mapping[str, Any],
    repository_root: Path,
    verify_assets: bool = True,
) -> dict[str, Any]:
    if manifest.get("schema") != OBSERVATION_SCHEMA or manifest.get("task_id") != TASK_ID:
        raise ContractError("observation manifest identity changed")
    if manifest.get("status") != "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH":
        raise ContractError("observation manifest is not frozen")
    records = registration["records_per_domain"]
    if manifest.get("records_per_domain") != records or manifest.get("cell_order") != list(CELL_ORDER):
        raise ContractError("observation records or cell order changed")
    for key in ("source_text_loaded", "target_labels_loaded", "truth_opened", "candidate_arrays_persisted"):
        if manifest.get(key) is not False:
            raise ContractError(f"observation manifest flag is open: {key}")
    for forbidden in ("record_ids", "token_ids", "source_text", "target_labels", "truth", "labels"):
        if forbidden in manifest:
            raise ContractError(f"observation manifest contains forbidden field: {forbidden}")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or [row.get("cell_id") for row in cells if isinstance(row, Mapping)] != list(CELL_ORDER):
        raise ContractError("observation cells are incomplete or reordered")
    paired: dict[str, str] = {}
    parsed: dict[str, Any] = {}
    for row in cells:
        if not isinstance(row, Mapping):
            raise ContractError("observation cell row is malformed")
        cell_id = row.get("cell_id")
        if cell_id not in CELL_ORDER:
            raise ContractError(f"unknown observation cell: {cell_id}")
        style, condition = str(cell_id).split("__", 1)
        if row.get("style") != style or row.get("condition") != condition or row.get("records") != records:
            raise ContractError(f"observation cell identity changed: {cell_id}")
        record_hash = _require_hash(row.get("record_ids_sha256"), description=f"record IDs {cell_id}")
        if style in paired and paired[style] != record_hash:
            raise ContractError(f"public-base/LoRA record pairing changed: {style}")
        paired[style] = record_hash
        binding = _observation_binding(row.get("observation"), cell_id=cell_id, records=records)
        file_row = _structural_file_record(binding, description=f"observation {cell_id}")
        if verify_assets:
            validate_file_record(binding, repository_root=repository_root, description=f"observation {cell_id}", verify=True)
        parsed[cell_id] = {
            "cell_id": cell_id,
            "style": style,
            "condition": condition,
            "records": records,
            "record_ids_sha256": record_hash,
            "observation": file_row | {
                key: binding[key] for key in (
                    "shape", "stored_sequence_tokens", "scored_sequence_tokens",
                    "scored_post_bos_tokens", "capture_batch_records",
                    "capture_sequence_tokens", "activations_key",
                    "attention_mask_key", "position_ids_key",
                    "public_full_forward", "producer_only_lora"
                )
            },
        }
    return {
        "records_per_domain": records,
        "cell_order": list(CELL_ORDER),
        "cells": parsed,
        "record_ids_sha256": paired,
    }


def load_observation_manifest(
    registration: Mapping[str, Any],
    *,
    repository_root: Path,
    verify_assets: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    binding = registration["observation_manifest"]
    path = resolve_path(binding["path"], repository_root=repository_root, description="observation manifest")
    actual = {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
    if actual != {"path": str(path), "bytes": binding["bytes"], "sha256": binding["sha256"]}:
        raise ContractError("observation manifest binding does not match file")
    manifest = load_json(path, description="public observation manifest")
    parsed = validate_observation_manifest(
        manifest, registration=registration, repository_root=repository_root, verify_assets=verify_assets
    )
    return manifest, parsed, actual


def normalize_prediction(raw: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(raw, dtype=torch.long).detach().cpu().contiguous()
    mask = torch.as_tensor(valid_mask, dtype=torch.bool).detach().cpu().contiguous()
    if values.ndim != 1 or mask.ndim != 1 or values.shape != mask.shape:
        raise ContractError("prediction and mask geometry differ")
    if mask.numel() != STORED_SEQUENCE_TOKENS or not bool(mask[0].item()):
        raise ContractError("prediction mask must contain BOS and 128 positions")
    output = torch.full_like(values, INVALID_TOKEN_ID)
    output[mask] = values[mask]
    output[0] = BOS_TOKEN_ID
    active = output[mask]
    if active.lt(0).any().item() or active.ge(VOCAB_SIZE).any().item():
        raise ContractError("prediction has an invalid active ID")
    return output


def validate_prediction_tensor(
    predictions: torch.Tensor,
    *,
    records: int,
    sequence_tokens: int = STORED_SEQUENCE_TOKENS,
) -> torch.Tensor:
    value = torch.as_tensor(predictions, dtype=torch.long).detach().cpu().contiguous()
    if tuple(value.shape) != (records, sequence_tokens):
        raise ContractError(f"prediction shape changed: expected {(records, sequence_tokens)}")
    if value[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise ContractError("prediction BOS column changed")
    active = value.ge(0)
    if active[:, 0].logical_not().any().item():
        raise ContractError("prediction BOS became invalid")
    if value[active].ge(VOCAB_SIZE).any().item():
        raise ContractError("prediction active IDs exceed vocabulary")
    if value[~active].ne(INVALID_TOKEN_ID).any().item():
        raise ContractError("prediction invalid rows are not marked -1")
    for row in range(records):
        invalid = (~active[row]).nonzero(as_tuple=False).flatten()
        if invalid.numel() and active[row, int(invalid[0].item()) + 1 :].any().item():
            raise ContractError("prediction padding is not a suffix")
    return value


def validate_prediction_artifact(
    path: Path,
    *,
    registration: Mapping[str, Any],
    cell: Mapping[str, Any],
    method_id: str,
    records: int,
    verify_hash: bool = True,
) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"prediction artifact is unavailable: {path}")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            metadata = dict(handle.metadata() or {})
            allowed = {"predictions"}
            if method_id == ANCHOR_METHOD_ID:
                allowed.add("a1_predictions")
            if not keys.issubset(allowed) or "predictions" not in keys:
                raise ContractError("prediction contains candidate arrays or unexpected tensors")
            if method_id == ANCHOR_METHOD_ID and "a1_predictions" not in keys:
                raise ContractError("A1 diagnostic prediction is absent from the bounded anchor")
            predictions = handle.get_tensor("predictions")
            a1_predictions = handle.get_tensor("a1_predictions") if "a1_predictions" in keys else None
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"prediction artifact is unreadable: {path}") from exc
    if metadata.get("schema") != PREDICTION_SCHEMA or metadata.get("task_id") != TASK_ID:
        raise ContractError("prediction artifact identity changed")
    if metadata.get("cell_id") != cell["cell_id"] or metadata.get("method_id") != method_id:
        raise ContractError("prediction artifact cell/method changed")
    if metadata.get("truth_opened") != "false" or metadata.get("candidate_arrays_persisted") != "false":
        raise ContractError("prediction artifact truth/candidate flags are open")
    if metadata.get("registration_sha256") != registration["registration_sha256"]:
        raise ContractError("prediction registration binding changed")
    if metadata.get("observation_sha256") != cell["observation"]["sha256"]:
        raise ContractError("prediction observation binding changed")
    geometry = metadata.get("geometry_json")
    if not isinstance(geometry, str):
        raise ContractError("prediction geometry metadata is absent")
    try:
        geometry_value = json.loads(geometry)
    except json.JSONDecodeError as exc:
        raise ContractError("prediction geometry metadata is invalid") from exc
    if geometry_value != {
        "records": records,
        "sequence_tokens": STORED_SEQUENCE_TOKENS,
        "hidden_size": HIDDEN_SIZE,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
    }:
        raise ContractError("prediction geometry changed")
    validate_prediction_tensor(predictions, records=records)
    a1_digest = None
    if method_id == ANCHOR_METHOD_ID:
        assert a1_predictions is not None
        validate_prediction_tensor(a1_predictions, records=records)
        a1_digest = tensor_digest(a1_predictions)
    artifact = {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
    return {
        "path": str(path),
        "artifact": artifact,
        "prediction_sha256": tensor_digest(predictions),
        "a1_prediction_sha256": a1_digest,
        "records": records,
        "cell_id": cell["cell_id"],
        "method_id": method_id,
    }


def expected_prediction_path(output_root: Path, *, cell_id: str, method_id: str) -> Path:
    if cell_id not in CELL_ORDER or method_id not in METHOD_ORDER:
        raise ContractError("unknown cell or method for prediction path")
    style, condition = cell_id.split("__", 1)
    return Path(output_root) / style / condition / f"{method_id}.safetensors"


def expected_timing_path(output_root: Path, *, cell_id: str, method_id: str) -> Path:
    return expected_prediction_path(output_root, cell_id=cell_id, method_id=method_id).with_suffix(".run.json")


def expected_method_cells(method_id: str) -> tuple[str, ...]:
    if method_id == ANCHOR_METHOD_ID:
        return BASE_CELL_ORDER
    if method_id in METHOD_ORDER:
        return CELL_ORDER
    raise ContractError(f"unknown method: {method_id}")


def write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise ContractError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise ContractError(f"refusing to overwrite artifact: {path}") from exc
