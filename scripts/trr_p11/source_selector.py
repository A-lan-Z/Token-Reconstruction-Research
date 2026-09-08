"""Fail-closed trusted-curator source selector for TRR-P11.

The selector validates the frozen replication contract, the complete opaque
identity-union export, and the exact Agent1 B0/B1 bank/development-input
bindings before it reads public rows.  Model-state identities are checked by
the later restore/prediction phase.  Trusted rerender/tokenization is
transient: outputs carry only approved identity metadata and opaque hashes.
Raw signed-int32 H40 is a P11 identity namespace kept separate from the
TRR-0002 tensor-header/int64 H40 namespace.
Capture, prediction, truth, and scoring are separate later phases and are not
invoked here.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

# Keep both ``python scripts/...`` and package imports working from a checkout.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _root in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

TASK_ID = "TRR-P11"
MANIFEST_SCHEMA = "token-reconstruction.trr-p11-replication-manifest.v1"
EXCLUSION_SCHEMA = "token-reconstruction.trr-p11-identity-exclusion-audit.v1"
SELECTION_SCHEMA = "token-reconstruction.trr-p11-source-selection.v1"
SELECTION_STATUS = "FROZEN_TRR-P11_SOURCE_SELECTION_NO_TRUTH"
OBSERVATION_SCHEMA = "token-reconstruction.trr-p11-public-observation-manifest.v1"
CAPTURE_SCHEMA = "token-reconstruction.trr-p11-public-capture.v1"
CAPTURE_STATUS = "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH"
PREDICTION_MANIFEST_SCHEMA = "token-reconstruction.trr-p11-prediction-manifest.v1"
REGISTRATION_SCHEMA = "token-reconstruction.trr-p11-evaluation-registration.v1"
REGISTRATION_STATUS = "FROZEN_EVALUATION_REGISTRATION_BEFORE_TRUTH"
FREEZE_SCHEMA = "token-reconstruction.trr-p11-public-freeze.v1"
FREEZE_STATUS = "P11_PUBLIC_PREDICTIONS_FROZEN_BEFORE_TRUTH"
TRUTH_SCHEMA = "token-reconstruction.trr-p11-truth-binding.v1"
SCORE_SCHEMA = "token-reconstruction.trr-p11-score.v1"

DOMAIN_ORDER = ("pile", "finance")
TARGET_ORDER = ("public_base", "public_lora_2601")
CELL_ORDER = tuple(f"{domain}__{target}" for domain in DOMAIN_ORDER for target in TARGET_ORDER)
METHOD_ORDER = ("new_current_fixed_B0", "new_expanded_fixed_B1")
OPTIONAL_COMPARATOR = "a1_a2_k256"
SELECTION_SEED = 5011
PILE_RANGE = (0, 2000)
FINANCE_RANGE = (20000, 28000)
SOURCE_RANGES = {"pile": PILE_RANGE, "finance": FINANCE_RANGE}
RECORDS_PER_DOMAIN = 256
STORED_SEQUENCE_TOKENS = 128
SCORED_POST_BOS_TOKENS = 127
A1_A2_RECORDS_PER_DOMAIN = 128
CAPTURE_BATCH_RECORDS = 8
CAPTURE_SEQUENCE_TOKENS = 192
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
BOS_TOKEN_ID = 128000
PADDING_TOKEN_ID = 128001
SCORER_RELATIVE_PATH = "scripts/trr0010_analysis.py"
SCORER_LOCAL_RELATIVE_PATH = "scripts/trr_p11/scorer/trr0010_analysis.py"
SCORER_SHA256 = "90078ac78bfcdcfb5f782a598c943b78cb0417f6895906a1e6d61056a3793cef"
SCORER_SOURCE_COMMIT = "70c57db7643913eea97cc606775b3f1f3807967a"
BOOTSTRAP_SEED = 9009
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_ALPHA = 0.025
# Future source IDs, observations, predictions, truth, and scores stay below
# this ignored root.  Compact hash-bound descriptors remain in experiments/.
PRIVATE_EVALUATION_ROOT_RELATIVE = Path("outputs") / TASK_ID / "private-evaluation"

_DATASET_META = {
    "pile": {
        "dataset_id": "NeelNanda/pile-10k",
        "split": "train",
        "revision": "127bfedcd5047750df5ccf3a12979a47bfa0bafa",
    },
    "finance": {
        "dataset_id": "Josephgflowers/Finance-Instruct-500k",
        "split": "train",
        "revision": "583a98fb0ec14d904e9423b671d9d0fea88891b6",
    },
}

_SHA256_HEX = frozenset("0123456789abcdef")
_COMMIT_HEX = frozenset("0123456789abcdef")
_P11_UNION_FIELDS = frozenset({
    "record_id",
    "source_index",
    "rendered_sha256",
    "tokenized_record_sha256",
    "h40_sequence_sha256",
    "h128_sequence_sha256",
    "h129_sequence_sha256",
    "trr0002_active_token_ids_sha256",
    "trr0002_h40_token_ids_sha256",
})
_FORBIDDEN_FALSE_KEYS = (
    "truth_opened",
    "truth_created",
    "target_labels_loaded",
    "source_text_written",
    "token_ids_written",
    "candidate_arrays_persisted",
    "p03_holdout_accessed",
)


class P11PipelineError(ValueError):
    """Raised when a P11 phase cannot satisfy its frozen boundary."""


class SelectionError(P11PipelineError):
    """Raised when trusted-curator selection fails closed."""


class CaptureError(P11PipelineError):
    """Raised when public target capture cannot satisfy its contract."""


class EvaluationError(P11PipelineError):
    """Raised when prediction, truth, or scoring bindings are incomplete."""


@dataclass(frozen=True)
class ExclusionContext:
    """In-memory identity union loaded from an independently audited receipt."""

    audit_record: dict[str, Any]
    audit: dict[str, Any]
    union_record: dict[str, Any]
    union: Any


@dataclass(frozen=True)
class SelectionContext:
    """Validated selection payload, file binding, rows, and source descriptors."""

    payload: dict[str, Any]
    record: dict[str, Any]
    rows: dict[str, list[dict[str, Any]]]
    counts: dict[str, int]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def scorer_contract() -> dict[str, Any]:
    """Return the explicit P11 scorer settings; never rely on scorer defaults."""
    return {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_DRAWS,
        "one_sided_alpha": BOOTSTRAP_ALPHA,
        "exact_route_alpha": 0.025,
    }


def _is_true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


def _is_false(value: Any) -> bool:
    return value is False or (isinstance(value, str) and value.lower() == "false")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P11PipelineError(f"asset is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise P11PipelineError("value is not canonical JSON") from exc


def _json_digest(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _require_sha(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _SHA256_HEX for char in value):
        raise P11PipelineError(f"{description} is not a lowercase SHA-256")
    return value


def _resolve(value: Any, *, root: Path, description: str, require_file: bool = True) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise P11PipelineError(f"{description} path is absent")
    raw = Path(value).expanduser()
    path = (raw if raw.is_absolute() else root / raw).resolve()
    if path.is_symlink():
        raise P11PipelineError(f"{description} is a symlink: {path}")
    if require_file and (not path.exists() or not path.is_file()):
        raise P11PipelineError(f"{description} is unavailable: {path}")
    return path


def _record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    resolved = _resolve(path, root=root, description=description)
    return {"path": str(resolved), "bytes": int(resolved.stat().st_size), "sha256": _sha256_file(resolved)}


def _record_declared(value: Any, *, root: Path, description: str) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        return _record(Path(value), root=root, description=description)
    if not isinstance(value, Mapping):
        raise P11PipelineError(f"{description} binding is malformed")
    actual = _record(Path(str(value.get("path", ""))), root=root, description=description)
    if "bytes" in value:
        try:
            declared_bytes = int(value["bytes"])
        except (TypeError, ValueError) as exc:
            raise P11PipelineError(f"{description} byte binding is malformed") from exc
        if declared_bytes != actual["bytes"]:
            raise P11PipelineError(f"{description} byte binding changed")
    if "sha256" in value and value.get("sha256") != actual["sha256"]:
        raise P11PipelineError(f"{description} hash binding changed")
    return actual


def _load_json(path: Path, *, root: Path, description: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _record(path, root=root, description=description)
    try:
        value = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P11PipelineError(f"{description} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise P11PipelineError(f"{description} must be a JSON object")
    return dict(value), record


def _write_create_only(path: Path, payload: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = root / raw
    path = raw.resolve()
    if path.exists() or path.is_symlink():
        raise P11PipelineError(f"{description} is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:  # pragma: no cover
        raise P11PipelineError(f"{description} is create-only: {path}") from exc
    return _record(path, root=root, description=description)


def _task_output(path: Path, *, root: Path, phase: str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = root / raw
    resolved = raw.resolve()
    allowed_roots = (
        (root / "experiments" / TASK_ID / phase).resolve(),
        (root / PRIVATE_EVALUATION_ROOT_RELATIVE / phase).resolve(),
    )
    if not any(resolved == allowed or allowed in resolved.parents for allowed in allowed_roots):
        rendered = ", ".join(str(value) for value in allowed_roots)
        raise P11PipelineError(f"{phase} output must be below one of: {rendered}")
    if resolved.exists() or resolved.is_symlink():
        raise P11PipelineError(f"{phase} output is create-only: {resolved}")
    return resolved


def _git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise P11PipelineError("cannot resolve current code commit") from exc
    if len(value) != 40 or any(char not in _COMMIT_HEX for char in value):
        raise P11PipelineError("current code commit is malformed")
    return value


def _load_manifest(
    manifest_path: Path,
    *,
    root: Path,
    require_state_bindings: bool = False,
    require_replication_inputs: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, record = _load_json(manifest_path, root=root, description="P11 replication manifest")
    validate_p11_manifest(
        payload,
        require_state_bindings=require_state_bindings,
        require_replication_inputs=require_replication_inputs,
    )
    return payload, record


def _validate_replication_inputs(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate hash-only B0/B1 bank and development-selection identities."""
    raw = manifest.get("replication_inputs")
    if not isinstance(raw, Mapping):
        raise SelectionError("P11 replication_inputs binding is absent")
    status = str(raw.get("status", "")).upper()
    if not status or any(marker in status for marker in ("PENDING", "NOT_BOUND", "UNAVAILABLE")):
        raise SelectionError("P11 replication input identities are not frozen")
    banks = raw.get("banks")
    if not isinstance(banks, Mapping):
        raise SelectionError("P11 replication input banks are absent")
    normalized: dict[str, Any] = {"banks": {}}
    for bank in ("B0", "B1"):
        item = banks.get(bank)
        if not isinstance(item, Mapping) or item.get("bank") != bank:
            raise SelectionError(f"P11 replication input bank binding changed: {bank}")
        bank_normalized: dict[str, Any] = {}
        for key in ("bank_manifest", "ordered_identity"):
            binding = item.get(key)
            if not isinstance(binding, Mapping):
                raise SelectionError(f"P11 {bank} {key} binding is absent")
            path = binding.get("path")
            if not isinstance(path, str) or not path:
                raise SelectionError(f"P11 {bank} {key} path is absent")
            digest = _require_sha(binding.get("sha256"), description=f"P11 {bank} {key}")
            bank_normalized[key] = {"path": path, "sha256": digest}
            if "bytes" in binding:
                try:
                    declared_bytes = int(binding["bytes"])
                except (TypeError, ValueError) as exc:
                    raise SelectionError(f"P11 {bank} {key} byte binding is malformed") from exc
                if declared_bytes < 0:
                    raise SelectionError(f"P11 {bank} {key} byte binding is malformed")
                bank_normalized[key]["bytes"] = declared_bytes
        normalized["banks"][bank] = bank_normalized
    development = raw.get("development_selection")
    if not isinstance(development, Mapping):
        raise SelectionError("P11 development selection binding is absent")
    development_path = development.get("path")
    if not isinstance(development_path, str) or not development_path:
        raise SelectionError("P11 development selection path is absent")
    development_digest = _require_sha(
        development.get("sha256"),
        description="P11 development selection",
    )
    normalized["development_selection"] = {
        "path": development_path,
        "sha256": development_digest,
    }
    if "bytes" in development:
        try:
            declared_bytes = int(development["bytes"])
        except (TypeError, ValueError) as exc:
            raise SelectionError("P11 development selection byte binding is malformed") from exc
        if declared_bytes < 0:
            raise SelectionError("P11 development selection byte binding is malformed")
        normalized["development_selection"]["bytes"] = declared_bytes
    return normalized


def _validate_new_state_bindings(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Require exact Agent1 B0/B1 bank, state, and selection identities."""
    decision = manifest.get("decision")
    if not isinstance(decision, Mapping):
        raise SelectionError("P11 decision binding is absent")
    decision_status = str(decision.get("new_state_identities", "")).upper()
    if not decision_status or any(marker in decision_status for marker in ("PENDING", "NOT_BOUND", "UNAVAILABLE")):
        raise SelectionError("P11 B0/B1 state identities are not bound")
    states = manifest.get("new_model_states")
    if not isinstance(states, Mapping):
        raise SelectionError("P11 new_model_states binding is absent")
    states_status = str(states.get("status", "")).upper()
    if not states_status or any(marker in states_status for marker in ("PENDING", "NOT_BOUND", "UNAVAILABLE")):
        raise SelectionError("P11 new_model_states are not frozen")
    expected = {
        "current_b0": ("B0", "new-b0"),
        "expanded_b1": ("B1", "new-b1"),
    }
    required = (
        "bank",
        "model_id",
        "selected_step",
        "state_id",
        "state_sha256",
        "bank_manifest_sha256",
        "selection_receipt_sha256",
    )
    bound: dict[str, dict[str, Any]] = {}
    receipt_hashes: set[str] = set()
    for arm, (bank, model_id) in expected.items():
        value = states.get(arm)
        if not isinstance(value, Mapping):
            raise SelectionError(f"P11 state identity is absent: {arm}")
        item = dict(value)
        if item.get("bank") != bank or item.get("model_id") != model_id:
            raise SelectionError(f"P11 {arm} bank/model identity changed")
        try:
            selected_step = int(item.get("selected_step", -1))
        except (TypeError, ValueError) as exc:
            raise SelectionError(f"P11 {arm} selected_step is malformed") from exc
        if selected_step < 0 or isinstance(item.get("selected_step"), bool):
            raise SelectionError(f"P11 {arm} selected_step is malformed")
        if not isinstance(item.get("state_id"), str) or not item["state_id"]:
            raise SelectionError(f"P11 {arm} state_id is absent")
        for key in ("state_sha256", "bank_manifest_sha256", "selection_receipt_sha256"):
            _require_sha(item.get(key), description=f"P11 {arm}/{key}")
        item["selected_step"] = selected_step
        bound[arm] = item
        receipt_hashes.add(item["selection_receipt_sha256"])
    if len(receipt_hashes) != 1:
        raise SelectionError("P11 B0/B1 selection receipts are not paired")
    return bound


def _validate_audit_replication_binding(
    audit: Mapping[str, Any],
    replication_inputs: Mapping[str, Any],
) -> None:
    """Require the audit to name the same B0/B1 input-bank identities."""
    audit_inputs = audit.get("replication_inputs")
    if not isinstance(audit_inputs, Mapping):
        raise SelectionError("complete exclusion audit lacks replication_inputs binding")
    audit_normalized = _validate_replication_inputs(
        {"replication_inputs": audit_inputs},
    )
    if _canonical_bytes(audit_normalized) != _canonical_bytes(dict(replication_inputs)):
        raise SelectionError("exclusion audit replication_inputs differ from frozen plan")


def validate_p11_manifest(
    manifest: Mapping[str, Any],
    *,
    require_state_bindings: bool = False,
    require_replication_inputs: bool = False,
) -> dict[str, Any]:
    """Validate the frozen P11 geometry and boundary without reading assets."""
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("task_id") != TASK_ID:
        raise P11PipelineError("P11 manifest schema or task identity changed")
    status = str(manifest.get("status", ""))
    if not (
        status.startswith("PLANNING_ONLY")
        or status.startswith("PREPARATION_IN_PROGRESS")
        or status.startswith("READY")
    ):
        raise P11PipelineError(f"P11 manifest status is not a runnable preparation state: {status}")
    decision = manifest.get("decision")
    if not isinstance(decision, Mapping) or decision.get("current_arm") != "current_bank_B0_fixed_public_readout" or decision.get("expanded_arm") != "expanded_bank_B1_fixed_public_readout":
        raise P11PipelineError("P11 B0/B1 decision binding changed")
    evaluation = manifest.get("evaluation_contract")
    if not isinstance(evaluation, Mapping):
        raise P11PipelineError("P11 evaluation contract is absent")
    if tuple(evaluation.get("domains", ())) != DOMAIN_ORDER or tuple(evaluation.get("targets", ())) != TARGET_ORDER or tuple(evaluation.get("cells", ())) != CELL_ORDER:
        raise P11PipelineError("P11 domain/target/cell order changed")
    expected_geometry = {
        "records_per_domain": RECORDS_PER_DOMAIN,
        "stored_tokens_including_bos": STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        "token_positions_per_cell": RECORDS_PER_DOMAIN * SCORED_POST_BOS_TOKENS,
        "exact_records_per_cell": RECORDS_PER_DOMAIN,
        "a1_a2_subset_records_per_domain": A1_A2_RECORDS_PER_DOMAIN,
        "a1_a2_subset_token_positions_per_cell": A1_A2_RECORDS_PER_DOMAIN * SCORED_POST_BOS_TOKENS,
    }
    if any(evaluation.get(key) != value for key, value in expected_geometry.items()):
        raise P11PipelineError("P11 evaluation geometry changed")
    reservation = manifest.get("source_reservation")
    if not isinstance(reservation, Mapping) or reservation.get("seed") != SELECTION_SEED or tuple(reservation.get("pile_half_open", ())) != PILE_RANGE or tuple(reservation.get("finance_half_open", ())) != FINANCE_RANGE:
        raise P11PipelineError("P11 source reservation changed")
    primary = evaluation.get("primary_contrast")
    if not isinstance(primary, Mapping) or primary.get("candidate") != "new_expanded_fixed_B1" or primary.get("control") != "new_current_fixed_B0" or primary.get("paired_unit") != "source_record" or tuple(primary.get("exact_record_inventory", ())) != ("gains", "losses", "ties") or primary.get("reported_separately_per_cell") is not True:
        raise P11PipelineError("P11 primary contrast binding changed")
    stats = manifest.get("statistical_contract")
    if not isinstance(stats, Mapping) or stats.get("bootstrap_unit") != "source_record" or stats.get("bootstrap_seed") != BOOTSTRAP_SEED or stats.get("bootstrap_resamples") != BOOTSTRAP_DRAWS or stats.get("automatic_ci_gate") is not False or stats.get("pooling") is not False or stats.get("owner") != "Agent2" or stats.get("separate_from_agent1_decoder_package") is not True:
        raise P11PipelineError("P11 statistical contract changed")
    for key in ("exact_record_interval", "token_contrast_function"):
        binding = stats.get(key)
        if not isinstance(binding, Mapping) or binding.get("path") != SCORER_RELATIVE_PATH or binding.get("source_commit") != SCORER_SOURCE_COMMIT or binding.get("source_sha256") != SCORER_SHA256:
            raise P11PipelineError(f"P11 scorer binding changed: {key}")
    if require_replication_inputs:
        _validate_replication_inputs(manifest)
    if require_state_bindings:
        _validate_new_state_bindings(manifest)
    truth = manifest.get("truth_boundary")
    if isinstance(truth, Mapping) and any(_is_true(truth.get(key)) for key in ("source_selection_started", "observations_captured", "predictions_started", "truth_opened", "p03_holdout_accessed")):
        raise P11PipelineError("P11 manifest records an already-opened phase")
    return dict(manifest)


def _normalize_source_inputs(source_inputs: Mapping[str, Any] | Path | str, *, root: Path, require_tokenizer_dir: bool = True) -> dict[str, Any]:
    if isinstance(source_inputs, (str, Path)):
        payload, _record_value = _load_json(Path(source_inputs), root=root, description="P11 source-input descriptor")
    elif isinstance(source_inputs, Mapping):
        payload = dict(source_inputs)
    else:
        raise P11PipelineError("source-input descriptor is malformed")
    normalized: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        descriptor = payload.get(domain)
        if not isinstance(descriptor, Mapping):
            raise P11PipelineError(f"source-input descriptor is absent: {domain}")
        expected_meta = _DATASET_META[domain]
        for key, expected in expected_meta.items():
            if descriptor.get(key) != expected:
                raise P11PipelineError(f"{domain} source {key} differs from the frozen dataset revision")
        files = descriptor.get("arrow_files")
        if not isinstance(files, Sequence) or isinstance(files, (str, bytes, bytearray)) or not files:
            raise P11PipelineError(f"{domain} Arrow files are absent")
        normalized_files = [_record_declared(item, root=root, description=f"{domain} Arrow file {index}") for index, item in enumerate(files)]
        normalized[domain] = {**expected_meta, "dataset_key": domain, "arrow_files": normalized_files}
    tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise P11PipelineError("tokenizer descriptor is absent")
    tokenizer_path = Path(str(tokenizer.get("path", ""))).expanduser()
    if not tokenizer_path.is_absolute():
        tokenizer_path = root / tokenizer_path
    tokenizer_path = tokenizer_path.resolve()
    if tokenizer_path.is_symlink() or not tokenizer_path.is_dir():
        raise P11PipelineError(f"tokenizer snapshot is unavailable: {tokenizer_path}")
    token_files: dict[str, Any] = {}
    declared_files = tokenizer.get("files", {})
    if declared_files is not None and not isinstance(declared_files, Mapping):
        raise P11PipelineError("tokenizer file descriptor is malformed")
    for name, item in (declared_files or {}).items():
        token_files[str(name)] = _record_declared(item, root=root, description=f"tokenizer file {name}")
    if require_tokenizer_dir and not token_files:
        # The actual tokenizer loader performs the semantic BOS/padding check;
        # this phase still binds at least one immutable snapshot file.
        raise P11PipelineError("tokenizer descriptor has no immutable file bindings")
    normalized["tokenizer"] = {"path": str(tokenizer_path), "files": token_files}
    return normalized


def _parse_union_namespace(value: str) -> Any:
    from scripts.trr_p10.build_exclusion_audit import Namespace
    parts = str(value).split("|")
    if len(parts) != 4:
        raise SelectionError(f"identity-union namespace is malformed: {value!r}")
    return Namespace(*parts)


def _load_identity_union(audit: Mapping[str, Any], *, root: Path) -> tuple[Any, dict[str, Any]]:
    binding = audit.get("identity_union_export")
    if not isinstance(binding, Mapping):
        raise SelectionError("complete audit must bind identity_union_export")
    path = _resolve(binding.get("path"), root=root, description="identity-union export")
    try:
        record = _record_declared(binding, root=root, description="identity-union export")
    except P11PipelineError as exc:
        raise SelectionError("identity-union export binding changed") from exc
    if record["bytes"] != int(binding.get("bytes", -1)) or record["sha256"] != binding.get("sha256"):
        raise SelectionError("identity-union export binding changed")
    payload, _ = _load_json(path, root=root, description="identity-union export")
    if payload.get("schema") != "token-reconstruction.trr-p11-identity-union.v1" or payload.get("task_id") != TASK_ID or payload.get("status") != "IDENTITY_UNION_COMPLETE_NO_PAYLOAD":
        raise SelectionError("identity-union export schema/status changed")
    if payload.get("source_text_or_token_ids_written") is not False or payload.get("truth_opened") is not False or payload.get("p03_holdout_accessed") is not False:
        raise SelectionError("identity-union export records forbidden payload access")
    fields = payload.get("fields")
    if not isinstance(fields, Mapping):
        raise SelectionError("identity-union export fields are absent")
    from scripts.trr_p10.build_exclusion_audit import IdentityBundle
    union = IdentityBundle("p11_identity_union_export", "union", Path("."), record["sha256"], record["bytes"], schema=payload["schema"], status=payload["status"])
    for field, by_namespace in fields.items():
        if not isinstance(field, str) or field not in _P11_UNION_FIELDS or not isinstance(by_namespace, Mapping):
            raise SelectionError(f"identity-union field is not an approved P11 namespace: {field!r}")
        for namespace_text, values in by_namespace.items():
            namespace = _parse_union_namespace(str(namespace_text))
            if not isinstance(values, list):
                raise SelectionError(f"identity-union values are malformed: {field}/{namespace_text}")
            for value in values:
                if field == "source_index":
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        raise SelectionError("identity-union source index is malformed")
                    union.add(field, value, namespace)
                elif field == "record_id":
                    if not isinstance(value, str) or not value:
                        raise SelectionError("identity-union record ID is malformed")
                    union.add(field, value, namespace)
                else:
                    text = str(value).lower()
                    if len(text) != 64 or any(char not in _SHA256_HEX for char in text):
                        raise SelectionError(f"identity-union hash is malformed: {field}")
                    union.add(field, text, namespace)
    counts = payload.get("identity_counts")
    if not isinstance(counts, Mapping) or dict(counts) != union.counts():
        raise SelectionError("identity-union internal counts changed")
    if isinstance(audit.get("union_identity_counts"), Mapping) and dict(audit["union_identity_counts"]) != union.counts():
        raise SelectionError("identity-union counts differ from exclusion audit")
    return union, record


def load_complete_exclusions(audit_path: Path, *, root: Path, pr20_root: Path | None = None) -> ExclusionContext:
    audit, audit_record = _load_json(audit_path, root=root, description="P11 exclusion audit")
    if audit.get("schema") != EXCLUSION_SCHEMA or audit.get("task_id") != TASK_ID:
        raise SelectionError("P11 exclusion audit schema or task identity changed")
    if audit.get("coverage_complete") is not True or audit.get("selection_release") is not True:
        raise SelectionError("source selection requires complete exclusion coverage and explicit release")
    boundary = audit.get("access_boundary")
    if not isinstance(boundary, Mapping) or boundary.get("p03_holdout_accessed") is not False:
        raise SelectionError("P03 holdout boundary is not closed")
    for key in ("source_or_token_payload_emitted", "truth_or_scores_read", "model_loaded", "new_panel_selected"):
        if boundary.get(key) is True:
            raise SelectionError(f"exclusion audit records forbidden access: {key}")
    inventory = audit.get("source_inventory")
    if not isinstance(inventory, list) or any("p03" in str(item.get("label", "")).lower() for item in inventory if isinstance(item, Mapping)):
        raise SelectionError("P03 is present in the exclusion inventory")
    union, union_record = _load_identity_union(audit, root=root)
    return ExclusionContext(audit_record=audit_record, audit=audit, union_record=union_record, union=union)


def _candidate_identity(candidate: Any) -> dict[str, Any]:
    """Convert a transient trusted-renderer row to hash-only metadata."""
    token_ids = tuple(int(value) for value in candidate.token_ids)
    if len(token_ids) < STORED_SEQUENCE_TOKENS:
        raise SelectionError("candidate lacks the H128 prefix required by the exclusion contract")
    # Reuse the producer's exact signed-int32 prefix convention.  Values are
    # transient and are never placed in the returned dictionary.
    from scripts.trr_p10 import build_exclusion_audit as p10
    fingerprints = p10.candidate_sequence_fingerprints(token_ids)
    if "h128_sequence_sha256" not in fingerprints:
        raise SelectionError("candidate H128 prefix fingerprint is unavailable")
    metadata = dict(candidate.selection_metadata())
    metadata["trr0002_active_token_ids_sha256"] = fingerprints["trr0002_active_token_ids_sha256"]
    metadata["trr0002_h40_token_ids_sha256"] = fingerprints.get("trr0002_h40_token_ids_sha256")
    # P11's raw signed-int32 first-40 namespace is distinct from the
    # TRR-0002 tensor-header/int64 H40 digest above.  The frozen P10 helper
    # computes the exact producer convention without modifying P10.
    metadata["h40_sequence_sha256"] = p10._raw_int32_digest(token_ids[:40])
    metadata["h128_sequence_sha256"] = fingerprints["h128_sequence_sha256"]
    metadata["h129_sequence_sha256"] = fingerprints.get("h129_sequence_sha256")
    metadata["final_sequence_sha256"] = fingerprints["h128_sequence_sha256"]
    return metadata


def _candidate_exclusion_reasons(candidate_metadata: Mapping[str, Any], union: Any) -> list[dict[str, str]]:
    from scripts.trr_p10 import build_exclusion_audit as p10
    candidate = {
        "record_id": candidate_metadata["record_id"],
        "rendered_sha256": candidate_metadata["public_record_sha256"],
        "h128_sequence_sha256": candidate_metadata["h128_sequence_sha256"],
        "source_index": int(candidate_metadata["source_index"]),
        "style": candidate_metadata["dataset_key"],
        "dataset_id": candidate_metadata["dataset_id"],
        "split": candidate_metadata["split"],
        "revision": candidate_metadata["revision"],
    }
    for field in (
        "h129_sequence_sha256",
        "trr0002_active_token_ids_sha256",
        "trr0002_h40_token_ids_sha256",
        "h40_sequence_sha256",
    ):
        value = candidate_metadata.get(field)
        if value is not None:
            candidate[field] = value
    reasons = p10.check_candidate(candidate, union)
    # Frozen P10 predates the P11 raw-int32 H40 namespace and must remain
    # byte-identical.  Apply this one additional global identity route here.
    raw_h40 = candidate_metadata.get("h40_sequence_sha256")
    if raw_h40 is not None:
        raw_h40 = str(raw_h40).casefold()
        for namespace, values in union.values.get("h40_sequence_sha256", {}).items():
            if raw_h40 in values:
                reasons.append({"field": "h40_sequence_sha256", "namespace": namespace.as_string()})
    return sorted(
        {(item["field"], item["namespace"]): item for item in reasons}.values(),
        key=lambda item: (item["field"], item["namespace"]),
    )


def _selection_row(candidate: Any) -> dict[str, Any]:
    metadata = _candidate_identity(candidate)
    allowed = (
        "record_id", "public_record_sha256", "dataset_key", "dataset_id", "split", "revision",
        "row_index", "source_index", "full_token_count", "post_bos_token_count", "valid_tokens",
        "final_sequence_sha256", "h40_sequence_sha256", "h128_sequence_sha256", "h129_sequence_sha256",
        "trr0002_active_token_ids_sha256", "trr0002_h40_token_ids_sha256",
    )
    return {key: metadata[key] for key in allowed}


def _validate_p11_index(domain: str, index: int) -> None:
    if domain not in SOURCE_RANGES or not isinstance(index, int) or isinstance(index, bool):
        raise SelectionError("source row index is malformed")
    start, stop = SOURCE_RANGES[domain]
    if not start <= index < stop:
        raise SelectionError(f"{domain} row {index} is outside the P11 reserved range [{start}, {stop})")


def select_sources(
    *,
    manifest_path: Path,
    audit_path: Path,
    source_inputs: Mapping[str, Any] | Path | str,
    output_path: Path,
    repository_root: Path,
    pr20_root: Path | None = None,
) -> dict[str, Any]:
    """Select the first 256 eligible rows per domain after all gates pass."""
    root = Path(repository_root).expanduser().resolve()
    manifest, manifest_record = _load_manifest(
        Path(manifest_path),
        root=root,
        require_replication_inputs=True,
    )
    replication_inputs = _validate_replication_inputs(manifest)
    exclusions = load_complete_exclusions(Path(audit_path), root=root, pr20_root=pr20_root)
    _validate_audit_replication_binding(exclusions.audit, replication_inputs)
    inputs = _normalize_source_inputs(source_inputs, root=root)
    output = _task_output(Path(output_path), root=root, phase="selection")

    try:
        from scripts import trr0005_produce_confirmation as trusted
        from token_reconstruction.trr0005_public_corpus import deterministic_row_order
        tokenizer = trusted._load_tokenizer(Path(inputs["tokenizer"]["path"]))
        datasets = {
            domain: trusted._load_arrow_dataset(tuple(Path(item["path"]) for item in inputs[domain]["arrow_files"]))
            for domain in DOMAIN_ORDER
        }
    except Exception as exc:
        raise SelectionError("trusted public tokenizer or Arrow source could not be loaded") from exc

    selected: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAIN_ORDER}
    diagnostics: dict[str, dict[str, int]] = {}
    seen_rendered: set[str] = set()
    seen_h128: set[str] = set()
    seen_ids: set[str] = set()
    for domain in DOMAIN_ORDER:
        start, stop = SOURCE_RANGES[domain]
        if len(datasets[domain]) < stop:
            raise SelectionError(f"{domain} source has {len(datasets[domain])} rows; need {stop}")
        counts = {"excluded_identity": 0, "duplicate_rendered": 0, "duplicate_h128": 0, "invalid_short_or_render": 0}
        order = deterministic_row_order(range(start, stop), dataset_key=f"trr-p11-{domain}", seed=SELECTION_SEED)
        for index in order:
            _validate_p11_index(domain, index)
            try:
                row = datasets[domain][index]
                candidate = trusted._render_row(domain, row, index, tokenizer)
                metadata = _candidate_identity(candidate)
            except trusted.ProducerError as exc:
                message = str(exc).lower()
                if "shorter than" in message or "no user/assistant" in message or "malformed" in message:
                    counts["invalid_short_or_render"] += 1
                    continue
                raise SelectionError(f"trusted renderer failed at {domain}/{index}") from exc
            reasons = _candidate_exclusion_reasons(metadata, exclusions.union)
            if reasons:
                counts["excluded_identity"] += 1
                continue
            rendered = str(metadata["public_record_sha256"])
            h128 = str(metadata["h128_sequence_sha256"])
            record_id = str(metadata["record_id"])
            if rendered in seen_rendered:
                counts["duplicate_rendered"] += 1
                continue
            if h128 in seen_h128:
                counts["duplicate_h128"] += 1
                continue
            if record_id in seen_ids:
                counts["duplicate_rendered"] += 1
                continue
            seen_rendered.add(rendered)
            seen_h128.add(h128)
            seen_ids.add(record_id)
            selected[domain].append({key: metadata[key] for key in (
                "record_id", "public_record_sha256", "dataset_key", "dataset_id", "split", "revision",
                "row_index", "source_index", "full_token_count", "post_bos_token_count", "valid_tokens",
                "final_sequence_sha256", "h40_sequence_sha256", "h128_sequence_sha256", "h129_sequence_sha256",
                "trr0002_active_token_ids_sha256", "trr0002_h40_token_ids_sha256",
            )})
            if len(selected[domain]) == RECORDS_PER_DOMAIN:
                break
        diagnostics[domain] = {**counts, "selected": len(selected[domain]), "pool_size": stop - start}
        if len(selected[domain]) != RECORDS_PER_DOMAIN:
            raise SelectionError(f"{domain} eligible pool yielded {len(selected[domain])}; need {RECORDS_PER_DOMAIN}")

    ids = {domain: [row["record_id"] for row in selected[domain]] for domain in DOMAIN_ORDER}
    h128 = {domain: [row["h128_sequence_sha256"] for row in selected[domain]] for domain in DOMAIN_ORDER}
    payload: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "task_id": TASK_ID,
        "status": SELECTION_STATUS,
        "created_utc": _utc_now(),
        "manifest": manifest_record,
        "manifest_sha256": manifest_record["sha256"],
        "exclusion_audit": exclusions.audit_record,
        "exclusion_audit_sha256": exclusions.audit_record["sha256"],
        "selection_seed": SELECTION_SEED,
        "source_ranges_half_open": {domain: list(SOURCE_RANGES[domain]) for domain in DOMAIN_ORDER},
        "records_by_domain": {domain: RECORDS_PER_DOMAIN for domain in DOMAIN_ORDER},
        "sequence_tokens_including_bos": STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": SCORED_POST_BOS_TOKENS,
        "target_conditions": list(TARGET_ORDER),
        "paired_conditions": True,
        "public_sources_frozen": inputs,
        "selection_rule": {
            "algorithm": "deterministic_row_order(dataset_key=trr-p11-{domain}, seed=5011); trusted public rerender/tokenization; reject complete audited identity union and rendered/H128 duplicates; retain first 256 eligible rows per domain; H129 is recorded where the rendered prefix provides it",
            "identity_exclusions": True,
            "public_rerender_and_tokenization": True,
            "source_text_or_token_ids_written": False,
            "record_ids_sha256": {domain: _json_digest(ids[domain]) for domain in DOMAIN_ORDER},
            "h128_sequence_sha256": {domain: _json_digest(h128[domain]) for domain in DOMAIN_ORDER},
            "records": selected,
        },
        "a1_a2_subset": {
            "status": "PROSPECTIVE_FIRST_128_PER_DOMAIN",
            "records_per_domain": A1_A2_RECORDS_PER_DOMAIN,
            "record_ids_sha256": {domain: _json_digest(ids[domain][:A1_A2_RECORDS_PER_DOMAIN]) for domain in DOMAIN_ORDER},
            "performance_based_drop": False,
        },
        "selection_diagnostics": diagnostics,
        "exclusion_union_counts": dict(exclusions.union.counts()),
        "execution": {
            "command": list(sys.argv),
            "code_commit": _git_head(root),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "network_used": False,
            "source_text_read": True,
            "public_rerender_and_tokenization": True,
            "model_loaded": False,
            "source_text_written": False,
            "token_ids_written": False,
            "truth_opened": False,
            "selection_performed": True,
        },
        "truth_opened": False,
        "truth_created": False,
        "target_labels_loaded": False,
        "source_text_written": False,
        "token_ids_written": False,
        "candidate_arrays_persisted": False,
        "p03_holdout_accessed": False,
        "selection_release": True,
    }
    record = _write_create_only(output, payload, root=root, description="P11 source selection")
    return {"task_id": TASK_ID, "status": SELECTION_STATUS, "selection": record, "records_by_domain": payload["records_by_domain"], "truth_opened": False}


def _validate_selection_payload(payload: Mapping[str, Any], *, root: Path) -> SelectionContext:
    if payload.get("schema") != SELECTION_SCHEMA or payload.get("task_id") != TASK_ID or payload.get("status") != SELECTION_STATUS:
        raise EvaluationError("P11 source selection is not frozen")
    for key in ("truth_opened", "truth_created", "target_labels_loaded", "source_text_written", "token_ids_written", "candidate_arrays_persisted", "p03_holdout_accessed"):
        if _is_true(payload.get(key)):
            raise EvaluationError(f"P11 selection records forbidden access: {key}")
    if payload.get("selection_release") is not True or payload.get("paired_conditions") is not True:
        raise EvaluationError("P11 selection release or target pairing is absent")
    if payload.get("selection_seed") != SELECTION_SEED or tuple(payload.get("target_conditions", ())) != TARGET_ORDER:
        raise EvaluationError("P11 selection constants changed")
    rows_raw = payload.get("selection_rule", {}).get("records") if isinstance(payload.get("selection_rule"), Mapping) else None
    if not isinstance(rows_raw, Mapping):
        raise EvaluationError("P11 selection rows are absent")
    counts_raw = payload.get("records_by_domain")
    if not isinstance(counts_raw, Mapping):
        raise EvaluationError("P11 selection counts are absent")
    rows: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    allowed = {
        "record_id", "public_record_sha256", "dataset_key", "dataset_id", "split", "revision", "row_index", "source_index",
        "full_token_count", "post_bos_token_count", "valid_tokens", "final_sequence_sha256", "h40_sequence_sha256", "h128_sequence_sha256", "h129_sequence_sha256",
        "trr0002_active_token_ids_sha256", "trr0002_h40_token_ids_sha256",
    }
    for domain in DOMAIN_ORDER:
        values = rows_raw.get(domain)
        if not isinstance(values, list) or len(values) != RECORDS_PER_DOMAIN or int(counts_raw.get(domain, -1)) != RECORDS_PER_DOMAIN:
            raise EvaluationError(f"P11 selection count changed: {domain}")
        checked: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, value in enumerate(values):
            if not isinstance(value, Mapping) or set(value) != allowed:
                raise EvaluationError(f"P11 selection row contains unapproved payload: {domain}/{index}")
            row = dict(value)
            if not isinstance(row.get("record_id"), str) or not row["record_id"] or row["record_id"] in seen_ids:
                raise EvaluationError(f"P11 selection record ID is malformed: {domain}/{index}")
            for key in ("public_record_sha256", "final_sequence_sha256", "h40_sequence_sha256", "h128_sequence_sha256", "trr0002_active_token_ids_sha256", "trr0002_h40_token_ids_sha256"):
                _require_sha(row.get(key), description=f"P11 selection {domain}/{index}/{key}")
            if row.get("h129_sequence_sha256") is not None:
                _require_sha(row.get("h129_sequence_sha256"), description=f"P11 selection {domain}/{index}/h129_sequence_sha256")
            if row["final_sequence_sha256"] != row["h128_sequence_sha256"] or row.get("dataset_key") != domain or row.get("valid_tokens") != STORED_SEQUENCE_TOKENS:
                raise EvaluationError(f"P11 selection identity binding changed: {domain}/{index}")
            if int(row.get("row_index", -1)) != int(row.get("source_index", -2)):
                raise EvaluationError(f"P11 selection source index binding changed: {domain}/{index}")
            _validate_p11_index(domain, int(row["row_index"]))
            seen_ids.add(row["record_id"])
            checked.append(row)
        rows[domain] = checked
        counts[domain] = RECORDS_PER_DOMAIN
    rule = payload.get("selection_rule")
    if not isinstance(rule, Mapping) or not isinstance(rule.get("record_ids_sha256"), Mapping) or not isinstance(rule.get("h128_sequence_sha256"), Mapping):
        raise EvaluationError("P11 selection digest bindings are absent")
    for domain in DOMAIN_ORDER:
        if rule["record_ids_sha256"].get(domain) != _json_digest([row["record_id"] for row in rows[domain]]) or rule["h128_sequence_sha256"].get(domain) != _json_digest([row["h128_sequence_sha256"] for row in rows[domain]]):
            raise EvaluationError(f"P11 selection digest changed: {domain}")
    sources = payload.get("public_sources_frozen")
    if not isinstance(sources, Mapping):
        raise EvaluationError("P11 selection sources are absent")
    return SelectionContext(payload=dict(payload), record={}, rows=rows, counts=counts)


def load_selection(selection_path: Path, *, root: Path) -> SelectionContext:
    payload, record = _load_json(selection_path, root=root, description="P11 source selection")
    context = _validate_selection_payload(payload, root=root)
    return SelectionContext(payload=context.payload, record=record, rows=context.rows, counts=context.counts)



def choose_identity_rows(candidates: Mapping[str, Sequence[Mapping[str, Any]]], *, union: Any, records_per_domain: int = RECORDS_PER_DOMAIN) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    """Select already-rendered identity metadata without reading source payloads."""
    allowed = {"record_id", "public_record_sha256", "dataset_key", "dataset_id", "split", "revision", "row_index", "source_index", "full_token_count", "post_bos_token_count", "valid_tokens", "final_sequence_sha256", "h40_sequence_sha256", "h128_sequence_sha256", "h129_sequence_sha256", "trr0002_active_token_ids_sha256", "trr0002_h40_token_ids_sha256"}
    chosen: dict[str, list[dict[str, Any]]] = {domain: [] for domain in DOMAIN_ORDER}
    diagnostics: dict[str, dict[str, int]] = {}
    seen_ids: set[str] = set(); seen_rendered: set[str] = set(); seen_h128: set[str] = set()
    for domain in DOMAIN_ORDER:
        values = candidates.get(domain)
        if not isinstance(values, Sequence): raise SelectionError(f"candidate metadata is absent: {domain}")
        stats = {"excluded_identity": 0, "duplicate_rendered": 0, "duplicate_h128": 0}
        for raw in values:
            if not isinstance(raw, Mapping) or set(raw) != allowed: raise SelectionError(f"candidate identity metadata is malformed: {domain}")
            row = dict(raw)
            if row.get("dataset_key") != domain or row.get("valid_tokens") != STORED_SEQUENCE_TOKENS: raise SelectionError(f"candidate contract changed: {domain}")
            for key in ("public_record_sha256", "final_sequence_sha256", "h40_sequence_sha256", "h128_sequence_sha256", "trr0002_active_token_ids_sha256", "trr0002_h40_token_ids_sha256"):
                value = row.get(key)
                if not isinstance(value, str) or len(value) != 64 or any(char not in _SHA256_HEX for char in value.lower()): raise SelectionError(f"candidate hash malformed: {domain}/{key}")
            if row.get("h129_sequence_sha256") is not None:
                value = row["h129_sequence_sha256"]
                if not isinstance(value, str) or len(value) != 64 or any(char not in _SHA256_HEX for char in value.lower()): raise SelectionError(f"candidate hash malformed: {domain}/h129_sequence_sha256")
            if row["final_sequence_sha256"] != row["h128_sequence_sha256"] or int(row["row_index"]) != int(row["source_index"]): raise SelectionError(f"candidate identity changed: {domain}")
            if _candidate_exclusion_reasons(row, union): stats["excluded_identity"] += 1; continue
            rendered, h128, record_id = row["public_record_sha256"].lower(), row["h128_sequence_sha256"].lower(), row["record_id"]
            if record_id in seen_ids or rendered in seen_rendered: stats["duplicate_rendered"] += 1; continue
            if h128 in seen_h128: stats["duplicate_h128"] += 1; continue
            seen_ids.add(record_id); seen_rendered.add(rendered); seen_h128.add(h128); chosen[domain].append(row)
            if len(chosen[domain]) == records_per_domain: break
        stats.update({"selected": len(chosen[domain]), "pool_size": len(values)}); diagnostics[domain] = stats
        if len(chosen[domain]) != records_per_domain: raise SelectionError(f"{domain} eligible pool yielded {len(chosen[domain])}; need {records_per_domain}")
    return chosen, diagnostics


__all__ = ["ExclusionContext", "SelectionContext", "SelectionError", "choose_identity_rows", "load_complete_exclusions", "scorer_contract", "select_sources", "validate_p11_manifest"]
