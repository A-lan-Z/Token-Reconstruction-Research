"""Truth-free native A1+A2 K256 runtime for the prospective P11 panel.

The adapter reuses the exact local TRR3 proposal/decode helpers and TRR4
public-prefix loader already present in the repository.  It does not reuse
historical prediction tensors.  A runtime binding records the current P11
selection's first-128 identities, current local code hashes, and every public
resource hash before the native loader is allowed to run.  The heavy imports
and CUDA path are reachable only from ``run_native_a1_a2(..., execute=True)``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from scripts.trr_p11 import public_capture
from scripts.trr_p11 import source_selector as selector

TASK_ID = selector.TASK_ID
METHOD_ID = "frozen_a1_a2_k256"
RUNTIME_SCHEMA = "token-reconstruction.trr-p11-a1-a2-runtime-binding.v1"
RUNTIME_STATUS = "BOUND_NATIVE_A1_A2_K256_BEFORE_TRUTH"
PREDICTION_SCHEMA = "token-reconstruction.trr-p11-a1-a2-prediction.v1"
PREDICTION_STATUS = "A1_A2_K256_PREDICTIONS_COMPLETE_NO_TRUTH"
QUALIFICATION_STATUS = "A1_A2_K256_QUALIFICATION_COMPLETE_NO_TRUTH"
LARGEST_QUALIFICATION_CELL = "finance__public_base"
DOMAIN_ORDER = selector.DOMAIN_ORDER
CELL_ORDER = selector.CELL_ORDER
A1_A2_RECORDS_PER_DOMAIN = selector.A1_A2_RECORDS_PER_DOMAIN
STORED_SEQUENCE_TOKENS = selector.STORED_SEQUENCE_TOKENS
SCORED_POST_BOS_TOKENS = selector.SCORED_POST_BOS_TOKENS
HIDDEN_SIZE = selector.HIDDEN_SIZE
DEFAULT_A1_CHUNK = 256
DEFAULT_A2_K = 256
DEFAULT_A2_PROPOSAL_K = 512
DEFAULT_RECORD_BATCH_SIZE = 1
DEFAULT_MINIMUM_FREE_GIB = 8.0
DEFAULT_MAXIMUM_RESERVED_GIB = 6.0
DEFAULT_MAXIMUM_RSS_GIB = 16.0
DEFAULT_MAX_SECONDS: float | None = None
DEFAULT_WATCHDOG_POLL_SECONDS = 1.0

_SHA256_HEX = frozenset("0123456789abcdef")

# Exact code bindings from the historical descriptor, verified against the
# current P11 checkout before runtime use.  The P11 adapter never imports the
# old worktree's code.
CODE_BINDINGS = {
    "legacy_confirmation_adapter": ("scripts/trr0004_predict_confirmation.py", "36f6aa7b4493c60b257b3896c975f523595da912e14a97dba3f6440d419e8427"),
    "fresh_confirmation_contract": ("scripts/trr0004_fresh_confirmation.py", "482c3a75d2fc641f90fe9c531584e2be14aefe914d97b246fe3633bdaa3d2945"),
    "native_driver": ("scripts/trr0003_footing_compare.py", "bb3690a481014f3ea12b411a3ec6eb58d7c888ef6c4135f9e03908027857f98e"),
    "a1_a2_configuration_search": ("src/token_reconstruction/a1a2_configuration_search.py", "608bead6291353cacaa700a85f3dd619fb3261375e6fd5ba19cbc43fd738315b"),
    "component_crossover": ("src/token_reconstruction/component_crossover.py", "2cd03fda1f29f30d0a92246252ba2d2d89d3890abc6739ae0c8bae0a5c105b7d"),
    "public_prefix": ("src/token_reconstruction/public_prefix.py", "b9dca1d8d56c7c07413015aea4bde79e8ceac827c32ee9bc0d1a236ea1dd35f6"),
}
EXPECTED_EMBEDDING_SHA256 = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"
EXPECTED_LENS_SHA256 = "33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742"
EXPECTED_REFERENCE_SHA256 = "10532a746cb8c30eb2caf338e206e1fa9d85e708d4db43a0d8fd4a2ff1a6f8bd"
EXPECTED_EMBEDDING_BYTES = 1050673488
EXPECTED_LENS_BYTES = 16787653


class A1A2RuntimeError(RuntimeError):
    """Raised when the native comparator cannot be bound or run safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise A1A2RuntimeError(f"asset is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise A1A2RuntimeError("value is not canonical JSON") from exc
    return hashlib.sha256(raw).hexdigest()


def _tensor_digest(value: Any) -> str:
    """Match the repository tensor digest used by the P11 evaluation gate."""
    import torch

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


def _require_sha(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _SHA256_HEX for char in value):
        raise A1A2RuntimeError(f"{description} is not a lowercase SHA-256")
    return value


def _record(path: Path, *, expected_sha256: str | None = None, expected_bytes: int | None = None, description: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise A1A2RuntimeError(f"{description} is unavailable: {resolved}")
    actual = {"path": str(resolved), "bytes": int(resolved.stat().st_size), "sha256": _sha256_file(resolved)}
    if expected_bytes is not None and actual["bytes"] != int(expected_bytes):
        raise A1A2RuntimeError(f"{description} byte binding changed")
    if expected_sha256 is not None and actual["sha256"] != _require_sha(expected_sha256, description=description):
        raise A1A2RuntimeError(f"{description} hash binding changed")
    return actual


def _write_create_only(path: Path, payload: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = root / resolved
    resolved = resolved.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise A1A2RuntimeError(f"{description} is create-only: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return {"path": str(resolved), "bytes": int(resolved.stat().st_size), "sha256": _sha256_file(resolved)}


def _load_json(path: Path, *, description: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _record(path, description=description)
    try:
        value = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise A1A2RuntimeError(f"{description} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise A1A2RuntimeError(f"{description} must be an object")
    return dict(value), record


def validate_native_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable historical A1+A2 K256 decision rule."""
    expected = {
        "candidate_budget": DEFAULT_A2_K,
        "proposal_budget": DEFAULT_A2_PROPOSAL_K,
        "proposal_chunk": DEFAULT_A1_CHUNK,
        "schedule": [DEFAULT_A2_K],
        "fast_path_id": "off",
        "score_rule": "direct_cosine",
        "terminal_action": "commit_last_winner",
        "reconstructed_prefix": True,
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise A1A2RuntimeError(f"A1+A2 policy changed at {key}: {policy.get(key)!r}")
    return dict(expected)


def _validate_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    if descriptor.get("method_id") != METHOD_ID:
        raise A1A2RuntimeError("A1+A2 descriptor method identity changed")
    status = descriptor.get("status")
    if status != "PUBLIC_READONLY_BINDING_ONLY_NO_INFERENCE":
        raise A1A2RuntimeError("A1+A2 descriptor is not the retained read-only binding")
    policy = descriptor.get("policy")
    if not isinstance(policy, Mapping):
        raise A1A2RuntimeError("A1+A2 policy binding is absent")
    normalized = validate_native_policy(policy)
    boundary = descriptor.get("truth_boundary")
    if not isinstance(boundary, Mapping) or boundary.get("truth_opened") is not False or boundary.get("target_labels_loaded") is not False:
        raise A1A2RuntimeError("A1+A2 descriptor truth boundary is open or absent")
    return normalized


def _resolve_asset(path_value: str | Path, *, root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _descriptor_expected_resource(descriptor: Mapping[str, Any], key: str, fallback: str | None = None) -> tuple[str, int | None]:
    value = descriptor.get(key)
    if not isinstance(value, Mapping):
        raise A1A2RuntimeError(f"A1+A2 descriptor resource is absent: {key}")
    digest = value.get("sha256", fallback)
    expected = _require_sha(digest, description=f"A1+A2 {key}")
    bytes_value = value.get("bytes")
    return expected, int(bytes_value) if bytes_value is not None else None


def _code_records(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for key, (relative, expected) in CODE_BINDINGS.items():
        record = _record(root / relative, expected_sha256=expected, description=f"P11 native A1+A2 code {key}")
        records[key] = {**record, "relative_path": relative}
    return records


def first128_subset_binding(selection: selector.SelectionContext) -> dict[str, Any]:
    """Bind the prospective comparator to the first 128 selected rows only."""
    ids: dict[str, list[str]] = {}
    h128: dict[str, list[str]] = {}
    for domain in DOMAIN_ORDER:
        rows = selection.rows.get(domain)
        if not isinstance(rows, list) or len(rows) != selector.RECORDS_PER_DOMAIN:
            raise A1A2RuntimeError(f"P11 selection does not expose {selector.RECORDS_PER_DOMAIN} rows: {domain}")
        subset = rows[:A1_A2_RECORDS_PER_DOMAIN]
        ids[domain] = [str(row["record_id"]) for row in subset]
        h128[domain] = [str(row["h128_sequence_sha256"]) for row in subset]
        if len(set(ids[domain])) != A1_A2_RECORDS_PER_DOMAIN or len(set(h128[domain])) != A1_A2_RECORDS_PER_DOMAIN:
            raise A1A2RuntimeError(f"P11 first-128 comparator rows are not unique: {domain}")
    return {
        "method": METHOD_ID,
        "records_per_domain": A1_A2_RECORDS_PER_DOMAIN,
        "first_records_per_domain": True,
        "performance_based_drop": False,
        "domains": list(DOMAIN_ORDER),
        "token_positions_per_cell": A1_A2_RECORDS_PER_DOMAIN * SCORED_POST_BOS_TOKENS,
        "record_ids": ids,
        "record_ids_sha256": {domain: _canonical_digest(ids[domain]) for domain in DOMAIN_ORDER},
        "h128_sequence_sha256": h128,
        "h128_sequence_sha256_digest": {domain: _canonical_digest(h128[domain]) for domain in DOMAIN_ORDER},
    }


def bind_native_runtime(
    *,
    descriptor_path: Path,
    selection_path: Path,
    repository_root: Path,
    public_embedding_path: Path,
    lens_path: Path,
    model_snapshot_path: Path,
    reference_path: Path,
    output_path: Path | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Bind exact local code/resources and the future selection's first-128 subset."""
    root = Path(repository_root).expanduser().resolve()
    descriptor, descriptor_record = _load_json(Path(descriptor_path), description="A1+A2 runtime descriptor")
    policy = _validate_descriptor(descriptor)
    selection = selector.load_selection(Path(selection_path), root=root)
    subset = first128_subset_binding(selection)
    code = _code_records(root)

    embedding_declared, embedding_bytes = _descriptor_expected_resource(descriptor, "public_embedding_table", EXPECTED_EMBEDDING_SHA256)
    lens_declared, lens_bytes = _descriptor_expected_resource(descriptor, "retained_a1_lens", EXPECTED_LENS_SHA256)
    reference_value = descriptor.get("public_reference") or descriptor.get("reference_binding")
    if not isinstance(reference_value, Mapping):
        raise A1A2RuntimeError("A1+A2 public reference binding is absent")
    reference_declared = _require_sha(reference_value.get("sha256", EXPECTED_REFERENCE_SHA256), description="A1+A2 public reference")
    model = descriptor.get("model_snapshot")
    if not isinstance(model, Mapping) or not isinstance(model.get("files"), Mapping):
        raise A1A2RuntimeError("A1+A2 model snapshot file bindings are absent")
    snapshot_path = _resolve_asset(model_snapshot_path, root=root)
    if snapshot_path.is_symlink() or not snapshot_path.is_dir():
        raise A1A2RuntimeError(f"A1+A2 model snapshot is unavailable: {snapshot_path}")

    def bind_resource(path: Path, expected: str, declared_bytes: int | None, description: str) -> dict[str, Any]:
        if verify_files:
            return _record(path, expected_sha256=expected, expected_bytes=declared_bytes, description=description)
        resolved = path.expanduser().resolve()
        if resolved.is_symlink() or not resolved.is_file():
            raise A1A2RuntimeError(f"{description} is unavailable: {resolved}")
        return {"path": str(resolved), "bytes": declared_bytes, "sha256": expected, "verified": False}

    resources: dict[str, Any] = {
        "public_embedding_table": bind_resource(Path(public_embedding_path), embedding_declared, embedding_bytes or EXPECTED_EMBEDDING_BYTES, "A1+A2 public embedding"),
        "retained_a1_lens": bind_resource(Path(lens_path), lens_declared, lens_bytes or EXPECTED_LENS_BYTES, "A1+A2 retained lens"),
        "public_reference": bind_resource(Path(reference_path), reference_declared, int(reference_value.get("bytes")) if reference_value.get("bytes") is not None else None, "A1+A2 public reference"),
    }
    snapshot_files: dict[str, Any] = {}
    for name, value in model["files"].items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise A1A2RuntimeError("A1+A2 model snapshot file binding is malformed")
        expected = _require_sha(value.get("sha256"), description=f"A1+A2 snapshot {name}")
        expected_bytes = int(value["bytes"]) if value.get("bytes") is not None else None
        snapshot_files[name] = bind_resource(snapshot_path / name, expected, expected_bytes, f"A1+A2 snapshot {name}")
    resources["model_snapshot"] = {"path": str(snapshot_path), "files": snapshot_files}
    return_payload: dict[str, Any] = {
        "schema": RUNTIME_SCHEMA,
        "task_id": TASK_ID,
        "status": RUNTIME_STATUS,
        "method_id": METHOD_ID,
        "descriptor": descriptor_record,
        "selection": selection.record,
        "selection_sha256": selection.record["sha256"],
        "subset": subset,
        "policy": policy,
        "code_bindings": code,
        "resources": resources,
        "resource_verification": "PASS" if verify_files else "DEFERRED_RUNTIME_HASH_CHECK",
        "candidate_arrays_persisted": False,
        "source_text_written": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "truth_opened": False,
        "p03_holdout_accessed": False,
    }
    if output_path is not None:
        return_payload["binding_record"] = _write_create_only(Path(output_path), return_payload, root=root, description="P11 A1+A2 runtime binding")
    return return_payload


def validate_runtime_binding(binding: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    if binding.get("schema") != RUNTIME_SCHEMA or binding.get("task_id") != TASK_ID or binding.get("status") != RUNTIME_STATUS:
        raise A1A2RuntimeError("P11 A1+A2 runtime binding schema/status changed")
    if binding.get("method_id") != METHOD_ID:
        raise A1A2RuntimeError("P11 A1+A2 runtime method identity changed")
    validate_native_policy(binding.get("policy") if isinstance(binding.get("policy"), Mapping) else {})
    for key in ("source_text_written", "token_ids_written", "target_labels_loaded", "truth_opened", "p03_holdout_accessed"):
        if binding.get(key) is not False:
            raise A1A2RuntimeError(f"P11 A1+A2 runtime boundary changed: {key}")
    subset = binding.get("subset")
    if not isinstance(subset, Mapping) or subset.get("records_per_domain") != A1_A2_RECORDS_PER_DOMAIN or subset.get("first_records_per_domain") is not True or subset.get("performance_based_drop") is not False:
        raise A1A2RuntimeError("P11 A1+A2 first-128 subset binding changed")
    for domain in DOMAIN_ORDER:
        if len(subset.get("record_ids", {}).get(domain, ())) != A1_A2_RECORDS_PER_DOMAIN:
            raise A1A2RuntimeError(f"P11 A1+A2 subset record count changed: {domain}")
    resources = binding.get("resources")
    code = binding.get("code_bindings")
    if not isinstance(resources, Mapping) or not isinstance(code, Mapping):
        raise A1A2RuntimeError("P11 A1+A2 code/resource bindings are absent")
    for key, (relative, expected) in CODE_BINDINGS.items():
        declared = code.get(key)
        if not isinstance(declared, Mapping) or declared.get("sha256") != expected:
            raise A1A2RuntimeError(f"P11 A1+A2 code binding changed: {key}")
        actual = _record(Path(str(declared.get("path", root / relative))), expected_sha256=expected, description=f"P11 A1+A2 code {key}")
        if actual["sha256"] != declared.get("sha256"):
            raise A1A2RuntimeError(f"P11 A1+A2 code file changed: {key}")
    for key in ("public_embedding_table", "retained_a1_lens", "public_reference"):
        declared = resources.get(key)
        if not isinstance(declared, Mapping):
            raise A1A2RuntimeError(f"P11 A1+A2 resource binding absent: {key}")
        _record(Path(str(declared.get("path", ""))), expected_sha256=_require_sha(declared.get("sha256"), description=key), description=f"P11 A1+A2 resource {key}")
    snapshot = resources.get("model_snapshot")
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("files"), Mapping):
        raise A1A2RuntimeError("P11 A1+A2 model snapshot binding is absent")
    snapshot_path = Path(str(snapshot.get("path", "")))
    if snapshot_path.is_symlink() or not snapshot_path.is_dir():
        raise A1A2RuntimeError(f"P11 A1+A2 model snapshot is unavailable: {snapshot_path}")
    for name, declared in snapshot["files"].items():
        if not isinstance(declared, Mapping):
            raise A1A2RuntimeError(f"P11 A1+A2 model snapshot binding is malformed: {name}")
        _record(snapshot_path / str(name), expected_sha256=_require_sha(declared.get("sha256"), description=f"snapshot {name}"), expected_bytes=int(declared["bytes"]) if declared.get("bytes") is not None else None, description=f"P11 A1+A2 snapshot {name}")
    return dict(binding)


def _load_observation_manifest(path: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, record = _load_json(path, description="P11 public observation manifest")
    if payload.get("schema") != selector.OBSERVATION_SCHEMA or payload.get("task_id") != TASK_ID or payload.get("status") != "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH":
        raise A1A2RuntimeError("P11 observations are not frozen before truth")
    for key in ("source_text_written", "token_ids_written", "target_labels_loaded", "truth_opened", "p03_holdout_accessed"):
        if payload.get(key) is not False:
            raise A1A2RuntimeError(f"P11 observation boundary changed: {key}")
    if payload.get("cell_order") != list(CELL_ORDER):
        raise A1A2RuntimeError("P11 observation cell order changed")
    return payload, record


def _validate_observation_descriptor(descriptor: Mapping[str, Any], *, cell: str) -> None:
    """Require the capture-side order/tensor bindings consumed by evaluation."""
    order = descriptor.get("record_order")
    if not isinstance(order, list) or len(order) != selector.RECORDS_PER_DOMAIN or any(not isinstance(item, str) or not item for item in order):
        raise A1A2RuntimeError(f"P11 observation record order is absent or malformed: {cell}")
    tensor_sha = descriptor.get("tensor_sha256")
    if not isinstance(tensor_sha, Mapping) or set(tensor_sha) != {"activations", "attention_mask", "position_ids"}:
        raise A1A2RuntimeError(f"P11 observation tensor bindings are incomplete: {cell}")
    for key in ("activations", "attention_mask", "position_ids"):
        _require_sha(tensor_sha.get(key), description=f"P11 observation {cell}/{key}")
    _require_sha(descriptor.get("record_order_sha256"), description=f"P11 observation {cell} record order")


def _load_prediction_tensors(path: Path) -> tuple[Any, Any, Any]:
    from safetensors.torch import load_file
    tensors = load_file(str(path), device="cpu")
    required = {"activations", "attention_mask", "position_ids"}
    if set(tensors) != required:
        raise A1A2RuntimeError(f"observation keys changed: {path}")
    return tensors["activations"], tensors["attention_mask"], tensors["position_ids"]


def execution_cells(
    *,
    qualification_cell: str | None = None,
    qualification_only: bool = False,
    reuse_qualification: bool = False,
) -> tuple[str, ...]:
    """Return the explicit restart-safe cell plan without loading numerical data."""
    cell = qualification_cell or LARGEST_QUALIFICATION_CELL
    if cell not in CELL_ORDER:
        raise A1A2RuntimeError(f"unknown qualification cell: {cell}")
    if qualification_only and reuse_qualification:
        raise A1A2RuntimeError("qualification_only cannot reuse an earlier qualification")
    if qualification_only:
        return (cell,)
    if reuse_qualification:
        return tuple(item for item in CELL_ORDER if item != cell)
    return tuple(CELL_ORDER)


def _load_qualification_receipt(
    path: Path,
    *,
    binding: Mapping[str, Any],
    observation_record: Mapping[str, Any],
    expected_cell: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the retained largest-cell artifacts before a resumed matrix."""
    payload, record = _load_json(path, description="P11 A1+A2 qualification receipt")
    if payload.get("schema") != PREDICTION_SCHEMA or payload.get("task_id") != TASK_ID or payload.get("status") != QUALIFICATION_STATUS:
        raise A1A2RuntimeError("qualification receipt is not a completed truth-free qualification")
    if payload.get("qualification_only") is not True or payload.get("truth_opened") is not False or payload.get("p03_holdout_accessed") is not False:
        raise A1A2RuntimeError("qualification receipt boundary is changed")
    if payload.get("selection_sha256") != binding.get("selection_sha256"):
        raise A1A2RuntimeError("qualification selection binding differs from the current runtime")
    observed = payload.get("observation_manifest")
    if not isinstance(observed, Mapping) or observed.get("sha256") != observation_record.get("sha256"):
        raise A1A2RuntimeError("qualification observation binding differs from the current capture")
    cell = payload.get("qualification_cell")
    if not isinstance(cell, str) or cell not in CELL_ORDER:
        raise A1A2RuntimeError("qualification cell identity is absent or changed")
    if expected_cell is not None and cell != expected_cell:
        raise A1A2RuntimeError("qualification cell does not match the requested restart")
    cells = payload.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != {cell}:
        raise A1A2RuntimeError("qualification receipt does not contain exactly one retained cell")
    cell_payload = cells[cell]
    if not isinstance(cell_payload, Mapping):
        raise A1A2RuntimeError("qualification cell payload is malformed")
    for key in ("file", "trace", "cost", "receipt"):
        value = cell_payload.get(key)
        if not isinstance(value, Mapping):
            raise A1A2RuntimeError(f"qualification {key} binding is absent")
        _record(
            Path(str(value.get("path", ""))),
            expected_sha256=str(value.get("sha256")),
            expected_bytes=int(value.get("bytes")) if value.get("bytes") is not None else None,
            description=f"qualification {key} {cell}",
        )
    return payload, record


def _watchdog_gpu_index(device: Any) -> int:
    value = str(device)
    if ":" in value:
        value = value.rsplit(":", 1)[1]
    try:
        return int(value)
    except ValueError as exc:
        raise A1A2RuntimeError(f"cannot determine CUDA device index: {device}") from exc


def _start_external_watchdog(
    *,
    root: Path,
    output: Path,
    device: Any,
    maximum_rss_gib: float,
    minimum_free_gpu_gib: float,
    maximum_seconds: float | None,
    phase: str,
) -> tuple[subprocess.Popen[Any], Path, list[str]]:
    script = root / "scripts" / "trr_p11" / "live_watchdog.py"
    if script.is_symlink() or not script.is_file():
        raise A1A2RuntimeError(f"P11 live watchdog is unavailable: {script}")
    receipt_path = output / f"resource_watchdog_{phase}.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise A1A2RuntimeError(f"P11 live watchdog receipt is create-only: {receipt_path}")
    command = [
        sys.executable,
        str(script),
        "--parent-pid",
        str(__import__("os").getpid()),
        "--output",
        str(receipt_path),
        "--minimum-free-gib",
        str(minimum_free_gpu_gib),
        "--maximum-rss-gib",
        str(maximum_rss_gib),
        "--poll-seconds",
        str(DEFAULT_WATCHDOG_POLL_SECONDS),
        "--gpu-index",
        str(_watchdog_gpu_index(device)),
    ]
    if maximum_seconds is not None:
        command.extend(["--maximum-seconds", str(maximum_seconds)])
    try:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        raise A1A2RuntimeError("failed to start external P11 live watchdog") from exc
    return process, receipt_path, command


def _stop_external_watchdog(
    process: subprocess.Popen[Any] | None,
    receipt_path: Path | None,
    *,
    root: Path,
    command: Sequence[str] | None,
) -> dict[str, Any] | None:
    if process is None or receipt_path is None:
        return None
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    if not receipt_path.exists() and not receipt_path.is_symlink():
        _write_create_only(
            receipt_path,
            {
                "schema": "token-reconstruction.trr-p11-live-watchdog.v1",
                "task_id": TASK_ID,
                "status": "STOPPED_BY_RUNTIME_AFTER_PASS",
                "parent_pid": __import__("os").getpid(),
                "command": list(command or ()),
            },
            root=root,
            description="P11 live watchdog receipt",
        )
    return {
        "status": "EXTERNAL_WATCHDOG_STOPPED",
        "command": list(command or ()),
        "process_returncode": process.returncode,
        "receipt": _record(receipt_path, description="P11 live watchdog receipt"),
    }


def run_native_a1_a2(
    *,
    binding_path: Path,
    observations_path: Path,
    output_root: Path,
    repository_root: Path,
    device: str = "cuda",
    execute: bool = False,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
    qualification_cell: str | None = None,
    qualification_only: bool = False,
    qualification_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Run native A1+A2 on first-128 rows of each fresh P11 observation cell."""
    if not execute:
        raise A1A2RuntimeError("P11 A1+A2 runtime requires explicit execute=True")
    root = Path(repository_root).expanduser().resolve()
    output = Path(output_root).expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    allowed = (root / "experiments" / TASK_ID / "evaluation").resolve()
    try:
        output.relative_to(allowed)
    except ValueError as exc:
        raise A1A2RuntimeError(f"A1+A2 output must be below {allowed}") from exc
    if output.exists() or output.is_symlink():
        if qualification_only or qualification_receipt_path is None or not output.is_dir():
            raise A1A2RuntimeError(f"A1+A2 output is create-only: {output}")
    else:
        output.mkdir(parents=True)
    if qualification_only and qualification_receipt_path is not None:
        raise A1A2RuntimeError("qualification_only cannot receive a prior qualification receipt")
    if not qualification_only and qualification_receipt_path is None:
        raise A1A2RuntimeError("full A1+A2 matrix requires a prior retained qualification receipt")
    started_utc = _utc_now()
    started_clock = time.perf_counter()
    failure_path = output / "failure.json"
    watchdog_process: subprocess.Popen[Any] | None = None
    watchdog_receipt_path: Path | None = None
    watchdog_command: list[str] | None = None
    try:
        binding, _binding_record = _load_json(Path(binding_path), description="P11 A1+A2 runtime binding")
        validate_runtime_binding(binding, root=root)
        observations, observation_record = _load_observation_manifest(Path(observations_path), root=root)
        if observations.get("selection_plan_sha256") != binding.get("selection_sha256"):
            raise A1A2RuntimeError("A1+A2 observations and binding use different P11 selection")
        # Only now cross-check the metadata-only first-128 identity digest.
        selection_record = binding.get("selection")
        if not isinstance(selection_record, Mapping):
            raise A1A2RuntimeError("A1+A2 selection binding is absent")
        selection, _ = _load_json(Path(str(selection_record["path"])), description="P11 selection for A1+A2")
        selection_context = selector._validate_selection_payload(selection, root=root)
        subset = first128_subset_binding(selection_context)
        if subset["record_ids_sha256"] != binding["subset"]["record_ids_sha256"] or subset["h128_sequence_sha256_digest"] != binding["subset"]["h128_sequence_sha256_digest"]:
            raise A1A2RuntimeError("A1+A2 first-128 selection identity changed")
        if qualification_only:
            cells_to_run = execution_cells(qualification_cell=qualification_cell, qualification_only=True)
            qualification_payload = None
            qualification_record = None
            reused_qualification_cell = None
        else:
            expected_qualification_cell = qualification_cell or LARGEST_QUALIFICATION_CELL
            qualification_path = Path(qualification_receipt_path).expanduser()
            if not qualification_path.is_absolute():
                qualification_path = root / qualification_path
            qualification_path = qualification_path.resolve()
            if qualification_path.parent != output:
                raise A1A2RuntimeError("qualification receipt and resumed matrix must share the output root")
            qualification_payload, qualification_record = _load_qualification_receipt(
                qualification_path,
                binding=binding,
                observation_record=observation_record,
                expected_cell=expected_qualification_cell,
            )
            reused_qualification_cell = str(qualification_payload["qualification_cell"])
            cells_to_run = execution_cells(qualification_cell=reused_qualification_cell, reuse_qualification=True)

        import gc
        import torch
        from safetensors.torch import save_file
        script_dir = root / "scripts"
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        from scripts import trr0003_footing_compare as footing
        from scripts import trr0004_predict_confirmation as legacy

        torch_device = torch.device(device)
        if torch_device.type != "cuda" or not torch.cuda.is_available():
            raise A1A2RuntimeError("native A1+A2 K256 requires CUDA")
        guard_args = type("GuardArgs", (), {
            "minimum_free_gib": 8.0,
            "maximum_reserved_gib": 6.0,
            "maximum_rss_gib": 16.0,
            "max_seconds": float(max_seconds) if max_seconds is not None else float("inf"),
        })()
        watchdog_phase = "qualification" if qualification_only else "matrix"
        watchdog_process, watchdog_receipt_path, watchdog_command = _start_external_watchdog(
            root=root,
            output=output,
            device=torch_device,
            maximum_rss_gib=float(guard_args.maximum_rss_gib),
            minimum_free_gpu_gib=float(guard_args.minimum_free_gib),
            maximum_seconds=max_seconds,
            phase=watchdog_phase,
        )
        preflight_events = [legacy._resource_preflight(guard_args, torch_device, stage="before_a1_a2_load", started=started_clock)]
        resources = binding["resources"]
        snapshot = Path(str(resources["model_snapshot"]["path"]))
        reference = Path(str(resources["public_reference"]["path"]))
        lens = Path(str(resources["retained_a1_lens"]["path"]))
        embedding = Path(str(resources["public_embedding_table"]["path"]))
        precut, lens_module, embeddings, public_evidence = legacy._load_public_prefix(
            snapshot=snapshot,
            reference_path=reference,
            lens_path=lens,
            embedding_path=embedding,
            device=torch_device,
        )
        policy = footing._fixed_k256_policy()
        adapter = legacy._A2Adapter(
            precut=precut,
            lens=lens_module,
            embeddings=embeddings,
            device=torch_device,
            policy=policy,
        )
        cells: dict[str, Any] = {}
        if qualification_payload is not None:
            cells.update({str(key): dict(value) for key, value in qualification_payload["cells"].items()})
        subset_indices = list(range(A1_A2_RECORDS_PER_DOMAIN))
        subset_indices_sha256 = _canonical_digest(subset_indices)
        state_file_binding = dict(resources["retained_a1_lens"])
        for cell in cells_to_run:
            descriptor = next((item.get("observation") for item in observations.get("cells", []) if isinstance(item, Mapping) and item.get("cell_id") == cell), None)
            if not isinstance(descriptor, Mapping):
                raise A1A2RuntimeError(f"P11 observation cell is absent: {cell}")
            _validate_observation_descriptor(descriptor, cell=cell)
            observation_file = _record(
                Path(str(descriptor.get("path", ""))),
                expected_sha256=str(descriptor.get("sha256")),
                description=f"P11 observation {cell}",
            )
            activation, mask, positions = _load_prediction_tensors(Path(observation_file["path"]))
            if tuple(activation.shape) != (selector.RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS, HIDDEN_SIZE) or tuple(mask.shape) != (selector.RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS) or tuple(positions.shape) != (selector.RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS) or str(activation.dtype) not in {"torch.bfloat16", "bfloat16"} or str(mask.dtype) not in {"torch.uint8", "uint8", "torch.bool", "bool"} or str(positions.dtype) not in {"torch.int64", "int64", "torch.long", "long"}:
                raise A1A2RuntimeError(f"P11 observation geometry or dtype changed: {cell}")
            cell_start = len(preflight_events)
            preflight_events.append(legacy._resource_preflight(guard_args, torch_device, stage=f"before_{cell}_a1_a2", started=started_clock))
            adapter.begin_cell()
            # Preserve the exact native warmup plus three measured calls.  The
            # first measured output is the accuracy output and the next two
            # must compare exactly; _A2Adapter retains the first measured
            # proposal for every record under this cadence.
            prediction_tensor, timing = legacy.fc.run_warmed_prediction(
                observations=activation[:A1_A2_RECORDS_PER_DOMAIN],
                attention_mask=mask[:A1_A2_RECORDS_PER_DOMAIN],
                position_ids=positions[:A1_A2_RECORDS_PER_DOMAIN],
                predictor=adapter,
                device=torch_device,
                warmup_runs=1,
                measured_runs=3,
            )
            if tuple(prediction_tensor.shape) != (A1_A2_RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS):
                raise A1A2RuntimeError(f"A1+A2 prediction geometry changed: {cell}")
            if not bool(prediction_tensor[:, 0].eq(selector.BOS_TOKEN_ID).all().item()):
                raise A1A2RuntimeError(f"A1+A2 BOS normalization changed: {cell}")
            if bool(prediction_tensor.lt(0).any().item()) or bool(prediction_tensor.ge(selector.VOCAB_SIZE).any().item()):
                raise A1A2RuntimeError(f"A1+A2 prediction token range changed: {cell}")
            candidates, candidate_scores = adapter.candidate_tensors(
                records=A1_A2_RECORDS_PER_DOMAIN,
                sequence_tokens=STORED_SEQUENCE_TOKENS,
            )
            if tuple(candidates.shape) != (A1_A2_RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS, DEFAULT_A2_PROPOSAL_K) or tuple(candidate_scores.shape) != tuple(candidates.shape):
                raise A1A2RuntimeError(f"A1+A2 candidate trace geometry changed: {cell}")
            prediction_tensor = prediction_tensor.to(device="cpu", dtype=torch.long).contiguous()
            prediction_sha256 = _tensor_digest(prediction_tensor)
            prediction_path = output / "predictions" / f"{cell}.safetensors"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {"predictions": prediction_tensor},
                str(prediction_path),
                metadata={
                    "schema": PREDICTION_SCHEMA,
                    "task_id": TASK_ID,
                    "method_id": METHOD_ID,
                    "cell_id": cell,
                    "records": str(A1_A2_RECORDS_PER_DOMAIN),
                    "stored_sequence_tokens": str(STORED_SEQUENCE_TOKENS),
                    "scored_post_bos_tokens": str(SCORED_POST_BOS_TOKENS),
                    "selection_sha256": str(binding["selection_sha256"]),
                    "prediction_tensor_sha256": prediction_sha256,
                    "candidate_arrays_persisted": "true",
                    "source_text_written": "false",
                    "token_ids_written": "false",
                    "target_labels_loaded": "false",
                    "truth_opened": "false",
                },
            )
            output_record = _record(prediction_path, description=f"A1+A2 prediction {cell}")
            trace_path = output / "traces" / f"{cell}.safetensors"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {
                    "candidates": candidates.to(device="cpu", dtype=torch.int64).contiguous(),
                    "candidate_scores": candidate_scores.to(device="cpu", dtype=torch.float32).contiguous(),
                },
                str(trace_path),
                metadata={
                    "schema": "token-reconstruction.trr-p11-a1-a2-candidate-trace.v1",
                    "task_id": TASK_ID,
                    "method_id": METHOD_ID,
                    "cell_id": cell,
                    "records": str(A1_A2_RECORDS_PER_DOMAIN),
                    "stored_sequence_tokens": str(STORED_SEQUENCE_TOKENS),
                    "proposal_budget": str(DEFAULT_A2_PROPOSAL_K),
                    "candidate_budget": str(DEFAULT_A2_K),
                    "selection_sha256": str(binding["selection_sha256"]),
                    "truth_opened": "false",
                },
            )
            trace_record = _record(trace_path, description=f"A1+A2 candidate trace {cell}")
            cell_preflight = preflight_events[cell_start:]
            adapter_evidence = adapter.evidence()
            cost_payload = {
                "schema": "token-reconstruction.trr-p11-a1-a2-cost.v1",
                "task_id": TASK_ID,
                "method_id": METHOD_ID,
                "cell_id": cell,
                "timing": timing,
                "adapter": adapter_evidence,
                "resource_preflight": cell_preflight,
                "truth_opened": False,
                "p03_holdout_accessed": False,
            }
            cost_record = _write_create_only(
                output / "costs" / f"{cell}.json",
                cost_payload,
                root=root,
                description=f"A1+A2 cost receipt {cell}",
            )
            receipt_payload = {
                "schema": PREDICTION_SCHEMA,
                "task_id": TASK_ID,
                "status": PREDICTION_STATUS,
                "method_id": METHOD_ID,
                "cell_id": cell,
                "output": output_record,
                "observations": dict(descriptor),
                "methods": {
                    "a1_a2_k256": {
                        "state_file_binding": state_file_binding,
                    }
                },
                "runtime": {
                    "dependencies": {
                        "code_bindings": binding["code_bindings"],
                        "resources": binding["resources"],
                    }
                },
                "trace": trace_record,
                "cost": cost_record,
                "tensor_sha256": prediction_sha256,
                "candidate_arrays_persisted": True,
                "source_text_loaded": False,
                "token_ids_loaded": False,
                "target_labels_loaded": False,
                "truth_opened": False,
                "p03_holdout_accessed": False,
            }
            receipt_record = _write_create_only(
                output / "predictions" / f"{cell}.receipt.json",
                receipt_payload,
                root=root,
                description=f"A1+A2 prediction receipt {cell}",
            )
            preflight_events.append(legacy._resource_preflight(guard_args, torch_device, stage=f"after_{cell}_a1_a2", started=started_clock))
            cells[cell] = {
                "cell_id": cell,
                "records": A1_A2_RECORDS_PER_DOMAIN,
                "file": output_record,
                "prediction": output_record,
                "tensor_key": "predictions",
                "tensor_sha256": prediction_sha256,
                "input_observation": observation_file,
                "observations": dict(descriptor),
                "subset": {
                    "parent_cell": cell,
                    "indices": subset_indices,
                    "indices_sha256": subset_indices_sha256,
                    "records": A1_A2_RECORDS_PER_DOMAIN,
                },
                "receipt": receipt_record,
                "trace": trace_record,
                "cost": cost_record,
                "timing": timing,
                "adapter": adapter_evidence,
                "candidate_arrays_persisted": True,
                "truth_opened": False,
            }
            del activation, mask, positions, prediction_tensor, candidates, candidate_scores
            gc.collect()
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
        watchdog_record = _stop_external_watchdog(
            watchdog_process,
            watchdog_receipt_path,
            root=root,
            command=watchdog_command,
        )
        watchdog_process = None
        run_status = QUALIFICATION_STATUS if qualification_only else PREDICTION_STATUS
        receipt = {
            "schema": PREDICTION_SCHEMA,
            "task_id": TASK_ID,
            "status": run_status,
            "qualification_only": qualification_only,
            "qualification_cell": next(iter(cells)) if qualification_only else reused_qualification_cell,
            "qualification_reused": qualification_payload is not None,
            "qualification_receipt": qualification_record,
            "method_id": METHOD_ID,
            "runtime_binding": _record(Path(binding_path), description="P11 A1+A2 runtime binding"),
            "observation_manifest": observation_record,
            "selection_sha256": binding["selection_sha256"],
            "subset": binding["subset"],
            "policy": validate_native_policy(binding["policy"]),
            "cells": cells,
            "trace_files": [cells[cell]["trace"] for cell in CELL_ORDER if cell in cells],
            "cost_files": [cells[cell]["cost"] for cell in CELL_ORDER if cell in cells],
            "candidate_arrays_persisted": True,
            "resource_preflight": preflight_events,
            "resource_watchdog": watchdog_record,
            "public_prefix": public_evidence,
            "execution": {
                "started_utc": started_utc,
                "ended_utc": _utc_now(),
                "elapsed_seconds": time.perf_counter() - started_clock,
                "command": list(sys.argv),
                "qualification_only": qualification_only,
                "qualification_cell": next(iter(cells)) if qualification_only else reused_qualification_cell,
                "code_commit": _git_head(root),
                "python": sys.executable,
                "python_version": platform.python_version(),
                "device": str(torch_device),
                "model_loaded": True,
                "source_text_loaded": False,
                "source_text_written": False,
                "token_ids_written": False,
                "target_labels_loaded": False,
                "truth_opened": False,
                "p03_holdout_accessed": False,
            },
        }
        receipt_name = "qualification.json" if qualification_only else "predictions.json"
        receipt_record = _write_create_only(output / receipt_name, receipt, root=root, description=f"P11 A1+A2 {receipt_name} receipt")
        result = {"task_id": TASK_ID, "status": run_status, "truth_opened": False}
        result["qualification_receipt" if qualification_only else "receipt"] = receipt_record
        return result
    except Exception as exc:
        if watchdog_process is not None:
            try:
                _stop_external_watchdog(
                    watchdog_process,
                    watchdog_receipt_path,
                    root=root,
                    command=watchdog_command,
                )
            except Exception:
                pass
        if not failure_path.exists() and not failure_path.is_symlink():
            _write_create_only(
                failure_path,
                {
                    "schema": PREDICTION_SCHEMA,
                    "task_id": TASK_ID,
                    "status": "A1_A2_K256_QUALIFICATION_OR_MATRIX_FAILED_NO_TRUTH",
                    "started_utc": started_utc,
                    "ended_utc": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "truth_opened": False,
                    "p03_holdout_accessed": False,
                },
                root=root,
                description="P11 A1+A2 failure receipt",
            )
        if isinstance(exc, A1A2RuntimeError):
            raise
        raise A1A2RuntimeError("P11 native A1+A2 run failed") from exc


def resource_plan() -> dict[str, Any]:
    """Return the fail-closed CUDA plan and its geometry-only preflight basis."""
    observation_elements = A1_A2_RECORDS_PER_DOMAIN * STORED_SEQUENCE_TOKENS * HIDDEN_SIZE
    observation_bytes = observation_elements * 2  # BF16
    candidate_elements = A1_A2_RECORDS_PER_DOMAIN * STORED_SEQUENCE_TOKENS * DEFAULT_A2_PROPOSAL_K
    candidate_bytes = candidate_elements * 8  # int64 proposal IDs
    candidate_score_bytes = candidate_elements * 4  # float32 proposal scores
    known_transient_bytes = observation_bytes + candidate_bytes + candidate_score_bytes
    return {
        "method_id": METHOD_ID,
        "device_required": "cuda",
        "model_dtype": "bfloat16",
        "records_per_domain": A1_A2_RECORDS_PER_DOMAIN,
        "stored_sequence_tokens": STORED_SEQUENCE_TOKENS,
        "capture_batch_records": selector.CAPTURE_BATCH_RECORDS,
        "candidate_proposal_budget": DEFAULT_A2_PROPOSAL_K,
        "candidate_budget": DEFAULT_A2_K,
        "proposal_chunk": DEFAULT_A1_CHUNK,
        "record_batch_size": DEFAULT_RECORD_BATCH_SIZE,
        "minimum_free_gpu_gib": DEFAULT_MINIMUM_FREE_GIB,
        "maximum_reserved_gpu_gib": DEFAULT_MAXIMUM_RESERVED_GIB,
        "maximum_host_rss_gib": DEFAULT_MAXIMUM_RSS_GIB,
        "maximum_wall_seconds": DEFAULT_MAX_SECONDS,
        "external_live_watchdog_required": True,
        "watchdog_poll_seconds": DEFAULT_WATCHDOG_POLL_SECONDS,
        "largest_representative_cell": LARGEST_QUALIFICATION_CELL,
        "qualification_required_before_full_run": True,
        "candidate_arrays_persisted": True,
        "truth_opened": False,
        "preflight_basis": {
            "basis": "geometry-only upper bound; no P11 runtime has executed",
            "observation_elements": observation_elements,
            "observation_bytes_bfloat16": observation_bytes,
            "candidate_elements": candidate_elements,
            "candidate_bytes_int64": candidate_bytes,
            "candidate_score_bytes_float32": candidate_score_bytes,
            "known_transient_bytes_per_qualified_cell": known_transient_bytes,
            "known_transient_gib_per_qualified_cell": known_transient_bytes / float(2**30),
            "public_embedding_bytes_bound": EXPECTED_EMBEDDING_BYTES,
            "model_workspace_bound": "runtime-dependent; enforced by free/reserved/RSS guards",
            "peak_memory_measurement": "required from the largest representative cell before the four-cell matrix",
            "wall_time_measurement": "required from the largest representative cell before the four-cell matrix",
            "external_watchdog": "separate stdlib child polls parent RSS, GPU free memory, temperature, and compute-app exclusivity between cell checks",
            "equivalence_check": "warmup output vs three measured outputs exact per record; first measured proposal trace retained for every record",
        },
    }


__all__ = [
    "A1A2RuntimeError",
    "CODE_BINDINGS",
    "METHOD_ID",
    "PREDICTION_SCHEMA",
    "bind_native_runtime",
    "execution_cells",
    "first128_subset_binding",
    "resource_plan",
    "run_native_a1_a2",
    "validate_native_policy",
    "validate_runtime_binding",
]
