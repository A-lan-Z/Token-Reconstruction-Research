"""Fail-closed TRR-P11 evaluation freeze, truth gate, and scorer adapter.

This module is the narrow bridge between Agent1's restored package outputs and
P10's validated paired-inventory/scorer primitives.  It consumes immutable
file bindings and sibling prediction receipts; it does not capture sources,
load a model, select records, or infer candidate ranks from final predictions.

The pretruth phase validates the complete four-cell matrix for B0 and B1 and
the explicitly bound first-128 A1 comparator.  A pre-registered technical A1
blocker produces an explicit qualified-partial freeze; a missing comparator is
never silently dropped.  The truth phase revalidates every pretruth binding
before opening one truth sidecar, then delegates inventory construction and
numerical scorer calls to the byte-identical P10/scorer implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open

from scripts.trr_p10 import execution as p10

TASK_ID = "TRR-P11"
MANIFEST_SCHEMA = "token-reconstruction.trr-p11-evaluation-manifest.v1"
FREEZE_SCHEMA = "token-reconstruction.trr-p11-evaluation-freeze.v1"
TRUTH_SCHEMA = "token-reconstruction.trr-p11-evaluation-truth.v1"
SCORE_SCHEMA = "token-reconstruction.trr-p11-evaluation-score.v1"
PREDICTION_RECEIPT_SCHEMA = "token-reconstruction.trr0012-prediction-receipt.v1"
RESTORE_STATUS = "PASS_RESTORED_SMOKE"
RESTORE_STATE_ASSETS = {
    "new_current_fixed_B0": "current_fixed",
    "new_expanded_fixed_B1": "expanded_fixed",
}
RESTORE_STATE_BANKS = {
    "new_current_fixed_B0": "B0",
    "new_expanded_fixed_B1": "B1",
}
RESTORE_TENSOR_ASSETS = ("current_fixed", "expanded_fixed", "public_readout")
RESTORE_DEPLOYMENT_ASSETS = (
    "public_readout",
    "loader_code",
    "decoder_code",
    "package_cli",
    "package_manifest",
    "frozen_config",
)

DOMAIN_ORDER = ("pile", "finance")
TARGET_ORDER = ("public_base", "public_lora_2601")
CELL_ORDER = tuple(f"{domain}__{target}" for domain in DOMAIN_ORDER for target in TARGET_ORDER)
PRIMARY_METHODS = ("new_current_fixed_B0", "new_expanded_fixed_B1")
COMPARATOR_METHOD = "a1_a2_k256"
METHOD_ORDER = PRIMARY_METHODS + (COMPARATOR_METHOD,)
METHOD_OUTPUT_KEYS = {
    "new_current_fixed_B0": "current_fixed",
    "new_expanded_fixed_B1": "expanded_fixed",
}
RECORDS_PER_DOMAIN = 256
COMPARATOR_RECORDS_PER_DOMAIN = 128
STORED_SEQUENCE_TOKENS = 128
SCORED_POST_BOS_TOKENS = 127
BOS_TOKEN_ID = 128000
VOCABULARY_SIZE = 128256
BOOTSTRAP_SEED = 9009
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_ALPHA = 0.025
SCORER_RELATIVE_PATH = "scripts/trr_p11/scorer/trr0010_analysis.py"
SCORER_SHA256 = "90078ac78bfcdcfb5f782a598c943b78cb0417f6895906a1e6d61056a3793cef"
SCORER_SOURCE_COMMIT = "70c57db7643913eea97cc606775b3f1f3807967a"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_TRUE = (
    "truth_opened",
    "truth_created",
    "target_labels_loaded",
    "source_text_written",
    "token_ids_written",
    "p03_holdout_accessed",
)
SUPPORT_HISTOGRAM_SCHEMA = "token-reconstruction.trr-p11-public-support-histogram.v1"
SUPPORT_EXECUTION_SCHEMA = "token-reconstruction.trr-p11-public-support-histogram-execution.v1"
SUPPORT_RECOVERY_EXECUTION_SCHEMA = "token-reconstruction.trr-p11-common-frequency-recovery-execution.v1"
SUPPORT_FREQUENCY_BINS = (
    ("unseen_0", 0, 0),
    ("seen_1", 1, 1),
    ("seen_2_4", 2, 4),
    ("seen_5_16", 5, 16),
    ("seen_17_64", 17, 64),
    ("seen_65_plus", 65, None),
)
SUPPORT_POSITION_BINS = (
    ("1-15", 1, 15, True),
    ("16-39", 16, 39, True),
    ("40-79", 40, 79, True),
    ("80-127", 80, 127, True),
    ("128-191", 128, 191, False),
)


class EvaluationError(ValueError):
    """Raised when the P11 evaluation boundary is incomplete or changed."""


class QualifiedPartial(EvaluationError):
    """Raised when a declared technical comparator blocker prevents full scoring."""


def _is_true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


def _is_false(value: Any) -> bool:
    return value is False or (isinstance(value, str) and value.lower() == "false")


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
        raise EvaluationError("canonical JSON is not serializable") from exc


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return p10.sha256_file(Path(path))


def _require_sha(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise EvaluationError(f"{description} hash is malformed")
    return value


def _path_value(value: Any, *, root: Path, description: str) -> Any:
    if isinstance(value, (str, Path)):
        return value
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{description} binding is malformed")
    for key in ("path", "loaded_path", "relative_path"):
        candidate = value.get(key)
        if candidate:
            if key == "relative_path":
                return root / str(candidate)
            return candidate
    raise EvaluationError(f"{description} path is absent")


def _record(
    value: Any,
    *,
    root: Path,
    description: str,
    declared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(_path_value(value, root=root, description=description)).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        actual = p10.file_record(path, base=root, description=description)
    except Exception as exc:
        if isinstance(exc, EvaluationError):
            raise
        raise EvaluationError(f"{description} cannot be recorded") from exc
    if declared is not None:
        declared_path = _path_value(declared, root=root, description=description)
        normalized = {
            "path": str(declared_path),
            "bytes": declared.get("bytes"),
            "sha256": declared.get("sha256"),
        }
        try:
            p10.file_record(path, base=root, description=description, declared=normalized)
        except Exception as exc:
            raise EvaluationError(f"{description} binding changed") from exc
    return {"path": actual["path"], "bytes": int(actual["bytes"]), "sha256": str(actual["sha256"])}


def _same_content(left: Mapping[str, Any], right: Mapping[str, Any], *, description: str) -> None:
    if int(left.get("bytes", -1)) != int(right.get("bytes", -2)):
        raise EvaluationError(f"{description} byte binding changed")
    if left.get("sha256") != right.get("sha256"):
        raise EvaluationError(f"{description} hash binding changed")


def _load_json_binding(value: Any, *, root: Path, description: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _record(value, root=root, description=description, declared=value if isinstance(value, Mapping) else None)
    try:
        payload = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvaluationError(f"{description} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"{description} JSON is not an object")
    return payload, record


def _write_json(path: Path, payload: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=False)
    if path.exists() or path.is_symlink():
        raise EvaluationError(f"{description} output is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise EvaluationError(f"{description} output is create-only: {path}") from exc
    return _record(path, root=root, description=description)


def _assert_pretruth(payload: Mapping[str, Any], *, description: str) -> None:
    for key in _FORBIDDEN_TRUE:
        if _is_true(payload.get(key)):
            raise EvaluationError(f"{description} records forbidden access: {key}")
    for key in _FORBIDDEN_TRUE:
        if key not in payload and key != "p03_holdout_accessed":
            # Older receipts may omit a false compatibility flag, but an
            # evaluation manifest/freeze must state the P03 boundary explicitly.
            continue
    if "source_text_or_target_labels" in payload and payload.get("source_text_or_target_labels") is not False:
        raise EvaluationError(f"{description} records source text or target labels")
    if payload.get("p03_holdout_accessed") is not False:
        raise EvaluationError(f"{description} does not explicitly close the P03 boundary")


def _expected_record_count(method: str) -> int:
    return COMPARATOR_RECORDS_PER_DOMAIN if method == COMPARATOR_METHOD else RECORDS_PER_DOMAIN


def _extract_record_ids(selection: Mapping[str, Any]) -> dict[str, str]:
    rule = selection.get("selection_rule")
    if not isinstance(rule, Mapping) or not isinstance(rule.get("record_ids_sha256"), Mapping):
        raise EvaluationError("selection record-order hashes are absent")
    result: dict[str, str] = {}
    for domain in DOMAIN_ORDER:
        result[domain] = _require_sha(rule["record_ids_sha256"].get(domain), description=f"selection {domain} record IDs")
    return result


def _validate_selection(value: Any, *, root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload, record = _load_json_binding(value, root=root, description="P11 selection")
    if payload.get("task_id") != TASK_ID or payload.get("status") != "FROZEN_TRR-P11_SOURCE_SELECTION_NO_TRUTH":
        raise EvaluationError("selection is not the frozen P11 pretruth selection")
    _assert_pretruth(payload, description="P11 selection")
    ids = _extract_record_ids(payload)
    rule = payload.get("selection_rule")
    if not isinstance(rule, Mapping) or not isinstance(rule.get("records"), Mapping):
        raise EvaluationError("selection rows are absent; canonical H128 binding cannot be checked")
    h128_by_domain: dict[str, list[str]] = {}
    record_ids_by_domain: dict[str, list[str]] = {}
    h128_list_sha256: dict[str, str] = {}
    declared_h128_lists = rule.get("h128_sequence_sha256")
    if not isinstance(declared_h128_lists, Mapping):
        raise EvaluationError("selection H128 list hashes are absent")
    for domain in DOMAIN_ORDER:
        rows = rule["records"].get(domain)
        if not isinstance(rows, list) or len(rows) != RECORDS_PER_DOMAIN:
            raise EvaluationError(f"selection rows are absent or incomplete: {domain}")
        record_ids: list[str] = []
        h128_values: list[str] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise EvaluationError(f"selection row is malformed: {domain}/{index}")
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise EvaluationError(f"selection record ID is malformed: {domain}/{index}")
            h128 = _require_sha(row.get("h128_sequence_sha256"), description=f"selection {domain}/{index} H128")
            if row.get("final_sequence_sha256") != h128:
                raise EvaluationError(f"selection final/H128 identity differs: {domain}/{index}")
            record_ids.append(record_id)
            h128_values.append(h128)
        if _json_digest(record_ids) != ids[domain]:
            raise EvaluationError(f"selection record-order rows differ: {domain}")
        declared_h128_list = _require_sha(declared_h128_lists.get(domain), description=f"selection {domain} H128 list")
        if _json_digest(h128_values) != declared_h128_list:
            raise EvaluationError(f"selection H128 rows differ: {domain}")
        record_ids_by_domain[domain] = record_ids
        h128_by_domain[domain] = h128_values
        h128_list_sha256[domain] = declared_h128_list
    counts = payload.get("records_by_domain")
    if not isinstance(counts, Mapping) or any(int(counts.get(domain, -1)) != RECORDS_PER_DOMAIN for domain in DOMAIN_ORDER):
        raise EvaluationError("selection record counts changed")
    return payload, record, {
        "record_ids_sha256": ids,
        "record_ids": record_ids_by_domain,
        "h128_sequence_sha256": h128_by_domain,
        "h128_sequence_list_sha256": h128_list_sha256,
    }


def _observation_tensor_digests(path: Path, *, cell: str) -> tuple[dict[str, str], dict[str, str]]:
    """Rehash raw capture tensors and exact Agent1 consumer-normalized tensors."""
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                raise EvaluationError(f"observation tensor keys changed: {cell}")
            tensors = {key: handle.get_tensor(key).detach().cpu().contiguous() for key in handle.keys()}
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"observation artifact is unreadable: {cell}") from exc
    raw = {key: p10.tensor_digest(tensors[key]) for key in ("activations", "attention_mask", "position_ids")}
    normalized_tensors = {
        "activations": tensors["activations"],
        "attention_mask": tensors["attention_mask"].to(torch.bool).contiguous(),
        "position_ids": tensors["position_ids"].to(torch.int64).contiguous(),
    }
    normalized = {key: p10.tensor_digest(normalized_tensors[key]) for key in normalized_tensors}
    return raw, normalized


def _validate_observation(
    value: Any,
    *,
    root: Path,
    cell: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"observation binding is malformed: {cell}")
    record = _record(value, root=root, description=f"observation {cell}", declared=value)
    capture_order = value.get("capture_record_order")
    if not isinstance(capture_order, list) or len(capture_order) != RECORDS_PER_DOMAIN or any(not isinstance(item, str) or not item for item in capture_order):
        raise EvaluationError(f"observation capture record order is absent or malformed: {cell}")
    capture_order_sha = _require_sha(value.get("capture_record_order_sha256"), description=f"observation {cell} capture record order")
    if capture_order_sha != _json_digest(capture_order):
        raise EvaluationError(f"observation capture record order digest changed: {cell}")
    record_order = value.get("record_order")
    expected_package_order = [f"record/{index:03d}" for index in range(RECORDS_PER_DOMAIN)]
    if record_order != expected_package_order:
        raise EvaluationError(f"observation package record order changed: {cell}")
    declared_order_sha = _require_sha(value.get("record_order_sha256"), description=f"observation {cell} record order")
    if declared_order_sha != _json_digest(record_order):
        raise EvaluationError(f"observation record order digest changed: {cell}")
    declared_raw = value.get("capture_tensor_sha256")
    declared_normalized = value.get("normalized_tensor_sha256")
    tensor_sha = value.get("tensor_sha256")
    if not isinstance(declared_raw, Mapping) or not isinstance(declared_normalized, Mapping) or not isinstance(tensor_sha, Mapping):
        raise EvaluationError(f"observation raw/normalized tensor bindings are incomplete: {cell}")
    if dict(declared_normalized) != dict(tensor_sha):
        raise EvaluationError(f"observation normalized tensor aliases differ: {cell}")
    raw_tensor_sha: dict[str, str] = {}
    normalized_tensor_sha: dict[str, str] = {}
    for key in ("activations", "attention_mask", "position_ids"):
        raw_tensor_sha[key] = _require_sha(declared_raw.get(key), description=f"observation {cell}/capture/{key}")
        normalized_tensor_sha[key] = _require_sha(declared_normalized.get(key), description=f"observation {cell}/normalized/{key}")
    actual_raw, actual_normalized = _observation_tensor_digests(Path(record["path"]), cell=cell)
    if raw_tensor_sha != actual_raw:
        raise EvaluationError(f"observation raw tensor binding changed: {cell}")
    if normalized_tensor_sha != actual_normalized:
        raise EvaluationError(f"observation normalized tensor binding changed: {cell}")
    return {
        **record,
        "capture_tensor_sha256": raw_tensor_sha,
        "normalized_tensor_sha256": normalized_tensor_sha,
        "tensor_sha256": normalized_tensor_sha,
        "capture_record_order": list(capture_order),
        "capture_record_order_sha256": capture_order_sha,
        "record_order": list(record_order),
        "record_order_sha256": declared_order_sha,
    }


def _validate_capture(
    value: Any,
    *,
    root: Path,
    selection_identity: Mapping[str, Any],
    observation_bindings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    payload, record = _load_json_binding(value, root=root, description="P11 capture")
    if payload.get("task_id") != TASK_ID or payload.get("status") != "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH":
        raise EvaluationError("capture is not the complete P11 pretruth capture")
    _assert_pretruth(payload, description="P11 capture")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise EvaluationError("capture cells are absent")
    by_cell = {str(item.get("cell_id")): item for item in cells if isinstance(item, Mapping)}
    if set(by_cell) != set(CELL_ORDER):
        raise EvaluationError("capture cell order/matrix changed")
    observations: dict[str, dict[str, Any]] = {}
    for cell in CELL_ORDER:
        domain = cell.split("__", 1)[0]
        item = by_cell[cell]
        if item.get("records") not in (None, RECORDS_PER_DOMAIN):
            raise EvaluationError(f"capture record count changed: {cell}")
        if item.get("record_ids_sha256") != selection_identity["record_ids_sha256"][domain]:
            raise EvaluationError(f"capture source order changed: {cell}")
        declared = observation_bindings.get(cell)
        if declared is None:
            raise EvaluationError(f"observation binding is absent: {cell}")
        observation = _validate_observation(declared, root=root, cell=cell)
        if observation["capture_record_order"] != selection_identity["record_ids"][domain]:
            raise EvaluationError(f"capture record order differs from frozen selection: {cell}")
        capture_observation = item.get("observation")
        if isinstance(capture_observation, Mapping):
            _same_content(observation, capture_observation, description=f"capture observation {cell}")
        observations[cell] = observation
    return payload, record, observations


def _validate_restore(value: Any, *, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, record = _load_json_binding(value, root=root, description="P11 restore receipt")
    if payload.get("task_id") != TASK_ID or payload.get("status") != RESTORE_STATUS:
        raise EvaluationError("restore receipt is not PASS_RESTORED_SMOKE")
    if payload.get("independent_evaluation_truth_opened") is not False:
        raise EvaluationError("restore receipt evaluation truth boundary is not closed")
    if payload.get("training_worktree_import") is not False:
        raise EvaluationError("restore receipt did not explicitly close the training-worktree boundary")
    if payload.get("source_boundary") != "secondary":
        raise EvaluationError("restore receipt did not record secondary-copy materialization")
    retrieval = payload.get("retrieval")
    if not isinstance(retrieval, Mapping) or retrieval.get("source_boundary") != "secondary":
        raise EvaluationError("restore retrieval did not record secondary-copy materialization")
    restore_manifest = payload.get("restore_manifest")
    if not isinstance(restore_manifest, Mapping):
        raise EvaluationError("restore receipt manifest binding is absent")
    _record(restore_manifest, root=root, description="restore manifest receipt", declared=restore_manifest)
    consumer_receipt = payload.get("consumer_receipt")
    if not isinstance(consumer_receipt, Mapping):
        raise EvaluationError("restore consumer receipt is absent")
    if consumer_receipt.get("training_worktree_import") is not False or consumer_receipt.get("temporary_dependency") is not False:
        raise EvaluationError("restore consumer receipt did not explicitly close the clean-runtime boundary")
    return payload, record


def _restore_copy_record(asset_name: str, asset: Mapping[str, Any], boundary: str) -> dict[str, Any]:
    copies = asset.get("copies")
    if not isinstance(copies, Mapping):
        raise EvaluationError(f"restore asset copies are absent: {asset_name}")
    value = copies.get(boundary)
    if not isinstance(value, Mapping):
        raise EvaluationError(f"restore asset {boundary} copy is absent: {asset_name}")
    try:
        bytes_value = int(value["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError(f"restore asset byte binding is malformed: {asset_name}/{boundary}") from exc
    if bytes_value < 0:
        raise EvaluationError(f"restore asset byte binding is negative: {asset_name}/{boundary}")
    digest = _require_sha(value.get("sha256"), description=f"restore asset {asset_name}/{boundary}")
    return {"bytes": bytes_value, "sha256": digest}


def _validate_restore_links(
    restore: Mapping[str, Any],
    states: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind evaluation states and deployment assets to the actual restore report.

    ``restore_gate.restore_and_run_smoke`` emits the normalized asset, tensor
    identity, and consumer receipt objects consumed here.  Hashes/pointers to
    an older restore run are insufficient: the state files and deployment
    assets in this report must match the files named by the evaluation method
    descriptors and the files observed by the clean consumer.
    """
    assets = restore.get("assets")
    identities = restore.get("tensor_identity")
    consumer = restore.get("consumer_receipt")
    if not isinstance(assets, Mapping):
        raise EvaluationError("restore report assets are absent")
    if not isinstance(identities, Mapping):
        raise EvaluationError("restore report tensor identities are absent")
    if not isinstance(consumer, Mapping):
        raise EvaluationError("restore report consumer receipt is absent")
    normalized_assets: dict[str, Any] = {}
    for name in RESTORE_DEPLOYMENT_ASSETS + ("current_fixed", "expanded_fixed", "selection_receipt", "smoke_input", "smoke_expected"):
        asset = assets.get(name)
        if not isinstance(asset, Mapping):
            raise EvaluationError(f"restore report asset is absent: {name}")
        relative = asset.get("relative_path")
        if not isinstance(relative, str) or not relative or relative.startswith("/"):
            raise EvaluationError(f"restore report asset path is malformed: {name}")
        primary = _restore_copy_record(name, asset, "primary")
        secondary = _restore_copy_record(name, asset, "secondary")
        _same_content(primary, secondary, description=f"restore asset copies {name}")
        normalized_assets[name] = {
            "relative_path": relative,
            "primary": primary,
            "secondary": secondary,
            "metadata": dict(asset.get("metadata", {})) if isinstance(asset.get("metadata"), Mapping) else {},
        }

    state_links: dict[str, Any] = {}
    for method, asset_name in RESTORE_STATE_ASSETS.items():
        state = states.get(method)
        if not isinstance(state, Mapping):
            raise EvaluationError(f"restore state descriptor is absent: {method}")
        asset = normalized_assets[asset_name]
        metadata = asset["metadata"]
        if metadata.get("bank") != RESTORE_STATE_BANKS[method]:
            raise EvaluationError(f"restore state bank differs: {method}")
        if metadata.get("model_id") != state.get("model_id"):
            raise EvaluationError(f"restore state model identity differs: {method}")
        try:
            restore_step = int(metadata["selected_step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationError(f"restore state selected step is absent: {method}") from exc
        if restore_step != state.get("selected_step"):
            raise EvaluationError(f"restore state selected step differs: {method}")
        _same_content(state["file"], asset["primary"], description=f"restore state file {method}")
        identity = identities.get(asset_name)
        if not isinstance(identity, Mapping):
            raise EvaluationError(f"restore tensor identity is absent: {asset_name}")
        identity_sha = identity.get("file_sha256")
        if identity_sha is None and isinstance(identity.get("payload"), Mapping):
            identity_sha = identity["payload"].get("file_sha256")
        if identity_sha != asset["primary"]["sha256"]:
            raise EvaluationError(f"restore tensor identity differs: {asset_name}")
        state_links[method] = {
            "asset_name": asset_name,
            "state_id": str(state["state_id"]),
            "model_id": str(state["model_id"]),
            "selected_step": int(state["selected_step"]),
            "file": dict(state["file"]),
            "restore_asset": dict(asset),
            "tensor_identity_file_sha256": str(identity_sha),
        }

    for asset_name in RESTORE_TENSOR_ASSETS:
        identity = identities.get(asset_name)
        if not isinstance(identity, Mapping):
            raise EvaluationError(f"restore tensor identity is absent: {asset_name}")
        identity_sha = identity.get("file_sha256")
        if identity_sha is None and isinstance(identity.get("payload"), Mapping):
            identity_sha = identity["payload"].get("file_sha256")
        if identity_sha != normalized_assets[asset_name]["primary"]["sha256"]:
            raise EvaluationError(f"restore tensor identity does not match asset: {asset_name}")

    loaded_bindings = consumer.get("loaded_file_bindings")
    if not isinstance(loaded_bindings, list) or not loaded_bindings:
        raise EvaluationError("restore consumer loaded file bindings are absent")
    normalized_loaded: list[dict[str, Any]] = []
    for index, item in enumerate(loaded_bindings):
        if not isinstance(item, Mapping):
            raise EvaluationError(f"restore consumer loaded binding is malformed: {index}")
        digest = _require_sha(item.get("sha256"), description=f"restore consumer loaded binding {index}")
        try:
            bytes_value = int(item["bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationError(f"restore consumer loaded byte binding is malformed: {index}") from exc
        path = item.get("path", item.get("loaded_path"))
        if not isinstance(path, str) or not path:
            raise EvaluationError(f"restore consumer loaded path is absent: {index}")
        normalized_loaded.append({"path": path, "bytes": bytes_value, "sha256": digest, "role": str(item.get("role", ""))})

    def require_loaded_asset(asset_name: str, *, role: str) -> dict[str, Any]:
        expected = normalized_assets[asset_name]["primary"]
        matches = [item for item in normalized_loaded if item["bytes"] == expected["bytes"] and item["sha256"] == expected["sha256"]]
        if not matches:
            raise EvaluationError(f"restore consumer did not load asset: {asset_name}")
        role_matches = [item for item in matches if item.get("role") in {role, f"module:{role}"}]
        return dict(role_matches[0] if role_matches else matches[0])

    loaded_asset_bindings = {
        "current_fixed": require_loaded_asset("current_fixed", role="current_fixed"),
        "expanded_fixed": require_loaded_asset("expanded_fixed", role="expanded_fixed"),
        "public_readout": require_loaded_asset("public_readout", role="public_readout"),
    }
    loaded_code_paths = consumer.get("loaded_code_paths")
    if not isinstance(loaded_code_paths, list) or not loaded_code_paths or any(not isinstance(path, str) or not path for path in loaded_code_paths):
        raise EvaluationError("restore consumer loaded code paths are absent")
    def require_loaded_code(asset_name: str) -> dict[str, Any]:
        expected = normalized_assets[asset_name]["primary"]
        relative = normalized_assets[asset_name]["relative_path"].replace("\\", "/")
        matches = [
            item
            for item in normalized_loaded
            if item["bytes"] == expected["bytes"]
            and item["sha256"] == expected["sha256"]
            and (str(item["path"]).replace("\\", "/").endswith("/" + relative) or str(item["path"]).replace("\\", "/").endswith(relative))
        ]
        if not matches:
            raise EvaluationError(f"restore consumer did not load code asset: {asset_name}")
        return dict(matches[0])

    loaded_code_bindings = {name: require_loaded_code(name) for name in ("loader_code", "decoder_code", "package_cli")}
    return {
        "assets": normalized_assets,
        "states": state_links,
        "tensor_identity": {name: {"file_sha256": normalized_assets[name]["primary"]["sha256"]} for name in RESTORE_TENSOR_ASSETS},
        "consumer": {"loaded_file_bindings": {**loaded_asset_bindings, **loaded_code_bindings}, "loaded_code_paths": list(loaded_code_paths)},
        "restore_manifest": dict(restore["restore_manifest"]) if isinstance(restore.get("restore_manifest"), Mapping) else {},
    }


def _validate_trace_costs(manifest: Mapping[str, Any], *, root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for field in ("trace_files", "cost_files"):
        raw = manifest.get(field)
        if isinstance(raw, Mapping):
            raw_values = list(raw.values())
        elif isinstance(raw, list):
            raw_values = raw
        elif raw is None:
            raw_values = []
        else:
            raise EvaluationError(f"{field} are malformed")
        if not raw_values:
            if field != "trace_files":
                raise EvaluationError(f"{field} are empty")
            unavailable = manifest.get("trace_unavailable")
            if not isinstance(unavailable, Mapping) or unavailable.get("preregistered") is not True or not isinstance(unavailable.get("reason"), str) or not unavailable.get("reason"):
                raise EvaluationError("trace files are empty without an explicit preregistered unavailability record")
            result.append([])
            continue
        checked: list[dict[str, Any]] = []
        for index, value in enumerate(raw_values):
            record = _record(value, root=root, description=f"{field}[{index}]", declared=value if isinstance(value, Mapping) else None)
            if record["path"].lower().endswith(".json"):
                payload, _ = _load_json_binding(record, root=root, description=f"{field}[{index}]")
                _assert_pretruth(payload, description=f"{field}[{index}]")
            checked.append(record)
        result.append(checked)
    return result[0], result[1]


def _validate_support(value: Any, *, root: Path) -> dict[str, Any] | None:
    """Validate the frozen public fitting-support preparation artifacts.

    The support map and histograms are public fitting diagnostics.  This check
    binds their bytes, fixed bins, bank denominators, and truth-free execution
    receipts before a prediction freeze.  It never reads evaluation truth or
    reconstruction outputs.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise EvaluationError("public fitting support binding is malformed")
    required = ("histogram", "common_frequency", "contract", "execution")
    missing = [name for name in required if not isinstance(value.get(name), Mapping)]
    if missing:
        raise EvaluationError(f"public fitting support bindings are absent: {', '.join(missing)}")

    histogram, histogram_record = _load_json_binding(
        value["histogram"], root=root, description="public support histogram"
    )
    if (
        histogram.get("schema") != SUPPORT_HISTOGRAM_SCHEMA
        or histogram.get("task_id") != TASK_ID
        or histogram.get("status") != "COMPLETE_PUBLIC_SUPPORT_HISTOGRAM"
    ):
        raise EvaluationError("public support histogram is not complete")
    truth_boundary = histogram.get("truth_boundary")
    if not isinstance(truth_boundary, Mapping):
        raise EvaluationError("public support histogram truth boundary is absent")
    for key in ("evaluation_truth_opened", "hidden_states_read", "model_opened", "reconstruction_predictions_read", "source_text_read"):
        if truth_boundary.get(key) is not False:
            raise EvaluationError(f"public support histogram crosses the truth/model boundary: {key}")
    counting_audit = histogram.get("counting_audit")
    if not isinstance(counting_audit, Mapping):
        raise EvaluationError("public support histogram counting audit is absent")
    for key in ("exclude_bos", "exclude_padding", "frequency_reference_is_common", "b0_prefix_not_concatenated_to_b1", "b1_full_file_counted_once"):
        if counting_audit.get(key) is not True:
            raise EvaluationError(f"public support counting audit changed: {key}")

    contract, contract_record = _load_json_binding(
        value["contract"], root=root, description="public support contract"
    )
    if contract.get("schema") != SUPPORT_HISTOGRAM_SCHEMA or contract.get("task_id") != TASK_ID:
        raise EvaluationError("public support contract identity changed")
    contract_bins = contract.get("bins")
    if not isinstance(contract_bins, Mapping):
        raise EvaluationError("public support bin contract is absent")
    contract_frequency_bins = contract_bins.get("frequency_bins")
    contract_position_bins = contract_bins.get("position_bins")
    if not isinstance(contract_frequency_bins, list) or not isinstance(contract_position_bins, list):
        raise EvaluationError("public support bin lists are malformed")
    expected_frequency_bins = [
        {"name": name, "lower": lower, "upper": upper}
        for name, lower, upper in SUPPORT_FREQUENCY_BINS
    ]
    expected_position_bins = [
        {"name": name, "lower": lower, "upper": upper, "evaluation_range": evaluation_range}
        for name, lower, upper, evaluation_range in SUPPORT_POSITION_BINS
    ]
    if contract_frequency_bins != expected_frequency_bins or contract_position_bins != expected_position_bins:
        raise EvaluationError("public support bin definitions changed")
    if contract_bins.get("evaluation_width") != STORED_SEQUENCE_TOKENS or contract_bins.get("support_width") != 192:
        raise EvaluationError("public support widths changed")
    if contract_bins.get("common_frequency_map_for_bin_assignment") is not True or contract_bins.get("bank_local_frequency_counts_reported_separately") is not True:
        raise EvaluationError("public support frequency assignment policy changed")

    common_record = _record(
        value["common_frequency"],
        root=root,
        description="public common frequency map",
        declared=value["common_frequency"],
    )
    histogram_config = histogram.get("config")
    if not isinstance(histogram_config, Mapping):
        raise EvaluationError("public support histogram contract binding is absent")
    histogram_config_file = histogram_config.get("file") if isinstance(histogram_config.get("file"), Mapping) else histogram_config
    if _require_sha(histogram_config_file.get("sha256"), description="public support histogram contract") != contract_record["sha256"]:
        raise EvaluationError("public support histogram/contract hash changed")
    if histogram_config_file.get("bytes") is not None and int(histogram_config_file["bytes"]) != contract_record["bytes"]:
        raise EvaluationError("public support histogram/contract byte binding changed")
    histogram_common = histogram.get("common_frequency_reference")
    contract_common = contract.get("common_frequency_reference")
    if not isinstance(histogram_common, Mapping) or not isinstance(contract_common, Mapping):
        raise EvaluationError("public common frequency reference metadata is absent")
    histogram_common_file = histogram_common.get("file")
    if not isinstance(histogram_common_file, Mapping):
        raise EvaluationError("public histogram common frequency file binding is absent")
    _same_content(common_record, histogram_common_file, description="public histogram/common frequency map")
    expected_common_sha = _require_sha(contract_common.get("sha256"), description="public contract common frequency")
    if expected_common_sha != common_record["sha256"] or int(contract_common.get("bytes", -1)) != common_record["bytes"]:
        raise EvaluationError("public contract common frequency map binding changed")
    for key in ("support_count", "positive_occurrences", "support_digest_trr0010", "support_ids_tensor_sha256", "support_counts_tensor_sha256", "frequency_vector_tensor_sha256"):
        if key not in histogram_common or key not in contract_common or histogram_common[key] != contract_common[key]:
            raise EvaluationError(f"public common frequency identity changed: {key}")
    if int(contract_common.get("vocabulary_size", -1)) != VOCABULARY_SIZE:
        raise EvaluationError("public common frequency vocabulary size changed")

    try:
        with safe_open(common_record["path"], framework="pt", device="cpu") as handle:
            expected_keys = {"frequency_counts", "support_ids", "support_counts"}
            if set(handle.keys()) != expected_keys:
                raise EvaluationError("public common frequency tensor keys changed")
            tensors = {key: handle.get_tensor(key).detach().cpu().contiguous() for key in expected_keys}
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError("public common frequency map is unreadable") from exc
    expected_shapes = {
        "frequency_counts": [VOCABULARY_SIZE],
        "support_ids": [int(contract_common["support_count"])],
        "support_counts": [int(contract_common["support_count"])],
    }
    expected_digests = {
        "frequency_counts": str(contract_common["frequency_vector_tensor_sha256"]),
        "support_ids": str(contract_common["support_ids_tensor_sha256"]),
        "support_counts": str(contract_common["support_counts_tensor_sha256"]),
    }
    tensor_records: dict[str, Any] = {}
    for key, tensor in tensors.items():
        if list(tensor.shape) != expected_shapes[key] or str(tensor.dtype) != "torch.int64":
            raise EvaluationError(f"public common frequency tensor geometry changed: {key}")
        digest = p10.tensor_digest(tensor)
        if digest != expected_digests[key]:
            raise EvaluationError(f"public common frequency tensor digest changed: {key}")
        tensor_records[key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": digest}

    execution, execution_record = _load_json_binding(
        value["execution"], root=root, description="public support histogram execution receipt"
    )
    if execution.get("schema") != SUPPORT_EXECUTION_SCHEMA or execution.get("task_id") != TASK_ID or execution.get("status") != "PASS_PUBLIC_SUPPORT_HISTOGRAM":
        raise EvaluationError("public support histogram execution receipt is not a pass")
    for key in ("gpu_used", "model_opened", "truth_opened"):
        if execution.get(key) is not False:
            raise EvaluationError(f"public support execution boundary changed: {key}")
    recovery_execution_record = None
    if isinstance(value.get("frequency_recovery_execution"), Mapping):
        recovery_execution, recovery_execution_record = _load_json_binding(
            value["frequency_recovery_execution"],
            root=root,
            description="public common frequency recovery execution receipt",
        )
        if recovery_execution.get("schema") != SUPPORT_RECOVERY_EXECUTION_SCHEMA or recovery_execution.get("task_id") != TASK_ID or recovery_execution.get("status") != "PASS_PUBLIC_B0_FREQUENCY_DERIVED":
            raise EvaluationError("public common frequency recovery execution is not a pass")
        for key in ("gpu_used", "model_opened", "truth_opened"):
            if recovery_execution.get(key) is not False:
                raise EvaluationError(f"public common frequency recovery boundary changed: {key}")

    histograms = histogram.get("histograms")
    if not isinstance(histograms, Mapping):
        raise EvaluationError("public support histograms are absent")
    expected_bank_geometry = {
        "B0": (1200, 124371, [0, 1200]),
        "B1": (12000, 1243710, [0, 12000]),
        "B1_additions": (10800, 1119339, [1200, 12000]),
    }
    bank_views: dict[str, Any] = {}
    for bank_name, (expected_records, expected_positions, expected_slice) in expected_bank_geometry.items():
        bank = histograms.get(bank_name)
        if not isinstance(bank, Mapping):
            raise EvaluationError(f"public support histogram bank is absent: {bank_name}")
        if bank.get("correctness_status") != "not_computed":
            raise EvaluationError(f"public support histogram correctness was computed: {bank_name}")
        if int(bank.get("records", -1)) != expected_records or int(bank.get("post_bos_positions", -1)) != expected_positions:
            raise EvaluationError(f"public support histogram denominator changed: {bank_name}")
        if bank.get("row_slice_in_source") != expected_slice:
            raise EvaluationError(f"public support histogram row slice changed: {bank_name}")
        common_bins = bank.get("common_reference_frequency", {}).get("bin_occurrences")
        if not isinstance(common_bins, Mapping) or set(common_bins) != {name for name, _, _ in SUPPORT_FREQUENCY_BINS}:
            raise EvaluationError(f"public support histogram frequency bins are incomplete: {bank_name}")
        if sum(int(common_bins[name]) for name, _, _ in SUPPORT_FREQUENCY_BINS) != expected_positions:
            raise EvaluationError(f"public support histogram frequency denominator changed: {bank_name}")
        position_bins = bank.get("position_bins")
        if not isinstance(position_bins, Mapping) or set(position_bins) != {name for name, _, _, _ in SUPPORT_POSITION_BINS}:
            raise EvaluationError(f"public support histogram position bins are incomplete: {bank_name}")
        position_total = 0
        for name, _, _, evaluation_range in SUPPORT_POSITION_BINS:
            item = position_bins[name]
            if not isinstance(item, Mapping) or item.get("in_evaluation_range") is not evaluation_range:
                raise EvaluationError(f"public support histogram position definition changed: {bank_name}/{name}")
            position_total += int(item.get("post_bos_positions", -1))
        if position_total != expected_positions:
            raise EvaluationError(f"public support histogram position denominator changed: {bank_name}")
        bank_views[bank_name] = {
            key: bank[key]
            for key in (
                "label", "records", "post_bos_positions", "row_slice_in_source", "coverage",
                "common_reference_frequency", "position_bins", "source_type_positions",
                "own_bank_frequency", "style_summary", "joint_style_position_frequency",
                "correctness_status",
            )
            if key in bank
        }

    return {
        "status": "AVAILABLE_PUBLIC_FIT_SUPPORT",
        "histogram": histogram_record,
        "common_frequency": common_record,
        "contract": contract_record,
        "execution": execution_record,
        "frequency_recovery_execution": recovery_execution_record,
        "frequency_map": {
            "file": dict(common_record),
            "tensor_keys": sorted(tensor_records),
            "tensors": tensor_records,
            "support_count": int(contract_common["support_count"]),
            "positive_occurrences": int(contract_common["positive_occurrences"]),
            "support_digest_trr0010": str(contract_common["support_digest_trr0010"]),
        },
        "definition": {
            "frequency_bins": expected_frequency_bins,
            "position_bins": expected_position_bins,
            "position_coordinate": "one-based post-BOS position",
            "labels": "token_ids[:, 1:][attention_mask[:, 1:]]",
            "exclude_bos": True,
            "exclude_padding": True,
            "evaluation_width": STORED_SEQUENCE_TOKENS,
            "support_width": 192,
            "assignment_source": "frozen common public B0 frequency map",
            "correctness_is_not_used_for_bin_selection": True,
        },
        "fit_bank_counts": bank_views,
    }


def _validate_state(method: str, value: Any, *, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"state descriptor is absent: {method}")
    state_id = value.get("state_id")
    model_id = value.get("model_id")
    if not isinstance(state_id, str) or not state_id or not isinstance(model_id, str) or not model_id:
        raise EvaluationError(f"state/model identity is absent: {method}")
    selected_step = value.get("selected_step")
    if selected_step is not None and (isinstance(selected_step, bool) or not isinstance(selected_step, int) or selected_step < 0):
        raise EvaluationError(f"selected step is malformed: {method}")
    state_file = value.get("file") or value.get("state_file")
    if state_file is None:
        raise EvaluationError(f"state file binding is absent: {method}")
    record = _record(state_file, root=root, description=f"state {method}", declared=state_file if isinstance(state_file, Mapping) else None)
    return {
        "state_id": state_id,
        "model_id": model_id,
        "selected_step": selected_step,
        "file": record,
    }, record


def _receipt_method_key(binding: Mapping[str, Any], method: str) -> str:
    key = binding.get("receipt_method_key")
    if key is None:
        key = METHOD_OUTPUT_KEYS.get(method)
    if not isinstance(key, str) or not key:
        raise EvaluationError(f"receipt output key is absent: {method}")
    return key


def _validate_prediction(
    method: str,
    cell: str,
    value: Any,
    *,
    root: Path,
    state: Mapping[str, Any],
    observation: Mapping[str, Any],
    restore_links: Mapping[str, Any],
) -> tuple[dict[str, Any], torch.Tensor]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"prediction binding is absent: {method}/{cell}")
    records = _expected_record_count(method)
    declared_records = value.get("records", records)
    if int(declared_records) != records:
        raise EvaluationError(f"prediction record count changed: {method}/{cell}")
    output_value = value.get("file") or value.get("output") or value
    output = _record(output_value, root=root, description=f"prediction {method}/{cell}", declared=output_value if isinstance(output_value, Mapping) else None)
    tensor_key = value.get("tensor_key")
    if not isinstance(tensor_key, str) or not tensor_key:
        raise EvaluationError(f"prediction tensor key is absent: {method}/{cell}")
    expected_key = METHOD_OUTPUT_KEYS.get(method)
    if expected_key is not None and tensor_key != expected_key:
        raise EvaluationError(f"Agent1 output key changed: {method}/{cell}")
    tensor_digest_declared = value.get("tensor_sha256") or value.get("prediction_sha256")
    _require_sha(tensor_digest_declared, description=f"prediction tensor {method}/{cell}")
    try:
        with safe_open(output["path"], framework="pt", device="cpu") as handle:
            available_keys = set(handle.keys())
            if tensor_key not in available_keys:
                raise EvaluationError(f"prediction tensor key is missing: {method}/{cell}/{tensor_key}")
            if method in PRIMARY_METHODS and available_keys != set(METHOD_OUTPUT_KEYS.values()):
                raise EvaluationError(f"primary prediction tensor keys changed: {method}/{cell}")
            tensor = handle.get_tensor(tensor_key).detach().cpu().contiguous()
            metadata = dict(handle.metadata() or {})
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"prediction artifact is unreadable: {method}/{cell}") from exc
    if tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise EvaluationError(f"prediction dtype is not integer: {method}/{cell}")
    if tuple(tensor.shape) != (records, STORED_SEQUENCE_TOKENS):
        raise EvaluationError(f"prediction geometry changed: {method}/{cell}")
    if tensor[:, 0].ne(BOS_TOKEN_ID).any().item() or tensor.lt(0).any().item() or tensor.ge(VOCABULARY_SIZE).any().item():
        raise EvaluationError(f"prediction token range changed: {method}/{cell}")
    if p10.tensor_digest(tensor) != tensor_digest_declared:
        raise EvaluationError(f"prediction tensor digest changed: {method}/{cell}")
    if str(metadata.get("method_id", method)) != method or str(metadata.get("cell_id", cell)) != cell:
        raise EvaluationError(f"prediction metadata identity changed: {method}/{cell}")
    for flag in _FORBIDDEN_TRUE:
        if _is_true(metadata.get(flag)):
            raise EvaluationError(f"prediction metadata crosses truth boundary: {method}/{cell}/{flag}")

    receipt_value = value.get("receipt") or value.get("sibling_receipt")
    if receipt_value is None:
        raise EvaluationError(f"sibling prediction receipt is absent: {method}/{cell}")
    receipt, receipt_record = _load_json_binding(receipt_value, root=root, description=f"prediction receipt {method}/{cell}")
    key = _receipt_method_key(value, method)
    if method == COMPARATOR_METHOD:
        # A1 is an independently frozen native method.  Its receipt contract
        # is bound by the adapter, but it is not forced through Agent1's
        # TRR-0012 schema or output-key names.
        if not isinstance(receipt.get("schema"), str) or not isinstance(receipt.get("task_id"), str) or not isinstance(receipt.get("status"), str):
            raise EvaluationError(f"native A1 receipt identity is absent: {method}/{cell}")
        for flag in ("truth_opened", "source_text_loaded", "token_ids_loaded", "target_labels_loaded"):
            if flag in receipt and receipt.get(flag) is not False:
                raise EvaluationError(f"native A1 receipt crosses truth boundary: {method}/{cell}/{flag}")
    else:
        if receipt.get("schema") != PREDICTION_RECEIPT_SCHEMA or receipt.get("task_id") != "TRR-0012" or receipt.get("status") != "PREDICTIONS_GENERATED_AFTER_SELECTION":
            raise EvaluationError(f"sibling prediction receipt status changed: {method}/{cell}")
        for flag, expected in (("complete_before_smoke", True), ("smoke_used_for_selection", False), ("independent_evaluation_truth_opened", False), ("truth_opened", False), ("source_text_loaded", False), ("token_ids_loaded", False)):
            if receipt.get(flag) is not expected:
                raise EvaluationError(f"prediction receipt boundary changed: {method}/{cell}/{flag}")
    output_receipt = receipt.get("output")
    if isinstance(output_receipt, Mapping):
        _same_content(output, output_receipt, description=f"prediction receipt output {method}/{cell}")
    else:
        raise EvaluationError(f"prediction receipt output binding is absent: {method}/{cell}")
    obs_receipt = receipt.get("observations")
    if not isinstance(obs_receipt, Mapping):
        raise EvaluationError(f"prediction receipt observation binding is absent: {method}/{cell}")
    _same_content(observation, obs_receipt, description=f"prediction receipt observation {method}/{cell}")
    receipt_tensor_sha = obs_receipt.get("consumer_tensor_sha256")
    if receipt_tensor_sha is None:
        receipt_tensor_sha = obs_receipt.get("tensor_sha256")
    if not isinstance(receipt_tensor_sha, Mapping) or dict(receipt_tensor_sha) != dict(observation["normalized_tensor_sha256"]):
        raise EvaluationError(f"prediction receipt consumer observation tensor binding changed: {method}/{cell}")
    receipt_order = obs_receipt.get("record_order")
    if method == COMPARATOR_METHOD and receipt_order == observation["capture_record_order"]:
        receipt_order_sha = _json_digest(receipt_order)
        if receipt_order_sha != observation["capture_record_order_sha256"]:
            raise EvaluationError(f"prediction receipt source observation order digest changed: {method}/{cell}")
    elif receipt_order == observation["record_order"]:
        receipt_order_sha = _json_digest(receipt_order)
        if receipt_order_sha != observation["record_order_sha256"]:
            raise EvaluationError(f"prediction receipt observation order digest changed: {method}/{cell}")
    else:
        raise EvaluationError(f"prediction receipt observation order changed: {method}/{cell}")
    if method != COMPARATOR_METHOD:
        readout = receipt.get("readout")
        if not isinstance(readout, Mapping):
            raise EvaluationError(f"prediction receipt readout binding is absent: {method}/{cell}")
        readout_binding = readout.get("file_binding") if isinstance(readout.get("file_binding"), Mapping) else readout
        if not isinstance(readout_binding, Mapping):
            raise EvaluationError(f"prediction receipt readout file binding is absent: {method}/{cell}")
        _same_content(restore_links["assets"]["public_readout"]["primary"], readout_binding, description=f"prediction receipt readout {method}/{cell}")
        descriptor_sha = _require_sha(receipt.get("package_descriptor_sha256"), description=f"prediction receipt package descriptor {method}/{cell}")
        if descriptor_sha != restore_links["assets"]["package_manifest"]["primary"]["sha256"]:
            raise EvaluationError(f"prediction receipt package descriptor changed: {method}/{cell}")
    receipt_methods = receipt.get("methods")
    method_receipt = receipt_methods.get(key) if isinstance(receipt_methods, Mapping) else None
    if method_receipt is None and method == COMPARATOR_METHOD:
        method_receipt = receipt.get("method") or receipt.get("state")
    if not isinstance(method_receipt, Mapping):
        raise EvaluationError(f"prediction receipt method binding is absent: {method}/{cell}/{key}")
    state_receipt = method_receipt.get("state_file_binding") or method_receipt.get("state")
    if isinstance(state_receipt, Mapping):
        _same_content(state["file"], state_receipt, description=f"prediction receipt state {method}/{cell}")
    else:
        raise EvaluationError(f"prediction receipt state binding is absent: {method}/{cell}")
    for identity_key in ("state_id", "model_id", "selected_step"):
        if identity_key in method_receipt and method_receipt.get(identity_key) != state.get(identity_key):
            raise EvaluationError(f"prediction receipt state identity changed: {method}/{cell}/{identity_key}")
    runtime = receipt.get("runtime") or receipt.get("environment")
    if method != COMPARATOR_METHOD and (not isinstance(runtime, Mapping) or not isinstance(runtime.get("dependencies"), Mapping) or not runtime.get("dependencies")):
        raise EvaluationError(f"prediction receipt runtime dependency binding is absent: {method}/{cell}")
    normalized = {
        "file": output,
        "tensor_key": tensor_key,
        "tensor_sha256": tensor_digest_declared,
        "records": records,
        "receipt": receipt_record,
        "receipt_method_key": key,
        "metadata": {str(k): str(v) for k, v in metadata.items()},
    }
    return normalized, tensor.to(dtype=torch.long)


def _validate_subset(method: str, value: Any) -> tuple[dict[str, Any], tuple[int, ...] | None]:
    if method != COMPARATOR_METHOD:
        return {}, None
    if not isinstance(value, Mapping):
        raise EvaluationError("A1 comparator subset binding is absent")
    raw = value.get("indices")
    if not isinstance(raw, list) or raw != list(range(COMPARATOR_RECORDS_PER_DOMAIN)):
        raise EvaluationError("A1 comparator is not explicitly bound to the first 128 rows")
    digest = _require_sha(value.get("indices_sha256"), description="A1 comparator indices")
    if digest != p10.indices_digest(raw):
        raise EvaluationError("A1 comparator subset index digest changed")
    if int(value.get("records", COMPARATOR_RECORDS_PER_DOMAIN)) != COMPARATOR_RECORDS_PER_DOMAIN:
        raise EvaluationError("A1 comparator subset count changed")
    return {"parent_cell": value.get("parent_cell"), "indices": list(raw), "indices_sha256": digest, "records": COMPARATOR_RECORDS_PER_DOMAIN}, tuple(raw)


@dataclass(frozen=True)
class FrozenEvaluation:
    freeze_path: Path
    freeze_record: dict[str, Any]
    payload: dict[str, Any]
    predictions: dict[str, torch.Tensor]
    prediction_bindings: dict[str, dict[str, Any]]
    states: dict[str, dict[str, Any]]
    records_by_cell: dict[str, int]
    subsets: dict[str, tuple[int, ...] | None]


def _validate_document(document: Mapping[str, Any], *, root: Path, allow_freeze: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str], str]:
    status = document.get("status")
    if document.get("task_id") != TASK_ID:
        raise EvaluationError("evaluation task identity changed")
    allowed_status = {"PREDICTIONS_READY_NO_TRUTH"} if not allow_freeze else {"PREDICTIONS_READY_NO_TRUTH", "P11_PREDICTIONS_FROZEN_BEFORE_TRUTH", "QUALIFIED_PARTIAL_A1_BLOCKER"}
    if status not in allowed_status:
        raise EvaluationError(f"evaluation document is not pretruth: {status!r}")
    _assert_pretruth(document, description="evaluation document")
    selection, selection_record, selection_identity = _validate_selection(document.get("selection"), root=root)
    selection_ids = selection_identity["record_ids_sha256"]
    capture, capture_record, observations = _validate_capture(document.get("capture"), root=root, selection_identity=selection_identity, observation_bindings=document.get("observations", {}))
    restore, restore_record = _validate_restore(document.get("restore"), root=root)
    support = _validate_support(document.get("support"), root=root)
    trace_files, cost_files = _validate_trace_costs(document, root=root)
    source_order = document.get("source_order")
    if not isinstance(source_order, Mapping):
        raise EvaluationError("source order bindings are absent")
    for cell in CELL_ORDER:
        item = source_order.get(cell)
        domain = cell.split("__", 1)[0]
        if not isinstance(item, Mapping) or item.get("record_ids_sha256") != selection_identity["record_ids_sha256"][domain]:
            raise EvaluationError(f"source order binding changed: {cell}")
        order_sha = _require_sha(item.get("record_order_sha256"), description=f"source order {cell}")
        if order_sha != observations[cell]["record_order_sha256"]:
            raise EvaluationError(f"source/observation order binding changed: {cell}")
    methods = document.get("methods")
    if not isinstance(methods, Mapping):
        raise EvaluationError("evaluation method matrix is absent")
    states: dict[str, dict[str, Any]] = {}
    for method in PRIMARY_METHODS:
        state, _ = _validate_state(method, methods.get(method, {}).get("state") if isinstance(methods.get(method), Mapping) else None, root=root)
        states[method] = state
    comparator = document.get("comparator")
    if not isinstance(comparator, Mapping):
        raise EvaluationError("A1 comparator status is absent; it cannot be silently dropped")
    comparator_status = comparator.get("status")
    if comparator_status == "READY":
        state, _ = _validate_state(COMPARATOR_METHOD, comparator.get("state"), root=root)
        states[COMPARATOR_METHOD] = state
    elif comparator_status == "BLOCKED_TECHNICAL":
        if comparator.get("preregistered") is not True or not isinstance(comparator.get("blocker_id"), str) or not comparator.get("blocker_id") or not isinstance(comparator.get("reason"), str) or not comparator.get("reason"):
            raise EvaluationError("A1 technical blocker is not preregistered with a concrete reason")
    else:
        raise EvaluationError("A1 comparator is neither complete nor explicitly blocked")
    restore_links = _validate_restore_links(restore, states)
    prediction_bindings: dict[str, dict[str, Any]] = {}
    predictions: dict[str, torch.Tensor] = {}
    subsets: dict[str, tuple[int, ...] | None] = {}
    for method in PRIMARY_METHODS:
        item = methods.get(method)
        cells = item.get("cells") if isinstance(item, Mapping) else None
        if not isinstance(cells, Mapping):
            raise EvaluationError(f"prediction cells are absent: {method}")
        for cell in CELL_ORDER:
            key = f"{method}::{cell}"
            normalized, tensor = _validate_prediction(method, cell, cells.get(cell), root=root, state=states[method], observation=observations[cell], restore_links=restore_links)
            prediction_bindings[key] = normalized
            predictions[key] = tensor
            subsets[key] = None
    if comparator_status == "READY":
        cells = comparator.get("cells")
        if not isinstance(cells, Mapping):
            raise EvaluationError("A1 comparator cells are absent")
        for cell in CELL_ORDER:
            key = f"{COMPARATOR_METHOD}::{cell}"
            cell_value = cells.get(cell)
            if isinstance(cell_value, Mapping):
                subset_value = cell_value.get("subset") or cell_value.get("source_subset")
            else:
                subset_value = None
            subset, indices = _validate_subset(COMPARATOR_METHOD, subset_value)
            normalized, tensor = _validate_prediction(COMPARATOR_METHOD, cell, cells.get(cell), root=root, state=states[COMPARATOR_METHOD], observation=observations[cell], restore_links=restore_links)
            prediction_bindings[key] = {**normalized, "source_subset": subset}
            predictions[key] = tensor
            subsets[key] = indices
    matrix_status = "P11_PREDICTIONS_FROZEN_BEFORE_TRUTH" if comparator_status == "READY" else "QUALIFIED_PARTIAL_A1_BLOCKER"
    if allow_freeze:
        declared_restore_links = document.get("restore_asset_bindings")
        if declared_restore_links != restore_links:
            raise EvaluationError("frozen restore asset bindings changed")
        if document.get("selection_h128_sequence_sha256") != selection_identity["h128_sequence_sha256"]:
            raise EvaluationError("frozen selection H128 bindings changed")
    return (
        dict(selection_record),
        dict(capture_record),
        dict(restore_record),
        {"payload": document.get("source_order"), "observations": observations, "selection_ids": selection_ids, "selection_identity": selection_identity, "restore_links": restore_links, "support": support},
        prediction_bindings,
        states,
        trace_files,
        cost_files,
        selection_ids,
        matrix_status,
    )


def freeze_predictions(*, manifest_path: Path, output_path: Path, repository_root: Path) -> dict[str, Any]:
    """Validate and freeze all prediction files before any truth is opened."""
    root = Path(repository_root).expanduser().resolve()
    manifest, manifest_record = _load_json_binding(manifest_path, root=root, description="P11 evaluation manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise EvaluationError("evaluation manifest schema changed")
    (
        selection_record,
        capture_record,
        restore_record,
        source_context,
        prediction_bindings,
        states,
        trace_files,
        cost_files,
        selection_ids,
        matrix_status,
    ) = _validate_document(manifest, root=root)
    methods: dict[str, Any] = {}
    for method in PRIMARY_METHODS:
        source = manifest["methods"][method]
        methods[method] = {"state": states[method], "cells": {cell: prediction_bindings[f"{method}::{cell}"] for cell in CELL_ORDER}}
    comparator = manifest["comparator"]
    comparator_payload: dict[str, Any]
    if comparator.get("status") == "READY":
        comparator_payload = {"status": "READY", "state": states[COMPARATOR_METHOD], "cells": {cell: prediction_bindings[f"{COMPARATOR_METHOD}::{cell}"] for cell in CELL_ORDER}}
    else:
        comparator_payload = {key: comparator[key] for key in ("status", "preregistered", "blocker_id", "reason")}
    payload = {
        "schema": FREEZE_SCHEMA,
        "task_id": TASK_ID,
        "status": matrix_status,
        "manifest": manifest_record,
        "selection": selection_record,
        "capture": capture_record,
        "restore": restore_record,
        "observations": source_context["observations"],
        "source_order": source_context["payload"],
        "restore_asset_bindings": source_context["restore_links"],
        "selection_record_ids_sha256": selection_ids,
        "selection_h128_sequence_sha256": source_context["selection_identity"]["h128_sequence_sha256"],
        "selection_h128_sequence_list_sha256": source_context["selection_identity"]["h128_sequence_list_sha256"],
        "methods": methods,
        "comparator": comparator_payload,
        "trace_files": trace_files,
        "cost_files": cost_files,
        "trace_unavailable": manifest.get("trace_unavailable") if not trace_files else None,
        "method_order": list(METHOD_ORDER),
        "cell_order": list(CELL_ORDER),
        "records_by_cell": {cell: RECORDS_PER_DOMAIN for cell in CELL_ORDER},
        "scoring_contract": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "one_sided_alpha": BOOTSTRAP_ALPHA,
            "shared_schedule_within_domain": True,
            "pooling": False,
        },
        "truth_opened": False,
        "source_text_or_target_labels": False,
        "candidate_arrays_persisted": False,
        "p03_holdout_accessed": False,
        "support": source_context["support"],
    }
    record = _write_json(Path(output_path), payload, root=root, description="P11 evaluation freeze")
    return {"task_id": TASK_ID, "status": matrix_status, "freeze": record, "truth_opened": False}


def _load_frozen(path: Path, *, root: Path) -> FrozenEvaluation:
    payload, record = _load_json_binding(path, root=root, description="P11 evaluation freeze")
    if payload.get("schema") != FREEZE_SCHEMA:
        raise EvaluationError("P11 freeze schema changed")
    (
        _selection_record,
        _capture_record,
        _restore_record,
        source_context,
        prediction_bindings,
        states,
        _trace_files,
        _cost_files,
        _selection_ids,
        matrix_status,
    ) = _validate_document(payload, root=root, allow_freeze=True)
    if payload.get("status") != matrix_status:
        raise EvaluationError("P11 freeze matrix status changed")
    predictions: dict[str, torch.Tensor] = {}
    subsets: dict[str, tuple[int, ...] | None] = {}
    for key, binding in prediction_bindings.items():
        method, cell = key.split("::", 1)
        predictions[key] = _load_prediction_again(method, cell, binding, root=root)
        raw_subset = binding.get("source_subset")
        subsets[key] = tuple(raw_subset["indices"]) if isinstance(raw_subset, Mapping) else None
    return FrozenEvaluation(
        freeze_path=Path(record["path"]),
        freeze_record=record,
        payload=payload,
        predictions=predictions,
        prediction_bindings=prediction_bindings,
        states=states,
        records_by_cell={cell: RECORDS_PER_DOMAIN for cell in CELL_ORDER},
        subsets=subsets,
    )


def _load_prediction_again(method: str, cell: str, binding: Mapping[str, Any], *, root: Path) -> torch.Tensor:
    output = _record(binding["file"], root=root, description=f"frozen prediction {method}/{cell}", declared=binding["file"])
    try:
        with safe_open(output["path"], framework="pt", device="cpu") as handle:
            key = str(binding["tensor_key"])
            if key not in set(handle.keys()):
                raise EvaluationError(f"frozen prediction key disappeared: {method}/{cell}")
            tensor = handle.get_tensor(key).detach().cpu().contiguous()
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"frozen prediction is unreadable: {method}/{cell}") from exc
    if p10.tensor_digest(tensor) != binding["tensor_sha256"]:
        raise EvaluationError(f"frozen prediction tensor hash changed: {method}/{cell}")
    return tensor.to(dtype=torch.long)


def _load_scorer(root: Path) -> Any:
    path = root / SCORER_RELATIVE_PATH
    actual = _sha256_file(path)
    if actual != SCORER_SHA256:
        raise EvaluationError("registered P11 scorer bytes changed")
    spec = importlib.util.spec_from_file_location("trr_p11_registered_scorer", path)
    if spec is None or spec.loader is None:
        raise EvaluationError("registered P11 scorer cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _h128_sequence_digest(values: Sequence[int]) -> str:
    """Use the P10 audit's raw little-endian signed-int32 H128 convention."""
    if len(values) != STORED_SEQUENCE_TOKENS:
        raise EvaluationError("H128 digest requires exactly 128 token IDs")
    normalized = [int(value) for value in values]
    if any(value < -(2**31) or value >= 2**31 for value in normalized):
        raise EvaluationError("H128 token ID is outside signed int32")
    return hashlib.sha256(struct.pack("<" + "i" * STORED_SEQUENCE_TOKENS, *normalized)).hexdigest()


def load_truth_after_freeze(*, freeze_path: Path, truth_descriptor_path: Path, repository_root: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Revalidate every pretruth binding, then open one bound truth sidecar."""
    root = Path(repository_root).expanduser().resolve()
    frozen = _load_frozen(Path(freeze_path), root=root)
    if frozen.payload.get("status") not in {"P11_PREDICTIONS_FROZEN_BEFORE_TRUTH", "QUALIFIED_PARTIAL_A1_BLOCKER"}:
        raise EvaluationError("truth opening requires a validated pretruth freeze")
    truth, truth_record = _load_json_binding(truth_descriptor_path, root=root, description="P11 truth descriptor")
    if truth.get("schema") != TRUTH_SCHEMA or truth.get("task_id") != TASK_ID or truth.get("status") != "TRUTH_PREPARED_AFTER_PREDICTION_FREEZE":
        raise EvaluationError("truth descriptor is not post-freeze")
    if truth.get("predictions_frozen_before_truth") is not True or truth.get("p03_holdout_accessed") is not False:
        raise EvaluationError("truth descriptor boundary changed")
    freeze_ref = truth.get("freeze")
    if not isinstance(freeze_ref, Mapping):
        raise EvaluationError("truth descriptor freeze binding is absent")
    _same_content(frozen.freeze_record, freeze_ref, description="truth descriptor/freeze")
    sidecar_ref = truth.get("sidecar")
    if not isinstance(sidecar_ref, Mapping):
        raise EvaluationError("truth descriptor sidecar binding is absent")
    sidecar = _record(sidecar_ref, root=root, description="P11 truth sidecar", declared=sidecar_ref)
    try:
        with safe_open(sidecar["path"], framework="pt", device="cpu") as handle:
            expected_keys = {f"{cell}__token_ids" for cell in CELL_ORDER}
            if set(handle.keys()) != expected_keys:
                raise EvaluationError("truth sidecar cell keys changed")
            values = {cell: handle.get_tensor(f"{cell}__token_ids").detach().cpu().contiguous() for cell in CELL_ORDER}
            metadata = dict(handle.metadata() or {})
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError("truth sidecar is unreadable") from exc
    for cell, tensor in values.items():
        if tuple(tensor.shape) != (RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS) or tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise EvaluationError(f"truth geometry/dtype changed: {cell}")
        if tensor[:, 0].ne(BOS_TOKEN_ID).any().item() or tensor.lt(0).any().item() or tensor.ge(VOCABULARY_SIZE).any().item():
            raise EvaluationError(f"truth token range changed: {cell}")
    expected_h128 = frozen.payload.get("selection_h128_sequence_sha256")
    if not isinstance(expected_h128, Mapping) or set(expected_h128) != set(DOMAIN_ORDER):
        raise EvaluationError("frozen selection H128 rows are absent")
    actual_h128: dict[str, list[str]] = {}
    for domain in DOMAIN_ORDER:
        expected_rows = expected_h128.get(domain)
        if not isinstance(expected_rows, list) or len(expected_rows) != RECORDS_PER_DOMAIN:
            raise EvaluationError(f"frozen selection H128 rows are incomplete: {domain}")
        base = values[f"{domain}__public_base"]
        lora = values[f"{domain}__public_lora_2601"]
        if not torch.equal(base, lora):
            raise EvaluationError(f"truth paired public conditions differ: {domain}")
        actual_rows = [_h128_sequence_digest(row.tolist()) for row in base]
        if actual_rows != expected_rows:
            for index, (actual, expected) in enumerate(zip(actual_rows, expected_rows)):
                if actual != expected:
                    raise EvaluationError(f"truth H128 identity differs: {domain}/{index}")
            raise EvaluationError(f"truth H128 identity differs: {domain}")
        actual_h128[domain] = actual_rows
    if metadata.get("freeze_sha256") != frozen.freeze_record["sha256"] or metadata.get("truth_opened") != "true":
        raise EvaluationError("truth sidecar freeze binding changed")
    return values, {
        "descriptor": truth_record,
        "sidecar": sidecar,
        "metadata": {str(k): str(v) for k, v in metadata.items()},
        "truth_opened": True,
        "matrix_status": frozen.payload.get("status"),
        "a1_comparator_available": frozen.payload.get("status") == "P11_PREDICTIONS_FROZEN_BEFORE_TRUTH",
        "selection_h128_sequence_sha256": actual_h128,
        "paired_conditions": {domain: {"public_base_equals_public_lora_2601": True} for domain in DOMAIN_ORDER},
    }


def _as_p10_frozen(frozen: FrozenEvaluation) -> p10.FrozenPackage:
    method_map = {
        "expanded_fixed": "new_expanded_fixed_B1",
        "current_fixed": "new_current_fixed_B0",
        "frozen_a1_a2_k256": COMPARATOR_METHOD,
    }
    predictions: dict[str, torch.Tensor] = {}
    prediction_records: dict[str, dict[str, Any]] = {}
    prediction_metadata: dict[str, dict[str, str]] = {}
    records_by_method_cell: dict[str, int] = {}
    subsets: dict[str, tuple[int, ...] | None] = {}
    active_method_map = dict(method_map)
    if not any(f"{COMPARATOR_METHOD}::{cell}" in frozen.predictions for cell in CELL_ORDER):
        active_method_map.pop("frozen_a1_a2_k256", None)
    for p10_method, p11_method in active_method_map.items():
        for cell in CELL_ORDER:
            source_key = f"{p11_method}::{cell}"
            target_key = f"{p10_method}::{cell}"
            predictions[target_key] = frozen.predictions[source_key]
            prediction_records[target_key] = dict(frozen.prediction_bindings[source_key]["file"])
            prediction_metadata[target_key] = {"method_id": p10_method, "cell_id": cell, "records": str(predictions[target_key].shape[0])}
            records_by_method_cell[target_key] = int(predictions[target_key].shape[0])
            subsets[target_key] = frozen.subsets[source_key]
    return p10.FrozenPackage(
        freeze_path=frozen.freeze_path,
        freeze_record=frozen.freeze_record,
        freeze_payload=frozen.payload,
        registration_path=frozen.freeze_path,
        registration_record=frozen.freeze_record,
        registration_payload=frozen.payload,
        run_manifest_record=None,
        trace_records={str(index): dict(item) for index, item in enumerate(frozen.payload.get("trace_files", []))},
        cells=tuple(p10.CELL_ORDER),
        methods=tuple(active_method_map),
        records_by_cell=dict(frozen.records_by_cell),
        records_by_method_cell=records_by_method_cell,
        subset_indices_by_method_cell=subsets,
        predictions=predictions,
        prediction_records=prediction_records,
        prediction_metadata=prediction_metadata,
    )


def _build_primary_inventory(
    frozen: FrozenEvaluation,
    truth: Mapping[str, torch.Tensor],
    *,
    truth_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact P10 pair categories for a qualified primary-only matrix."""
    cells: dict[str, Any] = {}
    for cell in CELL_ORDER:
        target = torch.as_tensor(truth[cell]).detach().cpu().contiguous()
        if tuple(target.shape) != (RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS):
            raise EvaluationError(f"truth geometry differs from frozen full panel: {cell}")
        per_method: dict[str, torch.Tensor] = {}
        method_summary: dict[str, Any] = {}
        for method, label in (("new_expanded_fixed_B1", "expanded_fixed"), ("new_current_fixed_B0", "current_fixed")):
            key = f"{method}::{cell}"
            correct = p10._correctness(frozen.predictions[key], target)
            per_method[label] = correct
            method_summary[label] = {
                "records": int(correct.shape[0]),
                "scored_post_bos_tokens": int(correct.numel()),
                "correct_tokens": int(correct.sum().item()),
                "token_errors": int((~correct).sum().item()),
                "exact_correct_records": int(correct.all(dim=1).sum().item()),
                "exact_error_records": int((~correct.all(dim=1)).sum().item()),
                "alignment": {"mode": "full_panel", "records": int(correct.shape[0]), "parent_records": RECORDS_PER_DOMAIN},
            }
        left = per_method["expanded_fixed"]
        right = per_method["current_fixed"]
        cells[cell] = {
            "records": RECORDS_PER_DOMAIN,
            "method_summary": method_summary,
            "pairwise": {
                "expanded_fixed_vs_current_fixed": {
                    "left_method": "expanded_fixed",
                    "right_method": "current_fixed",
                    "alignment": {"mode": "full_panel", "records": RECORDS_PER_DOMAIN, "parent_records": RECORDS_PER_DOMAIN},
                    "tokens": p10._pair_counts(left, right),
                    "records_exact": p10._record_pair_counts(left.all(dim=1), right.all(dim=1)),
                }
            },
            "rank_and_proposal_diagnostics": {
                "status": "UNAVAILABLE",
                "reason": "qualified_partial_A1_blocker",
            },
        }
    return {
        "schema": "token-reconstruction.trr-p10-paired-error-inventory.v1",
        "task_id": TASK_ID,
        "status": "SCORED_AFTER_PUBLIC_FREEZE_PRIMARY_ONLY",
        "freeze": frozen.freeze_record,
        "registration": frozen.freeze_record,
        "truth_binding": dict(truth_binding),
        "method_order": ["expanded_fixed", "current_fixed"],
        "cell_order": list(CELL_ORDER),
        "records_by_cell": {cell: RECORDS_PER_DOMAIN for cell in CELL_ORDER},
        "records_by_method_cell": {
            f"{method}::{cell}": RECORDS_PER_DOMAIN
            for method in ("expanded_fixed", "current_fixed")
            for cell in CELL_ORDER
        },
        "subset_bindings": {},
        "cells": cells,
        "decomposition_policy": {
            "a1_proposal_failures": "UNAVAILABLE because the preregistered A1 comparator blocker remains active",
            "decoder_ranking_failures": "UNAVAILABLE because the preregistered A1 comparator blocker remains active",
            "final_prediction_mismatch_is_not_a_candidate_rank": True,
        },
    }


def _parse_cost_files(cost_files: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Expose hash-bound cost receipts without imposing a new producer schema."""
    parsed: list[dict[str, Any]] = []
    for record in cost_files:
        path = Path(str(record["path"]))
        item: dict[str, Any] = {"file": dict(record)}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            item.update({"status": "UNPARSED", "reason": f"JSON load failed: {type(exc).__name__}"})
            parsed.append(item)
            continue
        if not isinstance(payload, Mapping):
            item.update({"status": "UNPARSED", "reason": "cost receipt is not a JSON object"})
            parsed.append(item)
            continue
        numeric: dict[str, float] = {}
        def collect(value: Any, prefix: str) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    collect(child, f"{prefix}.{key}" if prefix else str(key))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric[prefix] = float(value)
        collect(payload, "")
        item.update({
            "status": "PARSED_JSON",
            "schema": payload.get("schema"),
            "task_id": payload.get("task_id"),
            "method_id": payload.get("method_id"),
            "cell_id": payload.get("cell_id"),
            "numeric_fields": numeric,
        })
        parsed.append(item)
    return {"status": "AVAILABLE" if parsed else "UNAVAILABLE", "files": parsed}


def _build_support_strata(
    frozen: FrozenEvaluation,
    truth: Mapping[str, torch.Tensor],
    *,
    support: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Assign post-freeze target positions to the frozen public support bins.

    The common frequency map and bank histograms are fixed before evaluation.
    Target rows are used only after the prediction freeze to report correctness
    by the preregistered support/position strata; they never choose records,
    checkpoints, bins, or methods.
    """
    if not isinstance(support, Mapping) or support.get("status") != "AVAILABLE_PUBLIC_FIT_SUPPORT":
        raise EvaluationError("public fitting support is not validated")
    frequency_map = support.get("frequency_map")
    if not isinstance(frequency_map, Mapping) or not isinstance(frequency_map.get("file"), Mapping):
        raise EvaluationError("public common frequency map binding is absent")
    frequency_file = _record(
        frequency_map["file"],
        root=root,
        description="scoring common frequency map",
        declared=frequency_map["file"],
    )
    expected_tensor = frequency_map.get("tensors", {}).get("frequency_counts") if isinstance(frequency_map.get("tensors"), Mapping) else None
    if not isinstance(expected_tensor, Mapping):
        raise EvaluationError("public common frequency vector binding is absent")
    try:
        with safe_open(frequency_file["path"], framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"frequency_counts", "support_ids", "support_counts"}:
                raise EvaluationError("public common frequency tensor keys changed before scoring")
            frequency_counts = handle.get_tensor("frequency_counts").detach().cpu().contiguous()
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError("public common frequency vector cannot be reopened") from exc
    if list(frequency_counts.shape) != [VOCABULARY_SIZE] or str(frequency_counts.dtype) != "torch.int64":
        raise EvaluationError("public common frequency vector geometry changed before scoring")
    if p10.tensor_digest(frequency_counts) != expected_tensor.get("sha256"):
        raise EvaluationError("public common frequency vector changed before scoring")

    definition = support.get("definition")
    if not isinstance(definition, Mapping):
        raise EvaluationError("public support definition is absent")
    frequency_bins = tuple(SUPPORT_FREQUENCY_BINS)
    position_bins = tuple(SUPPORT_POSITION_BINS)
    cells: dict[str, Any] = {}
    for cell in CELL_ORDER:
        observation = frozen.payload.get("observations", {}).get(cell) if isinstance(frozen.payload.get("observations"), Mapping) else None
        if not isinstance(observation, Mapping):
            raise EvaluationError(f"support observation binding is absent: {cell}")
        observation_record = _record(
            observation,
            root=root,
            description=f"support observation {cell}",
            declared=observation,
        )
        try:
            with safe_open(observation_record["path"], framework="pt", device="cpu") as handle:
                if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                    raise EvaluationError(f"support observation tensor keys changed: {cell}")
                attention_mask = handle.get_tensor("attention_mask").detach().cpu().contiguous()
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(f"support observation cannot be reopened: {cell}") from exc
        if tuple(attention_mask.shape) != (RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS):
            raise EvaluationError(f"support observation mask geometry changed: {cell}")
        active = attention_mask.to(torch.bool)[:, 1:STORED_SEQUENCE_TOKENS]
        target = torch.as_tensor(truth[cell]).detach().cpu().contiguous()
        if tuple(target.shape) != (RECORDS_PER_DOMAIN, STORED_SEQUENCE_TOKENS):
            raise EvaluationError(f"support truth geometry changed: {cell}")
        target_evaluation = target[:, 1:STORED_SEQUENCE_TOKENS]
        if target_evaluation.numel() and (target_evaluation.lt(0).any().item() or target_evaluation.ge(VOCABULARY_SIZE).any().item()):
            raise EvaluationError(f"support truth token range changed: {cell}")
        frequencies = frequency_counts[target_evaluation.to(dtype=torch.long)]
        position_numbers = torch.arange(1, SCORED_POST_BOS_TOKENS + 1, dtype=torch.int64)
        position_bin_indices = torch.full((SCORED_POST_BOS_TOKENS,), -1, dtype=torch.int64)
        for index, (_name, lower, upper, _evaluation_range) in enumerate(position_bins):
            if lower <= SCORED_POST_BOS_TOKENS:
                end = min(int(upper), SCORED_POST_BOS_TOKENS) if upper is not None else SCORED_POST_BOS_TOKENS
                if end >= lower:
                    position_bin_indices[(position_numbers >= lower) & (position_numbers <= end)] = index
        frequency_bin_indices = torch.full_like(frequencies, -1)
        for index, (_name, lower, upper) in enumerate(frequency_bins):
            condition = frequencies.ge(lower)
            if upper is not None:
                condition &= frequencies.le(upper)
            frequency_bin_indices[condition] = index
        if (frequency_bin_indices.lt(0) & active).any().item():
            raise EvaluationError(f"support frequency bins do not cover active target positions: {cell}")
        method_tensors: dict[str, torch.Tensor] = {
            method: frozen.predictions[f"{method}::{cell}"][:, 1:STORED_SEQUENCE_TOKENS]
            for method in PRIMARY_METHODS
        }
        method_records: dict[str, int] = {method: RECORDS_PER_DOMAIN for method in PRIMARY_METHODS}
        if f"{COMPARATOR_METHOD}::{cell}" in frozen.predictions:
            method_tensors[COMPARATOR_METHOD] = frozen.predictions[f"{COMPARATOR_METHOD}::{cell}"][:, 1:STORED_SEQUENCE_TOKENS]
            method_records[COMPARATOR_METHOD] = COMPARATOR_RECORDS_PER_DOMAIN
        method_totals: dict[str, int] = {}
        method_correct: dict[str, int] = {}
        for method, prediction in method_tensors.items():
            rows = method_records[method]
            method_active = active[:rows]
            method_target = target_evaluation[:rows]
            if tuple(prediction.shape) != tuple(method_target.shape):
                raise EvaluationError(f"support prediction geometry changed: {method}/{cell}")
            method_totals[method] = int(method_active.sum().item())
            method_correct[method] = int((prediction.eq(method_target) & method_active).sum().item())

        strata: dict[str, dict[str, Any]] = {}
        for frequency_index, (frequency_name, _lower, _upper) in enumerate(frequency_bins):
            by_position: dict[str, Any] = {}
            frequency_mask = frequency_bin_indices.eq(frequency_index)
            for position_index, (position_name, _plower, _pupper, _evaluation_range) in enumerate(position_bins):
                position_mask = position_bin_indices.eq(position_index).unsqueeze(0)
                primary_mask = active & frequency_mask & position_mask
                denominator = int(primary_mask.sum().item())
                methods_report: dict[str, Any] = {}
                for method, prediction in method_tensors.items():
                    rows = method_records[method]
                    mask = primary_mask[:rows]
                    target_rows = target_evaluation[:rows]
                    correct = int((prediction.eq(target_rows) & mask).sum().item())
                    methods_report[method] = {
                        "records": rows,
                        "correct_tokens": correct,
                        "total_tokens": int(mask.sum().item()),
                    }
                by_position[position_name] = {
                    "denominator_tokens": denominator,
                    "methods": methods_report,
                }
            strata[frequency_name] = by_position
        denominator_by_frequency = {
            name: int((active & frequency_bin_indices.eq(index).to(active.device)).sum().item())
            for index, (name, _lower, _upper) in enumerate(frequency_bins)
        }
        denominator_by_position = {
            name: int((active & position_bin_indices.eq(index).unsqueeze(0)).sum().item())
            for index, (name, _lower, _upper, _evaluation_range) in enumerate(position_bins)
        }
        cells[cell] = {
            "records": RECORDS_PER_DOMAIN,
            "scored_post_bos_positions": SCORED_POST_BOS_TOKENS,
            "denominator_tokens": int(active.sum().item()),
            "denominator_by_frequency": denominator_by_frequency,
            "denominator_by_position": denominator_by_position,
            "methods": {
                method: {
                    "records": method_records[method],
                    "correct_tokens": method_correct[method],
                    "total_tokens": method_totals[method],
                }
                for method in method_tensors
            },
            "frequency_position_strata": strata,
            "observation": observation_record,
        }
    return {
        "status": "AVAILABLE_PUBLIC_FIT_SUPPORT",
        "assignment": {
            "frequency_reference": dict(frequency_map["file"]),
            "source": "frozen common public B0 frequency map plus post-freeze target rows",
            "selection_used": False,
            "checkpoint_selection_used": False,
            "truth_opened_before_prediction_freeze": False,
            "exclude_bos": True,
            "exclude_padding": True,
            "position_coordinate": "one-based post-BOS position",
        },
        "definition": dict(definition),
        "reference_artifacts": {
            "histogram": dict(support["histogram"]),
            "common_frequency": dict(support["common_frequency"]),
            "contract": dict(support["contract"]),
            "execution": dict(support["execution"]),
            "frequency_recovery_execution": dict(support["frequency_recovery_execution"])
            if isinstance(support.get("frequency_recovery_execution"), Mapping)
            else None,
        },
        "frequency_map": dict(frequency_map),
        "fit_bank_counts": dict(support.get("fit_bank_counts", {})),
        "cells": cells,
        "method_order": list(PRIMARY_METHODS) + ([COMPARATOR_METHOD] if any(f"{COMPARATOR_METHOD}::{cell}" in frozen.predictions for cell in CELL_ORDER) else []),
    }


def _support_strata_unavailable() -> dict[str, Any]:
    return {
        "status": "PENDING_FROZEN_FIT_SUPPORT_BINDING",
        "reason": "P11 evaluation manifest does not yet bind the common frozen P07 support-frequency reference and native B0/B1 fit support counts; support is not derived from post-truth target rows",
        "definition": {
            "frequency_bins": [
                {"name": "unseen_0", "lower": 0, "upper": 0},
                {"name": "seen_1", "lower": 1, "upper": 1},
                {"name": "seen_2_4", "lower": 2, "upper": 4},
                {"name": "seen_5_16", "lower": 5, "upper": 16},
                {"name": "seen_17_64", "lower": 17, "upper": 64},
                {"name": "seen_65_plus", "lower": 65, "upper": None},
            ],
            "position_bins_one_based": [
                {"name": "1-15", "lower": 1, "upper": 15},
                {"name": "16-39", "lower": 16, "upper": 39},
                {"name": "40-79", "lower": 40, "upper": 79},
                {"name": "80-127", "lower": 80, "upper": 127},
                {"name": "128-191", "lower": 128, "upper": 191},
            ],
            "common_reference_required": True,
            "required_pretruth_bindings": [
                "common_public_fit_frequency_reference",
                "B0_native_fit_position_attention_mask_counts",
                "B1_native_fit_position_attention_mask_counts",
            ],
            "p11_stored_width_note": "The 128-191 historical bin is retained for contract comparability; a 128-token P11 panel has no active positions in that bin.",
            "correctness_scoring": False,
        },
    }


def score_after_truth(*, freeze_path: Path, truth_descriptor_path: Path, output_path: Path, repository_root: Path) -> dict[str, Any]:
    """Score primary methods, and A1 paired comparisons when available."""
    root = Path(repository_root).expanduser().resolve()
    frozen = _load_frozen(Path(freeze_path), root=root)
    if frozen.payload.get("status") not in {"P11_PREDICTIONS_FROZEN_BEFORE_TRUTH", "QUALIFIED_PARTIAL_A1_BLOCKER"}:
        raise EvaluationError("scientific score requires a validated pretruth freeze")
    truth, truth_binding = load_truth_after_freeze(
        freeze_path=freeze_path,
        truth_descriptor_path=truth_descriptor_path,
        repository_root=root,
    )
    scorer = _load_scorer(root)
    comparator_available = frozen.payload.get("status") == "P11_PREDICTIONS_FROZEN_BEFORE_TRUTH"
    if comparator_available:
        p10_frozen = _as_p10_frozen(frozen)
        inventory = p10.build_error_inventory(p10_frozen, truth, truth_binding=truth_binding)
    else:
        inventory = _build_primary_inventory(frozen, truth, truth_binding=truth_binding)
    cells: dict[str, Any] = {}
    for cell in CELL_ORDER:
        target = truth[cell]
        b0 = frozen.predictions[f"new_current_fixed_B0::{cell}"]
        b1 = frozen.predictions[f"new_expanded_fixed_B1::{cell}"]
        b0_correct = b0[:, 1:].eq(target[:, 1:])
        b1_correct = b1[:, 1:].eq(target[:, 1:])
        exact = scorer.paired_exact_cp(
            b1_correct.all(dim=1).tolist(),
            b0_correct.all(dim=1).tolist(),
            alpha=BOOTSTRAP_ALPHA,
        )
        token_deltas = [
            (int(left.sum().item()) - int(right.sum().item())) / float(SCORED_POST_BOS_TOKENS)
            for left, right in zip(b1_correct, b0_correct)
        ]
        token = scorer.bootstrap_token_delta(
            token_deltas,
            seed=BOOTSTRAP_SEED,
            draws=BOOTSTRAP_DRAWS,
            one_sided_alpha=BOOTSTRAP_ALPHA,
        )
        cell_payload: dict[str, Any] = {
            "new_B1_minus_new_B0": {"exact_record": exact, "token_delta": token},
        }
        primary_position_strata = inventory["cells"][cell]["pairwise"]["expanded_fixed_vs_current_fixed"]["tokens"]["by_position"]
        cell_payload["new_B1_minus_new_B0_position_strata"] = primary_position_strata
        if comparator_available:
            a1 = frozen.predictions[f"{COMPARATOR_METHOD}::{cell}"]
            indices = frozen.subsets[f"{COMPARATOR_METHOD}::{cell}"]
            if indices != tuple(range(COMPARATOR_RECORDS_PER_DOMAIN)):
                raise EvaluationError(f"A1 comparator subset changed before scoring: {cell}")
            selector = torch.tensor(indices, dtype=torch.long)
            target_subset = target.index_select(0, selector)
            b1_subset = b1.index_select(0, selector)
            b1_a1 = b1_subset[:, 1:].eq(target_subset[:, 1:])
            a1_correct = a1[:, 1:].eq(target_subset[:, 1:])
            a1_exact = scorer.paired_exact_cp(
                b1_a1.all(dim=1).tolist(),
                a1_correct.all(dim=1).tolist(),
                alpha=BOOTSTRAP_ALPHA,
            )
            a1_token_deltas = [
                (int(left.sum().item()) - int(right.sum().item())) / float(SCORED_POST_BOS_TOKENS)
                for left, right in zip(b1_a1, a1_correct)
            ]
            a1_token = scorer.bootstrap_token_delta(
                a1_token_deltas,
                seed=BOOTSTRAP_SEED,
                draws=BOOTSTRAP_DRAWS,
                one_sided_alpha=BOOTSTRAP_ALPHA,
            )
            cell_payload["new_B1_minus_a1_a2_first128"] = {
                "exact_record": a1_exact,
                "token_delta": a1_token,
                "subset": {"records": COMPARATOR_RECORDS_PER_DOMAIN, "indices": list(indices), "indices_sha256": p10.indices_digest(list(indices))},
            }
            cell_payload["new_B1_minus_a1_a2_position_strata"] = inventory["cells"][cell]["pairwise"]["expanded_fixed_vs_a1_a2"]["tokens"]["by_position"]
            cell_payload["position_strata"] = cell_payload["new_B1_minus_a1_a2_position_strata"]
        else:
            cell_payload["position_strata"] = primary_position_strata
        cells[cell] = cell_payload
    partial = not comparator_available
    payload = {
        "schema": SCORE_SCHEMA,
        "task_id": TASK_ID,
        "status": "SCORE_COMPLETE_AFTER_TRUTH_QUALIFIED_PARTIAL_A1_BLOCKER" if partial else "SCORE_COMPLETE_AFTER_TRUTH",
        "freeze": frozen.freeze_record,
        "truth": truth_binding,
        "cell_order": list(CELL_ORDER),
        "cells": cells,
        "p10_inventory": inventory,
        "trace_files": [dict(item) for item in frozen.payload.get("trace_files", [])],
        "trace_status": ("AVAILABLE" if frozen.payload.get("trace_files") else "UNAVAILABLE_PREREGISTERED"),
        "cost_files": [dict(item) for item in frozen.payload.get("cost_files", [])],
        "parsed_costs": _parse_cost_files(frozen.payload.get("cost_files", [])),
        "support_strata": (
            _build_support_strata(frozen, truth, support=frozen.payload["support"], root=root)
            if isinstance(frozen.payload.get("support"), Mapping)
            else _support_strata_unavailable()
        ),
        "a1_diagnostics": (
            {"status": "UNAVAILABLE", "matrix_status": "QUALIFIED_PARTIAL_A1_BLOCKER", "reason": frozen.payload["comparator"].get("reason"), "blocker_id": frozen.payload["comparator"].get("blocker_id")}
            if partial
            else {"status": "AVAILABLE", "matrix_status": "P11_PREDICTIONS_FROZEN_BEFORE_TRUTH", "paired_subset_records": COMPARATOR_RECORDS_PER_DOMAIN}
        ),
        "scorer": {"path": SCORER_RELATIVE_PATH, "source_commit": SCORER_SOURCE_COMMIT, "sha256": SCORER_SHA256, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS, "one_sided_alpha": BOOTSTRAP_ALPHA, "functions": ["paired_exact_cp", "bootstrap_token_delta"]},
        "truth_opened": True,
        "p03_holdout_accessed": False,
        "pooling": False,
    }
    record = _write_json(Path(output_path), payload, root=root, description="P11 evaluation score")
    return {"task_id": TASK_ID, "status": payload["status"], "score": record, "truth_opened": True}


def scorer_contract() -> dict[str, Any]:
    return {"path": SCORER_RELATIVE_PATH, "source_commit": SCORER_SOURCE_COMMIT, "sha256": SCORER_SHA256, "functions": ["paired_exact_cp", "bootstrap_token_delta"], "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS, "one_sided_alpha": BOOTSTRAP_ALPHA, "pooling": False}


__all__ = [
    "CELL_ORDER",
    "COMPARATOR_METHOD",
    "EvaluationError",
    "FREEZE_SCHEMA",
    "MANIFEST_SCHEMA",
    "METHOD_ORDER",
    "QualifiedPartial",
    "SCORE_SCHEMA",
    "freeze_predictions",
    "load_truth_after_freeze",
    "score_after_truth",
    "scorer_contract",
]
