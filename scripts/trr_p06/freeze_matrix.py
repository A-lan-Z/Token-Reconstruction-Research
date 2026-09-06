#!/usr/bin/env python3
"""Assemble the TRR-P06 student/anchor matrix before truth access.

This adapter is deliberately metadata-only.  It reads JSON receipts and hashes
state, observation, and prediction files, but never deserializes a tensor or
opens an evaluator truth file.  It combines the public student manifest with
one or more anchor descriptor manifests and writes a canonical prediction
manifest plus a create-only joint-freeze receipt compatible with
``score_frozen.validate_joint_freeze``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.trr_p06 import score_frozen  # noqa: E402


TASK_ID = score_frozen.TASK_ID
FREEZE_SCHEMA = score_frozen.FREEZE_SCHEMA
PREDICTION_SCHEMA = score_frozen.PREDICTION_SCHEMA
DOMAINS = score_frozen.DOMAINS
TARGETS = score_frozen.TARGETS
METHOD_ORDER = score_frozen.METHOD_ORDER
REPLICATE_SEEDS = score_frozen.REPLICATE_SEEDS
ANCHOR_METHOD_ID = score_frozen.ANCHOR_METHOD_ID
CELL_ORDER = tuple(f"{domain}__{target}" for domain in DOMAINS for target in TARGETS)
ANCHOR_RECORDS_PER_DOMAIN = score_frozen.ANCHOR_RECORDS_PER_DOMAIN
RECORDS_PER_DOMAIN = score_frozen.RECORDS_PER_DOMAIN
SEQUENCE_TOKENS = score_frozen.SEQUENCE_TOKENS
SCORED_POST_BOS = score_frozen.SCORED_POST_BOS
_SHA256 = score_frozen._SHA256
_COMMIT = score_frozen._COMMIT


class FreezeMatrixError(RuntimeError):
    """Raised when the P06 joint-freeze boundary fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise FreezeMatrixError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FreezeMatrixError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise FreezeMatrixError(f"{description} must be a JSON object")
    return dict(value)


def _json_digest(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FreezeMatrixError("value is not canonical-JSON serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_digest(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FreezeMatrixError(f"{description} must be a lowercase SHA-256")
    return value


def _require_commit(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise FreezeMatrixError(f"{description} must be a full commit hash")
    return value


def _file_record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    try:
        return score_frozen._actual_record(path, root=root, description=description)
    except (OSError, score_frozen.P06ScoreError) as exc:
        raise FreezeMatrixError(str(exc)) from exc


def _verify_file_record(record: Mapping[str, Any], *, root: Path, description: str) -> dict[str, Any]:
    try:
        return score_frozen._verify_file_record(record, root=root, description=description)
    except (OSError, score_frozen.P06ScoreError, TypeError, ValueError) as exc:
        raise FreezeMatrixError(str(exc)) from exc


def _record_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return dict(left) == dict(right)


def _git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreezeMatrixError("cannot resolve assembler source commit") from exc
    value = result.stdout.strip()
    return _require_commit(value, description="assembler source commit")


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FreezeMatrixError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise FreezeMatrixError(f"refusing to overwrite create-only artifact: {path}") from exc


def _truth_closed(payload: Mapping[str, Any], *, description: str) -> None:
    for key in ("truth_opened", "source_text_loaded", "source_text_written", "token_ids_written", "token_ids_loaded", "target_labels_loaded", "payload_read"):
        if payload.get(key) is True:
            raise FreezeMatrixError(f"{description} reports forbidden access: {key}")


def _validate_plan(
    plan_path: Path,
    *,
    root: Path,
    approval_manifest_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    plan = _load_json(plan_path, description="P06 plan")
    if plan.get("task_id") != TASK_ID:
        raise FreezeMatrixError("plan task ID changed")
    _truth_closed(plan, description="plan")
    plan_record = _file_record(plan_path, root=root, description="P06 plan")

    approval_record: dict[str, Any] | None = None
    if approval_manifest_path is None:
        candidate = root / "experiments" / TASK_ID / "manifest.json"
        if candidate.is_file() and not candidate.is_symlink():
            approval_manifest_path = candidate
    if approval_manifest_path is not None:
        approval = _load_json(approval_manifest_path, description="P06 task manifest")
        if approval.get("task_id") != TASK_ID:
            raise FreezeMatrixError("task manifest task ID changed")
        review = approval.get("plan_review")
        if not isinstance(review, Mapping) or review.get("status") != "ROOT_APPROVED_FROZEN":
            raise FreezeMatrixError("task manifest does not record ROOT_APPROVED_FROZEN plan review")
        approval_record = _file_record(approval_manifest_path, root=root, description="P06 task manifest")
    return plan, plan_record, approval_record


def _validate_source_selection(
    selection_path: Path,
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection = _load_json(selection_path, description="P06 source selection")
    if selection.get("task_id") != TASK_ID:
        raise FreezeMatrixError("source selection task ID changed")
    _truth_closed(selection, description="source selection")
    if selection.get("paired_conditions") is not True:
        raise FreezeMatrixError("source selection does not declare paired target conditions")
    if selection.get("records_per_domain") != RECORDS_PER_DOMAIN:
        raise FreezeMatrixError("source selection record count changed")
    record = _file_record(selection_path, root=root, description="P06 source selection")
    try:
        identity = score_frozen._load_selection_identity(record, root=root)
    except (OSError, score_frozen.P06ScoreError, TypeError, ValueError) as exc:
        raise FreezeMatrixError(str(exc)) from exc
    return selection, record, identity


def _validate_observation_manifest(
    observation_path: Path,
    *,
    root: Path,
    selection_record: Mapping[str, Any],
    selection_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    observation = _load_json(observation_path, description="P06 observation manifest")
    if observation.get("schema") != "token-reconstruction.trr-p06-public-observation-manifest.v1":
        raise FreezeMatrixError("observation manifest schema changed")
    if observation.get("task_id") != TASK_ID or observation.get("status") != "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH":
        raise FreezeMatrixError("observation manifest is not the frozen no-truth status")
    _truth_closed(observation, description="observation manifest")
    if observation.get("records_per_domain") != RECORDS_PER_DOMAIN:
        raise FreezeMatrixError("observation record count changed")
    if observation.get("sequence_tokens_including_bos") != SEQUENCE_TOKENS or observation.get("scored_post_bos_tokens") != SCORED_POST_BOS:
        raise FreezeMatrixError("observation geometry changed")
    if observation.get("hidden_size") != 2048:
        raise FreezeMatrixError("observation hidden size changed")
    selection_binding = observation.get("selection_plan")
    if not isinstance(selection_binding, Mapping) or selection_binding.get("sha256") != selection_record.get("sha256"):
        raise FreezeMatrixError("observation selection binding changed")
    if observation.get("cell_order") != list(CELL_ORDER):
        raise FreezeMatrixError("observation cell order changed")
    pairing = observation.get("source_pairing")
    if not isinstance(pairing, Mapping) or pairing.get("same_record_ids_across_targets") is not True:
        raise FreezeMatrixError("observation target pairing is not declared")
    expected_ids = selection_identity["record_ids_sha256"]
    if dict(pairing.get("record_ids_sha256", {})) != dict(expected_ids):
        raise FreezeMatrixError("observation record order differs from source selection")

    cells_raw = observation.get("cells")
    if not isinstance(cells_raw, list) or [row.get("cell_id") for row in cells_raw if isinstance(row, Mapping)] != list(CELL_ORDER):
        raise FreezeMatrixError("observation cells are incomplete or reordered")
    cells: dict[str, dict[str, Any]] = {}
    for row in cells_raw:
        if not isinstance(row, Mapping):
            raise FreezeMatrixError("observation cell descriptor is malformed")
        cell_id = row.get("cell_id")
        if cell_id not in CELL_ORDER:
            raise FreezeMatrixError(f"unknown observation cell: {cell_id}")
        domain, target = str(cell_id).split("__", 1)
        if row.get("style") != domain or row.get("condition") != target:
            raise FreezeMatrixError(f"observation cell identity changed: {cell_id}")
        if row.get("record_ids_sha256") != expected_ids[domain]:
            raise FreezeMatrixError(f"observation record order changed: {cell_id}")
        descriptor = row.get("observation")
        if not isinstance(descriptor, Mapping):
            raise FreezeMatrixError(f"observation asset missing: {cell_id}")
        if list(descriptor.get("shape", ())) != [RECORDS_PER_DOMAIN, SEQUENCE_TOKENS, 2048]:
            raise FreezeMatrixError(f"observation tensor geometry changed: {cell_id}")
        if descriptor.get("record_ids_sha256") not in (None, expected_ids[domain]):
            raise FreezeMatrixError(f"observation asset source order changed: {cell_id}")
        if descriptor.get("selection_plan_sha256") not in (None, selection_record.get("sha256")):
            raise FreezeMatrixError(f"observation selection binding changed: {cell_id}")
        actual = _verify_file_record(descriptor, root=root, description=f"observation {cell_id}")
        cells[str(cell_id)] = {
            "record": actual,
            "descriptor": dict(descriptor),
            "domain": domain,
            "target": target,
        }
    return observation, _file_record(observation_path, root=root, description="P06 observation manifest"), cells


def _validate_capture_receipt(
    capture_path: Path,
    *,
    root: Path,
    selection_record: Mapping[str, Any],
    observation_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    capture = _load_json(capture_path, description="P06 capture receipt")
    if capture.get("task_id") != TASK_ID or capture.get("status") != "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH":
        raise FreezeMatrixError("capture receipt is not the completed no-truth status")
    _truth_closed(capture, description="capture receipt")
    if capture.get("source_pairing", {}).get("same_record_ids_across_targets") is not True:
        raise FreezeMatrixError("capture receipt lacks paired target binding")
    obs_binding = capture.get("observations")
    if not isinstance(obs_binding, Mapping) or obs_binding.get("sha256") != observation_record.get("sha256"):
        raise FreezeMatrixError("capture receipt observation binding changed")
    selection_binding = capture.get("selection_plan")
    if not isinstance(selection_binding, Mapping) or selection_binding.get("sha256") != selection_record.get("sha256"):
        raise FreezeMatrixError("capture receipt source selection binding changed")
    return capture, _file_record(capture_path, root=root, description="P06 capture receipt")


def _validate_prerequisite_receipt(
    path: Path,
    *,
    root: Path,
    role: str,
    statuses: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_json(path, description=f"P06 {role} receipt")
    if payload.get("task_id") != TASK_ID:
        raise FreezeMatrixError(f"{role} receipt task ID changed")
    _truth_closed(payload, description=f"{role} receipt")
    if payload.get("status") not in set(statuses):
        raise FreezeMatrixError(f"{role} receipt is not a passing status: {payload.get('status')}")
    return payload, _file_record(path, root=root, description=f"P06 {role} receipt")


def _validate_capacity(payload: Mapping[str, Any]) -> None:
    methods = payload.get("methods")
    if not isinstance(methods, list) or {row.get("method_id") for row in methods if isinstance(row, Mapping)} != set(METHOD_ORDER):
        raise FreezeMatrixError("capacity receipt method matrix changed")
    for row in methods:
        if not isinstance(row, Mapping) or row.get("status") != "PASS" or row.get("direct_affine_frozen") is not True:
            raise FreezeMatrixError("capacity receipt contains a non-passing arm")
        initial = row.get("initial_metrics")
        final = row.get("final_metrics")
        if not isinstance(initial, Mapping) or not isinstance(final, Mapping):
            raise FreezeMatrixError("capacity receipt metrics are missing")
        if initial.get("total") != 256 or initial.get("correct") != 0 or final.get("total") != 256 or int(final.get("correct", -1)) < int(row.get("pass_threshold_correct", 52)):
            raise FreezeMatrixError("capacity receipt did not pass the declared trainability gate")


def _validate_main_fit(
    payload: Mapping[str, Any],
    *,
    student: Mapping[str, Any],
) -> dict[tuple[int, str], str]:
    methods = payload.get("methods")
    expected = {(seed, method) for seed in REPLICATE_SEEDS for method in METHOD_ORDER}
    if not isinstance(methods, list):
        raise FreezeMatrixError("main-fit receipt method matrix is missing")
    state_hashes: dict[tuple[int, str], str] = {}
    for row in methods:
        if not isinstance(row, Mapping) or row.get("status") != "PASS":
            raise FreezeMatrixError("main-fit receipt contains a non-passing arm")
        seed = row.get("seed")
        method = row.get("method_id")
        if isinstance(seed, bool) or not isinstance(seed, int) or (seed, method) in state_hashes or (seed, method) not in expected:
            raise FreezeMatrixError("main-fit receipt seed/method matrix changed")
        state = row.get("state")
        if not isinstance(state, Mapping):
            raise FreezeMatrixError(f"main-fit selected state missing: {seed}/{method}")
        state_hashes[(seed, str(method))] = _require_digest(state.get("sha256"), description=f"main-fit state {seed}/{method}")
    if set(state_hashes) != expected:
        raise FreezeMatrixError("main-fit receipt does not contain all six arms")
    runtime = payload.get("runtime_components")
    if isinstance(runtime, Mapping) and any(runtime.get(key) is True for key in ("a2_student", "guessed_token_feedback", "source_token_access", "target_truth_access")):
        raise FreezeMatrixError("main-fit receipt reports forbidden student input")
    state_bindings = student.get("state_bindings")
    if not isinstance(state_bindings, Mapping):
        raise FreezeMatrixError("student manifest state bindings are missing")
    for (seed, method), expected_hash in state_hashes.items():
        binding = state_bindings.get(f"{seed}::{method}")
        if not isinstance(binding, Mapping) or binding.get("sha256") != expected_hash:
            raise FreezeMatrixError(f"student state differs from main-fit state: {seed}/{method}")
    return state_hashes


def _extract_file_record(value: Any, *, root: Path, description: str) -> dict[str, Any] | None:
    """Extract an artifact file record without interpreting tensor contents."""

    if isinstance(value, str) and value:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        return _file_record(path, root=root, description=description)
    if not isinstance(value, Mapping):
        return None
    if isinstance(value.get("file"), Mapping):
        return _extract_file_record(value["file"], root=root, description=description)
    if isinstance(value.get("path"), str):
        path = Path(str(value["path"])).expanduser()
        if not path.is_absolute():
            path = root / path
        actual = _file_record(path, root=root, description=description)
        if "bytes" in value and value.get("bytes") != actual["bytes"]:
            raise FreezeMatrixError(f"{description} byte binding changed")
        if "sha256" in value and value.get("sha256") != actual["sha256"]:
            raise FreezeMatrixError(f"{description} hash binding changed")
        return actual
    return None


def _anchor_cells_from_payload(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = payload.get("anchor_cells")
    if isinstance(raw, Mapping):
        cells = dict(raw)
    else:
        raw = payload.get("cells")
        if isinstance(raw, Mapping):
            cells = dict(raw)
        elif isinstance(raw, list):
            cells = {}
            for row in raw:
                if isinstance(row, Mapping) and isinstance(row.get("domain"), str):
                    cells[row["domain"]] = dict(row)
        elif isinstance(payload.get("domain"), str):
            cells = {str(payload["domain"]): dict(payload)}
        else:
            cells = {
                domain: payload[domain]
                for domain in DOMAINS
                if isinstance(payload.get(domain), Mapping)
            }
    if not set(cells).issubset(set(DOMAINS)) or not cells or any(not isinstance(cells[domain], Mapping) for domain in cells):
        raise FreezeMatrixError("anchor descriptor set contains an unknown or empty domain set")
    return {domain: dict(cells[domain]) for domain in cells}


def _validate_anchor_manifests(
    anchor_paths: Sequence[Path],
    *,
    root: Path,
    selection_identity: Mapping[str, Any],
    observation_cells: Mapping[str, Mapping[str, Any]],
    selection_record: Mapping[str, Any],
    observation_record: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], str, dict[str, str], dict[str, Any], list[dict[str, Any]], str | None, dict[str, Any]]:
    if not anchor_paths:
        raise FreezeMatrixError("at least one anchor descriptor manifest is required")
    cells: dict[str, dict[str, Any]] = {}
    semantic_state_hashes: set[str] = set()
    subset_hashes = {
        domain: _json_digest(selection_identity["record_ids"][domain][:ANCHOR_RECORDS_PER_DOMAIN])
        for domain in DOMAINS
    }
    state_file_records: list[dict[str, Any]] = []
    anchor_code_commits: set[str] = set()
    manifest_records: list[dict[str, Any]] = []
    anchor_sources: list[dict[str, Any]] = []
    anchor_input_bindings: dict[str, Any] = {}
    for path in anchor_paths:
        payload = _load_json(path, description="P06 anchor descriptor")
        if payload.get("task_id") not in (None, TASK_ID):
            raise FreezeMatrixError("anchor descriptor task ID changed")
        _truth_closed(payload, description="anchor descriptor")
        payload_cells = _anchor_cells_from_payload(payload)
        manifest_records.append(_file_record(path, root=root, description="P06 anchor descriptor"))
        if isinstance(payload.get("code_commit"), str):
            anchor_code_commits.add(_require_commit(payload["code_commit"], description="anchor code commit"))
        for binding_name, binding_value in (("selection", payload.get("selection")), ("observation_manifest", payload.get("observation_manifest"))):
            if isinstance(binding_value, Mapping) and isinstance(binding_value.get("path"), str):
                actual_binding = _verify_file_record(binding_value, root=root, description=f"anchor {binding_name}")
                expected_binding = selection_record if binding_name == "selection" else observation_record
                if actual_binding["sha256"] != expected_binding["sha256"]:
                    raise FreezeMatrixError(f"anchor {binding_name} differs from frozen P06 input")
                anchor_input_bindings.setdefault(binding_name, actual_binding)
        semantic = payload.get("anchor_state_sha256") or payload.get("state_sha256")
        for domain in payload_cells:
            descriptor = dict(payload_cells[domain])
            if domain in cells:
                raise FreezeMatrixError(f"duplicate anchor descriptor: {domain}")
            if descriptor.get("task_id") not in (None, TASK_ID):
                raise FreezeMatrixError(f"anchor task binding changed: {domain}")
            if descriptor.get("domain") not in (None, domain):
                raise FreezeMatrixError(f"anchor domain binding changed: {domain}")
            descriptor.update({"task_id": TASK_ID, "domain": domain, "target": "public_base"})
            descriptor.setdefault("method_id", ANCHOR_METHOD_ID)
            descriptor.setdefault("subset", "first64_public_base")
            if descriptor.get("method_id") != ANCHOR_METHOD_ID or descriptor.get("subset") != "first64_public_base":
                raise FreezeMatrixError(f"anchor identity changed: {domain}")
            descriptor.setdefault("records", ANCHOR_RECORDS_PER_DOMAIN)
            descriptor.setdefault("shape", [ANCHOR_RECORDS_PER_DOMAIN, SEQUENCE_TOKENS])
            descriptor.setdefault("scored_post_bos_tokens", SCORED_POST_BOS)
            if descriptor.get("records") != ANCHOR_RECORDS_PER_DOMAIN or list(descriptor.get("shape", ())) != [ANCHOR_RECORDS_PER_DOMAIN, SEQUENCE_TOKENS] or descriptor.get("scored_post_bos_tokens") != SCORED_POST_BOS:
                raise FreezeMatrixError(f"anchor geometry changed: {domain}")
            descriptor.setdefault("anchor_subset_record_ids_sha256", subset_hashes[domain])
            descriptor.setdefault("record_ids_sha256", subset_hashes[domain])
            if descriptor.get("anchor_subset_record_ids_sha256") != subset_hashes[domain] or descriptor.get("record_ids_sha256") != subset_hashes[domain]:
                raise FreezeMatrixError(f"anchor source subset changed: {domain}")
            for field in ("attention_mask_sha256", "position_ids_sha256"):
                _require_digest(descriptor.get(field), description=f"anchor {domain}.{field}")
            prediction = descriptor.get("prediction")
            _verify_file_record(prediction, root=root, description=f"anchor {domain} prediction")
            public_base_observation = observation_cells[f"{domain}__public_base"]["record"]["sha256"]
            if descriptor.get("observation_sha256") != public_base_observation:
                raise FreezeMatrixError(f"anchor observation binding changed: {domain}")
            if descriptor.get("truth_opened") is True or descriptor.get("candidate_arrays_persisted") is True:
                raise FreezeMatrixError(f"anchor {domain} reports forbidden payload state")
            local_semantic = descriptor.get("state_sha256") or semantic
            local_semantic = _require_digest(local_semantic, description=f"anchor {domain} state")
            if semantic is not None and local_semantic != semantic:
                raise FreezeMatrixError(f"anchor state differs across descriptor and manifest: {domain}")
            semantic_state_hashes.add(local_semantic)
            descriptor["state_sha256"] = local_semantic
            cells[domain] = descriptor

        # A state-file binding may be global or nested under one of the cells.
        candidates: list[Any] = []
        for key in ("anchor_state_binding", "state_binding", "anchor_state_file", "state_file", "anchor_state", "state_artifact", "state"):
            if key in payload:
                candidates.append(payload[key])
        for descriptor in payload_cells.values():
            if isinstance(descriptor, Mapping):
                for key in ("anchor_state_binding", "state_binding", "anchor_state_file", "state_file", "state_artifact", "state"):
                    if key in descriptor:
                        candidates.append(descriptor[key])
        for candidate in candidates:
            record = _extract_file_record(candidate, root=root, description="anchor state artifact")
            if record is not None and record not in state_file_records:
                state_file_records.append(record)
        runtime_assets = payload.get("runtime_assets")
        if isinstance(runtime_assets, Mapping):
            for asset_name in ("embedding_table", "retained_a1_state", "parent_reference"):
                if asset_name in runtime_assets:
                    record = _extract_file_record(runtime_assets[asset_name], root=root, description=f"anchor runtime asset {asset_name}")
                    if record is not None:
                        anchor_input_bindings[f"{asset_name}:{len(anchor_input_bindings)}"] = record
        anchor_sources.append({"path": str(path), "schema": payload.get("schema"), "code_commit": payload.get("code_commit")})

    if set(cells) != set(DOMAINS):
        raise FreezeMatrixError("anchor descriptor set is incomplete")
    if semantic_state_hashes != {cells[domain]["state_sha256"] for domain in DOMAINS} or len(semantic_state_hashes) != 1:
        raise FreezeMatrixError("anchor state hash is not common across domains")
    if len(state_file_records) != 1:
        raise FreezeMatrixError("anchor state artifact must have exactly one common file binding")
    if len(anchor_code_commits) > 1:
        raise FreezeMatrixError("anchor descriptor code commits differ")
    semantic_state = next(iter(semantic_state_hashes))
    return cells, semantic_state, subset_hashes, state_file_records[0], manifest_records, next(iter(anchor_code_commits), None), anchor_input_bindings


def _student_input_bindings(student: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fit_receipt = student.get("fit_receipt")
    if isinstance(fit_receipt, Mapping):
        result["fit_receipt"] = _verify_file_record(fit_receipt, root=root, description="student fit receipt")
    runtime_assets = student.get("runtime_assets")
    if isinstance(runtime_assets, Mapping):
        normalized_assets: dict[str, Any] = {}
        for key, value in runtime_assets.items():
            if isinstance(value, Mapping) and isinstance(value.get("path"), str):
                normalized_assets[key] = _verify_file_record(value, root=root, description=f"student runtime asset {key}")
            else:
                normalized_assets[key] = value
        result["runtime_assets"] = normalized_assets
    state_bindings = student.get("state_bindings")
    if isinstance(state_bindings, Mapping):
        result["state_bindings"] = {
            str(key): _verify_file_record(value, root=root, description=f"student state {key}")
            for key, value in state_bindings.items()
        }
    return result


def _prediction_inventory(student: Mapping[str, Any], anchors: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    student_predictions: dict[str, Any] = {}
    cells = student.get("student_cells")
    if not isinstance(cells, Mapping):
        raise FreezeMatrixError("student prediction cells are missing")
    for cell_id, cell in cells.items():
        if not isinstance(cell, Mapping) or not isinstance(cell.get("replicates"), Mapping):
            raise FreezeMatrixError(f"student prediction cell is malformed: {cell_id}")
        for seed, methods in cell["replicates"].items():
            if not isinstance(methods, Mapping):
                raise FreezeMatrixError(f"student prediction replicate is malformed: {cell_id}/{seed}")
            for method, descriptor in methods.items():
                if not isinstance(descriptor, Mapping):
                    raise FreezeMatrixError(f"student prediction descriptor is malformed: {cell_id}/{seed}/{method}")
                student_predictions[f"{cell_id}::{seed}::{method}"] = descriptor["prediction"]
    return {
        "student": student_predictions,
        "anchor": {domain: anchors[domain]["prediction"] for domain in DOMAINS},
    }


def _canonicalize_student_manifest(student: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the prediction runner's cell map to the scorer's frozen contract.

    ``run_predictions`` records each cell as ``cell_id -> seed -> method``.
    The scorer contract deliberately wraps that map with explicit ``domain``,
    ``target``, and ``replicates`` fields.  Keep the producer JSON as the
    immutable source binding and construct a separate canonical view for
    validation and the create-only frozen manifest.
    """

    raw_cells = student.get("student_cells")
    expected_cells = set(CELL_ORDER)
    if not isinstance(raw_cells, Mapping) or set(raw_cells) != expected_cells:
        raise FreezeMatrixError("student prediction matrix is not exactly the four registered target cells")
    canonical_cells: dict[str, dict[str, Any]] = {}
    for cell_id in CELL_ORDER:
        raw_cell = raw_cells[cell_id]
        if not isinstance(raw_cell, Mapping):
            raise FreezeMatrixError(f"student prediction cell is malformed: {cell_id}")
        domain, target = cell_id.split("__", 1)
        if isinstance(raw_cell.get("replicates"), Mapping):
            # Already canonical: preserve its explicit identity fields and
            # copy the wrapper so the producer object is never mutated.
            canonical = dict(raw_cell)
            canonical["replicates"] = dict(raw_cell["replicates"])
        else:
            # The production runner's no-truth manifest uses the compact
            # seed-keyed form.  The seed/method checks remain in the scorer;
            # this adapter only supplies the contract wrapper it requires.
            canonical = {"domain": domain, "target": target, "replicates": dict(raw_cell)}
        canonical_cells[cell_id] = canonical
    result = dict(student)
    result["student_cells"] = canonical_cells
    return result


def assemble_joint_freeze(
    *,
    repository_root: Path,
    student_manifest_path: Path,
    anchor_manifest_paths: Sequence[Path],
    plan_path: Path,
    source_selection_path: Path,
    observation_manifest_path: Path,
    capture_receipt_path: Path,
    preflight_receipt_path: Path,
    qualification_receipt_path: Path,
    capacity_receipt_path: Path,
    main_fit_receipt_path: Path,
    output_manifest_path: Path,
    output_freeze_path: Path,
    approval_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Create the canonical P06 matrix and no-truth joint receipt."""

    root = Path(repository_root).expanduser().resolve()
    manifest_out = Path(output_manifest_path).expanduser().resolve()
    freeze_out = Path(output_freeze_path).expanduser().resolve()
    if manifest_out == freeze_out:
        raise FreezeMatrixError("prediction manifest and freeze receipt must be distinct files")
    for path in (manifest_out, freeze_out):
        if path.exists() or path.is_symlink():
            raise FreezeMatrixError(f"create-only output already exists: {path}")

    plan, plan_record, approval_record = _validate_plan(plan_path, root=root, approval_manifest_path=approval_manifest_path)
    selection, selection_record, selection_identity = _validate_source_selection(source_selection_path, root=root)
    observation, observation_record, observation_cells = _validate_observation_manifest(
        observation_manifest_path,
        root=root,
        selection_record=selection_record,
        selection_identity=selection_identity,
    )
    capture, capture_record = _validate_capture_receipt(
        capture_receipt_path,
        root=root,
        selection_record=selection_record,
        observation_record=observation_record,
    )

    student = _load_json(student_manifest_path, description="P06 student prediction manifest")
    if student.get("schema") != "token-reconstruction.trr-p06-student-prediction-manifest.v1" or student.get("task_id") != TASK_ID:
        raise FreezeMatrixError("student manifest schema or task ID changed")
    if student.get("status") not in ("STUDENT_PREDICTIONS_COMPLETE_NO_TRUTH", "FROZEN_P06_PREDICTIONS_NO_TRUTH"):
        raise FreezeMatrixError("student manifest is not a completed no-truth manifest")
    _truth_closed(student, description="student prediction manifest")
    student_code_commit = _require_commit(student.get("code_commit"), description="student code commit")
    student_record = _file_record(student_manifest_path, root=root, description="student prediction manifest")

    # Adapt the runner's compact cell map to the scorer's explicit contract,
    # while retaining the raw producer manifest as an immutable source record.
    canonical_student = _canonicalize_student_manifest(student)

    # Validate the complete student matrix before adding the anchor bindings.
    try:
        normalized_students, student_meta = score_frozen._validate_student_matrix(canonical_student, root=root)
    except (OSError, score_frozen.P06ScoreError, TypeError, ValueError) as exc:
        raise FreezeMatrixError(str(exc)) from exc
    if student_meta["record_ids_sha256"] != selection_identity["record_ids_sha256"]:
        raise FreezeMatrixError("student source order differs from frozen source selection")
    for cell_id, digest in student_meta["observation_sha256"].items():
        if observation_cells[cell_id]["record"]["sha256"] != digest:
            raise FreezeMatrixError(f"student observation binding differs from capture: {cell_id}")

    # The runner's manifest may use an absolute observation path.  The joint
    # manifest canonicalizes it to the root-relative record used by the scorer.
    existing_observation = canonical_student.get("observation_manifest")
    if isinstance(existing_observation, Mapping):
        existing = _verify_file_record(existing_observation, root=root, description="student observation manifest")
        if existing["sha256"] != observation_record["sha256"]:
            raise FreezeMatrixError("student observation-manifest binding changed")

    anchors, anchor_state_sha, anchor_subset_hashes, anchor_state_file, anchor_manifest_records, anchor_code_commit, anchor_input_bindings = _validate_anchor_manifests(
        anchor_manifest_paths,
        root=root,
        selection_identity=selection_identity,
        observation_cells=observation_cells,
        selection_record=selection_record,
        observation_record=observation_record,
    )

    preflight, preflight_record = _validate_prerequisite_receipt(
        preflight_receipt_path, root=root, role="resource preflight", statuses=("SOURCE_ONLY_PREFLIGHT_PASS", "PASS")
    )
    qualification, qualification_record = _validate_prerequisite_receipt(
        qualification_receipt_path, root=root, role="largest-cell qualification", statuses=("PASS",)
    )
    capacity, capacity_record = _validate_prerequisite_receipt(
        capacity_receipt_path, root=root, role="capacity", statuses=("PASS",)
    )
    main_fit, main_fit_record = _validate_prerequisite_receipt(
        main_fit_receipt_path, root=root, role="main fit", statuses=("PASS",)
    )
    _validate_capacity(capacity)
    state_hashes = _validate_main_fit(main_fit, student=canonical_student)
    for (seed, method), state_sha in state_hashes.items():
        descriptor = normalized_students[CELL_ORDER[0]][str(seed)][method]
        if descriptor.get("state_sha256") != state_sha:
            raise FreezeMatrixError(f"student prediction state binding differs from main fit: {seed}/{method}")

    scientific_preconditions = {
        "plan_frozen": True,
        "resource_qualified": True,
        "capacity_qualified": True,
        "all_fits_finite": True,
        "source_pairing_validated": True,
    }
    # Explicitly preserve the receipt statuses in the no-truth receipt rather
    # than deriving a truth-dependent pass later.
    if any(value is not True for value in scientific_preconditions.values()):
        raise FreezeMatrixError("scientific preconditions are incomplete")

    normalized_manifest = dict(canonical_student)
    normalized_manifest.update(
        {
            "schema": PREDICTION_SCHEMA,
            "status": "FROZEN_P06_PREDICTIONS_NO_TRUTH",
            "truth_opened": False,
            "plan": plan_record,
            "source_selection": selection_record,
            "observation_manifest": observation_record,
            "capture_receipt": capture_record,
            "anchor_manifest": anchor_manifest_records,
            "anchor_code_commit": anchor_code_commit,
            "anchor_state_sha256": anchor_state_sha,
            "anchor_state_binding": anchor_state_file,
            "anchor_input_bindings": anchor_input_bindings,
            "anchor_subset_record_ids_sha256": anchor_subset_hashes,
            "anchor_cells": anchors,
            "student_manifest_source": student_record,
            "source_identity": {
                "record_ids_sha256": selection_identity["record_ids_sha256"],
                "record_count_per_domain": RECORDS_PER_DOMAIN,
                "anchor_subset_record_ids_sha256": anchor_subset_hashes,
            },
            "truth_gate": "complete student and anchor matrix frozen before any truth access",
        }
    )
    # Avoid carrying a file record written by an earlier producer instance.
    normalized_manifest["code_commit"] = student_code_commit

    input_bindings = _student_input_bindings(canonical_student, root=root)
    prediction_inventory = _prediction_inventory(normalized_manifest, anchors)
    prerequisite_receipts = {
        "preflight": {"file": preflight_record, "schema": preflight.get("schema"), "status": preflight.get("status"), "source_commit": preflight.get("source_commit")},
        "qualification": {"file": qualification_record, "schema": qualification.get("schema"), "status": qualification.get("status"), "source_commit": qualification.get("source_commit")},
        "capacity": {"file": capacity_record, "schema": capacity.get("schema"), "status": capacity.get("status"), "source_commit": capacity.get("source_commit")},
        "main_fit": {"file": main_fit_record, "schema": main_fit.get("schema"), "status": main_fit.get("status"), "source_commit": main_fit.get("source_commit")},
    }

    # Resolve the assembler commit before writing either create-only output.
    # The receipt hashes exactly the manifest bytes written below.
    assembler_commit = _git_head(root)
    _write_create_only(manifest_out, normalized_manifest)
    manifest_record = _file_record(manifest_out, root=root, description="joint prediction manifest")
    freeze = {
        "schema": FREEZE_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_P06_MATRIX_NO_TRUTH",
        "truth_opened": False,
        "created_utc": _utc_now(),
        "code_commit": student_code_commit,
        "assembler_code_commit": assembler_commit,
        "plan": plan_record,
        "task_manifest": approval_record,
        "source_selection": selection_record,
        "observation_manifest": observation_record,
        "capture_receipt": capture_record,
        "student_manifest": student_record,
        "anchor_manifests": anchor_manifest_records,
        "anchor_state_sha256": anchor_state_sha,
        "anchor_state_binding": anchor_state_file,
        "anchor_input_bindings": anchor_input_bindings,
        "anchor_subset_record_ids_sha256": anchor_subset_hashes,
        "prediction_manifest_sha256": manifest_record["sha256"],
        "prediction_manifest": manifest_record,
        "scientific_preconditions": scientific_preconditions,
        "prerequisite_receipts": prerequisite_receipts,
        "input_bindings": input_bindings,
        "prediction_bindings": prediction_inventory,
        "state_bindings": input_bindings.get("state_bindings", {}),
        "source_identity": {
            "record_ids_sha256": selection_identity["record_ids_sha256"],
            "final_sequence_sha256": selection_identity["final_sequence_sha256"],
            "anchor_subset_record_ids_sha256": anchor_subset_hashes,
        },
        "matrix": {
            "student_cells": list(CELL_ORDER),
            "student_replicates": list(REPLICATE_SEEDS),
            "student_methods": list(METHOD_ORDER),
            "student_prediction_count": len(CELL_ORDER) * len(REPLICATE_SEEDS) * len(METHOD_ORDER),
            "anchor_domains": list(DOMAINS),
            "anchor_records_per_domain": ANCHOR_RECORDS_PER_DOMAIN,
        },
        "access_assertions": {
            "truth_opened": False,
            "student_source_text_loaded": False,
            "student_target_labels_loaded": False,
            "student_candidate_arrays_persisted": False,
            "anchor_truth_opened": False,
        },
    }
    _write_create_only(freeze_out, freeze)
    freeze_record = _file_record(freeze_out, root=root, description="joint freeze receipt")
    return {
        "task_id": TASK_ID,
        "status": freeze["status"],
        "truth_opened": False,
        "prediction_manifest": manifest_record,
        "freeze_receipt": freeze_record,
        "scientific_preconditions": scientific_preconditions,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--student-manifest", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-selection", type=Path, required=True)
    parser.add_argument("--observation-manifest", type=Path, required=True)
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--qualification-receipt", type=Path, required=True)
    parser.add_argument("--capacity-receipt", type=Path, required=True)
    parser.add_argument("--main-fit-receipt", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-freeze", type=Path, required=True)
    parser.add_argument("--approval-manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = assemble_joint_freeze(
            repository_root=args.repository_root,
            student_manifest_path=args.student_manifest,
            anchor_manifest_paths=args.anchor_manifest,
            plan_path=args.plan,
            source_selection_path=args.source_selection,
            observation_manifest_path=args.observation_manifest,
            capture_receipt_path=args.capture_receipt,
            preflight_receipt_path=args.preflight_receipt,
            qualification_receipt_path=args.qualification_receipt,
            capacity_receipt_path=args.capacity_receipt,
            main_fit_receipt_path=args.main_fit_receipt,
            output_manifest_path=args.output_manifest,
            output_freeze_path=args.output_freeze,
            approval_manifest_path=args.approval_manifest,
        )
    except (FreezeMatrixError, OSError, ValueError, score_frozen.P06ScoreError) as exc:
        print(f"TRR-P06 freeze failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
