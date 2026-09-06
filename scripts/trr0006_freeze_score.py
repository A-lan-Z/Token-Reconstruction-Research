#!/usr/bin/env python3
"""Execute the TRR-0006 public freeze, one truth open, and paired score.

This is the task-local executable boundary for the main 1,536-record matrix.
It consumes the frozen plan, source selection, registration, public runner
receipt, and one producer truth binding.  The truth binding is inspected as
metadata before the public gate; its label sidecar is opened exactly once
only after :func:`trr0006_freeze_pair.validate_before_truth` succeeds.

The private producer binding uses this narrow interface:

* ``schema`` is ``token-reconstruction.trr0006-private-label-binding.v1``;
* ``truth_file`` is an absolute ``{path, bytes, sha256}`` record outside the
  reconstruction root and frozen prediction output root;
* ``decision_plan`` and ``source_selection`` are file records (the aliases
  ``plan`` and ``selection`` are accepted), and ``observation_sha256`` binds
  all four public cells;
* ``record_ids_sha256`` binds the two source orders and
  ``truth_tensor_keys`` is exactly ``["finance__token_ids",
  "pile__token_ids"]`` in any order;
* ``truth_opened`` and ``reconstruction_root_contains_truth`` are false.

The sidecar contains exactly those two int64 ``[records_per_domain, 128]``
label tensors.  The domain label is shared by its public and synthetic-LoRA
cells; masks and positions remain public observation tensors.  Sidecar
metadata carries the schema, task, and the same plan/selection/observation
bindings when available.  No labels are copied into result, report, manifest,
or execution receipt artifacts.
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

import torch
from safetensors import safe_open

try:  # Repository import form.
    from scripts import trr0006_freeze_pair as freeze
    from scripts import trr0006_prediction_contract as contract
    from scripts import trr0006_score_pair as scorer
except ModuleNotFoundError:  # Direct execution with PYTHONPATH=src:scripts.
    import trr0006_freeze_pair as freeze
    import trr0006_prediction_contract as contract
    import trr0006_score_pair as scorer


TASK_ID = "TRR-0006"
TRUTH_BINDING_SCHEMA = "token-reconstruction.trr0006-private-label-binding.v1"
TRUTH_SIDECAR_SCHEMA = "token-reconstruction.trr0006-private-label-sidecar.v1"
TRUTH_READY_STATUS = "PUBLIC_TRUTH_PREPARED_OUTSIDE_RECONSTRUCTION_ROOT"
EXECUTION_BINDING_SCHEMA = "token-reconstruction.trr0006-scoring-execution-binding.v1"
EXECUTION_BINDING_SPECS = (
    ("scoring_driver", "scripts/trr0006_freeze_score.py"),
    ("public_freeze_gate", "scripts/trr0006_freeze_pair.py"),
    ("pair_scorer", "scripts/trr0006_score_pair.py"),
)
SOURCE_SELECTION_SCHEMA = "token-reconstruction.trr0006-source-selection.v1"
SOURCE_SELECTION_STATUS = "FROZEN_TRR0006_SOURCE_SELECTION_NO_TRUTH"
EXPECTED_TRUTH_KEYS = ("finance__token_ids", "pile__token_ids")
EXPECTED_OBSERVATION_KEYS = set(freeze.CELL_ORDER)


class FreezeScoreError(RuntimeError):
    """Raised when the executable freeze/score boundary fails closed."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _root(value: Path | str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_symlink() or not path.is_dir():
        raise FreezeScoreError(f"repository root is unavailable: {path}")
    return path


def _canonical_path(value: Path | str, *, root: Path | None = None, description: str, require_file: bool = True) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        if root is None:
            raise FreezeScoreError(f"{description} path must be absolute")
        raw = root / raw
    # The private truth path is intentionally resolved without requiring that
    # it exists during the pre-gate phase.  The gate itself only does the same
    # path containment check; bytes are first inspected below the truth gate.
    path = raw.resolve(strict=False)
    if require_file and (path.is_symlink() or not path.is_file()):
        raise FreezeScoreError(f"{description} is unavailable: {path}")
    return path


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _actual_record(path: Path, *, root: Path | None = None, description: str = "asset") -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise FreezeScoreError(f"{description} is unavailable: {path}")
    try:
        digest = contract.sha256_file(path)
    except contract.ContractError as exc:
        raise FreezeScoreError(str(exc)) from exc
    return {
        "path": _relative(path, root) if root is not None else str(path),
        "bytes": int(path.stat().st_size),
        "sha256": digest,
    }


def _record_path(record: Mapping[str, Any], *, root: Path, description: str) -> Path:
    value = record.get("path")
    if not isinstance(value, str) or not value:
        raise FreezeScoreError(f"{description} path is absent")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = root / raw
    return raw.resolve()


def _same_record(actual: Mapping[str, Any], declared: Mapping[str, Any], *, root: Path, description: str) -> None:
    try:
        declared_path = _record_path(declared, root=root, description=description)
    except FreezeScoreError:
        raise
    actual_path = _record_path(actual, root=root, description=description)
    if actual_path != declared_path or actual.get("bytes") != declared.get("bytes") or actual.get("sha256") != declared.get("sha256"):
        raise FreezeScoreError(f"{description} binding changed")


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FreezeScoreError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FreezeScoreError(f"{description} must be a JSON object")
    return value


def _json_digest(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FreezeScoreError("record-order digest input is not JSON-safe") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_head(root: Path) -> str:
    try:
        value = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreezeScoreError("cannot resolve current executable commit") from exc
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise FreezeScoreError("current executable commit is not a full lowercase hash")
    return value


def _validate_execution_binding(registration: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    binding = registration.get("execution_binding")
    if not isinstance(binding, Mapping):
        raise FreezeScoreError("main registration has no scoring execution binding")
    if binding.get("schema") != EXECUTION_BINDING_SCHEMA:
        raise FreezeScoreError("scoring execution binding schema changed")
    current_head = _git_head(root)
    if registration.get("code_commit") != current_head or binding.get("code_commit") != current_head:
        raise FreezeScoreError("registration/scoring execution commit differs from current HEAD")
    rows = binding.get("files")
    if not isinstance(rows, list) or len(rows) != len(EXECUTION_BINDING_SPECS):
        raise FreezeScoreError("scoring execution file binding is incomplete")
    checked: list[dict[str, Any]] = []
    for index, (role, expected_path) in enumerate(EXECUTION_BINDING_SPECS):
        row = rows[index]
        if not isinstance(row, Mapping) or row.get("role") != role or row.get("path") != expected_path:
            raise FreezeScoreError(f"scoring execution binding changed: {role}")
        path = _record_path(row, root=root, description=f"scoring executable {role}")
        actual = _actual_record(path, root=root, description=f"scoring executable {role}")
        _same_record(actual, row, root=root, description=f"scoring executable {role}")
        checked.append({"role": role, **actual})
    return {"schema": EXECUTION_BINDING_SCHEMA, "code_commit": current_head, "files": checked}


def _validate_plan_binding(
    registration: Mapping[str, Any],
    plan_path: Path,
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    declared = registration.get("decision_plan")
    if not isinstance(declared, Mapping):
        raise FreezeScoreError("registration decision-plan binding is absent")
    path = _canonical_path(plan_path, root=root, description="decision plan")
    actual = _actual_record(path, root=root, description="decision plan")
    _same_record(actual, declared, root=root, description="decision plan")
    if registration.get("decision_plan_sha256") != actual["sha256"]:
        raise FreezeScoreError("registration decision-plan digest changed")
    plan = _load_json(path, description="decision plan")
    if plan.get("schema") != "token-reconstruction.trr0006-decision-plan.v1" or plan.get("task_id") != TASK_ID:
        raise FreezeScoreError("decision plan identity changed")
    if not str(plan.get("status", "")).startswith("FROZEN") or plan.get("sample_size_frozen") is False:
        raise FreezeScoreError("decision plan is not frozen")
    if plan.get("truth_opened") is True or plan.get("evaluation_truth_opened") is True:
        raise FreezeScoreError("decision plan was written after truth access")
    panel = plan.get("panel")
    comparison = plan.get("comparison")
    if not isinstance(panel, Mapping) or not isinstance(comparison, Mapping):
        raise FreezeScoreError("decision plan panel/comparison is absent")
    if panel.get("records_per_domain") != registration.get("records_per_domain"):
        raise FreezeScoreError("plan and registration record counts differ")
    if panel.get("clip_tokens_including_bos") != contract.STORED_SEQUENCE_TOKENS or panel.get("scored_post_bos_tokens") != contract.SCORED_POST_BOS_TOKENS:
        raise FreezeScoreError("decision plan clip geometry changed")
    if panel.get("unique_sources_total") != 2 * int(registration["records_per_domain"]):
        raise FreezeScoreError("decision plan source count changed")
    if panel.get("record_condition_evaluations_per_method") != 4 * int(registration["records_per_domain"]):
        raise FreezeScoreError("decision plan paired evaluation count changed")
    if comparison.get("cells") != list(contract.CELL_ORDER) or comparison.get("method_order") != list(contract.METHOD_IDS):
        raise FreezeScoreError("decision plan matrix order changed")
    return plan, actual


def _validate_selection_binding(
    registration: Mapping[str, Any],
    selection_path: Path | None,
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[str]]]:
    declared = registration.get("source_selection")
    if not isinstance(declared, Mapping):
        raise FreezeScoreError("main registration has no source-selection binding")
    path = _record_path(declared, root=root, description="source selection") if selection_path is None else _canonical_path(selection_path, root=root, description="source selection")
    actual = _actual_record(path, root=root, description="source selection")
    _same_record(actual, declared, root=root, description="source selection")
    selection = _load_json(path, description="source selection")
    if selection.get("schema") != SOURCE_SELECTION_SCHEMA or selection.get("task_id") != TASK_ID or selection.get("status") != SOURCE_SELECTION_STATUS:
        raise FreezeScoreError("source selection is not the frozen TRR-0006 selection")
    records_per_domain = registration.get("records_per_domain")
    if selection.get("records_per_domain") != records_per_domain or selection.get("paired_conditions") is not True:
        raise FreezeScoreError("source selection record count or pairing changed")
    if selection.get("truth_opened") is True or selection.get("source_text_or_token_ids_written") is True:
        raise FreezeScoreError("source selection contains a truth/source payload")
    rule = selection.get("selection_rule")
    if not isinstance(rule, Mapping) or rule.get("source_text_or_token_ids_written") is not False:
        raise FreezeScoreError("source selection is not source-free")
    rows = rule.get("records")
    declared_digests = rule.get("record_ids_sha256")
    if not isinstance(rows, Mapping) or set(rows) != {"pile", "finance"} or not isinstance(declared_digests, Mapping) or set(declared_digests) != {"pile", "finance"}:
        raise FreezeScoreError("source selection record order is incomplete")
    record_ids: dict[str, list[str]] = {}
    for domain in ("pile", "finance"):
        domain_rows = rows.get(domain)
        if not isinstance(domain_rows, list) or len(domain_rows) != int(records_per_domain):
            raise FreezeScoreError(f"source selection records changed: {domain}")
        ids: list[str] = []
        for row in domain_rows:
            if not isinstance(row, Mapping):
                raise FreezeScoreError(f"source selection row is malformed: {domain}")
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise FreezeScoreError(f"source selection record ID is malformed: {domain}")
            ids.append(record_id)
        if len(set(ids)) != len(ids) or _json_digest(ids) != declared_digests.get(domain):
            raise FreezeScoreError(f"source selection record order digest changed: {domain}")
        record_ids[domain] = ids
    registered_digests = registration.get("source_record_ids_sha256")
    if registered_digests != dict(declared_digests):
        raise FreezeScoreError("registration source-record digests changed")
    return selection, actual, record_ids


def _binding_file_record(binding: Mapping[str, Any], names: Sequence[str], *, description: str) -> Mapping[str, Any] | None:
    for name in names:
        value = binding.get(name)
        if isinstance(value, Mapping):
            return value
    return None


def _check_hash_alias(binding: Mapping[str, Any], names: Sequence[str], expected: str, *, description: str) -> None:
    for name in names:
        value = binding.get(name)
        if value is not None:
            if isinstance(value, Mapping):
                value = value.get("sha256")
            if value != expected:
                raise FreezeScoreError(f"truth binding {description} changed")
            return


def _truth_header(
    binding_path: Path,
    truth_path: Path,
    *,
    root: Path,
    plan_record: Mapping[str, Any],
    selection_record: Mapping[str, Any],
    record_ids: Mapping[str, Sequence[str]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read only private-binding metadata before the public gate."""

    # This record is the only pre-gate read of the producer binding file.  The
    # truth sidecar itself is deliberately not stat'ed or hashed here.
    binding_path = _canonical_path(binding_path, description="private truth binding")
    binding_record = _actual_record(binding_path, description="private truth binding")
    binding = _load_json(binding_path, description="private truth binding")
    if binding.get("schema") != TRUTH_BINDING_SCHEMA or binding.get("task_id") != TASK_ID or binding.get("status") != TRUTH_READY_STATUS:
        raise FreezeScoreError("private truth binding is not a prepared TRR-0006 binding")
    if binding.get("truth_opened") is not False or binding.get("reconstruction_root_contains_truth") is not False:
        raise FreezeScoreError("private truth binding is already opened or inside the reconstruction root")
    truth_record = binding.get("truth_file")
    if not isinstance(truth_record, Mapping):
        truth_record = binding.get("sidecar")
    if not isinstance(truth_record, Mapping):
        raise FreezeScoreError("private truth artifact record is absent")
    declared_truth_path = truth_record.get("path")
    if not isinstance(declared_truth_path, str) or not Path(declared_truth_path).expanduser().is_absolute():
        raise FreezeScoreError("private truth artifact path must be absolute")
    declared_truth_path = Path(declared_truth_path).expanduser().resolve(strict=False)
    supplied_truth_path = Path(truth_path).expanduser().absolute().resolve(strict=False)
    if declared_truth_path != supplied_truth_path:
        raise FreezeScoreError("private truth artifact path differs from its binding")
    if _inside(declared_truth_path, root):
        raise FreezeScoreError("private truth artifact is inside the reconstruction root")
    if not isinstance(truth_record.get("bytes"), int) or isinstance(truth_record.get("bytes"), bool) or truth_record["bytes"] <= 0:
        raise FreezeScoreError("private truth artifact byte count is invalid")
    truth_sha = truth_record.get("sha256")
    if not isinstance(truth_sha, str) or len(truth_sha) != 64 or any(c not in "0123456789abcdef" for c in truth_sha):
        raise FreezeScoreError("private truth artifact digest is invalid")
    plan_binding = _binding_file_record(binding, ("decision_plan", "plan", "selection_plan"), description="decision plan")
    if plan_binding is not None:
        _same_record(plan_record, plan_binding, root=root, description="truth decision plan")
    _check_hash_alias(binding, ("decision_plan_sha256", "selection_plan_sha256", "plan_sha256"), str(plan_record["sha256"]), description="decision plan")
    selection_binding = _binding_file_record(binding, ("source_selection", "selection"), description="source selection")
    if selection_binding is not None:
        _same_record(selection_record, selection_binding, root=root, description="truth source selection")
    _check_hash_alias(binding, ("source_selection_sha256", "selection_sha256"), str(selection_record["sha256"]), description="source selection")
    observed = binding.get("observation_sha256", binding.get("observations_sha256"))
    if not isinstance(observed, Mapping) or set(observed) != EXPECTED_OBSERVATION_KEYS:
        raise FreezeScoreError("truth observation bindings are incomplete")
    for value in observed.values():
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise FreezeScoreError("truth observation digest is malformed")
    observed_ids = binding.get("record_ids_sha256", binding.get("record_order_sha256"))
    expected_ids = {domain: _json_digest(list(values)) for domain, values in record_ids.items()}
    if observed_ids != expected_ids:
        raise FreezeScoreError("truth record-order digests differ from source selection")
    keys = binding.get("truth_tensor_keys")
    if not isinstance(keys, list) or set(keys) != set(EXPECTED_TRUTH_KEYS) or len(keys) != len(EXPECTED_TRUTH_KEYS):
        raise FreezeScoreError("truth tensor key binding is not the two-domain layout")
    # Labels must not be serialized into the binding metadata.  Public record
    # IDs and digests are fine; token/label payload fields are not.
    for forbidden in ("token_ids", "input_ids", "labels", "target_labels", "truth_tokens", "truth_labels"):
        if forbidden in binding:
            raise FreezeScoreError("private truth binding contains label payload")
    opaque = {
        "binding_manifest": binding_record,
        "truth_file": dict(truth_record),
        "truth_payload_read_before_gate": False,
    }
    return binding, opaque, {"path": str(binding_path), **binding_record}


def _validate_binding_against_public(
    binding: Mapping[str, Any],
    *,
    plan_record: Mapping[str, Any],
    selection_record: Mapping[str, Any],
    source_record_ids: Mapping[str, Sequence[str]],
    gate: Mapping[str, Any],
) -> None:
    observations = gate.get("observations")
    if not isinstance(observations, Mapping) or not isinstance(observations.get("cells"), Mapping):
        raise FreezeScoreError("public gate did not return observation bindings")
    expected_observations: dict[str, str] = {}
    for cell_id in freeze.CELL_ORDER:
        cell = observations["cells"].get(cell_id)
        if not isinstance(cell, Mapping) or not isinstance(cell.get("observation"), Mapping):
            raise FreezeScoreError(f"public gate observation is absent: {cell_id}")
        digest = cell["observation"].get("sha256")
        if not isinstance(digest, str):
            raise FreezeScoreError(f"public gate observation digest is absent: {cell_id}")
        expected_observations[cell_id] = digest
        if cell.get("record_ids_sha256") != _json_digest(list(source_record_ids[cell_id.split("__", 1)[0]])):
            raise FreezeScoreError(f"public source pairing differs from selection: {cell_id}")
    observed = binding.get("observation_sha256", binding.get("observations_sha256"))
    if dict(observed) != expected_observations:
        raise FreezeScoreError("truth observation bindings differ from the gated public matrix")
    # These aliases were checked before the gate too; repeat the hash-only
    # comparison after the gate to close a concurrent metadata replacement.
    _check_hash_alias(binding, ("decision_plan_sha256", "selection_plan_sha256", "plan_sha256"), str(plan_record["sha256"]), description="decision plan")
    _check_hash_alias(binding, ("source_selection_sha256", "selection_sha256"), str(selection_record["sha256"]), description="source selection")
    expected_ids = {domain: _json_digest(list(values)) for domain, values in source_record_ids.items()}
    if binding.get("record_ids_sha256", binding.get("record_order_sha256")) != expected_ids:
        raise FreezeScoreError("truth record-order digests changed after the public gate")


def _load_public_geometry(gate: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    observations = gate.get("observations")
    if not isinstance(observations, Mapping) or not isinstance(observations.get("cells"), Mapping):
        raise FreezeScoreError("public gate observation map is incomplete")
    masks: dict[str, torch.Tensor] = {}
    positions: dict[str, torch.Tensor] = {}
    count = int(gate["records_per_domain"])
    for cell_id in freeze.CELL_ORDER:
        row = observations["cells"].get(cell_id)
        if not isinstance(row, Mapping) or not isinstance(row.get("observation"), Mapping):
            raise FreezeScoreError(f"public observation is absent: {cell_id}")
        path = Path(str(row["observation"]["path"])).expanduser().resolve()
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                if set(handle.keys()) != {"activations", "attention_mask", "position_ids"}:
                    raise FreezeScoreError(f"public observation tensor keys changed: {cell_id}")
                mask = handle.get_tensor("attention_mask")
                pos = handle.get_tensor("position_ids")
        except FreezeScoreError:
            raise
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            raise FreezeScoreError(f"public observation could not be reopened: {cell_id}") from exc
        if tuple(mask.shape) != (count, contract.STORED_SEQUENCE_TOKENS) or tuple(pos.shape) != tuple(mask.shape):
            raise FreezeScoreError(f"public observation geometry changed: {cell_id}")
        mask = mask.to(dtype=torch.bool, device="cpu").contiguous()
        pos = pos.to(dtype=torch.int64, device="cpu").contiguous()
        if not bool(mask[:, 0].all().item()) or not bool(mask.all().item()):
            raise FreezeScoreError(f"public observation mask is not full 128-token geometry: {cell_id}")
        expected = torch.arange(contract.STORED_SEQUENCE_TOKENS, dtype=torch.int64).repeat(count, 1)
        if not torch.equal(pos, expected):
            raise FreezeScoreError(f"public observation positions changed: {cell_id}")
        masks[cell_id] = mask
        positions[cell_id] = pos
    return masks, positions


def _check_truth_tensor(value: torch.Tensor, *, count: int, domain: str) -> torch.Tensor:
    tensor = value.detach().cpu().contiguous()
    if tensor.dtype != torch.int64 or tuple(tensor.shape) != (count, contract.STORED_SEQUENCE_TOKENS):
        raise FreezeScoreError(f"private truth tensor geometry or dtype changed: {domain}")
    if not bool(torch.all(tensor[:, 0].eq(contract.BOS_TOKEN_ID)).item()):
        raise FreezeScoreError(f"private truth BOS changed: {domain}")
    if bool(torch.any(tensor.lt(0)).item()) or bool(torch.any(tensor.ge(contract.VOCAB_SIZE)).item()):
        raise FreezeScoreError(f"private truth ID range changed: {domain}")
    return tensor


def _check_sidecar_metadata(
    metadata: Mapping[str, str],
    binding: Mapping[str, Any],
    *,
    plan_record: Mapping[str, Any],
    selection_record: Mapping[str, Any],
    observations: Mapping[str, str],
    record_ids: Mapping[str, Sequence[str]],
) -> None:
    if metadata.get("schema") != TRUTH_SIDECAR_SCHEMA or metadata.get("task_id") != TASK_ID:
        raise FreezeScoreError("private truth sidecar schema/task metadata changed")
    if metadata.get("truth_opened") not in (None, "false", "False"):
        raise FreezeScoreError("private truth sidecar truth flag is open")
    for names, expected, description in (
        (("decision_plan_sha256", "selection_plan_sha256"), str(plan_record["sha256"]), "decision plan"),
        (("source_selection_sha256", "panel_sha256"), str(selection_record["sha256"]), "source selection"),
    ):
        present = [name for name in names if name in metadata]
        if present and any(metadata[name] != expected for name in present):
            raise FreezeScoreError(f"private truth sidecar {description} metadata changed")
    if "observation_sha256" in metadata:
        try:
            observed = json.loads(metadata["observation_sha256"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise FreezeScoreError("private truth sidecar observation metadata is invalid") from exc
        if observed != dict(observations):
            raise FreezeScoreError("private truth sidecar observation metadata changed")
    if "record_ids_sha256" in metadata:
        try:
            observed_ids = json.loads(metadata["record_ids_sha256"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise FreezeScoreError("private truth sidecar record-order metadata is invalid") from exc
        expected_ids = {domain: _json_digest(list(values)) for domain, values in record_ids.items()}
        if observed_ids != expected_ids:
            raise FreezeScoreError("private truth sidecar record-order metadata changed")


def _load_truth_after_gate(
    binding_path: Path,
    truth_path: Path,
    *,
    binding_before: Mapping[str, Any],
    opaque_before: Mapping[str, Any],
    root: Path,
    plan_record: Mapping[str, Any],
    selection_record: Mapping[str, Any],
    source_record_ids: Mapping[str, Sequence[str]],
    gate: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Verify and open the private sidecar once, after the public gate."""

    binding_now_record = _actual_record(binding_path, description="private truth binding")
    before_binding_record = opaque_before["binding_manifest"]
    if binding_now_record.get("sha256") != before_binding_record.get("sha256") or binding_now_record.get("bytes") != before_binding_record.get("bytes"):
        raise FreezeScoreError("private truth binding changed across the public gate")
    binding_now = _load_json(binding_path, description="private truth binding")
    if binding_now != dict(binding_before):
        raise FreezeScoreError("private truth binding metadata changed across the public gate")
    _validate_binding_against_public(
        binding_now,
        plan_record=plan_record,
        selection_record=selection_record,
        source_record_ids=source_record_ids,
        gate=gate,
    )
    declared_truth = binding_now.get("truth_file")
    if not isinstance(declared_truth, Mapping):
        declared_truth = binding_now.get("sidecar")
    if not isinstance(declared_truth, Mapping):
        raise FreezeScoreError("private truth artifact record disappeared")
    actual_truth = _actual_record(truth_path, description="private truth artifact")
    declared_path = Path(str(declared_truth["path"])).expanduser().resolve(strict=False)
    if declared_path != truth_path.expanduser().absolute().resolve(strict=False) or actual_truth["bytes"] != declared_truth.get("bytes") or actual_truth["sha256"] != declared_truth.get("sha256"):
        raise FreezeScoreError("private truth artifact bytes or digest changed")
    observations = {
        cell_id: str(gate["observations"]["cells"][cell_id]["observation"]["sha256"])
        for cell_id in freeze.CELL_ORDER
    }
    expected_keys = set(EXPECTED_TRUTH_KEYS)
    count = int(gate["records_per_domain"])
    labels_by_domain: dict[str, torch.Tensor] = {}
    try:
        # This is the sole safe_open call for the evaluator truth artifact.
        with safe_open(truth_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != expected_keys:
                raise FreezeScoreError("private truth sidecar tensor keys changed")
            metadata = dict(handle.metadata() or {})
            _check_sidecar_metadata(
                metadata,
                binding_now,
                plan_record=plan_record,
                selection_record=selection_record,
                observations=observations,
                record_ids=source_record_ids,
            )
            for domain in ("pile", "finance"):
                labels_by_domain[domain] = _check_truth_tensor(
                    handle.get_tensor(f"{domain}__token_ids"),
                    count=count,
                    domain=domain,
                )
    except FreezeScoreError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise FreezeScoreError("private truth sidecar could not be opened or validated") from exc
    tensor_digests = {
        f"{domain}__token_ids": contract.tensor_digest(labels_by_domain[domain])
        for domain in ("pile", "finance")
    }
    declared_digests = binding_now.get("truth_tensor_sha256", binding_now.get("tensor_sha256"))
    if declared_digests is not None and declared_digests != tensor_digests:
        raise FreezeScoreError("private truth tensor digest changed")
    truth = {
        cell_id: labels_by_domain[cell_id.split("__", 1)[0]]
        for cell_id in freeze.CELL_ORDER
    }
    return truth, {
        "status": "TRUTH_VERIFIED_AFTER_PUBLIC_GATE",
        "truth_file": actual_truth,
        "schema": TRUTH_SIDECAR_SCHEMA,
        "tensor_keys": sorted(expected_keys),
        "tensor_sha256": tensor_digests,
        "truth_payload_read_before_gate": False,
    }


def _task_output_path(value: Path | str, *, root: Path, description: str) -> Path:
    path = _canonical_path(value, root=root, description=description, require_file=False)
    task_root = (root / "experiments" / "TRR-0006").resolve()
    if not _inside(path, task_root):
        raise FreezeScoreError(f"{description} must be task-owned below {task_root}")
    if path.is_symlink():
        raise FreezeScoreError(f"{description} is a symlink: {path}")
    return path


def _ensure_create_only(paths: Sequence[Path]) -> None:
    if len(set(paths)) != len(paths):
        raise FreezeScoreError("main result paths must be distinct")
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FreezeScoreError(f"refusing to overwrite create-only result artifact: {path}")


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise FreezeScoreError(f"refusing to overwrite create-only artifact: {path}") from exc


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(value)
    except FileExistsError as exc:
        raise FreezeScoreError(f"refusing to overwrite create-only artifact: {path}") from exc


def _context_skeleton(plan: Mapping[str, Any], *, fixture_path: Path | None = None) -> dict[str, Any]:
    provenance = plan.get("provenance") if isinstance(plan.get("provenance"), Mapping) else {}
    limitations = plan.get("exclusion_limitations")
    if not isinstance(limitations, list):
        limitations = []
    fixture = {
        "status": "QUALIFICATION_FIXTURE_ONLY",
        "failure": None,
        "evidence_path": provenance.get("fixture_equivalence_path"),
        "evidence_sha256": provenance.get("fixture_equivalence_sha256"),
        "main_matrix_qualification_failure": None,
    }
    if fixture_path is not None and fixture_path.exists():
        fixture["qualification_summary"] = _actual_record(fixture_path, root=fixture_path.parents[2], description="qualification summary")
    return {
        "qualification": fixture,
        "exclusion_limitations": list(limitations),
        "historical_A2_gap": {
            "status": "SEPARATE_HISTORICAL_DENOMINATOR",
            "value_pp": None,
            "source": "TRR-0005 historical A1+A2 comparison; not recomputed in TRR-0006",
        },
        "costs": {
            "inherited_preparation_training_ratio": None,
            "inherited_preparation_training_status": "UNAVAILABLE_WITHOUT_NEW_TRAINING",
            "warmed_runtime_ratio": None,
            "note": "Quality inference is reported separately; missing historical preparation cost makes cost qualification unavailable.",
        },
    }


def _build_manifest(
    *,
    root: Path,
    result: Mapping[str, Any],
    plan_record: Mapping[str, Any],
    selection_record: Mapping[str, Any],
    registration_record: Mapping[str, Any],
    freeze_receipt_record: Mapping[str, Any],
    truth_binding_record: Mapping[str, Any],
    truth_verified: Mapping[str, Any],
    output_records: Mapping[str, Mapping[str, Any]],
    execution_binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "token-reconstruction.trr0006-analysis-manifest.v1",
        "task_id": TASK_ID,
        "status": "TRR6_ANALYSIS_COMPLETE_AFTER_PUBLIC_FREEZE",
        "code_commit": execution_binding["code_commit"],
        "execution_binding": dict(execution_binding),
        "decision_plan": dict(plan_record),
        "source_selection": dict(selection_record),
        "registration": dict(registration_record),
        "freeze_receipt": dict(freeze_receipt_record),
        "truth_binding": dict(truth_binding_record),
        "truth_verified_after_public_gate": dict(truth_verified),
        "outputs": {name: dict(record) for name, record in output_records.items()},
        "truth_opened": True,
        "private_truth_payload_persisted": False,
        "result_schema": result.get("schema"),
        "decision": result.get("decision"),
    }


def run(
    *,
    repository_root: Path,
    plan_path: Path,
    registration_path: Path,
    freeze_receipt_path: Path,
    truth_binding_path: Path,
    truth_path: Path,
    result_path: Path,
    report_path: Path,
    manifest_path: Path,
    execution_receipt_path: Path,
    source_selection_path: Path | None = None,
    observation_manifest_path: Path | None = None,
    runtime_ratio: float | None = None,
    training_ratio: float | None = None,
) -> dict[str, Any]:
    """Run the frozen public gate, one private truth load, and full score."""

    root = _root(repository_root)
    started_utc = _now()
    registration_file = _canonical_path(registration_path, root=root, description="prediction registration")
    registration = contract.load_registration(registration_file)
    execution_binding = _validate_execution_binding(registration, root=root)
    plan, plan_record = _validate_plan_binding(registration, plan_path, root=root)
    selection, selection_record, source_record_ids = _validate_selection_binding(
        registration,
        source_selection_path,
        root=root,
    )
    del selection  # The IDs and file digest are the only selection inputs used here.
    truth_binding, opaque_before, binding_record_before = _truth_header(
        truth_binding_path,
        truth_path,
        root=root,
        plan_record=plan_record,
        selection_record=selection_record,
        record_ids=source_record_ids,
    )
    result_file = _task_output_path(result_path, root=root, description="result")
    report_file = _task_output_path(report_path, root=root, description="report")
    manifest_file = _task_output_path(manifest_path, root=root, description="analysis manifest")
    execution_file = _task_output_path(execution_receipt_path, root=root, description="execution receipt")
    _ensure_create_only((result_file, report_file, manifest_file, execution_file))

    # The imported gate is the actual public matrix implementation.  It reads
    # only registered public files, and its truth_path argument performs a
    # containment check without opening or hashing evaluator truth.
    gate = freeze.validate_before_truth(
        repository_root=root,
        registration_path=registration_file,
        receipt_path=freeze_receipt_path,
        observation_manifest_path=observation_manifest_path,
        plan_path=plan_path,
        truth_path=truth_path,
    )
    if gate.get("verified_before_truth") is not True or gate.get("truth_opened") is not False:
        raise FreezeScoreError("public gate did not return a complete closed receipt")

    # Close the small metadata race between the pre-gate checks and the public
    # gate.  No truth bytes are inspected by these rechecks.
    _, plan_record_after = _validate_plan_binding(registration, plan_path, root=root)
    _, selection_record_after, source_record_ids_after = _validate_selection_binding(
        registration,
        source_selection_path,
        root=root,
    )
    if plan_record_after != plan_record or selection_record_after != selection_record or source_record_ids_after != source_record_ids:
        raise FreezeScoreError("plan or source selection changed across the public gate")
    binding_record_after = _actual_record(_canonical_path(truth_binding_path, description="private truth binding"), description="private truth binding")
    if binding_record_after.get("sha256") != binding_record_before.get("sha256") or binding_record_after.get("bytes") != binding_record_before.get("bytes"):
        raise FreezeScoreError("private truth binding changed across the public gate")
    _validate_binding_against_public(
        truth_binding,
        plan_record=plan_record,
        selection_record=selection_record,
        source_record_ids=source_record_ids,
        gate=gate,
    )

    masks, positions = _load_public_geometry(gate)
    truth, truth_verified = _load_truth_after_gate(
        _canonical_path(truth_binding_path, description="private truth binding"),
        Path(truth_path).expanduser().absolute().resolve(strict=False),
        binding_before=truth_binding,
        opaque_before=opaque_before,
        root=root,
        plan_record=plan_record,
        selection_record=selection_record,
        source_record_ids=source_record_ids,
        gate=gate,
    )
    scored = scorer.score_matrix(
        predictions=gate["prediction_tensors"],
        truth=truth,
        attention_masks=masks,
        record_ids={
            cell_id: source_record_ids[cell_id.split("__", 1)[0]]
            for cell_id in freeze.CELL_ORDER
        },
        position_ids=positions,
        runtime_ratio=runtime_ratio,
        training_ratio=training_ratio,
    )
    report = scorer.render_report(scored)
    context = _context_skeleton(plan)
    result = {
        "schema": "token-reconstruction.trr0006-final-result.v1",
        "task_id": TASK_ID,
        "status": "TRR6_ANALYSIS_COMPLETE_AFTER_PUBLIC_FREEZE",
        "decision": scored["decision"],
        "claim_scope": scored["claim_scope"],
        "matrix": scored["matrix"],
        "bootstrap": scored["bootstrap"],
        "exact_uncertainty": scored["exact_uncertainty"],
        "cells": scored["cells"],
        "method_scores": scored["method_scores"],
        "provenance": {
            "code_commit": execution_binding["code_commit"],
            "decision_plan": dict(plan_record),
            "source_selection": dict(selection_record),
            "registration": _actual_record(registration_file, root=root, description="prediction registration"),
            "freeze_receipt": _actual_record(_canonical_path(freeze_receipt_path, root=root, description="freeze receipt"), root=root, description="freeze receipt"),
            "truth_binding": dict(binding_record_before),
            "truth_file": dict(truth_verified["truth_file"]),
        },
        **context,
    }
    report += "\n\n## Qualification, limitations, and historical context\n\n"
    report += "Main-matrix qualification failure: **not recorded in this result skeleton**. The retained fixture qualification is separate and does not establish the main truth result.\n\n"
    report += "Exclusion limitations: " + "; ".join(context["exclusion_limitations"]) + "\n\n"
    report += "Historical A2 gap: retained as a separate historical denominator; TRR-0006 does not recompute it.\n\n"
    report += "Inherited preparation cost: unavailable without new training; quality inference and cost qualification remain separate.\n"

    # Build the result/report first, then bind their create-only records into the
    # manifest and receipt.  The receipt itself is intentionally not
    # self-hashed because a self-record would be circular.
    _write_json_exclusive(result_file, result)
    _write_text_exclusive(report_file, report)
    result_record = _actual_record(result_file, root=root, description="result")
    report_record = _actual_record(report_file, root=root, description="report")
    gate_record = _actual_record(_canonical_path(freeze_receipt_path, root=root, description="freeze receipt"), root=root, description="freeze receipt")
    registration_record = _actual_record(registration_file, root=root, description="prediction registration")
    outputs_for_manifest = {"result": result_record, "report": report_record}
    manifest = _build_manifest(
        root=root,
        result=result,
        plan_record=plan_record,
        selection_record=selection_record,
        registration_record=registration_record,
        freeze_receipt_record=gate_record,
        truth_binding_record=binding_record_before,
        truth_verified=truth_verified,
        output_records=outputs_for_manifest,
        execution_binding=execution_binding,
    )
    _write_json_exclusive(manifest_file, manifest)
    manifest_record = _actual_record(manifest_file, root=root, description="analysis manifest")
    execution_receipt = {
        "schema": "token-reconstruction.trr0006-execution-receipt.v1",
        "task_id": TASK_ID,
        "status": "TRR6_EXECUTED_SCORED_AFTER_PUBLIC_FREEZE",
        "started_utc": started_utc,
        "ended_utc": _now(),
        "code_commit": execution_binding["code_commit"],
        "execution_binding": execution_binding,
        "public_gate": {
            "status": gate["status"],
            "verified_before_truth": True,
            "truth_opened": False,
            "entry_count": gate["entry_count"],
            "assets_rehashed": gate["assets_rehashed"],
            "freeze_receipt": gate_record,
        },
        "truth_binding_recorded_before_gate": dict(opaque_before),
        "truth_verified_after_public_gate": dict(truth_verified),
        "outputs": {
            "result": result_record,
            "report": report_record,
            "manifest": manifest_record,
        },
        "truth_opened_once": True,
        "private_truth_payload_persisted": False,
        "decision": result["decision"],
    }
    _write_json_exclusive(execution_file, execution_receipt)
    return {
        "task_id": TASK_ID,
        "status": execution_receipt["status"],
        "result": result_record,
        "report": report_record,
        "manifest": manifest_record,
        "execution_receipt": _actual_record(execution_file, root=root, description="execution receipt"),
        "truth_opened_once": True,
        "truth_opened_before_public_gate": False,
        "decision": result["decision"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--source-selection", type=Path)
    parser.add_argument("--observation-manifest", type=Path)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--truth-binding", type=Path, required=True)
    parser.add_argument("--truth-path", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--runtime-ratio", type=float)
    parser.add_argument("--training-ratio", type=float)
    args = parser.parse_args(argv)
    try:
        result = run(
            repository_root=args.repository_root,
            plan_path=args.plan,
            registration_path=args.registration,
            source_selection_path=args.source_selection,
            observation_manifest_path=args.observation_manifest,
            freeze_receipt_path=args.freeze_receipt,
            truth_binding_path=args.truth_binding,
            truth_path=args.truth_path,
            result_path=args.result,
            report_path=args.report,
            manifest_path=args.manifest,
            execution_receipt_path=args.execution_receipt,
            runtime_ratio=args.runtime_ratio,
            training_ratio=args.training_ratio,
        )
    except (FreezeScoreError, contract.ContractError, freeze.FreezePairError, scorer.PairScoreError) as exc:
        print(f"TRR-0006 freeze/score failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
