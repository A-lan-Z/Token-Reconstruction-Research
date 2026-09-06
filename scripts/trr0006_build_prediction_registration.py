#!/usr/bin/env python3
"""Build the frozen TRR-0006 main prediction registration.

This builder accepts only the already-frozen decision plan, completed public
source selection, and a producer observation manifest.  It derives the
1,536-record count from the exact plan, verifies the plan and selection
digests and geometry, validates every source-free observation binding, records
the executable HEAD and code hashes, and writes a create-only registration.
It never opens truth, reads source text/token IDs, trains a model, or creates
observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from scripts import trr0006_prediction_contract as contract

EXPECTED_PLAN_PATH = "experiments/TRR-0006/decision_plan.json"
EXPECTED_PLAN_BYTES = 15768
EXPECTED_PLAN_SHA256 = "edeb1bb05a00ad3f580d415f1b9ab632f588b3c50fd1ed9c2f2e055737c766ea"
EXPECTED_RECORDS_PER_DOMAIN = 1536
EXPECTED_UNIQUE_SOURCES_TOTAL = 3072
EXPECTED_RECORD_CONDITION_EVALUATIONS_PER_METHOD = 6144
MAIN_MAX_SECONDS = 1800
NO_UNIVERSAL_EQUIVALENCE_CLAIM = "No universal/native equivalence claim; scoped fixture equivalence passed"
EXPECTED_SELECTION_PATH = "experiments/TRR-0006/source_selection.json"
EXPECTED_SELECTION_BYTES = 2135542
EXPECTED_SELECTION_SHA256 = "75909aaf0f9e40176c197d86c09651097010a11519855f1db3dc50fe5e754f43"
EXECUTION_BINDING_SCHEMA = "token-reconstruction.trr0006-scoring-execution-binding.v1"
EXECUTION_BINDING_SPECS = (
    ("scoring_driver", "scripts/trr0006_freeze_score.py"),
    ("public_freeze_gate", "scripts/trr0006_freeze_pair.py"),
    ("pair_scorer", "scripts/trr0006_score_pair.py"),
)


class RegistrationBuildError(contract.ContractError):
    """Raised when the main registration cannot be constructed fail-closed."""


def _root(value: Path) -> Path:
    path = value.expanduser().resolve()
    if path.is_symlink() or not path.is_dir():
        raise RegistrationBuildError(f"repository root is unavailable: {path}")
    return path


def _resolve_file(value: Path | str, *, root: Path, description: str) -> Path:
    try:
        return contract.resolve_path(str(value), repository_root=root, description=description)
    except contract.ContractError as exc:
        raise RegistrationBuildError(str(exc)) from exc


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise RegistrationBuildError(f"asset is unavailable: {path}")
    display = str(path)
    if root is not None:
        try:
            display = path.relative_to(root).as_posix()
        except ValueError:
            pass
    return {"path": display, "bytes": int(path.stat().st_size), "sha256": contract.sha256_file(path)}


def _git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RegistrationBuildError("cannot resolve executable HEAD") from exc
    value = result.stdout.strip()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise RegistrationBuildError("executable HEAD is not a full lowercase commit hash")
    return value


def _load_plan(path: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = _resolve_file(path, root=root, description="TRR-0006 decision plan")
    record = _record(resolved, root=root)
    if record["path"] != EXPECTED_PLAN_PATH or record["bytes"] != EXPECTED_PLAN_BYTES or record["sha256"] != EXPECTED_PLAN_SHA256:
        raise RegistrationBuildError("decision plan is not the frozen TRR-0006 plan")
    plan = contract.load_json(resolved, description="TRR-0006 decision plan")
    if plan.get("schema") != "token-reconstruction.trr0006-decision-plan.v1" or plan.get("task_id") != contract.TASK_ID or plan.get("status") != "FROZEN_BEFORE_NEW_SOURCE_SELECTION":
        raise RegistrationBuildError("decision plan is not frozen before source selection")
    panel = plan.get("panel")
    comparison = plan.get("comparison")
    provenance = plan.get("provenance")
    if not isinstance(panel, dict) or not isinstance(comparison, dict) or not isinstance(provenance, dict):
        raise RegistrationBuildError("decision plan is missing panel/comparison/provenance")
    exact_panel = {
        "records_per_domain": EXPECTED_RECORDS_PER_DOMAIN,
        "unique_sources_total": EXPECTED_UNIQUE_SOURCES_TOTAL,
        "record_condition_evaluations_per_method": EXPECTED_RECORD_CONDITION_EVALUATIONS_PER_METHOD,
        "clip_tokens_including_bos": contract.STORED_SEQUENCE_TOKENS,
        "scored_post_bos_tokens": contract.SCORED_POST_BOS_TOKENS,
        "capture_tokens": contract.CAPTURE_SEQUENCE_TOKENS,
        "capture_batch_records": contract.CAPTURE_BATCH_RECORDS,
        "prediction_chunk_records": contract.CAPTURE_BATCH_RECORDS,
        "selection_seed": 5005,
    }
    for key, expected in exact_panel.items():
        if panel.get(key) != expected:
            raise RegistrationBuildError(f"decision plan panel changed: {key}")
    if comparison.get("method_order") != list(contract.METHOD_IDS) or comparison.get("cells") != list(contract.CELL_ORDER):
        raise RegistrationBuildError("decision plan method or cell order changed")
    if provenance.get("scientific_source_commit") != contract.SCIENTIFIC_SOURCE_COMMIT:
        raise RegistrationBuildError("decision plan scientific source commit changed")
    return plan, record


def _load_source_selection(path: Path, *, root: Path) -> dict[str, Any]:
    """Bind the completed source-free selection without reading source rows."""

    resolved = _resolve_file(path, root=root, description="TRR-0006 source selection")
    record = _record(resolved, root=root)
    if (
        record["path"] != EXPECTED_SELECTION_PATH
        or record["bytes"] != EXPECTED_SELECTION_BYTES
        or record["sha256"] != EXPECTED_SELECTION_SHA256
    ):
        raise RegistrationBuildError("source selection is not the completed frozen TRR-0006 selection")
    selection = contract.load_json(resolved, description="TRR-0006 source selection")
    if (
        selection.get("schema") != "token-reconstruction.trr0006-source-selection.v1"
        or selection.get("task_id") != contract.TASK_ID
        or selection.get("status") != "FROZEN_TRR0006_SOURCE_SELECTION_NO_TRUTH"
        or selection.get("records_per_domain") != EXPECTED_RECORDS_PER_DOMAIN
        or selection.get("paired_conditions") is not True
        or selection.get("truth_opened") is True
    ):
        raise RegistrationBuildError("source selection is not a closed 1536-record TRR-0006 selection")
    rule = selection.get("selection_rule")
    if not isinstance(rule, dict) or rule.get("source_text_or_token_ids_written") is not False:
        raise RegistrationBuildError("source selection is not source-free")
    digests = rule.get("record_ids_sha256")
    if not isinstance(digests, dict) or set(digests) != {"pile", "finance"}:
        raise RegistrationBuildError("source selection record-order digests are incomplete")
    return {"record": record, "record_ids_sha256": dict(digests)}


def _load_observation_manifest(
    path: Path,
    *,
    root: Path,
    registration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    resolved = _resolve_file(path, root=root, description="producer observation manifest")
    record = _record(resolved, root=root)
    manifest = contract.load_json(resolved, description="producer observation manifest")
    # The producer reads public text to generate observations.  Its artifact
    # contract binds what was written, rather than whether it read that text.
    for key in ("truth_opened", "source_text_written", "token_ids_written", "target_labels_loaded"):
        if manifest.get(key) is not False:
            raise RegistrationBuildError(f"producer observation manifest private flag changed: {key}")
    if manifest.get("public_material_only") is not True:
        raise RegistrationBuildError("producer observation manifest is not public material only")
    try:
        parsed = contract.validate_observation_manifest(
            manifest,
            registration=registration,
            repository_root=root,
            verify_assets=True,
        )
    except contract.ContractError as exc:
        raise RegistrationBuildError(str(exc)) from exc
    return manifest, parsed, record


def _output_root(value: Path, *, root: Path) -> str:
    raw = value.expanduser()
    candidate = (raw if raw.is_absolute() else root / raw).resolve()
    task_root = (root / "experiments/TRR-0006").resolve()
    try:
        candidate.relative_to(task_root)
    except ValueError as exc:
        raise RegistrationBuildError(f"main output root must be task-owned below {task_root}") from exc
    if candidate.is_symlink():
        raise RegistrationBuildError(f"main output root is a symlink: {candidate}")
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError as exc:  # pragma: no cover - guarded above
        raise RegistrationBuildError("main output root escaped repository") from exc


def build(
    *,
    repository_root: Path,
    plan_path: Path,
    source_selection_path: Path,
    observation_manifest_path: Path,
    normalized_public_e_path: Path,
    output_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    root = _root(repository_root)
    plan, plan_record = _load_plan(plan_path, root=root)
    selection = _load_source_selection(source_selection_path, root=root)
    observation_path = _resolve_file(observation_manifest_path, root=root, description="producer observation manifest")
    observation_record = _record(observation_path, root=root)
    e_path = _resolve_file(normalized_public_e_path, root=root, description="normalized public E")
    e_record = _record(e_path)
    if e_record["bytes"] != contract.NORMALIZED_PUBLIC_E_BYTES or e_record["sha256"] != contract.NORMALIZED_PUBLIC_E_SHA256:
        raise RegistrationBuildError("normalized public E is not the retained shared table")
    # This skeleton is complete enough for the source-free manifest validator;
    # all method/code/runtime fields are filled before the final self-check.
    skeleton = {"records_per_domain": EXPECTED_RECORDS_PER_DOMAIN}
    try:
        manifest = contract.load_json(observation_path, description="producer observation manifest")
        _, parsed_observations, _ = _load_observation_manifest(
            observation_path,
            root=root,
            registration=skeleton,
        )
    except contract.ContractError as exc:
        raise RegistrationBuildError(str(exc)) from exc
    if parsed_observations["records_per_domain"] != EXPECTED_RECORDS_PER_DOMAIN:
        raise RegistrationBuildError("producer observation count is not the frozen 1536 per domain")

    code_bindings: list[dict[str, Any]] = []
    for role, relative_path in contract.CODE_BINDING_SPECS:
        source_path = _resolve_file(relative_path, root=root, description=f"required code {role}")
        code_bindings.append({"role": role, **_record(source_path, root=root)})
    execution_bindings: list[dict[str, Any]] = []
    for role, relative_path in EXECUTION_BINDING_SPECS:
        source_path = _resolve_file(relative_path, root=root, description=f"required scoring code {role}")
        execution_bindings.append({"role": role, **_record(source_path, root=root)})
    methods: dict[str, Any] = {}
    for method_id in contract.METHOD_IDS:
        state = contract.PUBLISHED_STATE_BINDINGS[method_id]
        methods[method_id] = {
            "base_method_id": contract.BASE_METHOD_IDS[method_id],
            "decision_rule": contract.METHOD_RULES[method_id],
            "state": {
                "path": state["path"],
                "bytes": state["bytes"],
                "sha256": state["sha256"],
                "source_commit": state["source_commit"],
            },
        }
    current_head = _git_head(root)
    output_registration = output_path.expanduser()
    if not output_registration.is_absolute():
        output_registration = root / output_registration
    output_registration = output_registration.resolve()
    output_root_value = _output_root(output_root, root=root)
    registration = {
        "schema": contract.REGISTRATION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_PREDICTION_REGISTRATION",
        "qualification_only": False,
        "code_commit": current_head,
        "code_bindings": code_bindings,
        "execution_binding": {
            "schema": EXECUTION_BINDING_SCHEMA,
            "code_commit": current_head,
            "files": execution_bindings,
        },
        "records_per_domain": EXPECTED_RECORDS_PER_DOMAIN,
        "cell_order": list(contract.CELL_ORDER),
        "method_ids": list(contract.METHOD_IDS),
        "geometry": {
            "capture_batch_records": contract.CAPTURE_BATCH_RECORDS,
            "capture_sequence_tokens": contract.CAPTURE_SEQUENCE_TOKENS,
            "stored_sequence_tokens": contract.STORED_SEQUENCE_TOKENS,
            "scored_sequence_tokens": contract.SCORED_SEQUENCE_TOKENS,
            "scored_post_bos_tokens": contract.SCORED_POST_BOS_TOKENS,
            "hidden_size": contract.HIDDEN_SIZE,
            "chunk_records": contract.CAPTURE_BATCH_RECORDS,
        },
        "runtime_assets": {
            "normalized_public_E": {
                **e_record,
                "shape": [contract.VOCAB_SIZE, contract.HIDDEN_SIZE],
                "dtype": "torch.float32",
            }
        },
        "methods": methods,
        "observation_manifest": observation_record,
        "source_selection": selection["record"],
        "source_record_ids_sha256": selection["record_ids_sha256"],
        "output_root": output_root_value,
        "decision_plan": plan_record,
        "decision_plan_sha256": EXPECTED_PLAN_SHA256,
        "timing_contract": {
            "warmup_runs_per_record": 1,
            "measured_runs_per_record": 1,
            "repeat_integrity": "Require warmup and measured predicted IDs to match exactly",
        },
        "resource_guard": {
            "minimum_free_gpu_bytes": contract.MIN_FREE_GPU_BYTES,
            "maximum_reserved_gpu_bytes": contract.MAX_RESERVED_GPU_BYTES,
            "maximum_rss_bytes": contract.MAX_RSS_BYTES,
            "minimum_host_available_bytes": contract.MIN_HOST_AVAILABLE_BYTES,
            "maximum_seconds": MAIN_MAX_SECONDS,
        },
        "numerical_settings": dict(contract.NUMERICAL_SETTINGS),
        "source_lineage": {
            "scientific_source_commit": contract.SCIENTIFIC_SOURCE_COMMIT,
            "published_parent_commit": contract.PUBLISHED_PARENT_COMMIT,
            "post_score_maintenance_commit": contract.POST_SCORE_MAINTENANCE_COMMIT,
            "maintenance_inference_equivalence": NO_UNIVERSAL_EQUIVALENCE_CLAIM,
        },
        "truth_opened": False,
        "candidate_arrays_persisted": False,
    }
    try:
        contract.validate_registration(registration)
    except contract.ContractError as exc:
        raise RegistrationBuildError(str(exc)) from exc
    contract.write_create_only(output_registration, registration)
    return {
        "schema": "token-reconstruction.trr0006-main-registration-builder.v1",
        "task_id": contract.TASK_ID,
        "status": "MAIN_REGISTRATION_CREATED_NO_TRUTH",
        "records_per_domain": EXPECTED_RECORDS_PER_DOMAIN,
        "decision_plan": plan_record,
        "observation_manifest": observation_record,
        "source_selection": selection["record"],
        "registration": _record(output_registration, root=root),
        "code_commit": current_head,
        "observations_validated": True,
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-selection", type=Path, required=True)
    parser.add_argument("--observation-manifest", type=Path, required=True)
    parser.add_argument("--normalized-public-E", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        repository_root=args.repository_root,
        plan_path=args.plan,
        source_selection_path=args.source_selection,
        observation_manifest_path=args.observation_manifest,
        normalized_public_e_path=args.normalized_public_E,
        output_path=args.output,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
