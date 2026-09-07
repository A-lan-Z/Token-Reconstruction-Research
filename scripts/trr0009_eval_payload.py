"""Assemble the create-only TRR-0009 registration payload from frozen receipts.

This helper reads only truth-free method, support, planning, selection, capture,
and loader-equivalence metadata.  It never loads model tensors, source rows, or
truth.  The payload is written only after every required binding is present and
hash-verified; registration itself remains a separate create-only command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts import trr0009_eval_contract as contract


class PayloadError(contract.ContractError):
    pass


METHOD_FREEZE_DEFAULT = Path("experiments/TRR-0009/training/method_freeze.json")
COMPLETION_DEFAULT = Path("experiments/TRR-0009/training/run_v1/completion_receipt.json")
PLAN_DEFAULT = Path("experiments/TRR-0009/planning/decision_contract.json")
SELECTION_DEFAULT = Path("experiments/TRR-0009/selection/source_selection.json")
CAPTURE_DEFAULT = Path("experiments/TRR-0009/evaluation/public_observations/capture.json")
OBSERVATIONS_DEFAULT = Path("experiments/TRR-0009/evaluation/public_observations/observations.json")
EQUIVALENCE_DEFAULT = Path("experiments/TRR-0009/evaluation/loader_qualification_v2/qualification.json")
FREQUENCY_DEFAULT = Path("experiments/TRR-0009/evaluation/frequency_reference_v1.json")
TIMING_PLAN_DEFAULT = Path("experiments/TRR-0009/evaluation/timing/plan.json")
EMBEDDING_DEFAULT = Path("/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors")
OUTPUT_ROOT_DEFAULT = "experiments/TRR-0009/evaluation/predictions_v1"
PAYLOAD_DEFAULT = Path("experiments/TRR-0009/evaluation/registration_payload_v1.json")

EVALUATOR_CODE_PATHS = (
    "scripts/trr0009_eval_contract.py",
    "scripts/trr0009_eval_capture.py",
    "scripts/trr0009_eval_register.py",
    "scripts/trr0009_eval_runner.py",
    "scripts/trr0009_eval_gate.py",
    "scripts/trr0009_eval_truth.py",
    "scripts/trr0009_score.py",
    "scripts/trr0009_timing.py",
    "scripts/trr0009_model.py",
)


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise PayloadError(f"repository root is unavailable: {root}")
    return root


def _resolve(value: Path | str, *, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _load(path: Path, *, description: str) -> dict[str, Any]:
    try:
        return contract.load_json(path, description=description)
    except contract.ContractError as exc:
        raise PayloadError(str(exc)) from exc


def _record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    path = _resolve(path, root=root)
    try:
        return contract.validate_file_record(
            {"path": str(path), "bytes": path.stat().st_size, "sha256": contract.sha256_file(path)},
            repository_root=root,
            description=description,
            verify=True,
        )
    except (OSError, contract.ContractError) as exc:
        raise PayloadError(str(exc)) from exc


def _same_record(actual: Mapping[str, Any], expected: Mapping[str, Any], *, description: str) -> None:
    for key in ("path", "bytes", "sha256"):
        if str(actual.get(key)) != str(expected.get(key)):
            raise PayloadError(f"{description} {key} binding changed")


def _require_truth_free(value: Mapping[str, Any], *, description: str) -> None:
    forbidden = (
        "truth_opened",
        "source_text_loaded",
        "source_text_written",
        "target_labels_loaded",
        "token_ids_written",
        "candidate_arrays_persisted",
        "private_or_truth_payload_read",
        "fresh_evaluation_started",
    )
    if any(value.get(key) is True for key in forbidden):
        raise PayloadError(f"{description} records forbidden access")


def _path_value(value: Any, *, root: Path, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise PayloadError(f"{description} path is absent")
    path = _resolve(value, root=root)
    if not path.is_file() or path.is_symlink():
        raise PayloadError(f"{description} is unavailable: {path}")
    return str(path)


def _fit_cost(
    *, method_id: str, completion: Mapping[str, Any], completion_record: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    if method_id in (contract.UNCHANGED_METHOD_ID, contract.PUBLISHED_REFERENCE_METHOD_ID):
        status = "NOT_FIT_PUBLISHED_ANCHOR" if method_id == contract.UNCHANGED_METHOD_ID else "RETAINED_PUBLISHED_CONTEXT_NO_REFIT"
        return {"status": status, "source_receipt": dict(completion_record)}
    arms = completion.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get(method_id), Mapping):
        raise PayloadError(f"completion receipt lacks fit arm: {method_id}")
    arm = arms[method_id]
    costs = arm.get("costs")
    if not isinstance(costs, Mapping):
        raise PayloadError(f"completion receipt lacks fit costs: {method_id}")
    return {
        "status": "TRAINING_COMPLETE_SELECTED_STEP",
        "arm": method_id,
        "selected_step": int(arm.get("selected_step")),
        "costs": dict(costs),
        "source_receipt": dict(completion_record),
        "selected_state": dict(arm.get("state", {})),
    }


def _deployment_cost(
    *,
    method_id: str,
    state: Mapping[str, Any],
    state_metadata: Mapping[str, Any] | None,
    loader: Mapping[str, Any],
) -> dict[str, Any]:
    # The verified file record deliberately contains only path/bytes/SHA.  The
    # frozen method row carries state semantics separately in ``state_metadata``;
    # do not silently turn a missing supported-token count into zero.
    metadata = state_metadata if isinstance(state_metadata, Mapping) else {}
    supported_token_count: int | None = None
    if method_id == contract.ADAPTABLE_METHOD_ID:
        try:
            supported_token_count = int(metadata["supported_token_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PayloadError("adaptable state metadata lacks supported_token_count") from exc
        if supported_token_count <= 0:
            raise PayloadError("adaptable supported_token_count must be positive")
    return {
        "status": "DIRECT_POST_LOGIT_PATH_STATIC_FOOTPRINT",
        "inference_path": "current_H_only_full_vocabulary_direct_post_logit",
        "materialization_performed": False,
        "materialized_dictionary_path": False,
        "state_bytes": int(state.get("bytes")),
        "state_sha256": str(state.get("sha256")),
        "loader_interface": str(loader.get("interface")),
        "supported_token_count": supported_token_count,
        "runtime_model_startup_reported_in_run_manifest": True,
    }


def _method_rows(*, freeze: Mapping[str, Any], completion: Mapping[str, Any], completion_record: Mapping[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    bindings = freeze.get("state_bindings")
    if not isinstance(bindings, Mapping):
        raise PayloadError("method freeze state bindings are absent")
    rows: dict[str, dict[str, Any]] = {}
    for evaluator_id in contract.METHOD_ORDER:
        freeze_id = "retained_reference" if evaluator_id == contract.PUBLISHED_REFERENCE_METHOD_ID else evaluator_id
        binding = bindings.get(freeze_id)
        if not isinstance(binding, Mapping):
            raise PayloadError(f"method freeze row is absent: {freeze_id}")
        state = binding.get("state")
        loader = binding.get("loader")
        if not isinstance(state, Mapping) or not isinstance(loader, Mapping):
            raise PayloadError(f"method freeze state/loader is malformed: {freeze_id}")
        state_path = _path_value(state.get("path"), root=root, description=f"{freeze_id} state")
        actual_state = _record(Path(state_path), root=root, description=f"{freeze_id} state")
        if int(state.get("bytes")) != actual_state["bytes"] or str(state.get("sha256")) != actual_state["sha256"]:
            raise PayloadError(f"method freeze state record changed: {freeze_id}")
        rows[evaluator_id] = {
            "id": evaluator_id,
            "role": contract.METHOD_ROLES[evaluator_id],
            "kind": "decoder",
            "loader": dict(loader),
            "state_path": state_path,
            "state_sha256": actual_state["sha256"],
            # ``retained_reference`` is the freeze's binding key, while the
            # evaluator and completion contract call this method
            # ``published_reference``.  Keep those namespaces separate:
            # dispatch fit costs by evaluator id and state files by freeze id.
            "fit_cost": _fit_cost(method_id=evaluator_id, completion=completion, completion_record=completion_record, root=root),
            "deployment_cost": _deployment_cost(
                method_id=evaluator_id,
                state=actual_state,
                state_metadata=binding.get("state_metadata"),
                loader=loader,
            ),
        }
    return rows


def _code_bindings(*, freeze: Mapping[str, Any], root: Path) -> list[str]:
    paths: list[str] = []
    for row in freeze.get("source_code", []):
        if isinstance(row, Mapping) and isinstance(row.get("path"), str):
            paths.append(_path_value(row["path"], root=root, description="method-freeze source code"))
    paths.extend(_path_value(path, root=root, description="evaluator source") for path in EVALUATOR_CODE_PATHS)
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def build_payload(
    *,
    repository_root: Path,
    method_freeze_path: Path = METHOD_FREEZE_DEFAULT,
    completion_path: Path = COMPLETION_DEFAULT,
    plan_path: Path = PLAN_DEFAULT,
    selection_path: Path = SELECTION_DEFAULT,
    capture_path: Path = CAPTURE_DEFAULT,
    observation_manifest_path: Path = OBSERVATIONS_DEFAULT,
    equivalence_path: Path = EQUIVALENCE_DEFAULT,
    frequency_reference_path: Path = FREQUENCY_DEFAULT,
    timing_plan_path: Path = TIMING_PLAN_DEFAULT,
    embedding_path: Path = EMBEDDING_DEFAULT,
    output_root: str = OUTPUT_ROOT_DEFAULT,
) -> dict[str, Any]:
    root = _root(repository_root)
    freeze = _load(_resolve(method_freeze_path, root=root), description="TRR-0009 method freeze")
    if freeze.get("schema") != "token-reconstruction.trr0009-selected-method-freeze.v1" or freeze.get("task_id") != contract.TASK_ID or freeze.get("status") != "FROZEN_TRR0009_METHODS_BEFORE_SOURCE_SELECTION" or freeze.get("state_selection_frozen") is not True:
        raise PayloadError("method freeze is not the approved preselection freeze")
    _require_truth_free(freeze, description="method freeze")
    completion = _load(_resolve(completion_path, root=root), description="TRR-0009 training completion receipt")
    if completion.get("task_id") != contract.TASK_ID or completion.get("status") != "FULL_MATRIX_COMPLETE_PUBLIC_FIT_ONLY":
        raise PayloadError("training completion receipt is not complete")
    _require_truth_free(completion.get("verification", {}), description="training completion verification")
    completion_record = _record(_resolve(completion_path, root=root), root=root, description="training completion receipt")

    selection_path = _resolve(selection_path, root=root)
    selection = _load(selection_path, description="TRR-0009 source selection")
    if selection.get("schema") != "token-reconstruction.trr0009-source-selection.v1" or selection.get("task_id") != contract.TASK_ID or selection.get("status") != "FROZEN_TRR0009_SOURCE_SELECTION_NO_TRUTH":
        raise PayloadError("source selection is not the frozen TRR-0009 identity ledger")
    _require_truth_free(selection, description="source selection")
    selection_record = _record(selection_path, root=root, description="source selection")
    counts = selection.get("records_by_domain")
    if counts != {"finance": 256, "pile": 128}:
        raise PayloadError(f"source selection counts differ from frozen panel: {counts!r}")

    capture_path = _resolve(capture_path, root=root)
    capture = _load(capture_path, description="TRR-0009 public capture receipt")
    if capture.get("schema") != "token-reconstruction.trr0009-public-capture.v1" or capture.get("task_id") != contract.TASK_ID or capture.get("status") != "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH":
        raise PayloadError("public capture is not complete")
    _require_truth_free(capture, description="public capture")
    capture_record = _record(capture_path, root=root, description="public capture receipt")
    if capture.get("selection_plan") != selection_record:
        raise PayloadError("capture selection binding changed")
    observations_path = _resolve(observation_manifest_path, root=root)
    observations = _load(observations_path, description="TRR-0009 observation manifest")
    if observations.get("schema") != contract.OBSERVATION_SCHEMA or observations.get("task_id") != contract.TASK_ID:
        raise PayloadError("observation manifest schema changed")
    _require_truth_free(observations, description="observation manifest")
    observation_record = _record(observations_path, root=root, description="observation manifest")
    capture_observation = capture.get("observations")
    if isinstance(capture_observation, Mapping) and capture_observation.get("path"):
        _same_record(capture_observation, observation_record, description="capture observation")

    equivalence = _load(_resolve(equivalence_path, root=root), description="TRR-0009 loader equivalence receipt")
    if equivalence.get("schema") != "token-reconstruction.trr0009-loader-equivalence-qualification.v1" or equivalence.get("task_id") != contract.TASK_ID or equivalence.get("status") != "TRR8_PUBLIC_FIXTURE_LOADER_EQUIVALENCE_PASS":
        raise PayloadError("loader equivalence receipt is not a PASS")
    _require_truth_free(equivalence, description="loader equivalence receipt")
    equivalence_record = _record(_resolve(equivalence_path, root=root), root=root, description="loader equivalence receipt")
    method_freeze_record = _record(_resolve(method_freeze_path, root=root), root=root, description="method freeze")
    if equivalence.get("method_freeze") != method_freeze_record:
        raise PayloadError("loader equivalence method-freeze binding changed")

    plan_path = _resolve(plan_path, root=root)
    plan = _load(plan_path, description="TRR-0009 decision contract plan")
    if plan.get("task_id") != contract.TASK_ID:
        raise PayloadError("decision-contract plan task identity changed")
    _require_truth_free(plan, description="decision-contract plan")
    _record(plan_path, root=root, description="decision-contract plan")
    frequency_path = _resolve(frequency_reference_path, root=root)
    timing_path = _resolve(timing_plan_path, root=root)
    embedding_path = _resolve(embedding_path, root=root)
    _record(frequency_path, root=root, description="public fitting-frequency reference")
    _record(timing_path, root=root, description="timing plan")
    _record(embedding_path, root=root, description="normalized public embedding")

    rows = _method_rows(freeze=freeze, completion=completion, completion_record=completion_record, root=root)
    payload = {
        "repository_root": str(root),
        "plan_path": str(plan_path),
        "observation_manifest_path": str(observations_path),
        "source_selection_path": str(selection_path),
        "capture_receipt_path": str(capture_path),
        "frequency_reference_path": str(frequency_path),
        "embedding_path": str(embedding_path),
        "timing_plan_path": str(timing_path),
        "method_rows": rows,
        "records_by_domain": {str(k): int(v) for k, v in counts.items()},
        "code_bindings": _code_bindings(freeze=freeze, root=root),
        "output_root": output_root,
        "initialization_equivalence": {
            "status": "PASS",
            "scope": "TRR8 public fixture loader/forward equivalence and exact current-H/full-vocabulary runner boundary before TRR9 source capture",
            "receipt": equivalence_record,
            "method_freeze": method_freeze_record,
            "code_commit": equivalence.get("code_commit"),
            "fresh_source_selection": False,
            "truth_opened": False,
        },
    }
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--method-freeze", type=Path, default=METHOD_FREEZE_DEFAULT)
    parser.add_argument("--completion", type=Path, default=COMPLETION_DEFAULT)
    parser.add_argument("--plan", type=Path, default=PLAN_DEFAULT)
    parser.add_argument("--selection", type=Path, default=SELECTION_DEFAULT)
    parser.add_argument("--capture", type=Path, default=CAPTURE_DEFAULT)
    parser.add_argument("--observations", type=Path, default=OBSERVATIONS_DEFAULT)
    parser.add_argument("--equivalence", type=Path, default=EQUIVALENCE_DEFAULT)
    parser.add_argument("--frequency-reference", type=Path, default=FREQUENCY_DEFAULT)
    parser.add_argument("--timing-plan", type=Path, default=TIMING_PLAN_DEFAULT)
    parser.add_argument("--embedding", type=Path, default=EMBEDDING_DEFAULT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--output", type=Path, default=PAYLOAD_DEFAULT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = build_payload(
            repository_root=args.repository_root,
            method_freeze_path=args.method_freeze,
            completion_path=args.completion,
            plan_path=args.plan,
            selection_path=args.selection,
            capture_path=args.capture,
            observation_manifest_path=args.observations,
            equivalence_path=args.equivalence,
            frequency_reference_path=args.frequency_reference,
            timing_plan_path=args.timing_plan,
            embedding_path=args.embedding,
            output_root=args.output_root,
        )
        contract.write_create_only(args.output, payload)
    except (PayloadError, OSError, ValueError) as exc:
        print(f"TRR-0009 registration payload assembly failed closed: {exc}")
        return 2
    print(json.dumps({"status": "PAYLOAD_READY", "output": str(args.output.resolve()), "methods": list(payload["method_rows"]), "records_by_domain": payload["records_by_domain"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
