#!/usr/bin/env python3
"""Freeze TRR-0005 method choices and bind the later public panel.

The first command is source-free: it records the eight selected method states,
the executable source bytes, the public validation choice, and the decision
plan before a fresh row is selected.  The second command consumes that marker
and a completed public panel/selection plan, adding only panel-bound
descriptors.  Neither command loads a model, tokenizer, dataset row, truth
sidecar, or activation tensor.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from token_reconstruction.footing import (
    FootingError,
    external_file_record,
    file_record,
)
from token_reconstruction.trr0005_contract import (
    ContractError,
    METHOD_IDS,
    METHOD_SPECS,
    REGISTRATION_SCHEMA,
    TASK_ID,
    build_registration as _contract_build_registration,
    validate_method_ids,
    validate_panel_descriptor,
    validate_registration,
    valid_sha256,
)


METHOD_FREEZE_SCHEMA = "token-reconstruction.trr0005-method-freeze.v1"
PUBLIC_SELECTION_SCHEMA = "token-reconstruction.trr0005-public-validation-selection.v1"
METHOD_FREEZE_STATUS = "FROZEN_METHOD_PRESELECTION"
REGISTRATION_STATUS = "FROZEN_METHOD_REGISTRATION"
A2_METHOD_ID = "frozen_a1_a2_k256"
CAUSAL_STATE = "affine_causal_h_attention128"
DIAGONAL_STATE = "affine_trained_diagonal_attention128"
AFFINE_STATE = "joint_full_affine"

DEFAULT_DECISION_PLAN = Path("experiments/TRR-0005/decision_plan.json")
DEFAULT_FIT_ROOT = Path("experiments/TRR-0005/joint_fit_v1")
DEFAULT_LENS = Path("experiments/TRR-0004/evidence/comparators/public_a1_lens.pt")
DEFAULT_EMBEDDING = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/"
    "outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors"
)
DEFAULT_P0_CHECKPOINT = Path(
    "/home/alanz/.cache/huggingface/hub/models--meta-llama--"
    "Llama-3.2-1B-Instruct/blobs/"
    "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"
)
DEFAULT_P0_CONFIG = Path(
    "/home/alanz/.cache/huggingface/hub/models--meta-llama--"
    "Llama-3.2-1B-Instruct/blobs/"
    "3e3aaf51a035cb5092d9f6827a0dc074657ba88c"
)
DEFAULT_A2_REFERENCE = Path(
    "experiments/TRR-0004/evidence/comparators/round001_teacher.py"
)

COMMON_CODE_PATHS = (
    "scripts/trr0005_bind_methods.py",
    "scripts/trr0005_run_predictions.py",
    "scripts/trr0005_predict_confirmation.py",
    "scripts/trr0005_score_confirmation.py",
    "scripts/trr0005_freeze_confirmation.py",
    "src/token_reconstruction/trr0005_contract.py",
    "src/token_reconstruction/trr0005_joint_decoder.py",
    "src/token_reconstruction/footing.py",
    "src/token_reconstruction/freeze.py",
    "scripts/trr0004_predict_confirmation.py",
    "scripts/trr0004_fresh_confirmation.py",
    "scripts/trr0003_footing_compare.py",
    "src/token_reconstruction/historical_inputlens_bridge.py",
    "src/token_reconstruction/public_prefix.py",
    "src/token_reconstruction/component_crossover.py",
    "src/token_reconstruction/a1a2_configuration_search.py",
    "src/token_reconstruction/dual_benchmark.py",
    "src/token_reconstruction/experiment_runtime.py",
    "src/token_reconstruction/inverse.py",
    "src/token_reconstruction/causal_decoder_extension.py",
    "src/token_reconstruction/historical_affine_ce.py",
)


class MethodBindingError(ContractError):
    """Raised when a source or panel binding cannot be made fail-closed."""


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise MethodBindingError(f"{description} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MethodBindingError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MethodBindingError(f"{description} must be a JSON object")
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise MethodBindingError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
            handle.write("\n")
    except OSError as exc:
        raise MethodBindingError(f"unable to write binding artifact: {path}") from exc


def _resolve_repo_path(path: Path, *, root: Path, description: str) -> Path:
    raw = path.expanduser()
    resolved = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise MethodBindingError(f"{description} must be inside the repository") from exc
    return resolved


def _repo_record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    path = _resolve_repo_path(path, root=root, description=description)
    try:
        return file_record(path, repository_root=root)
    except (FootingError, OSError, ValueError) as exc:
        raise MethodBindingError(f"{description} is unavailable: {path}") from exc


def _external_record(path: Path, *, description: str) -> dict[str, Any]:
    raw = path.expanduser()
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise MethodBindingError(f"{description} is unavailable: {raw}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise MethodBindingError(f"{description} must resolve to a regular file: {raw}")
    try:
        return external_file_record(resolved)
    except (FootingError, OSError, ValueError) as exc:
        raise MethodBindingError(f"{description} is unavailable: {resolved}") from exc


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MethodBindingError("unable to resolve executable git commit") from exc
    commit = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise MethodBindingError("executable git commit is not a full lowercase hash")
    return commit


def _resolve_commit(root: Path, supplied: str | None) -> str:
    if supplied is None:
        return _git_commit(root)
    commit = str(supplied).lower()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise MethodBindingError("supplied code commit is not a full lowercase hash")
    try:
        current = _git_commit(root)
    except MethodBindingError:
        return commit
    if current != commit:
        raise MethodBindingError(
            f"supplied code commit differs from executable HEAD: {commit} != {current}"
        )
    return commit


def _selection_validator() -> Any:
    try:
        import trr0005_score_confirmation as scorer
    except ModuleNotFoundError:
        from scripts import trr0005_score_confirmation as scorer
    return scorer.validate_public_validation_selection


def _validate_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(selection, Mapping):
        raise MethodBindingError("public-validation selection is absent")
    try:
        _selection_validator()(selection)
    except Exception as exc:
        raise MethodBindingError("public-validation selection is invalid") from exc
    return dict(selection)


def _reject_fresh_payload(value: Any, *, path: str = "method_freeze") -> None:
    forbidden = {
        "source_record_id",
        "public_record_sha256",
        "row_index",
        "raw_index",
        "token_ids",
        "input_ids",
        "labels",
        "panel_sha256",
        "fresh_panel",
        "holdout_ids",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold().replace("-", "_") in forbidden:
                raise MethodBindingError(f"{path}.{key} contains fresh source/panel state")
            _reject_fresh_payload(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_fresh_payload(child, path=f"{path}[{index}]")


def _descriptor_equal(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    return all(expected.get(key) == observed.get(key) for key in ("path", "bytes", "sha256"))


def _state_method(method_id: str) -> tuple[str, str | None]:
    if method_id in {"historical_alpaca_a1", A2_METHOD_ID}:
        return "anchor", None
    distribution, state = method_id.split("__", 1)
    return distribution, state




def build_public_validation_selection(
    *,
    decision_plan_sha256: str,
    repository_root: Path,
    fit_evidence_path: Path,
) -> dict[str, Any]:
    """Select the two public positionwise contenders from fit evidence."""

    try:
        decision_digest = valid_sha256(
            decision_plan_sha256, description="decision plan"
        )
    except ContractError as exc:
        raise MethodBindingError(str(exc)) from exc
    root = repository_root.expanduser().resolve()
    evidence_path = _resolve_repo_path(
        fit_evidence_path, root=root, description="fit evidence"
    )
    evidence = _load_json(evidence_path, description="joint fit evidence")
    if evidence.get("task_id") != TASK_ID:
        raise MethodBindingError("joint fit evidence task ID changed")
    if evidence.get("final_holdout_loaded") is True:
        raise MethodBindingError("joint fit evidence includes final holdout data")
    if evidence.get("current_evaluator_truth_accessed") is True:
        raise MethodBindingError("joint fit evidence accessed evaluator truth")
    distributions = evidence.get("distributions")
    if not isinstance(distributions, Mapping):
        raise MethodBindingError("joint fit evidence has no distributions")
    candidates_by_distribution: dict[str, list[dict[str, Any]]] = {}
    winners: dict[str, dict[str, Any]] = {}
    for distribution in ("original", "enriched"):
        row = distributions.get(distribution)
        methods = row.get("methods") if isinstance(row, Mapping) else None
        if not isinstance(methods, Mapping):
            raise MethodBindingError(
                f"joint fit evidence has no methods: {distribution}"
            )
        candidates: list[dict[str, Any]] = []
        for state in (AFFINE_STATE, DIAGONAL_STATE):
            method = methods.get(state)
            if not isinstance(method, Mapping):
                raise MethodBindingError(
                    f"joint fit evidence is missing {distribution}/{state}"
                )
            canonical = f"{distribution}__{state}"
            if method.get("canonical_method_id") not in (None, canonical):
                raise MethodBindingError(
                    f"fit evidence method identity changed: {canonical}"
                )
            score = method.get("best_validation_style_balanced_token_accuracy")
            selected_step = method.get("selected_step")
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise MethodBindingError(f"fit evidence score is invalid: {canonical}")
            if not isinstance(selected_step, int) or selected_step < 0:
                raise MethodBindingError(
                    f"fit evidence selected step is invalid: {canonical}"
                )
            curve = method.get("curve")
            if not isinstance(curve, Mapping):
                raise MethodBindingError(f"fit evidence curve is absent: {canonical}")
            curve_path = curve.get("path")
            if not isinstance(curve_path, str):
                raise MethodBindingError(f"fit evidence curve path is absent: {canonical}")
            curve_record = _repo_record(
                Path(curve_path), root=root, description=f"{canonical} curve"
            )
            if any(
                curve.get(key) != curve_record.get(key)
                for key in ("bytes", "sha256")
            ):
                raise MethodBindingError(f"fit evidence curve binding changed: {canonical}")
            candidates.append(
                {
                    "method_id": canonical,
                    "score": float(score),
                    "selected_step": selected_step,
                    "curve_file": curve_record,
                }
            )
        winner = max(candidates, key=lambda item: (item["score"], -item["selected_step"]))
        candidates_by_distribution[distribution] = candidates
        winners[distribution] = winner
    fit_record = _repo_record(
        evidence_path, root=root, description="joint fit evidence"
    )
    selection = {
        "schema": PUBLIC_SELECTION_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_PUBLIC_VALIDATION_SELECTION",
        "selection_stage": "public_validation_before_fresh_evaluation",
        "truth_accessed": False,
        "fresh_evaluation_accessed": False,
        "decision_plan_sha256": decision_digest,
        "fit_evidence": fit_record,
        "selection_metric": "validation_style_balanced_token_accuracy",
        "selection_rule": (
            "Select the maximum public validation-style accuracy mean from "
            "the affine and trained diagonal contenders per distribution; "
            "ties use the earliest selected step."
        ),
        "distributions": {
            distribution: {
                "candidate_method_ids": [
                    item["method_id"]
                    for item in candidates_by_distribution[distribution]
                ],
                "candidates": candidates_by_distribution[distribution],
                "selected_method_id": winners[distribution]["method_id"],
                "selected_score": winners[distribution]["score"],
                "selected_step": winners[distribution]["selected_step"],
            }
            for distribution in ("original", "enriched")
        },
    }
    _validate_selection(selection)
    return selection

def _state_path(
    *,
    method_id: str,
    fit_root: Path,
    causal_fit_root: Path,
    lens_path: Path,
) -> Path:
    if method_id in {"historical_alpaca_a1", A2_METHOD_ID}:
        return lens_path
    distribution, state = _state_method(method_id)
    state_root = causal_fit_root if state == CAUSAL_STATE else fit_root
    expected = {
        AFFINE_STATE,
        CAUSAL_STATE,
        DIAGONAL_STATE,
    }
    if distribution not in {"original", "enriched"} or state not in expected:
        raise MethodBindingError(f"unknown TRR5 fitted method: {method_id}")
    return state_root / distribution / state / "selected.safetensors"


def _code_paths(
    *,
    root: Path,
    method_id: str,
    a2_reference: Path,
    supplied: Mapping[str, Sequence[Path]] | Sequence[Path] | None,
) -> tuple[Path, ...]:
    if supplied is None:
        relative = list(COMMON_CODE_PATHS)
        if method_id == A2_METHOD_ID:
            relative.append(a2_reference.as_posix())
        values = [
            _resolve_repo_path(Path(value), root=root, description="code source")
            for value in relative
        ]
    elif isinstance(supplied, Mapping):
        values = [
            _resolve_repo_path(Path(value), root=root, description="code source")
            for value in supplied.get(method_id, ())
        ]
    else:
        values = [
            _resolve_repo_path(Path(value), root=root, description="code source")
            for value in supplied
        ]
    result: list[Path] = []
    for value in values:
        if value not in result:
            result.append(value)
    if not result:
        raise MethodBindingError(f"code binding is empty: {method_id}")
    return tuple(result)


def _attention_amendment(
    path: Path | None,
    *,
    root: Path,
) -> dict[str, Any] | None:
    if path is None:
        return None
    descriptor_path = _resolve_repo_path(
        path, root=root, description="attention amendment"
    )
    value = _load_json(descriptor_path, description="attention amendment")
    if value.get("task_id") not in (None, TASK_ID):
        raise MethodBindingError("attention amendment task ID changed")
    if value.get("truth_accessed") is True:
        raise MethodBindingError("attention amendment accessed truth")
    _reject_fresh_payload(value, path="attention_amendment")
    return _repo_record(
        descriptor_path, root=root, description="attention amendment"
    )


def _state_binding(
    *,
    method_id: str,
    root: Path,
    state_path: Path,
    decision_plan_record: Mapping[str, Any],
    attention_record: Mapping[str, Any] | None,
    code_paths: Sequence[Path],
    code_commit: str,
    embedding_record: Mapping[str, Any],
    p0_records: Mapping[str, Any],
) -> dict[str, Any]:
    state_record = _repo_record(
        state_path, root=root, description=f"{method_id} state"
    )
    configs = [dict(decision_plan_record)]
    if method_id.split("__", 1)[-1] == CAUSAL_STATE and attention_record:
        configs.append(dict(attention_record))
    binding = {
        "method_id": method_id,
        "method_rule": next(
            spec["rule"] for spec in METHOD_SPECS if spec["id"] == method_id
        ),
        "status": "FROZEN",
        "method_state": [state_record],
        "state_sha256": state_record["sha256"],
        "method_config": configs,
        "code": [
            _repo_record(path, root=root, description=f"{method_id} code")
            for path in code_paths
        ],
        "code_commit": code_commit,
        "runtime_assets": {
            "public_embedding_table": dict(embedding_record),
        },
    }
    if method_id == A2_METHOD_ID:
        binding["runtime_assets"].update(
            {
                "public_prefix_checkpoint": dict(
                    p0_records["public_prefix_checkpoint"]
                ),
                "public_prefix_config": dict(p0_records["public_prefix_config"]),
            }
        )
    return binding




def build_method_freeze(
    *,
    repository_root: Path,
    decision_plan_path: Path = DEFAULT_DECISION_PLAN,
    output_freeze: Path,
    output_selection: Path,
    fit_root: Path = DEFAULT_FIT_ROOT,
    causal_fit_root: Path | None = None,
    lens_path: Path = DEFAULT_LENS,
    embedding_path: Path = DEFAULT_EMBEDDING,
    p0_checkpoint: Path = DEFAULT_P0_CHECKPOINT,
    p0_config: Path = DEFAULT_P0_CONFIG,
    attention_amendment_path: Path | None = None,
    a2_reference: Path = DEFAULT_A2_REFERENCE,
    code_commit: str | None = None,
    code_paths: Mapping[str, Sequence[Path]] | Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Write the source-free selection and method-freeze markers.

    Only the decision plan, selected state files, executable source files, and
    public runtime resources are touched.  In particular, this function has
    no tokenizer/dataset/model imports and no reserved-row arguments.
    """

    root = repository_root.expanduser().resolve()
    decision_path = _resolve_repo_path(
        decision_plan_path, root=root, description="decision plan"
    )
    decision_value = _load_json(decision_path, description="decision plan")
    if decision_value.get("task_id") not in (None, TASK_ID):
        raise MethodBindingError("decision plan task ID changed")
    decision_record = _repo_record(
        decision_path, root=root, description="decision plan"
    )
    fit = _resolve_repo_path(fit_root, root=root, description="fit root")
    causal = (
        _resolve_repo_path(causal_fit_root, root=root, description="causal fit root")
        if causal_fit_root is not None
        else fit
    )
    if causal != fit and attention_amendment_path is None:
        raise MethodBindingError(
            "a separate causal state root requires an attention amendment binding"
        )
    lens = _resolve_repo_path(lens_path, root=root, description="retained lens")
    commit = _resolve_commit(root, code_commit)
    embedding_record = _external_record(
        embedding_path, description="normalized public embedding table"
    )
    p0_records = {
        "public_prefix_checkpoint": _external_record(
            p0_checkpoint, description="public P0 checkpoint"
        ),
        "public_prefix_config": _external_record(
            p0_config, description="public P0 config"
        ),
    }
    attention_record = _attention_amendment(
        attention_amendment_path, root=root
    )

    states: dict[str, Path] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for method_id in METHOD_IDS:
        state_path = _state_path(
            method_id=method_id,
            fit_root=fit,
            causal_fit_root=causal,
            lens_path=lens,
        )
        states[method_id] = state_path
        paths = _code_paths(
            root=root,
            method_id=method_id,
            a2_reference=a2_reference,
            supplied=code_paths,
        )
        bindings[method_id] = _state_binding(
            method_id=method_id,
            root=root,
            state_path=state_path,
            decision_plan_record=decision_record,
            attention_record=attention_record,
            code_paths=paths,
            code_commit=commit,
            embedding_record=embedding_record,
            p0_records=p0_records,
        )

    selection = build_public_validation_selection(
        decision_plan_sha256=decision_record["sha256"],
        repository_root=root,
        fit_evidence_path=fit / "run_evidence.json",
    )
    selection_path = _resolve_repo_path(
        output_selection, root=root, description="public selection output"
    )
    freeze_path = _resolve_repo_path(
        output_freeze, root=root, description="method freeze output"
    )
    if selection_path == freeze_path:
        raise MethodBindingError("method-freeze and selection outputs must differ")
    _write_create_only(selection_path, selection)
    selection_record = _repo_record(
        selection_path, root=root, description="public selection output"
    )
    payload: dict[str, Any] = {
        "schema": METHOD_FREEZE_SCHEMA,
        "task_id": TASK_ID,
        "status": METHOD_FREEZE_STATUS,
        "scope": (
            "Source-free method/state and public-validation freeze. Fresh "
            "source IDs, panel descriptors, observations, and truth are "
            "intentionally absent."
        ),
        "method_ids": list(METHOD_IDS),
        "methods": [
            {
                **dict(spec),
                "status": "FROZEN",
                "state_sha256": bindings[spec["id"]]["state_sha256"],
            }
            for spec in METHOD_SPECS
        ],
        "state_bindings": bindings,
        "code_commit": commit,
        "decision_plan": dict(decision_record),
        "decision_plan_sha256": decision_record["sha256"],
        "public_validation_selection": selection,
        "public_validation_selection_file": dict(selection_record),
        "state_roots": {
            "fitted": str(fit.relative_to(root).as_posix()),
            "causal": str(causal.relative_to(root).as_posix()),
        },
        "attention_amendment": (
            dict(attention_record) if attention_record is not None else None
        ),
        "truth_opened": False,
        "fresh_evaluation_started": False,
        "source_accessed": False,
        "target_weights_available_to_reconstruction": False,
    }
    _reject_fresh_payload(payload)
    _write_create_only(freeze_path, payload)
    result = dict(payload)
    result["method_freeze"] = _repo_record(
        freeze_path, root=root, description="method freeze output"
    )
    result["public_validation_selection_file"] = dict(selection_record)
    return result


# Explicit compatibility name for launch scripts and review tooling.
freeze_methods = build_method_freeze




def _validate_freeze(
    path: Path,
    *,
    root: Path,
    decision_plan_path: Path | None = None,
    public_selection_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Revalidate a source-free marker and every descriptor before panel bind."""

    freeze_path = _resolve_repo_path(
        path, root=root, description="method freeze"
    )
    freeze = _load_json(freeze_path, description="method freeze")
    if freeze.get("schema") != METHOD_FREEZE_SCHEMA:
        raise MethodBindingError("method freeze schema changed")
    if freeze.get("task_id") != TASK_ID or freeze.get("status") not in {
        METHOD_FREEZE_STATUS,
        REGISTRATION_STATUS,
    }:
        raise MethodBindingError("method freeze is not in a frozen state")
    _reject_fresh_payload(freeze)
    try:
        validate_method_ids(freeze.get("method_ids", ()))
    except ContractError as exc:
        raise MethodBindingError("method freeze method order changed") from exc
    if freeze.get("truth_opened") is True or freeze.get("fresh_evaluation_started") is True:
        raise MethodBindingError("method freeze was written after fresh evaluation")
    commit = freeze.get("code_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise MethodBindingError("method freeze lacks a full code commit")
    decision_digest = freeze.get("decision_plan_sha256")
    if not isinstance(decision_digest, str):
        raise MethodBindingError("method freeze lacks a decision-plan digest")
    try:
        valid_sha256(decision_digest, description="decision plan")
    except ContractError as exc:
        raise MethodBindingError(str(exc)) from exc
    if decision_plan_path is not None:
        decision_path = _resolve_repo_path(
            decision_plan_path, root=root, description="decision plan"
        )
        decision_record = _repo_record(
            decision_path, root=root, description="decision plan"
        )
        if decision_record["sha256"] != decision_digest:
            raise MethodBindingError("decision plan differs from method freeze")
    selection = freeze.get("public_validation_selection")
    if public_selection_path is not None:
        selected_path = _resolve_repo_path(
            public_selection_path, root=root,
            description="public-validation selection",
        )
        selected = _load_json(
            selected_path, description="public-validation selection"
        )
        selection_file = freeze.get("public_validation_selection_file")
        actual_selection_record = _repo_record(
            selected_path, root=root, description="public-validation selection"
        )
        if isinstance(selection_file, Mapping) and not _descriptor_equal(
            selection_file, actual_selection_record
        ):
            raise MethodBindingError("public selection file binding changed")
        if selection != selected:
            raise MethodBindingError("public selection differs from method freeze")
    if not isinstance(selection, Mapping):
        raise MethodBindingError("method freeze has no public validation selection")
    normalized_selection = _validate_selection(selection)

    bindings = freeze.get("state_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(METHOD_IDS):
        raise MethodBindingError("method freeze lacks all eight state bindings")
    for method_id in METHOD_IDS:
        binding = bindings.get(method_id)
        if not isinstance(binding, Mapping):
            raise MethodBindingError(f"method freeze binding is malformed: {method_id}")
        if binding.get("method_id") != method_id:
            raise MethodBindingError(f"method freeze binding identity changed: {method_id}")
        expected_rule = next(
            spec["rule"] for spec in METHOD_SPECS if spec["id"] == method_id
        )
        if binding.get("method_rule") != expected_rule:
            raise MethodBindingError(f"method freeze rule changed: {method_id}")
        if binding.get("status") in {"PENDING_STATE", "UNFIT", "UNSELECTED"}:
            raise MethodBindingError(f"method freeze state is incomplete: {method_id}")
        if binding.get("code_commit") != commit:
            raise MethodBindingError(f"method freeze code commit changed: {method_id}")
        states = binding.get("method_state")
        if not isinstance(states, list) or len(states) != 1:
            raise MethodBindingError(f"{method_id} must bind exactly one state")
        state_record = states[0]
        if not isinstance(state_record, Mapping):
            raise MethodBindingError(f"{method_id} state descriptor is malformed")
        actual_state = _repo_record(
            Path(str(state_record.get("path", ""))),
            root=root,
            description=f"{method_id} state",
        )
        if not _descriptor_equal(state_record, actual_state):
            raise MethodBindingError(f"{method_id} state binding changed")
        if binding.get("state_sha256") != actual_state["sha256"]:
            raise MethodBindingError(f"{method_id} state hash is inconsistent")
        configs = binding.get("method_config")
        if not isinstance(configs, list) or not configs:
            raise MethodBindingError(f"{method_id} has no method config")
        for index, descriptor in enumerate(configs):
            if not isinstance(descriptor, Mapping):
                raise MethodBindingError(
                    f"{method_id} method config is malformed: {index}"
                )
            actual = _repo_record(
                Path(str(descriptor.get("path", ""))),
                root=root,
                description=f"{method_id} method config",
            )
            if not _descriptor_equal(descriptor, actual):
                raise MethodBindingError(
                    f"{method_id} method config changed: {index}"
                )
        code = binding.get("code")
        if not isinstance(code, list) or not code:
            raise MethodBindingError(f"{method_id} has no executable code")
        for index, descriptor in enumerate(code):
            if not isinstance(descriptor, Mapping):
                raise MethodBindingError(f"{method_id} code is malformed: {index}")
            actual = _repo_record(
                Path(str(descriptor.get("path", ""))),
                root=root,
                description=f"{method_id} code",
            )
            if not _descriptor_equal(descriptor, actual):
                raise MethodBindingError(f"{method_id} code changed: {index}")
        assets = binding.get("runtime_assets")
        expected_roles = (
            {"public_embedding_table", "public_prefix_checkpoint",
             "public_prefix_config"}
            if method_id == A2_METHOD_ID
            else {"public_embedding_table"}
        )
        if not isinstance(assets, Mapping) or set(assets) != expected_roles:
            raise MethodBindingError(
                f"runtime asset roles changed: {method_id}"
            )
        for role, descriptor in assets.items():
            if not isinstance(descriptor, Mapping):
                raise MethodBindingError(
                    f"runtime asset is malformed: {method_id}/{role}"
                )
            raw_path = descriptor.get("path")
            if not isinstance(raw_path, str) or not Path(raw_path).expanduser().is_absolute():
                raise MethodBindingError(
                    f"runtime asset must use an absolute path: {method_id}/{role}"
                )
            try:
                actual = _external_record(
                    Path(raw_path),
                    description=f"{method_id}/{role}",
                )
            except MethodBindingError:
                raise
            if not _descriptor_equal(descriptor, actual):
                raise MethodBindingError(
                    f"runtime asset changed: {method_id}/{role}"
                )

    freeze_record = _repo_record(
        freeze_path, root=root, description="method freeze"
    )
    return freeze, normalized_selection, freeze_record




def _panel_descriptor_matches(
    descriptor: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    root: Path,
    description: str,
) -> None:
    if not isinstance(descriptor, Mapping):
        raise MethodBindingError(f"{description} is absent")
    raw = descriptor.get("path")
    if not isinstance(raw, str) or not raw:
        raise MethodBindingError(f"{description} path is absent")
    candidate = Path(raw).expanduser()
    actual_path = (
        candidate.resolve()
        if candidate.is_absolute()
        else _resolve_repo_path(candidate, root=root, description=description)
    )
    expected_path = _resolve_repo_path(
        Path(str(expected["path"])), root=root, description=description
    )
    if actual_path != expected_path:
        raise MethodBindingError(f"{description} path binding changed")
    if any(descriptor.get(key) != expected.get(key) for key in ("bytes", "sha256")):
        raise MethodBindingError(f"{description} byte/hash binding changed")


def bind_panel_registration(
    *,
    repository_root: Path,
    method_freeze_path: Path,
    panel_path: Path,
    selection_plan_path: Path,
    output_registration: Path,
    public_validation_selection_path: Path | None = None,
    decision_plan_path: Path | None = None,
) -> dict[str, Any]:
    """Add the fresh panel and selection-plan descriptors to a frozen map."""

    root = repository_root.expanduser().resolve()
    freeze, selection, freeze_record = _validate_freeze(
        method_freeze_path,
        root=root,
        decision_plan_path=decision_plan_path,
        public_selection_path=public_validation_selection_path,
    )
    # The executable prediction driver later enforces this same binding.  Do
    # it here when a real checkout is available so a stale freeze cannot be
    # attached to a new source tree.
    if (root / ".git").exists():
        _resolve_commit(root, str(freeze["code_commit"]))
    panel_file_path = _resolve_repo_path(
        panel_path, root=root, description="confirmation panel"
    )
    plan_file_path = _resolve_repo_path(
        selection_plan_path, root=root, description="selection plan"
    )
    plan = _load_json(plan_file_path, description="selection plan")
    panel = _load_json(panel_file_path, description="confirmation panel")
    if panel.get("task_id") not in (None, TASK_ID):
        raise MethodBindingError("confirmation panel task ID changed")
    try:
        validate_panel_descriptor(panel)
    except ContractError as exc:
        raise MethodBindingError("confirmation panel failed contract validation") from exc
    panel_record = _repo_record(
        panel_file_path, root=root, description="confirmation panel"
    )
    plan_record = _repo_record(
        plan_file_path, root=root, description="selection plan"
    )
    if panel.get("method_freeze_sha256") != freeze_record["sha256"]:
        raise MethodBindingError("panel is bound to a different method freeze")
    _panel_descriptor_matches(
        panel.get("selection_plan"),
        expected=plan_record,
        root=root,
        description="panel selection plan",
    )
    if plan.get("method_freeze_sha256") != freeze_record["sha256"]:
        raise MethodBindingError("selection plan is bound to a different method freeze")
    plan_selection = plan.get("public_validation_selection")
    if not isinstance(plan_selection, Mapping):
        raise MethodBindingError("selection plan has no public validation choice")
    if _validate_selection(plan_selection) != _validate_selection(selection):
        raise MethodBindingError("selection plan changed public validation choice")
    if isinstance(panel.get("public_validation_selection"), Mapping):
        if _validate_selection(panel["public_validation_selection"]) != _validate_selection(selection):
            raise MethodBindingError("panel changed public validation choice")

    frozen_bindings = freeze.get("state_bindings")
    if not isinstance(frozen_bindings, Mapping):
        raise MethodBindingError("method freeze state bindings are absent")
    bindings: dict[str, dict[str, Any]] = {}
    for method_id in METHOD_IDS:
        source = frozen_bindings.get(method_id)
        if not isinstance(source, Mapping):
            raise MethodBindingError(f"method freeze binding is absent: {method_id}")
        binding = copy.deepcopy(dict(source))
        if "panel" in binding:
            raise MethodBindingError(
                f"method freeze unexpectedly contains panel state: {method_id}"
            )
        binding["panel"] = dict(panel_record)
        bindings[method_id] = binding

    registration = _contract_build_registration(
        status=REGISTRATION_STATUS,
        code_commit=str(freeze["code_commit"]),
        state_bindings=bindings,
    )
    registration.update(
        {
            "panel": dict(panel_record),
            "selection_plan": dict(plan_record),
            "method_freeze": dict(freeze_record),
            "method_freeze_sha256": freeze_record["sha256"],
            "public_validation_selection": copy.deepcopy(selection),
            "public_validation_selection_file": (
                dict(
                    _repo_record(
                        public_validation_selection_path,
                        root=root,
                        description="public-validation selection",
                    )
                )
                if public_validation_selection_path is not None
                else dict(freeze.get("public_validation_selection_file", {}))
            ),
            "registration_stage": "PANEL_BOUND_AFTER_PUBLIC_SOURCE_SELECTION",
            "truth_opened": False,
            "fresh_evaluation_started": False,
        }
    )
    try:
        validate_registration(registration, require_frozen=True)
    except ContractError as exc:
        raise MethodBindingError("panel-bound registration failed contract validation") from exc
    registration_path = _resolve_repo_path(
        output_registration, root=root, description="registration output"
    )
    _write_create_only(registration_path, registration)
    result = dict(registration)
    result["registration"] = _repo_record(
        registration_path, root=root, description="registration output"
    )
    return result


# Explicit compatibility names for orchestration scripts.
build_panel_bound_registration = bind_panel_registration
build_registration_from_freeze = bind_panel_registration
bind_registration = bind_panel_registration
build_preselection = build_method_freeze
write_method_freeze = build_method_freeze
build_registration = bind_panel_registration




def _add_common_asset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--decision-plan", type=Path, default=DEFAULT_DECISION_PLAN)
    parser.add_argument("--fit-root", type=Path, default=DEFAULT_FIT_ROOT)
    parser.add_argument(
        "--causal-fit-root",
        type=Path,
        help="optional repaired causal-state root; requires --attention-amendment",
    )
    parser.add_argument("--lens", type=Path, default=DEFAULT_LENS)
    parser.add_argument("--embedding-table", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--p0-checkpoint", type=Path, default=DEFAULT_P0_CHECKPOINT)
    parser.add_argument("--p0-config", type=Path, default=DEFAULT_P0_CONFIG)
    parser.add_argument(
        "--attention-amendment",
        type=Path,
        help="public H-only causal score-rule amendment for a repaired root",
    )
    parser.add_argument(
        "--a2-reference", type=Path, default=DEFAULT_A2_REFERENCE
    )
    parser.add_argument(
        "--code-commit",
        help="test/replay override; a repository checkout must match it",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser(
        "preselect",
        help="write source-free public_validation_selection and method_freeze",
    )
    _add_common_asset_args(pre)
    pre.add_argument("--output-freeze", type=Path, required=True)
    pre.add_argument("--output-selection", type=Path, required=True)

    register = sub.add_parser(
        "register",
        help="bind an existing frozen method map to the public panel and plan",
    )
    register.add_argument("--repository-root", type=Path, default=Path("."))
    register.add_argument("--method-freeze", type=Path, required=True)
    register.add_argument("--panel", type=Path, required=True)
    register.add_argument("--selection-plan", type=Path, required=True)
    register.add_argument("--output-registration", type=Path, required=True)
    register.add_argument("--decision-plan", type=Path)
    register.add_argument("--public-validation-selection", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preselect":
            result = build_method_freeze(
                repository_root=args.repository_root,
                decision_plan_path=args.decision_plan,
                output_freeze=args.output_freeze,
                output_selection=args.output_selection,
                fit_root=args.fit_root,
                causal_fit_root=args.causal_fit_root,
                lens_path=args.lens,
                embedding_path=args.embedding_table,
                p0_checkpoint=args.p0_checkpoint,
                p0_config=args.p0_config,
                attention_amendment_path=args.attention_amendment,
                a2_reference=args.a2_reference,
                code_commit=args.code_commit,
            )
        elif args.command == "register":
            result = bind_panel_registration(
                repository_root=args.repository_root,
                method_freeze_path=args.method_freeze,
                panel_path=args.panel,
                selection_plan_path=args.selection_plan,
                output_registration=args.output_registration,
                public_validation_selection_path=args.public_validation_selection,
                decision_plan_path=args.decision_plan,
            )
        else:  # pragma: no cover
            raise MethodBindingError(f"unknown command: {args.command}")
    except (MethodBindingError, ContractError, OSError, ValueError) as exc:
        raise SystemExit(f"TRR-0005 method-binding error: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
