"""Pure in-memory TRR-P08 scorer after the joint freeze.

The file deliberately has no artifact or truth loader.  ``score_arrays`` is
called only after an external prediction/observation/truth gate has validated
the frozen matrix.  Its input is an explicit in-memory cell mapping, which
keeps this CPU layer testable without opening a real evaluator payload.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from token_reconstruction.trr_p08_metrics import (  # noqa: E402
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_BOOTSTRAP_SEED,
    DOMAINS,
    METHOD_ORDER,
    P08MetricsError,
    REPLICATE_SEEDS,
    TARGETS,
    classify_general_staging_gate,
    classify_interaction_gate,
    paired_cluster_bootstrap,
    paired_metrics_from_scores,
    score_method,
)

TASK_ID = "TRR-P08"
SCORE_SCHEMA = "token-reconstruction.trr-p08-score.v1"


class P08ScoreError(RuntimeError):
    """Raised when the already-frozen in-memory score matrix is malformed."""


def _cell_key(domain: str, target: str) -> str:
    return f"{domain}/{target}"


def _seed_scores(method_scores: Mapping[str, Mapping[str | int, Mapping[str, Any]]], seed: int) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for method in METHOD_ORDER:
        values = method_scores.get(method)
        if not isinstance(values, Mapping) or str(seed) not in {str(key) for key in values}:
            raise P08ScoreError(f"method matrix lacks {method}/{seed}")
        value = next(score for key, score in values.items() if str(key) == str(seed))
        result[method] = value
    return result


def _seed_interaction_points(method_scores: Mapping[str, Mapping[str | int, Mapping[str, Any]]]) -> list[dict[str, float | None]]:
    values: list[dict[str, float | None]] = []
    for seed in REPLICATE_SEEDS:
        scores = _seed_scores(method_scores, seed)
        staged = paired_metrics_from_scores(
            scores["p08_past_only_staged"],
            scores["p08_positionwise_staged"],
            contrast_id="past_staged_minus_positionwise_staged",
        )["metrics"]
        joint = paired_metrics_from_scores(
            scores["p08_past_only_joint"],
            scores["p08_positionwise_joint"],
            contrast_id="past_joint_minus_positionwise_joint",
        )["metrics"]
        token_staged = staged.get("token_delta_pp")
        token_joint = joint.get("token_delta_pp")
        exact_staged = staged.get("exact_delta_pp")
        exact_joint = joint.get("exact_delta_pp")
        values.append(
            {
                "token_delta_pp": None if token_staged is None or token_joint is None else float(token_staged - token_joint),
                "exact_delta_pp": None if exact_staged is None or exact_joint is None else float(exact_staged - exact_joint),
            }
        )
    return values


def _seed_general_points(method_scores: Mapping[str, Mapping[str | int, Mapping[str, Any]]]) -> dict[str, list[dict[str, float | None]]]:
    result: dict[str, list[dict[str, float | None]]] = {}
    for visibility, staged_method, joint_method in (
        ("past_staged_minus_past_joint", "p08_past_only_staged", "p08_past_only_joint"),
        ("positionwise_staged_minus_positionwise_joint", "p08_positionwise_staged", "p08_positionwise_joint"),
    ):
        points: list[dict[str, float | None]] = []
        for seed in REPLICATE_SEEDS:
            scores = _seed_scores(method_scores, seed)
            contrast = paired_metrics_from_scores(scores[staged_method], scores[joint_method], contrast_id=visibility)["metrics"]
            points.append({
                "token_delta_pp": contrast.get("token_delta_pp"),
                "exact_delta_pp": contrast.get("exact_delta_pp"),
            })
        result[visibility] = points
    return result


def _validate_matrix_cell(cell_id: str, cell: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...], np.ndarray, np.ndarray | None, np.ndarray | None, Mapping[str, Any]]:
    if not isinstance(cell, Mapping):
        raise P08ScoreError(f"{cell_id} is not an object")
    domain, target = cell.get("domain"), cell.get("target")
    if domain not in DOMAINS or target not in TARGETS:
        raise P08ScoreError(f"{cell_id} has an unregistered domain/target")
    ids = cell.get("record_ids")
    if not isinstance(ids, (list, tuple)) or not ids or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise P08ScoreError(f"{cell_id} record IDs are malformed")
    truth = np.asarray(cell.get("truth"))
    if truth.ndim != 2 or truth.shape[0] != len(ids) or truth.shape[1] != 128:
        raise P08ScoreError(f"{cell_id} truth geometry changed")
    predictions = cell.get("predictions")
    if not isinstance(predictions, Mapping):
        raise P08ScoreError(f"{cell_id} prediction matrix is missing")
    if set(predictions) != set(METHOD_ORDER):
        raise P08ScoreError(f"{cell_id} method matrix is incomplete")
    return str(domain), str(target), tuple(ids), truth, cell.get("attention_mask"), cell.get("position_ids"), predictions


def score_arrays(
    cells: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Score all eight arm/seed fits in an already-authorized matrix.

    ``cells`` must contain four domain/target entries.  Each entry supplies
    ``truth``, geometry, ordered ``record_ids``, and
    ``predictions[method][seed]``.  The function never reads paths and never
    chooses records or checkpoints.
    """

    if not isinstance(cells, Mapping):
        raise P08ScoreError("P08 cells are not a mapping")
    expected = {_cell_key(domain, target) for domain in DOMAINS for target in TARGETS}
    if set(cells) != expected:
        raise P08ScoreError(f"P08 requires exactly cells {sorted(expected)}")

    score_cells: dict[str, dict[str, Any]] = {}
    by_domain_ids: dict[str, tuple[str, ...]] = {}
    for cell_id, raw_cell in cells.items():
        domain, target, ids, truth, mask, positions, predictions = _validate_matrix_cell(cell_id, raw_cell)
        if domain in by_domain_ids and by_domain_ids[domain] != ids:
            raise P08ScoreError(f"source-record order differs across {domain} targets")
        by_domain_ids[domain] = ids
        score_maps: dict[str, dict[str, Any]] = {method: {} for method in METHOD_ORDER}
        reference_valid: np.ndarray | None = None
        reference_exact: np.ndarray | None = None
        for method in METHOD_ORDER:
            replicates = predictions[method]
            if not isinstance(replicates, Mapping) or {str(key) for key in replicates} != {str(seed) for seed in REPLICATE_SEEDS}:
                raise P08ScoreError(f"{cell_id}/{method} must contain seeds {REPLICATE_SEEDS}")
            for seed in REPLICATE_SEEDS:
                value = next(prediction for key, prediction in replicates.items() if str(key) == str(seed))
                try:
                    scored = score_method(
                        value,
                        truth,
                        record_ids=ids,
                        attention_mask=mask,
                        position_ids=positions,
                        method_id=method,
                    )
                except P08MetricsError as exc:
                    raise P08ScoreError(f"{cell_id}/{method}/{seed}: {exc}") from exc
                valid = np.asarray([row["valid_post_bos"] for row in scored["per_record"]], dtype=bool)
                exact = np.asarray([row["exact_eligible"] for row in scored["per_record"]], dtype=bool)
                if reference_valid is None:
                    reference_valid, reference_exact = valid, exact
                elif not np.array_equal(reference_valid, valid) or not np.array_equal(reference_exact, exact):
                    raise P08ScoreError(f"{cell_id} methods have different metric geometry")
                score_maps[method][str(seed)] = scored
        score_cells[cell_id] = {
            "domain": domain,
            "target": target,
            "records": len(ids),
            "record_ids": list(ids),
            "scores": score_maps,
            "seed_interaction": _seed_interaction_points(score_maps),
            "seed_general_staging": _seed_general_points(score_maps),
        }

    bootstrap_input = {
        cell_id: {
            "domain": cell["domain"],
            "target": cell["target"],
            "method_scores": cell["scores"],
        }
        for cell_id, cell in score_cells.items()
    }
    try:
        bootstrap = paired_cluster_bootstrap(bootstrap_input, draws=bootstrap_draws, seed=bootstrap_seed)
    except P08MetricsError as exc:
        raise P08ScoreError(str(exc)) from exc
    primary_summaries = {
        domain: bootstrap["cells"][_cell_key(domain, "public_base")]["bootstrap"]
        for domain in DOMAINS
    }
    seed_interactions = {
        domain: score_cells[_cell_key(domain, "public_base")]["seed_interaction"]
        for domain in DOMAINS
    }
    gate = classify_interaction_gate(primary_summaries, seed_interactions=seed_interactions)
    for cell_id, cell in score_cells.items():
        cell["general_staging"] = bootstrap["cells"][cell_id]["general_staging"]
    general_summaries = {
        domain: bootstrap["cells"][_cell_key(domain, "public_base")]["general_staging"]
        for domain in DOMAINS
    }
    general_seed_contrasts = {
        domain: score_cells[_cell_key(domain, "public_base")]["seed_general_staging"]
        for domain in DOMAINS
    }
    general_gate = classify_general_staging_gate(general_summaries, seed_contrasts=general_seed_contrasts)
    return {
        "schema": SCORE_SCHEMA,
        "task_id": TASK_ID,
        "status": "TRR-P08_SCORED_AFTER_JOINT_FREEZE",
        "cells": score_cells,
        "bootstrap": bootstrap,
        "gate": gate,
        "general_staging_gate": general_gate,
        "truth_opened": True,
        "truth_payload_persisted": False,
        "claim_scope": "exploratory task-local P08 staged-versus-joint interaction; not a canonical replacement or universal mechanism claim",
    }


__all__ = ["P08ScoreError", "SCORE_SCHEMA", "score_arrays"]
