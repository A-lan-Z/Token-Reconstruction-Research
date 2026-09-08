#!/usr/bin/env python3
"""Emit aggregate-only post-freeze reconstruction breakage diagnostics.

The command is deliberately separate from the P12 scorer.  It consumes the
already reviewed freeze receipt and evaluator truth descriptor, then compares
the frozen integer predictions to evaluator truth without loading a model,
source rows, or source identifiers.  The output contains only counts,
position ranges, first-error positions, and per-record count vectors.  It is a
post hoc descriptive report: the fixed position bins and summaries are chosen
before this command opens evaluator truth, but no confirmatory threshold or
selection rule is fit here.

The release marker is required so this script cannot accidentally open the
private truth descriptor before the root scoring release.  The marker itself
must state that the complete freeze was reviewed and that truth was still
closed at release time.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.trr_p12 import evaluation


TASK_ID = "TRR-P12"
SCHEMA = "token-reconstruction.trr-p12-postfreeze-breakage-descriptive.v1"
RELEASE_SCHEMA = "token-reconstruction.trr-p12-root-truth-release.v1"
RELEASE_STATUS = "COMPLETE_FREEZE_REVIEWED_EVALUATOR_SCORING_RELEASED"
B1_STAGES = (0, 64, 128, 256)
A1_STAGES = (0, 256)
B1_DOMAINS = ("finance", "pile")

# These are fixed before opening evaluator truth.  Positions are 1-based and
# exclude the BOS slot at stored position 0.  The final bin ends at 127 for
# the registered 128-token geometry.
POSITION_BINS = (
    ("positions_1_32", 1, 32),
    ("positions_33_64", 33, 64),
    ("positions_65_96", 65, 96),
    ("positions_97_127", 97, 127),
)


class BreakageError(ValueError):
    """Raised when frozen P12 inputs cannot support the fixed report."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: str | Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise BreakageError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise BreakageError(f"{label} is not a regular file: {resolved}")
    return resolved


def create_only(path: str | Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.exists() or raw.is_symlink():
        raise BreakageError(f"{label} is create-only: {raw}")
    resolved = raw.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BreakageError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise BreakageError(f"{label} must be a JSON object")
    return dict(value)


def validate_release_marker(release_path: Path, freeze_path: Path) -> dict[str, Any]:
    release = read_object(release_path, label="root truth release marker")
    if release.get("schema") != RELEASE_SCHEMA:
        raise BreakageError("root truth release marker schema differs")
    if release.get("task_id") != TASK_ID:
        raise BreakageError("root truth release marker task_id differs")
    if release.get("status") != RELEASE_STATUS:
        raise BreakageError("root truth scoring release is not complete")
    if release.get("truth_opened_at_release") is not False:
        raise BreakageError("root release marker does not prove truth was closed at release")
    if release.get("p03_holdout_accessed") is not False:
        raise BreakageError("root release marker records P03 access")
    binding = release.get("freeze")
    if not isinstance(binding, Mapping):
        raise BreakageError("root release marker lacks freeze binding")
    declared_path = Path(str(binding.get("path", ""))).expanduser().resolve()
    if declared_path != freeze_path.resolve():
        raise BreakageError("release marker freeze path differs from --freeze-receipt")
    declared_hash = str(binding.get("sha256", "")).lower()
    actual_hash = sha256_file(freeze_path)
    if declared_hash != actual_hash:
        raise BreakageError("release marker freeze hash differs from --freeze-receipt")
    return release


def _normalise_mask(mask: torch.Tensor | None, *, shape: tuple[int, int], label: str) -> torch.Tensor:
    if mask is None:
        value = torch.ones(shape, dtype=torch.bool)
    else:
        value = torch.as_tensor(mask).detach().cpu()
        if tuple(value.shape) != shape:
            raise BreakageError(f"{label} valid-mask geometry differs")
        if value.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise BreakageError(f"{label} valid mask is not boolean/integer")
        if value.dtype != torch.bool and bool(((value != 0) & (value != 1)).any().item()):
            raise BreakageError(f"{label} valid mask contains values other than 0/1")
        value = value.to(dtype=torch.bool)
    if shape[1] != 128:
        raise BreakageError(f"{label} requires the registered 128-token geometry")
    # BOS is never a scored reconstruction position, regardless of the mask
    # sidecar.  Reject a masked BOS only through the normal scorer contract;
    # normalizing it here keeps the report aligned with P12 scoring.
    value = value.clone()
    value[:, 0] = False
    return value


def _load_cell(
    freeze: Mapping[str, Any],
    truth_cells: Mapping[tuple[str, str, int], Mapping[str, Any]],
    *,
    method: str,
    domain: str,
    stage: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, str | None]]:
    try:
        freeze_domain = freeze["methods"][method]["domains"][domain]
        freeze_cell = freeze_domain["stages"][str(stage)]
        truth_cell = truth_cells[(method, domain, stage)]
    except (KeyError, TypeError) as exc:
        raise BreakageError(f"missing frozen cell {method}/{domain}/{stage}") from exc
    prediction, truth, mask = evaluation._load_cell_tensors(
        freeze_cell,
        truth_cell,
        label=f"{method}/{domain}/stage{stage}",
    )
    if prediction.ndim != 2 or tuple(prediction.shape) != tuple(truth.shape):
        raise BreakageError(f"{method}/{domain}/stage{stage} prediction/truth geometry differs")
    if prediction.dtype != torch.long or truth.dtype != torch.long:
        raise BreakageError(f"{method}/{domain}/stage{stage} tensors are not integer token IDs")
    valid = _normalise_mask(mask, shape=tuple(prediction.shape), label=f"{method}/{domain}/stage{stage}")
    if method == "B1" and int(prediction.shape[0]) != 128:
        raise BreakageError(f"B1/{domain}/stage{stage} must contain 128 records")
    if method == "A1+A2" and int(prediction.shape[0]) != 32:
        raise BreakageError(f"A1+A2/{domain}/stage{stage} must contain 32 records")
    bindings = {
        "prediction_sha256": str(freeze_cell["artifacts"]["prediction"].get("sha256")),
        "truth_sha256": str(truth_cell["truth"].get("sha256")),
        "valid_mask_sha256": None
        if truth_cell.get("valid_mask") is None
        else str(truth_cell["valid_mask"].get("sha256")),
    }
    return prediction, truth, valid, bindings


def _first_error_positions(error: torch.Tensor) -> list[int | None]:
    rows, positions = error.shape
    position_grid = torch.arange(positions, dtype=torch.long).unsqueeze(0).expand(rows, -1)
    sentinel = torch.full_like(position_grid, positions)
    first = torch.where(error, position_grid, sentinel).min(dim=1).values
    return [None if int(item) == positions else int(item) for item in first.tolist()]


def _first_error_histogram(first: list[int | None]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for value in first:
        key = "none" if value is None else str(value)
        histogram[key] = histogram.get(key, 0) + 1
    return {key: histogram[key] for key in sorted(histogram, key=lambda item: (item == "none", int(item) if item != "none" else 0))}


def _bin_summary(
    *,
    valid: torch.Tensor,
    error: torch.Tensor,
    broken: torch.Tensor | None = None,
    improved: torch.Tensor | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, start, end in POSITION_BINS:
        # Slice end is exclusive while the registered bins are inclusive.
        sl = slice(start, end + 1)
        valid_bin = valid[:, sl]
        error_bin = error[:, sl] & valid_bin
        item: dict[str, Any] = {
            "position_range_inclusive": [start, end],
            "valid_position_count": int(valid_bin.sum().item()),
            "error_position_count": int(error_bin.sum().item()),
            "error_record_count": int(error_bin.any(dim=1).sum().item()),
        }
        if broken is not None and improved is not None:
            broken_bin = broken[:, sl] & valid_bin
            improved_bin = improved[:, sl] & valid_bin
            item.update(
                {
                    "broken_position_count": int(broken_bin.sum().item()),
                    "broken_record_count": int(broken_bin.any(dim=1).sum().item()),
                    "improved_position_count": int(improved_bin.sum().item()),
                    "improved_record_count": int(improved_bin.any(dim=1).sum().item()),
                    "per_record_broken_token_counts": [int(value) for value in broken_bin.sum(dim=1).tolist()],
                    "per_record_improved_token_counts": [int(value) for value in improved_bin.sum(dim=1).tolist()],
                }
            )
        result[name] = item
    return result


def _absolute_summary(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    *,
    broken: torch.Tensor | None = None,
    improved: torch.Tensor | None = None,
) -> dict[str, Any]:
    correct = prediction.eq(truth) & valid
    error = valid & ~correct
    first = _first_error_positions(error)
    result: dict[str, Any] = {
        "record_count": int(prediction.shape[0]),
        "stored_position_count": int(prediction.shape[1]),
        "scored_position_count": int(valid.sum().item()),
        "correct_position_count": int(correct.sum().item()),
        "error_position_count": int(error.sum().item()),
        "error_record_count": int(error.any(dim=1).sum().item()),
        "first_error_position_by_record": first,
        "first_error_position_histogram": _first_error_histogram(first),
        "per_record_error_token_counts": [int(value) for value in error.sum(dim=1).tolist()],
        "bins": _bin_summary(valid=valid, error=error, broken=broken, improved=improved),
    }
    if broken is not None and improved is not None:
        result.update(
            {
                "broken_position_count": int(broken.sum().item()),
                "broken_record_count": int(broken.any(dim=1).sum().item()),
                "improved_position_count": int(improved.sum().item()),
                "improved_record_count": int(improved.any(dim=1).sum().item()),
                "per_record_broken_token_counts": [int(value) for value in broken.sum(dim=1).tolist()],
                "per_record_improved_token_counts": [int(value) for value in improved.sum(dim=1).tolist()],
            }
        )
    return result


def _paired_summary(
    baseline_prediction: torch.Tensor,
    stage_prediction: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, Any]:
    if tuple(baseline_prediction.shape) != tuple(stage_prediction.shape):
        raise BreakageError("baseline/stage prediction geometry differs")
    baseline_correct = baseline_prediction.eq(truth) & valid
    stage_correct = stage_prediction.eq(truth) & valid
    broken = baseline_correct & ~stage_correct
    improved = ~baseline_correct & stage_correct
    result = _absolute_summary(stage_prediction, truth, valid, broken=broken, improved=improved)
    result["baseline_error_position_count"] = int((valid & ~baseline_correct).sum().item())
    result["baseline_correct_position_count"] = int(baseline_correct.sum().item())
    return result


def _bind_truth_consistency(
    baseline_truth: torch.Tensor,
    baseline_valid: torch.Tensor,
    current_truth: torch.Tensor,
    current_valid: torch.Tensor,
    *,
    label: str,
) -> None:
    if not torch.equal(baseline_truth, current_truth):
        raise BreakageError(f"{label} evaluator truth differs across stages")
    if not torch.equal(baseline_valid, current_valid):
        raise BreakageError(f"{label} valid mask differs across stages")


def _cell_binding_record(bindings: Mapping[str, str | None]) -> dict[str, str | None]:
    # Paths are intentionally omitted.  Hashes bind the reviewed inputs while
    # keeping private truth locations out of the aggregate report.
    return {str(key): value for key, value in bindings.items()}


def build_report(
    *,
    release_marker_path: Path,
    freeze_receipt_path: Path,
    truth_descriptor_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    release_marker = validate_release_marker(release_marker_path, freeze_receipt_path)
    freeze_path, freeze = evaluation._load_freeze_receipt(
        freeze_receipt_path,
        strict_record_counts=True,
    )
    truth_path = regular_file(truth_descriptor_path, label="truth descriptor")
    truth_descriptor = evaluation._json_object(truth_path, label="truth descriptor")
    normalized_truth = evaluation._normalise_truth_descriptor(
        truth_descriptor,
        descriptor_path=truth_path,
        freeze_path=freeze_path,
        freeze=freeze,
    )
    truth_cells = normalized_truth["cells"]

    output: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "POSTFREEZE_DESCRIPTIVE_COMPLETE",
        "analysis_scope": "where reconstruction broke by fixed position bin and frozen record ordinal",
        "analysis_type": "post hoc descriptive reporting selected before truth opening",
        "confirmatory_threshold_or_selection_fit": False,
        "truth_opened_after_release_gate": True,
        "source_text_loaded": False,
        "target_weights_loaded": False,
        "position_convention": {
            "stored_positions": "0..127",
            "BOS_position": 0,
            "scored_positions": "1..127",
            "bins_inclusive": [[name, start, end] for name, start, end in POSITION_BINS],
            "record_vectors": "frozen source order; source identifiers and token values omitted",
        },
        "release": {
            "schema": str(release_marker["schema"]),
            "status": str(release_marker["status"]),
            "release_marker_sha256": sha256_file(release_marker_path),
            "freeze_receipt_sha256": sha256_file(freeze_path),
            "truth_descriptor_sha256": sha256_file(truth_path),
        },
        "methods": {},
    }

    b1_cells: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, str | None]]] = {}
    for domain in B1_DOMAINS:
        loaded: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, str | None]]] = {}
        for stage in B1_STAGES:
            loaded[stage] = _load_cell(
                freeze,
                truth_cells,
                method="B1",
                domain=domain,
                stage=stage,
            )
            b1_cells[(domain, stage)] = loaded[stage]
        baseline_prediction, baseline_truth, baseline_valid, baseline_bindings = loaded[0]
        domain_out: dict[str, Any] = {
            "record_count": int(baseline_prediction.shape[0]),
            "stages": {},
        }
        for stage in B1_STAGES:
            prediction, truth, valid, bindings = loaded[stage]
            _bind_truth_consistency(
                baseline_truth,
                baseline_valid,
                truth,
                valid,
                label=f"B1/{domain}",
            )
            if stage == 0:
                summary = _absolute_summary(prediction, truth, valid)
            else:
                summary = _paired_summary(baseline_prediction, prediction, truth, valid)
            summary["input_hashes"] = _cell_binding_record(bindings)
            domain_out["stages"][str(stage)] = summary
        output["methods"].setdefault("B1", {})[domain] = domain_out

    a1_output: dict[str, Any] = {}
    for domain in B1_DOMAINS:
        loaded_a1: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, str | None]]] = {}
        for stage in A1_STAGES:
            loaded_a1[stage] = _load_cell(
                freeze,
                truth_cells,
                method="A1+A2",
                domain=domain,
                stage=stage,
            )
        a1_baseline_prediction, a1_baseline_truth, a1_baseline_valid, _a1_baseline_bindings = loaded_a1[0]
        domain_out = {
            "record_count": int(a1_baseline_prediction.shape[0]),
            "reference": "B1 same-stage first32 records",
            "stages": {},
        }
        for stage in A1_STAGES:
            a1_prediction, a1_truth, a1_valid, a1_bindings = loaded_a1[stage]
            _bind_truth_consistency(
                a1_baseline_truth,
                a1_baseline_valid,
                a1_truth,
                a1_valid,
                label=f"A1+A2/{domain}",
            )
            b1_prediction, b1_truth, b1_valid, _b1_bindings = b1_cells[(domain, stage)]
            b1_prediction = b1_prediction[: int(a1_prediction.shape[0])]
            b1_truth = b1_truth[: int(a1_prediction.shape[0])]
            b1_valid = b1_valid[: int(a1_prediction.shape[0])]
            if not torch.equal(a1_truth, b1_truth):
                raise BreakageError(f"A1+A2/{domain}/stage{stage} truth differs from B1 common32")
            if not torch.equal(a1_valid, b1_valid):
                raise BreakageError(f"A1+A2/{domain}/stage{stage} valid mask differs from B1 common32")
            baseline_for_pair = b1_prediction.eq(a1_truth) & a1_valid
            candidate_correct = a1_prediction.eq(a1_truth) & a1_valid
            broken = baseline_for_pair & ~candidate_correct
            improved = ~baseline_for_pair & candidate_correct
            summary = _absolute_summary(a1_prediction, a1_truth, a1_valid, broken=broken, improved=improved)
            summary["reference_error_position_count"] = int((a1_valid & ~baseline_for_pair).sum().item())
            summary["input_hashes"] = _cell_binding_record(a1_bindings)
            domain_out["stages"][str(stage)] = summary
        # Keep the A1 endpoint-to-endpoint change separate from the B1
        # same-stage contrast so both meanings remain explicit.
        a1_final_prediction, _, a1_final_valid, _ = loaded_a1[256]
        if not torch.equal(a1_baseline_valid, a1_final_valid):
            raise BreakageError(f"A1+A2/{domain} endpoint valid masks differ")
        a1_endpoint_truth = a1_baseline_truth
        domain_out["endpoint_0_to_256"] = _paired_summary(
            a1_baseline_prediction,
            a1_final_prediction,
            a1_endpoint_truth,
            a1_baseline_valid,
        )
        a1_output[domain] = domain_out
    output["methods"]["A1+A2"] = a1_output
    output["elapsed_seconds"] = time.perf_counter() - started
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output["output_sha256"] = sha256_file(output_path)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-marker", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--truth-descriptor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output_path = create_only(args.output, label="post-freeze report")
        result = build_report(
            release_marker_path=regular_file(args.release_marker, label="root truth release marker"),
            freeze_receipt_path=regular_file(args.freeze_receipt, label="freeze receipt"),
            truth_descriptor_path=regular_file(args.truth_descriptor, label="truth descriptor"),
            output_path=output_path,
        )
    except (BreakageError, evaluation.EvaluationError, OSError, RuntimeError, ValueError) as exc:
        print(f"TRR-P12 post-freeze analysis error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output_path),
                "output_sha256": result["output_sha256"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
