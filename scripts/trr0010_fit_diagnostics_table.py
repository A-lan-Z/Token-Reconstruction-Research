"""Assemble the truth-free TRR-0010 four-arm fit diagnostic table.

This module joins the immutable fixed-readout learning-curve table with the
two future directional fit receipts.  It is deliberately a reporting adapter:
it does not load model tensors, source rows, observations, labels, or truth.
The fixed rows are copied from the reviewed v2 table.  A directional input may
be either the task's ``run_receipt.json`` wrapper or the nested A2
``raw_runner_result.json``.  Missing directional receipts remain explicitly
pending; a supplied malformed receipt raises instead of being treated as a
missing result.

The output keeps fitting diagnostics separate from final evaluation.  The
``cost.nonoverlap_wall_cost`` fields report provider preparation, runner wall,
and post-fit export phases once each.  Runner optimizer/stream/validation
times are nested inside runner wall and are retained as phase diagnostics, not
added to it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXED_TABLE = ROOT / "experiments/TRR-0010/report/fixed_fit_learning_curve_cost_table_v2.json"
DEFAULT_FIXED_STATE_BINDING = ROOT / "experiments/TRR-0010/setup/final_method_row_assembly_v1.json"
DEFAULT_OUTPUT = ROOT / "experiments/TRR-0010/report/combined_fit_diagnostics_cost_table_v1.json"

TASK_ID = "TRR-0010"
SCHEMA = "token-reconstruction.trr0010.combined-fit-diagnostics-cost-table.v1"
FIXED_TABLE_SCHEMA = "token-reconstruction.trr0010.fixed-fit-descriptive-table.v2"
EXPECTED_FIXED_TABLE_SHA256 = "4f760524a2d0269861f8424d24ed6d89584f94719c6ecfa888d605bce8924040"
CHECKPOINT_GRID = (0, 1000, 2000, 4000, 8000, 12000, 13000)
ARM_ORDER = (
    "current_fixed",
    "current_directional",
    "expanded_fixed",
    "expanded_directional",
)
ARM_TO_BANK = {
    "current_fixed": "B0",
    "current_directional": "B0",
    "expanded_fixed": "B1",
    "expanded_directional": "B1",
}


class FitTableError(ValueError):
    """Raised when the fit table inputs do not satisfy the frozen schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value: Path | str, *, description: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FitTableError(f"{description} is unavailable: {path}")
    return path


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FitTableError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise FitTableError(f"{description} must be a JSON object: {path}")
    return payload


def _mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FitTableError(f"{description} must be an object")
    return value


def _sequence(value: Any, *, description: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FitTableError(f"{description} must be an array")
    return value


def _optional_number(value: Any, *, description: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FitTableError(f"{description} must be numeric or null")
    return value


def _file_binding(path: Path, *, description: str, expected_sha256: str | None = None) -> dict[str, Any]:
    path = _path(path, description=description)
    actual_sha256 = _sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise FitTableError(
            f"{description} SHA-256 changed: expected {expected_sha256}, got {actual_sha256}"
        )
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual_sha256}


def _declared_artifact(value: Any, *, description: str) -> dict[str, Any]:
    """Copy a declared artifact binding without reading its payload.

    Selected states are model artifacts.  This reporting adapter records their
    declared bytes and digest, but deliberately does not rehash or deserialize
    them.
    """

    record = _mapping(value, description=description)
    path = record.get("path")
    bytes_value = record.get("bytes")
    sha256 = record.get("sha256")
    if not isinstance(path, str) or not path:
        raise FitTableError(f"{description}.path is absent")
    if isinstance(bytes_value, bool) or not isinstance(bytes_value, int) or bytes_value <= 0:
        raise FitTableError(f"{description}.bytes is invalid")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise FitTableError(f"{description}.sha256 is invalid")
    return {
        "path": path,
        "bytes": bytes_value,
        "sha256": sha256,
        **({"readonly": True} if record.get("readonly") is True else {}),
    }


def _load_fixed_table(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _file_binding(path, description="fixed v2 table", expected_sha256=EXPECTED_FIXED_TABLE_SHA256)
    table = _read_json(Path(source["path"]), description="fixed v2 table")
    if table.get("schema") != FIXED_TABLE_SCHEMA:
        raise FitTableError("fixed input is not the reviewed v2 table schema")
    if table.get("task_id") != TASK_ID:
        raise FitTableError("fixed input task identity changed")
    training = _mapping(table.get("training_contract"), description="fixed training contract")
    grid = tuple(int(value) for value in _sequence(training.get("checkpoint_grid"), description="fixed checkpoint grid"))
    if grid != CHECKPOINT_GRID:
        raise FitTableError(f"fixed checkpoint grid changed: {grid!r}")
    checkpoints = _sequence(table.get("checkpoint_table"), description="fixed checkpoint table")
    expected_keys = {(bank, step) for bank in ("B0", "B1") for step in CHECKPOINT_GRID}
    actual_keys: set[tuple[str, int]] = set()
    for row in checkpoints:
        row_map = _mapping(row, description="fixed checkpoint row")
        bank = row_map.get("bank")
        try:
            step = int(row_map.get("step"))
        except (TypeError, ValueError) as exc:
            raise FitTableError("fixed checkpoint step is invalid") from exc
        key = (bank, step)
        if key in actual_keys:
            raise FitTableError(f"duplicate fixed checkpoint row: {key!r}")
        actual_keys.add(key)
    if actual_keys != expected_keys:
        raise FitTableError("fixed checkpoint table does not contain both seven-point curves")
    for key in ("exposure_by_bank", "cost_and_resources_by_bank"):
        rows = _sequence(table.get(key), description=f"fixed {key}")
        banks = {str(_mapping(row, description=f"fixed {key} row").get("bank")) for row in rows}
        if banks != {"B0", "B1"}:
            raise FitTableError(f"fixed {key} is missing B0 or B1")
    return table, source


def _fixed_state_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    binding_path = _path(path, description="fixed state binding")
    payload = _read_json(binding_path, description="fixed state binding")
    methods = payload.get("methods", payload)
    if not isinstance(methods, Mapping):
        raise FitTableError("fixed state binding has no methods map")
    result: dict[str, dict[str, Any]] = {}
    for arm in ("current_fixed", "expanded_fixed"):
        row = methods.get(arm)
        if not isinstance(row, Mapping):
            continue
        state = row.get("state")
        if state is None and isinstance(row.get("resources"), Mapping):
            state = row["resources"].get("base_decoder_state")
        if state is not None:
            result[arm] = _declared_artifact(state, description=f"fixed state {arm}")
            result[arm]["binding_file"] = str(binding_path)
            result[arm]["binding_file_sha256"] = _sha256(binding_path)
            result[arm]["binding_file_bytes"] = binding_path.stat().st_size
    return result


def _fixed_curve(table: Mapping[str, Any], *, arm: str, state_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    bank = ARM_TO_BANK[arm]
    rows = [
        deepcopy(dict(_mapping(row, description="fixed checkpoint row")))
        for row in _sequence(table["checkpoint_table"], description="fixed checkpoint table")
        if _mapping(row, description="fixed checkpoint row").get("bank") == bank
    ]
    rows.sort(key=lambda row: int(row["step"]))
    if tuple(int(row["step"]) for row in rows) != CHECKPOINT_GRID:
        raise FitTableError(f"fixed {arm} curve is incomplete")
    selected = [row for row in rows if row.get("selected") is True]
    if len(selected) != 1:
        raise FitTableError(f"fixed {arm} selected checkpoint is ambiguous")
    selected_row = selected[0]
    selected_state = state_rows.get(arm)
    if selected_state is not None:
        expected_physical = selected_row.get("checkpoint_file_sha256")
        if selected_state.get("sha256") != expected_physical:
            raise FitTableError(f"fixed {arm} selected state digest disagrees with v2 table")
        selected_state = deepcopy(dict(selected_state))
        selected_state["logical_sha256"] = selected_row.get("state_logical_sha256")
        selected_state["selected_step"] = int(selected_row["step"])
    exposure_rows = {
        str(_mapping(row, description="fixed exposure row")["bank"]): deepcopy(dict(row))
        for row in _sequence(table["exposure_by_bank"], description="fixed exposure rows")
    }
    cost_rows = {
        str(_mapping(row, description="fixed cost row")["bank"]): deepcopy(dict(row))
        for row in _sequence(table["cost_and_resources_by_bank"], description="fixed cost rows")
    }
    exposure = exposure_rows[bank]
    cost = cost_rows[bank]
    nested_phases = {
        key: cost.get(key)
        for key in ("optimizer_update_seconds", "stream_load_seconds", "validation_seconds", "fitting_diagnostics_seconds")
    }
    return {
        "arm": arm,
        "bank": bank,
        "method_role": f"{arm}_readout",
        "status": "COMPLETE_FIXED_V2_BOUND",
        "fit_started": True,
        "truth_opened": False,
        "selected_step": int(selected_row["step"]),
        "selected_state": selected_state,
        "checkpoint_curve": rows,
        "diagnostics": [
            {
                "step": int(row["step"]),
                "frozen64": deepcopy(row.get("diagnostics", {}).get("frozen64")),
                "full_bank_endpoint": deepcopy(row.get("diagnostics", {}).get("full_bank_endpoint")),
            }
            for row in rows
        ],
        "exposure": exposure,
        "cost": {
            "process_wall_seconds": cost.get("process_wall_seconds"),
            "external_watchdog_seconds": cost.get("external_watchdog_seconds"),
            "nested_phase_seconds": nested_phases,
            "nonoverlap_wall_cost": {
                "measured_total_seconds": cost.get("process_wall_seconds"),
                "components": [
                    {"name": "fit_process_wall", "seconds": cost.get("process_wall_seconds"), "scope": "one process wall measurement"}
                ],
                "phase_sum_is_not_added": True,
                "note": "Optimizer, stream-load, validation, and diagnostic times are nested phase accounting; they are retained but not added to process wall.",
            },
            "raw": cost,
        },
        "source": {"fixed_table_bank": bank, "fixed_table_checkpoint_rows": len(rows)},
    }


def _find_artifact(value: Any, *, description: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _declared_artifact(value, description=description)


def _optional_int(value: Any, *, description: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise FitTableError(f"{description} must be an integer or null")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FitTableError(f"{description} must be an integer or null") from exc


def _directional_payload(path: Path, *, expected_arm: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = _file_binding(path, description=f"directional {expected_arm} receipt")
    payload = _read_json(Path(source["path"]), description=f"directional {expected_arm} receipt")
    if payload.get("task_id") not in (None, TASK_ID):
        raise FitTableError(f"directional {expected_arm} task identity changed")
    if payload.get("truth_opened") is True:
        raise FitTableError(f"directional {expected_arm} receipt opened truth")
    arm: Mapping[str, Any] | None = None
    runner: Mapping[str, Any] | None = None
    wrapper = payload.get("arms")
    if isinstance(wrapper, Mapping):
        candidate = wrapper.get(expected_arm)
        if not isinstance(candidate, Mapping):
            raise FitTableError(f"directional {expected_arm} wrapper has no arm entry")
        arm = candidate
        runner_value = candidate.get("runner_result")
        if isinstance(runner_value, Mapping):
            runner = runner_value
    elif isinstance(payload.get("learning_curve"), Sequence) and not isinstance(payload.get("learning_curve"), (str, bytes)):
        # This is a raw A2 runner result.  It has no post-fit export receipt.
        runner = payload
        arm = {"arm_name": expected_arm, "bank_role": ARM_TO_BANK[expected_arm]}
    else:
        raise FitTableError(f"directional {expected_arm} receipt is neither a run wrapper nor a raw runner result")
    if runner is None:
        raise FitTableError(f"directional {expected_arm} runner_result is absent")
    if runner.get("status") != "COMPLETED":
        raise FitTableError(f"directional {expected_arm} runner status is not COMPLETED")
    if arm is None:
        raise FitTableError(f"directional {expected_arm} arm entry is absent")
    if arm.get("truth_opened") is True:
        raise FitTableError(f"directional {expected_arm} arm receipt opened truth")
    if arm.get("arm_name", expected_arm) != expected_arm:
        raise FitTableError(f"directional {expected_arm} arm identity changed")
    if arm.get("bank_role", ARM_TO_BANK[expected_arm]) != ARM_TO_BANK[expected_arm]:
        raise FitTableError(f"directional {expected_arm} bank role changed")
    return dict(payload), dict(arm), {**dict(runner), "_source": source}


def _directional_curve(runner: Mapping[str, Any], *, arm: str) -> list[dict[str, Any]]:
    points = _sequence(runner.get("learning_curve"), description=f"directional {arm} learning curve")
    if tuple(int(_mapping(point, description=f"directional {arm} curve point").get("step")) for point in points) != CHECKPOINT_GRID:
        raise FitTableError(f"directional {arm} learning curve does not match the frozen grid")
    return [deepcopy(dict(_mapping(point, description=f"directional {arm} curve point"))) for point in points]


def _directional_diagnostics(arm_record: Mapping[str, Any], *, arm: str) -> tuple[list[dict[str, Any]], str | None]:
    events = arm_record.get("diagnostic_events")
    if events is None:
        return [], "directional run receipt has no post-fit diagnostic_events"
    events_seq = _sequence(events, description=f"directional {arm} diagnostic_events")
    normalized: list[dict[str, Any]] = []
    for event in events_seq:
        item = _mapping(event, description=f"directional {arm} diagnostic event")
        try:
            step = int(item.get("step"))
        except (TypeError, ValueError) as exc:
            raise FitTableError(f"directional {arm} diagnostic step is invalid") from exc
        fit_records = _optional_int(item.get("fit_records"), description=f"directional {arm} fit_records")
        if fit_records != 64:
            raise FitTableError(f"directional {arm} frozen-64 diagnostic has {fit_records!r} records")
        fit_digest = item.get("fit_records_sha256")
        if not isinstance(fit_digest, str) or len(fit_digest) != 64:
            raise FitTableError(f"directional {arm} frozen-64 diagnostic digest is invalid")
        untouched = item.get("selection_metric_untouched")
        if untouched is not True:
            raise FitTableError(f"directional {arm} diagnostic changes selection semantics")
        full_bank = item.get("full_bank")
        if full_bank is not None:
            full_bank = dict(_mapping(full_bank, description=f"directional {arm} full-bank diagnostic"))
            if full_bank.get("endpoint") not in {"start", "end"}:
                raise FitTableError(f"directional {arm} full-bank endpoint is invalid")
        normalized.append({
            "step": step,
            "fit_records": fit_records,
            "fit_records_sha256": fit_digest,
            "full_bank_endpoint": deepcopy(full_bank),
            "selection_metric_untouched": untouched,
            "elapsed_seconds": item.get("elapsed_seconds"),
        })
    normalized.sort(key=lambda item: item["step"])
    if tuple(item["step"] for item in normalized) != CHECKPOINT_GRID:
        raise FitTableError(f"directional {arm} diagnostics do not cover the frozen grid")
    endpoints = {item["full_bank_endpoint"].get("endpoint") for item in normalized if isinstance(item.get("full_bank_endpoint"), Mapping)}
    if endpoints != {"start", "end"}:
        return normalized, "directional diagnostics do not contain both full-bank start/end endpoints"
    return normalized, None


def _exposure_from_runner(runner: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    schedule = runner.get("schedule")
    schedule_map = _mapping(schedule, description=f"directional {arm} schedule")
    exposure = schedule_map.get("exposure", runner.get("exposure"))
    exposure_map = dict(_mapping(exposure, description=f"directional {arm} schedule exposure"))
    if int(schedule_map.get("steps", -1)) != 13000:
        raise FitTableError(f"directional {arm} schedule steps changed")
    exposure_map.setdefault("steps", schedule_map.get("steps"))
    exposure_map.setdefault("seed", schedule_map.get("seed"))
    exposure_map.setdefault("schedule_semantic_sha256", schedule_map.get("semantic_sha256"))
    # Keep all producer fields and make the common names explicit.  A2's
    # runner calls repeated pairs a count of pairs; the public audit may also
    # provide repeated draws/sample records, so do not conflate them.
    for key in (
        "total_draws",
        "unique_record_position_pairs",
        "repeated_record_position_pairs",
        "max_exposures_per_record_position",
        "used_replacement_steps",
        "draws_per_step",
        "repeated_record_position_draws",
        "sampled_records",
        "record_batch_slots",
    ):
        if key in exposure_map:
            _optional_int(exposure_map[key], description=f"directional {arm} exposure {key}")
    exposure_map["exposure_source"] = "runner_result.schedule.exposure"
    return exposure_map


def _support_count(*values: Any) -> int | None:
    keys = ("support_token_ids", "support_count", "supported_token_count", "support_ids_count")
    for value in values:
        if not isinstance(value, Mapping):
            continue
        for key in keys:
            if value.get(key) is not None:
                return _optional_int(value[key], description=f"support count {key}")
    return None


def _directional_state(arm_record: Mapping[str, Any], *, arm: str, runner: Mapping[str, Any]) -> dict[str, Any] | None:
    selected = _find_artifact(arm_record.get("selected_checkpoint"), description=f"directional {arm} selected checkpoint")
    if selected is None:
        selected = _find_artifact(arm_record.get("selected_state"), description=f"directional {arm} selected state")
    base = _find_artifact(arm_record.get("base_decoder_state"), description=f"directional {arm} base decoder state")
    readout = _find_artifact(arm_record.get("effective_readout"), description=f"directional {arm} effective readout")
    selected_step = runner.get("selected_step")
    if selected is None and base is None and readout is None:
        return None
    result: dict[str, Any] = {
        "selected_step": _optional_int(selected_step, description=f"directional {arm} selected_step"),
        "selected_checkpoint": selected,
        "base_decoder_state": base,
        "effective_readout_w": readout,
        "footprint_scope": "declared serialized artifacts; runtime tensors and process peaks are separate",
    }
    declared = [item["bytes"] for item in (base, readout) if item is not None]
    result["deployed_base_plus_effective_readout_bytes"] = sum(declared) if len(declared) == 2 else None
    return result


def _directional_cost(payload: Mapping[str, Any], arm_record: Mapping[str, Any], runner: Mapping[str, Any], diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    timing = dict(_mapping(runner.get("timing", {}), description="directional runner timing"))
    arm_timing = arm_record.get("timing")
    if arm_timing is not None:
        timing.update(dict(_mapping(arm_timing, description="directional wrapper timing")))
    # The wrapper adds preparation and post-fit exports outside runner wall.
    # A single arm elapsed value is optional; never manufacture it by adding
    # nested phase values to wrapper_seconds.
    provider = timing.get("provider_preparation_seconds")
    runner_wall = timing.get("whole_wall_seconds")
    restore = timing.get("restore_export_seconds")
    base_export = timing.get("base_decoder_export_seconds")
    explicit_components = {
        "provider_preparation_seconds": _optional_number(provider, description="directional provider preparation"),
        "fit_runner_wall_seconds": _optional_number(runner_wall, description="directional runner wall"),
        "post_fit_restore_export_seconds": _optional_number(restore, description="directional restore/export"),
        "post_fit_base_decoder_export_seconds": _optional_number(base_export, description="directional base export"),
    }
    measured = [value for value in explicit_components.values() if value is not None]
    receipt_elapsed = payload.get("elapsed_seconds")
    return {
        "process_wall_seconds": _optional_number(runner_wall, description="directional process wall"),
        "runner_call_seconds": _optional_number(timing.get("runner_call_seconds"), description="directional runner call"),
        "provider_preparation_seconds": explicit_components["provider_preparation_seconds"],
        "restore_export_seconds": explicit_components["post_fit_restore_export_seconds"],
        "base_decoder_export_seconds": explicit_components["post_fit_base_decoder_export_seconds"],
        "nested_phase_seconds": {
            key: timing.get(key)
            for key in ("optimizer_update_seconds", "stream_load_seconds", "validation_seconds")
            if timing.get(key) is not None
        },
        "fitting_diagnostics_seconds": sum(
            float(event["elapsed_seconds"])
            for event in diagnostics
            if isinstance(event.get("elapsed_seconds"), (int, float)) and not isinstance(event.get("elapsed_seconds"), bool)
        ) if diagnostics else None,
        "wrapper_seconds": _optional_number(timing.get("wrapper_seconds"), description="directional wrapper seconds"),
        "receipt_elapsed_seconds": _optional_number(receipt_elapsed, description="directional receipt elapsed"),
        "nonoverlap_wall_cost": {
            "components": [
                {"name": name, "seconds": seconds, "scope": "counted once"}
                for name, seconds in explicit_components.items()
                if seconds is not None
            ],
            "sum_explicit_components_seconds": sum(measured) if len(measured) == len(explicit_components) else None,
            "phase_sum_is_not_added": True,
            "note": "Provider preparation, runner process wall, and post-fit exports are separate components. Optimizer/stream/validation phases are nested inside runner wall; wrapper_seconds is not added again.",
        },
        "raw_timing": timing,
    }


def _directional_arm(path: Path, *, arm: str) -> dict[str, Any]:
    payload, arm_record, runner = _directional_payload(path, expected_arm=arm)
    curve = _directional_curve(runner, arm=arm)
    diagnostics, diagnostic_pending = _directional_diagnostics(arm_record, arm=arm)
    exposure = _exposure_from_runner(runner, arm=arm)
    state = _directional_state(arm_record, arm=arm, runner=runner)
    missing_fields: list[str] = []
    if diagnostic_pending:
        missing_fields.append("full_and_frozen64_diagnostics")
    if state is None:
        missing_fields.append("selected_state_footprints")
    support_count = _support_count(arm_record, runner, exposure)
    if support_count is None:
        missing_fields.append("support_token_ids")
    status = "COMPLETE_DIRECTIONAL_RECEIPT" if not missing_fields else "COMPLETE_WITH_METADATA_PENDING"
    return {
        "arm": arm,
        "bank": ARM_TO_BANK[arm],
        "method_role": f"{arm}_directional_readout",
        "status": status,
        "fit_started": True,
        "truth_opened": False,
        "selected_step": runner.get("selected_step"),
        "selected_state": state,
        "checkpoint_curve": curve,
        "diagnostics": diagnostics,
        "exposure": {**exposure, "support_token_ids": support_count},
        "cost": _directional_cost(payload, arm_record, runner, diagnostics),
        "source": {
            "receipt": runner.get("_source"),
            "raw_runner_status": runner.get("status"),
            "missing_fields": missing_fields,
            **({"diagnostic_pending_reason": diagnostic_pending} if diagnostic_pending else {}),
        },
    }


def _pending_arm(arm: str, *, requested_path: Path | None) -> dict[str, Any]:
    return {
        "arm": arm,
        "bank": ARM_TO_BANK[arm],
        "method_role": f"{arm}_directional_readout",
        "status": "PENDING_MISSING_RECEIPT",
        "fit_started": False,
        "truth_opened": False,
        "selected_step": None,
        "selected_state": None,
        "checkpoint_curve": [],
        "diagnostics": [],
        "exposure": None,
        "cost": None,
        "source": {
            "requested_receipt_path": None if requested_path is None else str(requested_path),
            "missing_reason": "directional fit receipt has not been supplied",
        },
    }


def _summary_row(arm: Mapping[str, Any]) -> dict[str, Any]:
    exposure = arm.get("exposure") if isinstance(arm.get("exposure"), Mapping) else {}
    cost = arm.get("cost") if isinstance(arm.get("cost"), Mapping) else {}
    state = arm.get("selected_state") if isinstance(arm.get("selected_state"), Mapping) else {}
    nonoverlap = cost.get("nonoverlap_wall_cost") if isinstance(cost.get("nonoverlap_wall_cost"), Mapping) else {}
    return {
        "arm": arm.get("arm"),
        "status": arm.get("status"),
        "selected_step": arm.get("selected_step"),
        "updates": exposure.get("steps"),
        "total_draws": exposure.get("total_draws"),
        "available_post_bos_positions": exposure.get("available_post_bos_positions"),
        "sampled_unique_record_position_pairs": exposure.get("actual_sampled_unique_record_position_pairs", exposure.get("unique_record_position_pairs")),
        "repeated_record_position_draws": exposure.get("actual_sampled_repeat_draws", exposure.get("repeated_record_position_draws")),
        "repeated_record_position_pairs": exposure.get("repeated_record_position_pairs"),
        "support_token_ids": exposure.get("support_token_ids"),
        "fit_wall_seconds": cost.get("process_wall_seconds"),
        "nonoverlap_wall_seconds": nonoverlap.get("sum_explicit_components_seconds", nonoverlap.get("measured_total_seconds")),
        "selected_state_bytes": state.get("bytes") if "bytes" in state else None,
    }


def assemble_table(
    *,
    fixed_table_path: Path | str = DEFAULT_FIXED_TABLE,
    current_directional_path: Path | str | None = None,
    expanded_directional_path: Path | str | None = None,
    fixed_state_binding_path: Path | str | None = DEFAULT_FIXED_STATE_BINDING,
) -> dict[str, Any]:
    """Return the combined table without writing a file."""

    fixed_path = _path(fixed_table_path, description="fixed v2 table")
    table, fixed_source = _load_fixed_table(fixed_path)
    state_path: Path | None = None
    if fixed_state_binding_path is not None:
        candidate = Path(fixed_state_binding_path).expanduser()
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.exists():
            state_path = candidate
    state_rows = _fixed_state_rows(state_path)
    arms: dict[str, dict[str, Any]] = {
        "current_fixed": _fixed_curve(table, arm="current_fixed", state_rows=state_rows),
        "expanded_fixed": _fixed_curve(table, arm="expanded_fixed", state_rows=state_rows),
    }
    requested = {
        "current_directional": None if current_directional_path is None else Path(current_directional_path),
        "expanded_directional": None if expanded_directional_path is None else Path(expanded_directional_path),
    }
    for arm in ("current_directional", "expanded_directional"):
        path = requested[arm]
        if path is None:
            arms[arm] = _pending_arm(arm, requested_path=None)
        else:
            arms[arm] = _directional_arm(_path(path, description=f"directional {arm} receipt"), arm=arm)
    ordered_arms = {arm: arms[arm] for arm in ARM_ORDER}
    pending = [arm for arm in ARM_ORDER if ordered_arms[arm]["status"].startswith("PENDING") or "PENDING" in ordered_arms[arm]["status"]]
    status = "COMPLETE_FOUR_ARM_FIT_DIAGNOSTICS" if not pending else "FIXED_ARMS_BOUND_DIRECTIONAL_ARMS_PENDING"
    return {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": status,
        "truth_opened": False,
        "final_quality_available": False,
        "checkpoint_grid": list(CHECKPOINT_GRID),
        "arm_order": list(ARM_ORDER),
        "source_bindings": {
            "fixed_fit_table_v2": fixed_source,
            "fixed_state_binding": None if state_path is None else _file_binding(state_path, description="fixed state binding"),
        },
        "arms": ordered_arms,
        "summary_rows": [_summary_row(ordered_arms[arm]) for arm in ARM_ORDER],
        "accounting": {
            "wall_cost_nonoverlap": True,
            "scope": "truth-free public fitting and checkpoint diagnostics",
            "note": "Fixed process wall and directional provider/runner/export components are measured once. Nested optimizer, stream-load, validation, and diagnostic phase times are retained for readability and are not added to process wall; wrapper elapsed is not added again.",
            "missing_directional_receipts": pending,
        },
        "quality_boundary": {
            "fit_diagnostics_only": True,
            "truth_opened": False,
            "final_evaluation_started": False,
            "final_quality_result_available": False,
            "interpretation": "Fit curves, support/exposure counts, serialized state footprints, and costs do not establish fresh reconstruction quality.",
        },
        "reproducibility": {
            "assembler": "scripts/trr0010_fit_diagnostics_table.py",
            "fixed_table_sha256": fixed_source["sha256"],
            "directional_inputs_are_create_only": True,
            "directional_input_contract": "run_receipt.json wrapper or raw_runner_result.json with the frozen seven-point curve; malformed supplied receipts fail closed",
        },
    }


def write_create_only(payload: Mapping[str, Any], output: Path | str) -> Path:
    destination = Path(output).expanduser()
    if not destination.is_absolute():
        destination = ROOT / destination
    destination = destination.resolve()
    try:
        destination.relative_to((ROOT / "experiments" / TASK_ID).resolve())
    except ValueError as exc:
        raise FitTableError("combined table output must remain under experiments/TRR-0010") from exc
    if destination.exists() or destination.is_symlink():
        raise FitTableError(f"combined table output is create-only: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-table", type=Path, default=DEFAULT_FIXED_TABLE)
    parser.add_argument("--fixed-state-binding", type=Path, default=DEFAULT_FIXED_STATE_BINDING)
    parser.add_argument("--current-directional", type=Path)
    parser.add_argument("--expanded-directional", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        payload = assemble_table(
            fixed_table_path=args.fixed_table,
            current_directional_path=args.current_directional,
            expanded_directional_path=args.expanded_directional,
            fixed_state_binding_path=args.fixed_state_binding,
        )
        destination = write_create_only(payload, args.output)
    except (FitTableError, OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": payload["status"], "output": str(destination), "pending": payload["accounting"]["missing_directional_receipts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
