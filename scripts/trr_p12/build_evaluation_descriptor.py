#!/usr/bin/env python3
"""Assemble the strict, pretruth TRR-P12 evaluation descriptor.

This binder reads only JSON receipts and file bytes.  It does not load tensor
payloads, source text, target weights, or evaluator truth.  B1 uses all four
paired stages; A1+A2 binds the endpoint predictions and candidate receipts for
the first 32 panel rows per domain.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trr_p12 import capture as public_capture


TASK_ID = "TRR-P12"
FREEZE_SCHEMA = "token-reconstruction.trr-p12-evaluation-freeze.v1"
DOMAINS = ("pile", "finance")
B1_STAGES = (0, 64, 128, 256)
A1_STAGES = (0, 256)
B1_RECORDS = 128
A1_RECORDS = 32
GEOMETRY_SCHEMA = "token-reconstruction.trr-p12-geometry-pipeline.v1"
FORECAST_SCHEMA = "token-reconstruction.trr-p12-direction-forecast-artifact.v1"
B1_RUN_SCHEMA = "token-reconstruction.trr-p12-b1-runner-receipt.v1"
B1_GEOMETRY_SCHEMA = "token-reconstruction.trr-p12-b1-compact-geometry.v1"
CAPTURE_SCHEMA = "token-reconstruction.trr-p12-public-capture.v1"
ORDER_SCHEMA = "token-reconstruction.trr-p12-public-order.v1"
A1_RUN_SCHEMA = "token-reconstruction.trr-p12-a1-run.v1"


class DescriptorError(RuntimeError):
    """Raised when a registered P12 input is absent or changed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = base / raw
    return raw.resolve()


def file_binding(path: str | Path, *, label: str, base: Path = ROOT) -> dict[str, Any]:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = base / raw
    if raw.is_symlink():
        raise DescriptorError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise DescriptorError(f"{label} is not a regular file: {resolved}")
    return {
        "label": label,
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def declared_binding(value: Any, *, label: str, base: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
        raise DescriptorError(f"{label} lacks a path binding")
    result = file_binding(value["path"], label=label, base=base)
    declared = value.get("sha256")
    if declared is not None and str(declared).lower() != result["sha256"]:
        raise DescriptorError(f"{label} SHA-256 changed")
    if value.get("bytes") is not None and int(value["bytes"]) != result["bytes"]:
        raise DescriptorError(f"{label} byte count changed")
    for key in ("key", "format", "role"):
        if key in value:
            result[key] = str(value[key])
    return result


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DescriptorError(f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DescriptorError(f"{label} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise DescriptorError(f"{label} must be an object")
    return dict(value)


def require(value: Any, *, label: str) -> Any:
    if value is None:
        raise DescriptorError(f"{label} is missing")
    return value


def require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DescriptorError(f"{label} must be an object")
    return value


def require_digest(value: Any, *, label: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise DescriptorError(f"{label} is not a SHA-256 digest")
    return result


def ordered_digest(values: Sequence[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def record_ids_digest(values: Sequence[str]) -> str:
    """Digest used by public capture order receipts (newline-delimited IDs)."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def runner_record_ids_digest(values: Sequence[str]) -> str:
    """Digest used by the P11 A1 runtime input manifest (canonical JSON IDs)."""
    payload = json.dumps(list(values), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_false_flags(value: Mapping[str, Any], *, label: str, flags: Sequence[str]) -> None:
    for flag in flags:
        if value.get(flag) is not False:
            raise DescriptorError(f"{label} violates {flag}=false")


def panel_bindings(panel_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    try:
        panel, pile_bindings, pile_rows = public_capture.load_panel(panel_path, "pile")
        _panel_again, finance_bindings, finance_rows = public_capture.load_panel(panel_path, "finance")
    except Exception as exc:
        raise DescriptorError("frozen P12 panel validation failed") from exc
    if pile_bindings["panel"]["sha256"] != finance_bindings["panel"]["sha256"]:
        raise DescriptorError("panel binding differs between domains")
    if pile_bindings["source_inputs"] != finance_bindings["source_inputs"]:
        raise DescriptorError("source-input binding differs between domains")
    rows = {"pile": pile_rows, "finance": finance_rows}
    orders: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        record_ids = [str(row["record_id"]) for row in rows[domain]]
        h128 = [require_digest(row["h128_sequence_sha256"], label=f"panel {domain} H128") for row in rows[domain]]
        if len(record_ids) != B1_RECORDS or len(set(record_ids)) != B1_RECORDS:
            raise DescriptorError(f"panel {domain} record order is not unique 128 rows")
        if len(h128) != B1_RECORDS or len(set(h128)) != B1_RECORDS:
            raise DescriptorError(f"panel {domain} H128 order is not unique 128 rows")
        orders[domain] = {
            "record_ids": record_ids,
            "h128": h128,
            "record_ids_sha256": record_ids_digest(record_ids),
            "h128_order_sha256": ordered_digest(h128),
            "source_order_sha256": ordered_digest(h128),
            "first32_record_ids_sha256": record_ids_digest(record_ids[:A1_RECORDS]),
            "first32_h128_order_sha256": ordered_digest(h128[:A1_RECORDS]),
        }
    if pile_bindings["panel"]["sha256"] != public_capture.PANEL_SHA256:
        raise DescriptorError("panel is not the frozen P12 release")
    if panel.get("task_id") != TASK_ID or panel.get("status") != "FROZEN_P12_SOURCE_PANEL_NO_TRUTH":
        raise DescriptorError("panel identity or status changed")
    return panel, pile_bindings["panel"], rows, {"source_inputs": pile_bindings["source_inputs"], **orders}


def same_binding_path(left: Mapping[str, Any], right: Mapping[str, Any], *, label: str) -> None:
    if resolve_path(str(left.get("path"))) != resolve_path(str(right.get("path"))):
        raise DescriptorError(f"{label} path differs")
    if str(left.get("sha256", "")).lower() != str(right.get("sha256", "")).lower():
        raise DescriptorError(f"{label} SHA-256 differs")


def load_capture_cell(
    capture_root: Path,
    *,
    domain: str,
    stage: int,
    panel_binding: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    order: Mapping[str, Any],
) -> dict[str, Any]:
    cell_root = capture_root / f"{domain}-stage{stage:04d}"
    capture_path = cell_root / "capture.json"
    order_path = cell_root / "order.json"
    observation_path = cell_root / "observation.safetensors"
    capture = read_json(capture_path, label=f"capture {domain}/{stage}")
    capture_binding = file_binding(capture_path, label=f"capture receipt {domain}/{stage}")
    if capture.get("schema") != CAPTURE_SCHEMA or capture.get("task_id") != TASK_ID:
        raise DescriptorError(f"capture {domain}/{stage} schema or task identity changed")
    if capture.get("status") != "PUBLIC_OBSERVATION_CELL_COMPLETE_NO_TRUTH":
        raise DescriptorError(f"capture {domain}/{stage} is not complete before truth")
    assert_false_flags(capture.get("contract", {}), label=f"capture {domain}/{stage} contract", flags=("truth_opened", "source_text_or_token_ids_written"))
    assert_false_flags(capture.get("execution", {}), label=f"capture {domain}/{stage} execution", flags=("truth_opened", "token_ids_written", "source_text_written", "target_weights_written"))
    if capture.get("panel", {}).get("sha256") != panel_binding["sha256"]:
        raise DescriptorError(f"capture {domain}/{stage} panel binding differs")
    if capture.get("source_inputs", {}).get("sha256") != source_binding["sha256"]:
        raise DescriptorError(f"capture {domain}/{stage} source-input binding differs")
    observation = declared_binding(capture.get("observation"), label=f"capture {domain}/{stage} observation", base=capture_path.parent)
    order_binding = declared_binding(capture.get("order"), label=f"capture {domain}/{stage} order", base=capture_path.parent)
    if resolve_path(observation["path"]) != observation_path.resolve():
        raise DescriptorError(f"capture {domain}/{stage} observation path differs")
    if resolve_path(order_binding["path"]) != order_path.resolve():
        raise DescriptorError(f"capture {domain}/{stage} order path differs")
    order_payload = read_json(order_path, label=f"capture order {domain}/{stage}")
    if order_payload.get("schema") != ORDER_SCHEMA or order_payload.get("task_id") != TASK_ID:
        raise DescriptorError(f"capture order {domain}/{stage} schema changed")
    if order_payload.get("status") != "FROZEN_PUBLIC_ORDER_NO_PAYLOAD":
        raise DescriptorError(f"capture order {domain}/{stage} status changed")
    assert_false_flags(order_payload, label=f"capture order {domain}/{stage}", flags=("truth_opened",))
    if order_payload.get("panel_sha256") != panel_binding["sha256"] or order_payload.get("domain") != domain or int(order_payload.get("stage_updates", -1)) != stage:
        raise DescriptorError(f"capture order {domain}/{stage} identity differs")
    ordered_h128 = [require_digest(item, label=f"capture order {domain}/{stage} H128") for item in require(order_payload.get("ordered_h128_sequence_sha256"), label=f"capture order {domain}/{stage} H128 order")]
    ordered_ids = [str(item) for item in require(order_payload.get("ordered_record_ids"), label=f"capture order {domain}/{stage} record order")]
    if ordered_h128 != list(order["h128"]) or ordered_ids != list(order["record_ids"]):
        raise DescriptorError(f"capture order {domain}/{stage} differs from the frozen panel")
    if order_payload.get("h128_order_sha256") != ordered_digest(ordered_h128):
        raise DescriptorError(f"capture order {domain}/{stage} H128 digest differs")
    if order_payload.get("record_ids_sha256") != record_ids_digest(ordered_ids):
        raise DescriptorError(f"capture order {domain}/{stage} record digest differs")
    if capture.get("observation", {}).get("sha256") != observation["sha256"] or capture.get("order", {}).get("sha256") != order_binding["sha256"]:
        raise DescriptorError(f"capture {domain}/{stage} receipt output hashes differ")
    contract = require_mapping(capture.get("contract"), label=f"capture {domain}/{stage} contract")
    if contract.get("stored_sequence_tokens") != 128 or contract.get("hidden_size") != 2048 or contract.get("activation_dtype") != "bfloat16":
        raise DescriptorError(f"capture {domain}/{stage} geometry contract changed")
    return {
        "root": cell_root,
        "capture": capture,
        "capture_binding": capture_binding,
        "observation": observation,
        "order_binding": order_binding,
        "order_payload": order_payload,
    }


def normalize_code_files(value: Mapping[str, Any], *, label: str, base: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for name, raw in value.items():
        if name in {"import_policy", "loaded_decoder_modules"}:
            continue
        if isinstance(raw, Mapping) and isinstance(raw.get("path"), str):
            files[str(name)] = declared_binding(raw, label=f"{label}/{name}", base=base)
    if not files:
        raise DescriptorError(f"{label} has no file bindings")
    return files


def load_b1_cell(
    b1_root: Path,
    capture: Mapping[str, Any],
    *,
    domain: str,
    stage: int,
    panel_binding: Mapping[str, Any],
) -> dict[str, Any]:
    cell_root = b1_root / f"{domain}-stage{stage:04d}"
    receipt_path = cell_root / "receipt.json"
    prediction_path = cell_root / "predictions.safetensors"
    projected_path = cell_root / "projected.safetensors"
    geometry_path = cell_root / "geometry.json"
    receipt = read_json(receipt_path, label=f"B1 receipt {domain}/{stage}")
    receipt_binding = file_binding(receipt_path, label=f"B1 receipt {domain}/{stage}")
    if receipt.get("schema") != B1_RUN_SCHEMA or receipt.get("task_id") != TASK_ID:
        raise DescriptorError(f"B1 receipt {domain}/{stage} schema or task identity changed")
    if receipt.get("status") != "PREDICTIONS_AND_PROJECTED_FEATURES_GENERATED":
        raise DescriptorError(f"B1 receipt {domain}/{stage} is not complete")
    assert_false_flags(receipt.get("access_boundary", {}), label=f"B1 receipt {domain}/{stage} access boundary", flags=("truth_opened", "source_text_loaded", "token_ids_loaded", "target_weights_loaded"))
    observation = require_mapping(receipt.get("observations"), label=f"B1 receipt {domain}/{stage} observation")
    capture_observation = capture["observation"]
    if str(observation.get("sha256", "")).lower() != capture_observation["sha256"] or resolve_path(str(observation.get("path"))) != resolve_path(capture_observation["path"]):
        raise DescriptorError(f"B1 receipt {domain}/{stage} observation differs from capture")
    outputs = require_mapping(receipt.get("outputs"), label=f"B1 receipt {domain}/{stage} outputs")
    prediction = declared_binding(outputs.get("predictions"), label=f"B1 prediction {domain}/{stage}", base=receipt_path.parent)
    projected = declared_binding(outputs.get("projected_hidden"), label=f"B1 projected hidden {domain}/{stage}", base=receipt_path.parent)
    geometry = declared_binding(outputs.get("compact_geometry"), label=f"B1 compact geometry {domain}/{stage}", base=receipt_path.parent)
    if resolve_path(prediction["path"]) != prediction_path.resolve() or resolve_path(projected["path"]) != projected_path.resolve() or resolve_path(geometry["path"]) != geometry_path.resolve():
        raise DescriptorError(f"B1 receipt {domain}/{stage} output path differs")
    geometry_payload = read_json(geometry_path, label=f"B1 geometry {domain}/{stage}")
    if geometry_payload.get("schema") != B1_GEOMETRY_SCHEMA or geometry_payload.get("task_id") != TASK_ID:
        raise DescriptorError(f"B1 geometry {domain}/{stage} schema changed")
    if geometry_payload.get("record_count") != B1_RECORDS or geometry_payload.get("prediction_output_geometry") != [B1_RECORDS, 128] or geometry_payload.get("projected_output_geometry") != [B1_RECORDS, 127, 2048]:
        raise DescriptorError(f"B1 geometry {domain}/{stage} shape changed")
    assert_false_flags(geometry_payload, label=f"B1 geometry {domain}/{stage}", flags=("truth_opened", "source_text_loaded", "token_ids_loaded"))
    code = normalize_code_files(require_mapping(receipt.get("code"), label=f"B1 receipt {domain}/{stage} code"), label=f"B1 code {domain}/{stage}", base=receipt_path.parent)
    package = require_mapping(receipt.get("package"), label=f"B1 package {domain}/{stage}")
    package_bindings = {
        "descriptor": declared_binding(package.get("descriptor"), label=f"B1 package descriptor {domain}/{stage}", base=receipt_path.parent),
        "b1_state": declared_binding(package.get("b1_state"), label=f"B1 package state {domain}/{stage}", base=receipt_path.parent),
        "public_readout": declared_binding(package.get("public_readout"), label=f"B1 package readout {domain}/{stage}", base=receipt_path.parent),
    }
    return {
        "receipt": receipt,
        "receipt_binding": receipt_binding,
        "prediction": {**prediction, "key": "expanded_fixed", "role": "B1 prediction"},
        "projected": {**projected, "key": "projected_hidden", "role": "B1 projected hidden"},
        "geometry": {**geometry, "role": "B1 compact geometry"},
        "code": code,
        "package": package_bindings,
        "costs": {
            "receipt": receipt_binding,
            "device": receipt.get("device"),
            "numeric_profile": receipt.get("numeric_profile"),
            "timing": receipt.get("timing"),
            "resource": receipt.get("resource"),
        },
        "panel_sha256": panel_binding["sha256"],
    }


def pick_file(directory: Path, *, label: str, names: Sequence[str]) -> Path:
    for name in names:
        path = directory / name
        if path.is_file() and not path.is_symlink():
            return path
    raise DescriptorError(f"{label} missing in geometry directory {directory}; expected one of {list(names)}")


def load_geometry_domain(
    directory: Path,
    *,
    domain: str,
    b1_cells: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise DescriptorError(f"geometry directory for {domain} is unavailable: {directory}")
    forecast_path = pick_file(directory, label=f"{domain} forecast", names=("forecast.json",))
    private_path = pick_file(directory, label=f"{domain} private geometry", names=("private_boundary.json", "private-boundary.json", "private.json"))
    public_path = pick_file(directory, label=f"{domain} public geometry", names=("public_aggregates.json", "public-aggregates.json", "public.json"))
    receipt_path = pick_file(directory, label=f"{domain} geometry receipt", names=("receipt.json", "geometry-receipt.json", "pipeline-receipt.json"))
    forecast = read_json(forecast_path, label=f"{domain} forecast")
    geometry_receipt = read_json(receipt_path, label=f"{domain} geometry receipt")
    public_payload = read_json(public_path, label=f"{domain} public geometry")
    private_payload = read_json(private_path, label=f"{domain} private geometry")
    if forecast.get("schema") != FORECAST_SCHEMA or forecast.get("task_id") != TASK_ID or forecast.get("status") != "FORECAST_FROZEN_BEFORE_LATER_FEATURE_LOAD":
        raise DescriptorError(f"{domain} forecast schema/status changed")
    if geometry_receipt.get("schema") != GEOMETRY_SCHEMA or geometry_receipt.get("task_id") != TASK_ID or geometry_receipt.get("status") != "FORECAST_AND_BOUNDARY_COMPLETE":
        raise DescriptorError(f"{domain} geometry receipt schema/status changed")
    assert_false_flags(forecast, label=f"{domain} forecast", flags=("truth_opened", "source_text_loaded", "target_weights_loaded"))
    assert_false_flags(public_payload, label=f"{domain} public geometry", flags=("truth_opened", "source_text_loaded", "target_weights_loaded"))
    assert_false_flags(private_payload, label=f"{domain} private geometry", flags=("truth_opened", "source_text_loaded", "target_weights_loaded"))
    if not isinstance(forecast.get("rows"), list) or len(forecast["rows"]) != B1_RECORDS:
        raise DescriptorError(f"{domain} forecast row count changed")
    if forecast.get("later_stages_not_loaded") != [128, 256]:
        raise DescriptorError(f"{domain} forecast later-stage boundary changed")
    stage_inputs = require_mapping(geometry_receipt.get("stage_inputs"), label=f"{domain} geometry stage inputs")
    if {int(str(key)) for key in stage_inputs} != set(B1_STAGES):
        raise DescriptorError(f"{domain} geometry stage set changed")
    for stage in B1_STAGES:
        raw = require_mapping(stage_inputs.get(str(stage), stage_inputs.get(stage)), label=f"{domain} geometry stage {stage}")
        for key, cell_key in (("predictions", "prediction"), ("projected_hidden", "projected"), ("geometry", "geometry")):
            declared = require_mapping(raw.get(key), label=f"{domain} geometry stage {stage}/{key}")
            if str(declared.get("sha256", "")).lower() != str(b1_cells[stage][cell_key]["sha256"]).lower():
                raise DescriptorError(f"{domain} geometry stage {stage}/{key} differs from B1 output")
        if int(raw.get("record_count", -1)) != B1_RECORDS:
            raise DescriptorError(f"{domain} geometry stage {stage} record count changed")
    early_inputs = require_mapping(forecast.get("early_stage_inputs"), label=f"{domain} forecast early inputs")
    for stage in (0, 64):
        raw = require_mapping(early_inputs.get(str(stage), early_inputs.get(stage)), label=f"{domain} forecast stage {stage}")
        geom_stage = require_mapping(stage_inputs.get(str(stage), stage_inputs.get(stage)), label=f"{domain} geometry stage {stage}")
        for key in ("predictions", "projected_hidden", "geometry"):
            if str(require_mapping(raw.get(key), label=f"{domain} forecast stage {stage}/{key}").get("sha256", "")).lower() != str(require_mapping(geom_stage.get(key), label=f"{domain} geometry stage {stage}/{key}").get("sha256", "")).lower():
                raise DescriptorError(f"{domain} forecast stage {stage}/{key} differs from geometry receipt")
    outputs = require_mapping(geometry_receipt.get("outputs"), label=f"{domain} geometry outputs")
    file_paths = {
        "forecast": forecast_path,
        "private_boundary": private_path,
        "public_aggregates": public_path,
    }
    for name, path in file_paths.items():
        declared = outputs.get(name)
        if not isinstance(declared, Mapping):
            raise DescriptorError(f"{domain} geometry output {name} binding is missing")
        output_binding = declared_binding(declared, label=f"{domain} geometry output {name}", base=receipt_path.parent)
        if output_binding["sha256"] != sha256_file(path):
            raise DescriptorError(f"{domain} geometry output {name} hash differs")
    code = normalize_code_files(require_mapping(geometry_receipt.get("code"), label=f"{domain} geometry code"), label=f"{domain} geometry code", base=receipt_path.parent)
    geometry_binding = {
        "directory": str(directory.resolve()),
        "forecast": file_binding(forecast_path, label=f"{domain} forecast"),
        "private_boundary": file_binding(private_path, label=f"{domain} private boundary"),
        "public_aggregates": file_binding(public_path, label=f"{domain} public aggregates"),
        "receipt": file_binding(receipt_path, label=f"{domain} geometry receipt"),
        "embedding": declared_binding(geometry_receipt.get("embedding"), label=f"{domain} geometry embedding", base=receipt_path.parent),
        "code": code,
        "stage_inputs": {
            str(stage): {
                key: declared_binding(require_mapping(require_mapping(stage_inputs.get(str(stage), stage_inputs.get(stage)), label=f"{domain} geometry stage {stage}").get(key), label=f"{domain} geometry stage {stage}/{key}"), label=f"{domain} geometry stage {stage}/{key}", base=receipt_path.parent)
                for key in ("predictions", "projected_hidden", "geometry")
            }
            for stage in B1_STAGES
        },
    }
    return geometry_binding


def load_a1_receipt(
    path: Path,
    *,
    panel_orders: Mapping[str, Mapping[str, Any]],
    capture_cells: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = read_json(path, label="A1 aggregate receipt")
    receipt_binding = file_binding(path, label="A1 aggregate receipt")
    if receipt.get("schema") != A1_RUN_SCHEMA or receipt.get("task_id") != TASK_ID:
        raise DescriptorError("A1 aggregate receipt schema or task identity changed")
    if receipt.get("status") != "A1_A2_RUN_COMPLETE_NO_TRUTH":
        raise DescriptorError("A1 aggregate receipt is not a complete production run")
    if receipt.get("qualification_only") is not False:
        raise DescriptorError("A1 aggregate receipt is qualification-only")
    for flag in ("truth_opened", "source_text_loaded", "token_ids_loaded", "target_labels_loaded", "target_weights_loaded", "p03_holdout_accessed"):
        if receipt.get(flag) is not False:
            raise DescriptorError(f"A1 aggregate receipt violates {flag}=false")
    scope = require_mapping(receipt.get("scope"), label="A1 scope")
    if scope.get("domains") != list(DOMAINS) or scope.get("stages") != list(A1_STAGES) or int(scope.get("records_per_domain", -1)) != A1_RECORDS:
        raise DescriptorError("A1 aggregate scope differs")
    input_manifest = require_mapping(receipt.get("input_manifest"), label="A1 input manifest summary")
    manifest_orders_raw = input_manifest.get("source_order")
    manifest_orders = manifest_orders_raw if isinstance(manifest_orders_raw, Mapping) else {}
    input_order_digests = require_mapping(input_manifest.get("source_order_sha256"), label="A1 input manifest order digests")
    input_cells = require_mapping(receipt.get("input_cells"), label="A1 input cells")
    cells = require_mapping(receipt.get("cells"), label="A1 output cells")
    expected_cell_ids = {f"{domain}_stage{stage:04d}" for domain in DOMAINS for stage in A1_STAGES}
    expected_input_ids = {f"{domain}_{stage}" for domain in DOMAINS for stage in A1_STAGES}
    if set(str(key) for key in cells) != expected_cell_ids or set(str(key) for key in input_cells) != expected_input_ids:
        raise DescriptorError("A1 aggregate cells are incomplete")
    code_summary = require_mapping(receipt.get("binding"), label="A1 runtime binding summary")
    code_files = {"a1_runner": declared_binding(receipt.get("runner_code"), label="A1 runner code", base=path.parent)}
    raw_code = require_mapping(code_summary.get("code_bindings"), label="A1 P11 code bindings")
    for name, value in raw_code.items():
        code_files[f"p11_{name}"] = declared_binding(value, label=f"A1 P11 code/{name}", base=path.parent)
    a1_code = {"files": code_files}
    result_cells: dict[tuple[str, int], dict[str, Any]] = {}
    for domain in DOMAINS:
        expected_h128 = list(panel_orders[domain]["h128"][:A1_RECORDS])
        expected_h128_digest = panel_orders[domain]["first32_h128_order_sha256"]
        expected_id_digest = runner_record_ids_digest([str(item) for item in panel_orders[domain]["record_ids"][:A1_RECORDS]])
        if manifest_orders:
            manifest_domain = require_mapping(manifest_orders.get(domain), label=f"A1 manifest order/{domain}")
            if manifest_domain.get("first32_h128_sequence_sha256") != expected_h128_digest or manifest_domain.get("first32_record_count") != A1_RECORDS:
                raise DescriptorError(f"A1 manifest H128 order differs for {domain}")
        if str(input_order_digests.get(domain, "")).lower() != expected_id_digest:
            raise DescriptorError(f"A1 runner order digest differs for {domain}")
        for stage in A1_STAGES:
            cell_id = f"{domain}_stage{stage:04d}"
            input_id = f"{domain}_{stage}"
            raw = require_mapping(cells.get(cell_id), label=f"A1 cell {cell_id}")
            input_raw = require_mapping(input_cells.get(input_id), label=f"A1 input cell {input_id}")
            if int(raw.get("stage", -1)) != stage or raw.get("domain") != domain or int(raw.get("records", -1)) != A1_RECORDS:
                raise DescriptorError(f"A1 cell {cell_id} identity or count changed")
            if str(raw.get("source_order_sha256", "")).lower() != expected_id_digest:
                raise DescriptorError(f"A1 cell {cell_id} runner source digest differs")
            input_observation = declared_binding(input_raw.get("observation"), label=f"A1 input observation {cell_id}", base=path.parent)
            capture_observation = capture_cells[(domain, stage)]["observation"]
            if input_observation["sha256"] != capture_observation["sha256"] or resolve_path(input_observation["path"]) != resolve_path(capture_observation["path"]):
                raise DescriptorError(f"A1 input observation {cell_id} differs from capture endpoint")
            cell_receipt = declared_binding(raw.get("receipt"), label=f"A1 cell receipt {cell_id}", base=path.parent)
            cell_prediction = declared_binding(raw.get("prediction"), label=f"A1 prediction {cell_id}", base=path.parent)
            cell_trace = declared_binding(raw.get("trace"), label=f"A1 candidate trace {cell_id}", base=path.parent)
            cell_cost = declared_binding(raw.get("cost"), label=f"A1 cost {cell_id}", base=path.parent)
            cell_receipt_payload = read_json(Path(cell_receipt["path"]), label=f"A1 cell receipt payload {cell_id}")
            if cell_receipt_payload.get("schema") != "token-reconstruction.trr-p12-a1-cell-receipt.v1" or cell_receipt_payload.get("status") != "A1_A2_CELL_COMPLETE_NO_TRUTH":
                raise DescriptorError(f"A1 cell receipt {cell_id} schema/status changed")
            if cell_receipt_payload.get("method_id") != "frozen_a1_a2_k256" or cell_receipt_payload.get("domain") != domain or int(cell_receipt_payload.get("stage", -1)) != stage or int(cell_receipt_payload.get("records", -1)) != A1_RECORDS:
                raise DescriptorError(f"A1 cell receipt {cell_id} identity changed")
            assert_false_flags(cell_receipt_payload, label=f"A1 cell receipt {cell_id}", flags=("truth_opened", "source_text_loaded", "token_ids_loaded", "target_labels_loaded", "target_weights_loaded", "p03_holdout_accessed"))
            nested_input = declared_binding(cell_receipt_payload.get("input_observation"), label=f"A1 cell receipt {cell_id} input observation", base=Path(cell_receipt["path"]).parent)
            if nested_input["sha256"] != input_observation["sha256"]:
                raise DescriptorError(f"A1 cell receipt {cell_id} input observation differs")
            for nested_name, aggregate_binding in (("prediction", cell_prediction), ("trace", cell_trace), ("cost", cell_cost)):
                nested_binding = declared_binding(cell_receipt_payload.get(nested_name), label=f"A1 cell receipt {cell_id}/{nested_name}", base=Path(cell_receipt["path"]).parent)
                if nested_binding["sha256"] != aggregate_binding["sha256"]:
                    raise DescriptorError(f"A1 cell receipt {cell_id} {nested_name} differs")
            result_cells[(domain, stage)] = {
                "source_order": {"hashes": expected_h128, "sha256": expected_h128_digest, "record_count": A1_RECORDS},
                "observation": {
                    **input_observation,
                    "key": "activations",
                    "role": "A1 sanitized capture endpoint",
                    "record_slice": [0, A1_RECORDS],
                },
                "cell_receipt_metadata": {**cell_receipt, "role": "A1 cell receipt metadata"},
                "prediction": {**cell_prediction, "key": "predictions", "role": "A1 prediction"},
                "candidate_receipt": {**cell_receipt, "role": "A1 candidate receipt"},
                "trace": {**cell_trace, "role": "A1 candidate trace"},
                "cost": cell_cost,
                "costs": {"receipt": cell_cost, "timing": raw.get("timing"), "peak_memory": raw.get("peak_memory")},
                "code": a1_code,
                "runner_source_order_sha256": expected_id_digest,
            }
    return {"receipt": receipt, "receipt_binding": receipt_binding, "cells": result_cells, "code": a1_code}


def decoder_registration(b1_cell: Mapping[str, Any], *, receipt_path: Path) -> dict[str, Any]:
    package = require_mapping(b1_cell["receipt"].get("package"), label="B1 package")
    package_files = {
        "package_descriptor": {**b1_cell["package"]["descriptor"], "role": "decoder package manifest"},
        "b1_state": {**b1_cell["package"]["b1_state"], "role": "decoder B1 state"},
        "public_readout": {**b1_cell["package"]["public_readout"], "role": "decoder public readout"},
    }
    decoder_files: dict[str, dict[str, Any]] = {}
    raw_modules = require_mapping(b1_cell["receipt"].get("code"), label="B1 decoder code").get("loaded_decoder_modules")
    if not isinstance(raw_modules, list) or not raw_modules:
        raise DescriptorError("B1 decoder module bindings are missing")
    for index, raw in enumerate(raw_modules):
        module = require_mapping(raw, label=f"B1 decoder module {index}")
        loaded_path = module.get("loaded_path")
        if not isinstance(loaded_path, str):
            raise DescriptorError(f"B1 decoder module {index} path is missing")
        binding = file_binding(loaded_path, label=f"decoder module {index}")
        if binding["sha256"] != require_digest(module.get("sha256"), label=f"decoder module {index} hash"):
            raise DescriptorError(f"B1 decoder module {index} hash changed")
        binding["role"] = "decoder module"
        decoder_files[f"decoder_module_{index:02d}"] = binding
    frozen = {"decoder": decoder_files, "package": package_files}
    if len(decoder_files) + len(package_files) < 3:
        raise DescriptorError("decoder/package registration has fewer than three files")
    return frozen


def write_create_only(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise DescriptorError(f"descriptor output is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file_binding(path, label="evaluation freeze descriptor")


def build(args: argparse.Namespace) -> dict[str, Any]:
    panel_path = resolve_path(args.panel)
    panel, panel_binding, rows, panel_data = panel_bindings(panel_path)
    source_binding = panel_data["source_inputs"]
    orders = {domain: panel_data[domain] for domain in DOMAINS}
    capture_root = resolve_path(args.capture_root)
    b1_root = resolve_path(args.b1_root)
    capture_cells: dict[tuple[str, int], dict[str, Any]] = {}
    b1_cells: dict[tuple[str, int], dict[str, Any]] = {}
    for domain in DOMAINS:
        for stage in B1_STAGES:
            capture_cell = load_capture_cell(capture_root, domain=domain, stage=stage, panel_binding=panel_binding, source_binding=source_binding, order=orders[domain])
            b1_cell = load_b1_cell(b1_root, capture_cell, domain=domain, stage=stage, panel_binding=panel_binding)
            capture_cells[(domain, stage)] = capture_cell
            b1_cells[(domain, stage)] = b1_cell
    package_fingerprint: tuple[tuple[str, str], ...] | None = None
    decoder_fingerprint: tuple[tuple[str, str], ...] | None = None
    for (domain, stage), cell in sorted(b1_cells.items()):
        package_now = tuple(sorted((name, binding["sha256"]) for name, binding in cell["package"].items()))
        modules = require_mapping(cell["receipt"].get("code"), label=f"B1 decoder code {domain}/{stage}").get("loaded_decoder_modules")
        if not isinstance(modules, list):
            raise DescriptorError(f"B1 decoder module bindings are missing for {domain}/{stage}")
        decoder_now = tuple((str(require_mapping(item, label=f"B1 decoder module {domain}/{stage}").get("loaded_path")), require_digest(require_mapping(item, label=f"B1 decoder module {domain}/{stage}").get("sha256"), label=f"B1 decoder module {domain}/{stage} hash")) for item in modules)
        if package_fingerprint is None:
            package_fingerprint = package_now
            decoder_fingerprint = decoder_now
        elif package_now != package_fingerprint or decoder_now != decoder_fingerprint:
            raise DescriptorError(f"B1 decoder/package binding differs at {domain}/{stage}")
    geometry_dirs = parse_geometry_dirs(args.geometry_dir)
    geometry: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        geometry[domain] = load_geometry_domain(geometry_dirs[domain], domain=domain, b1_cells={stage: b1_cells[(domain, stage)] for stage in B1_STAGES})
    a1 = load_a1_receipt(resolve_path(args.a1_receipt), panel_orders=orders, capture_cells={key: capture_cells[key] for key in capture_cells if key[1] in A1_STAGES})
    first_b1 = b1_cells[("pile", 0)]
    frozen_decoder = decoder_registration(first_b1, receipt_path=resolve_path(args.b1_root))
    b1_code = {"files": first_b1["code"], "commit": first_b1["receipt"].get("code_commit")}
    methods: dict[str, Any] = {
        "B1": {"record_scope": "all128_per_domain", "domains": {}},
        "A1+A2": {"record_scope": "first32_per_domain", "domains": {}},
    }
    for domain in DOMAINS:
        source_order = {"hashes": orders[domain]["h128"], "sha256": orders[domain]["source_order_sha256"], "record_count": B1_RECORDS}
        b1_domain: dict[str, Any] = {"source_order": source_order, "forecast": geometry[domain]["forecast"], "stages": {}}
        for stage in B1_STAGES:
            cell = b1_cells[(domain, stage)]
            b1_domain["stages"][str(stage)] = {
                "stage": stage,
                "record_count": B1_RECORDS,
                "source_order_sha256": source_order["sha256"],
                "artifacts": {
                    "observation": {**capture_cells[(domain, stage)]["observation"], "key": "activations", "role": "public activation observation"},
                    "prediction": cell["prediction"],
                    "projected": cell["projected"],
                    "geometry": cell["geometry"],
                    "capture": capture_cells[(domain, stage)]["capture_binding"],
                    "order": capture_cells[(domain, stage)]["order_binding"],
                    "receipt": cell["receipt_binding"],
                },
                "costs": cell["costs"],
                "code": b1_code,
            }
        methods["B1"]["domains"][domain] = b1_domain
        a1_source_order = {"hashes": orders[domain]["h128"][:A1_RECORDS], "sha256": orders[domain]["first32_h128_order_sha256"], "record_count": A1_RECORDS}
        a1_domain: dict[str, Any] = {"source_order": a1_source_order, "stages": {}}
        for stage in A1_STAGES:
            cell = a1["cells"][(domain, stage)]
            a1_domain["stages"][str(stage)] = {
                "stage": stage,
                "record_count": A1_RECORDS,
                "source_order_sha256": a1_source_order["sha256"],
                "runner_source_order_sha256": cell["runner_source_order_sha256"],
                "artifacts": {
                    "observation": cell["observation"],
                    "prediction": cell["prediction"],
                    "candidate_receipt": cell["candidate_receipt"],
                    "input_observation": cell["observation"],
                    "trace": cell["trace"],
                    "cost": cell["cost"],
                },
                "costs": cell["costs"],
                "code": cell["code"],
            }
        methods["A1+A2"]["domains"][domain] = a1_domain
    descriptor: dict[str, Any] = {
        "schema": FREEZE_SCHEMA,
        "task_id": TASK_ID,
        "status": "FREEZE_COMPLETE_BEFORE_TRUTH",
        "created_utc": utc_now(),
        "panel": panel_binding,
        "source_inputs": source_binding,
        "source_orders": {
            domain: {
                "hashes": orders[domain]["h128"],
                "sha256": orders[domain]["source_order_sha256"],
                "record_count": B1_RECORDS,
                "first32_sha256": orders[domain]["first32_h128_order_sha256"],
            }
            for domain in DOMAINS
        },
        "methods": methods,
        "registered_geometry": {
            "stage_pairs": ["0_to_64", "0_to_128", "0_to_256"],
            "full_vocabulary_required": True,
            "retrospective_only": True,
            "no_nearest_euclidean_claim_from_runner_up": True,
            "artifacts_by_domain": {
                domain: {
                    "private_boundary": geometry[domain]["private_boundary"],
                    "public_aggregates": geometry[domain]["public_aggregates"],
                    "pipeline_receipt": geometry[domain]["receipt"],
                }
                for domain in DOMAINS
            },
        },
        "geometry": geometry,
        "frozen_decoder_binding": frozen_decoder,
        "a1_run_receipt": a1["receipt_binding"],
        "a1_runner_code": a1["code"],
        "source_order_conventions": {
            "B1": "ordered panel H128 sequence digests",
            "A1+A2": "ordered first32 panel H128 sequence digests; runner record-ID digests retained per cell as runner_source_order_sha256",
        },
        "truth_opened": False,
        "source_text_loaded": False,
        "target_weights_loaded": False,
        "target_labels_loaded": False,
        "token_ids_loaded": False,
        "p03_holdout_accessed": False,
        "execution": {
            "command": list(sys.argv),
            "code_commit": git_commit(),
            "code": file_binding(Path(__file__), label="descriptor assembler"),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "network_used": False,
            "tensor_payloads_loaded": False,
            "source_text_loaded": False,
            "truth_opened": False,
        },
    }
    if args.dry_run:
        return {
            "status": "READY_FREEZE_DESCRIPTOR_NO_TENSOR_OR_TRUTH_LOAD",
            "descriptor": descriptor,
            "output": str(resolve_path(args.output)),
        }
    output_binding = write_create_only(resolve_path(args.output), descriptor)
    return {"status": "FREEZE_DESCRIPTOR_WRITTEN_BEFORE_TRUTH", "descriptor": output_binding}


def parse_geometry_dirs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise DescriptorError("--geometry-dir values must be DOMAIN=PATH")
        domain, raw = value.split("=", 1)
        if domain not in DOMAINS or domain in result or not raw:
            raise DescriptorError("--geometry-dir must contain one unique pile=PATH and finance=PATH")
        result[domain] = resolve_path(raw)
    if set(result) != set(DOMAINS):
        raise DescriptorError("--geometry-dir must specify both pile and finance")
    return result


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "outputs/TRR-P12/sources-r2/panel.json")
    parser.add_argument("--capture-root", type=Path, default=ROOT / "outputs/TRR-P12/capture-r1")
    parser.add_argument("--b1-root", type=Path, default=ROOT / "outputs/TRR-P12/b1-r1")
    parser.add_argument("--geometry-dir", action="append", required=True, metavar="DOMAIN=PATH")
    parser.add_argument("--a1-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/TRR-P12/evaluation-freeze-descriptor-r1.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.execute == args.dry_run:
        raise SystemExit("pass exactly one of --execute or --dry-run")
    try:
        print(json.dumps(build(args), indent=2, sort_keys=True))
    except DescriptorError as exc:
        print(f"P12 descriptor refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
