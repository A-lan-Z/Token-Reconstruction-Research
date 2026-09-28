#!/usr/bin/env python3
"""Capture one frozen TRR-P12 public observation cell.

The worker rerenders the frozen public rows transiently, verifies all published
identity hashes, runs the validated cut-4 public prefix in fixed B8 x 192
batches, and writes only BF16 activations, masks, positions, and opaque order
metadata.  It is create-only and has no truth-opening path.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import torch
from safetensors.torch import save_file

from scripts import trr0005_produce_confirmation as trusted
from scripts.trr_p11 import source_selector as p11
from scripts.trr_p12 import qualify_capture as qualified
from token_reconstruction.public_activation import (
    capture_public_prefix,
    pad_public_token_sequences,
    record_ids_sha256,
    validate_activation_tensor,
)


TASK_ID = "TRR-P12"
PANEL_SCHEMA = "token-reconstruction.trr-p12-source-panel.v1"
PANEL_STATUS = "FROZEN_P12_SOURCE_PANEL_NO_TRUTH"
PANEL_SHA256 = "f48a126fbae6a5e88e1292caf0675d8973f67ab415b706b7fd35e966659ad08e"
PANEL_BYTES = 627329
TARGET_BUNDLE_SHA256 = "5f717f0b1ba9195d98566b681c378de30a8397d5cefa2b337b521d99c0437b6d"
TRAJECTORY_SCHEMA = "token-reconstruction.trr-p12-target-trajectory.v1.run"
TRAJECTORY_STATUS = "FULL_TRAJECTORY_COMPLETE"
CAPTURE_SCHEMA = "token-reconstruction.trr-p12-public-capture.v1"
OBSERVATION_SCHEMA = "token-reconstruction.trr-p12-public-observation.v1"
ORDER_SCHEMA = "token-reconstruction.trr-p12-public-order.v1"
STAGES = (0, 64, 128, 256)
DOMAINS = ("pile", "finance")
RECORDS = 128
BATCH_RECORDS = 8
FULL_TOKENS = 192
STORED_TOKENS = 128
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
BOS = 128000
PAD = 128001
CUT_DEPTH = 4
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
BASE_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
SOURCE_INPUTS_SHA256 = "1fa6aea4485ad602d396e6a57dc53257977a0f4581375892ef22ab176d52b409"
DEFAULT_SOURCE_INPUTS = ROOT / "experiments/TRR-P11/selector/public_source_inputs_r1.json"
DEFAULT_TARGET_BUNDLE = ROOT / "outputs/TRR-P12/target-sources-r1/bundle.json"
DEFAULT_OUTPUT = ROOT / "outputs/TRR-P12/capture-cell-r1"
DEFAULT_MODEL = Path(
    "/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct"
) / "snapshots" / BASE_REVISION
DEFAULT_TRAJECTORY = ROOT / "outputs/TRR-P12/target-trajectory-r1/run.json"
PANEL_RANGES = {"pile": (0, 10000), "finance": (50000, 52000)}
HEX = frozenset("0123456789abcdef")
ROW_FIELDS = frozenset(
    {
        "record_id",
        "public_record_sha256",
        "dataset_key",
        "dataset_id",
        "split",
        "revision",
        "row_index",
        "source_index",
        "full_token_count",
        "post_bos_token_count",
        "valid_tokens",
        "final_sequence_sha256",
        "h40_sequence_sha256",
        "h128_sequence_sha256",
        "h129_sequence_sha256",
        "trr0002_active_token_ids_sha256",
        "trr0002_h40_token_ids_sha256",
    }
)


class CaptureError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_binding(
    path: str | Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    base: Path = ROOT,
    allow_symlink: bool = False,
) -> dict[str, Any]:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = base / raw
    resolved = raw.resolve()
    if raw.is_symlink() and not allow_symlink:
        raise CaptureError(f"{label} must not be a symlink: {raw}")
    if not resolved.is_file():
        raise CaptureError(f"{label} is unavailable: {resolved}")
    actual_sha = sha256_file(resolved)
    actual_bytes = int(resolved.stat().st_size)
    if expected_sha256 is not None and actual_sha != expected_sha256.lower():
        raise CaptureError(f"{label} SHA-256 changed")
    if expected_bytes is not None and actual_bytes != int(expected_bytes):
        raise CaptureError(f"{label} byte count changed")
    return {"path": str(resolved), "bytes": actual_bytes, "sha256": actual_sha, "readonly": True}


def resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CaptureError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"{label} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise CaptureError(f"{label} must be an object")
    return dict(value)


def write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise CaptureError(f"create-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file_binding(path, label="new JSON output")


def hex64(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c.lower() not in HEX for c in value):
        raise CaptureError(f"{label} is not a SHA-256 hex string")
    return value.lower()


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


def code_bindings() -> dict[str, dict[str, Any]]:
    paths = {
        "capture": Path(__file__).resolve(),
        "trusted_renderer": Path(trusted.__file__).resolve(),
        "source_selector": Path(p11.__file__).resolve(),
        "qualified_loader": Path(qualified.__file__).resolve(),
        "public_activation": SRC / "token_reconstruction/public_activation.py",
        "public_prefix": SRC / "token_reconstruction/public_prefix.py",
        "target_update": SRC / "token_reconstruction/target_update.py",
    }
    return {name: file_binding(path, label=f"code/{name}") for name, path in paths.items()}


def panel_rows(panel: Mapping[str, Any], domain: str) -> list[dict[str, Any]]:
    if panel.get("schema") != PANEL_SCHEMA or panel.get("task_id") != TASK_ID:
        raise CaptureError("P12 panel schema or task identity changed")
    if panel.get("status") != PANEL_STATUS:
        raise CaptureError("P12 panel is not frozen before truth")
    if panel.get("records_per_domain") != RECORDS:
        raise CaptureError("P12 panel record count changed")
    if tuple(panel.get("paired_stages", ())) != STAGES:
        raise CaptureError("P12 panel stage pairing changed")
    if panel.get("same_records_all_snapshots") is not True:
        raise CaptureError("P12 panel does not declare paired records")
    if panel.get("release_sha256") != "302cef73349927615e39b116e985f796a9c70b6f9c97932a119c0acb1d3f0a8d":
        raise CaptureError("P12 panel study-manifest binding changed")
    if panel.get("amendment_sha256") != "e960951e52922bd65dfeb48d14780f68c3e3c3a79e041f7234af378a3b2fad42":
        raise CaptureError("P12 panel amendment binding changed")
    if panel.get("union", {}).get("sha256") != "bd2e641f5f10d249595b89f48aeeba2d97a2b71688190177f4ebe4f4ac022a0b":
        raise CaptureError("P12 panel exclusion-union binding changed")
    if panel.get("source_ranges_half_open") != {
        "pile": [0, 10000],
        "finance": [50000, 52000],
    }:
        raise CaptureError("P12 panel amended ranges changed")
    for flag in ("truth_opened", "target_labels_loaded", "p03_holdout_accessed"):
        if panel.get(flag) is not False:
            raise CaptureError(f"P12 panel boundary flag {flag} is open")
    selection = panel.get("selection_rule")
    if not isinstance(selection, Mapping) or not isinstance(selection.get("records"), Mapping):
        raise CaptureError("P12 panel records are missing")
    raw_rows = selection["records"].get(domain)
    if not isinstance(raw_rows, list) or len(raw_rows) != RECORDS:
        raise CaptureError(f"P12 panel {domain} count changed")
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_h128: set[str] = set()
    for number, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise CaptureError(f"P12 panel {domain} row {number} is malformed")
        row = dict(raw)
        if set(row) - ROW_FIELDS:
            raise CaptureError(f"P12 panel {domain} row has unapproved payload")
        if row.get("dataset_key") != domain:
            raise CaptureError(f"P12 panel {domain} dataset key changed")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
            raise CaptureError(f"P12 panel {domain} record IDs are not unique")
        h128 = hex64(row.get("h128_sequence_sha256"), label=f"{domain}/{number}/h128")
        if h128 in seen_h128 or row.get("final_sequence_sha256") != h128:
            raise CaptureError(f"P12 panel {domain} H128 order changed")
        if row.get("source_index") != row.get("row_index") or row.get("valid_tokens") != STORED_TOKENS:
            raise CaptureError(f"P12 panel {domain} source geometry changed")
        for key in (
            "public_record_sha256",
            "h40_sequence_sha256",
            "h129_sequence_sha256",
            "trr0002_active_token_ids_sha256",
            "trr0002_h40_token_ids_sha256",
        ):
            if row.get(key) is not None:
                hex64(row[key], label=f"{domain}/{number}/{key}")
        seen_ids.add(record_id)
        seen_h128.add(h128)
        result.append(row)
    return result


def load_panel(path: Path, domain: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    binding = file_binding(path, label="P12 source panel", expected_sha256=PANEL_SHA256, expected_bytes=PANEL_BYTES)
    panel = read_json(path, label="P12 source panel")
    rows = panel_rows(panel, domain)
    source = panel.get("source_inputs")
    if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
        raise CaptureError("P12 source-input descriptor binding is missing")
    source_binding = file_binding(
        source["path"],
        label="P12 source-input descriptor",
        expected_sha256=SOURCE_INPUTS_SHA256,
        expected_bytes=source.get("bytes"),
        base=ROOT,
    )
    return panel, {"panel": binding, "source_inputs": source_binding}, rows


def validate_stage_receipt(
    path: Path,
    *,
    stage: int,
    adapter: Path | None,
    target_bundle_sha256: str | None,
) -> dict[str, Any]:
    binding = file_binding(path, label="P12 target trajectory receipt")
    receipt = read_json(path, label="P12 target trajectory receipt")
    if receipt.get("schema") != TRAJECTORY_SCHEMA or receipt.get("task_id") != TASK_ID:
        raise CaptureError("target trajectory receipt schema changed")
    if receipt.get("status") != TRAJECTORY_STATUS:
        raise CaptureError("target trajectory is not complete")
    model = receipt.get("model")
    if not isinstance(model, Mapping) or model.get("model_id") != BASE_MODEL_ID or model.get("revision") != BASE_REVISION:
        raise CaptureError("target trajectory model binding changed")
    trajectory = receipt.get("trajectory")
    expected = {
        "seed": 6012,
        "layers": [0, 1, 2, 3],
        "modules": ["q_proj", "v_proj"],
        "rank": 8,
        "alpha": 16.0,
        "batch_size": 2,
        "sequence_length": STORED_TOKENS,
        "train_rows": 256,
        "validation_rows": 64,
        "stages": [64, 128, 256],
    }
    if not isinstance(trajectory, Mapping) or any(trajectory.get(k) != v for k, v in expected.items()):
        raise CaptureError("target trajectory geometry/configuration changed")
    source_bundle = receipt.get("source_bundle")
    if target_bundle_sha256 is not None and (
        not isinstance(source_bundle, Mapping) or source_bundle.get("manifest_sha256") != target_bundle_sha256
    ):
        raise CaptureError("target trajectory is bound to a different target source bundle")
    if stage == 0:
        if adapter is not None:
            raise CaptureError("baseline stage cannot receive an adapter")
        stage_binding: dict[str, Any] = {"present": False, "target_weights": "baseline"}
    else:
        stage_binding = {}
        records = receipt.get("stage_records")
        if not isinstance(records, list):
            raise CaptureError("target trajectory stage records are missing")
        matching = [item for item in records if isinstance(item, Mapping) and item.get("stage_updates") == stage]
        if len(matching) != 1 or not isinstance(matching[0].get("adapter"), Mapping):
            raise CaptureError(f"target trajectory stage {stage} adapter is missing")
        declared = matching[0]["adapter"]
        actual = file_binding(
            declared.get("path", ""),
            label=f"target trajectory stage {stage} adapter",
            expected_sha256=declared.get("sha256"),
            expected_bytes=declared.get("bytes"),
            base=path.parent,
        )
        if adapter is None or resolve_path(adapter) != Path(actual["path"]):
            raise CaptureError(f"stage {stage} adapter argument differs from the trajectory receipt")
        stage_binding = {
            "present": True,
            "binding": actual,
            "config": {
                "layers": [0, 1, 2, 3],
                "modules": ["q_proj", "v_proj"],
                "rank": 8,
                "alpha": 16.0,
                "seed": 6012,
            },
            "target_weights": "evaluator-only; never serialized in public observations",
        }
    return {
        "receipt": binding,
        "stage_updates": stage,
        "model_snapshot": str(model.get("snapshot", "")),
        "adapter": stage_binding,
    }


def rerender_batches(
    *,
    rows: Sequence[Mapping[str, Any]],
    domain: str,
    source_inputs: Path,
) -> tuple[Any, dict[str, Any]]:
    try:
        normalized = p11._normalize_source_inputs(source_inputs, root=ROOT, require_tokenizer_dir=True)
        tokenizer = trusted._load_tokenizer(Path(normalized["tokenizer"]["path"]))
        dataset = trusted._load_arrow_dataset(
            tuple(Path(item["path"]) for item in normalized[domain]["arrow_files"])
        )
    except Exception as exc:
        raise CaptureError("trusted public tokenizer or Arrow source could not be loaded") from exc
    start, stop = PANEL_RANGES[domain]
    if len(dataset) < stop:
        raise CaptureError(f"{domain} dataset is shorter than the amended P12 range")
    sequences: list[list[int]] = []
    ids: list[str] = []
    h128s: list[str] = []
    for declared in rows:
        index = declared.get("row_index")
        if isinstance(index, bool) or not isinstance(index, int) or not start <= index < stop:
            raise CaptureError(f"{domain} panel row escaped its amended range")
        try:
            candidate = trusted._render_row(domain, dataset[index], index, tokenizer)
            actual = p11._candidate_identity(candidate)
        except Exception as exc:
            raise CaptureError(f"trusted P11 renderer failed for {domain}/{index}") from exc
        for key in ROW_FIELDS:
            if str(actual.get(key)) != str(declared.get(key)):
                raise CaptureError(f"canonical rerender changed {domain}/{index}/{key}")
        token_ids = [int(value) for value in candidate.token_ids]
        if len(token_ids) < STORED_TOKENS or token_ids[0] != BOS:
            raise CaptureError(f"{domain}/{index} lacks a valid H128 prefix")
        sequences.append(token_ids[:STORED_TOKENS])
        ids.append(str(actual["record_id"]))
        h128s.append(str(actual["h128_sequence_sha256"]))
    try:
        batch = pad_public_token_sequences(
            sequences,
            maximum_tokens=FULL_TOKENS,
            pad_token_id=PAD,
            bos_token_id=BOS,
            vocab_size=VOCAB_SIZE,
        )
    except Exception as exc:
        raise CaptureError("P12 B8x192 padded batch construction failed") from exc
    if tuple(batch.token_ids.shape) != (RECORDS, FULL_TOKENS):
        raise CaptureError("P12 padded batch geometry changed")
    if not batch.attention_mask[:, :STORED_TOKENS].eq(1).all().item():
        raise CaptureError("P12 first 128 positions are not all active")
    order = {
        "domain": domain,
        "record_count": RECORDS,
        "ordered_record_ids": ids,
        "ordered_h128_sequence_sha256": h128s,
        "record_ids_sha256": record_ids_sha256(ids),
        "h128_order_sha256": hashlib.sha256(
            json.dumps(h128s, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_text_or_token_ids_written": False,
    }
    return batch, order


def save_observation(
    path: Path,
    *,
    domain: str,
    stage: int,
    activations: torch.Tensor,
    batch: Any,
    order: Mapping[str, Any],
    panel_sha256: str,
    trajectory_sha256: str,
) -> dict[str, Any]:
    expected = (RECORDS, FULL_TOKENS, HIDDEN_SIZE)
    if tuple(activations.shape) != expected or activations.dtype != torch.bfloat16:
        raise CaptureError("full public activation geometry/dtype changed")
    validate_activation_tensor(activations, batch, hidden_size=HIDDEN_SIZE, require_bfloat16=True)
    compact = activations[:, :STORED_TOKENS].detach().cpu().contiguous()
    mask = batch.attention_mask[:, :STORED_TOKENS].to(torch.uint8).contiguous()
    positions = batch.position_ids[:, :STORED_TOKENS].to(torch.int64).contiguous()
    if not mask.eq(1).all().item():
        raise CaptureError("stored public mask changed")
    expected_positions = torch.arange(STORED_TOKENS, dtype=torch.int64).repeat(RECORDS, 1)
    if not torch.equal(positions, expected_positions):
        raise CaptureError("stored public positions changed")
    if path.exists() or path.is_symlink():
        raise CaptureError(f"public observation is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"activations": compact, "attention_mask": mask, "position_ids": positions},
        str(path),
        metadata={
            "schema": OBSERVATION_SCHEMA,
            "task_id": TASK_ID,
            "domain": domain,
            "stage_updates": str(stage),
            "cut_depth": str(CUT_DEPTH),
            "public_full_forward": "true",
            "capture_batch_records": str(BATCH_RECORDS),
            "capture_sequence_tokens": str(FULL_TOKENS),
            "stored_sequence_tokens": str(STORED_TOKENS),
            "hidden_size": str(HIDDEN_SIZE),
            "activation_dtype": "bfloat16",
            "record_ids_sha256": str(order["record_ids_sha256"]),
            "h128_order_sha256": str(order["h128_order_sha256"]),
            "panel_sha256": panel_sha256,
            "trajectory_sha256": trajectory_sha256,
            "source_text_written": "false",
            "token_ids_written": "false",
            "target_weights_written": "false",
            "truth_opened": "false",
        },
    )
    return {
        **file_binding(path, label="public observation"),
        "keys": ["activations", "attention_mask", "position_ids"],
        "shape": [RECORDS, STORED_TOKENS, HIDDEN_SIZE],
        "activation_dtype": "bfloat16",
        "domain": domain,
        "stage_updates": stage,
        "record_ids_sha256": order["record_ids_sha256"],
        "h128_order_sha256": order["h128_order_sha256"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.stage not in STAGES:
        raise CaptureError(f"stage must be one of {STAGES}")
    panel_path = resolve_path(args.panel)
    panel, bindings, rows = load_panel(panel_path, args.domain)
    requested_source_inputs = resolve_path(args.source_inputs)
    if requested_source_inputs != Path(bindings["source_inputs"]["path"]):
        raise CaptureError("source-inputs argument differs from the panel-bound descriptor")
    target_bundle = file_binding(
        args.target_bundle,
        label="P12 target source bundle",
        expected_sha256=TARGET_BUNDLE_SHA256,
    )
    target_payload = read_json(resolve_path(args.target_bundle), label="P12 target source bundle")
    if target_payload.get("schema") != "token-reconstruction.trr-p12-target-source-bundle.v1":
        raise CaptureError("P12 target source bundle schema changed")
    if target_payload.get("task_id") != TASK_ID:
        raise CaptureError("P12 target source bundle task identity changed")
    if target_payload.get("panel", {}).get("sha256") != bindings["panel"]["sha256"]:
        raise CaptureError("P12 target source bundle panel binding changed")
    target_bundle_sha = target_bundle["sha256"]
    stage_receipt = validate_stage_receipt(
        resolve_path(args.stage_receipt),
        stage=args.stage,
        adapter=None if args.adapter is None else resolve_path(args.adapter),
        target_bundle_sha256=target_bundle_sha,
    )
    if not args.execute:
        return {
            "status": "READY_NO_SOURCE_READ_NO_MODEL_LOAD",
            "panel": bindings["panel"],
            "source_inputs": bindings["source_inputs"],
            "stage_receipt": stage_receipt,
            "target_source_bundle": target_bundle,
            "domain": args.domain,
            "stage_updates": args.stage,
            "output_root": str(resolve_path(args.output_root)),
            "capture_contract": {
                "forward_path": "ContiguousPublicPrefix.forward_full",
                "batch_records": BATCH_RECORDS,
                "padded_tokens": FULL_TOKENS,
                "stored_tokens": STORED_TOKENS,
                "cut_depth": CUT_DEPTH,
                "hidden_size": HIDDEN_SIZE,
                "activation_dtype": "bfloat16",
                "truth_opened": False,
            },
        }
    snapshot = resolve_path(args.model_snapshot)
    receipt_snapshot = resolve_path(
        stage_receipt["model_snapshot"],
        base=resolve_path(args.stage_receipt).parent,
    )
    if snapshot != receipt_snapshot:
        raise CaptureError("model snapshot argument differs from the trajectory receipt")
    model_binding = qualified.bind_base_snapshot(snapshot)
    if model_binding["revision"] != BASE_REVISION:
        raise CaptureError("base model revision changed")
    source_inputs = Path(bindings["source_inputs"]["path"])
    batch, order = rerender_batches(rows=rows, domain=args.domain, source_inputs=source_inputs)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CaptureError("CUDA requested but unavailable")
    output_root = resolve_path(args.output_root)
    allowed = (ROOT / "outputs" / "TRR-P12").resolve()
    try:
        output_root.relative_to(allowed)
    except ValueError as exc:
        raise CaptureError("output root must be below outputs/TRR-P12") from exc
    if output_root.exists() or output_root.is_symlink():
        raise CaptureError(f"output root is create-only: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    adapter = None if args.adapter is None else resolve_path(args.adapter)
    prefix = None
    try:
        prefix = qualified.load_prefix(snapshot, device=device, adapter=adapter)
        def guard() -> None:
            qualified.guard_resources(
                device,
                phase="capture_batch",
                started=started,
                timeout_seconds=float(args.timeout_seconds),
                min_free_gpu_bytes=qualified.MIN_FREE_GPU_BYTES,
                max_reserved_gpu_bytes=qualified.MAX_RESERVED_GPU_BYTES,
                max_rss_bytes=qualified.MAX_HOST_RSS_BYTES,
            )
        guard()
        with torch.inference_mode():
            activations = capture_public_prefix(
                prefix,
                batch,
                device=device,
                batch_size=BATCH_RECORDS,
                hidden_size=HIDDEN_SIZE,
                resource_check=guard,
            )
        observation = save_observation(
            output_root / "observation.safetensors",
            domain=args.domain,
            stage=args.stage,
            activations=activations,
            batch=batch,
            order=order,
            panel_sha256=bindings["panel"]["sha256"],
            trajectory_sha256=stage_receipt["receipt"]["sha256"],
        )
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError("P12 public-prefix capture failed") from exc
    finally:
        if prefix is not None:
            del prefix
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    order_payload = {
        "schema": ORDER_SCHEMA,
        "task_id": TASK_ID,
        "status": "FROZEN_PUBLIC_ORDER_NO_PAYLOAD",
        "panel_sha256": bindings["panel"]["sha256"],
        "domain": args.domain,
        "stage_updates": args.stage,
        **order,
        "truth_opened": False,
    }
    order_binding = write_json(output_root / "order.json", order_payload)
    receipt = {
        "schema": CAPTURE_SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_OBSERVATION_CELL_COMPLETE_NO_TRUTH",
        "created_utc": utc_now(),
        "panel": bindings["panel"],
        "source_inputs": bindings["source_inputs"],
        "target_source_bundle": target_bundle,
        "stage_receipt": stage_receipt["receipt"],
        "model": model_binding,
        "adapter": stage_receipt["adapter"],
        "observation": observation,
        "order": order_binding,
        "contract": {
            "forward_path": "ContiguousPublicPrefix.forward_full",
            "cut_depth": CUT_DEPTH,
            "capture_batch_records": BATCH_RECORDS,
            "capture_sequence_tokens": FULL_TOKENS,
            "stored_sequence_tokens": STORED_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "activation_dtype": "bfloat16",
            "same_frozen_panel_order": True,
            "source_text_or_token_ids_written": False,
            "truth_opened": False,
        },
        "execution": {
            "command": list(sys.argv),
            "code_commit": git_commit(),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "device": str(device),
            "gpu_used": device.type == "cuda",
            "network_used": False,
            "source_text_materialized_transiently": True,
            "token_ids_materialized_transiently": True,
            "target_weights_loaded_evaluator_only": args.stage != 0,
            "target_weights_written": False,
            "source_text_written": False,
            "token_ids_written": False,
            "truth_opened": False,
            "elapsed_seconds": time.monotonic() - started,
            "resource_final": qualified.resource_snapshot(device),
        },
        "code": code_bindings(),
    }
    capture_binding = write_json(output_root / "capture.json", receipt)
    return {"status": receipt["status"], "capture": capture_binding, "observation": observation, "order": order_binding}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=ROOT / "outputs/TRR-P12/sources-r2/panel.json")
    p.add_argument("--target-bundle", type=Path, default=DEFAULT_TARGET_BUNDLE)
    p.add_argument("--stage-receipt", type=Path, required=True)
    p.add_argument("--stage", type=int, required=True, choices=STAGES)
    p.add_argument("--model-snapshot", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--domain", choices=DOMAINS, required=True)
    p.add_argument("--source-inputs", type=Path, default=DEFAULT_SOURCE_INPUTS)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--timeout-seconds", type=float, default=3600.0)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.execute and args.dry_run:
        raise SystemExit("use one of --execute or --dry-run")
    if not args.execute and not args.dry_run:
        raise SystemExit("pass --dry-run or --execute")
    try:
        print(json.dumps(run(args), sort_keys=True))
    except CaptureError as exc:
        print(f"P12 capture refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
