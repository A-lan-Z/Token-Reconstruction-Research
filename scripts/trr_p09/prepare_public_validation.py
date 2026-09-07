#!/usr/bin/env python3
"""Prepare the already-opened TRR-0009 public validation labels.

This adapter is deliberately narrower than public observation capture.  It
uses the frozen TRR-0009 selection and the trusted renderer/batcher to create
the public token labels, masks, and positions for the first 128 positions
including BOS.  The public_base H payloads are bound by their existing
observation-manifest records and row order; their tensor contents are never
opened here.

The output directory is create-only and contains one safetensors payload per
domain, a metadata-only record-row file, a manifest, and an execution receipt.
No source text is serialized and no model, forward, LoRA, evaluation truth,
or private/final evaluation artifact is loaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import torch
from safetensors.torch import save as save_safetensors

from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0006_capture_public as trr6_capture
from scripts import trr0009_eval_capture as trr9_capture
from scripts.trr_p09.fixed_control_caller import join_public_validation_labels


SCHEMA = "token-reconstruction.trr-p09-public-validation-preparation.v1"
ROWS_SCHEMA = "token-reconstruction.trr-p09-public-validation-rows.v1"
PAYLOAD_SCHEMA = "token-reconstruction.trr-p09-public-validation-tensors.v1"
TASK_ID = "TRR-P09"
PARENT_TASK_ID = "TRR-0009"
SEQUENCE_TOKENS = 128
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
DOMAINS = ("Finance", "Pile")
STYLE_BY_DOMAIN = {"Finance": "finance", "Pile": "pile"}
EXPECTED_SELECTION_SHA256 = "c2e996514f7f45e55d7bfadc27fc8048bdb07a2c979b208de8716972d8d1def2"
EXPECTED_OBSERVATIONS_SHA256 = "8ad89e2598e7b1e774c43b3fd1d0f1a990615069cd1f488e8aa139463087d430"
EXPECTED_RECORDS = {"Finance": 256, "Pile": 128}
PUBLIC_BASE = "public_base"


class PreparationError(RuntimeError):
    """Raised when the frozen public validation contract cannot be met."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_record(path: Path, *, label: str, hash_payload: bool = True) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"{label} is not a regular file: {path}")
    record: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "bytes": int(path.stat().st_size),
    }
    if hash_payload:
        record["sha256"] = _sha256_file(path)
    return record


def _declared_file_record(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path_raw = value.get("path")
    if not isinstance(path_raw, str) or not path_raw:
        raise PreparationError(f"{label} lacks a payload path")
    try:
        declared_bytes = int(value["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreparationError(f"{label} lacks a valid byte count") from exc
    declared_sha = value.get("sha256")
    if not isinstance(declared_sha, str) or len(declared_sha) != 64 or any(c not in "0123456789abcdef" for c in declared_sha):
        raise PreparationError(f"{label} lacks a valid declared SHA-256")
    path = Path(path_raw).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"{label} payload is unavailable: {path}")
    actual_bytes = int(path.stat().st_size)
    if actual_bytes != declared_bytes:
        raise PreparationError(f"{label} byte count changed: declared={declared_bytes}, actual={actual_bytes}")
    # This function is specifically used for the existing H payloads.  The
    # manifest's hash is retained as the immutable binding; the payload is
    # intentionally not read or rehashed.
    return {
        "path": str(path),
        "bytes": declared_bytes,
        "sha256": declared_sha,
        "payload_read": False,
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"{label} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise PreparationError(f"{label} must be a JSON object")
    return dict(value)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise PreparationError(f"create-only JSON destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return _file_record(path, label=path.name)


def _write_safetensors_exclusive(
    path: Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise PreparationError(f"create-only safetensors destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = save_safetensors(
            {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()},
            metadata={str(key): str(value) for key, value in metadata.items()},
        )
    except Exception as exc:
        raise PreparationError(f"could not serialize validation tensors: {path}") from exc
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return _file_record(path, label=path.name)


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _resource_snapshot() -> dict[str, Any]:
    rss_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_bytes = rss_raw * 1024 if sys.platform != "darwin" else rss_raw
    available: int | None = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == "MemAvailable:" and fields[1].isdigit():
                available = int(fields[1]) * 1024
                break
    except (OSError, UnicodeError, ValueError):
        available = None
    return {"host_rss_bytes": rss_bytes, "host_available_bytes": available}


def _resolve_path(value: Path | str, *, root: Path) -> Path:
    raw = Path(value).expanduser()
    return (raw if raw.is_absolute() else root / raw).resolve()


def _observation_cells(observations: Mapping[str, Any], *, root: Path) -> dict[str, dict[str, Any]]:
    if observations.get("schema") != "token-reconstruction.trr0009-public-observation-manifest.v1":
        raise PreparationError("observation manifest schema changed")
    if observations.get("task_id") != PARENT_TASK_ID or observations.get("status") != "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH":
        raise PreparationError("observation manifest is not the frozen public TRR-0009 manifest")
    for key in ("public_material_only", "source_text_loaded", "source_text_written", "token_ids_written", "target_labels_loaded", "truth_opened"):
        if observations.get(key) is not (True if key == "public_material_only" else False):
            raise PreparationError(f"observation access flag changed: {key}")
    if int(observations.get("sequence_tokens_including_bos", -1)) != SEQUENCE_TOKENS:
        raise PreparationError("public validation sequence width changed")
    if int(observations.get("hidden_size", -1)) != HIDDEN_SIZE:
        raise PreparationError("public validation hidden size changed")
    declared_counts = observations.get("records_by_domain")
    if not isinstance(declared_counts, Mapping):
        raise PreparationError("observation manifest lacks domain counts")
    for domain, count in EXPECTED_RECORDS.items():
        if int(declared_counts.get(domain.lower(), -1)) != count:
            raise PreparationError(f"public validation record count changed for {domain}")
    cells = observations.get("cells")
    if not isinstance(cells, list):
        raise PreparationError("observation manifest lacks cells")
    result: dict[str, dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise PreparationError("observation cell is malformed")
        if cell.get("condition") != PUBLIC_BASE:
            continue
        style = str(cell.get("style", ""))
        domain = {"finance": "Finance", "pile": "Pile"}.get(style)
        if domain is None:
            raise PreparationError(f"unknown public_base observation style: {style!r}")
        if domain in result:
            raise PreparationError(f"duplicate public_base observation cell: {domain}")
        if int(cell.get("records", -1)) != EXPECTED_RECORDS[domain]:
            raise PreparationError(f"public_base observation record count changed for {domain}")
        obs = cell.get("observation")
        if not isinstance(obs, Mapping):
            raise PreparationError(f"public_base observation descriptor is missing for {domain}")
        if obs.get("shape") != [EXPECTED_RECORDS[domain], SEQUENCE_TOKENS, HIDDEN_SIZE]:
            raise PreparationError(f"public_base H geometry changed for {domain}")
        for key in ("activations_key", "attention_mask_key", "position_ids_key"):
            if not isinstance(obs.get(key), str) or not obs[key]:
                raise PreparationError(f"public_base observation key missing for {domain}: {key}")
        h_record = _declared_file_record(obs, label=f"public_base H {domain}")
        if obs.get("stored_sequence_tokens") != SEQUENCE_TOKENS:
            raise PreparationError(f"public_base stored sequence width changed for {domain}")
        declared_ids_sha = cell.get("record_ids_sha256")
        if not isinstance(declared_ids_sha, str) or len(declared_ids_sha) != 64:
            raise PreparationError(f"public_base record-ID digest missing for {domain}")
        result[domain] = {
            "style": style,
            "cell_id": str(cell.get("cell_id", f"{style}__{PUBLIC_BASE}")),
            "records": int(cell["records"]),
            "record_ids_sha256": declared_ids_sha,
            "h": h_record,
            "activations_key": str(obs["activations_key"]),
            "attention_mask_key": str(obs["attention_mask_key"]),
            "position_ids_key": str(obs["position_ids_key"]),
            "shape": list(obs["shape"]),
            "capture_batch_records": int(obs.get("capture_batch_records", 8)),
            "capture_sequence_tokens": int(obs.get("capture_sequence_tokens", 192)),
            "producer_only_lora": bool(obs.get("producer_only_lora", False)),
        }
    if set(result) != set(DOMAINS):
        raise PreparationError(f"public_base observations are incomplete: {sorted(result)}")
    if any(value["producer_only_lora"] for value in result.values()):
        raise PreparationError("public_base validation unexpectedly binds a LoRA producer")
    # ``root`` is used to make the path check explicit at the call site.  H is
    # intentionally only stat-checked in _declared_file_record above.
    del root
    return result


def _validate_public_batches(
    batches: Mapping[str, Any],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, torch.Tensor]]:
    prepared: dict[str, dict[str, torch.Tensor]] = {}
    for style, rows in records.items():
        if style not in {"finance", "pile"}:
            raise PreparationError(f"unexpected materialized style: {style}")
        batch = batches.get(style)
        if batch is None:
            raise PreparationError(f"trusted batch is missing for {style}")
        count = len(rows)
        tokens = batch.token_ids[:, :SEQUENCE_TOKENS].contiguous()
        mask = batch.attention_mask[:, :SEQUENCE_TOKENS].contiguous()
        positions = batch.position_ids[:, :SEQUENCE_TOKENS].contiguous()
        if tuple(tokens.shape) != (count, SEQUENCE_TOKENS):
            raise PreparationError(f"{style} public labels have the wrong geometry")
        if tuple(mask.shape) != tuple(tokens.shape) or tuple(positions.shape) != tuple(tokens.shape):
            raise PreparationError(f"{style} public mask/position geometry differs")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise PreparationError(f"{style} public token IDs have an unexpected dtype")
        if mask.dtype not in (torch.bool, torch.uint8) or not mask.to(torch.bool).all().item():
            raise PreparationError(f"{style} public H128 mask is not fully active")
        expected_positions = torch.arange(SEQUENCE_TOKENS, dtype=torch.long).expand(count, -1)
        if not torch.equal(positions.to(torch.long), expected_positions):
            raise PreparationError(f"{style} public H128 positions are not 0..127")
        if int(tokens[:, 0].unique().item()) != int(trusted.BOS_TOKEN_ID):
            raise PreparationError(f"{style} public labels do not retain the declared BOS token")
        if tokens.lt(0).any().item() or tokens.ge(VOCAB_SIZE).any().item():
            raise PreparationError(f"{style} public labels contain an out-of-vocabulary ID")
        prepared[style] = {
            "token_ids": tokens.to(dtype=torch.int32).contiguous(),
            "attention_mask": mask.to(dtype=torch.uint8).contiguous(),
            "position_ids": positions.to(dtype=torch.int64).contiguous(),
        }
    return prepared


def _build_rows_and_joins(
    selected_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    cells: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[int]], list[dict[str, Any]], str, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    rows_by_domain: dict[str, list[int]] = {domain: [] for domain in DOMAINS}
    h_joins: list[dict[str, Any]] = []
    source_id_digests = trr9_capture._digest_record_ids(selected_rows)
    offset = 0
    for domain in DOMAINS:
        style = STYLE_BY_DOMAIN[domain]
        declared_rows = selected_rows[style]
        cell = cells[domain]
        if len(declared_rows) != int(cell["records"]):
            raise PreparationError(f"selection/observation row count differs for {domain}")
        if source_id_digests[style] != cell["record_ids_sha256"]:
            raise PreparationError(f"selection/observation record-ID digest differs for {domain}")
        for local_row, declared in enumerate(declared_rows):
            global_row = offset + local_row
            record_id = str(declared["record_id"])
            rows.append({"global_row": global_row, "record_id": record_id, "domain": domain})
            rows_by_domain[domain].append(global_row)
            h_joins.append(
                {
                    "global_row": global_row,
                    "record_id": record_id,
                    "domain": domain,
                    "observation_row": local_row,
                    "observation_cell_id": cell["cell_id"],
                    "h_path": cell["h"]["path"],
                    "h_sha256": cell["h"]["sha256"],
                    "h_bytes": cell["h"]["bytes"],
                    "activations_key": cell["activations_key"],
                    "attention_mask_key": cell["attention_mask_key"],
                    "position_ids_key": cell["position_ids_key"],
                    "row_order": "frozen_selection_order",
                    "h_payload_read": False,
                }
            )
        offset += len(declared_rows)
    label_join = join_public_validation_labels(rows, required_domains=DOMAINS)
    label_join_metadata = label_join.metadata()
    return rows, rows_by_domain, h_joins, label_join_metadata["semantic_sha256"], source_id_digests


def _tensor_digest(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)}))
    digest.update(contiguous.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=_ROOT)
    parser.add_argument("--selection", type=Path, default=Path("experiments/TRR-0009/selection_v2/source_selection.json"))
    parser.add_argument("--observations", type=Path, default=Path("experiments/TRR-0009/evaluation/public_observations_v2/observations.json"))
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/trr-p09-runtime/public-validation-r1"))
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise PreparationError(f"repository root is unavailable: {root}")
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise PreparationError(f"output root is create-only and already exists: {output_root}")
    output_root.mkdir(parents=True)
    started_utc = _utc_now()
    started_clock = time.perf_counter()
    before_resource = _resource_snapshot()

    selection_path = _resolve_path(args.selection, root=root)
    observations_path = _resolve_path(args.observations, root=root)
    selection, selection_record, selected_rows, counts = trr9_capture.load_selection(
        selection_path,
        repository_root=root,
        expected_counts={style: EXPECTED_RECORDS[{"finance": "Finance", "pile": "Pile"}[style]] for style in ("finance", "pile")},
    )
    if selection_record["sha256"] != EXPECTED_SELECTION_SHA256:
        raise PreparationError("selection SHA-256 differs from the frozen public validation binding")
    observations_record = _file_record(observations_path, label="public observation manifest")
    if observations_record["sha256"] != EXPECTED_OBSERVATIONS_SHA256:
        raise PreparationError("observation-manifest SHA-256 differs from the frozen public validation binding")
    observations = _load_json(observations_path, label="public observation manifest")
    cells = _observation_cells(observations, root=root)

    pile_paths = tuple(trr9_capture._source_paths(selection, style="pile", root=root))
    finance_paths = tuple(trr9_capture._source_paths(selection, style="finance", root=root))
    tokenizer_path = trr9_capture._tokenizer_path(selection, root=root)
    trr6_capture._validate_source_descriptors(
        selection,
        pile_paths=pile_paths,
        finance_paths=finance_paths,
        tokenizer_path=tokenizer_path,
    )
    tokenizer = trusted._load_tokenizer(tokenizer_path)
    datasets = {
        "pile": trusted._load_arrow_dataset(pile_paths),
        "finance": trusted._load_arrow_dataset(finance_paths),
    }
    materialized = trr6_capture._materialize_selected(selected_rows, datasets=datasets, tokenizer=tokenizer)
    batches = trr6_capture._batches(materialized)
    prepared = _validate_public_batches(batches, materialized)
    rows, rows_by_domain, h_joins, label_join_sha256, source_id_digests = _build_rows_and_joins(selected_rows, cells)

    source_bindings: dict[str, Any] = {}
    output_payloads: dict[str, Any] = {}
    for domain in DOMAINS:
        style = STYLE_BY_DOMAIN[domain]
        dataset_descriptor = trusted._dataset_descriptor(
            pile_paths if style == "pile" else finance_paths,
            style=style,
        )
        tokenizer_descriptor = trusted._tokenizer_descriptor(tokenizer_path)
        tensor_set = prepared[style]
        metadata = {
            "schema": PAYLOAD_SCHEMA,
            "task_id": TASK_ID,
            "domain": domain,
            "style": style,
            "records": str(len(selected_rows[style])),
            "sequence_tokens": str(SEQUENCE_TOKENS),
            "selection_sha256": selection_record["sha256"],
            "observation_manifest_sha256": observations_record["sha256"],
            "record_ids_sha256": source_id_digests[style],
            "h_reference_sha256": cells[domain]["h"]["sha256"],
            "h_reference_path": cells[domain]["h"]["path"],
            "h_payload_read": "false",
            "source_text_written": "false",
            "truth_opened": "false",
            "model_loaded": "false",
            "public_forward_count": "0",
        }
        payload_path = output_root / f"{style}_validation_h128.safetensors"
        payload_record = _write_safetensors_exclusive(payload_path, tensor_set, metadata=metadata)
        output_payloads[domain] = {
            "file": payload_record,
            "tensor_keys": list(tensor_set),
            "shape": list(tensor_set["token_ids"].shape),
            "dtype": {name: str(value.dtype) for name, value in tensor_set.items()},
            "tensor_sha256": {name: _tensor_digest(value) for name, value in tensor_set.items()},
        }
        selected_binding_rows = []
        for global_row, declared in zip(rows_by_domain[domain], selected_rows[style], strict=True):
            selected_binding_rows.append(
                {
                    "global_row": int(global_row),
                    "record_id": str(declared["record_id"]),
                    "source_row_index": int(declared["row_index"]),
                    "public_record_sha256": str(declared["public_record_sha256"]),
                    "final_sequence_sha256": str(declared["final_sequence_sha256"]),
                }
            )
        source_bindings[domain] = {
            "dataset_descriptor": dataset_descriptor,
            "tokenizer_descriptor": tokenizer_descriptor,
            "selected_rows": selected_binding_rows,
            "record_ids_sha256": source_id_digests[style],
            "observation_record_ids_sha256": cells[domain]["record_ids_sha256"],
            "observation_h": dict(cells[domain]["h"]),
        }

    rows_payload = {
        "schema": ROWS_SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_VALIDATION_LABELS_PREPARED_NO_TRUTH",
        "required_domains": list(DOMAINS),
        "record_count": len(rows),
        "rows": rows,
        "rows_by_domain": rows_by_domain,
        "label_join_sha256": label_join_sha256,
        "selection_metric": "domain-balanced public validation token accuracy",
        "domain_balanced_selection": {
            "enabled": True,
            "aggregation": "equal arithmetic mean of Finance and Pile token accuracy",
            "domains": list(DOMAINS),
            "row_counts": {domain: len(rows_by_domain[domain]) for domain in DOMAINS},
        },
        "source_selection": dict(selection_record),
        "observation_manifest": dict(observations_record),
        "source_text_written": False,
        "truth_opened": False,
    }
    rows_record = _write_json_exclusive(output_root / "validation_rows.json", rows_payload)

    manifest_payload = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_VALIDATION_PREPARED_NO_TRUTH",
        "parent_task": PARENT_TASK_ID,
        "domains": list(DOMAINS),
        "records_by_domain": {domain: len(rows_by_domain[domain]) for domain in DOMAINS},
        "record_count": len(rows),
        "sequence_tokens_including_bos": SEQUENCE_TOKENS,
        "hidden_size": HIDDEN_SIZE,
        "global_row_order": "Finance_then_Pile",
        "selection_metric": "domain-balanced public validation token accuracy",
        "label_join": {
            "schema": "token-reconstruction.trr-p09-public-validation-label-join.v1",
            "required_domains": list(DOMAINS),
            "rows_by_domain": rows_by_domain,
            "semantic_sha256": label_join_sha256,
            "record_count": len(rows),
        },
        "rows": rows_record,
        "payloads": output_payloads,
        "observation_h_join": h_joins,
        "source_bindings": source_bindings,
        "input_bindings": {
            "source_selection": dict(selection_record),
            "observation_manifest": dict(observations_record),
            "public_base_only": True,
            "h_payloads_opened": False,
            "h_payloads_rehashed": False,
            "model_loaded": False,
            "public_forward_count": 0,
            "lora_loaded": False,
            "truth_opened": False,
            "source_text_written": False,
        },
        "trusted_materialization": {
            "load_selection": "scripts.trr0009_eval_capture.load_selection",
            "validate_source_descriptors": "scripts.trr0006_capture_public._validate_source_descriptors",
            "load_tokenizer": "scripts.trr0005_produce_confirmation._load_tokenizer",
            "load_arrow_dataset": "scripts.trr0005_produce_confirmation._load_arrow_dataset",
            "materialize_selected": "scripts.trr0006_capture_public._materialize_selected",
            "batches": "scripts.trr0006_capture_public._batches",
            "crop": "first 128 positions including BOS after trusted 192-token batching",
        },
        "source_text_persisted": False,
        "truth_opened": False,
    }
    manifest_record = _write_json_exclusive(output_root / "validation_manifest.json", manifest_payload)
    finished_utc = _utc_now()
    receipt_payload = {
        "schema": "token-reconstruction.trr-p09-public-validation-preparation-receipt.v1",
        "task_id": TASK_ID,
        "status": "PASS_PUBLIC_VALIDATION_PREPARED_NO_TRUTH",
        "command": list(sys.argv),
        "repository_root": str(root),
        "source_commit": _git_commit(root),
        "script": _file_record(Path(__file__), label="preparation helper"),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "elapsed_seconds": time.perf_counter() - started_clock,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu-only-materialization",
            "pid": os.getpid(),
        },
        "resource": {
            "before": before_resource,
            "after": _resource_snapshot(),
            "external_watchdog": {
                "required": True,
                "max_rss_gib": 8,
                "minimum_host_available_gib": 10,
                "timeout_seconds": 900,
            },
        },
        "inputs": {
            "source_selection": dict(selection_record),
            "observation_manifest": dict(observations_record),
            "public_base_h_payloads": {
                domain: {
                    "path": cells[domain]["h"]["path"],
                    "bytes": cells[domain]["h"]["bytes"],
                    "sha256": cells[domain]["h"]["sha256"],
                    "payload_read": False,
                }
                for domain in DOMAINS
            },
        },
        "outputs": {
            "validation_manifest": manifest_record,
            "validation_rows": rows_record,
            "payloads": output_payloads,
        },
        "label_join_sha256": label_join_sha256,
        "records_by_domain": {domain: len(rows_by_domain[domain]) for domain in DOMAINS},
        "h_join_rows": len(h_joins),
        "source_text_written": False,
        "truth_opened": False,
        "model_loaded": False,
        "public_forward_count": 0,
        "lora_loaded": False,
    }
    receipt_record = _write_json_exclusive(output_root / "preparation_receipt.json", receipt_payload)
    return {
        "status": receipt_payload["status"],
        "output_root": str(output_root),
        "manifest": manifest_record,
        "rows": rows_record,
        "receipt": receipt_record,
        "payloads": output_payloads,
        "label_join_sha256": label_join_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    result = run(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PreparationError, OSError, ValueError, KeyError) as exc:
        print(f"TRR-P09 public validation preparation refused: {exc}", file=sys.stderr)
        raise SystemExit(125) from exc
