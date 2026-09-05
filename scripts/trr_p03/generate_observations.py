#!/usr/bin/env python3
"""Generate grouped opaque boundary observations for one target bundle.

This command is the evaluator-side boundary between source prompts and the
reconstruction process.  It may read a setup panel and its private truth while
constructing activations, but it writes only opaque grouped observations to
``public/``.  The reconstruction CLI receives that public directory and never
receives this command's panel or truth paths.

The default Stage-1 output has four files per target bundle, one for each
declared length, with six records per file.  ``--stage stage2_holdout`` is
available for the post-gate holdout and is intentionally opt-in.
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
import sys
import time
from typing import Any, Mapping

# Keep CUDA hidden even when a host has an available device.  The import is
# below this assignment so the runtime records the effective setting.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
for _path in (_SOURCE_ROOT, _SOURCE_ROOT / "src", _SOURCE_ROOT / "scripts" / "trr_p01"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import load_model
from safetensors import safe_open
from token_reconstruction.trr_p03.io import (
    BOS_TOKEN_ID,
    CUT_DEPTH,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    OBSERVATION_INDEX_SCHEMA,
    P03IOError,
    VOCAB_SIZE,
    create_only_directory,
    create_only_file,
    file_record,
    read_json,
    read_jsonl,
    save_observation_bundle,
    sha256_file,
    write_json_exclusive,
)


TASK_ID = "TRR-P03"
DEFAULT_REQUIRED_BYTES = 10 * 1024**3
DEFAULT_EXPECTED_PEAK_BYTES = 8 * 1024**3
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEED = 20260906
PANEL_SCHEMAS = {
    "token-reconstruction.trr-p03-natural-panel.v1",
    "token-reconstruction.trr-p03-setup-panel.v1",
    "token-reconstruction.trr-p03-panel.v1",
}


class GenerationError(RuntimeError):
    """Raised when source preparation or observation generation is unsafe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _runtime(seed: int, *, device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "platform": platform.platform(),
        "kernel": platform.uname()._asdict(),
        "pid": os.getpid(),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": int(seed),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _memory() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemAvailable", "MemTotal"}:
                values[key] = int(rest.strip().split()[0]) * 1024
    except (OSError, UnicodeError, ValueError, IndexError):
        pass
    # Linux reports ru_maxrss in KiB.  Keep a platform fallback so the receipt
    # remains explicit if this script is inspected outside Linux.
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "linux":
        rss *= 1024
    return {
        "available_bytes": int(values.get("MemAvailable", 0)),
        "total_bytes": int(values.get("MemTotal", 0)),
        "process_max_rss_bytes": rss * 1024 if sys.platform == "linux" else rss,
    }


def _guard(required_bytes: int, expected_peak_bytes: int) -> dict[str, Any]:
    if required_bytes <= expected_peak_bytes:
        raise GenerationError("resource reservation must exceed expected peak")
    snapshot = _memory()
    if snapshot["available_bytes"] <= 0 or snapshot["total_bytes"] <= 0:
        raise GenerationError("CPU resource guard failed closed: memory status unavailable")
    if snapshot["available_bytes"] < required_bytes:
        raise GenerationError(
            "CPU resource guard failed closed: "
            f"available={snapshot['available_bytes']} required={required_bytes}"
        )
    return {
        "status": "PASS",
        "required_bytes": int(required_bytes),
        "expected_peak_bytes": int(expected_peak_bytes),
        "safety_margin_bytes": int(required_bytes - expected_peak_bytes),
        "available_bytes_before": snapshot["available_bytes"],
        "total_bytes": snapshot["total_bytes"],
        "process_max_rss_before": snapshot["process_max_rss_bytes"],
        "cuda_allocation": False,
    }


def _check_live_guard(guard: Mapping[str, Any]) -> dict[str, int]:
    snapshot = _memory()
    required = int(guard["required_bytes"])
    expected = int(guard["expected_peak_bytes"])
    if snapshot["available_bytes"] < required:
        raise GenerationError(
            "live CPU resource guard failed closed: "
            f"available={snapshot['available_bytes']} required={required}"
        )
    if snapshot["process_max_rss_bytes"] > expected:
        raise GenerationError(
            "live process RSS cap exceeded: "
            f"rss={snapshot['process_max_rss_bytes']} expected_peak={expected}"
        )
    return snapshot


def _append_progress(path: Path, event: str, **details: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "schema": "token-reconstruction.trr-p03-phase-progress.v1",
                    "event": event,
                    "timestamp_utc": _utc_now(),
                    **details,
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl_truth(path: Path) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for row in read_jsonl(path):
        record_id = row.get("record_id")
        token_ids = row.get("token_ids", row.get("input_ids"))
        if not isinstance(record_id, str) or not record_id or record_id in result:
            raise GenerationError("truth JSONL record IDs are missing or duplicated")
        if not isinstance(token_ids, list) or len(token_ids) < 2:
            raise GenerationError("truth JSONL token sequence is invalid")
        result[record_id] = [int(value) for value in token_ids]
    if not result:
        raise GenerationError("truth JSONL is empty")
    return result


def _read_safetensors_truth(path: Path, truth_index: Path | None) -> dict[str, list[int]]:
    values: dict[str, list[int]] = {}
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key).to(torch.long).reshape(-1)
                if tensor.numel() < 2:
                    raise GenerationError("truth sequence is too short")
                values[key] = [int(value) for value in tensor.tolist()]
    except GenerationError:
        raise
    except Exception as exc:
        raise GenerationError(f"truth safetensors is invalid: {path}") from exc
    if truth_index is None or not truth_index.is_file():
        return {
            key.removesuffix("__input_ids").replace("_", "-"): token_ids
            for key, token_ids in values.items()
        }
    index = read_json(truth_index)
    rows = index.get("records") if isinstance(index, Mapping) else None
    if not isinstance(rows, list):
        raise GenerationError("truth index records are missing")
    result: dict[str, list[int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise GenerationError("truth index row is invalid")
        record_id, key = row.get("record_id"), row.get("tensor_key")
        if not isinstance(record_id, str) or not isinstance(key, str) or key not in values:
            raise GenerationError("truth index does not match truth tensors")
        if record_id in result:
            raise GenerationError("truth index IDs are duplicated")
        result[record_id] = values[key]
    return result


def _load_truth(path: Path, truth_index: Path | None) -> dict[str, list[int]]:
    if path.is_symlink() or not path.is_file():
        raise GenerationError(f"truth artifact is missing: {path}")
    if path.suffix.lower() in {".jsonl", ".json"}:
        return _read_jsonl_truth(path)
    return _read_safetensors_truth(path, truth_index)


def _stage_aliases(stage: str) -> tuple[str, ...]:
    if stage == "stage1":
        return ("stage1", "s1")
    if stage == "stage2_holdout":
        return ("stage2_holdout", "stage2", "s2")
    raise GenerationError(f"unsupported stage: {stage}")


def _select_order(panel: Mapping[str, Any], rows_by_id: Mapping[str, Mapping[str, Any]], stage: str) -> list[str]:
    section = panel.get(stage)
    if isinstance(section, Mapping) and isinstance(section.get("record_order"), list):
        order = [str(value) for value in section["record_order"]]
    else:
        aliases = set(_stage_aliases(stage))
        order = [
            record_id
            for record_id, row in rows_by_id.items()
            if str(row.get("stage", "stage1")) in aliases
        ]
    if not order:
        raise GenerationError(f"setup panel has no records for stage {stage}")
    if len(set(order)) != len(order):
        raise GenerationError("setup panel stage order contains duplicate IDs")
    if any(record_id not in rows_by_id for record_id in order):
        raise GenerationError("setup panel stage order references an unknown record")
    return order


def _truth_path_from_panel(panel_path: Path, panel: Mapping[str, Any], stage: str) -> Path | None:
    private = panel.get("private_truth")
    if isinstance(private, Mapping):
        for key in (stage, "stage1" if stage == "stage1" else "stage2_holdout"):
            value = private.get(key) or private.get(f"{key}_path")
            if isinstance(value, str):
                candidate = (panel_path.parent / value).resolve()
                if candidate.is_file():
                    return candidate
    candidates = [
        panel_path.parent / stage / "private_truth.jsonl",
        panel_path.parent / ("stage1" if stage == "stage1" else "stage2_holdout") / "private_truth.jsonl",
        panel_path.parent / "private_truth.jsonl",
        panel_path.parent / "private_truth.safetensors",
    ]
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def _truth_index_from_panel(panel_path: Path, truth_path: Path | None, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.resolve()
    if truth_path is None:
        return None
    candidate = truth_path.with_name("truth_index.json")
    return candidate.resolve() if candidate.is_file() else None


def _read_panel(
    panel_path: Path,
    *,
    truth_path: Path | None,
    truth_index_path: Path | None,
    stage: str,
    open_truth: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path | None]:
    panel_value = read_json(panel_path)
    if not isinstance(panel_value, Mapping) or panel_value.get("schema") not in PANEL_SCHEMAS:
        raise GenerationError("unsupported TRR-P03 setup panel schema")
    if panel_value.get("truth_opened") is True:
        raise GenerationError("setup panel is already truth-opened")
    raw_rows = panel_value.get("records")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise GenerationError("setup panel has no records")
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str):
            raise GenerationError("setup panel record is malformed")
        record_id = str(row["record_id"])
        if record_id in rows_by_id:
            raise GenerationError("setup panel record IDs are duplicated")
        rows_by_id[record_id] = row
    order = _select_order(panel_value, rows_by_id, stage)
    source_truth = truth_path.resolve() if truth_path is not None else _truth_path_from_panel(panel_path, panel_value, stage)
    source_truth_index = _truth_index_from_panel(panel_path, source_truth, truth_index_path)
    # Metadata-only callers (including the parser smoke) must not open a
    # private sidecar merely because one is adjacent to the evaluator panel.
    truth_by_id = (
        _load_truth(source_truth, source_truth_index)
        if open_truth and source_truth is not None
        else {}
    )
    selected: list[dict[str, Any]] = []
    for panel_position, record_id in enumerate(order):
        row = rows_by_id[record_id]
        token_values = row.get("token_ids", row.get("input_ids"))
        if not isinstance(token_values, list):
            token_values = truth_by_id.get(record_id)
        if not isinstance(token_values, list) or len(token_values) < 2:
            raise GenerationError(f"no evaluator token sequence for {record_id}")
        token_ids = [int(value) for value in token_values]
        if token_ids[0] != BOS_TOKEN_ID or any(value < 0 or value >= VOCAB_SIZE for value in token_ids):
            raise GenerationError(f"invalid token sequence for {record_id}")
        declared = row.get("sequence_length")
        if declared is not None and int(declared) != len(token_ids):
            raise GenerationError(f"declared sequence length differs for {record_id}")
        selected.append(
            {
                "record_id": record_id,
                "token_ids": token_ids,
                "sequence_length": len(token_ids),
                "panel_position": panel_position,
                "stage": stage,
            }
        )
    return dict(panel_value), selected, source_truth


def _read_record_ids(path: Path) -> list[str]:
    value = read_json(path.resolve())
    if isinstance(value, list):
        values = value
    elif isinstance(value, Mapping) and isinstance(value.get("record_ids"), list):
        values = value["record_ids"]
    else:
        raise GenerationError("record selection must be a JSON list or record_ids object")
    result = [str(item) for item in values]
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise GenerationError("record selection IDs are missing or duplicated")
    return result


def _configure_runtime(seed: int) -> None:
    if seed < 0:
        raise GenerationError("seed must be non-negative")
    try:
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise GenerationError("Torch thread configuration must happen before any model work") from exc
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _padded_batch(rows: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(int(row["sequence_length"]) for row in rows)
    input_ids = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    positions = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        values = torch.tensor(row["token_ids"], dtype=torch.long, device=device)
        length = int(values.numel())
        input_ids[index, :length] = values
        mask[index, :length] = 1
        positions[index, :length] = torch.arange(length, dtype=torch.long, device=device)
    return input_ids, mask, positions


def _digest_tensor(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    descriptor = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(descriptor + b"\0" + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _asset_record(path: Path) -> dict[str, Any]:
    """Describe a file or checkpoint directory without rehashing large weights."""
    path = path.resolve()
    if path.is_file():
        return file_record(path)
    if not path.is_dir():
        raise GenerationError(f"model asset is missing: {path}")
    files = []
    for child in sorted(path.rglob("*")):
        if child.is_file():
            files.append(
                {
                    "path": child.relative_to(path).as_posix(),
                    "bytes": child.stat().st_size,
                }
            )
    if not files:
        raise GenerationError(f"model asset directory is empty: {path}")
    return {"path": str(path), "kind": "checkpoint_directory", "files": files}


def _bundle_descriptor(
    *,
    bundle_id: str,
    stage: str,
    length: int,
    record_ids: list[str],
    artifact_path: Path,
    relative_path: Path,
    digest: str,
    mask_digest: str,
    position_digest: str,
) -> dict[str, Any]:
    """Describe a written bundle using its actual path and public relative name."""

    artifact_path = artifact_path.resolve()
    if artifact_path.is_symlink() or not artifact_path.is_file():
        raise GenerationError(f"observation bundle was not written: {artifact_path}")
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise GenerationError(f"observation bundle relative path is invalid: {relative_path}")
    sequence = length + 1
    return {
        "bundle_id": bundle_id,
        "stage": stage,
        "scored_tokens": length,
        "sequence_length": sequence,
        "record_ids": record_ids,
        "relative_path": relative_path.as_posix(),
        "keys": {"activations": "activations", "attention_mask": "attention_mask", "position_ids": "position_ids"},
        "expected_shapes": {
            "activations": [len(record_ids), sequence, HIDDEN_SIZE],
            "attention_mask": [len(record_ids), sequence],
            "position_ids": [len(record_ids), sequence],
        },
        "bytes": artifact_path.stat().st_size,
        "sha256": digest,
        "mask_digest": mask_digest,
        "position_digest": position_digest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bundle-id", choices=("bundle-a", "bundle-b"), required=True)
    parser.add_argument("--stage", choices=("stage1", "stage2_holdout", "all"), default="stage1")
    parser.add_argument(
        "--record-ids",
        type=Path,
        default=None,
        help="optional evaluator-only JSON selection for a predeclared qualification subset",
    )
    parser.add_argument("--truth", type=Path, default=None)
    parser.add_argument("--truth-index", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--required-bytes", type=int, default=DEFAULT_REQUIRED_BYTES)
    parser.add_argument("--expected-peak-bytes", type=int, default=DEFAULT_EXPECTED_PEAK_BYTES)
    parser.add_argument("--implementation-commit", default="UNBOUND_PRECOMMIT")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise GenerationError("batch size must be positive")
    device = torch.device(args.device)
    if device.type != "cpu":
        raise GenerationError("P03 observation generation is CPU-only")
    _configure_runtime(args.seed)
    panel_path = args.panel.resolve()
    model_path = args.model_path.resolve()
    stage_names = ("stage1", "stage2_holdout") if args.stage == "all" else (args.stage,)
    root = create_only_directory(args.output_root.resolve())
    public_root = root / "public"
    observations_root = public_root / "observations" / args.bundle_id
    public_root.mkdir()
    observations_root.mkdir(parents=True)
    progress_path = root / "phase_progress.jsonl"
    create_only_file(progress_path)
    progress_path.touch()
    started = time.perf_counter()
    started_utc = _utc_now()
    guard = _guard(int(args.required_bytes), int(args.expected_peak_bytes))
    _append_progress(progress_path, "resource_guard", **guard)

    all_rows: list[dict[str, Any]] = []
    source_truth_paths: list[Path] = []
    panel: dict[str, Any] = {}
    for stage in stage_names:
        panel, stage_rows, source_truth = _read_panel(
            panel_path,
            truth_path=args.truth,
            truth_index_path=args.truth_index,
            stage=stage,
            open_truth=True,
        )
        all_rows.extend(stage_rows)
        if source_truth is not None and source_truth not in source_truth_paths:
            source_truth_paths.append(source_truth)
    if args.record_ids is not None:
        requested_ids = _read_record_ids(args.record_ids)
        by_id = {str(row["record_id"]): row for row in all_rows}
        missing = [record_id for record_id in requested_ids if record_id not in by_id]
        if missing:
            raise GenerationError(f"record selection is outside the requested stage: {missing}")
        requested = set(requested_ids)
        all_rows = [row for row in all_rows if str(row["record_id"]) in requested]
        if len(all_rows) != len(requested_ids):
            raise GenerationError("record selection changed the panel order")
    if not all_rows:
        raise GenerationError("no records selected for generation")
    panel_hash = sha256_file(panel_path)

    load_started = time.perf_counter()
    model = load_model(device=device, model_path=model_path)
    model.eval()
    if int(model.config.hidden_size) != HIDDEN_SIZE or int(model.config.vocab_size) != VOCAB_SIZE:
        raise GenerationError("target model geometry changed")
    load_seconds = time.perf_counter() - load_started
    _append_progress(progress_path, "model_loaded", elapsed_seconds=load_seconds, model_path=str(model_path))

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in all_rows:
        grouped.setdefault(int(row["sequence_length"]) - 1, []).append(row)
    descriptors: list[dict[str, Any]] = []
    generation_started = time.perf_counter()
    with torch.inference_mode():
        for length in sorted(grouped):
            rows = grouped[length]
            activations: list[torch.Tensor] = []
            masks: list[torch.Tensor] = []
            positions: list[torch.Tensor] = []
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                input_ids, mask, position_ids = _padded_batch(batch, device)
                output = model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = getattr(output, "hidden_states", None)
                if hidden_states is None or len(hidden_states) <= CUT_DEPTH:
                    raise GenerationError("target model did not return the declared cut state")
                hidden = hidden_states[CUT_DEPTH].detach().to(device="cpu", dtype=torch.bfloat16)
                for offset, row in enumerate(batch):
                    sequence = int(row["sequence_length"])
                    activations.append(hidden[offset : offset + 1, :sequence, :].contiguous())
                    masks.append(torch.ones((1, sequence), dtype=torch.int64))
                    positions.append(torch.arange(sequence, dtype=torch.int64).view(1, -1))
                del output, hidden_states, hidden, input_ids, mask, position_ids
                live = _check_live_guard(guard)
                _append_progress(progress_path, "batch_complete", stage=str(rows[0]["stage"]), length=length, batch_start=start, batch_size=len(batch), process_max_rss_bytes=live["process_max_rss_bytes"], available_bytes=live["available_bytes"])
            activation_tensor = torch.cat(activations, dim=0)
            mask_tensor = torch.cat(masks, dim=0)
            position_tensor = torch.cat(positions, dim=0)
            relative = f"observations/{args.bundle_id}/{rows[0]['stage']}_len{length}.safetensors"
            path = public_root / relative
            digest = save_observation_bundle(
                activations=activation_tensor,
                attention_mask=mask_tensor,
                position_ids=position_tensor,
                path=path,
                bundle_id=args.bundle_id,
                stage=str(rows[0]["stage"]),
                record_ids=[str(row["record_id"]) for row in rows],
            )
            descriptors.append(
                _bundle_descriptor(
                    bundle_id=args.bundle_id,
                    stage=str(rows[0]["stage"]),
                    length=length,
                    record_ids=[str(row["record_id"]) for row in rows],
                    artifact_path=path,
                    relative_path=Path(relative),
                    digest=digest,
                    mask_digest=_digest_tensor(mask_tensor),
                    position_digest=_digest_tensor(position_tensor),
                )
            )
            del activation_tensor, mask_tensor, position_tensor, activations, masks, positions
    generation_seconds = time.perf_counter() - generation_started
    _append_progress(progress_path, "observations_generated", elapsed_seconds=generation_seconds, records=len(all_rows), bundles=len(descriptors))

    index_path = public_root / "observation_index.json"
    write_json_exclusive(
        index_path,
        {
            "schema": OBSERVATION_INDEX_SCHEMA,
            "task_id": TASK_ID,
            "status": "OBSERVATIONS_READY_BEFORE_RECONSTRUCTION",
            "truth_opened": False,
            "source_truth_included": False,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "cut_depth": CUT_DEPTH,
            "bos_token_id": BOS_TOKEN_ID,
            "bundle_id": args.bundle_id,
            "record_order": [str(row["record_id"]) for row in all_rows],
            "bundles": descriptors,
        },
    )
    for descriptor in descriptors:
        (public_root / str(descriptor["relative_path"])).chmod(0o444)
    index_path.chmod(0o444)
    evidence_path = root / "generation_evidence.json"
    source_records = [file_record(path) for path in source_truth_paths]
    write_json_exclusive(
        evidence_path,
        {
            "schema": "token-reconstruction.trr-p03-generation-evidence.v1",
            "task_id": TASK_ID,
            "truth_opened": False,
            "source_truth_included": False,
            "status": "OPAQUE_OBSERVATIONS_WRITTEN_AS_DISTINCT_ARTIFACTS",
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "command": {"argv": [str(value) for value in sys.argv], "cwd": os.getcwd()},
            "environment": _runtime(args.seed, device=device),
            "implementation_commit": args.implementation_commit,
            "panel": {"path": str(panel_path), "sha256": panel_hash, "records": len(all_rows)},
            "record_selection": file_record(args.record_ids.resolve()) if args.record_ids else None,
            "source_truth_inputs_evaluator_only": source_records,
            "model": _asset_record(model_path) | {"cut_depth": CUT_DEPTH},
            "bundle": {"bundle_id": args.bundle_id, "stages": list(stage_names)},
            "geometry": {"records": len(all_rows), "scored_tokens": sum(int(row["sequence_length"]) - 1 for row in all_rows), "lengths": sorted(grouped)},
            "phases": {"model_load_seconds": load_seconds, "observation_generation_seconds": generation_seconds, "total_seconds": time.perf_counter() - started},
            "resource_guard": guard,
            "observation_index": file_record(index_path, root=root),
            "observation_bundles": [file_record(public_root / str(item["relative_path"]), root=root) for item in descriptors],
            "phase_progress": file_record(progress_path, root=root),
            "peak_memory": {"process_max_rss_bytes": _memory()["process_max_rss_bytes"]},
        },
    )
    evidence_path.chmod(0o444)
    finish_path = root / "generation_finish.json"
    write_json_exclusive(
        finish_path,
        {
            "schema": "token-reconstruction.trr-p03-generation-finish.v1",
            "task_id": TASK_ID,
            "status": "OBSERVATION_GENERATION_COMPLETE",
            "truth_opened": False,
            "bundle_id": args.bundle_id,
            "records": len(all_rows),
            "observation_index": file_record(index_path, root=root),
            "evidence": file_record(evidence_path, root=root),
            "phase_progress": file_record(progress_path, root=root),
        },
    )
    finish_path.chmod(0o444)
    print(json.dumps({"status": "prepared", "bundle_id": args.bundle_id, "records": len(all_rows), "observation_index": str(index_path), "evidence": str(evidence_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GenerationError, P03IOError) as exc:
        print(f"TRR-P03 generation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
