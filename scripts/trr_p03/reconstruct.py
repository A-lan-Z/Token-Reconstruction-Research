#!/usr/bin/env python3
"""Truth-free TRR-P03 reconstruction for the natural public panel.

The command receives only a sanitized observation index, public prototype
assets, and (for the fitted-origin comparators) the already published public
Alpaca lens. It writes all requested predictions and diagnostics before a
create-only freeze receipt is written. The scorer is a separate command.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Iterable, Mapping

from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
for _path in (_SOURCE_ROOT, _SOURCE_ROOT / "src", _SOURCE_ROOT / "scripts" / "trr_p01"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from token_reconstruction.public_prefix import ContiguousPublicPrefix
from token_reconstruction.trr_p01 import PrototypeTable
from token_reconstruction.trr_p01.historical_comparators import (
    HISTORICAL_LENS_ARTIFACT_SHA256,
    run_fixed_k256_a1_a2,
    load_published_frozen_lens,
)
from token_reconstruction.trr_p03.io import (
    BOS_TOKEN_ID,
    CUT_DEPTH,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    P03IOError,
    PREDICTION_SCHEMA,
    create_only_directory,
    file_record,
    freeze_prediction_bundle,
    load_index_and_observations,
    read_json,
    sha256_file,
    write_freeze_receipt,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from token_reconstruction.trr_p03.readouts import (
    LENS_SHA256,
    PROJECTED_SCHEMA,
    project_prototypes,
    rank_a1,
    rank_projected,
    rank_raw,
)
from common import load_model


TASK_ID = "TRR-P03"
VOCAB_SIZE = 128256
EXPECTED_TABLE_SHA256 = "51abc304d51134777d55347b219fe659817b9f0319add99756eeac6e9b6dd9a3"
DEFAULT_REQUIRED_BYTES = 10 * 1024**3
DEFAULT_EXPECTED_PEAK_BYTES = 8 * 1024**3
DEFAULT_QUERY_CHUNK_SIZE = 256
DEFAULT_PROTOTYPE_CHUNK_SIZE = 8192
METHOD_ALIASES = {
    "raw_boundary": "raw_boundary.cosine",
    "raw_boundary.cosine": "raw_boundary.cosine",
    "raw_boundary.l2": "raw_boundary.l2",
    "projected_boundary": "projected_boundary.cosine",
    "projected_boundary.cosine": "projected_boundary.cosine",
    "historical_a1": "historical_a1.cosine",
    "historical_a1.cosine": "historical_a1.cosine",
    "historical_a1_a2_anchor": "historical_a1_a2_anchor.cosine",
    "historical_a1_a2_anchor.cosine": "historical_a1_a2_anchor.cosine",
}


def _configure_runtime(seed: int) -> None:
    if seed < 0:
        raise ReconstructionError("seed must be non-negative")
    try:
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise ReconstructionError(
            "Torch thread configuration must happen before reconstruction"
        ) from exc
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class ReconstructionError(RuntimeError):
    """Raised when the opaque reconstruction contract cannot be preserved."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _append_progress(path: Path, event: str, **details: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        payload = {
            "schema": "token-reconstruction.trr-p03-phase-progress.v1",
            "event": event,
            "timestamp_utc": _utc_now(),
            **details,
        }
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _guard(required: int, expected: int) -> dict[str, Any]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemAvailable", "MemTotal"}:
                values[key] = int(rest.strip().split()[0]) * 1024
    except (OSError, UnicodeError, ValueError, IndexError):
        pass
    available = int(values.get("MemAvailable", 0))
    total = int(values.get("MemTotal", 0))
    if available <= 0 or total <= 0 or available < int(required):
        raise ReconstructionError(
            f"CPU resource guard failed closed: available={available} required={required} total={total}"
        )
    if int(required) <= int(expected):
        raise ReconstructionError("resource reservation must exceed expected peak")
    return {
        "status": "PASS",
        "required_bytes": int(required),
        "expected_peak_bytes": int(expected),
        "safety_margin_bytes": int(required - expected),
        "available_bytes_before": available,
        "total_bytes": total,
        "cuda_allocation": False,
    }


def _normalise_methods(raw: str) -> tuple[str, ...]:
    names: list[str] = []
    for value in raw.split(","):
        name = value.strip()
        if not name:
            continue
        if name not in METHOD_ALIASES:
            raise ReconstructionError(f"unknown reconstruction method: {name}")
        canonical = METHOD_ALIASES[name]
        if canonical not in names:
            names.append(canonical)
    if not names:
        raise ReconstructionError("at least one reconstruction method is required")
    return tuple(names)


def _read_anchor_ids(path: Path | None) -> list[str]:
    if path is None:
        return []
    value = read_json(path)
    if isinstance(value, list):
        values = value
    elif isinstance(value, Mapping) and isinstance(value.get("record_ids"), list):
        values = value["record_ids"]
    else:
        raise ReconstructionError("anchor declaration must be a JSON list or record_ids object")
    result = [str(item) for item in values]
    if len(set(result)) != len(result):
        raise ReconstructionError("anchor record IDs are duplicated")
    return result


def _load_projected(path: Path, *, raw_hash: str) -> torch.Tensor:
    if path.is_symlink() or not path.is_file():
        raise ReconstructionError(f"projected prototype artifact is missing: {path}")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"prototypes"}:
                raise ReconstructionError("projected prototype tensor fields changed")
            metadata = handle.metadata() or {}
            expected = {
                "schema",
                "task_id",
                "model_id",
                "model_revision",
                "cut_depth",
                "vocab_size",
                "hidden_size",
                "dtype",
                "source_prototype_sha256",
                "lens_sha256",
                "truth_opened",
            }
            if set(metadata) != expected or metadata.get("schema") != PROJECTED_SCHEMA:
                raise ReconstructionError("projected prototype metadata changed")
            if metadata.get("source_prototype_sha256") != raw_hash or metadata.get("lens_sha256") != LENS_SHA256:
                raise ReconstructionError("projected prototype source identity changed")
            if metadata.get("truth_opened") != "false":
                raise ReconstructionError("projected prototype truth state changed")
            projected = handle.get_tensor("prototypes")
    except ReconstructionError:
        raise
    except Exception as exc:
        raise ReconstructionError(f"projected prototype artifact is invalid: {path}") from exc
    if tuple(projected.shape) != (VOCAB_SIZE, HIDDEN_SIZE) or projected.dtype != torch.float32:
        raise ReconstructionError("projected prototype geometry or dtype changed")
    if not torch.isfinite(projected).all().item():
        raise ReconstructionError("projected prototype data is non-finite")
    return projected.contiguous()


def _flatten_observations(
    records: list[dict[str, Any]], observations: list[Any]
) -> tuple[torch.Tensor, list[int]]:
    queries: list[torch.Tensor] = []
    lengths: list[int] = []
    for record, observation in zip(records, observations, strict=True):
        active = observation.attention_mask[0].to(torch.bool)
        if not active.all().item():
            raise ReconstructionError("P03 observation rows must be fully active")
        if int(observation.position_ids[0, 0]) != 0:
            raise ReconstructionError("first active position must be zero")
        expected_positions = torch.arange(
            int(observation.activation.shape[1]), dtype=torch.long
        )
        if not torch.equal(observation.position_ids[0].to(torch.long), expected_positions):
            raise ReconstructionError("observation positions are not contiguous")
        if int(observation.activation.shape[1]) != int(record["sequence_length"]):
            raise ReconstructionError("observation length differs from index")
        value = observation.activation[0, 1:, :].contiguous()
        queries.append(value)
        lengths.append(int(value.shape[0]))
    if not queries:
        raise ReconstructionError("observation index has no query rows")
    return torch.cat(queries, dim=0), lengths


def _fill_prediction(
    rows: list[dict[str, Any]],
    lengths: list[int],
    flat_ids: torch.Tensor,
    *,
    max_length: int,
) -> torch.Tensor:
    result = torch.full((len(rows), max_length), -1, dtype=torch.int32)
    offset = 0
    for index, length in enumerate(lengths):
        result[index, 0] = BOS_TOKEN_ID
        result[index, 1 : length + 1] = flat_ids[offset : offset + length].to(torch.int32)
        offset += length
    if offset != int(flat_ids.numel()):
        raise ReconstructionError("prediction flattening offset changed")
    return result


def _fill_float(
    lengths: list[int], flat_values: torch.Tensor, *, max_length: int, fill: float = float("nan")
) -> torch.Tensor:
    result = torch.full((len(lengths), max_length), fill, dtype=torch.float32)
    offset = 0
    for index, length in enumerate(lengths):
        result[index, 1 : length + 1] = flat_values[offset : offset + length].float()
        offset += length
    if offset != int(flat_values.numel()):
        raise ReconstructionError("diagnostic flattening offset changed")
    return result


def _fill_int(
    lengths: list[int], flat_values: torch.Tensor, *, max_length: int
) -> torch.Tensor:
    result = torch.zeros((len(lengths), max_length), dtype=torch.int32)
    offset = 0
    for index, length in enumerate(lengths):
        result[index, 1 : length + 1] = flat_values[offset : offset + length].to(torch.int32)
        offset += length
    if offset != int(flat_values.numel()):
        raise ReconstructionError("tie diagnostic flattening offset changed")
    return result


def _phase_estimate(methods: tuple[str, ...], table_path: Path, model_needed: bool, projected_needed: bool, a2_needed: bool, args: argparse.Namespace) -> dict[str, Any]:
    table_bytes = int(table_path.stat().st_size)
    scratch = int(args.query_chunk_size) * VOCAB_SIZE * 4
    estimate = {
        "raw_table_bytes": table_bytes,
        "projected_table_bytes": VOCAB_SIZE * HIDDEN_SIZE * 4 if projected_needed else 0,
        "embedding_table_bytes": VOCAB_SIZE * HIDDEN_SIZE * 4 if model_needed else 0,
        "model_bytes": 2_500_000_000 if model_needed else 0,
        "query_score_scratch_bytes": scratch,
        "historical_cache_margin_bytes": 2_000_000_000 if a2_needed else 0,
        "expected_peak_bytes": int(args.expected_peak_bytes),
        "guard_required_bytes": int(args.required_bytes),
        "methods": list(methods),
    }
    return estimate


def _method_field(method: str) -> str:
    return method.replace(".", "_")


def _source_files() -> list[Path]:
    return [
        Path(__file__).resolve(),
        _SOURCE_ROOT / "src/token_reconstruction/trr_p03/io.py",
        _SOURCE_ROOT / "src/token_reconstruction/trr_p03/ranking.py",
        _SOURCE_ROOT / "src/token_reconstruction/trr_p03/readouts.py",
        _SOURCE_ROOT / "src/token_reconstruction/trr_p01/historical_comparators.py",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-index", type=Path, required=True)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--historical-lens", type=Path, default=None)
    parser.add_argument("--projected-prototype", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--anchor-records", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--methods", default="raw_boundary.cosine,projected_boundary.cosine,historical_a1.cosine")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--query-chunk-size", type=int, default=DEFAULT_QUERY_CHUNK_SIZE)
    parser.add_argument("--prototype-chunk-size", type=int, default=DEFAULT_PROTOTYPE_CHUNK_SIZE)
    parser.add_argument("--required-bytes", type=int, default=DEFAULT_REQUIRED_BYTES)
    parser.add_argument("--expected-peak-bytes", type=int, default=DEFAULT_EXPECTED_PEAK_BYTES)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--implementation-commit", default="UNBOUND_PRECOMMIT")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_runtime(args.seed)
    methods = _normalise_methods(args.methods)
    if args.query_chunk_size <= 0 or args.prototype_chunk_size < 2:
        raise ReconstructionError("chunk sizes are invalid")
    if args.seed < 0:
        raise ReconstructionError("seed must be non-negative")
    if torch.device(args.device).type != "cpu":
        raise ReconstructionError("P03 reconstruction is CPU-only in this stage")
    if any(option in sys.argv[1:] for option in ("--truth", "--source", "--dataset", "--target-model", "--target-lora", "--condition")):
        raise ReconstructionError("reconstruction accepts only opaque public observations")
    if "historical_a1_a2_anchor.cosine" in methods and args.anchor_records is None:
        raise ReconstructionError("A1+A2 anchor requires the predeclared anchor record list")
    if any(method in methods for method in ("projected_boundary.cosine", "historical_a1.cosine", "historical_a1_a2_anchor.cosine")) and args.historical_lens is None:
        raise ReconstructionError("historical lens is required by the requested method")

    index_path = args.observation_index.resolve()
    index, records, observations = load_index_and_observations(index_path)
    root = create_only_directory(args.output_root.resolve())
    progress_path = root / "phase_progress.jsonl"
    progress_path.touch()
    started = time.perf_counter()
    started_utc = _utc_now()
    _append_progress(progress_path, "input_loaded", records=len(records), methods=list(methods))

    table_path = args.prototype.resolve()
    lens_needed = any(
        method in methods
        for method in (
            "projected_boundary.cosine",
            "historical_a1.cosine",
            "historical_a1_a2_anchor.cosine",
        )
    )
    model_needed = any(
        method in methods
        for method in ("historical_a1.cosine", "historical_a1_a2_anchor.cosine")
    )
    a2_needed = "historical_a1_a2_anchor.cosine" in methods
    projected_needed = "projected_boundary.cosine" in methods
    estimate = _phase_estimate(
        methods, table_path, model_needed, projected_needed, a2_needed, args
    )
    # Fail closed before opening the 0.5 GiB raw table or any model asset.
    guard = _guard(int(args.required_bytes), int(args.expected_peak_bytes))
    preflight_path = root / "preflight.json"
    write_json_exclusive(
        preflight_path,
        {
            "schema": "token-reconstruction.trr-p03-reconstruction-preflight.v1",
            "task_id": TASK_ID,
            "truth_opened": False,
            "source_truth_included": False,
            "started_utc": started_utc,
            "implementation_commit": args.implementation_commit,
            "methods": list(methods),
            "input": file_record(index_path),
            "prototype": file_record(table_path),
            "estimate": estimate,
            "resource_guard": guard,
            "numerics": {
                "device": "cpu",
                "float32_scores": True,
                "query_chunk_size": int(args.query_chunk_size),
                "prototype_chunk_size": int(args.prototype_chunk_size),
                "deterministic_algorithms": True,
                "top1_tie_rule": "descending score, ascending token ID",
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        },
    )
    _append_progress(progress_path, "preflight_complete", elapsed_seconds=time.perf_counter() - started, **guard)

    raw_hash = sha256_file(table_path)
    if raw_hash != EXPECTED_TABLE_SHA256:
        raise ReconstructionError("boundary prototype identity changed")
    table = PrototypeTable.load(
        table_path,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_cut_depth=CUT_DEPTH,
        expected_vocab_size=VOCAB_SIZE,
        expected_hidden_size=HIDDEN_SIZE,
    )
    queries, lengths = _flatten_observations(records, observations)
    max_length = max(int(row["sequence_length"]) for row in records)
    prediction_values: dict[str, torch.Tensor] = {}
    score_values: dict[str, torch.Tensor] = {}
    margin_values: dict[str, torch.Tensor] = {}
    tie_values: dict[str, torch.Tensor] = {}
    method_meta: dict[str, dict[str, Any]] = {}
    candidate_sets: torch.Tensor | None = None
    model = None
    prefix = None
    lens = None
    projected = None
    embedding = None

    if lens_needed:
        lens_path = args.historical_lens.resolve()
        if sha256_file(lens_path) != HISTORICAL_LENS_ARTIFACT_SHA256:
            raise ReconstructionError("historical lens identity changed")
        lens = load_published_frozen_lens(lens_path, device=torch.device("cpu"))
        _append_progress(progress_path, "lens_loaded", lens_sha256=HISTORICAL_LENS_ARTIFACT_SHA256)
    if projected_needed:
        projected_started = time.perf_counter()
        if args.projected_prototype is not None:
            projected = _load_projected(args.projected_prototype.resolve(), raw_hash=raw_hash)
            projected_source = "prepared_artifact"
        else:
            projected = project_prototypes(
                table.prototypes,
                lens,
                prototype_chunk_size=args.prototype_chunk_size,
            )
            projected_source = "constructed_once_in_reconstruction"
        _append_progress(
            progress_path,
            "projected_table_ready",
            elapsed_seconds=time.perf_counter() - projected_started,
            source=projected_source,
        )
    if model_needed:
        model_started = time.perf_counter()
        model = load_model(
            device=torch.device("cpu"),
            model_path=args.model_path.resolve() if args.model_path else None,
        )
        model.eval()
        embedding = F.normalize(
            model.get_input_embeddings().weight.detach().float().cpu(), dim=-1
        ).contiguous()
        if tuple(embedding.shape) != (VOCAB_SIZE, HIDDEN_SIZE):
            raise ReconstructionError("public embedding geometry changed")
        if a2_needed:
            prefix = ContiguousPublicPrefix(model, CUT_DEPTH).to(torch.device("cpu")).eval()
        else:
            # A1 needs only the normalized public embedding table. Releasing
            # the full decoder keeps table-only diagnostics memory bounded.
            del model
            model = None
            gc.collect()
        _append_progress(
            progress_path,
            "public_assets_ready",
            elapsed_seconds=time.perf_counter() - model_started,
            retained_decoder=bool(prefix is not None),
        )

    static_started = time.perf_counter()
    for method in methods:
        if method == "raw_boundary.cosine":
            result = rank_raw(
                queries,
                table.prototypes,
                metric="cosine",
                query_chunk_size=args.query_chunk_size,
                prototype_chunk_size=args.prototype_chunk_size,
            )
            units = "cosine"
        elif method == "raw_boundary.l2":
            result = rank_raw(
                queries,
                table.prototypes,
                metric="l2",
                query_chunk_size=args.query_chunk_size,
                prototype_chunk_size=args.prototype_chunk_size,
            )
            units = "negative_squared_l2"
        elif method == "projected_boundary.cosine":
            if lens is None or projected is None:
                raise ReconstructionError("projected method assets are unavailable")
            result = rank_projected(
                queries,
                table.prototypes,
                lens,
                projected_prototypes=projected,
                metric="cosine",
                query_chunk_size=args.query_chunk_size,
                prototype_chunk_size=args.prototype_chunk_size,
            )
            units = "projected_cosine"
        elif method == "historical_a1.cosine":
            if lens is None or embedding is None:
                raise ReconstructionError("A1 method assets are unavailable")
            a1 = rank_a1(
                queries,
                lens,
                embedding,
                query_chunk_size=args.query_chunk_size,
                prototype_chunk_size=args.prototype_chunk_size,
            )
            result = a1.ranking
            units = a1.score_units
            method_meta[method] = {
                "score_scale_exp_s": a1.score_scale,
                "score_units": a1.score_units,
                "cosine_equivalent_units": a1.cosine_equivalent_units,
            }
        elif method == "historical_a1_a2_anchor.cosine":
            continue
        else:
            raise ReconstructionError(f"unsupported canonical method: {method}")
        prediction_values[method] = _fill_prediction(
            records, lengths, result.top1_ids, max_length=max_length
        )
        score_values[method] = _fill_float(
            lengths, result.top1_scores, max_length=max_length, fill=float("nan")
        )
        margin_values[method] = _fill_float(
            lengths, result.margins, max_length=max_length, fill=float("nan")
        )
        tie_values[method] = _fill_int(
            lengths, result.top1_tie_count, max_length=max_length
        )
        method_meta.setdefault(method, {"score_units": units})
    _append_progress(
        progress_path,
        "static_predictions_ready",
        elapsed_seconds=time.perf_counter() - static_started,
        methods=[method for method in methods if method != "historical_a1_a2_anchor.cosine"],
        query_rows=int(queries.shape[0]),
    )

    if a2_needed:
        assert prefix is not None and lens is not None
        anchor_ids = _read_anchor_ids(args.anchor_records.resolve())
        by_id = {str(record["record_id"]): (index, record) for index, record in enumerate(records)}
        if not anchor_ids:
            raise ReconstructionError("anchor declaration is empty")
        missing = [record_id for record_id in anchor_ids if record_id not in by_id]
        if missing:
            raise ReconstructionError(f"anchor records are absent from observation index: {missing}")
        anchor_indices = [by_id[record_id][0] for record_id in anchor_ids]
        if any(int(records[index]["sequence_length"]) != 40 for index in anchor_indices):
            raise ReconstructionError("native A1+A2 anchor requires exact 40-slot observations")
        anchor_observations = torch.cat(
            [observations[index].activation for index in anchor_indices], dim=0
        ).contiguous()
        a2_started = time.perf_counter()
        native = run_fixed_k256_a1_a2(
            observations=anchor_observations,
            public_prefix=prefix,
            frozen_lens=lens,
            device=torch.device("cpu"),
            record_batch_size=8,
        )
        anchor_matrix = torch.full((len(records), max_length), -1, dtype=torch.int32)
        anchor_scores = torch.full((len(records), max_length), float("nan"), dtype=torch.float32)
        anchor_margins = torch.full((len(records), max_length), float("nan"), dtype=torch.float32)
        anchor_ties = torch.zeros((len(records), max_length), dtype=torch.int32)
        candidate_sets = torch.full(
            (len(records), max_length, native.candidates.shape[2]), -1, dtype=torch.int32
        )
        selection = native.selection_scores[:, 1:, :].float()
        for local, global_index in enumerate(anchor_indices):
            anchor_matrix[global_index, :40] = native.predictions[local].to(torch.int32)
            anchor_scores[global_index, 1:40] = selection[local].amax(dim=1)
            top = torch.topk(selection[local], k=2, dim=1, largest=True, sorted=True).values
            anchor_margins[global_index, 1:40] = top[:, 0] - top[:, 1]
            anchor_ties[global_index, 1:40] = selection[local].eq(
                selection[local].amax(dim=1, keepdim=True)
            ).sum(dim=1).to(torch.int32)
            candidate_sets[global_index, :40] = native.candidates[local].to(torch.int32)
        prediction_values["historical_a1_a2_anchor.cosine"] = anchor_matrix
        score_values["historical_a1_a2_anchor.cosine"] = anchor_scores
        margin_values["historical_a1_a2_anchor.cosine"] = anchor_margins
        tie_values["historical_a1_a2_anchor.cosine"] = anchor_ties
        method_meta["historical_a1_a2_anchor.cosine"] = {
            "score_units": "public_prefix_candidate_cosine",
            "coverage_records": len(anchor_indices),
            "anchor_record_ids": anchor_ids,
            "candidate_budget": 256,
            "proposal_budget": 512,
            "candidate_simulations": int(native.candidate_simulations),
            "executed_candidate_simulations": int(native.executed_candidate_simulations),
            "public_prefix_calls": int(native.public_prefix_calls),
            "a1_forward_calls": int(native.a1_forward_calls),
            "native_policy_id": native.policy_id,
            "native_a1_topk_tie_rule": "published torch.topk proposal order",
        }
        _append_progress(
            progress_path,
            "native_a1_a2_anchor_ready",
            elapsed_seconds=time.perf_counter() - a2_started,
            anchor_records=len(anchor_indices),
            candidate_simulations=int(native.candidate_simulations),
        )

    # Predictions and diagnostics are written before any evidence, freeze, or
    # scorer operation. Their methods and record masks are committed in the
    # safetensors metadata for a stable downstream join.
    method_masks = {
        method: [int(value[0].item() >= 0) for value in prediction_values[method]]
        for method in prediction_values
    }
    field_map = {_method_field(method): method for method in prediction_values}
    prediction_path = root / "predictions.safetensors"
    save_file(
        {_method_field(method): value for method, value in prediction_values.items()},
        prediction_path,
        metadata={
            "schema": PREDICTION_SCHEMA,
            "task_id": TASK_ID,
            "truth_opened": "false",
            "methods_json": json.dumps(list(prediction_values), separators=(",", ":")),
            "field_map_json": json.dumps(field_map, sort_keys=True, separators=(",", ":")),
            "record_order_json": json.dumps([record["record_id"] for record in records], separators=(",", ":")),
            "sequence_lengths_json": json.dumps([int(record["sequence_length"]) for record in records], separators=(",", ":")),
            "method_masks_json": json.dumps(method_masks, sort_keys=True, separators=(",", ":")),
        },
    )
    diagnostics_path = root / "lookup_diagnostics.safetensors"
    diagnostic_tensors: dict[str, torch.Tensor] = {}
    for method in prediction_values:
        field = _method_field(method)
        diagnostic_tensors[f"{field}__scores"] = score_values[method].contiguous()
        diagnostic_tensors[f"{field}__margins"] = margin_values[method].contiguous()
        diagnostic_tensors[f"{field}__top1_tie_count"] = tie_values[method].contiguous()
    save_file(
        diagnostic_tensors,
        diagnostics_path,
        metadata={
            "schema": "token-reconstruction.trr-p03-lookup-diagnostics.v1",
            "task_id": TASK_ID,
            "truth_opened": "false",
            "field_map_json": json.dumps(field_map, sort_keys=True, separators=(",", ":")),
        },
    )
    candidate_path: Path | None = None
    if candidate_sets is not None:
        candidate_path = root / "candidate_sets.safetensors"
        save_file(
            {"historical_a1_a2_anchor_candidates": candidate_sets},
            candidate_path,
            metadata={
                "schema": "token-reconstruction.trr-p03-candidate-sets.v1",
                "task_id": TASK_ID,
                "truth_opened": "false",
                "record_order_json": json.dumps([record["record_id"] for record in records], separators=(",", ":")),
            },
        )

    prediction_rows: list[dict[str, Any]] = []
    for method in prediction_values:
        for index, record in enumerate(records):
            if method_masks[method][index] == 0:
                continue
            length = int(record["sequence_length"])
            prediction_rows.append(
                {
                    "record_id": str(record["record_id"]),
                    "method": method,
                    "sequence_length": length,
                    "prediction_tokens": [int(value) for value in prediction_values[method][index, :length].tolist()],
                    "top1_tie_count": [int(value) for value in tie_values[method][index, 1:length].tolist()],
                    "top1_scores": [float(value) for value in score_values[method][index, 1:length].tolist()],
                    "top1_runner_margins": [float(value) for value in margin_values[method][index, 1:length].tolist()],
                    "score_units": method_meta[method].get("score_units", "unspecified"),
                    "observation_sha256": str(records[index].get("sha256")),
                    "truth_opened": False,
                }
            )
    rows_path = root / "predictions.jsonl"
    write_jsonl_exclusive(rows_path, prediction_rows)
    io_started = time.perf_counter()
    _append_progress(
        progress_path,
        "prediction_artifacts_written",
        elapsed_seconds=time.perf_counter() - io_started,
        methods=list(prediction_values),
        rows=len(prediction_rows),
    )

    evidence_path = root / "reconstructor_evidence.json"
    source_records = []
    for path in _source_files():
        if path.is_file():
            source_records.append(file_record(path))
    evidence = {
        "schema": "token-reconstruction.trr-p03-reconstructor-evidence.v1",
        "task_id": TASK_ID,
        "truth_opened": False,
        "source_truth_included": False,
        "status": "PREDICTIONS_READY_FOR_FREEZE",
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "command": {
            "argv": [str(value) for value in sys.argv],
            "cwd": os.getcwd(),
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "HF_HUB_OFFLINE",
                    "HF_DATASETS_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "CUDA_VISIBLE_DEVICES",
                    "PYTHONPATH",
                )
            },
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "platform": platform.platform(),
            "kernel": platform.uname()._asdict(),
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "seed": int(args.seed),
        },
        "implementation_commit": args.implementation_commit,
        "decision_predeclaration": file_record(args.plan.resolve()) if args.plan else file_record(index_path),
        "observation_index": file_record(index_path),
        "prototype": file_record(table_path),
        "historical_lens": file_record(args.historical_lens.resolve()) if lens_needed and args.historical_lens else None,
        "projected_prototype": file_record(args.projected_prototype.resolve()) if projected is not None and args.projected_prototype else None,
        "methods": list(prediction_values),
        "method_metadata": method_meta,
        "records": len(records),
        "scored_tokens": int(sum(int(record["sequence_length"]) - 1 for record in records)),
        "candidate_simulations": int(method_meta.get("historical_a1_a2_anchor.cosine", {}).get("candidate_simulations", 0)),
        "phase_progress": file_record(progress_path),
        "preflight": file_record(preflight_path),
        "peak_memory": {
            "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        },
        "code_files": source_records,
    }
    # Add the prediction hashes only after their bytes are complete.
    evidence["prediction_artifacts"] = {
        "predictions": file_record(prediction_path),
        "diagnostics": file_record(diagnostics_path),
        "prediction_rows": file_record(rows_path),
        "candidate_sets": file_record(candidate_path) if candidate_path else None,
    }
    write_json_exclusive(evidence_path, evidence)
    frozen_artifacts = [
        preflight_path,
        progress_path,
        prediction_path,
        diagnostics_path,
        rows_path,
        evidence_path,
    ]
    if candidate_path is not None:
        frozen_artifacts.append(candidate_path)
    plan_hash = sha256_file(args.plan.resolve()) if args.plan else sha256_file(index_path)
    freeze_payload = freeze_prediction_bundle(
        root=root,
        plan_hash=plan_hash,
        implementation_commit=args.implementation_commit,
        artifacts=frozen_artifacts,
        metadata={
            "methods": list(prediction_values),
            "records": len(records),
            "record_ids": [str(record["record_id"]) for record in records],
            "anchor_record_ids": (
                list(_read_anchor_ids(args.anchor_records.resolve()))
                if a2_needed and args.anchor_records is not None
                else []
            ),
            "truth_opened_before_freeze": False,
            "phase_receipt": "phase_progress.jsonl",
        },
    )
    freeze_path = write_freeze_receipt(root, freeze_payload)
    finish_path = root / "finish_receipt.json"
    write_json_exclusive(
        finish_path,
        {
            "schema": "token-reconstruction.trr-p03-reconstruction-finish.v1",
            "task_id": TASK_ID,
            "status": "PREDICTIONS_FROZEN_BEFORE_TRUTH",
            "truth_opened": False,
            "created_utc": _utc_now(),
            "implementation_commit": args.implementation_commit,
            "methods": list(prediction_values),
            "records": len(records),
            "predictions": file_record(prediction_path),
            "diagnostics": file_record(diagnostics_path),
            "prediction_rows": file_record(rows_path),
            "freeze_receipt": file_record(freeze_path),
            "evidence": file_record(evidence_path),
            "phase_progress": file_record(progress_path),
            "peak_memory": {
                "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            },
        },
    )
    finish_path.chmod(0o444)
    print(
        json.dumps(
            {
                "status": "PREDICTIONS_FROZEN_BEFORE_TRUTH",
                "output_root": str(root),
                "methods": list(prediction_values),
                "records": len(records),
                "freeze_receipt": str(freeze_path),
                "finish_receipt": str(finish_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionError, P03IOError) as exc:
        print(f"TRR-P03 reconstruction failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
