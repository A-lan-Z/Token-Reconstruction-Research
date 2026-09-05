#!/usr/bin/env python3
"""Run the bounded TRR-P02 public representation-geometry diagnosis.

The runner deliberately uses fully known public teacher-prefix inputs.  It is
not a reconstructor and never accepts a source record, target model, private
truth tensor, or correctness signal.  A single small activation panel drives
the cache/reference checks, shared-offset summaries, local-N=8 rankings, and
the predeclared twelve-row full-vocabulary lens comparison.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
for _path in (_SOURCE_ROOT / "src", _SOURCE_ROOT / "scripts" / "trr_p01"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from token_reconstruction.public_prefix import ContiguousPublicPrefix  # noqa: E402
from token_reconstruction.trr_p01 import PrototypeTable  # noqa: E402
from token_reconstruction.trr_p01.historical_comparators import (  # noqa: E402
    HISTORICAL_LENS_ARTIFACT_SHA256,
    load_published_frozen_lens,
)
from token_reconstruction.trr_p02 import (  # noqa: E402
    ContextSpec,
    GeometryDiagnosticError,
    pairwise_token_deformation,
    rank_metrics,
    reference_corrected_query,
    separation_summary,
    summarize_offsets,
)
from common import load_public_model  # noqa: E402


TASK_ID = "TRR-P02"
SCHEMA = "token-reconstruction.trr-p02-geometry-diagnostics.v1"
STRICT_RANK_DEFINITION = (
    "strict rank = 1 + count(score > true_score); equal scores are reported in "
    "true_equal_count, while top-1/runner IDs use descending score then ascending ID"
)
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
BOS_TOKEN_ID = 128000
CUT_DEPTH = 4
VOCAB_SIZE = 128256
HIDDEN_SIZE = 2048
MAX_RANKING_QUERY_ROWS = 16
RANKING_SCORE_BUFFER_BYTES = MAX_RANKING_QUERY_ROWS * VOCAB_SIZE * 4
TABLE_SHA256 = "51abc304d51134777d55347b219fe659817b9f0319add99756eeac6e9b6dd9a3"
LENS_SHA256 = HISTORICAL_LENS_ARTIFACT_SHA256
EXPECTED_CANDIDATE_IDS = (13, 32, 198, 220, 2048, 4096, 16384, 29871)
EXPECTED_REFERENCE_ID = 220
EXPECTED_POSITION_IDS = (220, 2048, 4096)
EXPECTED_CONTEXTS = (
    ("C0_bos_baseline", (BOS_TOKEN_ID,)),
    ("C1_same_length_13", (BOS_TOKEN_ID, 13)),
    ("C2_same_length_198", (BOS_TOKEN_ID, 198)),
    ("C3_same_length_1024", (BOS_TOKEN_ID, 1024)),
    ("C4_same_length_29871", (BOS_TOKEN_ID, 29871)),
    ("C5_repeat_13_length_2", (BOS_TOKEN_ID, 13, 13)),
    ("C6_repeat_13_length_3", (BOS_TOKEN_ID, 13, 13, 13)),
)
PRIMARY_CONTEXT_INDICES = (0, 1, 2, 3, 4)
REPEATED_CONTEXT_INDICES = (0, 1, 5, 6)
REPEATED_ENDPOINT_IDS = EXPECTED_POSITION_IDS
TARGETED_CONTEXT_INDICES = (1, 2, 3, 4)
TARGETED_ENDPOINT_IDS = EXPECTED_POSITION_IDS
TORCH_THREADS = 8
TORCH_INTEROP_THREADS = 1
SEED = 314159
EXPECTED_PEAK_RSS_BYTES = 8 * 1024**3
RSS_CEILING_BYTES = 8 * 1024**3
GUARD_REQUIRED_BYTES = 10 * 1024**3
QUALIFICATION_CONTEXT_INDEX = 6


class DiagnosticError(RuntimeError):
    """Raised when the P02 public diagnostic contract cannot be preserved."""


@dataclass
class Counters:
    full_calls: int = 0
    full_input_tokens: int = 0
    cached_calls: int = 0
    cached_input_tokens: int = 0

    @property
    def public_prefix_calls(self) -> int:
        return self.full_calls + self.cached_calls

    @property
    def public_prefix_input_token_evaluations(self) -> int:
        return self.full_input_tokens + self.cached_input_tokens


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise DiagnosticError(f"artifact must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    label = str(path)
    if root is not None:
        try:
            label = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            label = str(path)
    return {"path": label, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _read_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DiagnosticError(f"plan must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"invalid JSON plan: {path}") from exc
    if not isinstance(value, Mapping):
        raise DiagnosticError("plan root must be an object")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise DiagnosticError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _require_regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise DiagnosticError(f"{label} must be a regular file: {path}")
    return path.resolve()


def _create_only_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise DiagnosticError(f"output directory already exists: {path}")
    path.mkdir(parents=True)
    return path.resolve()


def _host_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                values[key] = int(rest.strip().split()[0]) * 1024
    except (OSError, UnicodeError, ValueError, IndexError):
        pass
    values["process_max_rss_bytes"] = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ) * 1024
    return values


def _resource_guard(required_bytes: int) -> dict[str, Any]:
    memory = _host_memory()
    available = int(memory.get("MemAvailable", 0))
    total = int(memory.get("MemTotal", 0))
    if available <= 0 or total <= 0 or available < int(required_bytes):
        raise DiagnosticError(
            "CPU resource guard failed closed: "
            f"available={available} required={required_bytes} total={total}"
        )
    return {
        "status": "PASS",
        "selected_device": "cpu",
        "required_bytes": int(required_bytes),
        "available_bytes_before": available,
        "total_bytes": total,
        "expected_peak_rss_bytes": EXPECTED_PEAK_RSS_BYTES,
        "safety_margin_bytes_above_expected": int(required_bytes - EXPECTED_PEAK_RSS_BYTES),
        "allocation_bytes": 0,
        "device_guard": "CPU_NO_CUDA_GUARD",
        "memory": memory,
    }


def _rss_ceiling_check(stage: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Record the process high-water RSS and fail closed above the reservation."""

    memory = _host_memory()
    observed = int(memory["process_max_rss_bytes"])
    result = {
        "stage": str(stage),
        "status": "PASS" if observed <= RSS_CEILING_BYTES else "FAIL",
        "process_max_rss_bytes": observed,
        "ceiling_bytes": RSS_CEILING_BYTES,
        "memory": memory,
    }
    checks.append(result)
    if observed > RSS_CEILING_BYTES:
        raise DiagnosticError(
            "process RSS reservation exceeded at "
            f"{stage}: observed={observed} ceiling={RSS_CEILING_BYTES}"
        )
    return result


def _runtime_record() -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "platform": platform.platform(),
        "kernel": platform.uname()._asdict(),
        "pid": os.getpid(),
        "selected_device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_allocated": False,
        "seed": SEED,
        "ranking_score_buffer_max_bytes": RANKING_SCORE_BUFFER_BYTES,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def _digest_tensor(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    descriptor = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(descriptor + b"\0" + raw).hexdigest()


def _stats(value: torch.Tensor) -> dict[str, float | int]:
    value = value.detach().float().reshape(-1)
    if value.numel() <= 0 or not torch.isfinite(value).all().item():
        raise DiagnosticError("metric values must be finite and non-empty")
    quantiles = torch.quantile(value, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32))
    return {
        "count": int(value.numel()),
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "min": float(value.min().item()),
        "p10": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
        "max": float(value.max().item()),
    }


def _finite(value: torch.Tensor, label: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point:
        raise DiagnosticError(f"{label} must be a floating-point tensor")
    if not torch.isfinite(value).all().item():
        raise DiagnosticError(f"{label} contains non-finite values")


def _validate_plan(plan: Mapping[str, Any]) -> tuple[list[int], int, list[ContextSpec]]:
    if plan.get("schema") != "token-reconstruction.trr-p02-geometry-plan.v2":
        raise DiagnosticError("P02 plan schema changed")
    if plan.get("task_id") != TASK_ID or plan.get("status") != "PREDECLARED_PUBLIC_DIAGNOSTIC":
        raise DiagnosticError("P02 plan identity changed")
    if plan.get("truth_opened") is not False or plan.get("source_truth_included") is not False:
        raise DiagnosticError("P02 plan truth state changed")
    model = plan.get("model")
    if not isinstance(model, Mapping) or {
        model.get("id"),
        model.get("revision"),
        model.get("cut_depth"),
        model.get("hidden_size"),
        model.get("vocab_size"),
        model.get("bos_token_id"),
        model.get("dtype"),
    } != {
        MODEL_ID,
        MODEL_REVISION,
        CUT_DEPTH,
        HIDDEN_SIZE,
        VOCAB_SIZE,
        BOS_TOKEN_ID,
        "bfloat16",
    }:
        raise DiagnosticError("P02 public model identity changed")
    tokens = plan.get("tokens")
    if not isinstance(tokens, Mapping):
        raise DiagnosticError("P02 token declaration missing")
    candidates = tokens.get("candidate_ids")
    reference = tokens.get("reference_id")
    position_ids = tokens.get("position_control_ids")
    if (
        not isinstance(candidates, list)
        or tuple(int(value) for value in candidates) != EXPECTED_CANDIDATE_IDS
        or reference != EXPECTED_REFERENCE_ID
        or not isinstance(position_ids, list)
        or tuple(int(value) for value in position_ids) != EXPECTED_POSITION_IDS
    ):
        raise DiagnosticError("P02 token declaration changed")
    contexts_raw = plan.get("contexts")
    if not isinstance(contexts_raw, list) or len(contexts_raw) != len(EXPECTED_CONTEXTS):
        raise DiagnosticError("P02 context declaration changed")
    runtime = plan.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("device") != "cpu"
        or int(runtime.get("torch_threads", -1)) != TORCH_THREADS
        or int(runtime.get("inter_op_threads", -1)) != TORCH_INTEROP_THREADS
        or runtime.get("deterministic_algorithms") is not True
        or int(runtime.get("seed", -1)) != SEED
        or int(runtime.get("ranking_score_buffer_max_bytes", -1)) != RANKING_SCORE_BUFFER_BYTES
        or int(runtime.get("expected_peak_rss_bytes", -1)) != EXPECTED_PEAK_RSS_BYTES
        or int(runtime.get("rss_ceiling_bytes", -1)) != RSS_CEILING_BYTES
        or int(runtime.get("resource_guard_required_bytes", -1)) != GUARD_REQUIRED_BYTES
    ):
        raise DiagnosticError("P02 runtime/resource declaration changed")
    contexts: list[ContextSpec] = []
    for observed, expected in zip(contexts_raw, EXPECTED_CONTEXTS, strict=True):
        if not isinstance(observed, Mapping):
            raise DiagnosticError("P02 context entry is not an object")
        name, token_ids = expected
        if observed.get("name") != name or tuple(observed.get("token_ids", ())) != token_ids:
            raise DiagnosticError(f"P02 context changed: {name}")
        context = ContextSpec(name=name, token_ids=token_ids)
        context.validate(bos_token_id=BOS_TOKEN_ID, vocab_size=VOCAB_SIZE)
        contexts.append(context)
    panel = plan.get("panel")
    if not isinstance(panel, Mapping):
        raise DiagnosticError("P02 panel declaration missing")
    if tuple(panel.get("primary_context_indices", ())) != PRIMARY_CONTEXT_INDICES:
        raise DiagnosticError("P02 primary context declaration changed")
    if tuple(panel.get("repeated_context_indices", ())) != REPEATED_CONTEXT_INDICES:
        raise DiagnosticError("P02 repeated context declaration changed")
    if tuple(panel.get("repeated_endpoint_ids", ())) != REPEATED_ENDPOINT_IDS:
        raise DiagnosticError("P02 repeated endpoint declaration changed")
    if int(panel.get("total_activation_rows", -1)) != 46:
        raise DiagnosticError("P02 panel row count changed")
    return [int(value) for value in candidates], int(reference), contexts


def _last_hidden(value: Any) -> torch.Tensor:
    if isinstance(value, tuple):
        if not value:
            raise DiagnosticError("public prefix returned an empty tuple")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise DiagnosticError("public prefix returned a non-tensor")
    if value.ndim == 3:
        value = value[:, -1, :]
    if value.ndim != 2 or value.shape[1] != HIDDEN_SIZE:
        raise DiagnosticError("public prefix returned invalid hidden geometry")
    _finite(value, "public prefix output")
    return value


def _full_last(
    prefix: Any, input_ids: torch.Tensor, counters: Counters
) -> torch.Tensor:
    output = _last_hidden(prefix.forward_full(input_ids))
    counters.full_calls += 1
    counters.full_input_tokens += int(input_ids.numel())
    return output.detach().cpu().contiguous()


def _cached_last(
    prefix: Any, input_ids: torch.Tensor, cache: Any, start_pos: int, counters: Counters
) -> torch.Tensor:
    output = _last_hidden(prefix.run_cached(input_ids, cache, start_pos))
    counters.cached_calls += 1
    counters.cached_input_tokens += int(input_ids.numel())
    return output.detach().cpu().contiguous()


def _compare(left: torch.Tensor, right: torch.Tensor, label: str) -> dict[str, Any]:
    left = left.detach().cpu()
    right = right.detach().cpu()
    if left.shape != right.shape:
        raise DiagnosticError(f"{label} shape changed: {tuple(left.shape)} != {tuple(right.shape)}")
    difference = (left.float() - right.float()).abs()
    _finite(difference, f"{label} difference")
    return {
        "shape": list(left.shape),
        "dtype_left": str(left.dtype).replace("torch.", ""),
        "dtype_right": str(right.dtype).replace("torch.", ""),
        "torch_equal": bool(torch.equal(left, right)),
        "maximum_absolute_difference": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_absolute_difference": float(difference.mean().item()) if difference.numel() else 0.0,
    }


def _cache_lengths(prefix: Any, cache: Any) -> dict[str, Any]:
    result = {"logical_length": int(getattr(cache, "length", -1))}
    method = getattr(prefix, "_cache_layer_lengths", None)
    if callable(method):
        result["layer_lengths"] = [int(value) for value in method(cache)]
    return result


def _collect_panel(
    prefix: Any,
    contexts: Sequence[ContextSpec],
    candidate_ids: Sequence[int],
    counters: Counters,
) -> tuple[torch.Tensor, dict[int, dict[int, torch.Tensor]], dict[int, torch.Tensor]]:
    """Collect one batched endpoint pass per context plus full context outputs."""

    primary = torch.empty(
        (len(PRIMARY_CONTEXT_INDICES), len(candidate_ids), HIDDEN_SIZE), dtype=torch.bfloat16
    )
    endpoint: dict[int, dict[int, torch.Tensor]] = {}
    context_last: dict[int, torch.Tensor] = {}
    candidate_by_context: dict[int, tuple[int, ...]] = {
        index: tuple(candidate_ids) for index in PRIMARY_CONTEXT_INDICES
    }
    for index in REPEATED_CONTEXT_INDICES[2:]:
        candidate_by_context[index] = REPEATED_ENDPOINT_IDS
    for index, context in enumerate(contexts):
        context_tensor = torch.tensor([context.token_ids], dtype=torch.long)
        context_last[index] = _full_last(prefix, context_tensor, counters)[0]
        values = candidate_by_context[index]
        rows = [tuple(context.token_ids) + (int(token),) for token in values]
        inputs = torch.tensor(rows, dtype=torch.long)
        outputs = _full_last(prefix, inputs, counters)
        endpoint[index] = {
            int(token): outputs[row].detach().clone() for row, token in enumerate(values)
        }
        if index in PRIMARY_CONTEXT_INDICES:
            primary_row = PRIMARY_CONTEXT_INDICES.index(index)
            primary[primary_row] = outputs
    _finite(primary, "primary public activation panel")
    return primary, endpoint, context_last


def _qualify_short_cell(
    prefix: Any,
    context: ContextSpec,
    candidate_ids: Sequence[int],
    counters: Counters,
) -> dict[str, Any]:
    """Qualify the largest short endpoint batch before collecting the panel.

    The batch and one-row implementations are both public teacher-prefix
    executions.  The diagnostic uses the batch path only after exact ordered
    output equality has been established.
    """

    rows = [tuple(context.token_ids) + (int(token),) for token in candidate_ids]
    batched_input = torch.tensor(rows, dtype=torch.long)
    batched = _last_hidden(prefix.forward_full(batched_input)).detach().cpu().contiguous()
    counters.full_calls += 1
    counters.full_input_tokens += int(batched_input.numel())
    individual: list[torch.Tensor] = []
    for row in rows:
        one = torch.tensor([row], dtype=torch.long)
        individual.append(_last_hidden(prefix.forward_full(one))[0].detach().cpu().contiguous())
        counters.full_calls += 1
        counters.full_input_tokens += int(one.numel())
    alternate = torch.stack(individual)
    check = _compare(batched, alternate, "short-cell batch equivalence")
    if not check["torch_equal"]:
        raise DiagnosticError(
            "largest representative short-cell batch is not output-equivalent; "
            "the planned batched diagnostic is excluded"
        )
    return {
        "status": "QUALIFIED_EQUIVALENT",
        "context_name": context.name,
        "context_token_ids": list(context.token_ids),
        "endpoint_position": len(context.token_ids),
        "candidate_ids": [int(value) for value in candidate_ids],
        "batch_shape": list(batched.shape),
        "batch_forward_calls": 1,
        "alternate_forward_calls": len(individual),
        "equivalence": check,
        "selection": "batch path retained after exact ordered equality",
    }


def _cache_and_reference_checks(
    prefix: Any,
    contexts: Sequence[ContextSpec],
    endpoint: Mapping[int, Mapping[int, torch.Tensor]],
    reference_id: int,
    counters: Counters,
) -> tuple[list[dict[str, Any]], torch.Tensor, list[dict[str, Any]]]:
    cache_rows: list[dict[str, Any]] = []
    reference_outputs: list[torch.Tensor] = []
    reference_rows: list[dict[str, Any]] = []
    for context_index, context in enumerate(contexts):
        context_ids = torch.tensor([context.token_ids], dtype=torch.long)
        full_context = _full_last(prefix, context_ids, counters)[0]

        block_cache = prefix.new_cache()
        cached_block = _cached_last(prefix, context_ids, block_cache, 0, counters)[0]
        block_check = _compare(cached_block, full_context, "cached context block")
        block_lengths = _cache_lengths(prefix, block_cache)

        incremental_cache = prefix.new_cache()
        incremental_outputs: list[torch.Tensor] = []
        for position, token in enumerate(context.token_ids):
            token_tensor = torch.tensor([[int(token)]], dtype=torch.long)
            incremental_outputs.append(
                _cached_last(prefix, token_tensor, incremental_cache, position, counters)[0]
            )
        incremental_check = _compare(
            incremental_outputs[-1], full_context, "incremental cached context"
        )
        incremental_lengths = _cache_lengths(prefix, incremental_cache)

        endpoint_checks: list[dict[str, Any]] = []
        for token in endpoint[context_index]:
            probe_cache = copy.deepcopy(block_cache)
            before = _cache_lengths(prefix, block_cache)
            probe_tensor = torch.tensor([[int(token)]], dtype=torch.long)
            probe_output = _cached_last(
                prefix, probe_tensor, probe_cache, len(context.token_ids), counters
            )[0]
            after_original = _cache_lengths(prefix, block_cache)
            expected = endpoint[context_index][int(token)]
            endpoint_checks.append(
                {
                    "token_id": int(token),
                    "cached_probe_vs_full_endpoint": _compare(
                        probe_output, expected, "cached endpoint"
                    ),
                    "persistent_cache_unchanged": bool(before == after_original),
                    "persistent_before": before,
                    "persistent_after": after_original,
                    "probe_after": _cache_lengths(prefix, probe_cache),
                }
            )
            if before != after_original:
                raise DiagnosticError("reference/candidate probe mutated the persistent cache")

        ref_cache = copy.deepcopy(block_cache)
        ref_before = _cache_lengths(prefix, block_cache)
        ref_tensor = torch.tensor([[int(reference_id)]], dtype=torch.long)
        ref_output = _cached_last(
            prefix, ref_tensor, ref_cache, len(context.token_ids), counters
        )[0]
        ref_after = _cache_lengths(prefix, block_cache)
        reference_outputs.append(ref_output)
        reference_rows.append(
            {
                "context_index": context_index,
                "context_name": context.name,
                "reference_token_id": int(reference_id),
                "reference_probe_vs_full": _compare(
                    ref_output, endpoint[context_index].get(int(reference_id), ref_output),
                    "cached reference probe",
                ),
                "persistent_cache_unchanged": bool(ref_before == ref_after),
                "persistent_before": ref_before,
                "persistent_after": ref_after,
                "probe_after": _cache_lengths(prefix, ref_cache),
                "endpoint_position": len(context.token_ids),
            }
        )
        if ref_before != ref_after:
            raise DiagnosticError("reference probe mutated the persistent cache")
        cache_rows.append(
            {
                "context_index": context_index,
                "context_name": context.name,
                "context_token_ids": list(context.token_ids),
                "endpoint_position": len(context.token_ids),
                "context_sequence_length_with_endpoint": len(context.token_ids) + 1,
                "block": block_check,
                "block_cache_after": block_lengths,
                "incremental": incremental_check,
                "incremental_cache_after": incremental_lengths,
                "endpoint_checks": endpoint_checks,
            }
        )
    reference_tensor = torch.stack(reference_outputs).contiguous()
    _finite(reference_tensor, "reference outputs")
    return cache_rows, reference_tensor, reference_rows


def _top_k_neighbors(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    query_token_ids: Sequence[int],
    k: int = 8,
    prototype_chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find each query's fixed N=8 nearest *other* public prototypes."""

    queries = queries.detach().cpu().float()
    prototypes = prototypes.detach().cpu().float()
    if queries.ndim != 2 or prototypes.ndim != 2 or queries.shape[1] != prototypes.shape[1]:
        raise DiagnosticError("neighbor geometry changed")
    rows = int(queries.shape[0])
    vocab = int(prototypes.shape[0])
    query_ids = torch.tensor([int(value) for value in query_token_ids], dtype=torch.long)
    if query_ids.numel() != rows or query_ids.lt(0).any().item() or query_ids.ge(vocab).any().item():
        raise DiagnosticError("neighbor query IDs changed")
    if k <= 0 or k >= vocab:
        raise DiagnosticError("neighbor k must leave room for the explicit true token")
    q = F.normalize(queries, dim=1, eps=1e-12)
    best_scores = torch.full((rows, k), -float("inf"), dtype=torch.float32)
    best_ids = torch.full((rows, k), vocab, dtype=torch.long)
    for start in range(0, vocab, prototype_chunk_size):
        stop = min(start + prototype_chunk_size, vocab)
        p = F.normalize(prototypes[start:stop], dim=1, eps=1e-12)
        block = q @ p.transpose(0, 1)
        ids = torch.arange(start, stop, dtype=torch.long).view(1, -1).expand(rows, -1)
        # Remove each query's own prototype before deterministic top-k merge.
        block = block.masked_fill(ids.eq(query_ids[:, None]), -float("inf"))
        merged_scores = torch.cat((best_scores, block), dim=1)
        merged_ids = torch.cat((best_ids, ids), dim=1)
        by_id = torch.argsort(merged_ids, dim=1, stable=True)
        merged_ids = merged_ids.gather(1, by_id)
        merged_scores = merged_scores.gather(1, by_id)
        by_score = torch.argsort(merged_scores, dim=1, descending=True, stable=True)[:, :k]
        best_scores = merged_scores.gather(1, by_score)
        best_ids = merged_ids.gather(1, by_score)
    if best_ids.ge(vocab).any().item() or not torch.isfinite(best_scores).all().item():
        raise DiagnosticError("neighbor scan did not produce finite full-vocabulary results")
    if best_ids.eq(query_ids[:, None]).any().item():
        raise DiagnosticError("neighbor scan retained a query's own prototype")
    return best_ids, best_scores


def _restricted_rank(
    queries: torch.Tensor,
    true_ids: Sequence[int],
    neighbor_ids: torch.Tensor,
    prototypes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    queries = queries.detach().cpu().float()
    prototypes = prototypes.detach().cpu().float()
    neighbors = neighbor_ids.detach().cpu().long()
    labels = torch.tensor([int(value) for value in true_ids], dtype=torch.long)
    if queries.ndim != 2 or neighbors.ndim != 2 or queries.shape[0] != neighbors.shape[0]:
        raise DiagnosticError("restricted ranking geometry changed")
    if labels.numel() != queries.shape[0] or neighbors.shape[1] < 2:
        raise DiagnosticError("restricted ranking labels or dictionary changed")
    if labels.lt(0).any().item() or labels.ge(prototypes.shape[0]).any().item():
        raise DiagnosticError("restricted ranking labels are outside the public dictionary")
    if neighbors.lt(0).any().item() or neighbors.ge(prototypes.shape[0]).any().item():
        raise DiagnosticError("restricted ranking dictionary IDs are outside the public vocabulary")
    sorted_neighbors_for_check = torch.sort(neighbors, dim=1).values
    if sorted_neighbors_for_check[:, 1:].eq(sorted_neighbors_for_check[:, :-1]).any().item():
        raise DiagnosticError("restricted ranking dictionary contains duplicate IDs")
    # [rows, hidden] @ [rows, hidden, k] -> [rows,k].
    scores = torch.bmm(
        F.normalize(queries, dim=1, eps=1e-12).unsqueeze(1),
        F.normalize(prototypes[neighbors.reshape(-1)], dim=1, eps=1e-12).reshape(
            neighbors.shape[0], neighbors.shape[1], -1
        ).transpose(1, 2),
    ).squeeze(1)
    in_neighbors = neighbors.eq(labels[:, None])
    if not in_neighbors.any(dim=1).all().item():
        raise DiagnosticError("true token missing from declared local dictionary")
    true_index = in_neighbors.to(torch.long).argmax(dim=1)
    true_score = scores.gather(1, true_index[:, None]).squeeze(1)
    by_id = torch.argsort(neighbors, dim=1, stable=True)
    sorted_ids = neighbors.gather(1, by_id)
    sorted_scores = scores.gather(1, by_id)
    by_score = torch.argsort(sorted_scores, dim=1, descending=True, stable=True)
    sorted_scores = sorted_scores.gather(1, by_score)
    sorted_ids = sorted_ids.gather(1, by_score)
    top1 = sorted_ids[:, 0]
    top1_score = sorted_scores[:, 0]
    runner = sorted_scores[:, 1]
    other = sorted_scores.masked_fill(sorted_ids.eq(labels[:, None]), -float("inf"))
    best_other = other.max(dim=1).values
    return {
        "top1_ids": top1,
        "runner_up_ids": sorted_ids[:, 1],
        "top1_scores": top1_score,
        "runner_up_scores": runner,
        "top1_runner_margin": top1_score - runner,
        "true_scores": true_score,
        "best_other_scores": best_other,
        "true_other_margin": true_score - best_other,
        # Match full-vocabulary/A1 semantics: strict greater-than rank, while
        # stable ID ordering is used only to choose top-1 and runner-up IDs.
        "true_rank": (scores > true_score[:, None]).sum(dim=1) + 1,
        "true_equal_count": scores.eq(true_score[:, None]).sum(dim=1),
        "top1_is_true": top1.eq(labels),
        "true_ids": labels,
    }


def _rank_summary(
    result: Mapping[str, torch.Tensor],
    *,
    row_context_indices: Sequence[int],
    row_token_ids: Sequence[int],
    variant: str,
    dictionary: str,
    score_scale: float = 1.0,
    native_score_units: str = "cosine",
) -> dict[str, Any]:
    top1 = result["top1_ids"].detach().cpu().long()
    runner = result["runner_up_ids"].detach().cpu().long()
    top1_score = result["top1_scores"].detach().cpu().float()
    runner_score = result["runner_up_scores"].detach().cpu().float()
    top1_margin = result["top1_runner_margin"].detach().cpu().float()
    true_score = result["true_scores"].detach().cpu().float()
    best_other = result["best_other_scores"].detach().cpu().float()
    true_margin = result["true_other_margin"].detach().cpu().float()
    true_rank = result["true_rank"].detach().cpu().long()
    true_ids = result["true_ids"].detach().cpu().long()
    top1_is_true = top1.eq(true_ids)
    if not math.isfinite(float(score_scale)) or float(score_scale) <= 0.0:
        raise DiagnosticError("ranking score scale must be finite and positive")
    score_scale = float(score_scale)
    reported_top1_score = top1_score / score_scale
    reported_runner_score = runner_score / score_scale
    reported_top1_margin = top1_margin / score_scale
    reported_true_score = true_score / score_scale
    reported_best_other = best_other / score_scale
    reported_true_margin = true_margin / score_scale
    if len(row_context_indices) != int(top1.numel()) or len(row_token_ids) != int(top1.numel()):
        raise DiagnosticError("ranking row metadata changed")
    rows: list[dict[str, Any]] = []
    for index in range(int(top1.numel())):
        rows.append(
            {
                "row_index": index,
                "context_index": int(row_context_indices[index]),
                "token_id": int(row_token_ids[index]),
                "top1_id": int(top1[index]),
                "runner_up_id": int(runner[index]),
                "true_rank": int(true_rank[index]),
                "top1_is_true": bool(top1_is_true[index]),
                "top1_score": float(reported_top1_score[index]),
                "runner_up_score": float(reported_runner_score[index]),
                "top1_runner_margin": float(reported_top1_margin[index]),
                "true_score": float(reported_true_score[index]),
                "best_other_score": float(reported_best_other[index]),
                "true_other_margin": float(reported_true_margin[index]),
                "native_top1_score": float(top1_score[index]),
                "native_runner_up_score": float(runner_score[index]),
                "native_top1_runner_margin": float(top1_margin[index]),
                "native_true_score": float(true_score[index]),
                "native_best_other_score": float(best_other[index]),
                "native_true_other_margin": float(true_margin[index]),
            }
        )
    unique, counts = torch.unique(true_rank, return_counts=True)
    return {
        "variant": variant,
        "dictionary": dictionary,
        "rank_definition": STRICT_RANK_DEFINITION,
        "reported_score_units": "cosine-equivalent",
        "native_score_units": native_score_units,
        "native_score_scale": score_scale,
        "rows": rows,
        "summary": {
            "count": int(top1.numel()),
            "top1_correct_count": int(top1_is_true.sum().item()),
            "top1_correct_rate": float(top1_is_true.float().mean().item()),
            "true_rank": _stats(true_rank.float()),
            "true_rank_histogram": {str(int(k)): int(v) for k, v in zip(unique, counts, strict=True)},
            "top1_runner_margin": _stats(reported_top1_margin),
            "true_other_margin": _stats(reported_true_margin),
            "true_score": _stats(reported_true_score),
            "native_top1_runner_margin": _stats(top1_margin),
            "native_true_other_margin": _stats(true_margin),
            "native_true_score": _stats(true_score),
        },
    }


def _rank_from_logits(logits: torch.Tensor, true_ids: Sequence[int]) -> dict[str, torch.Tensor]:
    logits = logits.detach().cpu().float()
    labels = torch.tensor([int(value) for value in true_ids], dtype=torch.long)
    if logits.ndim != 2 or labels.numel() != logits.shape[0] or logits.shape[1] != VOCAB_SIZE:
        raise DiagnosticError("A1 logits geometry changed")
    vocab = int(logits.shape[1])
    ids = torch.arange(vocab, dtype=torch.long).view(1, -1).expand(logits.shape[0], -1)
    by_id = torch.argsort(ids, dim=1, stable=True)
    scores = logits.gather(1, by_id)
    sorted_ids = ids.gather(1, by_id)
    by_score = torch.argsort(scores, dim=1, descending=True, stable=True)
    scores = scores.gather(1, by_score)
    sorted_ids = sorted_ids.gather(1, by_score)
    true_score = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.masked_fill(ids.eq(labels[:, None]), -float("inf"))
    best_other = other.max(dim=1).values
    true_rank = (logits > true_score[:, None]).sum(dim=1) + 1
    return {
        "top1_ids": sorted_ids[:, 0],
        "runner_up_ids": sorted_ids[:, 1],
        "top1_scores": scores[:, 0],
        "runner_up_scores": scores[:, 1],
        "top1_runner_margin": scores[:, 0] - scores[:, 1],
        "true_scores": true_score,
        "best_other_scores": best_other,
        "true_other_margin": true_score - best_other,
        "true_rank": true_rank,
        "true_equal_count": (logits == true_score[:, None]).sum(dim=1),
        "top1_is_true": sorted_ids[:, 0].eq(labels),
        "true_ids": labels,
    }


def _tensor_json(value: torch.Tensor) -> list[Any]:
    value = value.detach().cpu()
    if value.dtype == torch.bool:
        return [bool(item) for item in value.reshape(-1).tolist()]
    if value.dtype.is_floating_point:
        return [float(item) for item in value.reshape(-1).tolist()]
    return [int(item) for item in value.reshape(-1).tolist()]


def _build_projected_prototypes(
    table: PrototypeTable, lens: torch.nn.Module, *, chunk_size: int = 8192
) -> torch.Tensor:
    """Apply the frozen affine lens once to the shared table in streamed chunks."""

    output = torch.empty_like(table.prototypes, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, table.vocab_size, chunk_size):
            stop = min(start + chunk_size, table.vocab_size)
            block = lens.projected(table.prototypes[start:stop].float())
            _finite(block, "projected prototype block")
            output[start:stop] = block.detach().cpu().float()
    _finite(output, "projected prototype table")
    return output.contiguous()


def _make_figures(
    output_root: Path,
    *,
    context_names: Sequence[str],
    offset_rows: Sequence[Mapping[str, Any]],
    raw_separation: Mapping[str, Any],
    projected_separation: Mapping[str, Any],
    targeted_summaries: Mapping[str, Mapping[str, Any]],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    offset_path = output_root / "offset_geometry.png"
    lens_path = output_root / "lens_geometry.png"
    for path in (offset_path, lens_path):
        if path.exists() or path.is_symlink():
            raise DiagnosticError(f"figure output already exists: {path}")

    labels = [name.replace("_", "\n") for name in context_names]
    ratio = [float(row["residual_to_delta_ratio"]["median"]) for row in offset_rows]
    deformation = [
        float(row["relative_deformation_median"])
        for row in offset_rows
    ]
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    positions = torch.arange(len(labels)).numpy()
    width = 0.38
    axis.bar(positions - width / 2, ratio, width, label="offset residual / offset norm")
    axis.bar(positions + width / 2, deformation, width, label="pair deformation / baseline pair")
    axis.set_xticks(positions, labels, fontsize=8)
    axis.set_ylabel("median relative magnitude")
    axis.set_title("Public teacher-prefix context deformation")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(offset_path, dpi=150)
    plt.close(figure)

    names = ["raw hidden", "frozen lens projected"]
    same = [
        float(raw_separation["same_token_cross_context"]["l2"]["mean"]),
        float(projected_separation["same_token_cross_context"]["l2"]["mean"]),
    ]
    different = [
        float(raw_separation["different_token_within_context"]["l2"]["mean"]),
        float(projected_separation["different_token_within_context"]["l2"]["mean"]),
    ]
    rank_names = ["raw_boundary", "projected_prototype", "historical_A1"]
    rates = [
        float(targeted_summaries[name]["summary"]["top1_correct_rate"])
        for name in rank_names
    ]
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
    axes[0].bar(names, same, label="same token, cross context")
    axes[0].bar(names, different, bottom=same, label="different token, within context")
    axes[0].set_ylabel("mean L2 distance")
    axes[0].set_title("Lens geometry spread (C1-C4 equal position)")
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[0].legend(fontsize=7)
    axes[1].bar(rank_names, rates)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("top-1 rate on 12 targeted rows")
    axes[1].set_title("Same-row full-vocabulary ranking")
    axes[1].tick_params(axis="x", labelrotation=25)
    figure.tight_layout()
    figure.savefig(lens_path, dpi=150)
    plt.close(figure)
    return [offset_path, lens_path]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--historical-lens", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--prototype-chunk-size", type=int, default=8192)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prototype_chunk_size != 8192:
        raise DiagnosticError("only the predeclared prototype chunk size is accepted")
    plan_path = _require_regular(args.plan.resolve(), "P02 plan")
    plan = _read_json(plan_path)
    candidate_ids, reference_id, contexts = _validate_plan(plan)
    prototype_path = _require_regular(args.prototype.resolve(), "P01 prototype table")
    lens_path = _require_regular(args.historical_lens.resolve(), "published frozen lens")
    table_hash = _sha256_file(prototype_path)
    lens_hash = _sha256_file(lens_path)
    if table_hash != TABLE_SHA256:
        raise DiagnosticError(f"P01 prototype table hash changed: {table_hash}")
    if lens_hash != LENS_SHA256:
        raise DiagnosticError(f"published lens hash changed: {lens_hash}")
    if args.model_path is not None and args.model_path.is_symlink():
        raise DiagnosticError("public model path must not be a symlink")

    output_root = _create_only_directory(args.output_root.resolve())
    manifest_path = args.manifest_path.resolve()
    if manifest_path.exists() or manifest_path.is_symlink():
        raise DiagnosticError(f"manifest output already exists: {manifest_path}")
    started_utc = _utc_now()
    started = time.perf_counter()
    preflight_started = time.perf_counter()
    guard = _resource_guard(GUARD_REQUIRED_BYTES)
    preflight = {
        "schema": "token-reconstruction.trr-p02-preflight.v1",
        "task_id": TASK_ID,
        "truth_opened": False,
        "source_truth_included": False,
        "status": "PREPARED_BEFORE_MODEL_LOAD",
        "started_utc": started_utc,
        "implementation_commit": args.implementation_commit,
        "selected_device": "cpu",
        "seed": SEED,
        "threads": {"torch": TORCH_THREADS, "interop": TORCH_INTEROP_THREADS},
        "ranking_score_buffer_max_bytes": RANKING_SCORE_BUFFER_BYTES,
        "estimate": {
            "model_bytes_estimate": 2_500_000_000,
            "prototype_table_bytes": 525_336_576,
            "normalized_embedding_bytes": VOCAB_SIZE * HIDDEN_SIZE * 4,
            "projected_prototype_bytes": VOCAB_SIZE * HIDDEN_SIZE * 4,
            "expected_peak_rss_bytes": EXPECTED_PEAK_RSS_BYTES,
            "ranking_score_buffer_max_bytes": RANKING_SCORE_BUFFER_BYTES,
            "guard_required_bytes": GUARD_REQUIRED_BYTES,
            "safety_fraction": 0.8,
        },
        "guard": guard,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "cut_depth": CUT_DEPTH,
            "hidden_size": HIDDEN_SIZE,
            "vocab_size": VOCAB_SIZE,
            "model_path": str(args.model_path.resolve()) if args.model_path else None,
        },
        "prototype": _file_record(prototype_path),
        "lens": _file_record(lens_path),
        "plan": _file_record(plan_path),
    }
    _write_json_exclusive(output_root / "preflight.json", preflight)
    preflight_seconds = time.perf_counter() - preflight_started

    torch.set_num_threads(TORCH_THREADS)
    torch.set_num_interop_threads(TORCH_INTEROP_THREADS)
    torch.use_deterministic_algorithms(True)
    torch.random.manual_seed(SEED)
    device = torch.device("cpu")
    counters = Counters()
    phase_timing: dict[str, float] = {}
    rss_checks: list[dict[str, Any]] = []

    phase_started = time.perf_counter()
    table = PrototypeTable.load(
        prototype_path,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
        expected_cut_depth=CUT_DEPTH,
        expected_vocab_size=VOCAB_SIZE,
        expected_hidden_size=HIDDEN_SIZE,
    )
    if table.prototypes.dtype != torch.bfloat16:
        raise DiagnosticError("P01 prototype table dtype changed")
    model = load_public_model(device=device, model_path=args.model_path)
    prefix = ContiguousPublicPrefix(model, CUT_DEPTH).to(device).eval()
    if int(prefix.embed_tokens.num_embeddings) != VOCAB_SIZE:
        raise DiagnosticError("public prefix vocabulary changed")
    if int(prefix.embed_tokens.embedding_dim) != HIDDEN_SIZE:
        raise DiagnosticError("public prefix hidden size changed")
    lens = load_published_frozen_lens(lens_path, device=device)
    for parameter in lens.parameters():
        if parameter.requires_grad:
            raise DiagnosticError("published lens unexpectedly requires gradients")
    lens_s = getattr(lens, "s", None)
    if not isinstance(lens_s, torch.Tensor) or lens_s.numel() != 1:
        raise DiagnosticError("published lens scale parameter changed")
    lens_s = lens_s.detach().float().reshape(())
    _finite(lens_s, "frozen lens scale")
    lens_logit_scale = float(lens_s.exp().item())
    if not math.isfinite(lens_logit_scale) or lens_logit_scale <= 0.0:
        raise DiagnosticError("published lens native score scale is invalid")
    normalized_embeddings = F.normalize(prefix.embed_tokens.weight.detach().float(), dim=-1)
    _finite(normalized_embeddings, "normalized public embeddings")
    qualification_started = time.perf_counter()
    qualification = _qualify_short_cell(
        prefix, contexts[QUALIFICATION_CONTEXT_INDEX], candidate_ids, counters
    )
    phase_timing["largest_short_cell_qualification_seconds"] = time.perf_counter() - qualification_started
    _rss_ceiling_check("after_model_load_and_longest_context_qualification", rss_checks)
    qualification_path = output_root / "qualification.json"
    _write_json_exclusive(
        qualification_path,
        {
            "schema": "token-reconstruction.trr-p02-qualification.v1",
            "task_id": TASK_ID,
            "truth_opened": False,
            "source_truth_included": False,
            "implementation_commit": args.implementation_commit,
            **qualification,
        },
    )

    config_commit = getattr(getattr(model, "config", None), "_commit_hash", None)
    if config_commit is not None and str(config_commit) != MODEL_REVISION:
        raise DiagnosticError(
            "public model config revision differs from the declared snapshot: "
            f"{config_commit!r} != {MODEL_REVISION!r}"
        )
    model_identity = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "config_commit": config_commit,
        "cut_depth": CUT_DEPTH,
        "hidden_size": HIDDEN_SIZE,
        "vocab_size": VOCAB_SIZE,
        "dtype": str(prefix.embed_tokens.weight.dtype).replace("torch.", ""),
    }
    # The full-model shell owns layers after the cut.  Release it after the
    # prefix has retained its first four layers, reducing the measured CPU RSS
    # while leaving the public computation unchanged.
    del model
    gc.collect()
    phase_timing["load_table_model_lens_seconds"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    primary_panel, endpoint, context_last = _collect_panel(
        prefix, contexts, candidate_ids, counters
    )
    cache_rows, reference_outputs, reference_rows = _cache_and_reference_checks(
        prefix, contexts, endpoint, reference_id, counters
    )
    phase_timing["public_panel_and_cache_checks_seconds"] = time.perf_counter() - phase_started
    _rss_ceiling_check("after_public_panel_and_cache_checks", rss_checks)

    baseline = primary_panel[0]
    baseline_ids_tensor = torch.tensor(candidate_ids, dtype=torch.long)
    table_rows = table.prototypes[baseline_ids_tensor].detach().cpu()
    baseline_table_difference = (baseline.float() - table_rows.float()).abs()
    baseline_table_check = {
        "candidate_ids": candidate_ids,
        "recomputed_dtype": str(baseline.dtype).replace("torch.", ""),
        "table_dtype": str(table_rows.dtype).replace("torch.", ""),
        "torch_equal_after_dtype_match": bool(torch.equal(baseline, table_rows)),
        "maximum_absolute_difference": float(baseline_table_difference.max().item()),
        "mean_absolute_difference": float(baseline_table_difference.mean().item()),
        "recomputed_digest": _digest_tensor(baseline),
        "reused_table_rows_digest": _digest_tensor(table_rows),
        "interpretation": "P02 recomputes C0 rows under eight intra-op threads; equality is measured rather than assumed",
    }
    if not torch.isfinite(baseline_table_difference).all().item():
        raise DiagnosticError("recomputed baseline/table comparison is non-finite")

    primary_offsets = primary_panel.float() - baseline.float()[None, :, :]
    offset_result = summarize_offsets(primary_offsets)
    pair_result = pairwise_token_deformation(primary_panel, token_ids=candidate_ids)
    repeated_values = torch.stack(
        [
            torch.stack([endpoint[index][token] for token in REPEATED_ENDPOINT_IDS])
            for index in REPEATED_CONTEXT_INDICES
        ]
    )
    repeated_result = separation_summary(repeated_values)
    repeated_position_rows: list[dict[str, Any]] = []
    for repeated_row, context_index in enumerate(REPEATED_CONTEXT_INDICES):
        for token_row, token in enumerate(REPEATED_ENDPOINT_IDS):
            repeated_position_rows.append(
                {
                    "context_index": context_index,
                    "context_name": contexts[context_index].name,
                    "token_id": int(token),
                    "endpoint_position": len(contexts[context_index].token_ids),
                    "activation_digest": _digest_tensor(repeated_values[repeated_row, token_row]),
                }
            )

    reference_prototype = table.prototypes[int(reference_id)].float()
    reference_offsets = reference_outputs.float() - reference_prototype[None, :]
    primary_reference_offsets = reference_offsets[list(PRIMARY_CONTEXT_INDICES)]
    minus_queries = torch.empty_like(primary_panel.float())
    plus_queries = torch.empty_like(primary_panel.float())
    for row, context_index in enumerate(PRIMARY_CONTEXT_INDICES):
        offset = primary_reference_offsets[row]
        minus_queries[row] = reference_corrected_query(
            primary_panel[row],
            reference_outputs[context_index].view(1, -1).expand(len(candidate_ids), -1),
            reference_prototype.view(1, -1).expand(len(candidate_ids), -1),
            sign=-1,
        )
        plus_queries[row] = reference_corrected_query(
            primary_panel[row],
            reference_outputs[context_index].view(1, -1).expand(len(candidate_ids), -1),
            reference_prototype.view(1, -1).expand(len(candidate_ids), -1),
            sign=1,
        )
        if not torch.equal(offset, primary_reference_offsets[row]):
            raise DiagnosticError("reference offset assembly changed")
    centered_queries = primary_panel.float() - offset_result["context_means"].float()[:, None, :]
    self_consistency_rows: list[dict[str, Any]] = []
    reference_index = candidate_ids.index(reference_id)
    for row, context_index in enumerate(PRIMARY_CONTEXT_INDICES):
        minus_error = (minus_queries[row, reference_index] - baseline[reference_index].float()).abs()
        plus_error = (plus_queries[row, reference_index] - baseline[reference_index].float()).abs()
        self_consistency_rows.append(
            {
                "context_index": context_index,
                "context_name": contexts[context_index].name,
                "reference_token_id": reference_id,
                "minus_query_vs_recomputed_b_r_max_abs": float(minus_error.max().item()),
                "plus_query_vs_recomputed_b_r_max_abs": float(plus_error.max().item()),
                "minus_query_digest": _digest_tensor(minus_queries[row, reference_index]),
                "plus_query_digest": _digest_tensor(plus_queries[row, reference_index]),
            }
        )
    sign_check = {
        "reference_id": reference_id,
        "subtraction_rule": "q_minus = h - (z(C,r)-b_r)",
        "opposite_sign_control": "q_plus = h + (z(C,r)-b_r)",
        "self_consistency": self_consistency_rows,
        "reference_offset_norm": _stats(torch.linalg.vector_norm(primary_reference_offsets, dim=-1)),
    }

    phase_started = time.perf_counter()
    local_neighbor_ids, local_neighbor_scores = _top_k_neighbors(
        table.prototypes[baseline_ids_tensor],
        table.prototypes,
        query_token_ids=candidate_ids,
        k=8,
        prototype_chunk_size=args.prototype_chunk_size,
    )
    # The N=8 nearest-other scan is retained separately; each local ranking
    # dictionary appends its known query token, yielding exactly nine IDs.
    local_dictionary_ids = torch.cat(
        (local_neighbor_ids, baseline_ids_tensor[:, None]), dim=1
    )
    if local_dictionary_ids.shape != (len(candidate_ids), 9):
        raise DiagnosticError("local N=8 plus true dictionary cardinality changed")
    local_dictionary_rows = [
        {
            "token_id": int(token),
            "neighbor_ids": [int(value) for value in local_neighbor_ids[row].tolist()],
            "neighbor_cosine_scores": [float(value) for value in local_neighbor_scores[row].tolist()],
            "dictionary_ids": [int(value) for value in local_dictionary_ids[row].tolist()],
            "dictionary_size": int(local_dictionary_ids.shape[1]),
        }
        for row, token in enumerate(candidate_ids)
    ]
    row_contexts = [int(context_index) for context_index in PRIMARY_CONTEXT_INDICES for _ in candidate_ids]
    row_tokens = list(candidate_ids) * len(PRIMARY_CONTEXT_INDICES)
    local_variants: list[dict[str, Any]] = []
    primary_flat = primary_panel.float().reshape(-1, HIDDEN_SIZE)
    centered_flat = centered_queries.reshape(-1, HIDDEN_SIZE)
    minus_flat = minus_queries.reshape(-1, HIDDEN_SIZE)
    plus_flat = plus_queries.reshape(-1, HIDDEN_SIZE)
    for variant, queries in (
        ("raw_boundary", primary_flat),
        ("oracle_context_mean_centered", centered_flat),
        ("reference_subtraction_control", minus_flat),
        ("reference_opposite_sign_control", plus_flat),
    ):
        local_result = _restricted_rank(
            queries,
            row_tokens,
            local_dictionary_ids.repeat(len(PRIMARY_CONTEXT_INDICES), 1),
            table.prototypes,
        )
        local_variants.append(
            _rank_summary(
                local_result,
                row_context_indices=row_contexts,
                row_token_ids=row_tokens,
                variant=variant,
                dictionary="fixed N=8 nearest OTHER reused P01 BOS prototypes plus the known true ID (9 candidates)",
            )
        )

    targeted_contexts: list[int] = []
    targeted_tokens: list[int] = []
    targeted_queries: list[torch.Tensor] = []
    for context_index in TARGETED_CONTEXT_INDICES:
        row = PRIMARY_CONTEXT_INDICES.index(context_index)
        for token in TARGETED_ENDPOINT_IDS:
            targeted_contexts.append(context_index)
            targeted_tokens.append(int(token))
            targeted_queries.append(primary_panel[row, candidate_ids.index(int(token))].float())
    targeted_tensor = torch.stack(targeted_queries)
    targeted_phase_rows = [
        {
            "row_index": index,
            "context_index": targeted_contexts[index],
            "context_name": contexts[targeted_contexts[index]].name,
            "token_id": targeted_tokens[index],
            "endpoint_position": len(contexts[targeted_contexts[index]].token_ids),
        }
        for index in range(len(targeted_tokens))
    ]
    targeted_raw = rank_metrics(
        targeted_tensor,
        table.prototypes,
        targeted_tokens,
        metric="cosine",
        query_chunk_size=16,
        prototype_chunk_size=args.prototype_chunk_size,
    )
    targeted_centered_queries = torch.stack(
        [
            centered_queries[PRIMARY_CONTEXT_INDICES.index(context_index), candidate_ids.index(token)]
            for context_index, token in zip(targeted_contexts, targeted_tokens, strict=True)
        ]
    )
    targeted_minus_queries = torch.stack(
        [
            minus_queries[PRIMARY_CONTEXT_INDICES.index(context_index), candidate_ids.index(token)]
            for context_index, token in zip(targeted_contexts, targeted_tokens, strict=True)
        ]
    )
    targeted_centered = rank_metrics(
        targeted_centered_queries,
        table.prototypes,
        targeted_tokens,
        metric="cosine",
        query_chunk_size=16,
        prototype_chunk_size=args.prototype_chunk_size,
    )
    targeted_minus = rank_metrics(
        targeted_minus_queries,
        table.prototypes,
        targeted_tokens,
        metric="cosine",
        query_chunk_size=16,
        prototype_chunk_size=args.prototype_chunk_size,
    )
    phase_timing["local_and_targeted_raw_rankings_seconds"] = time.perf_counter() - phase_started
    _rss_ceiling_check("after_local_and_targeted_rankings", rss_checks)

    phase_started = time.perf_counter()
    projected_prototypes = _build_projected_prototypes(
        table, lens, chunk_size=args.prototype_chunk_size
    )
    projected_primary = lens.projected(primary_flat).detach().cpu().float().reshape(primary_panel.shape)
    _finite(projected_primary, "projected primary panel")
    # Keep the C0-relative panel for explicit baseline displacement, and derive
    # a separate C1-C4 summary for equal-endpoint-position content effects.
    raw_separation = separation_summary(primary_panel)
    projected_separation = separation_summary(projected_primary)
    raw_equal_position_separation = separation_summary(primary_panel[1:])
    projected_equal_position_separation = separation_summary(projected_primary[1:])
    targeted_projected_queries = lens.projected(targeted_tensor).detach().cpu().float()
    targeted_projected = rank_metrics(
        targeted_projected_queries,
        projected_prototypes,
        targeted_tokens,
        metric="cosine",
        query_chunk_size=16,
        prototype_chunk_size=args.prototype_chunk_size,
    )
    with torch.inference_mode():
        targeted_a1_logits = lens(targeted_tensor, normalized_embeddings).detach().cpu().float()
    _finite(targeted_a1_logits, "historical A1 logits")
    targeted_a1 = _rank_from_logits(targeted_a1_logits, targeted_tokens)
    phase_timing["projected_table_and_lens_rankings_seconds"] = time.perf_counter() - phase_started
    _rss_ceiling_check("after_projected_table_and_lens_rankings", rss_checks)

    targeted_summaries = {
        "raw_boundary": _rank_summary(
            targeted_raw,
            row_context_indices=targeted_contexts,
            row_token_ids=targeted_tokens,
            variant="raw_boundary",
            dictionary="reused P01 boundary prototypes",
        ),
        "oracle_mean_centered": _rank_summary(
            targeted_centered,
            row_context_indices=targeted_contexts,
            row_token_ids=targeted_tokens,
            variant="oracle_mean_centered",
            dictionary="reused P01 boundary prototypes",
        ),
        "reference_subtraction": _rank_summary(
            targeted_minus,
            row_context_indices=targeted_contexts,
            row_token_ids=targeted_tokens,
            variant="reference_subtraction",
            dictionary="reused P01 boundary prototypes",
        ),
        "projected_prototype": _rank_summary(
            targeted_projected,
            row_context_indices=targeted_contexts,
            row_token_ids=targeted_tokens,
            variant="projected_prototype",
            dictionary="one streamed frozen-lens projection of reused P01 prototypes",
        ),
        "historical_A1": _rank_summary(
            targeted_a1,
            row_context_indices=targeted_contexts,
            row_token_ids=targeted_tokens,
            variant="historical_A1",
            dictionary="raw public input embeddings via original frozen lens.forward",
            score_scale=lens_logit_scale,
            native_score_units="FrozenAffineLens.forward cosine * exp(s)",
        ),
    }
    phase_started = time.perf_counter()
    figures = _make_figures(
        output_root,
        context_names=[context.name for context in contexts],
        offset_rows=[
            {
                **row,
                "relative_deformation_median": float(
                    torch.tensor(
                        [
                            pair["relative_deformation"]
                            for pair in pair_result["pairs"]
                            if int(pair["context_index"]) == index
                        ],
                        dtype=torch.float32,
                    ).median().item()
                ),
            }
            for index, row in enumerate(offset_result["context_rows"])
        ],
        raw_separation=raw_equal_position_separation,
        projected_separation=projected_equal_position_separation,
        targeted_summaries={
            key: value for key, value in targeted_summaries.items() if key in {"raw_boundary", "projected_prototype", "historical_A1"}
        },
    )
    phase_timing["figures_seconds"] = time.perf_counter() - phase_started

    # The JSON intentionally contains summaries and row-level metrics, while
    # the activation artifact retains the exact small panel and correction
    # arrays for later review without exposing any private truth.
    serialization_started = time.perf_counter()
    tensor_path = output_root / "activation_panel.safetensors"
    if tensor_path.exists() or tensor_path.is_symlink():
        raise DiagnosticError(f"tensor artifact already exists: {tensor_path}")
    save_file(
        {
            "primary_activations": primary_panel.contiguous(),
            "repeated_endpoint_activations": repeated_values.contiguous(),
            "reference_outputs": reference_outputs.contiguous(),
            "primary_offsets": primary_offsets.contiguous(),
            "primary_context_means": offset_result["context_means"].contiguous(),
            "primary_offset_residuals": offset_result["residuals"].contiguous(),
            "reference_offsets": reference_offsets.contiguous(),
            "oracle_centered_queries": centered_queries.contiguous(),
            "reference_minus_queries": minus_queries.contiguous(),
            "reference_plus_queries": plus_queries.contiguous(),
            "local_neighbor_ids": local_neighbor_ids.to(torch.int32).contiguous(),
            "local_neighbor_scores": local_neighbor_scores.contiguous(),
            "recomputed_baseline": baseline.contiguous(),
            "reused_table_rows": table_rows.contiguous(),
        },
        tensor_path,
        metadata={
            "schema": "token-reconstruction.trr-p02-activation-panel.v1",
            "task_id": TASK_ID,
            "truth_opened": "false",
            "source_truth_included": "false",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "cut_depth": str(CUT_DEPTH),
            "candidate_ids": ",".join(str(value) for value in candidate_ids),
            "reference_id": str(reference_id),
            "panel_rows": "46",
        },
    )
    phase_timing["serialization_seconds"] = time.perf_counter() - serialization_started
    _rss_ceiling_check("after_activation_panel_serialization", rss_checks)
    diagnostics_path = output_root / "diagnostics.json"
    diagnostics = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_TEACHER_PREFIX_DIAGNOSTIC_COMPLETE",
        "truth_opened": False,
        "source_truth_included": False,
        "seed": SEED,
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "implementation_commit": args.implementation_commit,
        "model": model_identity,
        "assets": {
            "prototype": _file_record(prototype_path),
            "prototype_sha256_expected": TABLE_SHA256,
            "historical_lens": _file_record(lens_path),
            "historical_lens_sha256_expected": LENS_SHA256,
            "plan": _file_record(plan_path),
        },
        "qualification": {
            **qualification,
            "artifact": _file_record(qualification_path),
        },
        "teacher_prefix_control": {
            "label": "fully known public token IDs and contexts; not a reconstruction score",
            "candidate_ids": candidate_ids,
            "reference_id": reference_id,
            "contexts": [
                {
                    "context_index": index,
                    "name": context.name,
                    "token_ids": list(context.token_ids),
                    "endpoint_position": len(context.token_ids),
                    "endpoint_sequence_length": len(context.token_ids) + 1,
                    "role": "public teacher-prefix control",
                }
                for index, context in enumerate(contexts)
            ],
            "primary_panel_shape": list(primary_panel.shape),
            "total_unique_activation_rows": 46,
        },
        "wiring_and_position_checks": {
            "cache_rows": cache_rows,
            "reference_rows": reference_rows,
            "position_semantics": "C0 endpoint is position 1; C1-C4 endpoints are position 2 with same visible length; C5/C6 are explicit repeated-13 context-length controls at positions 3/4",
            "no_position_only_claim": True,
            "sign_check": sign_check,
            "baseline_table_recomputation": baseline_table_check,
        },
        "shared_offset": {
            "definition": "delta(C,v)=z(C,v)-b_v, b_v=z(C0,v)",
            "summary": {
                "geometry": offset_result["geometry"],
                "context_rows": offset_result["context_rows"],
                "global_residual_to_delta_ratio": offset_result["global_residual_to_delta_ratio"],
            },
            "mean_offset_status": "ORACLE_PUBLIC_PANEL_CENTERING_DIAGNOSTIC; each query participates in its context mean; never a deployable no-fit method",
            "pairwise_token_deformation": {
                "summary": {
                    "geometry": pair_result["geometry"],
                    "pair_count": pair_result["pair_count"],
                    "deformation_norm": pair_result["deformation_norm"],
                    "relative_deformation": pair_result["relative_deformation"],
                },
                "rows": pair_result["pairs"],
            },
            "local_n8_neighbors": {
                "dictionary": "fixed N=8 neighbors derived from reused P01 BOS table rows",
                "rows": local_dictionary_rows,
                "rankings": local_variants,
            },
            "repeated_13_context_length": {
                "endpoint_ids": list(REPEATED_ENDPOINT_IDS),
                "contexts": [int(value) for value in REPEATED_CONTEXT_INDICES],
                "separation": repeated_result,
                "rows": repeated_position_rows,
            },
            "targeted_full_vocab": {
                "row_declaration": targeted_phase_rows,
                "variants": targeted_summaries,
                "limit": "12 predeclared primary rows; no full table rebuilt and no per-context projected table",
            },
        },
        "lens_diagnostic": {
            "label": "frozen fitted-lens diagnostic; not fitting-free",
            "original_loader_and_forward": "reference.strict_bos.round001_teacher.load_frozen_lens and FrozenAffineLens.forward",
            "native_lens_s": float(lens_s.item()),
            "native_logit_scale_exp_s": lens_logit_scale,
            "score_comparison": "A1 native logits are cosine * exp(s); reported A1 scores/margins are divided by positive exp(s) for cosine-equivalent comparison; native values remain in each row/summary",
            "projected_prototype_definition": "g(h) against one streamed g(b_v) table derived from the reused P01 table",
            "raw_separation_primary_C0_to_C4": raw_separation,
            "projected_separation_primary_C0_to_C4": projected_separation,
            "raw_separation_equal_position_C1_to_C4": raw_equal_position_separation,
            "projected_separation_equal_position_C1_to_C4": projected_equal_position_separation,
            "targeted_same_rows": targeted_summaries,
            "projected_prototype_materialization": "one streamed shared table retained in memory; no per-context projected table written",
        },
        "timing": {
            "preflight_seconds": preflight_seconds,
            **phase_timing,
            "total_seconds": time.perf_counter() - started,
        },
        "resource": {
            "preflight_guard": guard,
            "peak_memory": _host_memory(),
            "expected_peak_rss_bytes": EXPECTED_PEAK_RSS_BYTES,
            "rss_ceiling_bytes": RSS_CEILING_BYTES,
            "rss_ceiling_checks": rss_checks,
            "ranking_score_buffer_max_bytes": RANKING_SCORE_BUFFER_BYTES,
            "largest_representative_cell": "46 public teacher-prefix endpoint rows plus one streamed projected vocabulary table",
        },
        "public_model_cost": {
            "public_prefix_calls": counters.public_prefix_calls,
            "public_prefix_input_token_evaluations": counters.public_prefix_input_token_evaluations,
            "full_forward_calls": counters.full_calls,
            "full_forward_input_token_evaluations": counters.full_input_tokens,
            "cached_forward_calls": counters.cached_calls,
            "cached_forward_input_token_evaluations": counters.cached_input_tokens,
            "candidate_simulations": 0,
            "target_model_calls": 0,
        },
        "artifact": {
            "activation_panel": {"path": str(tensor_path), "sha256": _sha256_file(tensor_path)},
            "figures": [{"path": str(path), "sha256": _sha256_file(path)} for path in figures],
        },
    }
    _write_json_exclusive(diagnostics_path, diagnostics)
    diagnostics_record = _file_record(diagnostics_path)

    evidence_files = [
        output_root / "preflight.json",
        qualification_path,
        tensor_path,
        *figures,
        diagnostics_path,
    ]
    manifest = {
        "schema": "token-reconstruction.trr-p02-manifest.v1",
        "task_id": TASK_ID,
        "status": "COMPLETE_PUBLIC_DIAGNOSTIC",
        "truth_opened": False,
        "source_truth_included": False,
        "seed": SEED,
        "implementation_commit": args.implementation_commit,
        "plan": _file_record(plan_path),
        "diagnostics": diagnostics_record,
        "source_files": [
            _file_record(Path(__file__)),
            _file_record(_SOURCE_ROOT / "src/token_reconstruction/trr_p02/geometry.py"),
            _file_record(_SOURCE_ROOT / "src/token_reconstruction/public_prefix.py"),
            _file_record(_SOURCE_ROOT / "src/token_reconstruction/trr_p01/boundary_prototype.py"),
            _file_record(_SOURCE_ROOT / "src/token_reconstruction/trr_p01/historical_comparators.py"),
            _file_record(_SOURCE_ROOT / "reference/strict_bos/round001_teacher.py"),
        ],
        "assets": {
            "prototype": _file_record(prototype_path),
            "historical_lens": _file_record(lens_path),
        },
        "evidence_files": [_file_record(path, root=output_root) for path in evidence_files],
        "runtime": _runtime_record(),
        "command": {
            "argv": [str(value) for value in sys.argv],
            "cwd": os.getcwd(),
            "selected_device": "cpu",
        },
        "model": model_identity,
        "decision_scope": {
            "rank_definition": STRICT_RANK_DEFINITION,
            "no_benchmark_score": True,
            "no_private_truth": True,
            "teacher_prefix_controls_explicit": True,
            "oracle_centering_not_method": True,
            "fitted_lens_not_method": True,
            "full_vocab_rows": 12,
            "local_dictionary_n": 8,
        },
    }
    _write_json_exclusive(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": diagnostics["status"],
                "diagnostics": str(diagnostics_path),
                "manifest": str(manifest_path),
                "public_prefix_calls": counters.public_prefix_calls,
                "elapsed_seconds": diagnostics["timing"]["total_seconds"],
                "peak_rss_bytes": diagnostics["resource"]["peak_memory"]["process_max_rss_bytes"],
                "truth_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DiagnosticError, GeometryDiagnosticError) as exc:
        print(f"P02 diagnostic failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
