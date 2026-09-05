#!/usr/bin/env python3
"""Guarded public preparation for the TRR-P01 boundary prototype table.

This command never accepts a record, source token, target model, or truth path.
It first qualifies the declared full-vocabulary boundary pass on the fixed
256-token probe set, then (unless ``--qualification-only`` is selected) builds
and saves the full ascending-ID table.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import torch
from safetensors.torch import save_file

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT / "src"))

from token_reconstruction.public_prefix import ContiguousPublicPrefix  # noqa: E402
from token_reconstruction.trr_p01 import PrototypeTable  # noqa: E402
from common import (  # noqa: E402
    CUT_DEPTH,
    digest_tensor,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    TASK_ID,
    VOCAB_SIZE,
    artifact_entry,
    command_record,
    environment_record,
    estimate_resource_need,
    file_record,
    load_json,
    require_create_only_file,
    load_public_model,
    peak_memory,
    resource_guard,
    require_create_only_directory,
    seed_everything,
    sha256_file,
    utc_now,
    validate_public_plan,
    write_json_exclusive,
)


TABLE_FILENAME = "boundary_prototypes.safetensors"
PREflight_FILENAME = "preflight.json"
QUALIFICATION_FILENAME = "qualification.json"
BUILD_EVIDENCE_FILENAME = "build_evidence.json"
MODEL_BYTES_ESTIMATE = 2_500_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda", help="explicit torch device (default: cuda)")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--alternate-batch-size", type=int, default=128)
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--implementation-commit", default=None)
    return parser.parse_args()


def _probe_ids(plan: dict[str, Any]) -> list[int]:
    qualification = plan.get("qualification")
    if not isinstance(qualification, dict):
        raise RuntimeError("qualification section is missing")
    values = qualification.get("probe_token_ids")
    if not isinstance(values, list) or len(values) != 256:
        raise RuntimeError("qualification probe geometry changed")
    ids = [int(value) for value in values]
    if sorted(ids) != ids or len(set(ids)) != len(ids) or ids[0] < 0 or ids[-1] >= VOCAB_SIZE:
        raise RuntimeError("qualification probe IDs are not a fixed ascending full-vocabulary sample")
    if qualification.get("initial_batch_size") != 256 or qualification.get("alternate_batch_size") != 128:
        raise RuntimeError("qualification batch contract changed")
    return ids


def _forward_probe(
    prefix: ContiguousPublicPrefix,
    probe_ids: Iterable[int],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, float]:
    if batch_size <= 0:
        raise RuntimeError("probe batch size must be positive")
    ids = list(probe_ids)
    started = time.perf_counter()
    values: list[torch.Tensor] = []
    forward_calls = 0
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            block = torch.tensor(ids[start : start + batch_size], dtype=torch.long, device=device)
            bos = torch.full_like(block, 128000)
            input_ids = torch.stack((bos, block), dim=1)
            output = prefix.forward_full(input_ids)
            if output.ndim != 3 or tuple(output.shape[:2]) != (block.shape[0], 2):
                raise RuntimeError("public prefix qualification geometry changed")
            values.append(output[:, 1, :].detach().to(device="cpu"))
            forward_calls += 1
    result = torch.cat(values, dim=0).contiguous()
    if tuple(result.shape) != (len(ids), HIDDEN_SIZE):
        raise RuntimeError("public prefix qualification hidden geometry changed")
    if result.dtype != torch.bfloat16:
        raise RuntimeError(f"public prefix output dtype changed: {result.dtype}")
    if not torch.isfinite(result).all().item():
        raise RuntimeError("public prefix qualification produced non-finite values")
    return result, forward_calls, time.perf_counter() - started


def run_qualification(
    prefix: ContiguousPublicPrefix,
    probe_ids: list[int],
    *,
    chosen_batch_size: int,
    alternate_batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    chosen, chosen_calls, chosen_seconds = _forward_probe(
        prefix, probe_ids, batch_size=chosen_batch_size, device=device
    )
    alternate, alternate_calls, alternate_seconds = _forward_probe(
        prefix, probe_ids, batch_size=alternate_batch_size, device=device
    )
    exact_equal = bool(torch.equal(chosen, alternate))
    # The declared production batch remains 256 even when an alternate fails;
    # an excluded alternate is evidence, never a silent numerical workaround.
    return {
        "schema": "token-reconstruction.trr-p01-qualification.v1",
        "task_id": TASK_ID,
        "truth_opened": False,
        "probe_count": len(probe_ids),
        "probe_token_ids": probe_ids,
        "chosen_batch_size": int(chosen_batch_size),
        "alternate_batch_size": int(alternate_batch_size),
        "chosen_forward_calls": chosen_calls,
        "alternate_forward_calls": alternate_calls,
        "chosen_seconds": chosen_seconds,
        "alternate_seconds": alternate_seconds,
        "float_output_dtype": str(chosen.dtype).replace("torch.", ""),
        "exact_output_equivalence": exact_equal,
        "alternate_status": "ACCEPTED_EQUIVALENT" if exact_equal else "EXCLUDED_NON_EQUIVALENT",
        "production_batch_selected": int(chosen_batch_size),
        "prediction_equivalence_rule": "torch.equal on ordered probe output tensors before any distance calculation",
        "chosen_output_sha256": digest_tensor(chosen),
        "alternate_output_sha256": digest_tensor(alternate),
        "maximum_absolute_difference": float((chosen.float() - alternate.float()).abs().max().item()),
    }, chosen, alternate


def _reset_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def main() -> int:
    args = parse_args()
    plan = load_json(args.plan)
    validate_public_plan(plan)
    if args.batch_size != 256 or args.alternate_batch_size != 128:
        raise RuntimeError("only the preregistered build batches are accepted")
    if args.model_path is not None and args.model_path.is_symlink():
        raise RuntimeError("model path must be a regular local directory")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    root = require_create_only_directory(args.output_root.resolve())
    started_utc = utc_now()
    seed_everything(1701, device)
    expected_table_bytes = VOCAB_SIZE * HIDDEN_SIZE * 2
    estimate = estimate_resource_need(
        table_bytes=expected_table_bytes,
        model_bytes=MODEL_BYTES_ESTIMATE,
        query_rows=256,
    )
    guard = resource_guard(
        device=device,
        required_bytes=int(estimate["guard_required_bytes"]),
        allocation_bytes=0,
    )
    preflight_path = root / PREflight_FILENAME
    write_json_exclusive(
        preflight_path,
        {
            "schema": "token-reconstruction.trr-p01-preflight.v1",
            "task_id": TASK_ID,
            "truth_opened": False,
            "started_utc": started_utc,
            "selected_device": str(device),
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "model_path": str(args.model_path) if args.model_path else None,
                "estimated_bytes": MODEL_BYTES_ESTIMATE,
            },
            "table": {
                "vocab_size": VOCAB_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "storage_dtype": "bfloat16",
                "expected_bytes": expected_table_bytes,
            },
            "estimate": estimate,
            "guard": guard,
            "batch_size": args.batch_size,
            "alternate_batch_size": args.alternate_batch_size,
            "numerics": {
                "tf32": False,
                "float32_distance_only": True,
                "deterministic_algorithms": True,
            },
            "status": "PREPARED_BEFORE_MODEL_LOAD",
        },
    )

    _reset_cuda(device)
    timer_started = time.perf_counter()
    model = load_public_model(device=device, model_path=args.model_path)
    prefix = ContiguousPublicPrefix(model, CUT_DEPTH).to(device).eval()
    if int(prefix.embed_tokens.num_embeddings) != VOCAB_SIZE:
        raise RuntimeError("public prefix vocabulary changed")
    probe_ids = _probe_ids(plan)
    qualification, chosen_probe, alternate_probe = run_qualification(
        prefix,
        probe_ids,
        chosen_batch_size=args.batch_size,
        alternate_batch_size=args.alternate_batch_size,
        device=device,
    )
    qualification["model"] = {"id": MODEL_ID, "revision": MODEL_REVISION, "cut_depth": CUT_DEPTH}
    qualification["peak_memory_after_probe"] = peak_memory(device)
    qualification["chosen_output_dtype"] = str(chosen_probe.dtype).replace("torch.", "")
    qualification["alternate_output_dtype"] = str(alternate_probe.dtype).replace("torch.", "")
    chosen_probe_path = root / "qualification_chosen.safetensors"
    alternate_probe_path = root / "qualification_alternate.safetensors"
    for path in (chosen_probe_path, alternate_probe_path):
        require_create_only_file(path)
    save_file(
        {"prototype_outputs": chosen_probe},
        chosen_probe_path,
        metadata={
            "schema": "token-reconstruction.trr-p01-qualification-output.v1",
            "task_id": TASK_ID,
            "batch_size": str(args.batch_size),
            "truth_opened": "false",
        },
    )
    save_file(
        {"prototype_outputs": alternate_probe},
        alternate_probe_path,
        metadata={
            "schema": "token-reconstruction.trr-p01-qualification-output.v1",
            "task_id": TASK_ID,
            "batch_size": str(args.alternate_batch_size),
            "truth_opened": "false",
        },
    )
    qualification["chosen_output"] = file_record(chosen_probe_path)
    qualification["alternate_output"] = file_record(alternate_probe_path)
    qualification_path = root / QUALIFICATION_FILENAME
    write_json_exclusive(qualification_path, qualification)

    table_path: Path | None = None
    build_stats: dict[str, Any] | None = None
    table_guard: dict[str, Any] | None = None
    if not args.qualification_only:
        post_model_estimate = estimate_resource_need(
            table_bytes=expected_table_bytes, model_bytes=0, query_rows=256
        )
        table_guard = resource_guard(
            device=device,
            required_bytes=int(post_model_estimate["guard_required_bytes"]),
            allocation_bytes=0,
        )
        with torch.inference_mode():
            table, stats = PrototypeTable.build(
                prefix,
                vocab_size=VOCAB_SIZE,
                bos_token_id=128000,
                cut_depth=CUT_DEPTH,
                model_id=MODEL_ID,
                model_revision=MODEL_REVISION,
                batch_size=args.batch_size,
                storage_dtype=torch.bfloat16,
                device=device,
                return_stats=True,
            )
        table_path = root / TABLE_FILENAME
        table_digest = table.save(table_path)
        build_stats = {
            "vocab_size": stats.vocab_size,
            "hidden_size": stats.hidden_size,
            "batch_size": stats.batch_size,
            "forward_calls": stats.forward_calls,
            "input_token_evaluations": stats.input_token_evaluations,
            "committed_tokens_compat": stats.committed_tokens,
            "elapsed_seconds": stats.elapsed_seconds,
            "output_dtype": stats.output_dtype,
            "table_bytes": table_path.stat().st_size,
            "table_sha256": table_digest,
        }
    ended_utc = utc_now()
    evidence = {
        "schema": "token-reconstruction.trr-p01-build-evidence.v1",
        "task_id": TASK_ID,
        "truth_opened": False,
        "source_truth_included": False,
        "status": "QUALIFICATION_ONLY" if args.qualification_only else "PUBLIC_TABLE_BUILT",
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "command": command_record(device),
        "environment": environment_record(device),
        "selected_device": str(device),
        "plan": file_record(args.plan),
        "preflight": file_record(preflight_path),
        "qualification": file_record(qualification_path),
        "qualification_chosen_output": file_record(chosen_probe_path),
        "qualification_alternate_output": file_record(alternate_probe_path),
        "post_model_table_guard": table_guard,
        "build_elapsed_seconds": time.perf_counter() - timer_started,
        "peak_memory": peak_memory(device),
        "prototype_build": build_stats,
        "table": file_record(table_path) if table_path else None,
        "public_prefix_token_evaluations": 2 * VOCAB_SIZE if table_path else 2 * len(probe_ids),
        "candidate_simulations": 0,
        "implementation_commit": args.implementation_commit or os.environ.get("TRR_P01_IMPLEMENTATION_COMMIT", "UNBOUND_PRECOMMIT"),
        "source_files": [
            file_record(_SOURCE_ROOT / "src/token_reconstruction/trr_p01/boundary_prototype.py"),
            file_record(_SOURCE_ROOT / "scripts/trr_p01/common.py"),
            file_record(Path(__file__)),
        ],
        "phase_timing": {
            "model_load_and_prefix_seconds": None,
            "qualification_seconds": qualification["chosen_seconds"] + qualification["alternate_seconds"],
            "table_build_seconds": build_stats["elapsed_seconds"] if build_stats else None,
            "table_save_and_hash_seconds": None,
        },
    }
    evidence_path = root / BUILD_EVIDENCE_FILENAME
    write_json_exclusive(evidence_path, evidence)
    print(
        {
            "status": evidence["status"],
            "qualification": str(qualification_path),
            "table": str(table_path) if table_path else None,
            "truth_opened": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
