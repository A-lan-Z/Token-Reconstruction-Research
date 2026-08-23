#!/usr/bin/env python3
"""Retrospective target/surrogate transfer panel for TRR-0002 owner R2.

The ``predict`` subcommand deliberately has no truth argument and never
reconstructs the Finance rows. It receives only the retained target
activations plus the public A1 lens and public Llama prefix. A separate
``score`` process verifies the frozen prediction artifact before loading the
already-open historical Finance truth.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file, save_file
import torch
import transformers

import trr0001_r2_dual_benchmark as r2
from token_reconstruction.a1a2_configuration_search import (
    decode_policy,
    resolved_policy_from_dict,
)
from token_reconstruction.component_crossover import propose_public_a1
from token_reconstruction.dual_benchmark import (
    paired_record_differences,
    score_predictions,
    scored_mask,
)
from token_reconstruction.experiment_runtime import (
    command_record,
    peak_memory,
    seed_everything,
)
from token_reconstruction.metrics import bootstrap_mean


TASK_ID = "TRR-0002"
REVISION_ID = "TRR-0002-OWNER-REVISION-R2"
TABLE_SHA256 = "89d0b1393c8afdab49d31ed474057117bb5765ad57b25486682a53a98ef8d59c"
SOURCE_CONFIG_RELATIVE = "config/a1_a2_source300_static_20260809.json"
SOURCE_CONFIG_SHA256 = "79ef9c9ca7400f003b94950dc299a8f3c314073258c76432ef273a920b3a460c"
SOURCE_TRACE_SHA256 = "0c880019a7b497fbf5d784b2cddb54f67b740c58c505fed8895697a1a1b4477a"
PUBLIC_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
PUBLIC_MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"

# The order is frozen for execution. It is deliberately not ordered by any
# target metric: low-cost controls, public leaders, and distinct adaptive rules
# are interleaved to make the scientific coverage explicit.
SHORTLIST: tuple[dict[str, str], ...] = (
    {
        "label": "fixed_k64_direct",
        "policy_id": "a1a2_589f6e179eb4626877c2",
        "reason": "low-cost fixed-budget control",
    },
    {
        "label": "calibrated_k32_to_k64",
        "policy_id": "a1a2_c316cdf581012bd81cfa",
        "reason": "previous scale-calibrated balanced control",
    },
    {
        "label": "fixed_k128_direct",
        "policy_id": "a1a2_422b282c012ff665ee2e",
        "reason": "middle point on the fixed-budget cost curve",
    },
    {
        "label": "fixed_k256_direct",
        "policy_id": "a1a2_43ea0bb737bc075531ca",
        "reason": "frozen exhaustive-search winner and public rank 1",
    },
    {
        "label": "fixed_k256_centered",
        "policy_id": "a1a2_cb89b524f27c2d5e25eb",
        "reason": "public rank 2; isolates score centering at K256",
    },
    {
        "label": "fixed_k512_direct",
        "policy_id": "a1a2_cce5e6b5435e9b1bee34",
        "reason": "public rank 3; tests whether mismatch needs more candidates",
    },
    {
        "label": "fixed_k512_centered",
        "policy_id": "a1a2_13f73c306bf8946e9a28",
        "reason": "public rank 4; tests centering at the maximum budget",
    },
    {
        "label": "fast_a1_099_then_k256_direct",
        "policy_id": "a1a2_91503c1f37fac38c4e20",
        "reason": "best public near-perfect speed configuration",
    },
    {
        "label": "fast_a1_099_then_k256_centered",
        "policy_id": "a1a2_ae35177bb01fa67279c3",
        "reason": "score-centering counterpart to the A1 fast path",
    },
    {
        "label": "multistage_historical_gate",
        "policy_id": "a1a2_a49923b51936a41a41fb",
        "reason": "best distinct public multi-stage low-simulation finalist",
    },
    {
        "label": "adaptive_k256_to_k512_rms_q05",
        "policy_id": "a1a2_d1700b9c0f2b1b32ec13",
        "reason": "best public fitted scale-normalized expansion finalist",
    },
    {
        "label": "historical_strict_anchor",
        "policy_id": "a1a2_6de800ba92c3d0ec0808",
        "reason": "exact historical strict adaptive decision-rule control",
    },
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    preregister = subparsers.add_parser("preregister")
    preregister.add_argument("--repository-root", type=Path, default=Path("."))
    preregister.add_argument("--historical-root", type=Path, required=True)
    preregister.add_argument("--table", type=Path, required=True)
    preregister.add_argument("--request", type=Path, required=True)
    preregister.add_argument("--output", type=Path, required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--repository-root", type=Path, default=Path("."))
    predict.add_argument("--historical-root", type=Path, required=True)
    predict.add_argument("--plan", type=Path, required=True)
    predict.add_argument("--prediction-artifact", type=Path, required=True)
    predict.add_argument("--evidence", type=Path, required=True)
    predict.add_argument("--freeze-receipt", type=Path, required=True)
    predict.add_argument("--record-batch-size", type=int, default=8)

    score = subparsers.add_parser("score")
    score.add_argument("--repository-root", type=Path, default=Path("."))
    score.add_argument("--historical-root", type=Path, required=True)
    score.add_argument("--plan", type=Path, required=True)
    score.add_argument("--prediction-artifact", type=Path, required=True)
    score.add_argument("--evidence", type=Path, required=True)
    score.add_argument("--freeze-receipt", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()


def assert_file_record(record: Mapping[str, Any], path: Path, label: str) -> None:
    observed = r2.file_record(path)
    for field in ("bytes", "sha256"):
        if observed[field] != record.get(field):
            raise RuntimeError(f"{label} {field} changed after freeze")


def counts(values: torch.Tensor, mask: torch.Tensor) -> dict[str, int]:
    selected = values[mask].to(torch.long)
    unique, frequencies = torch.unique(selected, return_counts=True)
    return {
        str(int(key.item())): int(value.item())
        for key, value in zip(unique, frequencies, strict=True)
    }


def table_index(table: Mapping[str, Any]) -> dict[str, tuple[int, Mapping[str, Any]]]:
    rows = table.get("rows")
    if not isinstance(rows, list) or len(rows) != 57:
        raise RuntimeError("frozen public causal table changed")
    indexed: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for rank, row in enumerate(rows, start=1):
        policy_id = str(row["policy_id"])
        if policy_id in indexed:
            raise RuntimeError("duplicate policy in public causal table")
        policy = resolved_policy_from_dict(row["policy"])
        if policy.policy_id != policy_id:
            raise RuntimeError("public causal policy identity changed")
        indexed[policy_id] = (rank, row)
    return indexed


def public_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    domains = row["domains"]
    return {
        domain: {
            "token_accuracy": domains[domain]["metrics"]["token_accuracy"],
            "correct_tokens": domains[domain]["metrics"]["correct_tokens"],
            "scored_tokens": domains[domain]["metrics"]["scored_tokens"],
            "exact_records": domains[domain]["metrics"]["exact_records"],
            "records": domains[domain]["metrics"]["records"],
            "median_selection_seconds": domains[domain]["timing"][
                "median_selection_seconds"
            ],
            "candidate_simulations": domains[domain]["timing"][
                "candidate_simulations"
            ],
        }
        for domain in ("pile", "finance")
    }


def build_preregistration(
    *,
    repository_root: Path,
    historical_root: Path,
    table_path: Path,
    request_path: Path,
) -> dict[str, Any]:
    if r2.sha256_file(table_path) != TABLE_SHA256:
        raise RuntimeError("public causal table no longer matches the reviewed table")
    source_config_path = historical_root / SOURCE_CONFIG_RELATIVE
    if r2.sha256_file(source_config_path) != SOURCE_CONFIG_SHA256:
        raise RuntimeError("historical source-300 config changed")
    source_config = load_json(source_config_path)
    source_trace_path = historical_root / str(source_config["source"]["path"])
    if r2.sha256_file(source_trace_path) != SOURCE_TRACE_SHA256:
        raise RuntimeError("historical source-300 activation trace changed")
    table = load_json(table_path)
    indexed = table_index(table)
    if len({entry["policy_id"] for entry in SHORTLIST}) != len(SHORTLIST):
        raise RuntimeError("shortlist contains duplicate policy IDs")
    shortlist = []
    for entry in SHORTLIST:
        policy_id = entry["policy_id"]
        if policy_id not in indexed:
            raise RuntimeError(f"shortlist policy is absent from frozen table: {policy_id}")
        public_rank, row = indexed[policy_id]
        shortlist.append(
            {
                **entry,
                "public_rank": public_rank,
                "policy": row["policy"],
                "public_selection_summary": public_summary(row),
            }
        )
    identity_path = (
        historical_root
        / "research/adaptive_a1_a2_strict_bos_20260817_goal_01a00b08"
        / "audit/AUDIT-0004-public-teacher-identity-v2.json"
    )
    lens_path = historical_root / "inversion_20260730/out/lens_alpaca.pt"
    return {
        "schema": "token-reconstruction.trr0002-owner-r2-finance-target-shortlist-preregistration.v1",
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "FROZEN_BEFORE_FINANCE_TARGET_SHORTLIST_RECONSTRUCTION",
        "created_utc": r2.utc_now(),
        "created_from_commit": git_head(repository_root),
        "owner_request": r2.file_record(request_path),
        "scientific_question": (
            "Do publicly leading A1+A2 configurations separate when the observed "
            "prefix belongs to the generation-300 Finance-Instruct target but A1 "
            "and A2 retain only the untouched public surrogate?"
        ),
        "target": {
            "model_family": PUBLIC_MODEL_ID,
            "base_revision": PUBLIC_MODEL_REVISION,
            "fine_tuning_dataset": source_config["truth"]["dataset"],
            "checkpoint_generation": source_config["source"]["checkpoint_generation"],
            "target_step": source_config["source"]["target_step"],
            "weight_version": source_config["source"]["weight_version"],
            "boundary_layer": source_config["model"]["boundary_layer"],
            "records": 128,
            "positions": 128,
            "expected_valid_positions_including_bos": source_config["source"][
                "expected_valid_positions"
            ],
            "source_config": r2.file_record(source_config_path),
            "activation_trace": r2.file_record(source_trace_path),
        },
        "surrogate": {
            "model_id": PUBLIC_MODEL_ID,
            "revision": PUBLIC_MODEL_REVISION,
            "prefix_layers": [0, 1, 2, 3],
            "target_fine_tune_available_to_a1_or_a2": False,
            "target_prefix_calls_permitted": 0,
            "a1": "pinned public Alpaca affine lens over the target activation",
            "a2": "untouched public Llama layers 0 through 3 with reconstructed prefix",
            "public_teacher_identity": r2.file_record(identity_path),
            "public_a1_lens": r2.file_record(lens_path),
        },
        "shortlist_selection": {
            "source_table": r2.file_record(table_path),
            "source_table_expected_sha256": TABLE_SHA256,
            "selected_without_finance_target_metrics": True,
            "selection_rule": (
                "retain public ranks 1-5, the centered fast-path counterpart, "
                "the best distinct multistage and fitted adaptive finalists, "
                "fixed K64/K128 cost controls, the previous calibrated control, "
                "and the exact historical strict control"
            ),
            "shortlist": shortlist,
        },
        "execution_protocol": {
            "prediction_process_truth_argument": None,
            "prediction_process_dataset_inputs": 0,
            "prediction_process_target_prefix_calls": 0,
            "candidate_order": "public A1 top-512, unchanged",
            "policy_rules": "exact serialized policies from the frozen public table",
            "record_batch_size": 8,
            "timing_passes": 1,
            "freeze": "create-only safetensors artifact and SHA-256 receipt",
            "scoring": "separate process after receipt verification",
        },
        "ranking": {
            "diagnostic_only": True,
            "order": [
                "highest target token accuracy",
                "highest exact-record count",
                "lowest measured method compute",
                "public rank",
            ],
        },
        "claim_limits": [
            "historical Finance truth was already open before R2",
            "result is retrospective target-shift stress evidence, not fresh blind confirmation",
            "target ranking cannot replace the previously frozen exhaustive winner",
            "no cross-setup overall-best claim may be made from this single diagnostic",
        ],
    }


def validate_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    if (
        plan.get("schema")
        != "token-reconstruction.trr0002-owner-r2-finance-target-shortlist-preregistration.v1"
        or plan.get("status")
        != "FROZEN_BEFORE_FINANCE_TARGET_SHORTLIST_RECONSTRUCTION"
        or plan.get("revision_id") != REVISION_ID
    ):
        raise RuntimeError("R2 Finance-target plan identity changed")
    if plan["shortlist_selection"]["source_table"]["sha256"] != TABLE_SHA256:
        raise RuntimeError("R2 plan does not bind the frozen public causal table")
    entries = list(plan["shortlist_selection"]["shortlist"])
    expected_ids = [entry["policy_id"] for entry in SHORTLIST]
    observed_ids = [entry["policy_id"] for entry in entries]
    if observed_ids != expected_ids:
        raise RuntimeError("R2 Finance-target shortlist order changed")
    for entry in entries:
        policy = resolved_policy_from_dict(entry["policy"])
        if policy.policy_id != entry["policy_id"]:
            raise RuntimeError("R2 shortlist serialized policy changed")
    return entries


def command_preregister(args: argparse.Namespace) -> int:
    repository_root = args.repository_root.resolve(strict=True)
    historical_root = args.historical_root.resolve(strict=True)
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("R2 preregistration output is create-only")
    payload = build_preregistration(
        repository_root=repository_root,
        historical_root=historical_root,
        table_path=args.table.resolve(strict=True),
        request_path=args.request.resolve(strict=True),
    )
    r2.write_json_exclusive(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "shortlist_count": len(payload["shortlist_selection"]["shortlist"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def load_public_components(
    repository_root: Path,
    historical_root: Path,
) -> tuple[Any, Any, torch.Tensor, torch.device, dict[str, Any], Path, Path]:
    reference_path = repository_root / "reference/strict_bos/round001_teacher.py"
    reference = r2.import_path("trr0002_r2_finance_reference", reference_path)
    identity_path = (
        historical_root
        / "research/adaptive_a1_a2_strict_bos_20260817_goal_01a00b08"
        / "audit/AUDIT-0004-public-teacher-identity-v2.json"
    )
    lens_path = historical_root / "inversion_20260730/out/lens_alpaca.pt"
    precut, lens, embeddings, device, identity = reference.load_public_teacher(
        r2.MODEL_SPEC,
        load_json(identity_path),
        lens_path=lens_path,
    )
    return precut, lens, embeddings, device, identity, identity_path, lens_path


def command_predict(args: argparse.Namespace) -> int:
    repository_root = args.repository_root.resolve(strict=True)
    historical_root = args.historical_root.resolve(strict=True)
    for output in (args.prediction_artifact, args.evidence, args.freeze_receipt):
        if output.exists() or output.is_symlink():
            raise RuntimeError(f"R2 prediction output is create-only: {output}")
    if args.record_batch_size != 8:
        raise RuntimeError("R2 preregistered record batch size is 8")
    plan = load_json(args.plan)
    entries = validate_plan(plan)
    started_utc = r2.utc_now()
    seed_everything(20260831)
    execution_commit = git_head(repository_root)

    source_path = historical_root / "scripts/score_a1_a2_source300_20260809.py"
    source300 = r2.import_path("trr0002_r2_finance_source300", source_path)
    config, _captures, observations, attention_mask, position_ids = r2.historical_inputs(
        historical_root, source300
    )
    if r2.sha256_file(historical_root / SOURCE_CONFIG_RELATIVE) != SOURCE_CONFIG_SHA256:
        raise RuntimeError("target source config changed after preregistration")
    trace_path = source300.resolve_inside_ersoy(config["source"]["path"])
    if r2.sha256_file(trace_path) != SOURCE_TRACE_SHA256:
        raise RuntimeError("target activation trace changed after preregistration")

    precut, lens, embeddings, device, public_identity, identity_path, lens_path = (
        load_public_components(repository_root, historical_root)
    )
    torch.cuda.reset_peak_memory_stats(device)
    proposal = propose_public_a1(
        observations=observations,
        attention_mask=attention_mask,
        lens=lens,
        normalized_embeddings=embeddings,
    )
    tensors: dict[str, torch.Tensor] = {
        "common.candidates_top512": proposal.candidates.to(torch.int32),
        "common.a1_confidence": proposal.top1_confidence.float(),
        "common.attention_mask": attention_mask.to(torch.uint8),
        "common.position_ids": position_ids.to(torch.int32),
    }
    policy_costs: dict[str, Any] = {}
    for index, entry in enumerate(entries, start=1):
        policy = resolved_policy_from_dict(entry["policy"])
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        result = decode_policy(
            observations=observations,
            attention_mask=attention_mask,
            position_ids=position_ids,
            candidates=proposal.candidates,
            a1_confidence=proposal.top1_confidence,
            precut=precut,
            device=device,
            policy=policy,
            record_batch_size=args.record_batch_size,
        )
        prefix = policy.policy_id
        tensors[f"{prefix}.predictions"] = result.predictions.to(torch.int32)
        tensors[f"{prefix}.routes"] = result.routes.to(torch.int8)
        tensors[f"{prefix}.selected_k"] = result.selected_k.to(torch.int16)
        tensors[f"{prefix}.selected_signal"] = result.selected_signal.float()
        policy_costs[policy.policy_id] = {
            "label": entry["label"],
            "selection_seconds": result.elapsed_seconds,
            "method_compute_seconds": proposal.elapsed_seconds + result.elapsed_seconds,
            "candidate_simulations": result.candidate_simulations,
            "executed_candidate_simulations": result.executed_candidate_simulations,
            "prefix_commit_tokens": result.prefix_commit_tokens,
            "record_batch_size": result.record_batch_size,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        print(
            json.dumps(
                {
                    "status": "FINANCE_TARGET_PREDICTION_PROGRESS",
                    "completed": index,
                    "total": len(entries),
                    "label": entry["label"],
                    "policy_id": policy.policy_id,
                    "selection_seconds": result.elapsed_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    args.prediction_artifact.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.contiguous() for name, value in tensors.items()},
        args.prediction_artifact,
        metadata={
            "schema": "token-reconstruction.trr0002-owner-r2-finance-target-prediction-freeze.v1",
            "task_id": TASK_ID,
            "revision_id": REVISION_ID,
            "execution_commit": execution_commit,
            "plan_sha256": r2.sha256_file(args.plan),
            "truth_status": "not_loaded_by_prediction_process",
            "target_prefix_calls": "0",
        },
    )
    prediction_frozen_utc = r2.utc_now()
    evidence = {
        "schema": "token-reconstruction.trr0002-owner-r2-finance-target-reconstruction-evidence.v1",
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "PREDICTIONS_FROZEN_WITHOUT_TRUTH",
        "started_utc": started_utc,
        "ended_utc": prediction_frozen_utc,
        "execution_commit": execution_commit,
        "command": command_record(),
        "exit_status": 0,
        "plan": r2.file_record(args.plan),
        "prediction_artifact": r2.file_record(args.prediction_artifact),
        "target_observation": {
            "source_config": r2.file_record(historical_root / SOURCE_CONFIG_RELATIVE),
            "activation_trace": r2.file_record(trace_path),
            "checkpoint_generation": config["source"]["checkpoint_generation"],
            "weight_version": config["source"]["weight_version"],
            "records": int(observations.shape[0]),
            "positions": int(observations.shape[1]),
            "hidden_size": int(observations.shape[2]),
        },
        "access": {
            "truth_paths_or_arguments": [],
            "dataset_inputs": 0,
            "truth_token_inputs": 0,
            "target_prefix_calls": 0,
            "target_weights_available_to_method": False,
            "public_surrogate_prefix_calls_only": True,
        },
        "proposal": {
            "method": "pinned public Alpaca A1 lens",
            "seconds": proposal.elapsed_seconds,
            "candidate_budget": 512,
        },
        "policies": policy_costs,
        "public_teacher_identity": public_identity,
        "artifacts": {
            "public_teacher_identity": r2.file_record(identity_path),
            "public_a1_lens": r2.file_record(lens_path),
            "method_source": r2.file_record(
                repository_root / "src/token_reconstruction/a1a2_configuration_search.py"
            ),
            "runner_source": r2.file_record(Path(__file__)),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(device),
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_memory": peak_memory(),
            "pid": os.getpid(),
        },
    }
    r2.write_json_exclusive(args.evidence, evidence)
    receipt = {
        "schema": "token-reconstruction.trr0002-owner-r2-finance-target-freeze-receipt.v1",
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "VERIFIED_BEFORE_SEPARATE_SCORING_PROCESS",
        "created_utc": r2.utc_now(),
        "execution_commit": execution_commit,
        "plan": r2.file_record(args.plan),
        "prediction_artifact": r2.file_record(args.prediction_artifact),
        "reconstruction_evidence": r2.file_record(args.evidence),
        "prediction_process_truth_loaded": False,
        "prediction_process_target_prefix_calls": 0,
    }
    r2.write_json_exclusive(args.freeze_receipt, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "policies": len(entries),
                "prediction_sha256": receipt["prediction_artifact"]["sha256"],
                "receipt": str(args.freeze_receipt),
            },
            sort_keys=True,
        )
    )
    return 0

def expected_tensor_keys(entries: Sequence[Mapping[str, Any]]) -> set[str]:
    keys = {
        "common.candidates_top512",
        "common.a1_confidence",
        "common.attention_mask",
        "common.position_ids",
    }
    for entry in entries:
        policy_id = str(entry["policy_id"])
        for suffix in ("predictions", "routes", "selected_k", "selected_signal"):
            keys.add(f"{policy_id}.{suffix}")
    return keys


def candidate_recall(
    candidates: torch.Tensor,
    truth: torch.Tensor,
    attention_mask: torch.Tensor,
    k: int,
) -> dict[str, Any]:
    mask = scored_mask(attention_mask)
    expected = truth[mask].to(torch.long)
    hits = candidates[mask, :k].to(torch.long).eq(expected[:, None]).any(dim=1)
    return {
        "k": k,
        "hits": int(hits.sum().item()),
        "scored_tokens": int(hits.numel()),
        "recall": int(hits.sum().item()) / int(hits.numel()),
    }


def command_score(args: argparse.Namespace) -> int:
    repository_root = args.repository_root.resolve(strict=True)
    historical_root = args.historical_root.resolve(strict=True)
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("R2 Finance-target score output is create-only")
    plan = load_json(args.plan)
    entries = validate_plan(plan)
    receipt = load_json(args.freeze_receipt)
    evidence = load_json(args.evidence)
    if (
        receipt.get("status") != "VERIFIED_BEFORE_SEPARATE_SCORING_PROCESS"
        or receipt.get("prediction_process_truth_loaded") is not False
        or receipt.get("prediction_process_target_prefix_calls") != 0
        or evidence.get("status") != "PREDICTIONS_FROZEN_WITHOUT_TRUTH"
        or evidence["access"]["truth_token_inputs"] != 0
        or evidence["access"]["target_prefix_calls"] != 0
    ):
        raise RuntimeError("prediction freeze/access receipt is invalid")
    assert_file_record(receipt["plan"], args.plan, "plan")
    assert_file_record(
        receipt["prediction_artifact"], args.prediction_artifact, "predictions"
    )
    assert_file_record(
        receipt["reconstruction_evidence"], args.evidence, "reconstruction evidence"
    )
    started_utc = r2.utc_now()
    frozen = load_file(args.prediction_artifact, device="cpu")
    if set(frozen) != expected_tensor_keys(entries):
        raise RuntimeError("frozen prediction tensor registry changed")

    source_path = historical_root / "scripts/score_a1_a2_source300_20260809.py"
    source300 = r2.import_path("trr0002_r2_finance_score_source300", source_path)
    config, captures, _observations, attention_mask, _position_ids = r2.historical_inputs(
        historical_root, source300
    )
    truth, record_ids = r2.load_old_truth(source300, captures, config)
    truth_loaded_utc = r2.utc_now()
    if not torch.equal(attention_mask.to(torch.uint8), frozen["common.attention_mask"]):
        raise RuntimeError("frozen attention mask changed before scoring")
    candidates = frozen["common.candidates_top512"].to(torch.long)
    if candidates.shape != (128, 128, 512):
        raise RuntimeError("frozen A1 candidate geometry changed")
    mask = scored_mask(attention_mask)

    scored_rows: list[dict[str, Any]] = []
    per_record_by_policy: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        policy = resolved_policy_from_dict(entry["policy"])
        policy_id = policy.policy_id
        predictions = frozen[f"{policy_id}.predictions"].to(torch.long)
        routes = frozen[f"{policy_id}.routes"].to(torch.int8)
        selected_k = frozen[f"{policy_id}.selected_k"].to(torch.int16)
        maximum_executed_k = int(selected_k[mask].max().item())
        scoring_k = max(1, maximum_executed_k)
        metrics, per_record = score_predictions(
            predictions=predictions,
            truth=truth,
            attention_mask=attention_mask,
            candidates=candidates[:, :, :scoring_k],
            record_ids=record_ids,
        )
        per_record_by_policy[policy_id] = per_record
        cost = evidence["policies"][policy_id]
        scored_rows.append(
            {
                "label": entry["label"],
                "policy_id": policy_id,
                "public_rank": entry["public_rank"],
                "policy": entry["policy"],
                "metrics": metrics,
                "per_record": per_record,
                "routes": counts(routes, mask),
                "selected_k": counts(selected_k, mask),
                "maximum_executed_k": maximum_executed_k,
                "cost": cost,
            }
        )

    ordered = sorted(
        scored_rows,
        key=lambda row: (
            row["metrics"]["token_accuracy"],
            row["metrics"]["exact_records"],
            -row["cost"]["method_compute_seconds"],
            -row["public_rank"],
        ),
        reverse=True,
    )
    baseline_id = "a1a2_43ea0bb737bc075531ca"
    baseline = next(row for row in scored_rows if row["policy_id"] == baseline_id)
    baseline_seconds = float(baseline["cost"]["method_compute_seconds"])
    baseline_simulations = int(baseline["cost"]["candidate_simulations"])
    for rank, row in enumerate(ordered, start=1):
        row["target_diagnostic_rank"] = rank
        row["correct_token_gap_from_target_best"] = (
            ordered[0]["metrics"]["correct_tokens"] - row["metrics"]["correct_tokens"]
        )
        row["runtime_relative_to_fixed_k256_direct"] = (
            row["cost"]["method_compute_seconds"] / baseline_seconds
        )
        row["simulations_relative_to_fixed_k256_direct"] = (
            row["cost"]["candidate_simulations"] / baseline_simulations
        )
        differences = paired_record_differences(
            per_record_by_policy[row["policy_id"]],
            per_record_by_policy[baseline_id],
        )
        row["paired_vs_fixed_k256_direct"] = {
            "mean_record_accuracy_difference": statistics.mean(differences),
            "bootstrap_95": bootstrap_mean(
                differences, draws=10000, seed=20260900 + rank
            ),
            "better_records": sum(value > 0 for value in differences),
            "tied_records": sum(value == 0 for value in differences),
            "worse_records": sum(value < 0 for value in differences),
        }

    accuracy_values = sorted(
        {row["metrics"]["token_accuracy"] for row in ordered}, reverse=True
    )
    result = {
        "schema": "token-reconstruction.trr0002-owner-r2-finance-target-shortlist-result.v1",
        "task_id": TASK_ID,
        "revision_id": REVISION_ID,
        "status": "RETROSPECTIVE_FINANCE_TARGET_SHORTLIST_SCORED",
        "started_utc": started_utc,
        "truth_loaded_utc": truth_loaded_utc,
        "ended_utc": r2.utc_now(),
        "scoring_commit": git_head(repository_root),
        "prediction_execution_commit": receipt["execution_commit"],
        "command": command_record(),
        "exit_status": 0,
        "truth_status": (
            "historical Finance truth already open before R2; predictions were "
            "frozen in a separate no-truth process and receipt-verified first"
        ),
        "target_surrogate_separation": {
            "target": (
                "generation-300 Finance-Instruct Llama-3.2-1B-Instruct, "
                "weight victim_post_000299, retained layer-4 activations"
            ),
            "a1": "public Alpaca affine lens; no target-fine-tune weights",
            "a2": (
                "untouched public Llama-3.2-1B-Instruct layers 0-3 at revision "
                f"{PUBLIC_MODEL_REVISION}"
            ),
            "target_prefix_calls_by_reconstruction": 0,
        },
        "dataset": {
            "id": config["truth"]["dataset"],
            "records": int(truth.shape[0]),
            "positions": int(truth.shape[1]),
            "scored_tokens": int(mask.sum().item()),
            "boundary_layer": config["model"]["boundary_layer"],
        },
        "candidate_recall_curve": [
            candidate_recall(candidates, truth, attention_mask, k)
            for k in (32, 64, 128, 256, 512)
        ],
        "diagnostic_ranking": ordered,
        "differentiation": {
            "configuration_count": len(ordered),
            "distinct_token_accuracy_values": len(accuracy_values),
            "accuracy_values": accuracy_values,
            "best_correct_tokens": ordered[0]["metrics"]["correct_tokens"],
            "worst_correct_tokens": ordered[-1]["metrics"]["correct_tokens"],
            "correct_token_range": (
                ordered[0]["metrics"]["correct_tokens"]
                - ordered[-1]["metrics"]["correct_tokens"]
            ),
            "perfect_configurations": sum(
                row["metrics"]["token_accuracy"] == 1.0 for row in ordered
            ),
        },
        "artifacts": {
            "plan": r2.file_record(args.plan),
            "prediction_artifact": r2.file_record(args.prediction_artifact),
            "reconstruction_evidence": r2.file_record(args.evidence),
            "freeze_receipt": r2.file_record(args.freeze_receipt),
            "target_source_config": r2.file_record(
                historical_root / SOURCE_CONFIG_RELATIVE
            ),
            "target_activation_trace": r2.file_record(
                source300.resolve_inside_ersoy(config["source"]["path"])
            ),
            "runner_source": r2.file_record(Path(__file__)),
        },
        "claim_limits": plan["claim_limits"],
    }
    r2.write_json_exclusive(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "best": {
                    "label": ordered[0]["label"],
                    "policy_id": ordered[0]["policy_id"],
                    "accuracy": ordered[0]["metrics"]["token_accuracy"],
                    "correct_tokens": ordered[0]["metrics"]["correct_tokens"],
                },
                "distinct_accuracy_values": len(accuracy_values),
                "result": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command_name == "preregister":
        return command_preregister(args)
    if args.command_name == "predict":
        return command_predict(args)
    if args.command_name == "score":
        return command_score(args)
    raise RuntimeError("unknown R2 Finance-target command")


if __name__ == "__main__":
    raise SystemExit(main())
