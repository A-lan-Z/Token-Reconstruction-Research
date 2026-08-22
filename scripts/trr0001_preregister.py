#!/usr/bin/env python3
"""Create the TRR-0001 preregistration from public, pre-truth resources."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

from token_reconstruction.records import record_ids_sha256, select_record_splits


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "TRR-0001"
PACKET_ID = "TRR-PACKET-TRR-0000-ACCEPT-TRR-0001-20260822-EA3A83F35AAD288D502C44D5"
CHARTER_SHA256 = "ab0fbe9dfad39eddee48c14f4cb8201f8c3f02d1c58668d8a8e59be5a250700d"
REQUEST_SHA256 = "e12296750323ecd8d90955f052f6ca2a1140e2e6452dc3cedadf4cb636daca6e"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
DATASET_ID = "NeelNanda/pile-10k"
DATASET_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
SEQUENCE_TOKENS = 40
SOURCE_TOKENS = SEQUENCE_TOKENS - 1
SELECTION_SEED = 20260822
SPLIT_SIZES = OrderedDict(
    (
        ("target_update_train", 64),
        ("inverse_train", 128),
        ("development", 32),
        ("blind_evaluation", 64),
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, size: int | None = None, digest: str | None = None) -> None:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"required file unavailable: {path}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"required regular file unavailable: {path}")
    # Hugging Face snapshots intentionally expose immutable blob files by symlink.
    if size is not None and path.stat().st_size != size:
        raise RuntimeError(f"size mismatch for {path}")
    if digest is not None and sha256_file(path) != digest:
        raise RuntimeError(f"SHA-256 mismatch for {path}")


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def split_payload(records: list[Any]) -> dict[str, Any]:
    return {
        "count": len(records),
        "record_ids_sha256": record_ids_sha256(records),
        "records": [record.as_json() for record in records],
    }


def build_plan(created_utc: str, splits: dict[str, list[Any]]) -> dict[str, Any]:
    return {
        "schema": "token-reconstruction.trr0001.preregistration.v1",
        "task_id": TASK_ID,
        "packet_id": PACKET_ID,
        "created_utc": created_utc,
        "status": "COMMITTED_BEFORE_BLIND_TRUTH",
        "truth_opened": False,
        "authority": {
            "path": "RESEARCH_CHARTER.md",
            "sha256": CHARTER_SHA256,
            "role": "sole_authoritative_research_definition",
        },
        "request": {
            "path": "coordination/requests/TRR-0001.md",
            "sha256": REQUEST_SHA256,
            "bytes": 24375,
        },
        "accepted_predecessor": {
            "task_id": "TRR-0000",
            "pull_request": 1,
            "accepted_head": "b087365766f00432077476bf32a6afdf2e854841",
            "merge_commit": "0f641e5f071dd38331d2e2b7821d40fc74941c2e",
            "accepted_head_is_merge_parent": True,
        },
        "seeds": {
            "record_selection": SELECTION_SEED,
            "global_execution": 1729,
            "target_update": 1730,
            "inverse_training": 1731,
            "record_bootstrap": 1732,
        },
        "resources": {
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "local_files_only": True,
                "dtype": "bfloat16",
                "attention_implementation": "sdpa",
                "architecture": "LlamaForCausalLM",
                "layers": 16,
                "hidden_size": 2048,
                "vocabulary_size": 128256,
            },
            "tokenizer": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "add_special_tokens_for_source_text": False,
                "declared_bos_token_id": 128000,
                "bos_insertion": "prepend exactly once after source tokenization",
            },
            "dataset": {
                "id": DATASET_ID,
                "revision": DATASET_REVISION,
                "split": "train",
                "rows": 10000,
                "license": "bigscience-bloom-rail-1.0",
                "parquet_sha256": "a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31",
                "role": "public auxiliary, development, and blind source",
            },
        },
        "data": {
            "sequence_tokens_including_bos": SEQUENCE_TOKENS,
            "scored_tokens_per_record": SOURCE_TOKENS,
            "blind_records": 64,
            "blind_scored_tokens": 64 * SOURCE_TOKENS,
            "padding": "none; only rows with at least 39 source tokens are eligible",
            "truncation": "first 39 source tokens in tokenizer order",
            "selection": {
                "eligibility": "tokenized source length >= 39 with add_special_tokens=false",
                "ordering": (
                    "ascending SHA-256 of "
                    "'TRR-0001|<dataset_revision>|row:<zero_based_index>|seed:20260822'; "
                    "text content is not part of the ordering key"
                ),
                "split_order": list(SPLIT_SIZES),
                "record_order": "the same ascending selection-key order within each split",
                "splits": {name: split_payload(records) for name, records in splits.items()},
                "disjoint": True,
            },
        },
        "conditions": {
            "primary": {
                "id": "unavailable_target_lora",
                "target_prefix": (
                    "base embedding plus target-specific LoRA-modified decoder layers; "
                    "callable target prefix and LoRA state remain evaluator-only"
                ),
                "unavailable_to_reconstructor": [
                    "target LoRA tensors",
                    "callable target prefix",
                    "source text",
                    "source token IDs except BOS",
                    "correctness feedback",
                ],
            },
            "matched_public_control": {
                "id": "matched_public",
                "target_prefix": "the pinned public checkpoint without the target update",
                "role": "implementation and target-surrogate mismatch control",
                "blind_records": "same 64 records, reconstructed and frozen before shared truth opening",
            },
            "permitted_to_reconstructor": [
                "boundary activation",
                "attention mask",
                "position IDs",
                "cut depth and non-truth metadata",
                "pinned public model and tokenizer",
                "public-data inverse states",
                "declared BOS token",
                "already reconstructed prefix",
            ],
        },
        "target_prefix_update": {
            "family": "evaluator-only low-rank public-data adaptation",
            "training_split": "target_update_train",
            "objective": "next-token cross-entropy on BOS plus first 39 public source tokens",
            "modules": "q_proj and v_proj in decoder layers 0,1,2,3",
            "rank": 4,
            "alpha": 8.0,
            "scale": 2.0,
            "dropout": 0.0,
            "initialization": "A normal std=0.01; B zeros",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "steps": 40,
            "batch_records": 8,
            "gradient_clip_norm": 1.0,
            "base_parameters": "frozen",
            "decision_time": "preregistered before any blind target observation or truth exists",
            "artifact": {
                "path": "outputs/TRR-0001/evaluator_private/target_lora.safetensors",
                "committed": False,
                "reconstructor_access": False,
            },
        },
        "cut_depths": {
            "evaluated": [0, 4, 8],
            "primary": 4,
            "roles": {
                "0": "embedding-boundary sanity condition",
                "4": "primary shallow cut aligned with the historical comparator",
                "8": "deeper stress condition",
            },
        },
        "baseline_families": {
            "direct_inverse": {
                "family": "positionwise public-data inverse",
                "cut_0": "identity map from observed embedding to normalized public embeddings",
                "cuts_4_and_8": (
                    "independent residual affine 2048-to-2048 maps trained on public-surrogate "
                    "activations to predict normalized input embeddings"
                ),
                "training_split": "inverse_train",
                "loss": "mean one-minus-cosine similarity",
                "optimizer": "AdamW",
                "steps_per_cut": 300,
                "position_batch_size": 512,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "gradient_clip_norm": 1.0,
                "hyperparameter_selection": "fixed here; no blind or development metric selects a variant",
                "candidate_scores": "cosine similarity to every normalized public input embedding",
                "direct_output": "highest-scoring token with token-ID ascending tie break",
                "frozen_candidate_budget": 16,
            },
            "causal_public_surrogate_search": {
                "family": "causal candidate generation and public-surrogate simulation",
                "candidate_source": "the direct inverse's frozen top-16 lists",
                "surrogate": f"{MODEL_ID}@{MODEL_REVISION}",
                "prefix": "declared BOS followed only by already committed reconstructed tokens",
                "score": (
                    "one-minus-cosine between the candidate's public-prefix cut activation "
                    "and the permitted observed activation at that position"
                ),
                "commit": "lowest score; proposal rank then token-ID ascending tie break",
                "candidate_simulations_per_scored_token": 16,
                "cache": "fork public DynamicCache for batched candidates; commit only the winner",
                "target_prefix_calls": 0,
            },
            "common_comparison": {
                "records": "same blind records and observations",
                "candidate_budget": 16,
                "abstention": "none",
                "record_order": "fixed selection order",
                "stopping": "reconstruct all 39 non-BOS positions",
                "cross_record_adaptation": "none",
            },
        },
        "historical_comparator": {
            "status": "RUNNABLE_BUT_INCOMPATIBLE_WITH_DECLARED_CONDITION",
            "reason": (
                "The exact code and public teacher assets are runnable, but the frozen comparator "
                "requires a 128x128 source geometry and a three-episode prior-trace adaptation "
                "contract. Applying it to the 64x40 Pile/LoRA condition would alter its fixed "
                "contract and would not be the exact historical comparator."
            ),
            "action": "do not run it as a scientific arm; retain its verified preflight and audit",
        },
        "execution_order": {
            "pre_truth": [
                "train target update on target_update_train",
                "train inverse states on inverse_train using only the public surrogate",
                "generate evaluator-only target and matched-public observations",
                "run both baseline families for conditions matched_public then unavailable_target_lora",
                "run cuts 0,4,8 and records in fixed order",
                "freeze outputs, candidates, queries, state, configuration, order, and timings",
                "create and verify the immutable freeze receipt",
            ],
            "truth_open": "only after the freeze receipt verifies every declared hash",
            "post_truth": [
                "score frozen outputs",
                "compute exact proposal ranks from frozen query vectors",
                "run preregistered teacher-prefix counterfactual diagnostics without revising outputs",
                "bootstrap records and produce tables/plots",
            ],
        },
        "freeze_contract": {
            "receipt_path": "experiments/TRR-0001/freeze_receipt.json",
            "hashes": [
                "reconstructed token outputs",
                "candidate IDs and scores",
                "frozen inverse query vectors",
                "method state",
                "record order",
                "configuration",
                "routing and stopping decisions",
                "timing records",
            ],
            "mutation_policy": "create-only outputs made read-only; any hash mismatch fails closed",
            "truth_gate": "scoring refuses to open truth until receipt verification succeeds",
            "determinism": "canonical JSON, sorted paths, SHA-256, explicit freeze timestamp",
        },
        "metrics": {
            "primary": "token accuracy excluding BOS and padding at cut 4 in unavailable_target_lora",
            "required": [
                "token accuracy excluding BOS and padding",
                "exact sequence-match rate",
                "correct prefix length and first-error position",
                "coverage and selective accuracy",
                "frozen top-k true-token recall and exact true-token rank",
                "conditional selection accuracy when truth is in the frozen candidates",
                "end-to-end runtime per record and scored token",
                "phase timings including preparation, training, proposal, reconstruction, synchronization, and I/O",
                "peak CPU and GPU memory",
                "candidate simulations or equivalent evaluations",
                "trainable parameter count and persisted state bytes",
                "implementation-complexity summary",
            ],
        },
        "statistics": {
            "unit": "record",
            "confidence_interval": "95% percentile bootstrap",
            "draws": 10000,
            "seed": 1732,
            "paired_differences": (
                "causal minus direct within each record, condition, and cut; "
                "target minus matched control for mismatch"
            ),
            "effect_sizes": "absolute percentage-point quality differences and multiplicative cost ratios",
            "improvement_rule": (
                "claim improvement only for supported quality gain at like-for-like budget, "
                "supported cost reduction at like-for-like quality, or a reported Pareto improvement; "
                "otherwise call the comparison inconclusive"
            ),
        },
        "post_truth_diagnostics": {
            "proposal_vs_ranking": "top-16 recall and causal conditional accuracy",
            "cut_depth": [0, 4, 8],
            "position": "physical non-BOS position 1 through 39",
            "error_propagation": [
                "accuracy before and after each record's first frozen output error",
                "teacher-prefix public-surrogate counterfactual after truth opening",
            ],
            "frequency_bins_from_auxiliary_counts": ["unseen", "1-4", "5-19", "20-or-more"],
            "token_groups": {
                "whitespace_prefixed": "decoded token begins whitespace or raw token begins Ġ",
                "punctuation": "all non-whitespace decoded characters are Unicode punctuation",
                "numeric": "contains a Unicode decimal digit and all other non-whitespace characters are punctuation",
                "other": "all remaining scored tokens",
            },
            "activation_matching": "frozen chosen and candidate score distributions labeled correct/incorrect after truth",
            "target_surrogate_mismatch": "paired unavailable-target versus matched-public differences",
            "output_revision": "forbidden",
        },
        "cost_accounting": {
            "timers": "CUDA-synchronized perf_counter intervals plus UTC phase timestamps",
            "memory": "torch peak allocated/reserved and process max RSS",
            "model_evaluations": "candidate simulations counted exactly; training steps and token examples separate",
            "persisted_state": "byte counts and SHA-256 for inverse and evaluator-only LoRA state",
        },
        "artifacts": {
            "committed": {
                "plan": "experiments/TRR-0001/plan.json",
                "resource_audit": "experiments/TRR-0001/resource_audit.json",
                "freeze_receipt": "experiments/TRR-0001/freeze_receipt.json",
                "manifest": "experiments/TRR-0001/manifest.json",
                "metrics": "experiments/TRR-0001/metrics.json",
                "per_record": "experiments/TRR-0001/per_record_metrics.jsonl",
                "result": "coordination/results/TRR-0001.md",
            },
            "uncommitted_private_root": "outputs/TRR-0001",
            "retention": "retain locally through review; record paths, byte counts, and SHA-256",
        },
        "claim_scope": {
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "target_update": "the exact evaluator-only rank-4 LoRA artifact produced by this plan",
            "surrogate": f"{MODEL_ID}@{MODEL_REVISION}",
            "cuts": [0, 4, 8],
            "dataset": f"{DATASET_ID}@{DATASET_REVISION}",
            "sequence_tokens": SEQUENCE_TOKENS,
            "candidate_budget": 16,
            "methods": ["direct_inverse", "causal_public_surrogate_search"],
            "generalization_beyond_scope": "not supported",
        },
        "deviation_policy": {
            "material_changes": (
                "record what changed, why, decision time, whether pre-truth, and effects on "
                "comparability or claim scope"
            ),
            "blind_truth_retuning": "forbidden",
            "current_deviations": [],
        },
    }


def build_resource_audit(created_utc: str) -> dict[str, Any]:
    home = Path.home()
    model = home / ".cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots" / MODEL_REVISION
    dataset = home / ".cache/huggingface/hub/datasets--NeelNanda--pile-10k/snapshots" / DATASET_REVISION
    model_files = {
        "config.json": (877, "2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb"),
        "generation_config.json": (189, "88effbb63300dbbc7390143fbbdd9d9fa50587b37e8bfd16c8c90d4970a74a36"),
        "model.safetensors": (2471645608, "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"),
        "special_tokens_map.json": (296, "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec"),
        "tokenizer.json": (9085657, "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4"),
        "tokenizer_config.json": (54528, "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e"),
    }
    for name, (size, digest) in model_files.items():
        require_file(model / name, size, digest)

    parquet = dataset / "data/train-00000-of-00001-4746b8785c874cc7.parquet"
    require_file(
        parquet,
        digest="a1a9475a8684ac8f1b17a36eccb2ec49c127edd7aae9beb2f240726972d93f31",
    )
    require_file(ROOT / "coordination/requests/TRR-0001.md", 24375, REQUEST_SHA256)
    require_file(ROOT / "RESEARCH_CHARTER.md", digest=CHARTER_SHA256)

    lens = Path("/home/alanz/spartan/punim2939/backdoor_lora/ersoy2026/inversion_20260730/out/lens_alpaca.pt")
    teacher_identity = Path(
        "/home/alanz/spartan/punim2939/backdoor_lora/ersoy2026/research/"
        "adaptive_a1_a2_strict_bos_20260817_goal_01a00b08/audit/"
        "AUDIT-0004-public-teacher-identity-v2.json"
    )
    boundary = Path(
        "/mnt/c/Backdoor_LoRA/Stage2Trace/20260810/trace_staging/"
        "chunk_000051_000100/boundary.safetensors"
    )
    historical_truth = Path(
        "/mnt/c/Backdoor_LoRA/Stage2Trace/20260810/trace_staging/"
        "chunk_000051_000100/oracle_tokens.safetensors"
    )
    require_file(lens, 16787653, "33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742")
    require_file(teacher_identity, 1247, "fd457c6e33f18b9341f9699ebcd204aa5994423cef373affb99bcbdf9c621061")
    require_file(boundary, 10079438000)
    require_file(historical_truth, 6554392)

    return {
        "schema": "token-reconstruction.trr0001.resource-audit.v1",
        "task_id": TASK_ID,
        "created_utc": created_utc,
        "environment": {
            "python": "3.12.3",
            "torch": "2.10.0+cu128",
            "torch_cuda": "12.8",
            "transformers": "5.3.0",
            "safetensors": "0.7.0",
            "datasets": "4.8.3",
            "numpy": "1.26.4",
            "pytest": "8.4.2",
            "gpu": "NVIDIA GeForce RTX 5080",
            "gpu_driver": "610.88",
            "gpu_total_memory_mib": 16303,
        },
        "primary_model": {
            "status": "AVAILABLE_AND_LOADABLE_OFFLINE",
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot_path": str(model),
            "files": {
                name: {"bytes": size, "sha256": digest}
                for name, (size, digest) in model_files.items()
            },
            "observed": {
                "layers": 16,
                "hidden_size": 2048,
                "vocabulary_size": 128256,
                "bos_token_id": 128000,
                "tokenizer_size": 128256,
            },
        },
        "dataset": {
            "status": "AVAILABLE_AND_LOADABLE_OFFLINE",
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "snapshot_path": str(dataset),
            "rows": 10000,
            "features": ["text", "meta"],
            "license": "bigscience-bloom-rail-1.0",
            "parquet_path": str(parquet),
            "parquet_bytes": parquet.stat().st_size,
            "parquet_sha256": sha256_file(parquet),
            "arrow_cache_sha256": "77ddf02e2a69373a944bc8bc8ac8f7b9926f5c62203d727341a24d709bf81113",
        },
        "historical_comparator": {
            "status": "RUNNABLE_BUT_INCOMPATIBLE_WITH_DECLARED_CONDITION",
            "exact_public_assets": {
                "lens": {
                    "path": str(lens),
                    "bytes": lens.stat().st_size,
                    "sha256": sha256_file(lens),
                },
                "teacher_identity": {
                    "path": str(teacher_identity),
                    "bytes": teacher_identity.stat().st_size,
                    "sha256": sha256_file(teacher_identity),
                },
                "model": f"{MODEL_ID}@{MODEL_REVISION}",
            },
            "historical_private_assets": {
                "boundary": {
                    "path": str(boundary),
                    "bytes": boundary.stat().st_size,
                    "expected_sha256_from_historical_protocol": "c2db96da30f8792bdcff2b5b8987faa56de1123241e6f3e2953b3c5223994aaa",
                    "verification": "existence and exact byte count checked; content not opened",
                },
                "truth_sidecar": {
                    "path": str(historical_truth),
                    "bytes": historical_truth.stat().st_size,
                    "expected_sha256_from_historical_protocol": "a47ace7fed712551e41bb2b3b767122686154ff40ea44e72bdf509540ebe6736",
                    "verification": "existence and exact byte count checked; content not opened",
                },
            },
            "public_teacher_identity_check": {
                "command": (
                    "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                    "python3 -c '<load reference/strict_bos round001_teacher.load_public_teacher "
                    "with pinned spec, lens, and AUDIT-0004 identity>'"
                ),
                "exit_status": 0,
                "output": [
                    "STRICT_BOS_PUBLIC_TEACHER_IDENTITY=VERIFIED",
                    "PREFIX_SHA256=3c897c5ad27073c85496ca24a4a11f2c37de1e5695f65535d7ceeb68e3f1c9ba",
                    "LENS_STATE_SHA256=98cd91c2dc4c5d8ac5a827672c027b7e6dad0d1acb3c45e387d118c9ad2ead04",
                ],
                "elapsed_seconds": 6.52,
                "max_rss_kib": 3370776,
            },
            "synthetic_preflight": {
                "command": (
                    "python3 reference/strict_bos/preflight_wavefront.py "
                    "--output /tmp/trr0001-strict-bos-preflight.json"
                ),
                "exit_status": 0,
                "output_bytes": 1628,
                "output_sha256": "9ff64cac95efd0f39779a7eac53db3e582cca1c002d577ac28a8c16127e99947",
                "checks_passed": True,
                "elapsed_seconds": 1.18,
                "max_rss_kib": 1033776,
            },
            "incompatibility": [
                "fixed 128x128 source geometry versus TRR-0001's 64 records of 40 tokens",
                "fixed three-episode completed-record adapter schedule tied to historical source indices",
                "historical target traces and target-prefix construction differ from the declared Pile/LoRA primary condition",
                "changing those contracts would no longer run the exact frozen comparator",
            ],
            "decision": "retain audit and preflight; do not include as a scientific comparison arm",
        },
        "merge_evidence": {
            "operation": "GitHub REST PUT /repos/A-lan-Z/Token-Reconstruction-Research/pulls/1/merge",
            "payload": {
                "sha": "b087365766f00432077476bf32a6afdf2e854841",
                "merge_method": "merge",
                "commit_title": "Merge pull request #1 from A-lan-Z/task/TRR-0000",
                "commit_message": "[TRR-0000] Bootstrap and validate research relay",
            },
            "exit_status": 0,
            "merged": True,
            "merged_at_utc": "2026-08-22T08:26:27Z",
            "merge_commit": "0f641e5f071dd38331d2e2b7821d40fc74941c2e",
            "accepted_head_ancestor_check": {
                "command": (
                    "git merge-base --is-ancestor "
                    "b087365766f00432077476bf32a6afdf2e854841 refs/remotes/origin/main"
                ),
                "exit_status": 0,
            },
            "pull_request_state": "MERGED",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "experiments/TRR-0001/plan.json"
    )
    parser.add_argument(
        "--resource-audit",
        type=Path,
        default=ROOT / "experiments/TRR-0001/resource_audit.json",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    dataset = load_dataset(
        DATASET_ID, revision=DATASET_REVISION, split="train"
    )

    def clipped_token_length(text: str) -> int:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=SOURCE_TOKENS,
        )
        return len(encoded["input_ids"])

    splits = select_record_splits(
        (row["text"] for row in dataset),
        token_length=clipped_token_length,
        dataset_revision=DATASET_REVISION,
        seed=SELECTION_SEED,
        minimum_tokens=SOURCE_TOKENS,
        split_sizes=SPLIT_SIZES,
    )
    flat = [record.index for records in splits.values() for record in records]
    if len(flat) != len(set(flat)):
        raise RuntimeError("record splits overlap")

    write_json_exclusive(args.plan, build_plan(args.created_utc, splits))
    write_json_exclusive(args.resource_audit, build_resource_audit(args.created_utc))
    print(f"PLAN={args.plan}")
    print(f"PLAN_SHA256={sha256_file(args.plan)}")
    print(f"RESOURCE_AUDIT={args.resource_audit}")
    print(f"RESOURCE_AUDIT_SHA256={sha256_file(args.resource_audit)}")
    for name, records in splits.items():
        print(f"{name.upper()}_COUNT={len(records)}")
        print(f"{name.upper()}_IDS_SHA256={record_ids_sha256(records)}")


if __name__ == "__main__":
    main()
