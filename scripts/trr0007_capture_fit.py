#!/usr/bin/env python3
"""TRR-0007 public P0 capture preflight and post-capture verifier.

The registered TRR-0005 producer already accepts the TRR-0007 materialized
corpus plan and complete token artifact.  This small adapter keeps the model
forward in that producer and supplies two CPU-only guards around it:

* ``preflight`` binds the support plan, token rows, TRR5 source artifacts,
  public model snapshot, geometry, and the exact capture command;
* ``verify`` checks the captured artifact and requires bit-exact activations on
  all 1,080 natural slots retained from the TRR-0005 enriched bank.

No model forward, private truth, target checkpoint, or source-text read occurs
in this file.  Capture remains an explicit invocation of
``scripts/trr0005_prepare_public_activations.py`` at fixed batch 8 x 192 and
cut depth 4.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from typing import Any

import torch
from safetensors import safe_open

# Keep the hash and padded-token validation definitions identical to the
# registered producer's public activation module.
from token_reconstruction.public_activation import tensor_sha256


TASK_ID = "TRR-0007"
TRR5_TASK_ID = "TRR-0005"
PLAN_SCHEMA = "token-reconstruction.trr0005-public-corpus-plan.v1"
SUPPORT_SCHEMA = "token-reconstruction.trr0007-public-broader-bank.v1"
PRODUCER_RELATIVE = "scripts/trr0005_prepare_public_activations.py"
PRODUCER_SHA256 = "e7d3f82df8d9f1c68eecdd7c077112eafc555acde25955549cb34836bd3b2322"
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
VOCAB_SIZE = 128256
RECORDS = 1200
WIDTH = 192
HIDDEN = 2048
POST_BOS = 124371
VALIDATION_RECORDS = 48
VALIDATION_POST_BOS = 3133
BATCH_RECORDS = 8
CUT_DEPTH = 4
FIT_STEPS = 3000
FIT_BATCH = 512
FIT_SEED = 4005
NATURAL_ROWS = 1080
CONTROLLED_ROWS = 120
EXPECTED_LENGTH_DIGEST = "b8b3392d6984e7e109ad70108cb36aaa861555eccce4b4bf14e3aea5c2846bb8"
EXPECTED_NATURAL_INDEX_HASH = "468e3a01f805086151588ffbf0547fe97fbb1268265f6d7f1fd05630e1b12877"
EXPECTED_CONTROLLED_INDEX_HASH = "808a63e652361d9acd403651a64786da3260c04e16c68174192b26e1a542a0e8"
EXPECTED_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
EXPECTED_MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
EXPECTED_MODEL_CONFIG_SHA256 = "2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb"
EXPECTED_MODEL_GENERATION_SHA256 = "88effbb63300dbbc7390143fbbdd9d9fa50587b37e8bfd16c8c90d4970a74a36"
EXPECTED_MODEL_WEIGHTS_SHA256 = "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"
EXPECTED_MODEL_WEIGHTS_BYTES = 2471645608
TRR5_CAPTURE_WALL_SECONDS = 11.447659793004277
TRR5_CAPTURE_PEAK_ALLOCATED_BYTES = 2472678400
TRR5_CAPTURE_PEAK_RESERVED_BYTES = 2480930816
TRR5_CAPTURE_PEAK_HOST_RSS_BYTES = 6063869952


class CaptureContractError(RuntimeError):
    """Raised when capture inputs or post-capture equality are not accepted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_ints(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(f"{int(value)}\n".encode("utf-8"))
    return digest.hexdigest()


def _regular(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise CaptureContractError(f"{label} must be a regular file: {path}")
    return path


def _json(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureContractError(f"cannot parse {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise CaptureContractError(f"{label} must be a JSON object")
    return dict(value)


def _file_record(path: Path, *, label: str, hash_bytes: bool = True) -> dict[str, Any]:
    path = _regular(path, label=label)
    record: dict[str, Any] = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
    }
    if hash_bytes:
        record["sha256"] = _sha256_file(path)
    return record


def _header(path: Path, key: str, *, label: str) -> dict[str, Any]:
    path = _regular(path, label=label)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if key not in keys:
                raise CaptureContractError(f"{label} has no {key!r} tensor")
            view = handle.get_slice(key)
            result = {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
                "tensor_key": key,
                "shape": list(view.get_shape()),
                "dtype": str(view.get_dtype()),
                "keys": sorted(keys),
            }
    except CaptureContractError:
        raise
    except Exception as exc:
        raise CaptureContractError(f"cannot inspect {label}: {path}") from exc
    return result


def _tensor(path: Path, key: str, *, label: str) -> torch.Tensor:
    path = _regular(path, label=label)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if key not in set(handle.keys()):
                raise CaptureContractError(f"{label} has no {key!r} tensor")
            return handle.get_tensor(key).contiguous()
    except CaptureContractError:
        raise
    except Exception as exc:
        raise CaptureContractError(f"cannot load {label}: {path}") from exc


def _resolve(value: str | Path, *, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _default_paths(root: Path) -> dict[str, Path]:
    return {
        "corpus_plan": root / "experiments/TRR-0007/support/broader_bank_v1/corpus_plan.json",
        "bank_receipt": root / "experiments/TRR-0007/support/broader_bank_v1/bank_construction_receipt.json",
        "constructed_tokens": root / "experiments/TRR-0007/support/broader_bank_v1/constructed_public_tokens.safetensors",
        "current_tokens": root / "../TRR-0005/experiments/TRR-0005/corpus/coverage_mix_v1/constructed_public_tokens.safetensors",
        "current_activations": root / "../TRR-0005/outputs/TRR-0005/enriched_fit_cut4.safetensors",
        "original_artifact": root / "../TRR-0004/outputs/TRR-0004/public_activation_v2/train_large_cut4.safetensors",
        "original_records": root / "../TRR-0004/experiments/TRR-0004/fit/adapter_v2/affine_fit_records.json",
        "validation_artifact": root / "../TRR-0004/experiments/TRR-0004/fit/adapter_v2/validation_mixed_cut4.safetensors",
        "validation_records": root / "../TRR-0004/experiments/TRR-0004/fit/adapter_v2/affine_validation_records.json",
        "embedding_table": root / "../../outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors",
        "model": Path("/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"),
        "capture_root": root / "experiments/TRR-0007/support/broader_capture_v1",
        "preflight_receipt": root / "experiments/TRR-0007/support/broader_capture_v1_preflight.json",
    }


def _load_plan(path: Path) -> dict[str, Any]:
    plan = _json(path, label="TRR-0007 support corpus plan")
    if plan.get("schema") != PLAN_SCHEMA or plan.get("task_id") != TRR5_TASK_ID:
        raise CaptureContractError("support corpus plan must retain the accepted TRR-0005 plan schema")
    if plan.get("status") != "PREPARED_PUBLIC_DATA_NO_MODEL_FORWARD":
        raise CaptureContractError("support corpus plan status changed from the no-forward preparation contract")
    design = plan.get("design")
    if not isinstance(design, Mapping):
        raise CaptureContractError("support corpus plan has no design")
    expected_design = {
        "record_count": RECORDS,
        "stored_rows_including_bos": RECORDS + POST_BOS,
        "post_bos_positions": POST_BOS,
        "max_sequence_length": WIDTH,
        "length_vector_digest": EXPECTED_LENGTH_DIGEST,
    }
    for key, expected in expected_design.items():
        if design.get(key) != expected:
            raise CaptureContractError(f"support corpus design changed at {key}: {design.get(key)!r}")
    exposure = plan.get("joint_training_exposure")
    if not isinstance(exposure, Mapping):
        raise CaptureContractError("support corpus plan has no joint training exposure")
    for key, expected in {
        "batch_size": FIT_BATCH,
        "steps": FIT_STEPS,
        "seed": FIT_SEED,
        "post_bos_positions": POST_BOS,
    }.items():
        if int(exposure.get(key, -1)) != expected:
            raise CaptureContractError(f"support exposure changed at {key}")
    support = plan.get("trr0007_support")
    if not isinstance(support, Mapping) or support.get("schema") != SUPPORT_SCHEMA:
        raise CaptureContractError("support corpus plan has no TRR-0007 broader-bank binding")
    matched = support.get("matched_geometry")
    if not isinstance(matched, Mapping):
        raise CaptureContractError("support corpus plan has no matched geometry")
    if matched.get("length_vector_digest") != EXPECTED_LENGTH_DIGEST:
        raise CaptureContractError("support matched length digest changed")
    natural = support.get("natural_slot_indices")
    controlled = support.get("controlled_slot_indices")
    if not isinstance(natural, list) or len(natural) != NATURAL_ROWS:
        raise CaptureContractError("support natural slot count changed")
    if not isinstance(controlled, list) or len(controlled) != CONTROLLED_ROWS:
        raise CaptureContractError("support controlled slot count changed")
    natural_ints = [int(value) for value in natural]
    controlled_ints = [int(value) for value in controlled]
    if natural_ints != sorted(set(natural_ints)) or controlled_ints != sorted(set(controlled_ints)):
        raise CaptureContractError("support slot indices are not sorted and unique")
    if set(natural_ints) | set(controlled_ints) != set(range(RECORDS)):
        raise CaptureContractError("support natural/controlled slots do not partition 1,200 rows")
    if _sha256_ints(natural_ints) != EXPECTED_NATURAL_INDEX_HASH:
        raise CaptureContractError("support natural slot index hash changed")
    if _sha256_ints(controlled_ints) != EXPECTED_CONTROLLED_INDEX_HASH:
        raise CaptureContractError("support controlled slot index hash changed")
    arms = plan.get("arms")
    if not isinstance(arms, Mapping):
        raise CaptureContractError("support corpus plan has no arms")
    coverage = arms.get("coverage_mix_v1")
    if not isinstance(coverage, Mapping):
        raise CaptureContractError("support corpus plan has no coverage_mix_v1 arm")
    rows = coverage.get("records")
    if not isinstance(rows, list) or len(rows) != RECORDS:
        raise CaptureContractError("support coverage arm record count changed")
    for slot, row in enumerate(rows):
        if not isinstance(row, Mapping) or int(row.get("slot", -1)) != slot:
            raise CaptureContractError(f"support coverage record ordering changed at slot {slot}")
        if not isinstance(row.get("record_id"), str) or not row["record_id"]:
            raise CaptureContractError(f"support coverage record {slot} has no record ID")
        target = int(row.get("target_post_bos_token_count", -1))
        if target != int(row.get("target_full_token_count", -1)) - 1 or not 31 <= target < WIDTH:
            raise CaptureContractError(f"support coverage record {slot} has invalid target length")
    token_entry = coverage.get("token_artifact")
    if not isinstance(token_entry, Mapping):
        raise CaptureContractError("support coverage arm has no token artifact descriptor")
    return plan


def _load_tokens(path: Path, *, label: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    token_ids = _tensor(path, "token_ids", label=f"{label} token IDs")
    mask = _tensor(path, "attention_mask", label=f"{label} attention mask")
    if tuple(token_ids.shape) != (RECORDS, WIDTH) or tuple(mask.shape) != (RECORDS, WIDTH):
        raise CaptureContractError(f"{label} token/mask geometry changed")
    if token_ids.dtype not in (torch.int32, torch.int64) or mask.dtype != torch.uint8:
        raise CaptureContractError(f"{label} token/mask dtypes changed")
    if token_ids[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise CaptureContractError(f"{label} rows lost BOS")
    if token_ids.lt(0).any().item() or token_ids.ge(VOCAB_SIZE).any().item():
        raise CaptureContractError(f"{label} token IDs escaped the public vocabulary")
    if mask.lt(0).any().item() or mask.gt(1).any().item():
        raise CaptureContractError(f"{label} mask is not binary")
    lengths: list[int] = []
    for row in range(RECORDS):
        active = int(mask[row].sum().item())
        if active <= 1 or active > WIDTH:
            raise CaptureContractError(f"{label} row {row} has invalid active length")
        if not mask[row, :active].eq(1).all().item() or not mask[row, active:].eq(0).all().item():
            raise CaptureContractError(f"{label} row {row} mask is not contiguous right-padding")
        if not token_ids[row, active:].eq(PAD_TOKEN_ID).all().item():
            raise CaptureContractError(f"{label} row {row} padding labels changed")
        lengths.append(active - 1)
    digest = hashlib.sha256()
    for slot, length in enumerate(lengths):
        digest.update(f"{slot}\t{length}\n".encode("utf-8"))
    info = {
        "shape": [RECORDS, WIDTH],
        "token_dtype": str(token_ids.dtype),
        "mask_dtype": str(mask.dtype),
        "token_ids_sha256": tensor_sha256(token_ids),
        "attention_mask_sha256": tensor_sha256(mask),
        "length_vector_digest": digest.hexdigest(),
        "post_bos_positions": int(sum(lengths)),
    }
    return token_ids, mask, info


def _natural_token_equivalence(
    support_tokens: torch.Tensor,
    support_mask: torch.Tensor,
    current_tokens: torch.Tensor,
    current_mask: torch.Tensor,
    natural_slots: Sequence[int],
) -> dict[str, Any]:
    if not torch.equal(support_mask, current_mask):
        raise CaptureContractError("support and current masks differ")
    mismatched_tokens = [slot for slot in natural_slots if not torch.equal(support_tokens[slot], current_tokens[slot])]
    mismatched_masks = [slot for slot in natural_slots if not torch.equal(support_mask[slot], current_mask[slot])]
    if mismatched_tokens or mismatched_masks:
        raise CaptureContractError(
            f"natural token/mask equivalence failed: token rows={mismatched_tokens[:5]}, mask rows={mismatched_masks[:5]}"
        )
    return {
        "status": "PASS",
        "natural_rows": len(natural_slots),
        "natural_slot_indices_sha256": _sha256_ints(natural_slots),
        "token_rows_equal": len(mismatched_tokens) == 0,
        "mask_rows_equal": len(mismatched_masks) == 0,
        "all_masks_equal_current": True,
        "support_token_ids_sha256": tensor_sha256(support_tokens),
        "current_token_ids_sha256": tensor_sha256(current_tokens),
        "support_attention_mask_sha256": tensor_sha256(support_mask),
        "current_attention_mask_sha256": tensor_sha256(current_mask),
    }


def _model_file(path: Path, *, label: str) -> Path:
    """Accept the Hugging Face snapshot symlink while binding its target."""

    path = path.expanduser()
    if not path.is_file():
        raise CaptureContractError(f"{label} is unavailable: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise CaptureContractError(f"{label} target is unavailable: {path}")
    return resolved


def _model_file_record(path: Path, *, label: str, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = _model_file(path, label=label)
    record: dict[str, Any] = {
        "path": str(path),
        "resolved_path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "symlink": bool(path.is_symlink()),
    }
    if expected_sha256 is not None:
        record["sha256_expected"] = expected_sha256
    else:
        record["sha256"] = _sha256_file(resolved)
    return record


def _model_binding(model: Path) -> dict[str, Any]:
    model = model.expanduser().resolve()
    if not model.is_dir():
        raise CaptureContractError(f"public model snapshot is unavailable: {model}")
    config = _model_file(model / "config.json", label="public model config")
    generation = _model_file(model / "generation_config.json", label="public generation config")
    weights = _model_file(model / "model.safetensors", label="public model weights")
    config_hash = _sha256_file(config)
    generation_hash = _sha256_file(generation)
    if config_hash != EXPECTED_MODEL_CONFIG_SHA256:
        raise CaptureContractError("public model config hash changed")
    if generation_hash != EXPECTED_MODEL_GENERATION_SHA256:
        raise CaptureContractError("public generation config hash changed")
    if int(weights.stat().st_size) != EXPECTED_MODEL_WEIGHTS_BYTES:
        raise CaptureContractError("public model weight byte count changed")
    return {
        "model_id": EXPECTED_MODEL_ID,
        "revision": EXPECTED_MODEL_REVISION,
        "snapshot_path": str(model),
        "config": _model_file_record(model / "config.json", label="public model config", expected_sha256=EXPECTED_MODEL_CONFIG_SHA256),
        "generation_config": _model_file_record(model / "generation_config.json", label="public generation config", expected_sha256=EXPECTED_MODEL_GENERATION_SHA256),
        "weights": {
            **_model_file_record(model / "model.safetensors", label="public model weights", expected_sha256=EXPECTED_MODEL_WEIGHTS_SHA256),
            "hash_scope": "full model.safetensors; producer verifies during load",
        },
    }


def _source_code_binding(root: Path) -> dict[str, Any]:
    paths = [
        root / PRODUCER_RELATIVE,
        root / "src/token_reconstruction/public_activation.py",
        root / "src/token_reconstruction/public_prefix.py",
        root / "src/token_reconstruction/trr0005_public_corpus.py",
    ]
    return {
        str(path.relative_to(root)): _file_record(path, label="capture source")
        for path in paths
    }


def _require_header(descriptor: Mapping[str, Any], *, shape: Sequence[int], dtype: str, label: str) -> None:
    if descriptor.get("shape") != list(shape) or descriptor.get("dtype") != dtype:
        raise CaptureContractError(
            f"{label} geometry/dtype changed: shape={descriptor.get('shape')!r}, dtype={descriptor.get('dtype')!r}"
        )


def _resource_binding(paths: Mapping[str, Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    result["constructed_tokens"] = {
        **_header(paths["constructed_tokens"], "token_ids", label="constructed public tokens"),
        "attention_mask": _header(paths["constructed_tokens"], "attention_mask", label="constructed public mask"),
    }
    result["current_tokens"] = {
        **_header(paths["current_tokens"], "token_ids", label="current enriched tokens"),
        "attention_mask": _header(paths["current_tokens"], "attention_mask", label="current enriched mask"),
    }
    result["current_activations"] = {
        **_header(paths["current_activations"], "activations", label="current enriched activations"),
        "token_ids": _header(paths["current_activations"], "token_ids", label="current enriched labels"),
        "attention_mask": _header(paths["current_activations"], "attention_mask", label="current enriched mask"),
    }
    result["original_artifact"] = {
        **_header(paths["original_artifact"], "activations", label="TRR4 original activations"),
        "token_ids": _header(paths["original_artifact"], "token_ids", label="TRR4 original labels"),
        "attention_mask": _header(paths["original_artifact"], "attention_mask", label="TRR4 original mask"),
    }
    result["original_records"] = _file_record(paths["original_records"], label="TRR4 original records")
    result["validation_artifact"] = {
        **_header(paths["validation_artifact"], "activations", label="common validation activations"),
        "token_ids": _header(paths["validation_artifact"], "token_ids", label="common validation labels"),
        "attention_mask": _header(paths["validation_artifact"], "attention_mask", label="common validation mask"),
    }
    result["validation_records"] = _file_record(paths["validation_records"], label="common validation records")
    result["embedding_table"] = _header(paths["embedding_table"], "embeddings", label="normalized public embedding table")
    return result


def _expected_capture_command(root: Path, paths: Mapping[str, Path]) -> list[str]:
    return [
        "env",
        "PYTHONPATH=.:src:scripts",
        "OMP_NUM_THREADS=8",
        "MKL_NUM_THREADS=8",
        "OPENBLAS_NUM_THREADS=8",
        "python3",
        PRODUCER_RELATIVE,
        "--mode", "capture",
        "--corpus-plan", os.path.relpath(paths["corpus_plan"], root),
        "--original-artifact", os.path.relpath(paths["original_artifact"], root),
        "--original-records", os.path.relpath(paths["original_records"], root),
        "--common-validation-artifact", os.path.relpath(paths["validation_artifact"], root),
        "--common-validation-records", os.path.relpath(paths["validation_records"], root),
        "--embedding-table", os.path.relpath(paths["embedding_table"], root),
        "--constructed-token-artifact", os.path.relpath(paths["constructed_tokens"], root),
        "--output-root", os.path.relpath(paths["capture_root"], root),
        "--enriched-activation-artifact", os.path.relpath(paths["capture_root"] / "enriched_fit_cut4.safetensors", root),
        "--model", str(paths["model"]),
        "--device", "cuda",
        "--batch-records", str(BATCH_RECORDS),
        "--cut-depth", str(CUT_DEPTH),
        "--min-free-gpu-gib", "8",
        "--max-reserved-gpu-gib", "8",
        "--max-host-rss-gib", "16",
    ]


def _expected_verify_command(root: Path, paths: Mapping[str, Path]) -> list[str]:
    return [
        "env",
        "PYTHONPATH=.:src",
        "python3",
        "scripts/trr0007_capture_fit.py",
        "--mode", "verify",
        "--corpus-plan", os.path.relpath(paths["corpus_plan"], root),
        "--constructed-tokens", os.path.relpath(paths["constructed_tokens"], root),
        "--current-tokens", os.path.relpath(paths["current_tokens"], root),
        "--current-activations", os.path.relpath(paths["current_activations"], root),
        "--capture-artifact", os.path.relpath(paths["capture_root"] / "enriched_fit_cut4.safetensors", root),
        "--receipt", os.path.relpath(paths["capture_root"] / "capture_verification_receipt.json", root),
    ]


def _base_paths(args: argparse.Namespace, root: Path) -> dict[str, Path]:
    defaults = _default_paths(root)
    names = (
        "corpus_plan", "bank_receipt", "constructed_tokens", "current_tokens",
        "current_activations", "original_artifact", "original_records",
        "validation_artifact", "validation_records", "embedding_table", "model",
        "capture_root",
    )
    result = {}
    for name in names:
        value = getattr(args, name, None)
        result[name] = _resolve(value, root=root) if value is not None else defaults[name].resolve()
    result["preflight_receipt"] = (
        _resolve(args.receipt, root=root) if args.receipt is not None else defaults["preflight_receipt"].resolve()
    )
    return result


def _preflight(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    paths = _base_paths(args, root)
    started = _utc_now()
    plan = _load_plan(paths["corpus_plan"])
    bank_receipt = _json(paths["bank_receipt"], label="support bank construction receipt")
    if bank_receipt.get("schema") != SUPPORT_SCHEMA or bank_receipt.get("status") != "METADATA_BANK_READY_FOR_REAL_P0_CAPTURE":
        raise CaptureContractError("support bank receipt is not ready for real P0 capture")
    capture_contract = bank_receipt.get("capture_contract")
    if not isinstance(capture_contract, Mapping) or capture_contract.get("producer") != PRODUCER_RELATIVE:
        raise CaptureContractError("support bank producer binding changed")
    if capture_contract.get("required") != "one real P0/public-prefix forward over all 1,200 complete constructed sequences; no activation splicing":
        raise CaptureContractError("support bank capture requirement changed")
    support_outputs = bank_receipt.get("outputs")
    if not isinstance(support_outputs, Mapping):
        raise CaptureContractError("support bank receipt has no outputs")
    declared_plan_hash = support_outputs.get("corpus_plan", {}).get("sha256") if isinstance(support_outputs.get("corpus_plan"), Mapping) else None
    if declared_plan_hash != _sha256_file(paths["corpus_plan"]):
        raise CaptureContractError("support bank receipt no longer binds the supplied corpus plan")
    token_output = support_outputs.get("token_artifact")
    declared_token_hash = None
    if isinstance(token_output, Mapping):
        declared_token_hash = token_output.get("sha256", token_output.get("content_hash"))
    if declared_token_hash != _sha256_file(paths["constructed_tokens"]):
        raise CaptureContractError("support bank receipt no longer binds the supplied token artifact")
    coverage_rows = plan["arms"]["coverage_mix_v1"]["records"]
    natural_slots = [int(value) for value in plan["trr0007_support"]["natural_slot_indices"]]
    support_tokens, support_mask, support_info = _load_tokens(paths["constructed_tokens"], label="support constructed tokens")
    current_tokens, current_mask, current_info = _load_tokens(paths["current_tokens"], label="current enriched tokens")
    if support_info["length_vector_digest"] != EXPECTED_LENGTH_DIGEST or current_info["length_vector_digest"] != EXPECTED_LENGTH_DIGEST:
        raise CaptureContractError("support/current length vector changed")
    natural_equivalence = _natural_token_equivalence(
        support_tokens, support_mask, current_tokens, current_mask, natural_slots
    )
    current_h = _header(paths["current_activations"], "activations", label="current enriched activations")
    if current_h["shape"] != [RECORDS, WIDTH, HIDDEN] or current_h["dtype"] != "BF16":
        raise CaptureContractError("current enriched activation geometry/dtype changed")
    resources = _resource_binding(paths)
    _require_header(resources["current_activations"], shape=(RECORDS, WIDTH, HIDDEN), dtype="BF16", label="current enriched activations")
    _require_header(resources["original_artifact"], shape=(RECORDS, WIDTH, HIDDEN), dtype="BF16", label="TRR4 original activations")
    _require_header(resources["validation_artifact"], shape=(VALIDATION_RECORDS, WIDTH, HIDDEN), dtype="BF16", label="common validation activations")
    _require_header(resources["embedding_table"], shape=(VOCAB_SIZE, HIDDEN), dtype="F32", label="normalized public embedding table")
    _require_header(resources["validation_artifact"]["token_ids"], shape=(VALIDATION_RECORDS, WIDTH), dtype="I32", label="common validation labels")
    _require_header(resources["validation_artifact"]["attention_mask"], shape=(VALIDATION_RECORDS, WIDTH), dtype="U8", label="common validation mask")
    model = _model_binding(paths["model"])
    producer = _file_record(root / PRODUCER_RELATIVE, label="TRR5 capture producer")
    if producer["sha256"] != PRODUCER_SHA256:
        raise CaptureContractError("TRR5 capture producer source hash changed")
    if paths["capture_root"].exists() or paths["capture_root"].is_symlink():
        raise CaptureContractError(f"capture output root must not exist before lease: {paths['capture_root']}")
    forecast = {
        "basis": "retained TRR5 capture receipt at identical model, cut, batch, and 1,200x192 geometry",
        "prior_capture_wall_seconds": TRR5_CAPTURE_WALL_SECONDS,
        "prior_peak_cuda_allocated_bytes": TRR5_CAPTURE_PEAK_ALLOCATED_BYTES,
        "prior_peak_cuda_reserved_bytes": TRR5_CAPTURE_PEAK_RESERVED_BYTES,
        "prior_peak_host_rss_bytes": TRR5_CAPTURE_PEAK_HOST_RSS_BYTES,
        "expected_output_bytes": 947176760,
        "live_guard": {
            "minimum_free_gpu_bytes": 8 * 1024**3,
            "maximum_reserved_gpu_bytes": 8 * 1024**3,
            "maximum_host_rss_bytes": 16 * 1024**3,
            "failure_policy": "stop and preserve failure on OOM, non-finite activation, allocator/driver anomaly, or source mismatch",
        },
        "qualification": "producer performs fixed batch-8 x 192 active-output bit-equivalence qualification before full capture",
    }
    command = _expected_capture_command(root, paths)
    verify_command = _expected_verify_command(root, paths)
    receipt = {
        "schema": "token-reconstruction.trr0007-p0-capture-preflight.v1",
        "task_id": TASK_ID,
        "status": "PASS_CPU_PREFLIGHT_READY_FOR_ROOT_GPU_LEASE",
        "execution": {
            "started_utc": started,
            "ended_utc": _utc_now(),
            "git_commit": _git_commit(root),
            "python": sys.version,
            "platform": platform.platform(),
            "resource_usage": {
                "user_cpu_seconds": float(resource.getrusage(resource.RUSAGE_SELF).ru_utime),
                "system_cpu_seconds": float(resource.getrusage(resource.RUSAGE_SELF).ru_stime),
                "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            },
        },
        "input_contract": {
            "plan_schema": PLAN_SCHEMA,
            "plan_status": plan["status"],
            "coverage_arm": "coverage_mix_v1",
            "record_count": RECORDS,
            "stored_width": WIDTH,
            "post_bos_positions": POST_BOS,
            "natural_rows": NATURAL_ROWS,
            "controlled_rows": CONTROLLED_ROWS,
            "natural_slot_indices_sha256": EXPECTED_NATURAL_INDEX_HASH,
            "controlled_slot_indices_sha256": EXPECTED_CONTROLLED_INDEX_HASH,
            "real_p0_required": True,
            "activation_splicing_allowed": False,
            "support_rows_contract": "arms.coverage_mix_v1.records plus token_artifact {token_ids, attention_mask}; rows ordered by slot 0..1199; target lengths must match mask",
        },
        "sources": {
            "corpus_plan": _file_record(paths["corpus_plan"], label="support corpus plan"),
            "bank_receipt": _file_record(paths["bank_receipt"], label="support bank receipt"),
            "constructed_tokens": resources["constructed_tokens"],
            "current_tokens": resources["current_tokens"],
            "current_activations": resources["current_activations"],
            "original_artifact": resources["original_artifact"],
            "original_records": resources["original_records"],
            "validation_artifact": resources["validation_artifact"],
            "validation_records": resources["validation_records"],
            "embedding_table": resources["embedding_table"],
            "model": model,
        },
        "checks": {
            "support_bank_ready": True,
            "support_plan_binding": True,
            "support_token_binding": True,
            "support_current_length_vector_equal": True,
            "natural_token_mask_equivalence": natural_equivalence,
            "producer_source": producer,
            "producer_source_hash_expected": PRODUCER_SHA256,
            "common_validation_geometry": [VALIDATION_RECORDS, WIDTH, HIDDEN],
            "common_validation_post_bos": VALIDATION_POST_BOS,
            "shared_embedding_geometry": [VOCAB_SIZE, HIDDEN],
        },
        "forecast": forecast,
        "source_code": {
            "adapter": _file_record(root / "scripts/trr0007_capture_fit.py", label="TRR7 capture adapter"),
            **_source_code_binding(root),
        },
        "commands": {
            "capture": command,
            "verify": verify_command,
            "working_directory": str(root),
            "capture_output_root": str(paths["capture_root"]),
        },
        "access_contract": {
            "public_prefix_only": True,
            "private_truth_accessed": False,
            "target_weights_accessed": False,
            "holdout_rows_accessed": False,
            "preflight_model_forward": False,
        },
    }
    receipt_path = paths["preflight_receipt"]
    if receipt_path.exists() or receipt_path.is_symlink():
        raise CaptureContractError(f"preflight receipt is create-only and already exists: {receipt_path}")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt"] = {"path": str(receipt_path), "bytes": int(receipt_path.stat().st_size), "sha256": _sha256_file(receipt_path)}
    # The receipt is intentionally written once; adding its own descriptor
    # afterward would make the recorded file hash self-referential.
    return receipt


def _contiguous_runs(indices: Sequence[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value != previous + 1:
            runs.append((start, previous + 1))
            start = value
        previous = value
    runs.append((start, previous + 1))
    return runs


def _verify(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    paths = _base_paths(args, root)
    if args.capture_artifact is None:
        raise CaptureContractError("--capture-artifact is required in verify mode")
    capture_artifact = _resolve(args.capture_artifact, root=root)
    plan = _load_plan(paths["corpus_plan"])
    natural_slots = [int(value) for value in plan["trr0007_support"]["natural_slot_indices"]]
    support_tokens, support_mask, support_info = _load_tokens(paths["constructed_tokens"], label="support constructed tokens")
    current_tokens, current_mask, current_info = _load_tokens(paths["current_tokens"], label="current enriched tokens")
    if support_info["length_vector_digest"] != EXPECTED_LENGTH_DIGEST or current_info["length_vector_digest"] != EXPECTED_LENGTH_DIGEST:
        raise CaptureContractError("support/current length vector changed during verification")
    natural_tokens = _natural_token_equivalence(
        support_tokens, support_mask, current_tokens, current_mask, natural_slots
    )
    capture_tokens, capture_mask, capture_header = _load_tokens(capture_artifact, label="captured public tokens")
    if not torch.equal(capture_tokens, support_tokens):
        raise CaptureContractError("captured token IDs do not equal the support constructed token artifact")
    if not torch.equal(capture_mask, support_mask):
        raise CaptureContractError("captured attention masks do not equal the support constructed token artifact")
    activation_header = _header(capture_artifact, "activations", label="captured public activations")
    if activation_header["shape"] != [RECORDS, WIDTH, HIDDEN] or activation_header["dtype"] != "BF16":
        raise CaptureContractError("captured activation geometry/dtype changed")
    metadata: dict[str, str] = {}
    with safe_open(str(capture_artifact), framework="pt", device="cpu") as handle:
        metadata = {str(k): str(v) for k, v in (handle.metadata() or {}).items()}
    plan_hash = _sha256_file(paths["corpus_plan"])
    if metadata.get("source_plan_sha256") != plan_hash:
        raise CaptureContractError("captured artifact source plan hash does not match support plan")
    if metadata.get("model_revision") != EXPECTED_MODEL_REVISION or metadata.get("cut_depth") != str(CUT_DEPTH):
        raise CaptureContractError("captured artifact model or cut-depth binding changed")
    natural_mismatch_slots: list[int] = []
    max_abs_delta = 0.0
    compared_values = 0
    with safe_open(str(capture_artifact), framework="pt", device="cpu") as captured, safe_open(str(paths["current_activations"]), framework="pt", device="cpu") as current:
        captured_view = captured.get_slice("activations")
        current_view = current.get_slice("activations")
        if captured_view.get_shape() != current_view.get_shape():
            raise CaptureContractError("captured/current activation shapes differ")
        for start, stop in _contiguous_runs(natural_slots):
            lhs = captured_view[start:stop]
            rhs = current_view[start:stop]
            compared_values += int(lhs.numel())
            if not torch.equal(lhs, rhs):
                diff = (lhs.float() - rhs.float()).abs()
                max_abs_delta = max(max_abs_delta, float(diff.max().item()))
                for offset in range(stop - start):
                    if not torch.equal(lhs[offset], rhs[offset]):
                        natural_mismatch_slots.append(start + offset)
    if natural_mismatch_slots:
        raise CaptureContractError(
            f"natural activation equality failed on {len(natural_mismatch_slots)} slots: {natural_mismatch_slots[:8]}"
        )
    receipt = {
        "schema": "token-reconstruction.trr0007-p0-capture-verification.v1",
        "task_id": TASK_ID,
        "status": "PASS_CAPTURED_H_EXACT_ON_1080_NATURAL_ROWS",
        "execution": {
            "ended_utc": _utc_now(),
            "git_commit": _git_commit(root),
            "python": sys.version,
            "platform": platform.platform(),
            "resource_usage": {
                "user_cpu_seconds": float(resource.getrusage(resource.RUSAGE_SELF).ru_utime),
                "system_cpu_seconds": float(resource.getrusage(resource.RUSAGE_SELF).ru_stime),
                "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            },
        },
        "sources": {
            "corpus_plan": _file_record(paths["corpus_plan"], label="support corpus plan"),
            "constructed_tokens": _file_record(paths["constructed_tokens"], label="support constructed tokens"),
            "current_tokens": _file_record(paths["current_tokens"], label="current enriched tokens"),
            "current_activations": _file_record(paths["current_activations"], label="current enriched activations"),
            "capture_artifact": _file_record(capture_artifact, label="captured public activations"),
        },
        "geometry": {
            "capture_activation_shape": activation_header["shape"],
            "capture_activation_dtype": activation_header["dtype"],
            "natural_rows": NATURAL_ROWS,
            "natural_slot_indices_sha256": _sha256_ints(natural_slots),
            "compared_activation_values": compared_values,
            "max_absolute_delta": max_abs_delta,
            "decision_rule": "torch.equal on each natural row; no tolerance",
        },
        "checks": {
            "captured_tokens_equal_support": True,
            "captured_masks_equal_support": True,
            "captured_metadata_source_plan_equal": True,
            "natural_token_mask_equivalence": natural_tokens,
            "natural_activation_equivalence": {
                "status": "PASS",
                "rows_compared": NATURAL_ROWS,
                "rows_mismatched": 0,
                "max_absolute_delta": max_abs_delta,
                "decision_rule": "torch.equal",
            },
        },
        "access_contract": {
            "public_prefix_only": True,
            "private_truth_accessed": False,
            "target_weights_accessed": False,
            "holdout_rows_accessed": False,
            "verification_model_forward": False,
        },
    }
    receipt_path = _resolve(args.receipt, root=root) if args.receipt else capture_artifact.parent / "capture_verification_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise CaptureContractError(f"verification receipt is create-only and already exists: {receipt_path}")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt"] = {"path": str(receipt_path), "bytes": int(receipt_path.stat().st_size), "sha256": _sha256_file(receipt_path)}
    return receipt


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    defaults = _default_paths(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "verify"), required=True)
    parser.add_argument("--corpus-plan", type=Path, default=defaults["corpus_plan"])
    parser.add_argument("--bank-receipt", type=Path, default=defaults["bank_receipt"])
    parser.add_argument("--constructed-tokens", type=Path, default=defaults["constructed_tokens"])
    parser.add_argument("--current-tokens", type=Path, default=defaults["current_tokens"])
    parser.add_argument("--current-activations", type=Path, default=defaults["current_activations"])
    parser.add_argument("--original-artifact", type=Path, default=defaults["original_artifact"])
    parser.add_argument("--original-records", type=Path, default=defaults["original_records"])
    parser.add_argument("--validation-artifact", type=Path, default=defaults["validation_artifact"])
    parser.add_argument("--validation-records", type=Path, default=defaults["validation_records"])
    parser.add_argument("--embedding-table", type=Path, default=defaults["embedding_table"])
    parser.add_argument("--model", type=Path, default=defaults["model"])
    parser.add_argument("--capture-root", type=Path, default=defaults["capture_root"])
    parser.add_argument("--capture-artifact", type=Path)
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        result = _preflight(args, root) if args.mode == "preflight" else _verify(args, root)
    except (CaptureContractError, OSError, RuntimeError) as exc:
        if args.mode == "verify":
            try:
                capture = _resolve(args.capture_artifact, root=root) if args.capture_artifact else None
                failure_path = _resolve(args.receipt, root=root) if args.receipt else (
                    capture.parent / "capture_verification_failure.json" if capture is not None else root / "experiments/TRR-0007/support/capture_verification_failure.json"
                )
                if not failure_path.exists() and not failure_path.is_symlink():
                    failure_path.parent.mkdir(parents=True, exist_ok=True)
                    failure_path.write_text(json.dumps({
                        "schema": "token-reconstruction.trr0007-p0-capture-verification.v1",
                        "task_id": TASK_ID,
                        "status": "FAIL_CAPTURE_VERIFICATION",
                        "error": str(exc),
                        "ended_utc": _utc_now(),
                        "capture_artifact": None if capture is None else str(capture),
                        "private_truth_accessed": False,
                    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            except (OSError, RuntimeError):
                pass
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": result["status"],
        "task_id": TASK_ID,
        "mode": args.mode,
        "receipt": result.get("receipt"),
        "natural_rows": result.get("geometry", {}).get("natural_rows", result.get("input_contract", {}).get("natural_rows")),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
