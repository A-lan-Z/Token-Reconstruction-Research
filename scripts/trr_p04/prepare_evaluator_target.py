#!/usr/bin/env python3
"""Prepare the predeclared TRR-P04 evaluator-only LoRA target update.

The target corpus is the pinned local ``HuggingFaceH4/no_robots`` train split.
Rows are selected by the frozen seed/order in ``evaluator_target_plan.json``;
rendered text and token IDs exist only transiently in this process.  The
script audits those transient fingerprints against every available P04 public
ledger, then trains only the declared rank-4 q/v LoRA parameters with a fixed
30-step cyclic schedule.  It writes a private update artifact and a compact
receipt; it never opens evaluator truth and never writes source text or token
IDs.

Use ``--preflight-only`` for metadata and geometry checks, ``--audit-only``
to materialize the pinned target rows and run the no-model overlap audit,
``--qualify-only`` for one bounded worst-geometry training step, and
``--execute`` for the declared 30-step target preparation.  All output roots
are create-only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from token_reconstruction.target_update import (
    TargetLoRAConfig,
    install_target_lora,
    target_lora_parameters,
)


TASK_ID = "TRR-P04"
PLAN_SCHEMA = "token-reconstruction.trr-p04-evaluator-target-plan.v1"
TARGET_SCHEMA = "token-reconstruction.evaluator-target-lora.v1"
TARGET_CONDITION = "p04_evaluator_target_update_v1"
LINEAGE_ID = "p04-target-lora-no-robots-v1-seed20260910"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
MODEL_WEIGHTS_BYTES = 2_471_645_608
MODEL_WEIGHTS_SHA256 = "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"
BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
VOCAB_SIZE = 128256
HIDDEN_SIZE = 2048
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 64
MODULE_OUTPUT_SIZES = {
    "q_proj": HIDDEN_SIZE,
    "v_proj": NUM_KEY_VALUE_HEADS * HEAD_DIM,
}
MAXIMUM_TOKENS = 192
TARGET_ROWS = 256
TARGET_SEED = 20260910
TARGET_STEPS = 30
TARGET_BATCH_SIZE = 8
TARGET_LR = 0.00075
TARGET_WEIGHT_DECAY = 0.0
TARGET_CLIP = 1.0
TARGET_LAYERS = (0, 1, 2, 3)
TARGET_MODULES = ("q_proj", "v_proj")
TARGET_RANK = 4
TARGET_ALPHA = 4.0
DATE_STRING = "06 Aug 2026"
DATASET_ID = "HuggingFaceH4/no_robots"
DATASET_REVISION = "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b"
DATASET_EXPECTED_ROWS = 9_500
DEFAULT_ARROW = Path(
    "/home/alanz/.cache/huggingface/datasets/HuggingFaceH4___no_robots/default/0.0.0/"
    "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b/no_robots-train.arrow"
)
DEFAULT_DATASET_INFO = DEFAULT_ARROW.parent / "dataset_info.json"
DEFAULT_MODEL_SNAPSHOT = Path(
    "/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/"
    "snapshots/9213176726f574b556790deb65791e0c5aa438b6"
)
DEFAULT_SELECTION = Path("experiments/TRR-P04/setup/public_selection-r2.json")
DEFAULT_PLAN = Path("experiments/TRR-P04/setup/evaluator_target_plan.json")
DEFAULT_TARGET_ARTIFACT = Path(
    "experiments/TRR-P04/private/evaluator_target_update/"
    "p04_evaluator_target_update_v1.safetensors"
)
PR7_ROOT = Path(
    "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004"
)
# These are public metadata ledgers only.  Missing optional runtime copies are
# recorded and do not weaken the required fit/validation/panel audit.
REQUIRED_LEDGER_PATHS = (
    PR7_ROOT / "experiments/TRR-0004/fit/affine_fit_records.json",
    PR7_ROOT / "experiments/TRR-0004/fit/affine_validation_records.json",
    PR7_ROOT / "experiments/TRR-0004/fresh_confirmation_v1/panel_capture/panel.json",
)
TEXT_HASH_KEYS = frozenset(
    {
        "rendered_sha256",
        "public_record_sha256",
        "text_sha256",
        "content_sha256",
        "record_hash",
    }
)
SEQUENCE_HASH_KEYS = frozenset(
    {"truncated_sequence_sha256", "sequence_sha256", "token_ids_sha256"}
)
OPTIONAL_LEDGER_RELATIVE_PATHS = (
    Path("experiments/TRR-P04/setup/public_selection-r2.json"),
    Path("experiments/TRR-P04/setup/public-pools-r2/correction_records.json"),
    Path("experiments/TRR-P04/setup/public-pools-r2/validation_records.json"),
    Path("experiments/TRR-P04/setup/public-pools-r2/fresh_panel_index.json"),
    Path("experiments/TRR-P04/runtime/public-pool-capture-r1/correction_records.json"),
    Path("experiments/TRR-P04/runtime/public-pool-capture-r1/validation_records.json"),
)
SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "PYTHONPATH",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)
# The target update uses the full-vocabulary causal loss at the worst fixed
# geometry.  The internal GPU bounds were approved after a complete one-step
# Adam qualification measured 8.505 GiB peak reserved and 6.014 GiB free; the
# revised 10/5 GiB bounds retain approximately 1.495/1.014 GiB margin.
MIN_FREE_GPU_BYTES = 5 * 2**30
MAX_RESERVED_GPU_BYTES = 10 * 2**30
MAX_HOST_RSS_BYTES = 16 * 2**30
MIN_HOST_AVAILABLE_BYTES = 10 * 2**30


class TargetPreparationError(RuntimeError):
    """Raised when target preparation fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sequence_hash(values: Sequence[int]) -> str:
    try:
        payload = struct.pack("<" + "i" * len(values), *(int(value) for value in values))
    except (struct.error, TypeError, ValueError) as exc:
        raise TargetPreparationError("target sequence cannot be represented as int32") from exc
    return _sha256_bytes(payload)


def _safe_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in SAFE_ENVIRONMENT_KEYS if key in os.environ}


def _git_head() -> str | None:
    try:
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value if len(value) == 40 else None


def _max_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TargetPreparationError(f"{label} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetPreparationError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TargetPreparationError(f"{label} must be a JSON object")
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise TargetPreparationError(f"refusing to overwrite create-only output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _ensure_output_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise TargetPreparationError(f"output root must be create-only: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _descriptor(path: Path, *, label: str, hash_file: bool = True) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TargetPreparationError(f"{label} is unavailable: {path}")
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
    }
    if hash_file:
        result["sha256"] = _sha256_file(path)
    return result


def _validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("task_id") != TASK_ID:
        raise TargetPreparationError("evaluator target plan identity changed")
    if plan.get("condition_id") != TARGET_CONDITION or plan.get("lineage_id") != LINEAGE_ID:
        raise TargetPreparationError("evaluator target lineage changed")
    base = plan.get("base_model")
    if not isinstance(base, Mapping) or base.get("id") != MODEL_ID or base.get("revision") != MODEL_REVISION:
        raise TargetPreparationError("evaluator target base model changed")
    update = plan.get("update")
    expected_update = {
        "family": "evaluator_only_lora",
        "layers": list(TARGET_LAYERS),
        "modules": list(TARGET_MODULES),
        "rank": TARGET_RANK,
        "alpha": TARGET_ALPHA,
        "initialization_seed": TARGET_SEED,
        "optimizer": "AdamW",
        "learning_rate": TARGET_LR,
        "weight_decay": TARGET_WEIGHT_DECAY,
        "gradient_clip_norm": TARGET_CLIP,
        "steps": TARGET_STEPS,
        "record_batch_size": TARGET_BATCH_SIZE,
        "maximum_tokens_including_bos": MAXIMUM_TOKENS,
        "target_update_path": "experiments/TRR-P04/private/evaluator_target_update/p04_evaluator_target_update_v1.safetensors",
    }
    if not isinstance(update, Mapping) or any(update.get(key) != value for key, value in expected_update.items()):
        raise TargetPreparationError("evaluator target update configuration changed")
    corpus = plan.get("update_corpus")
    if not isinstance(corpus, Mapping):
        raise TargetPreparationError("evaluator target corpus metadata is absent")
    expected_corpus = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": "train",
        "expected_source_rows": DATASET_EXPECTED_ROWS,
    }
    if any(corpus.get(key) != value for key, value in expected_corpus.items()):
        raise TargetPreparationError("evaluator target corpus identity changed")
    selection = corpus.get("selection")
    if not isinstance(selection, Mapping) or selection.get("seed") != TARGET_SEED or selection.get("records") != TARGET_ROWS or selection.get("selected_row_order_sha256") != "42fb5bb7dfc58dba8ccf9e3b288e787fd88cc1788e936ee934cfbbdc86de2fd2":
        raise TargetPreparationError("evaluator target row selection changed")
    for key in ("student_training_access", "teacher_access", "fresh_panel_access"):
        if corpus.get(key) is not False:
            raise TargetPreparationError(f"evaluator target access boundary changed: {key}")
    schedule = plan.get("schedule")
    expected_schedule = {
        "batch_order": "cyclic sequential selected rows",
        "formula": "row=(step*record_batch_size+offset) mod records",
        "steps": TARGET_STEPS,
        "record_batch_size": TARGET_BATCH_SIZE,
    }
    if schedule != expected_schedule:
        raise TargetPreparationError("evaluator target batch schedule changed")
    return {
        "condition_id": TARGET_CONDITION,
        "lineage_id": LINEAGE_ID,
        "config": expected_update,
        "selection": {
            "seed": TARGET_SEED,
            "records": TARGET_ROWS,
            "row_order_sha256": str(selection["selected_row_order_sha256"]),
        },
    }


def _target_indices(size: int) -> list[int]:
    if size != DATASET_EXPECTED_ROWS:
        raise TargetPreparationError(f"no_robots row count changed: {size}")
    return sorted(
        range(size),
        key=lambda index: (
            _sha256_bytes(f"TRR-P04|target-update-v1|row:{index}|seed:{TARGET_SEED}".encode("utf-8")),
            index,
        ),
    )[:TARGET_ROWS]


def _validate_index_digest(indices: Sequence[int], expected: str) -> None:
    actual = _digest_lines(str(index) for index in indices)
    if actual != expected:
        raise TargetPreparationError(f"target row-order digest changed: {actual}")


def _dataset_descriptor(arrow_path: Path, info_path: Path, *, hash_arrow: bool) -> dict[str, Any]:
    arrow_path = arrow_path.expanduser().resolve()
    info_path = info_path.expanduser().resolve()
    if arrow_path.is_symlink() or not arrow_path.is_file():
        raise TargetPreparationError(f"pinned target Arrow file is unavailable: {arrow_path}")
    if info_path.is_symlink() or not info_path.is_file():
        raise TargetPreparationError(f"pinned target dataset info is unavailable: {info_path}")
    result = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": "train",
        "arrow": {
            "path": str(arrow_path),
            "bytes": int(arrow_path.stat().st_size),
            "sha256": _sha256_file(arrow_path) if hash_arrow else None,
            "hash_recorded": bool(hash_arrow),
        },
        "dataset_info": _descriptor(info_path, label="target dataset info"),
    }
    if int(arrow_path.stat().st_size) != 16_503_208:
        raise TargetPreparationError("pinned no_robots Arrow size changed")
    return result


def _normalise_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not value:
        raise TargetPreparationError("chat template returned no token IDs")
    result: list[int] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0 or token >= VOCAB_SIZE:
            raise TargetPreparationError("target tokenizer returned an invalid token ID")
        result.append(int(token))
    return result


def _messages(row: Mapping[str, Any], row_index: int) -> list[dict[str, str]]:
    raw = row.get("messages")
    if not isinstance(raw, list) or not raw:
        raise TargetPreparationError(f"no_robots row {row_index} has no messages list")
    result: list[dict[str, str]] = []
    for message_index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise TargetPreparationError(f"no_robots row {row_index} message {message_index} is malformed")
        role = value.get("role")
        content = value.get("content")
        if not isinstance(role, str) or not isinstance(content, str) or not role or not content:
            raise TargetPreparationError(f"no_robots row {row_index} message {message_index} is incomplete")
        result.append({"role": role, "content": content})
    return result


def _materialize_records(dataset: Any, tokenizer: Any, indices: Sequence[int]) -> list[dict[str, Any]]:
    if len(indices) != TARGET_ROWS:
        raise TargetPreparationError("target selection quota changed")
    if int(getattr(tokenizer, "bos_token_id", -1)) != BOS_TOKEN_ID:
        raise TargetPreparationError("target tokenizer BOS changed")
    # The pinned Llama tokenizer may leave pad_token_id unset.  Target
    # training pads its fixed batch explicitly with the declared model PAD ID;
    # an explicitly configured tokenizer pad ID, when present, must agree.
    tokenizer_pad_id = getattr(tokenizer, "pad_token_id", None)
    if tokenizer_pad_id is not None and int(tokenizer_pad_id) != PAD_TOKEN_ID:
        raise TargetPreparationError("target tokenizer padding token changed")
    records: list[dict[str, Any]] = []
    for row_index in indices:
        row = dataset[int(row_index)]
        messages = _messages(row, int(row_index))
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            date_string=DATE_STRING,
        )
        if not isinstance(rendered, str) or not rendered:
            raise TargetPreparationError(f"no_robots row {row_index} rendered to empty text")
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            date_string=DATE_STRING,
        )
        token_ids = _normalise_ids(encoded)
        if token_ids[0] != BOS_TOKEN_ID or len(token_ids) < 2:
            raise TargetPreparationError(f"no_robots row {row_index} lost BOS or is too short")
        active_count = min(len(token_ids), MAXIMUM_TOKENS)
        values = token_ids[:MAXIMUM_TOKENS]
        records.append(
            {
                "row_index": int(row_index),
                "record_id": f"no-robots-train-{int(row_index):05d}-{_sha256_bytes(rendered.encode('utf-8'))[:16]}",
                "rendered_sha256": _sha256_bytes(rendered.encode("utf-8")),
                "truncated_sequence_sha256": _sequence_hash(values[: 1 + 128]),
                "input_ids": values,
                "active_count": active_count,
                "full_token_count": len(token_ids),
            }
        )
    return records


def _collect_fingerprints(value: Any, *, text_hashes: set[str], sequence_hashes: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                continue
            normalized = key.casefold().replace("-", "_")
            if normalized in {"token_ids", "input_ids", "labels", "source_text", "truth", "oracle", "target_weights"}:
                continue
            if isinstance(child, str) and len(child) == 64:
                try:
                    int(child, 16)
                except ValueError:
                    pass
                else:
                    if normalized in TEXT_HASH_KEYS:
                        text_hashes.add(child)
                    elif normalized in SEQUENCE_HASH_KEYS:
                        sequence_hashes.add(child)
            _collect_fingerprints(child, text_hashes=text_hashes, sequence_hashes=sequence_hashes)
    elif isinstance(value, list):
        for child in value:
            _collect_fingerprints(child, text_hashes=text_hashes, sequence_hashes=sequence_hashes)


def _ledger_paths(repo_root: Path, selection_path: Path) -> list[tuple[Path, bool]]:
    result: list[tuple[Path, bool]] = [(path, True) for path in REQUIRED_LEDGER_PATHS]
    result.append((selection_path, True))
    result.extend((repo_root / relative, False) for relative in OPTIONAL_LEDGER_RELATIVE_PATHS)
    deduplicated: list[tuple[Path, bool]] = []
    seen: set[Path] = set()
    for path, required in result:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduplicated.append((resolved, required))
    return deduplicated


def _overlap_audit(records: Sequence[Mapping[str, Any]], *, ledger_paths: Sequence[tuple[Path, bool]]) -> dict[str, Any]:
    target_text = {str(row["rendered_sha256"]) for row in records}
    target_sequence = {str(row["truncated_sequence_sha256"]) for row in records}
    if len(target_text) != len(records) or len(target_sequence) != len(records):
        raise TargetPreparationError("selected target rows contain duplicate rendered/token fingerprints")
    ledger_results: list[dict[str, Any]] = []
    total_text_collisions = 0
    total_sequence_collisions = 0
    missing_required: list[str] = []
    for path, required in ledger_paths:
        descriptor: dict[str, Any] = {
            "path": str(path),
            "required": bool(required),
            "available": False,
            "text_hash_collisions": 0,
            "sequence_hash_collisions": 0,
        }
        if not path.is_file() or path.is_symlink():
            if required:
                missing_required.append(str(path))
            ledger_results.append(descriptor)
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TargetPreparationError(f"target overlap ledger is invalid JSON: {path}") from exc
        prior_text: set[str] = set()
        prior_sequence: set[str] = set()
        _collect_fingerprints(value, text_hashes=prior_text, sequence_hashes=prior_sequence)
        text_collisions = len(target_text.intersection(prior_text))
        sequence_collisions = len(target_sequence.intersection(prior_sequence))
        descriptor.update(
            {
                "available": True,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
                "text_fingerprint_count": len(prior_text),
                "sequence_fingerprint_count": len(prior_sequence),
                "text_hash_collisions": text_collisions,
                "sequence_hash_collisions": sequence_collisions,
            }
        )
        total_text_collisions += text_collisions
        total_sequence_collisions += sequence_collisions
        ledger_results.append(descriptor)
    if missing_required:
        raise TargetPreparationError(f"required target overlap ledgers are unavailable: {missing_required}")
    if total_text_collisions or total_sequence_collisions:
        raise TargetPreparationError(
            f"target source overlap detected: text={total_text_collisions} sequence={total_sequence_collisions}"
        )
    return {
        "status": "PASS_NO_EXACT_TEXT_OR_TRUNCATED_SEQUENCE_OVERLAP",
        "target_records": len(records),
        "target_text_fingerprint_count": len(target_text),
        "target_sequence_fingerprint_count": len(target_sequence),
        "text_collisions": total_text_collisions,
        "sequence_collisions": total_sequence_collisions,
        "ledgers": ledger_results,
    }


def _length_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in records:
        bucket = str(min(int(row["full_token_count"]), MAXIMUM_TOKENS))
        summary[bucket] = summary.get(bucket, 0) + 1
    return dict(sorted(summary.items(), key=lambda item: int(item[0])))


def _read_mem_available_bytes() -> int:
    try:
        text = Path("/proc/meminfo").read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise TargetPreparationError("host MemAvailable is unreadable") from exc
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if key != "MemAvailable" or not separator:
            continue
        fields = value.strip().split()
        if not fields:
            break
        try:
            number = int(fields[0])
        except ValueError:
            break
        unit = fields[1].lower() if len(fields) > 1 else "kb"
        multiplier = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}.get(unit)
        if multiplier is None or number <= 0:
            break
        return number * multiplier
    raise TargetPreparationError("host MemAvailable is unavailable")


def _cuda_guard(device: torch.device, *, stage: str, started: float) -> dict[str, Any]:
    if device.type != "cuda":
        raise TargetPreparationError("target preparation requires CUDA for model training")
    if not torch.cuda.is_available():
        raise TargetPreparationError("CUDA is unavailable")
    try:
        free, total = torch.cuda.mem_get_info(device)
        reserved = int(torch.cuda.memory_reserved(device))
    except Exception as exc:
        raise TargetPreparationError(f"CUDA resource data unavailable at {stage}") from exc
    host_available = _read_mem_available_bytes()
    rss = _max_rss_bytes()
    # Include a compact numeric snapshot in any fail-closed diagnostic.  The
    # main failure receipt remains metadata-only, but this preserves the actual
    # allocation at the boundary where a child exits before its normal peak
    # summary can be written.
    allocated = int(torch.cuda.memory_allocated(device))
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    snapshot = (
        f"stage={stage} free={int(free)} total={int(total)} reserved={reserved} "
        f"allocated={allocated} peak_allocated={peak_allocated} peak_reserved={peak_reserved} "
        f"host_available={int(host_available)} rss={int(rss)}"
    )
    if free < MIN_FREE_GPU_BYTES:
        raise TargetPreparationError(f"target resource guard free GPU limit: {snapshot}")
    if reserved > MAX_RESERVED_GPU_BYTES:
        raise TargetPreparationError(f"target resource guard reserved GPU limit: {snapshot}")
    if host_available < MIN_HOST_AVAILABLE_BYTES:
        raise TargetPreparationError(f"target resource guard host availability limit: {snapshot}")
    if rss > MAX_HOST_RSS_BYTES:
        raise TargetPreparationError(f"target resource guard host RSS limit: {snapshot}")
    return {
        "stage": stage,
        "status": "PASS",
        "free_bytes": int(free),
        "total_bytes": int(total),
        "reserved_bytes": reserved,
        "host_mem_available_bytes": int(host_available),
        "rss_bytes": rss,
        "minimum_free_gpu_bytes": MIN_FREE_GPU_BYTES,
        "maximum_reserved_gpu_bytes": MAX_RESERVED_GPU_BYTES,
        "minimum_host_available_bytes": MIN_HOST_AVAILABLE_BYTES,
        "maximum_host_rss_bytes": MAX_HOST_RSS_BYTES,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def _expected_lora_parameter_count() -> int:
    return sum(
        TARGET_RANK * (HIDDEN_SIZE + MODULE_OUTPUT_SIZES[module])
        for _layer in TARGET_LAYERS
        for module in TARGET_MODULES
    )


def _validate_target_projection_geometry(model: torch.nn.Module) -> None:
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise TargetPreparationError("target model exposes no decoder layers")
    for layer_index in TARGET_LAYERS:
        try:
            attention = layers[layer_index].self_attn
        except (AttributeError, IndexError) as exc:
            raise TargetPreparationError(f"target layer {layer_index} is unavailable") from exc
        for module_name in TARGET_MODULES:
            module = getattr(attention, module_name, None)
            expected_out = MODULE_OUTPUT_SIZES[module_name]
            if (
                module is None
                or int(getattr(module, "in_features", -1)) != HIDDEN_SIZE
                or int(getattr(module, "out_features", -1)) != expected_out
            ):
                raise TargetPreparationError(
                    f"target projection geometry changed for layer {layer_index}.{module_name}"
                )


def _preflight_estimate() -> dict[str, Any]:
    lora_modules = len(TARGET_LAYERS) * len(TARGET_MODULES)
    lora_parameter_count = _expected_lora_parameter_count()
    logits_bf16 = TARGET_BATCH_SIZE * (MAXIMUM_TOKENS - 1) * VOCAB_SIZE * 2
    logits_fp32 = TARGET_BATCH_SIZE * (MAXIMUM_TOKENS - 1) * VOCAB_SIZE * 4
    return {
        "status": "ESTIMATE_ONLY_NOT_MEASURED",
        "worst_geometry": {
            "record_batch_size": TARGET_BATCH_SIZE,
            "tokens_including_bos": MAXIMUM_TOKENS,
            "vocabulary_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
        },
        "model_weights_bytes": MODEL_WEIGHTS_BYTES,
        "lora_modules": lora_modules,
        "module_output_sizes": dict(MODULE_OUTPUT_SIZES),
        "lora_parameter_count": lora_parameter_count,
        "one_step_logits_bytes_bfloat16": logits_bf16,
        "one_step_logits_bytes_float32_loss_copy": logits_fp32,
        "resource_limits": {
            "minimum_free_gpu_bytes": MIN_FREE_GPU_BYTES,
            "maximum_reserved_gpu_bytes": MAX_RESERVED_GPU_BYTES,
            "minimum_host_available_bytes": MIN_HOST_AVAILABLE_BYTES,
            "maximum_host_rss_bytes": MAX_HOST_RSS_BYTES,
        },
        "runtime_estimate": "not measured until the bounded one-step qualification; full 30-step runtime is then recorded",
    }


def build_preflight(*, plan_path: Path, selection_path: Path, arrow_path: Path, tokenizer_path: Path, model_snapshot: Path, output_root: Path, argv: Sequence[str]) -> dict[str, Any]:
    started = time.perf_counter()
    plan = _load_json(plan_path, label="evaluator target plan")
    target = _validate_plan(plan)
    selection = _load_json(selection_path, label="P04 public selection")
    if selection.get("schema") != "token-reconstruction.trr-p04-public-selection.v1" or selection.get("task_id") != TASK_ID:
        raise TargetPreparationError("P04 public selection identity changed")
    panel = selection.get("panel")
    if not isinstance(panel, Mapping) or panel.get("independent_source_records") != 72 or panel.get("anchor_record_count") != 12:
        raise TargetPreparationError("P04 panel geometry changed")
    output = _ensure_output_root(output_root)
    arrow_path = arrow_path.expanduser().resolve()
    tokenizer_path = tokenizer_path.expanduser().resolve()
    model_snapshot = model_snapshot.expanduser().resolve()
    dataset = _dataset_descriptor(arrow_path, arrow_path.parent / "dataset_info.json", hash_arrow=False)
    if not tokenizer_path.is_dir() or tokenizer_path.is_symlink():
        raise TargetPreparationError(f"pinned target tokenizer snapshot is unavailable: {tokenizer_path}")
    if not model_snapshot.is_dir() or model_snapshot.is_symlink():
        raise TargetPreparationError(f"pinned target model snapshot is unavailable: {model_snapshot}")
    value = {
        "schema": f"{TARGET_SCHEMA}-preflight.v1",
        "task_id": TASK_ID,
        "status": "PASS_NO_MODEL_NO_TARGET_NO_EVALUATION_TRUTH",
        "created_utc": _utc_now(),
        "plan": {"path": str(plan_path.resolve()), "sha256": _sha256_file(plan_path), **target},
        "selection": {"path": str(selection_path.resolve()), "sha256": _sha256_file(selection_path), "records": 72, "anchors": 12},
        "dataset": {**dataset, "expected_rows": DATASET_EXPECTED_ROWS, "hash_deferred_until_target_prep": True},
        "assets": {
            "tokenizer_snapshot": str(tokenizer_path),
            "model_snapshot": str(model_snapshot),
            "model_weights": {"bytes": MODEL_WEIGHTS_BYTES, "sha256": MODEL_WEIGHTS_SHA256, "hash_source": "pinned prior snapshot metadata"},
        },
        "geometry": _preflight_estimate(),
        "access": {
            "dataset_rows_read": False,
            "source_text_materialized": False,
            "source_tokens_materialized": False,
            "model_loaded": False,
            "target_update_written": False,
            "evaluation_truth_opened": False,
            "student_or_teacher_access": False,
        },
        "execution": {
            "argv": list(argv),
            "safe_environment": _safe_environment(),
            "git_commit": _git_head(),
            "python": sys.version,
            "platform": platform.platform(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "max_rss_bytes": _max_rss_bytes(),
        },
    }
    _write_create_only(output / "target_preparation_preflight.json", value)
    return value


def _load_dataset_and_tokenizer(arrow_path: Path, tokenizer_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    try:
        from datasets import Dataset
        from transformers import AutoTokenizer
        dataset = Dataset.from_file(str(arrow_path.expanduser().resolve()))
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path.expanduser().resolve()), local_files_only=True)
    except Exception as exc:
        raise TargetPreparationError("pinned target dataset/tokenizer loading failed") from exc
    if len(dataset) != DATASET_EXPECTED_ROWS:
        raise TargetPreparationError(f"no_robots row count changed: {len(dataset)}")
    tokenizer.padding_side = "right"
    return dataset, tokenizer, _dataset_descriptor(arrow_path, arrow_path.expanduser().resolve().parent / "dataset_info.json", hash_arrow=True)


def audit_target(
    *,
    plan_path: Path,
    selection_path: Path,
    arrow_path: Path,
    tokenizer_path: Path,
    output_root: Path,
    argv: Sequence[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    plan = _load_json(plan_path, label="evaluator target plan")
    target = _validate_plan(plan)
    selection = _load_json(selection_path, label="P04 public selection")
    output = _ensure_output_root(output_root)
    dataset, tokenizer, dataset_descriptor = _load_dataset_and_tokenizer(arrow_path, tokenizer_path)
    indices = _target_indices(len(dataset))
    _validate_index_digest(indices, target["selection"]["row_order_sha256"])
    records = _materialize_records(dataset, tokenizer, indices)
    audit = _overlap_audit(records, ledger_paths=_ledger_paths(Path.cwd(), selection_path))
    value = {
        "schema": f"{TARGET_SCHEMA}-selection-audit.v1",
        "task_id": TASK_ID,
        "status": "PASS_TARGET_SELECTION_AUDIT_NO_MODEL_NO_EVALUATION_TRUTH",
        "created_utc": _utc_now(),
        "plan": {"path": str(plan_path.resolve()), "sha256": _sha256_file(plan_path), **target},
        "selection": {"path": str(selection_path.resolve()), "sha256": _sha256_file(selection_path)},
        "dataset": dataset_descriptor,
        "selected_rows": {
            "count": len(records),
            "row_order_sha256": _digest_lines(str(row["row_index"]) for row in records),
            "active_length_summary": _length_summary(records),
            "row_ids_or_source_text_serialized": False,
        },
        "overlap_audit": audit,
        "access": {
            "dataset_rows_read": True,
            "source_text_materialized_transiently": True,
            "source_tokens_materialized_transiently": True,
            "source_text_serialized": False,
            "source_tokens_serialized": False,
            "model_loaded": False,
            "target_update_written": False,
            "evaluation_truth_opened": False,
            "student_or_teacher_access": False,
        },
        "execution": {
            "argv": list(argv),
            "safe_environment": _safe_environment(),
            "git_commit": _git_head(),
            "python": sys.version,
            "platform": platform.platform(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "max_rss_bytes": _max_rss_bytes(),
        },
    }
    _write_create_only(output / "target_selection_audit.json", value)
    return value


def _set_runtime(seed: int) -> None:
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise TargetPreparationError("target preparation requires CUDA")
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def _load_model(snapshot: Path, *, device: torch.device) -> torch.nn.Module:
    snapshot = snapshot.expanduser().resolve()
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise TargetPreparationError(f"pinned target model snapshot is unavailable: {snapshot}")
    weights = snapshot / "model.safetensors"
    # Hugging Face snapshots commonly expose immutable blobs through a
    # symlink. The snapshot directory itself is pinned above; accept that
    # representation while still requiring the resolved weight asset and
    # declared byte size to be present.
    if not weights.is_file() or not weights.resolve().is_file() or int(weights.stat().st_size) != MODEL_WEIGHTS_BYTES:
        raise TargetPreparationError("pinned target model weight asset changed")
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot.expanduser().resolve()),
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()
    except Exception as exc:
        raise TargetPreparationError("pinned target base model loading failed") from exc
    if int(model.config.hidden_size) != HIDDEN_SIZE or int(model.config.vocab_size) != VOCAB_SIZE:
        raise TargetPreparationError("target model geometry changed")
    _validate_target_projection_geometry(model)
    model.requires_grad_(False)
    model.config.use_cache = False
    return model


def _tensor_batch(records: Sequence[Mapping[str, Any]], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.full((len(records), MAXIMUM_TOKENS), PAD_TOKEN_ID, dtype=torch.long)
    mask = torch.zeros((len(records), MAXIMUM_TOKENS), dtype=torch.bool)
    for row_index, record in enumerate(records):
        values = [int(value) for value in record["input_ids"]]
        if len(values) < 2 or len(values) > MAXIMUM_TOKENS or values[0] != BOS_TOKEN_ID:
            raise TargetPreparationError("target training sequence geometry changed")
        ids[row_index, : len(values)] = torch.tensor(values, dtype=torch.long)
        mask[row_index, : len(values)] = True
    return ids.to(device), mask.to(device)


def _train_steps(
    model: torch.nn.Module,
    installed: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    steps: int,
    started: float,
    guard: list[dict[str, Any]],
) -> dict[str, Any]:
    parameters = target_lora_parameters(installed.values())
    if not parameters:
        raise TargetPreparationError("target LoRA has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=TARGET_LR, weight_decay=TARGET_WEIGHT_DECAY)
    model.train()
    losses: list[float] = []
    gradient_norms: list[float] = []
    step_seconds: list[float] = []
    for step_index in range(steps):
        step_started = time.perf_counter()
        start = (step_index * TARGET_BATCH_SIZE) % len(records)
        batch_records = [records[(start + offset) % len(records)] for offset in range(TARGET_BATCH_SIZE)]
        input_ids, mask = _tensor_batch(batch_records, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=mask.to(torch.long), use_cache=False)
        if tuple(output.logits.shape) != (TARGET_BATCH_SIZE, MAXIMUM_TOKENS, VOCAB_SIZE):
            raise TargetPreparationError("target full-vocabulary logits geometry changed")
        logits = output.logits[:, :-1].float()
        labels = input_ids[:, 1:].clone()
        labels.masked_fill_(~mask[:, 1:], -100)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss).item():
            raise TargetPreparationError(f"target loss became non-finite at step {step_index}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, TARGET_CLIP, error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(norm.detach().cpu()))
        step_seconds.append(time.perf_counter() - step_started)
        guard.append(_cuda_guard(device, stage=f"after_training_step_{step_index + 1:02d}", started=started))
        del output, logits, labels, loss, input_ids, mask
    model.eval()
    return {
        "steps": steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "losses": losses,
        "gradient_norms": gradient_norms,
        "step_seconds": step_seconds,
        "mean_step_seconds": sum(step_seconds) / len(step_seconds),
    }


def execute_target(
    *,
    mode: str,
    plan_path: Path,
    selection_path: Path,
    arrow_path: Path,
    tokenizer_path: Path,
    model_snapshot: Path,
    output_root: Path,
    device: torch.device,
    argv: Sequence[str],
) -> dict[str, Any]:
    started_perf = time.perf_counter()
    started_utc = _utc_now()
    plan = _load_json(plan_path, label="evaluator target plan")
    target = _validate_plan(plan)
    selection = _load_json(selection_path, label="P04 public selection")
    output = _ensure_output_root(output_root)
    dataset, tokenizer, dataset_descriptor = _load_dataset_and_tokenizer(arrow_path, tokenizer_path)
    indices = _target_indices(len(dataset))
    _validate_index_digest(indices, target["selection"]["row_order_sha256"])
    records = _materialize_records(dataset, tokenizer, indices)
    overlap = _overlap_audit(records, ledger_paths=_ledger_paths(Path.cwd(), selection_path))
    audit_receipt = {
        "schema": f"{TARGET_SCHEMA}-selection-audit.v1",
        "task_id": TASK_ID,
        "status": "PASS_TARGET_SELECTION_AUDIT_NO_EVALUATION_TRUTH",
        "plan_sha256": _sha256_file(plan_path),
        "selection_sha256": _sha256_file(selection_path),
        "dataset": dataset_descriptor,
        "selected_rows": {
            "count": len(records),
            "row_order_sha256": _digest_lines(str(row["row_index"]) for row in records),
            "active_length_summary": _length_summary(records),
            "row_ids_or_source_text_serialized": False,
        },
        "overlap_audit": overlap,
    }
    _write_create_only(output / "target_selection_audit.json", audit_receipt)
    guard: list[dict[str, Any]] = []
    guard.append(_cuda_guard(device, stage="before_target_model_load", started=started_perf))
    _set_runtime(TARGET_SEED)
    model = _load_model(model_snapshot, device=device)
    guard.append(_cuda_guard(device, stage="after_target_model_load", started=started_perf))
    config = TargetLoRAConfig(
        layers=TARGET_LAYERS,
        modules=TARGET_MODULES,
        rank=TARGET_RANK,
        alpha=TARGET_ALPHA,
        seed=TARGET_SEED,
    )
    installed = install_target_lora(model, config)
    parameters = target_lora_parameters(installed.values())
    expected_parameters = _expected_lora_parameter_count()
    if sum(int(parameter.numel()) for parameter in parameters) != expected_parameters:
        raise TargetPreparationError("target LoRA parameter geometry changed")
    guard.append(_cuda_guard(device, stage="after_target_lora_install", started=started_perf))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    steps = 1 if mode == "qualify" else TARGET_STEPS
    training = _train_steps(model, installed, records, device=device, steps=steps, started=started_perf, guard=guard)
    peak = {
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "host_max_rss_bytes": _max_rss_bytes(),
    }
    if peak["cuda_peak_reserved_bytes"] > MAX_RESERVED_GPU_BYTES or peak["host_max_rss_bytes"] > MAX_HOST_RSS_BYTES:
        raise TargetPreparationError("target training peak exceeded resource limits")
    if mode == "qualify":
        value = {
            "schema": f"{TARGET_SCHEMA}-qualification.v1",
            "task_id": TASK_ID,
            "status": "PASS_TARGET_WORST_GEOMETRY_QUALIFICATION_NO_ARTIFACT",
            "created_utc": _utc_now(),
            "mode": mode,
            "plan": {"path": str(plan_path.resolve()), "sha256": _sha256_file(plan_path), **target},
            "selection": {"path": str(selection_path.resolve()), "sha256": _sha256_file(selection_path), "records": len(records)},
            "dataset": dataset_descriptor,
            "geometry": {"batch_records": TARGET_BATCH_SIZE, "tokens_including_bos": MAXIMUM_TOKENS, "hidden_size": HIDDEN_SIZE, "vocabulary_size": VOCAB_SIZE},
            "training": training,
            "resource_guard": {"samples": guard, "peak": peak, "status": "PASS"},
            "access": {"evaluation_truth_opened": False, "source_text_serialized": False, "source_tokens_serialized": False, "target_update_written": False},
            "execution": {"argv": list(argv), "safe_environment": _safe_environment(), "git_commit": _git_head(), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "device": str(device), "elapsed_seconds": round(time.perf_counter() - started_perf, 6), "max_rss_bytes": _max_rss_bytes()},
        }
        _write_create_only(output / "target_qualification_receipt.json", value)
        del model, installed
        gc.collect()
        torch.cuda.empty_cache()
        return value
    artifact_path = output / "p04_evaluator_target_update_v1.safetensors"
    if artifact_path.exists() or artifact_path.is_symlink():
        raise TargetPreparationError(f"target artifact is create-only: {artifact_path}")
    tensors = {f"layers.{layer}.self_attn.{module}.{suffix}": value.detach().cpu().contiguous() for layer in TARGET_LAYERS for module in TARGET_MODULES for suffix, value in (("A", installed[f"layers.{layer}.self_attn.{module}"].A), ("B", installed[f"layers.{layer}.self_attn.{module}"].B))}
    save_file(
        tensors,
        str(artifact_path),
        metadata={
            "schema": TARGET_SCHEMA,
            "task_id": TASK_ID,
            "condition_id": TARGET_CONDITION,
            "lineage_id": LINEAGE_ID,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "layers": json.dumps(list(TARGET_LAYERS)),
            "modules": json.dumps(list(TARGET_MODULES)),
            "rank": str(TARGET_RANK),
            "alpha": str(TARGET_ALPHA),
            "initialization_seed": str(TARGET_SEED),
            "steps": str(TARGET_STEPS),
            "record_batch_size": str(TARGET_BATCH_SIZE),
            "maximum_tokens_including_bos": str(MAXIMUM_TOKENS),
            "schedule": "cyclic sequential selected rows; batch start=(step*8) mod 256",
            "evaluation_truth_opened": "false",
            "source_text_serialized": "false",
            "source_tokens_serialized": "false",
            "weights_available_to_reconstructor": "false",
        },
    )
    if not artifact_path.is_file():
        raise TargetPreparationError("target artifact was not written")
    value = {
        "schema": f"{TARGET_SCHEMA}-preparation-receipt.v1",
        "task_id": TASK_ID,
        "status": "PASS_EVALUATOR_TARGET_UPDATE_READY_NO_EVALUATION_TRUTH",
        "created_utc": _utc_now(),
        "mode": mode,
        "plan": {"path": str(plan_path.resolve()), "sha256": _sha256_file(plan_path), **target},
        "selection": {"path": str(selection_path.resolve()), "sha256": _sha256_file(selection_path), "records": len(records), "row_order_sha256": _digest_lines(str(row["row_index"]) for row in records)},
        "dataset": dataset_descriptor,
        "overlap_audit": overlap,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "snapshot": str(model_snapshot.resolve()), "weights_bytes": MODEL_WEIGHTS_BYTES, "weights_sha256": MODEL_WEIGHTS_SHA256, "weight_hash_scope": "pinned prior snapshot metadata; not rehashed by this run"},
        "geometry": {"batch_records": TARGET_BATCH_SIZE, "tokens_including_bos": MAXIMUM_TOKENS, "hidden_size": HIDDEN_SIZE, "vocabulary_size": VOCAB_SIZE},
        "training": {"config": {"steps": TARGET_STEPS, "record_batch_size": TARGET_BATCH_SIZE, "learning_rate": TARGET_LR, "weight_decay": TARGET_WEIGHT_DECAY, "gradient_clip_norm": TARGET_CLIP, "schedule": "cyclic sequential selected rows; batch start=(step*8) mod 256"}, **training},
        "target_artifact": {"path": str(artifact_path.resolve()), "bytes": int(artifact_path.stat().st_size), "sha256": _sha256_file(artifact_path), "serialized_source_text": False, "serialized_source_tokens": False},
        "resource_guard": {"samples": guard, "peak": peak, "status": "PASS"},
        "access": {"dataset_rows_read": True, "source_text_materialized_transiently": True, "source_tokens_materialized_transiently": True, "source_text_serialized": False, "source_tokens_serialized": False, "model_loaded": True, "target_update_written": True, "evaluation_truth_opened": False, "student_or_teacher_access": False},
        "execution": {"argv": list(argv), "safe_environment": _safe_environment(), "git_commit": _git_head(), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "device": str(device), "started_utc": started_utc, "ended_utc": _utc_now(), "elapsed_seconds": round(time.perf_counter() - started_perf, 6), "max_rss_bytes": _max_rss_bytes()},
    }
    _write_create_only(output / "target_preparation_receipt.json", value)
    del model, installed
    gc.collect()
    torch.cuda.empty_cache()
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--qualify-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--dataset-arrow", type=Path, default=DEFAULT_ARROW)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_MODEL_SNAPSHOT)
    parser.add_argument("--model-snapshot", type=Path, default=DEFAULT_MODEL_SNAPSHOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    effective_argv = list(sys.argv if argv is None else [sys.argv[0], *argv])
    try:
        if args.preflight_only:
            value = build_preflight(plan_path=args.plan, selection_path=args.selection, arrow_path=args.dataset_arrow, tokenizer_path=args.tokenizer, model_snapshot=args.model_snapshot, output_root=args.output_root, argv=effective_argv)
        elif args.audit_only:
            value = audit_target(plan_path=args.plan, selection_path=args.selection, arrow_path=args.dataset_arrow, tokenizer_path=args.tokenizer, output_root=args.output_root, argv=effective_argv)
        else:
            value = execute_target(mode="qualify" if args.qualify_only else "execute", plan_path=args.plan, selection_path=args.selection, arrow_path=args.dataset_arrow, tokenizer_path=args.tokenizer, model_snapshot=args.model_snapshot, output_root=args.output_root, device=torch.device(args.device), argv=effective_argv)
        print(json.dumps({"status": value["status"], "output_root": str(args.output_root.expanduser().resolve())}, sort_keys=True))
        return 0
    except Exception as exc:
        output = args.output_root.expanduser().resolve()
        if output.is_dir() and not (output / "failure.json").exists():
            try:
                _write_create_only(
                    output / "failure.json",
                    {
                        "schema": f"{TARGET_SCHEMA}-failure.v1",
                        "task_id": TASK_ID,
                        "status": "FAILED_CLOSED",
                        "created_utc": _utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "evaluation_truth_opened": False,
                        "source_text_or_tokens_serialized": False,
                        "execution": {"argv": effective_argv, "safe_environment": _safe_environment(), "git_commit": _git_head()},
                    },
                )
            except Exception:
                pass
        print(f"P04 evaluator target preparation failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
