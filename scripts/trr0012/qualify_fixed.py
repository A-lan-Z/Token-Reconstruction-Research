#!/usr/bin/env python3
"""Run the bounded TRR-0012 fixed-readout resource qualification.

This entrypoint is deliberately separate from ``run_fixed_pair.py``.  It uses
the exact P09 runner and public fixed hook for the first eight updates of the
bound B1 schedule, records resource/timing evidence, and writes no decoder
checkpoint.  The production pair remains owned by ``run_fixed_pair.py`` and
still consumes the complete 13,000-update schedule.

``--mode prepare`` performs only standard-library binding and JSON checks.
``--mode run`` performs a fresh live preflight, then launches ``--mode child``
through the native fail-closed P09 watchdog.  The child is the only process
that imports torch or opens model/activation tensors.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


TASK_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = TASK_ROOT.parent.parent
TASK_EXPERIMENT = TASK_ROOT / "experiments" / "TRR-0012"
TASK_OUTPUT = TASK_ROOT / "outputs" / "TRR-0012"
P10_ROOT = TASK_ROOT.parent / "TRR-P10"
P09_GUARD = P10_ROOT / "scripts" / "trr_p09" / "fixed_control_guarded_launch.py"
QUALIFICATION_ROOT = TASK_EXPERIMENT / "execution" / "qualification_fixed_b1_v2"
CONFIG_PATH = QUALIFICATION_ROOT / "qualification_config.json"
INPUT_BINDING_PATH = QUALIFICATION_ROOT / "qualification_input_binding.json"
PREFLIGHT_PATH = QUALIFICATION_ROOT / "qualification_preflight.json"
LAUNCH_PATH = QUALIFICATION_ROOT / "qualification_launch.json"
WATCHDOG_ROOT = QUALIFICATION_ROOT / "watchdog"
OUTPUT_ROOT = TASK_OUTPUT / "qualification_fixed_b1_v2"

B0_BINDING = P10_ROOT / "experiments" / "TRR-P09" / "setup" / "b0-immutable-loader-binding-r1.json"
B1_MANIFEST = TASK_EXPERIMENT / "capture" / "b1_activation_date07" / "bank_manifest.json"
FIXED_DIAGNOSTIC = P10_ROOT / "experiments" / "TRR-P09" / "setup" / "fixed-diagnostic-binding-r1.json"
SELECTION_SUPPLEMENT = CANONICAL_ROOT / ".worktrees" / "TRR-0010" / "experiments" / "TRR-0010" / "planning" / "stage1_template_compatible_selection_supplement_r1.json"
SHARED_CONTRACT = TASK_EXPERIMENT / "shared_contract_v1.json"
VALIDATION_MANIFEST = TASK_EXPERIMENT / "preparation" / "public_validation_r1" / "validation_manifest.json"
VALIDATION_ROWS = TASK_EXPERIMENT / "preparation" / "public_validation_r1" / "validation_rows.json"
SCHEDULE = TASK_EXPERIMENT / "preparation" / "schedules_r4" / "schedule-b1-seed4010.safetensors"
STARTING_STATE = CANONICAL_ROOT / ".worktrees" / "TRR-0009" / "experiments" / "TRR-0009" / "training" / "run_v1" / "continued_fixed_readout" / "selected.safetensors"
PUBLIC_EMBEDDING = CANONICAL_ROOT / "outputs" / "TRR-0003" / "track_b" / "public_fit_v2" / "public_normalized_embeddings.safetensors"
COORDINATOR_RELEASE = TASK_EXPERIMENT / "execution" / "coordinator_fit_release_v1.json"
CONFIGURATION_DRY_RUN = TASK_EXPERIMENT / "execution" / "configuration_dry_run_r2" / "configuration_dry_run_receipt.json"

SOURCE_COMMIT = "26e08098135c47a5607aff915e8025b57a747e20"
P10_HEAD = "cab7d455c305aa01c3d3bfac225fcffa3e66640b"
START_SHA = "5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14"
EMBEDDING_SHA = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"
B1_SCHEDULE_SHA = "f334697433746814b672e91f1a08cdca8e4810511b6d6648a0d73432e9072e61"
DIAGNOSTIC_SHA = "8b899bc168b55570e4cfd526ff4e8656b70050e840386cba9390ad6adb1f54c3"
SELECTION_SHA = "760adf507438111b03c7308740e6666b0961088e6fc8868f999240d58d2814f7"
CONFIGURATION_DRY_RUN_STATUS = "PASS_B0_AND_B1_NATIVE_CPU_CONFIGURATION"
GIB = 2**30

PROBE_STEPS = 8
FULL_STEPS = 13_000
RECORD_BATCH_SIZE = 8
POSITION_BUDGET = 512
TRAIN_SEQUENCE_TOKENS = 192
VALIDATION_SEQUENCE_TOKENS = 128
HIDDEN_SIZE = 2048
VOCABULARY_SIZE = 128_256
SEED = 4010
LEARNING_RATE = 2.0e-4
WEIGHT_DECAY = 0.0
GRADIENT_CLIP_NORM = 1.0

MAX_SECONDS = 300.0
MAX_RSS_BYTES = 12 * GIB
MIN_HOST_AVAILABLE_BYTES = 8 * GIB
MIN_GPU_FREE_BYTES = 2 * GIB
MAX_GPU_RESERVED_BYTES = 8 * GIB
MIN_DISK_FREE_BYTES = 20 * GIB
MAX_OUTPUT_BYTES = 2 * GIB


class QualificationError(RuntimeError):
    """Raised when the bounded qualifier cannot certify its inputs or run."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"{label} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must be a JSON object: {path}")
    return value


def write_create_only(path: Path, value: Mapping[str, Any]) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise QualificationError(f"create-only artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def descriptor_matches(path: Path, expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = launcher_record(path, label=label)
    for key in ("path", "bytes", "sha256"):
        if str(actual.get(key)) != str(expected.get(key)):
            raise QualificationError(
                f"{label} changed after binding: {key} {actual.get(key)!r} != {expected.get(key)!r}"
            )
    return actual


def launcher_record(path: Path, *, label: str) -> dict[str, Any]:
    # Imported lazily so ``--prepare`` remains standard-library-only apart
    # from the task launcher, which itself has no torch/model imports.
    return launcher.record(Path(path), label=label)


def _launcher_module() -> Any:
    global launcher
    try:
        return launcher
    except NameError:
        if str(TASK_ROOT) not in sys.path:
            sys.path.insert(0, str(TASK_ROOT))
        from scripts.trr0012 import run_fixed_pair as launcher_module

        launcher = launcher_module
        return launcher


def _require_status(path: Path, *, label: str, status: str) -> dict[str, Any]:
    value = read_json(path, label=label)
    if value.get("status") != status:
        raise QualificationError(f"{label} status differs: {value.get('status')!r} != {status!r}")
    return value


def _record_optional(path: Path, *, label: str) -> dict[str, Any]:
    return launcher_record(path, label=label)


def build_input_binding() -> dict[str, Any]:
    """Bind all files used by the native B1 probe without opening tensors."""

    global launcher
    launcher = _launcher_module()
    if launcher.git_head(P10_ROOT) != P10_HEAD:
        raise QualificationError(f"P09 worktree head changed: expected {P10_HEAD}")
    source = launcher.source_bindings()
    qualifier_record = launcher_record(Path(__file__), label="fixed qualification source")
    b1 = launcher.validate_b1_manifest(B1_MANIFEST)
    schedule_path, schedule_sha, schedule_semantic = launcher.schedule("B1")
    if schedule_sha != B1_SCHEDULE_SHA:
        raise QualificationError("B1 schedule signed hash differs")
    if schedule_semantic != "8acdb2c4f8e5afba546ad01cbe0adae340eb841c8e3322c919e976434d29ede5":
        raise QualificationError("B1 schedule signed semantic digest differs")
    dry_run = _require_status(
        CONFIGURATION_DRY_RUN,
        label="native B0/B1 configuration dry-run receipt",
        status=CONFIGURATION_DRY_RUN_STATUS,
    )
    release = read_json(COORDINATOR_RELEASE, label="coordinator fit release")
    if release.get("status") != "AUTHORIZED" or release.get("substantive_fit") is not True:
        raise QualificationError("coordinator fit release is not substantive AUTHORIZED")
    if release.get("paid_compute") is True or release.get("evaluation_truth_access") is True:
        raise QualificationError("coordinator fit release crosses a prohibited boundary")
    b0_record = launcher_record(B0_BINDING, label="B0 immutable fit binding")
    b1_record = b1["manifest"]
    fit_asset, combined_fit_sha = launcher.fit_binding(B1_MANIFEST)
    if combined_fit_sha != "13c7442b481ba6bdaad2db14e3fd0b2a4300efefc48653dc6955e937d091c2d2":
        raise QualificationError("combined B0+B1 fit binding differs from the native dry-run")
    embedding = launcher_record(PUBLIC_EMBEDDING, label="public normalized embedding")
    if embedding["sha256"] != EMBEDDING_SHA:
        raise QualificationError("public embedding hash differs from signed fixed E")
    starting_state = launcher_record(STARTING_STATE, label="exact common starting state")
    if starting_state["sha256"] != START_SHA:
        raise QualificationError("starting state hash differs from the sole approved state")
    schedule_record = launcher_record(schedule_path, label="B1 common schedule")
    validation_manifest = launcher_record(VALIDATION_MANIFEST, label="public validation manifest")
    validation_rows = launcher_record(VALIDATION_ROWS, label="public validation rows")
    diagnostic = launcher_record(FIXED_DIAGNOSTIC, label="fixed diagnostic binding")
    if diagnostic["sha256"] != DIAGNOSTIC_SHA:
        raise QualificationError("fixed diagnostic binding hash differs")
    supplement = launcher_record(SELECTION_SUPPLEMENT, label="signed selection supplement")
    if supplement["sha256"] != SELECTION_SHA:
        raise QualificationError("selection supplement hash differs")
    contract = launcher_record(SHARED_CONTRACT, label="TRR-0012 shared contract")
    coordinator = launcher_record(COORDINATOR_RELEASE, label="coordinator fit release")
    dry_run_record = launcher_record(CONFIGURATION_DRY_RUN, label="native configuration dry-run")
    guard_source = launcher_record(P09_GUARD, label="native P09 watchdog source")
    return {
        "schema": "token-reconstruction.trr0012-fixed-qualification-input-binding.v1",
        "task_id": "TRR-0012",
        "status": "READY_FOR_FIXED_QUALIFICATION",
        "created_utc": utc_now(),
        "source": source,
        "qualification_source": qualifier_record,
        "native_watchdog": guard_source,
        "artifacts": {
            "b0_binding": b0_record,
            "b1_manifest": b1_record,
            "b1_sidecars": b1["sidecars"],
            "fit_binding": fit_asset,
            "combined_fit_binding_sha256": combined_fit_sha,
            "schedule": {
                "file": schedule_record,
                "expected_sha256": schedule_sha,
                "semantic_sha256": schedule_semantic,
                "bank": "B1",
                "steps": FULL_STEPS,
                "record_batch_size": RECORD_BATCH_SIZE,
                "position_budget": POSITION_BUDGET,
                "sequence_tokens": TRAIN_SEQUENCE_TOKENS,
            },
            "validation_manifest": validation_manifest,
            "validation_rows": validation_rows,
            "embedding": embedding,
            "starting_state": starting_state,
            "fixed_diagnostic": diagnostic,
            "selection_supplement": supplement,
            "shared_contract": contract,
            "coordinator_release": coordinator,
            "configuration_dry_run": dry_run_record,
        },
        "b1_geometry": b1["geometry"],
        "b1_bank": b1["bank"],
        "training": {
            "probe_steps": PROBE_STEPS,
            "full_steps": FULL_STEPS,
            "seed": SEED,
            "record_batch_size": RECORD_BATCH_SIZE,
            "position_budget": POSITION_BUDGET,
            "train_sequence_tokens": TRAIN_SEQUENCE_TOKENS,
            "validation_sequence_tokens": VALIDATION_SEQUENCE_TOKENS,
            "hidden_size": HIDDEN_SIZE,
            "vocabulary_size": VOCABULARY_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "scheduler": "CosineAnnealingLR(T_max=13000)",
            "compute_base_logits": True,
            "selection_metric": "domain_balanced_token_accuracy",
            "checkpoint_steps": [0, PROBE_STEPS],
            "checkpoint_retained": False,
            "method_id": "continued_fixed_readout",
        },
        "limits": {
            "maximum_discarded_updates": PROBE_STEPS,
            "timeout_seconds": MAX_SECONDS,
            "maximum_group_rss_bytes": MAX_RSS_BYTES,
            "maximum_gpu_reserved_bytes": MAX_GPU_RESERVED_BYTES,
            "minimum_host_available_bytes": MIN_HOST_AVAILABLE_BYTES,
            "minimum_gpu_free_bytes": MIN_GPU_FREE_BYTES,
            "minimum_disk_free_bytes": MIN_DISK_FREE_BYTES,
            "maximum_output_bytes": MAX_OUTPUT_BYTES,
        },
        "paths": {
            "qualification_root": str(QUALIFICATION_ROOT),
            "config": str(CONFIG_PATH),
            "input_binding": str(INPUT_BINDING_PATH),
            "preflight": str(PREFLIGHT_PATH),
            "launch": str(LAUNCH_PATH),
            "watchdog_root": str(WATCHDOG_ROOT),
            "output_root": str(OUTPUT_ROOT),
        },
        "truth_boundary": {
            "evaluation_truth_opened": False,
            "P03_holdout_touched": False,
            "selected_contender_retained": False,
            "checkpoint_written": False,
        },
    }


def build_config(binding: Mapping[str, Any]) -> dict[str, Any]:
    config = {
        "schema": "token-reconstruction.trr0012-fixed-qualification-config.v1",
        "task_id": "TRR-0012",
        "status": "READY_FOR_FIXED_QUALIFICATION",
        "created_utc": utc_now(),
        "source_commit": SOURCE_COMMIT,
        "p10_head": P10_HEAD,
        "input_binding": {"path": str(INPUT_BINDING_PATH)},
        "training": dict(binding["training"]),
        "limits": dict(binding["limits"]),
        "paths": dict(binding["paths"]),
        "device": "cuda",
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "discard_policy": {
            "all_updates_discarded": True,
            "selected_step_discarded": True,
            "checkpoint_callback": None,
            "checkpoint_written": False,
            "output_contains_only_receipts": True,
            "starting_state_reused_read_only": True,
        },
        "truth_boundary": {
            "evaluation_truth_opened": False,
            "P03_holdout_touched": False,
            "source_plaintext_read": False,
        },
    }
    # The binding is kept in a separate create-only file so its hash can be
    # frozen and independently checked without a self-referential JSON field.
    config["input_binding"]["sha256"] = hashlib.sha256(canonical_bytes(binding)).hexdigest()
    return config


def prepare() -> int:
    global launcher
    launcher = _launcher_module()
    if QUALIFICATION_ROOT.exists() and any(QUALIFICATION_ROOT.iterdir()):
        raise QualificationError(f"qualification root is create-only and nonempty: {QUALIFICATION_ROOT}")
    QUALIFICATION_ROOT.mkdir(parents=True, exist_ok=True)
    binding = build_input_binding()
    config = build_config(binding)
    write_create_only(CONFIG_PATH, config)
    # The config's binding digest is canonical over the exact serialized
    # binding object, while this receipt records the config identity too.
    binding_with_config = {
        **binding,
        "config": launcher_record(CONFIG_PATH, label="fixed qualification config"),
    }
    write_create_only(INPUT_BINDING_PATH, binding_with_config)
    # Rewrite is intentionally forbidden; the config points at the original
    # binding digest, so verify the separate receipt before reporting success.
    if config["input_binding"]["sha256"] != hashlib.sha256(
        canonical_bytes(binding)
    ).hexdigest():
        raise QualificationError("input binding digest construction failed")
    print(
        json.dumps(
            {
                "status": "PREPARED",
                "config": str(CONFIG_PATH),
                "input_binding": str(INPUT_BINDING_PATH),
                "config_sha256": launcher_record(CONFIG_PATH, label="fixed qualification config")["sha256"],
                "input_binding_sha256": launcher_record(INPUT_BINDING_PATH, label="fixed qualification input binding")["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _read_mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "MemAvailable:" and fields[1].isdigit():
            return int(fields[1]) * 1024
    raise QualificationError("MemAvailable is unavailable")


def _nvidia_query(query: str, *, compute_apps: bool = False) -> list[list[str]]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-compute-apps={query}" if compute_apps else f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise QualificationError(f"nvidia-smi query failed: {query}") from exc
    rows: list[list[str]] = []
    for line in raw.splitlines():
        if line.strip():
            rows.append([field.strip() for field in line.split(",")])
    return rows


def fresh_preflight() -> dict[str, Any]:
    started = utc_now()
    mem_available = _read_mem_available_bytes()
    disk_free = int(shutil.disk_usage(QUALIFICATION_ROOT.parent).free)
    gpu_rows = _nvidia_query("index,memory.free,memory.total,temperature.gpu,utilization.gpu")
    process_rows = _nvidia_query("pid,process_name,used_memory", compute_apps=True)
    if len(gpu_rows) != 1:
        raise QualificationError(f"expected exactly one visible GPU, got {len(gpu_rows)}")
    if process_rows:
        raise QualificationError(f"conflicting CUDA compute job is active: {process_rows!r}")
    gpu_index, gpu_free_mib, gpu_total_mib, temperature_c, utilization_pct = gpu_rows[0]
    gpu_free = int(float(gpu_free_mib) * 1024 * 1024)
    gpu_total = int(float(gpu_total_mib) * 1024 * 1024)
    if mem_available < MIN_HOST_AVAILABLE_BYTES:
        raise QualificationError(f"host available memory below floor: {mem_available}")
    if disk_free < MIN_DISK_FREE_BYTES:
        raise QualificationError(f"disk free below floor: {disk_free}")
    if gpu_free < MIN_GPU_FREE_BYTES:
        raise QualificationError(f"GPU free memory below floor: {gpu_free}")
    value = {
        "schema": "token-reconstruction.trr0012-fixed-qualification-preflight.v1",
        "task_id": "TRR-0012",
        "status": "PASS_FRESH_EXCLUSIVE_PREFLIGHT",
        "started_utc": started,
        "finished_utc": utc_now(),
        "checks": {
            "no_conflicting_cuda_compute_job": True,
            "host_available_floor_bytes": MIN_HOST_AVAILABLE_BYTES,
            "gpu_free_floor_bytes": MIN_GPU_FREE_BYTES,
            "disk_free_floor_bytes": MIN_DISK_FREE_BYTES,
        },
        "observed": {
            "host_available_bytes": mem_available,
            "disk_free_bytes": disk_free,
            "gpu_index": gpu_index,
            "gpu_free_bytes": gpu_free,
            "gpu_total_bytes": gpu_total,
            "gpu_temperature_c": temperature_c,
            "gpu_utilization_percent": utilization_pct,
            "compute_apps": process_rows,
        },
        "truth_boundary": {
            "model_loaded": False,
            "evaluation_truth_opened": False,
        },
    }
    write_create_only(PREFLIGHT_PATH, value)
    return value


def verify_config_and_binding(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    global launcher
    launcher = _launcher_module()
    config = read_json(config_path, label="fixed qualification config")
    if config.get("schema") != "token-reconstruction.trr0012-fixed-qualification-config.v1":
        raise QualificationError("qualification config schema differs")
    if config.get("task_id") != "TRR-0012" or config.get("status") != "READY_FOR_FIXED_QUALIFICATION":
        raise QualificationError("qualification config is not ready")
    binding_path = Path(str(config.get("input_binding", {}).get("path", INPUT_BINDING_PATH))).expanduser().resolve()
    binding = read_json(binding_path, label="fixed qualification input binding")
    if binding.get("schema") != "token-reconstruction.trr0012-fixed-qualification-input-binding.v1":
        raise QualificationError("qualification input binding schema differs")
    if binding.get("status") != "READY_FOR_FIXED_QUALIFICATION":
        raise QualificationError("qualification input binding is not ready")
    expected_binding_sha = config.get("input_binding", {}).get("sha256")
    # The config digest intentionally excludes the later ``config`` descriptor
    # added to the binding receipt.
    binding_without_config = {key: value for key, value in binding.items() if key != "config"}
    actual_binding_sha = hashlib.sha256(canonical_bytes(binding_without_config)).hexdigest()
    if actual_binding_sha != expected_binding_sha:
        raise QualificationError("qualification input binding digest differs")
    source = launcher.source_bindings()
    if source != binding.get("source"):
        raise QualificationError("P09/task source bindings changed after preparation")
    qualifier_record = binding.get("qualification_source")
    if not isinstance(qualifier_record, Mapping):
        raise QualificationError("qualification source binding is missing")
    descriptor_matches(Path(__file__), qualifier_record, label="fixed qualification source")
    return config, binding


def _verify_receipt_files(binding: Mapping[str, Any]) -> None:
    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise QualificationError("input binding artifacts are missing")
    records = (
        (B0_BINDING, artifacts.get("b0_binding"), "B0 immutable fit binding"),
        (B1_MANIFEST, artifacts.get("b1_manifest"), "B1 bank manifest"),
        (SCHEDULE, artifacts.get("schedule", {}).get("file") if isinstance(artifacts.get("schedule"), Mapping) else None, "B1 common schedule"),
        (VALIDATION_MANIFEST, artifacts.get("validation_manifest"), "public validation manifest"),
        (VALIDATION_ROWS, artifacts.get("validation_rows"), "public validation rows"),
        (PUBLIC_EMBEDDING, artifacts.get("embedding"), "public normalized embedding"),
        (STARTING_STATE, artifacts.get("starting_state"), "exact common starting state"),
        (FIXED_DIAGNOSTIC, artifacts.get("fixed_diagnostic"), "fixed diagnostic binding"),
        (SELECTION_SUPPLEMENT, artifacts.get("selection_supplement"), "signed selection supplement"),
        (SHARED_CONTRACT, artifacts.get("shared_contract"), "TRR-0012 shared contract"),
        (COORDINATOR_RELEASE, artifacts.get("coordinator_release"), "coordinator fit release"),
        (CONFIGURATION_DRY_RUN, artifacts.get("configuration_dry_run"), "native configuration dry-run"),
    )
    for path, expected, label in records:
        if not isinstance(expected, Mapping):
            raise QualificationError(f"input binding lacks {label}")
        descriptor_matches(path, expected, label=label)
    b1 = launcher.validate_b1_manifest(B1_MANIFEST)
    expected_sidecars = artifacts.get("b1_sidecars")
    if not isinstance(expected_sidecars, list) or len(expected_sidecars) != len(b1["sidecars"]):
        raise QualificationError("B1 sidecar inventory changed")
    for index, expected in enumerate(expected_sidecars):
        if not isinstance(expected, Mapping):
            raise QualificationError(f"B1 sidecar binding {index} is malformed")
        descriptor_matches(Path(str(expected["path"])), expected, label=f"B1 shard {index} sidecar")
    if b1["bank"] != binding.get("b1_bank") or b1["geometry"] != binding.get("b1_geometry"):
        raise QualificationError("B1 metadata changed after binding")
    fit_asset, fit_sha = launcher.fit_binding(B1_MANIFEST)
    if fit_asset != artifacts.get("fit_binding") or fit_sha != artifacts.get("combined_fit_binding_sha256"):
        raise QualificationError("combined fit binding changed after preparation")


def _validate_preflight() -> dict[str, Any]:
    value = _require_status(PREFLIGHT_PATH, label="fresh live preflight", status="PASS_FRESH_EXCLUSIVE_PREFLIGHT")
    observed = value.get("observed")
    if not isinstance(observed, Mapping) or observed.get("compute_apps"):
        raise QualificationError("fresh preflight does not prove exclusive CUDA compute")
    if int(observed.get("host_available_bytes", -1)) < MIN_HOST_AVAILABLE_BYTES:
        raise QualificationError("fresh preflight host availability is below the floor")
    if int(observed.get("gpu_free_bytes", -1)) < MIN_GPU_FREE_BYTES:
        raise QualificationError("fresh preflight GPU free memory is below the floor")
    if int(observed.get("disk_free_bytes", -1)) < MIN_DISK_FREE_BYTES:
        raise QualificationError("fresh preflight disk free space is below the floor")
    return value


def run_guarded(config_path: Path) -> int:
    config, binding = verify_config_and_binding(config_path)
    _verify_receipt_files(binding)
    preflight = fresh_preflight()
    child = [
        "env",
        *[f"{key}={value}" for key, value in config["environment"].items()],
        f"PYTHONPATH={P10_ROOT}:{P10_ROOT / 'src'}",
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "child",
        "--config",
        str(config_path.resolve()),
    ]
    guard = [
        "env",
        f"PYTHONPATH={P10_ROOT}:{P10_ROOT / 'src'}",
        sys.executable,
        str(P09_GUARD),
        "--output-root",
        str(WATCHDOG_ROOT),
        "--child-output-root",
        str(OUTPUT_ROOT),
        "--timeout-seconds",
        str(MAX_SECONDS),
        "--poll-seconds",
        "0.5",
        "--max-rss-bytes",
        str(MAX_RSS_BYTES),
        "--min-available-bytes",
        str(MIN_HOST_AVAILABLE_BYTES),
        "--min-disk-free-bytes",
        str(MIN_DISK_FREE_BYTES),
        "--max-output-bytes",
        str(MAX_OUTPUT_BYTES),
        "--kill-grace-seconds",
        "2",
        "--cwd",
        str(P10_ROOT),
        "--label",
        "TRR-0012-fixed-qualification-B1",
        "--watchdog-source",
        str(P09_GUARD),
        "--",
        *child,
    ]
    if WATCHDOG_ROOT.exists() or OUTPUT_ROOT.exists():
        raise QualificationError("qualification watchdog/output path is not create-only")
    started = utc_now()
    completed = subprocess.run(guard, cwd=str(TASK_ROOT), check=False)
    finished = utc_now()
    finish_path = WATCHDOG_ROOT / "finish.json"
    child_receipt = OUTPUT_ROOT / "qualification_receipt.json"
    child_failure = OUTPUT_ROOT / "qualification_failure.json"
    finish = read_json(finish_path, label="native watchdog finish") if finish_path.exists() else None
    if completed.returncode != 0 or not isinstance(finish, Mapping) or finish.get("status") != "PASS":
        launch_status = "QUALIFICATION_FAILED_CLOSED"
    elif not child_receipt.is_file():
        launch_status = "QUALIFICATION_FAILED_NO_CHILD_RECEIPT"
    else:
        launch_status = "QUALIFICATION_PASS"
    launch = {
        "schema": "token-reconstruction.trr0012-fixed-qualification-launch.v1",
        "task_id": "TRR-0012",
        "status": launch_status,
        "started_utc": started,
        "finished_utc": finished,
        "command": guard,
        "child_command": child,
        "return_code": int(completed.returncode),
        "config": launcher_record(config_path, label="fixed qualification config"),
        "input_binding": launcher_record(INPUT_BINDING_PATH, label="fixed qualification input binding"),
        "preflight": launcher_record(PREFLIGHT_PATH, label="fresh live preflight"),
        "watchdog_finish": launcher_record(finish_path, label="native watchdog finish") if finish_path.exists() else None,
        "watchdog_guard": launcher_record(WATCHDOG_ROOT / "resource_guard.json", label="native watchdog guard") if (WATCHDOG_ROOT / "resource_guard.json").exists() else None,
        "child_receipt": launcher_record(child_receipt, label="fixed qualification receipt") if child_receipt.exists() else None,
        "child_failure": launcher_record(child_failure, label="fixed qualification failure") if child_failure.exists() else None,
        "truth_boundary": {"evaluation_truth_opened": False, "P03_holdout_touched": False},
    }
    write_create_only(LAUNCH_PATH, launch)
    print(json.dumps({"status": launch_status, "launch": str(LAUNCH_PATH), "child_receipt": str(child_receipt), "return_code": completed.returncode}, sort_keys=True))
    return int(completed.returncode)


def _child_imports() -> tuple[Any, Any, Any, Any, Any]:
    if str(P10_ROOT) not in sys.path:
        sys.path.insert(0, str(P10_ROOT))
    if str(P10_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(P10_ROOT / "src"))
    import torch
    from scripts.trr_p09 import fixed_control_caller as caller
    from scripts.trr_p09 import fixed_control_cli as native
    from scripts.trr_p09 import fixed_control_runner as runner
    from token_reconstruction.trr_p09_fixed_control_adapter import FixedPublicReadoutHook

    return torch, caller, native, runner, FixedPublicReadoutHook


def run_child(config_path: Path) -> int:
    config, binding = verify_config_and_binding(config_path)
    _verify_receipt_files(binding)
    preflight = _validate_preflight()
    started = time.monotonic()
    started_utc = utc_now()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    tracker: Any = None
    try:
        torch, caller, native, runner, FixedPublicReadoutHook = _child_imports()
        device = torch.device(str(config["device"]))
        if device.type != "cuda" or not torch.cuda.is_available():
            raise QualificationError("fixed qualification requires an available CUDA device")
        tracker = native.ResourceTracker(
            device=device,
            max_rss_bytes=MAX_RSS_BYTES,
            min_host_available_bytes=MIN_HOST_AVAILABLE_BYTES,
            min_gpu_free_bytes=MIN_GPU_FREE_BYTES,
            max_gpu_reserved_bytes=MAX_GPU_RESERVED_BYTES,
        )
        tracker.check("before_native_input_load")
        artifacts = binding["artifacts"]
        prefix_path = B0_BINDING
        addition_path = B1_MANIFEST
        fit_manifest, prefix_rows = native._read_bank_rows(prefix_path)
        addition_manifest, addition_rows = native._read_bank_rows(addition_path)
        fit_rows = [*prefix_rows, *addition_rows]
        if len(fit_rows) != 12_000:
            raise QualificationError(f"combined B0+B1 row count changed: {len(fit_rows)}")
        if int(fit_manifest["bank"].get("expanded_row_origin", -1)) != 0:
            raise QualificationError("B0 fit bank does not begin at global row zero")
        if int(addition_manifest["bank"].get("expanded_row_origin", -1)) != len(prefix_rows):
            raise QualificationError("B1 fit bank is not adjacent to B0")
        valid_mask = native._valid_mask(fit_rows, sequence_tokens=TRAIN_SEQUENCE_TOKENS)
        expected_mask_sha = native.tensor_digest(valid_mask)
        schedule = caller.load_serialized_schedule(
            SCHEDULE,
            expected_seed=SEED,
            expected_steps=FULL_STEPS,
            expected_record_batch_size=RECORD_BATCH_SIZE,
            expected_position_budget=POSITION_BUDGET,
            expected_sequence_tokens=TRAIN_SEQUENCE_TOKENS,
            expected_global_row_exclusive=len(fit_rows),
            expected_bank="B1",
            expected_valid_mask_semantic_sha256=expected_mask_sha,
            expected_file_sha256=B1_SCHEDULE_SHA,
        )
        prefix_steps = tuple(itertools.islice(schedule.iter_steps(), PROBE_STEPS))
        if len(prefix_steps) != PROBE_STEPS or tuple(step.step for step in prefix_steps) != tuple(range(PROBE_STEPS)):
            raise QualificationError("serialized B1 schedule did not provide a contiguous eight-step prefix")
        prefix_plan = caller.SchedulePlan.from_steps(seed=SEED, steps=prefix_steps)
        prefix_plan.validate(
            record_batch_size=RECORD_BATCH_SIZE,
            position_budget=POSITION_BUDGET,
            sequence_tokens=TRAIN_SEQUENCE_TOKENS,
        )
        tracker.check("after_schedule_and_bank_metadata")

        validation_loader = native.PublicValidationLoader(VALIDATION_MANIFEST, VALIDATION_ROWS)
        labels = validation_loader.label_join
        validation_source = runner.RandomAccessLoaderSource(validation_loader)

        def validation_factory(_step: int, _domain: str, rows: tuple[int, ...]) -> Iterable[Any]:
            if len(rows) % RECORD_BATCH_SIZE:
                raise QualificationError("public validation domain is not batch aligned")
            return [
                validation_source.batch_for_global_rows(rows[index : index + RECORD_BATCH_SIZE])
                for index in range(0, len(rows), RECORD_BATCH_SIZE)
            ]

        validation_callback = caller.make_domain_validation_callback(labels, validation_factory)
        fit_loader = native.CombinedB0StreamedBankLoader(prefix_path, addition_path, device="cpu")
        fit_source = runner.RandomAccessLoaderSource(fit_loader)
        tracker.check("after_public_validation_and_loader_setup")

        embedding, embedding_meta = native._load_embedding(PUBLIC_EMBEDDING, device=device)
        decoder, state_meta = native._load_decoder(STARTING_STATE, device=device)
        hook = FixedPublicReadoutHook(method_id=native.METHOD_ID, embedding_sha256=EMBEDDING_SHA)
        tracker.check("after_fixed_model_and_embedding_load")
        optimizer = torch.optim.AdamW(
            decoder.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            foreach=False,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FULL_STEPS)
        config_obj = runner.RunnerConfig(
            steps=PROBE_STEPS,
            record_batch_size=RECORD_BATCH_SIZE,
            position_budget=POSITION_BUDGET,
            validation_every=PROBE_STEPS,
            selection_metric=runner.DOMAIN_BALANCED_SELECTION_METRIC,
            seed=SEED,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            gradient_clip_norm=GRADIENT_CLIP_NORM,
            train_sequence_tokens=TRAIN_SEQUENCE_TOKENS,
            hidden_size=HIDDEN_SIZE,
            expected_activation_dtype=str(torch.bfloat16),
        )
        tracker.check("after_optimizer_setup")
        remaining = MAX_SECONDS - (time.monotonic() - started)
        if remaining <= 0:
            raise QualificationError("qualification wall cap exhausted before native runner")
        result = runner.run_training(
            decoder,
            hook,
            fit_source,
            prefix_steps,
            schedule_steps_count=PROBE_STEPS,
            schedule_seed=SEED,
            schedule_semantic_sha256=prefix_plan.semantic_sha256,
            schedule_exposure={
                **prefix_plan.exposure_summary(),
                "full_schedule_steps": FULL_STEPS,
                "full_schedule_semantic_sha256": schedule.semantic_sha256,
                "prefix_of_serialized_schedule": True,
            },
            optimizer=optimizer,
            embedding=embedding,
            config=config_obj,
            validation_callback=validation_callback,
            validation_sequence_tokens=VALIDATION_SEQUENCE_TOKENS,
            validation_batch_records=RECORD_BATCH_SIZE,
            validation_activation_dtype=torch.bfloat16,
            training_activation_dtype=torch.bfloat16,
            checkpoint_steps=(0, PROBE_STEPS),
            scheduler=scheduler,
            compute_base_logits=True,
            checkpoint_callback=None,
            deadline_seconds=remaining,
            resource_guard_callback=tracker.check,
        )
        tracker.check("after_discarded_probe")
        output_files = [path for path in OUTPUT_ROOT.rglob("*") if path.is_file()]
        if any(path.name.endswith((".safetensors", ".pt", ".bin")) for path in output_files):
            raise QualificationError("qualification output contains a retained tensor/checkpoint")
        receipt = {
            "schema": "token-reconstruction.trr0012-fixed-native-qualification.v1",
            "task_id": "TRR-0012",
            "status": "QUALIFICATION_PASS",
            "mode": "fixed_public_readout_native_runner",
            "bank": "B1",
            "source_commit": SOURCE_COMMIT,
            "p10_head": P10_HEAD,
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "wall_seconds": time.monotonic() - started,
            "discarded_updates": True,
            "all_updates_discarded": True,
            "contender_selection": False,
            "retained_fitted_arm": False,
            "checkpoint_callback": None,
            "checkpoint_written": False,
            "output_files": [str(path.relative_to(OUTPUT_ROOT)) for path in output_files],
            "selected_step_discarded": result.get("selected_step"),
            "selected_state_sha256_discarded": result.get("selected_state_sha256"),
            "training_contract": {
                "probe_steps": PROBE_STEPS,
                "full_fit_steps": FULL_STEPS,
                "record_batch_size": RECORD_BATCH_SIZE,
                "position_budget": POSITION_BUDGET,
                "train_sequence_tokens": TRAIN_SEQUENCE_TOKENS,
                "validation_sequence_tokens": VALIDATION_SEQUENCE_TOKENS,
                "hidden_size": HIDDEN_SIZE,
                "vocabulary_size": VOCABULARY_SIZE,
                "seed": SEED,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "scheduler": "CosineAnnealingLR(T_max=13000)",
                "compute_base_logits": True,
                "loss": "torch.nn.functional.cross_entropy",
                "selection_metric": runner.DOMAIN_BALANCED_SELECTION_METRIC,
                "validation_domains": list(labels.required_domains),
            },
            "schedule": {
                "full_schedule": launcher_record(SCHEDULE, label="B1 common schedule"),
                "full_steps": FULL_STEPS,
                "full_semantic_sha256": schedule.semantic_sha256,
                "prefix_steps": PROBE_STEPS,
                "prefix_semantic_sha256": prefix_plan.semantic_sha256,
                "prefix_exposure": prefix_plan.exposure_summary(),
                "prefix_is_first_serialized_steps": True,
                "valid_mask_semantic_sha256": expected_mask_sha,
            },
            "inputs": {
                "binding": launcher_record(INPUT_BINDING_PATH, label="fixed qualification input binding"),
                "preflight": launcher_record(PREFLIGHT_PATH, label="fresh live preflight"),
                "embedding": embedding_meta,
                "starting_state": state_meta,
                "combined_fit_binding_sha256": binding["artifacts"]["combined_fit_binding_sha256"],
                "validation_manifest": binding["artifacts"]["validation_manifest"],
                "validation_rows": binding["artifacts"]["validation_rows"],
            },
            "timing": dict(result.get("timing", {})),
            "resource_guard": {
                "in_process_peak": dict(tracker.peak),
                "in_process_low_water": dict(tracker.low_water),
                "checks": list(tracker.checks),
                "external_watchdog": {
                    "root": str(WATCHDOG_ROOT),
                    "limits": {
                        "timeout_seconds": MAX_SECONDS,
                        "max_group_rss_bytes": MAX_RSS_BYTES,
                        "min_host_available_bytes": MIN_HOST_AVAILABLE_BYTES,
                        "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
                        "max_output_bytes": MAX_OUTPUT_BYTES,
                    },
                },
            },
            "learning_curve_discarded": list(result.get("learning_curve", [])),
            "truth_boundary": {
                "evaluation_truth_opened": False,
                "P03_holdout_touched": False,
                "source_plaintext_read": False,
                "public_validation_only": True,
            },
        }
        write_create_only(OUTPUT_ROOT / "qualification_receipt.json", receipt)
        print(json.dumps({"status": receipt["status"], "receipt": str(OUTPUT_ROOT / "qualification_receipt.json"), "wall_seconds": receipt["wall_seconds"]}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": "token-reconstruction.trr0012-fixed-native-qualification-failure.v1",
            "task_id": "TRR-0012",
            "status": "QUALIFICATION_FAILED_CLOSED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "wall_seconds": time.monotonic() - started,
            "discarded_updates": True,
            "contender_selection": False,
            "retained_fitted_arm": False,
            "preflight": preflight,
            "resource_guard": None
            if tracker is None
            else {
                "in_process_peak": dict(tracker.peak),
                "in_process_low_water": dict(tracker.low_water),
                "checks": list(tracker.checks),
            },
            "truth_boundary": {"evaluation_truth_opened": False, "P03_holdout_touched": False},
        }
        failure_path = OUTPUT_ROOT / "qualification_failure.json"
        if not failure_path.exists():
            write_create_only(failure_path, failure)
        raise


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "child"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(list(argv) if argv is not None else None)
    config = Path(args.config).expanduser().resolve()
    if args.mode == "prepare":
        return prepare()
    if args.mode == "run":
        return run_guarded(config)
    return run_child(config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (QualificationError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"TRR-0012 fixed qualification refused: {exc}", file=sys.stderr)
        raise SystemExit(125) from exc
