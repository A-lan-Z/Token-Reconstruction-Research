"""Truth-free diagnostics for the completed TRR-P04 teacher-ranking run.

The module deliberately has no optimizer or fit path.  It reuses the frozen
P04 student/objective helpers, evaluates public correction observations, and
computes no-update objective gradients at stored checkpoints.  All outputs
are create-only so a failed later phase cannot replace earlier evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from .p04_objectives import (
    DEFAULT_HARD_MARGIN,
    DEFAULT_HARD_WEIGHT,
    DEFAULT_RANK_WEIGHT,
    DEFAULT_STUDENT_TEMPERATURE,
    hard_confusion_loss,
    pairwise_teacher_loss,
    student_objective,
)
from .p04_student import (
    ALL_METHODS,
    METHOD_AFFINE,
    METHOD_D,
    METHOD_H,
    METHOD_S,
    AffineStudent,
    P04StudentError,
    StudentArchitectureConfig,
    build_student,
    load_student_state,
    state_tensor_digest,
    validate_embedding_table,
)
from .p04_training import (
    PublicPool,
    TeacherEvidence,
    canonical_hash,
    combine_public_pools,
    file_sha256,
    load_public_pool,
    load_teacher_evidence,
    tensor_sha256,
)


TASK_ID = "TRR-P05"
PLAN_SCHEMA = "token-reconstruction.trr-p05-plan.v1"
SAMPLE_SCHEMA = "token-reconstruction.trr-p05-sample-index.v1"
FORWARD_SCHEMA = "token-reconstruction.trr-p05-forward-row.v1"
GRADIENT_SCHEMA = "token-reconstruction.trr-p05-gradient-cell.v1"
RECEIPT_SCHEMA = "token-reconstruction.trr-p05-diagnostic-receipt.v1"
FAILURE_SCHEMA = "token-reconstruction.trr-p05-diagnostic-failure.v1"
SELECTION_SEED = 20260909
CONTROL_COUNT = 384
TEACHER_COUNT = 384
SCHEDULE_STEPS = (0, 999, 1999, 2999)
METHODS = (METHOD_S, METHOD_H, METHOD_D)
GRADIENT_METHODS = (METHOD_H, METHOD_D)
GRADIENT_CHECKPOINTS = ("selected", "final")
# Exactly three stored gradient states per seed are in scope.  H is included
# as the hard-term comparator; rank at H weights is a diagnostic counterfactual.
GRADIENT_STATE_RULE = ((METHOD_H, "selected"), (METHOD_D, "selected"), (METHOD_D, "final"))
GRADIENT_BATCH_COUNT = len(SCHEDULE_STEPS)
EXPECTED_HIDDEN = 2048
EXPECTED_VOCAB = 128256
EXPECTED_TIME = 192
EXPECTED_CANDIDATE_K = 32
DEFAULT_RECORD_BATCH = 8
DEFAULT_PROJECTION_CHUNK = 512
MIN_FREE_GPU_BYTES = 8 * 1024**3
MAX_RESERVED_GPU_BYTES = 6 * 1024**3
MAX_HOST_RSS_BYTES = 16 * 1024**3


class P05DiagnosticError(RuntimeError):
    """Raised when a P05 input, integrity, or resource contract fails closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def source_commit() -> str | None:
    try:
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value if len(value) == 40 else None


def safe_environment() -> dict[str, str]:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTHONPATH",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


def regular_file(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P05DiagnosticError(f"{label} must be a regular file: {path}")
    return path


def descriptor(path: Path, *, label: str) -> dict[str, Any]:
    path = regular_file(path, label=label)
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": file_sha256(path)}


def write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise P05DiagnosticError(f"output is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_jsonl_create_only(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise P05DiagnosticError(f"output is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False, default=str) + "\n")


def current_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    multiplier = 1024 if sys.platform.startswith("linux") else 1
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * multiplier


def max_rss_bytes() -> int:
    multiplier = 1024 if sys.platform.startswith("linux") else 1
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * multiplier


def runtime_setup(*, seed: int = SELECTION_SEED, threads: int = 4, interop_threads: int = 1) -> None:
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(int(interop_threads))
    except RuntimeError:
        # This is expected when a caller has already executed a torch op.
        pass
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def cuda_snapshot(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": "cpu", "available": False, "free_bytes": None, "total_bytes": None, "max_reserved_bytes": None, "max_allocated_bytes": None}
    if not torch.cuda.is_available():
        raise P05DiagnosticError("CUDA was requested but is unavailable")
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "device": str(device),
        "available": True,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "current_rss_bytes": current_rss_bytes(),
        "max_rss_bytes": max_rss_bytes(),
    }


def check_resources(device: torch.device, *, phase: str, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    snap = {"phase": phase, "utc": utc_now(), **cuda_snapshot(device)}
    snapshots.append(snap)
    rss = int(snap.get("max_rss_bytes") or max_rss_bytes())
    if rss > MAX_HOST_RSS_BYTES:
        raise P05DiagnosticError(f"host RSS guard exceeded during {phase}: {rss} > {MAX_HOST_RSS_BYTES}")
    if device.type == "cuda":
        if int(snap["free_bytes"]) < MIN_FREE_GPU_BYTES:
            raise P05DiagnosticError(f"GPU free-memory guard exceeded during {phase}: {snap['free_bytes']} < {MIN_FREE_GPU_BYTES}")
        if int(snap["max_reserved_bytes"]) > MAX_RESERVED_GPU_BYTES:
            raise P05DiagnosticError(f"GPU reserved-memory guard exceeded during {phase}: {snap['max_reserved_bytes']} > {MAX_RESERVED_GPU_BYTES}")
    return snap


def _load_affine_state(path: Path, *, hidden_size: int) -> dict[str, torch.Tensor]:
    path = regular_file(path, label="affine initialization")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"W", "b", "s"}:
                raise P05DiagnosticError("affine initialization fields changed")
            state = {key: handle.get_tensor(key).contiguous() for key in handle.keys()}
    except P05DiagnosticError:
        raise
    except Exception as exc:
        raise P05DiagnosticError(f"cannot load affine initialization: {path}") from exc
    if state["W"].shape != (hidden_size, hidden_size) or state["b"].shape != (hidden_size,) or state["s"].ndim != 0:
        raise P05DiagnosticError("affine initialization geometry changed")
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise P05DiagnosticError("affine initialization contains non-finite values")
    return state


def _load_candidate_ids(path: Path) -> tuple[torch.Tensor, dict[str, str]]:
    path = regular_file(path, label="candidate preparation")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if "candidate_ids" not in set(handle.keys()):
                raise P05DiagnosticError("candidate preparation lacks candidate_ids")
            values = handle.get_tensor("candidate_ids").contiguous().to(dtype=torch.int64)
            metadata = dict(handle.metadata() or {})
    except P05DiagnosticError:
        raise
    except Exception as exc:
        raise P05DiagnosticError(f"cannot load candidate preparation: {path}") from exc
    if values.ndim != 3 or values.shape[2] != EXPECTED_CANDIDATE_K:
        raise P05DiagnosticError("candidate preparation geometry changed")
    return values, metadata


def _hash_control(seed: int, record_id: str, position: int) -> str:
    payload = f"trr-p05-control-v1\0{seed}\0{record_id}\0{int(position)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _record_order_hash(pool: PublicPool) -> str:
    return canonical_hash(list(pool.record_ids))


def _teacher_lookup(pool: PublicPool, evidence: TeacherEvidence) -> dict[tuple[int, int], int]:
    row_for_id = {record_id: index for index, record_id in enumerate(pool.record_ids)}
    result: dict[tuple[int, int], int] = {}
    for evidence_row, (record_id, position) in enumerate(zip(evidence.record_ids, evidence.positions)):
        if record_id not in row_for_id:
            raise P05DiagnosticError(f"teacher evidence record is outside the public pool: {record_id}")
        key = (row_for_id[record_id], int(position))
        if key in result:
            raise P05DiagnosticError("teacher evidence contains duplicate coordinates")
        if position <= 0 or position >= pool.positions or not bool(pool.valid_mask[key[0], position].item()):
            raise P05DiagnosticError("teacher evidence contains an inactive/BOS coordinate")
        result[key] = evidence_row
    return result


def _teacher_rows_for_sample(pool: PublicPool, evidence: TeacherEvidence) -> tuple[list[dict[str, Any]], dict[tuple[int, int], int]]:
    lookup = _teacher_lookup(pool, evidence)
    rows: list[dict[str, Any]] = []
    for evidence_row, (record_id, position, kind) in enumerate(zip(evidence.record_ids, evidence.positions, evidence.row_kind)):
        row = pool.record_ids.index(record_id)
        rows.append({
            "sample_kind": "teacher",
            "teacher_kind": str(kind),
            "teacher_row": int(evidence_row),
            "record_id": str(record_id),
            "pool_row": int(row),
            "position": int(position),
        })
    counts = Counter(row["teacher_kind"] for row in rows)
    if counts != Counter({"difficult_a1_error": 256, "uniform_audit": 128}):
        raise P05DiagnosticError(f"teacher kind partition changed: {dict(counts)}")
    return rows, lookup


def _control_rows(pool: PublicPool, teacher_lookup: Mapping[tuple[int, int], int], *, seed: int, count: int) -> list[dict[str, Any]]:
    candidates: list[tuple[str, str, int, int]] = []
    for row, record_id in enumerate(pool.record_ids):
        for position in range(1, pool.positions):
            if not bool(pool.valid_mask[row, position].item()) or (row, position) in teacher_lookup:
                continue
            key = _hash_control(seed, record_id, position)
            candidates.append((key, record_id, row, position))
    candidates.sort(key=lambda value: (value[0], value[1], value[3]))
    if len(candidates) < count:
        raise P05DiagnosticError("public correction pool has too few non-teacher controls")
    return [
        {"sample_kind": "control", "teacher_kind": None, "teacher_row": None, "record_id": record_id, "pool_row": int(row), "position": int(position)}
        for _key, record_id, row, position in candidates[:count]
    ]


def _schedule_payload(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
    path = regular_file(path, label="P04 position schedule")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(("record_indices", "selected_mask")) - set(handle.keys()):
                raise P05DiagnosticError("P04 schedule lacks required tensors")
            record_indices = handle.get_tensor("record_indices").contiguous().to(dtype=torch.long)
            selected_mask = handle.get_tensor("selected_mask").contiguous().to(dtype=torch.bool)
            metadata = dict(handle.metadata() or {})
    except P05DiagnosticError:
        raise
    except Exception as exc:
        raise P05DiagnosticError(f"cannot load P04 schedule: {path}") from exc
    if record_indices.ndim != 2 or selected_mask.ndim != 3 or tuple(record_indices.shape) != tuple(selected_mask.shape[:2]):
        raise P05DiagnosticError("P04 schedule geometry changed")
    if selected_mask.shape[2] != EXPECTED_TIME or record_indices.shape[1] != DEFAULT_RECORD_BATCH:
        raise P05DiagnosticError("P04 schedule batch geometry changed")
    return record_indices, selected_mask, metadata


def _gradient_batches(
    combined: PublicPool,
    correction: PublicPool,
    teacher_lookup_combined: Mapping[tuple[int, int], int],
    evidence: TeacherEvidence,
    schedules: Mapping[int, Path],
    *,
    steps: Sequence[int] = SCHEDULE_STEPS,
) -> list[dict[str, Any]]:
    correction_offset = combined.rows - correction.rows
    if correction_offset <= 0:
        raise P05DiagnosticError("combined public pool does not have replay rows")
    result: list[dict[str, Any]] = []
    for seed in sorted(schedules):
        record_indices, selected_masks, metadata = _schedule_payload(schedules[seed])
        expected_order = _record_order_hash(combined)
        if metadata.get("pool_record_order_sha256") not in (None, expected_order):
            raise P05DiagnosticError("P04 schedule record order does not match combined public pool")
        for step in steps:
            if step < 0 or step >= record_indices.shape[0]:
                raise P05DiagnosticError(f"P04 schedule step is outside schedule: {seed}:{step}")
            indices = record_indices[step]
            mask = selected_masks[step]
            if int((indices < correction_offset).sum().item()) != 6 or int((indices >= correction_offset).sum().item()) != 2:
                raise P05DiagnosticError(f"P04 schedule batch {seed}:{step} lost its 6+2 replay/correction composition")
            if mask[:, 0].any().item() or (mask & ~combined.valid_mask.index_select(0, indices)).any().item():
                raise P05DiagnosticError(f"P04 schedule batch {seed}:{step} selects invalid positions")
            records: list[dict[str, Any]] = []
            teacher_count = 0
            teacher_kinds: Counter[str] = Counter()
            for local, pool_row in enumerate(indices.tolist()):
                positions = torch.nonzero(mask[local], as_tuple=False).reshape(-1).tolist()
                row_teacher: list[dict[str, Any]] = []
                for position in positions:
                    evidence_row = teacher_lookup_combined.get((int(pool_row), int(position)))
                    if evidence_row is not None:
                        teacher_count += 1
                        kind = str(evidence.row_kind[evidence_row])
                        teacher_kinds[kind] += 1
                        row_teacher.append({"position": int(position), "teacher_row": int(evidence_row), "teacher_kind": kind})
                records.append({
                    "local_row": int(local),
                    "pool_row": int(pool_row),
                    "record_id": str(combined.record_ids[pool_row]),
                    "pool": "replay" if pool_row < correction_offset else "correction",
                    "selected_positions": [int(value) for value in positions],
                    "teacher_positions": row_teacher,
                })
            result.append({
                "seed": int(seed),
                "step": int(step),
                "schedule": descriptor(schedules[seed], label="P04 position schedule"),
                "record_indices": [int(value) for value in indices.tolist()],
                "selected_rows": int(mask.sum().item()),
                "selected_mask_sha256": tensor_sha256(mask),
                "teacher_rows": int(teacher_count),
                "teacher_kind_counts": dict(sorted(teacher_kinds.items())),
                "records": records,
            })
    return result


def build_sample_index(
    correction: PublicPool,
    combined: PublicPool,
    evidence: TeacherEvidence,
    schedules: Mapping[int, Path],
    *,
    seed: int = SELECTION_SEED,
    control_count: int = CONTROL_COUNT,
    schedule_steps: Sequence[int] = SCHEDULE_STEPS,
) -> dict[str, Any]:
    """Build the create-only public sample ledger before model diagnostics."""

    teacher_rows, correction_teacher_lookup = _teacher_rows_for_sample(correction, evidence)
    correction_offset = combined.rows - correction.rows
    combined_teacher_lookup = {(correction_offset + row, position): evidence_row for (row, position), evidence_row in correction_teacher_lookup.items()}
    controls = _control_rows(correction, correction_teacher_lookup, seed=seed, count=control_count)
    gradient_batches = _gradient_batches(combined, correction, combined_teacher_lookup, evidence, schedules, steps=schedule_steps)
    forward_rows = teacher_rows + controls
    source = {
        "correction_observations": descriptor(Path(correction.source_path), label="correction observations"),
        "correction_records": descriptor(Path(correction.records_path), label="correction records"),
        "teacher_evidence": descriptor(Path(evidence.source_path), label="teacher evidence"),
        "schedule_paths": {str(seed_value): descriptor(path, label="P04 position schedule") for seed_value, path in sorted(schedules.items())},
        "correction_record_order_sha256": _record_order_hash(correction),
        "combined_record_order_sha256": _record_order_hash(combined),
    }
    sample_core = {
        "teacher_rows": forward_rows[:TEACHER_COUNT],
        "control_rows": forward_rows[TEACHER_COUNT:],
        "gradient_batches": gradient_batches,
        "selection_seed": int(seed),
        "schedule_steps": [int(value) for value in schedule_steps],
    }
    return {
        "schema": SAMPLE_SCHEMA,
        "task_id": TASK_ID,
        "status": "PUBLIC_SAMPLE_FROZEN_NO_MODEL_LOADED",
        "plan_schema": PLAN_SCHEMA,
        "selection_seed": int(seed),
        "source": source,
        "forward": {
            "teacher_count": TEACHER_COUNT,
            "control_count": int(control_count),
            "total_count": len(forward_rows),
            "teacher_partition": dict(Counter(row["teacher_kind"] for row in teacher_rows)),
            "rows": forward_rows,
        },
        "gradient": {
            "schedule_steps": [int(value) for value in schedule_steps],
            "batch_count": len(gradient_batches),
            "batches": gradient_batches,
            "composition": "P04 schedule batches with six replay and two correction records",
        },
        "selection_sha256": canonical_hash(sample_core),
    }


def load_sample_index(path: Path) -> dict[str, Any]:
    path = regular_file(path, label="P05 sample index")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P05DiagnosticError(f"cannot parse P05 sample index: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SAMPLE_SCHEMA or payload.get("task_id") != TASK_ID:
        raise P05DiagnosticError("P05 sample index schema/task changed")
    rows = payload.get("forward", {}).get("rows") if isinstance(payload.get("forward"), dict) else None
    if not isinstance(rows, list) or len(rows) != TEACHER_COUNT + CONTROL_COUNT:
        raise P05DiagnosticError("P05 forward sample count changed")
    if payload.get("status") != "PUBLIC_SAMPLE_FROZEN_NO_MODEL_LOADED":
        raise P05DiagnosticError("P05 sample index is not a frozen pre-model ledger")
    return payload


@dataclass(frozen=True)
class StateSpec:
    method_id: str
    seed: int | None
    checkpoint: str
    path: Path
    source: str

    @property
    def state_id(self) -> str:
        if self.seed is None:
            return "affine_initial_function"
        return f"{self.method_id}-{self.seed}-{self.checkpoint}"


def collect_state_specs(manifest_path: Path) -> tuple[list[StateSpec], list[dict[str, Any]]]:
    manifest_path = regular_file(manifest_path, label="P04 state manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P05DiagnosticError(f"cannot parse P04 state manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("task_id") != "TRR-P04":
        raise P05DiagnosticError("P04 state manifest identity changed")
    specs: list[StateSpec] = []
    unavailable: list[dict[str, Any]] = []
    selected = manifest.get("states")
    final = manifest.get("excluded_final_states")
    if not isinstance(selected, list) or not isinstance(final, list):
        raise P05DiagnosticError("P04 state manifest lacks selected/final arrays")
    for checkpoint, values in (("selected", selected), ("final", final)):
        for value in values:
            if not isinstance(value, Mapping):
                continue
            method_id = str(value.get("method_id", ""))
            if method_id not in METHODS:
                continue
            seed = int(value["seed"])
            path = Path(str(value.get("path", ""))).expanduser()
            if path.is_file() and not path.is_symlink():
                specs.append(StateSpec(method_id=method_id, seed=seed, checkpoint=checkpoint, path=path.resolve(), source="p04_stored_checkpoint"))
            else:
                unavailable.append({"method_id": method_id, "seed": seed, "checkpoint": checkpoint, "path": str(path), "reason": "stored checkpoint unavailable"})
    specs.sort(key=lambda spec: (int(spec.seed or 0), spec.method_id, 0 if spec.checkpoint == "selected" else 1))
    if len(specs) != 12:
        # Missing final artifacts are allowed by the packet, but selected S/H/D
        # states must all be present for a complete bounded comparison.
        selected_count = sum(spec.checkpoint == "selected" for spec in specs)
        if selected_count != 6:
            raise P05DiagnosticError(f"P04 selected S/H/D state count changed: {selected_count}")
    return specs, unavailable


def build_initial_affine(path: Path, *, config: StudentArchitectureConfig, device: torch.device) -> AffineStudent:
    state = _load_affine_state(path, hidden_size=config.hidden_size)
    model = AffineStudent(config, affine_state=state).to(device).eval()
    return model


def _rank_layout(
    candidate_ids: torch.Tensor,
    teacher_scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    sigma_q: float,
    tie_tolerance: float,
) -> list[dict[str, Any]]:
    ids = candidate_ids.detach().cpu().to(dtype=torch.int64).tolist()
    scores = teacher_scores.detach().cpu().float().tolist()
    labels_cpu = labels.detach().cpu().to(dtype=torch.int64).tolist()
    rows: list[dict[str, Any]] = []
    for row, (row_ids, row_scores, label) in enumerate(zip(ids, scores, labels_cpu)):
        entries = [(float(score), int(token_id), col) for col, (token_id, score) in enumerate(zip(row_ids, row_scores)) if int(token_id) != int(label)]
        entries.sort(key=lambda item: (-item[0], item[1], item[2]))
        pairs: list[dict[str, Any]] = []
        omitted = 0
        for left, right in zip(entries, entries[1:]):
            delta = float(left[0] - right[0])
            if delta <= float(tie_tolerance):
                omitted += 1
                continue
            pairs.append({"left": int(left[2]), "right": int(right[2]), "sign": 1.0 if delta > 0 else -1.0, "weight": min(abs(delta) / float(sigma_q), 1.0)})
        rows.append({"row": int(row), "pairs": pairs, "omitted_ties": int(omitted), "empty": not bool(pairs)})
    return rows


def _row_rank_metric(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_ids: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    sigma_q: float,
    tie_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    layouts = _rank_layout(candidate_ids, teacher_scores, labels, sigma_q=sigma_q, tie_tolerance=tie_tolerance)
    candidate_logits = logits.gather(1, candidate_ids.to(device=logits.device, dtype=torch.int64))
    all_rows: list[dict[str, Any]] = []
    weighted_sum = torch.zeros((), dtype=logits.dtype, device=logits.device)
    weight_sum = torch.zeros((), dtype=logits.dtype, device=logits.device)
    retained = 0
    omitted = 0
    agreeing = 0
    student_ties = 0
    for row, layout in enumerate(layouts):
        row_weighted = 0.0
        row_weights = 0.0
        row_agree = 0
        row_tie = 0
        for pair in layout["pairs"]:
            margin = candidate_logits[row, int(pair["left"])] - candidate_logits[row, int(pair["right"])]
            term = F.softplus(-float(pair["sign"]) * margin)
            weight = float(pair["weight"])
            weighted_sum = weighted_sum + term * weight
            weight_sum = weight_sum + weight
            row_weighted += float((term.detach() * weight).cpu().item())
            row_weights += weight
            retained += 1
            if (float(margin.detach().cpu().item()) > 0 and pair["sign"] > 0) or (float(margin.detach().cpu().item()) < 0 and pair["sign"] < 0):
                agreeing += 1
                row_agree += 1
            elif float(margin.detach().cpu().item()) == 0:
                student_ties += 1
                row_tie += 1
        omitted += int(layout["omitted_ties"])
        all_rows.append({
            "rank_pairs": len(layout["pairs"]),
            "rank_omitted_ties": int(layout["omitted_ties"]),
            "rank_empty": bool(layout["empty"]),
            "rank_loss": row_weighted / row_weights if row_weights else None,
            "rank_weighted_sum": row_weighted,
            "rank_pair_weight_sum": row_weights,
            "pair_order_agree": int(row_agree),
            "pair_order_ties": int(row_tie),
        })
    aggregate = {
        "rank_rows": sum(bool(layout["pairs"]) for layout in layouts),
        "rank_pairs": int(retained),
        "omitted_tie_pairs": int(omitted),
        "omitted_empty_pairs": sum(bool(layout["empty"]) for layout in layouts),
        "pair_weight_sum": float(weight_sum.detach().cpu().item()),
        "pair_order_agree": int(agreeing),
        "pair_order_student_ties": int(student_ties),
        "pair_order_agreement": (agreeing + 0.5 * student_ties) / retained if retained else None,
        "rank_loss_from_rows": float((weighted_sum / weight_sum.clamp_min(torch.finfo(logits.dtype).eps)).detach().cpu().item()) if retained else 0.0,
    }
    return all_rows, aggregate


def _top2_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maxima = logits.amax(dim=1, keepdim=True)
    tie_count = logits.eq(maxima).sum(dim=1).to(dtype=torch.int32)
    # ``argmax`` returns the first maximal column, which is the required
    # ascending-token-ID tie rule because vocabulary columns are token IDs.
    top1 = logits.argmax(dim=1)
    # The value of the best other token is all that is needed for the margin;
    # topk supplies it without materializing a masked vocabulary-sized copy.
    values = torch.topk(logits, k=2, dim=1, largest=True, sorted=True).values
    labels = labels.to(device=logits.device, dtype=torch.int64)
    gold = logits.gather(1, labels[:, None]).squeeze(1)
    best_other = torch.where(top1.eq(labels), values[:, 1], values[:, 0])
    return top1, tie_count, gold - best_other, gold


def _sample_groups(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = sample["forward"]["rows"]
    return [dict(row) for row in rows]


def forward_state(
    model: torch.nn.Module,
    pool: PublicPool,
    sample: Mapping[str, Any],
    evidence: TeacherEvidence,
    table: torch.Tensor,
    *,
    device: torch.device,
    state_id: str,
    record_batch_size: int = DEFAULT_RECORD_BATCH,
    projection_chunk: int = DEFAULT_PROJECTION_CHUNK,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run full-vocabulary forward diagnostics and retain no logits."""

    rows = _sample_groups(sample)
    by_pool_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pool_row[int(row["pool_row"])].append(row)
    teacher_by_coord = {(pool.record_ids.index(record_id), int(position)): evidence_row for evidence_row, (record_id, position) in enumerate(zip(evidence.record_ids, evidence.positions))}
    output_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(by_pool_row), record_batch_size):
            selected_pool_rows = sorted(by_pool_row)[start : start + record_batch_size]
            activation = pool.observations.index_select(0, torch.tensor(selected_pool_rows, dtype=torch.long)).to(device=device, dtype=torch.float32)
            projected = model.projected_hidden(activation)
            local_coords: list[tuple[int, int, dict[str, Any]]] = []
            for local, pool_row in enumerate(selected_pool_rows):
                for row in by_pool_row[pool_row]:
                    local_coords.append((local, int(row["position"]), row))
            for chunk_start in range(0, len(local_coords), projection_chunk):
                chunk = local_coords[chunk_start : chunk_start + projection_chunk]
                hidden = torch.stack([projected[local, position] for local, position, _row in chunk], dim=0)
                logits = hidden @ table.transpose(0, 1)
                scale = model.logit_scale.float().exp()
                logits = logits * scale
                labels = torch.tensor([int(pool.labels[selected_pool_rows[local], position].item()) for local, position, _row in chunk], dtype=torch.long, device=device)
                top1, ties, margins, gold = _top2_metrics(logits, labels)
                teacher_local: list[int] = []
                for index, (_local, position, row) in enumerate(chunk):
                    if (selected_pool_rows[_local], position) in teacher_by_coord:
                        teacher_local.append(index)
                row_rank_metrics: dict[int, dict[str, Any]] = {}
                chunk_rank_aggregate: dict[str, Any] | None = None
                if teacher_local:
                    idx = torch.tensor(teacher_local, dtype=torch.long, device=device)
                    cand = torch.tensor([evidence.candidate_ids[teacher_by_coord[(selected_pool_rows[chunk[i][0]], chunk[i][1])]].tolist() for i in teacher_local], dtype=torch.long, device=device)
                    scores = torch.tensor([evidence.teacher_scores[teacher_by_coord[(selected_pool_rows[chunk[i][0]], chunk[i][1])]].tolist() for i in teacher_local], dtype=torch.float32, device=device)
                    rank_labels = labels.index_select(0, idx)
                    per_row, chunk_rank_aggregate = _row_rank_metric(logits.index_select(0, idx), rank_labels, cand, scores, sigma_q=evidence.sigma_q, tie_tolerance=evidence.tie_tolerance)
                    original_loss, original_diag = pairwise_teacher_loss(logits.index_select(0, idx), rank_labels, cand, scores, sigma_q=evidence.sigma_q, tie_tolerance=evidence.tie_tolerance)
                    if abs(float(original_loss.detach().cpu().item()) - float(chunk_rank_aggregate["rank_loss_from_rows"])) > 1.0e-5:
                        raise P05DiagnosticError("P04 rank-loss implementation did not reconcile with diagnostic decomposition")
                    chunk_rank_aggregate["p04_reconciled"] = True
                    chunk_rank_aggregate["p04_diagnostics"] = {str(key): int(value) for key, value in original_diag.items()}
                    for local_index, metric in zip(teacher_local, per_row):
                        row_rank_metrics[local_index] = metric
                for index, (local, position, row) in enumerate(chunk):
                    pool_row = selected_pool_rows[local]
                    rank_metric = row_rank_metrics.get(index)
                    output_by_key[(pool_row, position)] = {
                        "schema": FORWARD_SCHEMA,
                        "task_id": TASK_ID,
                        "state_id": state_id,
                        "sample_kind": str(row["sample_kind"]),
                        "teacher_kind": row.get("teacher_kind"),
                        "teacher_row": row.get("teacher_row"),
                        "record_id": str(row["record_id"]),
                        "pool_row": int(pool_row),
                        "position": int(position),
                        "label": int(labels[index].detach().cpu().item()),
                        "predicted_token": int(top1[index].detach().cpu().item()),
                        "tie_count": int(ties[index].detach().cpu().item()),
                        "correct": bool(top1[index].eq(labels[index]).item()),
                        "gold_margin": float(margins[index].detach().cpu().item()),
                        "gold_logit": float(gold[index].detach().cpu().item()),
                        "teacher_order_agreement": None if rank_metric is None else (rank_metric["pair_order_agree"] + 0.5 * rank_metric["pair_order_ties"]) / rank_metric["rank_pairs"] if rank_metric["rank_pairs"] else None,
                        "teacher_pair_order_strict_fraction": None if rank_metric is None else rank_metric["pair_order_agree"] / rank_metric["rank_pairs"] if rank_metric["rank_pairs"] else None,
                        "teacher_pair_count": None if rank_metric is None else int(rank_metric["rank_pairs"]),
                        "teacher_pair_student_ties": None if rank_metric is None else int(rank_metric["pair_order_ties"]),
                        "teacher_omitted_ties": None if rank_metric is None else int(rank_metric["rank_omitted_ties"]),
                        "teacher_rank_loss": None if rank_metric is None else rank_metric["rank_loss"],
                        "teacher_rank_weighted_sum": None if rank_metric is None else rank_metric["rank_weighted_sum"],
                        "teacher_pair_weight_sum": None if rank_metric is None else rank_metric["rank_pair_weight_sum"],
                    }
    if len(output_by_key) != len(rows):
        raise P05DiagnosticError(f"forward output row count mismatch: {len(output_by_key)} != {len(rows)}")
    ordered = [output_by_key[(int(row["pool_row"]), int(row["position"]))] for row in rows]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        if row["sample_kind"] == "teacher":
            groups[str(row["teacher_kind"])].append(row)
        else:
            groups["control"].append(row)
    summary_groups: dict[str, Any] = {}
    for name, values in sorted(groups.items()):
        correct = sum(bool(row["correct"]) for row in values)
        margins = [float(row["gold_margin"]) for row in values]
        ties = sum(int(row["tie_count"]) > 1 for row in values)
        payload: dict[str, Any] = {
            "rows": len(values),
            "correct": correct,
            "accuracy": correct / len(values) if values else None,
            "mean_gold_margin": sum(margins) / len(margins) if margins else None,
            "median_gold_margin": float(torch.tensor(margins, dtype=torch.float64).median().item()) if margins else None,
            "rows_with_top1_tie": ties,
            "top1_tie_count_sum": sum(int(row["tie_count"]) for row in values),
        }
        teacher_values = [row for row in values if row["teacher_pair_count"] is not None]
        if teacher_values:
            pair_count = sum(int(row["teacher_pair_count"]) for row in teacher_values)
            agreement = sum(float(row["teacher_order_agreement"]) * int(row["teacher_pair_count"]) for row in teacher_values) / pair_count if pair_count else None
            pair_weight_sum = sum(float(row["teacher_pair_weight_sum"]) for row in teacher_values)
            weighted_rank_sum = sum(float(row["teacher_rank_weighted_sum"]) for row in teacher_values)
            omitted_ties = sum(int(row["teacher_omitted_ties"]) for row in teacher_values)
            student_ties = sum(int(row["teacher_pair_student_ties"]) for row in teacher_values)
            row_losses = [float(row["teacher_rank_loss"]) for row in teacher_values if row["teacher_rank_loss"] is not None]
            payload.update({
                "teacher_pairs": pair_count,
                "teacher_pair_weight_sum": pair_weight_sum,
                "teacher_order_agreement": agreement,
                "teacher_student_ties": student_ties,
                "teacher_omitted_ties": omitted_ties,
                "teacher_rank_loss_global": weighted_rank_sum / pair_weight_sum if pair_weight_sum else 0.0,
                "teacher_rank_loss_row_mean": sum(row_losses) / len(row_losses) if row_losses else None,
                "teacher_rank_reduction": "global sum(weight*softplus)/global sum(weight)",
            })
        summary_groups[name] = payload
    summary = {
        "schema": "token-reconstruction.trr-p05-forward-summary.v1",
        "task_id": TASK_ID,
        "state_id": state_id,
        "total_rows": len(ordered),
        "groups": summary_groups,
        "correct": sum(bool(row["correct"]) for row in ordered),
        "accuracy": sum(bool(row["correct"]) for row in ordered) / len(ordered),
        "tie_rows": sum(int(row["tie_count"]) > 1 for row in ordered),
        "tie_count_sum": sum(int(row["tie_count"]) for row in ordered),
        "teacher_rows": sum(row["sample_kind"] == "teacher" for row in ordered),
        "control_rows": sum(row["sample_kind"] == "control" for row in ordered),
    }
    return ordered, summary


def _flatten_grads(grads: Sequence[torch.Tensor | None], params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for grad, parameter in zip(grads, params):
        values.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1) if grad is None else grad.detach().float().reshape(-1))
    return torch.cat(values) if values else torch.zeros(0, dtype=torch.float32)


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float()).cpu().item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_norm = torch.linalg.vector_norm(left.float())
    right_norm = torch.linalg.vector_norm(right.float())
    if left_norm.item() == 0.0 or right_norm.item() == 0.0:
        return None
    return float((torch.dot(left.float(), right.float()) / (left_norm * right_norm)).cpu().item())


def _gradient_batch_tensors(
    combined: PublicPool,
    batch: Mapping[str, Any],
    candidate_ids: torch.Tensor,
    teacher_lookup: Mapping[tuple[int, int], int],
    evidence: TeacherEvidence,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    indices = torch.tensor(batch["record_indices"], dtype=torch.long)
    selected = torch.zeros((len(batch["records"]), combined.positions), dtype=torch.bool)
    for record in batch["records"]:
        local = int(record["local_row"])
        selected[local, torch.tensor(record["selected_positions"], dtype=torch.long)] = True
    activation = combined.observations.index_select(0, indices).to(device=device, dtype=torch.float32)
    labels = combined.labels.index_select(0, indices).to(device=device, dtype=torch.long)
    candidates = candidate_ids.index_select(0, indices)[selected].to(device=device, dtype=torch.long)
    teacher_scores = torch.zeros_like(candidates, dtype=torch.float32, device=device)
    rank_mask = torch.zeros((int(selected.sum().item()),), dtype=torch.bool, device=device)
    selected_keys: list[tuple[int, int]] = []
    for local, record in enumerate(batch["records"]):
        for position in record["selected_positions"]:
            selected_keys.append((int(record["pool_row"]), int(position)))
    if len(selected_keys) != int(selected.sum().item()):
        raise P05DiagnosticError("gradient sample selected-position count changed")
    for row, key in enumerate(selected_keys):
        evidence_row = teacher_lookup.get(key)
        if evidence_row is not None:
            rank_mask[row] = True
            teacher_scores[row] = evidence.teacher_scores[evidence_row].to(device=device, dtype=torch.float32)
            expected = evidence.candidate_ids[evidence_row].to(device=device, dtype=torch.long)
            if not torch.equal(candidates[row], expected):
                raise P05DiagnosticError("gradient candidate IDs do not match cached teacher evidence")
    return activation, labels, selected.to(device=device), candidates, teacher_scores, {"rank_mask": rank_mask, "selected_keys": selected_keys}


def gradient_cell(
    model: torch.nn.Module,
    method_id: str,
    combined: PublicPool,
    batch: Mapping[str, Any],
    candidate_ids: torch.Tensor,
    teacher_lookup: Mapping[tuple[int, int], int],
    evidence: TeacherEvidence,
    table: torch.Tensor,
    *,
    device: torch.device,
    state_id: str,
) -> dict[str, Any]:
    """Measure P04 component gradients at one stored state with no update.

    H's *actual* P04 objective is CE + .25 hard.  Its rank gradient is also
    measured at the H weights as a diagnostic counterfactual, so the effect
    of rank can be compared at both the H and D checkpoints without pretending
    that H was trained with teacher scores.  D's actual objective includes all
    three terms.  Neither total is passed to an optimizer.
    """

    activation, labels, selected, candidates, teacher_scores, extra = _gradient_batch_tensors(
        combined, batch, candidate_ids, teacher_lookup, evidence, device=device
    )
    rank_mask = extra["rank_mask"]
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.eval()
    before_digest = state_tensor_digest(model.state_dict())
    projected = model.projected_hidden(activation)
    logits = projected[selected] @ table.transpose(0, 1)
    logits = logits * model.logit_scale.float().exp()
    if not torch.isfinite(logits).all().item():
        raise P05DiagnosticError("gradient diagnostic logits are non-finite")
    row_labels = labels[selected]
    objective = student_objective(
        logits,
        row_labels,
        method_id=method_id,
        candidate_ids=candidates,
        teacher_scores=teacher_scores if method_id == METHOD_D else None,
        rank_mask=rank_mask if method_id == METHOD_D else None,
        hard_weight=DEFAULT_HARD_WEIGHT,
        hard_margin=DEFAULT_HARD_MARGIN,
        rank_weight=DEFAULT_RANK_WEIGHT,
        sigma_q=evidence.sigma_q if method_id == METHOD_D else None,
        tie_tolerance=evidence.tie_tolerance if method_id == METHOD_D else None,
        student_temperature=DEFAULT_STUDENT_TEMPERATURE,
    )
    active_teacher = rank_mask.nonzero(as_tuple=False).reshape(-1)
    if active_teacher.numel():
        diagnostic_rank, rank_diag = pairwise_teacher_loss(
            logits.index_select(0, active_teacher),
            row_labels.index_select(0, active_teacher),
            candidates.index_select(0, active_teacher),
            teacher_scores.index_select(0, active_teacher),
            sigma_q=evidence.sigma_q,
            tie_tolerance=evidence.tie_tolerance,
            student_temperature=DEFAULT_STUDENT_TEMPERATURE,
        )
        rank_layout = _rank_layout(
            candidates.index_select(0, active_teacher).detach().cpu(),
            teacher_scores.index_select(0, active_teacher).detach().cpu(),
            row_labels.index_select(0, active_teacher).detach().cpu(),
            sigma_q=evidence.sigma_q,
            tie_tolerance=evidence.tie_tolerance,
        )
    else:
        diagnostic_rank = logits.sum() * 0.0
        rank_diag = {"rank_rows": 0, "rank_pairs": 0, "omitted_tie_pairs": 0, "omitted_empty_pairs": 0}
        rank_layout = []
    if method_id == METHOD_D:
        if abs(float(objective.rank.detach().cpu().item()) - float(diagnostic_rank.detach().cpu().item())) > 1.0e-5:
            raise P05DiagnosticError("D rank loss did not reconcile with the P04 implementation")
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    ce_grad = _flatten_grads(torch.autograd.grad(objective.ce, params, retain_graph=True, allow_unused=True), params)
    hard_grad = _flatten_grads(torch.autograd.grad(objective.hard, params, retain_graph=True, allow_unused=True), params)
    rank_grad = _flatten_grads(torch.autograd.grad(diagnostic_rank, params, retain_graph=True, allow_unused=True), params)
    actual_total = objective.total if method_id == METHOD_D else objective.ce + DEFAULT_HARD_WEIGHT * objective.hard
    hypothetical_d_total = objective.ce + DEFAULT_HARD_WEIGHT * objective.hard + DEFAULT_RANK_WEIGHT * diagnostic_rank
    actual_grad = _flatten_grads(torch.autograd.grad(actual_total, params, retain_graph=True, allow_unused=True), params)
    hypothetical_grad = _flatten_grads(torch.autograd.grad(hypothetical_d_total, params, retain_graph=True, allow_unused=True), params)
    values = torch.topk(logits, k=2, dim=1, largest=True, sorted=True).values
    top1 = logits.argmax(dim=1)
    gold = logits.gather(1, row_labels[:, None]).squeeze(1)
    best_other = torch.where(top1.eq(row_labels), values[:, 1], values[:, 0])
    # This loss is negative gold margin.  Descent on it increases the desired
    # gold margin, so a positive rank-vs-this cosine is aligned with margin
    # improvement.
    negative_gold_margin = -(gold - best_other).mean()
    margin_grad = _flatten_grads(torch.autograd.grad(negative_gold_margin, params, retain_graph=True, allow_unused=True), params)
    state_after = state_tensor_digest(model.state_dict())
    if state_after != before_digest:
        raise P05DiagnosticError(f"stored state changed during no-update gradient diagnostic: {state_id}")
    weighted_components = ce_grad + DEFAULT_HARD_WEIGHT * hard_grad + DEFAULT_RANK_WEIGHT * rank_grad
    actual_norm = _norm(actual_grad)
    hypothetical_norm = _norm(hypothetical_grad)
    weighted_component_norm = _norm(weighted_components)
    # torch.nn.utils.clip_grad_norm_ uses max_norm/(norm+1e-6), clamped to 1.
    clip_factor = min(1.0, 1.0 / (actual_norm + 1.0e-6)) if actual_norm else 1.0
    pair_weight_sum = sum(float(pair["weight"]) for row in rank_layout for pair in row["pairs"])
    rank_pairs_layout = sum(len(row["pairs"]) for row in rank_layout)
    reductions = objective.diagnostics.as_dict()
    reductions.update({
        "diagnostic_rank_rows": int(rank_diag["rank_rows"]),
        "diagnostic_rank_pairs": int(rank_diag["rank_pairs"]),
        "diagnostic_omitted_tie_pairs": int(rank_diag["omitted_tie_pairs"]),
        "diagnostic_rank_pair_weight_sum": pair_weight_sum,
        "diagnostic_rank_pairs_from_layout": rank_pairs_layout,
        "rank_mask_rows": int(rank_mask.sum().item()),
        "p04_rank_reduction": "global sum(weight*softplus)/global sum(weight)",
        "p04_h_actual_reduction": "CE + 0.25*hard (rank measured counterfactually at H weights)",
    })
    return {
        "schema": GRADIENT_SCHEMA,
        "task_id": TASK_ID,
        "state_id": state_id,
        "method_id": method_id,
        "seed": int(batch["seed"]),
        "schedule_step": int(batch["step"]),
        "selected_rows": int(selected.sum().item()),
        "teacher_rows": int(rank_mask.sum().item()),
        "teacher_kind_counts": dict(batch.get("teacher_kind_counts", {})),
        "losses": {
            "ce": float(objective.ce.detach().cpu().item()),
            "hard": float(objective.hard.detach().cpu().item()),
            "rank": float(diagnostic_rank.detach().cpu().item()),
            "p04_total": float(objective.total.detach().cpu().item()),
            "actual_total": float(actual_total.detach().cpu().item()),
            "hypothetical_d_total": float(hypothetical_d_total.detach().cpu().item()),
            "negative_gold_margin": float(negative_gold_margin.detach().cpu().item()),
        },
        "gradient_norms": {
            "ce": _norm(ce_grad),
            "hard_raw": _norm(hard_grad),
            "hard_weighted": DEFAULT_HARD_WEIGHT * _norm(hard_grad),
            "rank_raw": _norm(rank_grad),
            "rank_weighted": DEFAULT_RANK_WEIGHT * _norm(rank_grad),
            "actual_total_preclip": actual_norm,
            "hypothetical_d_total_preclip": hypothetical_norm,
            "weighted_component_sum": weighted_component_norm,
            "post_clip_actual_total": min(actual_norm, 1.0),
            "clip_factor": clip_factor,
            "negative_gold_margin": _norm(margin_grad),
        },
        "gradient_cosines": {
            "ce_hard_weighted": _cosine(ce_grad, DEFAULT_HARD_WEIGHT * hard_grad),
            "ce_rank_weighted": _cosine(ce_grad, DEFAULT_RANK_WEIGHT * rank_grad),
            "hard_rank_weighted": _cosine(DEFAULT_HARD_WEIGHT * hard_grad, DEFAULT_RANK_WEIGHT * rank_grad),
            "rank_negative_gold_margin": _cosine(rank_grad, margin_grad),
            "actual_total_negative_gold_margin": _cosine(actual_grad, margin_grad),
            "hypothetical_d_total_negative_gold_margin": _cosine(hypothetical_grad, margin_grad),
        },
        "reductions": reductions,
        "state_tensor_digest_before": before_digest,
        "state_tensor_digest_after": state_after,
        "optimizer_step_called": False,
        "parameter_update_applied": False,
    }

def _state_hash(path: Path) -> str:
    return file_sha256(path)


def _validate_candidate_binding(
    candidate: torch.Tensor,
    combined: PublicPool,
    correction: PublicPool,
    evidence: TeacherEvidence,
) -> dict[str, Any]:
    offset = combined.rows - correction.rows
    if tuple(candidate.shape[:2]) != (combined.rows, combined.positions):
        raise P05DiagnosticError("candidate table does not match combined public pool")
    lookup = _teacher_lookup(correction, evidence)
    checked = 0
    for (correction_row, position), evidence_row in lookup.items():
        actual = candidate[offset + correction_row, position]
        expected = evidence.candidate_ids[evidence_row].to(dtype=torch.int64)
        if not torch.equal(actual, expected):
            raise P05DiagnosticError(f"candidate identity mismatch at public correction row {correction_row}:{position}")
        checked += 1
    return {"candidate_shape": list(candidate.shape), "teacher_rows_checked": checked, "candidate_tensor_sha256": tensor_sha256(candidate)}


def prepare_sample_from_paths(
    *,
    correction_observations: Path,
    correction_records: Path,
    replay_observations: Path,
    replay_records: Path,
    teacher_evidence_path: Path,
    schedule_paths: Mapping[int, Path],
    output: Path,
) -> dict[str, Any]:
    correction = load_public_pool(correction_observations, correction_records, embedding_vocab_size=EXPECTED_VOCAB)
    replay = load_public_pool(replay_observations, replay_records, embedding_vocab_size=EXPECTED_VOCAB)
    combined = combine_public_pools(replay, correction)
    evidence = load_teacher_evidence(teacher_evidence_path, expected_candidate_k=EXPECTED_CANDIDATE_K)
    sample = build_sample_index(correction, combined, evidence, schedule_paths)
    write_json_create_only(output, sample)
    return sample


def run_diagnostics(
    *,
    sample_path: Path,
    correction_observations: Path,
    correction_records: Path,
    replay_observations: Path,
    replay_records: Path,
    teacher_evidence_path: Path,
    candidate_preparation: Path,
    embedding_table_path: Path,
    state_manifest_path: Path,
    affine_initial_path: Path,
    schedule_paths: Mapping[int, Path],
    output_root: Path,
    device: torch.device,
    mode: str = "full",
    threads: int = 4,
    interop_threads: int = 1,
) -> dict[str, Any]:
    """Execute qualification or the frozen truth-free diagnostic matrix."""

    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_utc = utc_now()
    snapshots: list[dict[str, Any]] = []
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "task_id": TASK_ID,
        "status": "RUNNING",
        "mode": mode,
        "source_commit": source_commit(),
        "started_utc": started_utc,
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "safe_environment": safe_environment(),
        "no_truth_access": True,
        "optimizer_step_called": False,
        "resource_guard": {"minimum_free_gpu_bytes": MIN_FREE_GPU_BYTES, "maximum_reserved_gpu_bytes": MAX_RESERVED_GPU_BYTES, "maximum_host_rss_bytes": MAX_HOST_RSS_BYTES, "snapshots": snapshots},
    }
    try:
        runtime_setup(seed=SELECTION_SEED, threads=threads, interop_threads=interop_threads)
        if device.type == "cuda":
            if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
                raise P05DiagnosticError("CUDA_VISIBLE_DEVICES must explicitly select the diagnostic GPU")
            if not torch.cuda.is_available():
                raise P05DiagnosticError("CUDA requested but unavailable")
            torch.cuda.reset_peak_memory_stats(device)
        check_resources(device, phase="preflight", snapshots=snapshots)
        sample = load_sample_index(sample_path)
        correction = load_public_pool(correction_observations, correction_records, embedding_vocab_size=EXPECTED_VOCAB)
        replay = load_public_pool(replay_observations, replay_records, embedding_vocab_size=EXPECTED_VOCAB)
        combined = combine_public_pools(replay, correction)
        evidence = load_teacher_evidence(teacher_evidence_path, expected_candidate_k=EXPECTED_CANDIDATE_K)
        correction_teacher_lookup = _teacher_lookup(correction, evidence)
        correction_offset = combined.rows - correction.rows
        combined_teacher_lookup = {(correction_offset + row, position): evidence_row for (row, position), evidence_row in correction_teacher_lookup.items()}
        # Revalidate the sample against the current immutable source inputs.
        regenerated = build_sample_index(correction, combined, evidence, schedule_paths)
        if regenerated["selection_sha256"] != sample.get("selection_sha256"):
            raise P05DiagnosticError("P05 sample index does not match public inputs or schedule ledgers")
        candidate, candidate_metadata = _load_candidate_ids(candidate_preparation)
        candidate_binding = _validate_candidate_binding(candidate, combined, correction, evidence)
        table = load_file(str(regular_file(embedding_table_path, label="public embedding table")), device="cpu").get("embeddings")
        if table is None:
            raise P05DiagnosticError("public embedding table lacks embeddings")
        table = table.contiguous().float()
        validate_embedding_table(table, hidden_size=EXPECTED_HIDDEN, vocab_size=EXPECTED_VOCAB, require_unit_norm=True)
        table_device = table.to(device=device, dtype=torch.float32)
        config = StudentArchitectureConfig(hidden_size=EXPECTED_HIDDEN, vocab_size=EXPECTED_VOCAB, gru_width=256)
        state_specs, unavailable = collect_state_specs(state_manifest_path)
        receipt["inputs"] = {
            "sample_index": descriptor(sample_path, label="P05 sample index"),
            "correction_observations": descriptor(correction_observations, label="correction observations"),
            "correction_records": descriptor(correction_records, label="correction records"),
            "replay_observations": descriptor(replay_observations, label="replay observations"),
            "replay_records": descriptor(replay_records, label="replay records"),
            "teacher_evidence": descriptor(teacher_evidence_path, label="teacher evidence"),
            "candidate_preparation": descriptor(candidate_preparation, label="candidate preparation"),
            "embedding_table": descriptor(embedding_table_path, label="public embedding table"),
            "state_manifest": descriptor(state_manifest_path, label="P04 state manifest"),
            "affine_initial": descriptor(affine_initial_path, label="affine initialization"),
            "schedules": {str(seed): descriptor(path, label="P04 position schedule") for seed, path in sorted(schedule_paths.items())},
        }
        receipt["candidate_binding"] = {**candidate_binding, "metadata_keys": sorted(candidate_metadata)}
        receipt["sample"] = {"selection_sha256": sample["selection_sha256"], "forward_count": len(sample["forward"]["rows"]), "gradient_batch_count": len(sample["gradient"]["batches"])}
        receipt["states"] = {"unavailable": unavailable, "forward": [], "gradient": []}
        check_resources(device, phase="public_assets_loaded", snapshots=snapshots)
        # First forward state is the exact affine initial-function reference.
        initial_model = build_initial_affine(affine_initial_path, config=config, device=device)
        initial_digest = state_tensor_digest(initial_model.state_dict())
        forward_rows, forward_summary = forward_state(initial_model, correction, sample, evidence, table_device, device=device, state_id="affine_initial_function")
        if state_tensor_digest(initial_model.state_dict()) != initial_digest:
            raise P05DiagnosticError("affine initial-function state changed during forward diagnostic")
        forward_path = output_root / "forward-affine_initial_function.jsonl"
        summary_path = output_root / "summary-affine_initial_function.json"
        write_jsonl_create_only(forward_path, forward_rows)
        write_json_create_only(summary_path, forward_summary)
        receipt["states"]["forward"].append({"state_id": "affine_initial_function", "source": "recorded_affine_initial", "forward_rows": str(forward_path), "summary": str(summary_path), "state_tensor_digest": initial_digest})
        del initial_model
        check_resources(device, phase="forward_affine_initial_function", snapshots=snapshots)
        if mode == "qualify":
            # Qualify the largest observed backward cell from the frozen
            # schedule ledger: most teacher-active, then most selected rows,
            # then lower seed and lower step.  This choice is made before any
            # P05 diagnostic values are computed and is shared with the
            # selected D forward state below.
            qualifier_batch = max(
                sample["gradient"]["batches"],
                key=lambda batch: (
                    int(batch.get("teacher_rows", 0)),
                    int(batch.get("selected_rows", 0)),
                    -int(batch["seed"]),
                    -int(batch["step"]),
                ),
            )
            gradient_batches = [qualifier_batch]
            qualifier_seed = int(qualifier_batch["seed"])
            selected_specs = [
                spec
                for spec in state_specs
                if spec.method_id == METHOD_D and spec.seed == qualifier_seed and spec.checkpoint == "selected"
            ]
        else:
            selected_specs = list(state_specs)
            gradient_batches = list(sample["gradient"]["batches"])
        for spec in selected_specs:
            model = load_student_state(spec.path, method_id=spec.method_id, device=device, config=config)
            before = state_tensor_digest(model.state_dict())
            rows, summary = forward_state(model, correction, sample, evidence, table_device, device=device, state_id=spec.state_id)
            after = state_tensor_digest(model.state_dict())
            if before != after:
                raise P05DiagnosticError(f"stored state changed during forward diagnostic: {spec.state_id}")
            forward_path = output_root / f"forward-{spec.state_id}.jsonl"
            summary_path = output_root / f"summary-{spec.state_id}.json"
            write_jsonl_create_only(forward_path, rows)
            write_json_create_only(summary_path, summary)
            receipt["states"]["forward"].append({"state_id": spec.state_id, "method_id": spec.method_id, "seed": spec.seed, "checkpoint": spec.checkpoint, "state": descriptor(spec.path, label="stored P04 checkpoint"), "forward_rows": str(forward_path), "summary": str(summary_path), "state_tensor_digest": before})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            check_resources(device, phase=f"forward_{spec.state_id}", snapshots=snapshots)
        if mode == "qualify":
            gradient_specs = [
                spec
                for spec in state_specs
                if spec.method_id == METHOD_D and spec.seed == qualifier_seed and spec.checkpoint == "selected"
            ]
        else:
            gradient_specs = [
                spec
                for spec in state_specs
                if (spec.method_id, spec.checkpoint) in GRADIENT_STATE_RULE and spec.seed in (1737, 2711)
            ]
        for spec in gradient_specs:
            model = load_student_state(spec.path, method_id=spec.method_id, device=device, config=config)
            batches = [batch for batch in gradient_batches if int(batch["seed"]) == int(spec.seed)]
            for batch in batches:
                cell = gradient_cell(model, spec.method_id, combined, batch, candidate, combined_teacher_lookup, evidence, table_device, device=device, state_id=spec.state_id)
                cell["state"] = descriptor(spec.path, label="stored P04 checkpoint")
                cell_path = output_root / f"gradient-{spec.state_id}-step{int(batch['step']):04d}.json"
                write_json_create_only(cell_path, cell)
                receipt["states"]["gradient"].append({"state_id": spec.state_id, "step": int(batch["step"]), "path": str(cell_path), "teacher_rows": cell["teacher_rows"], "selected_rows": cell["selected_rows"]})
                check_resources(device, phase=f"gradient_{spec.state_id}_step{int(batch['step']):04d}", snapshots=snapshots)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        receipt["status"] = "PASS"
        receipt["ended_utc"] = utc_now()
        receipt["wall_seconds"] = time.perf_counter() - started
        receipt["peak_rss_bytes"] = max_rss_bytes()
        receipt["resource_guard"]["status"] = "PASS"
        receipt_path = output_root / "diagnostic_receipt.json"
        receipt["receipt_path"] = str(receipt_path)
        write_json_create_only(receipt_path, receipt)
        return receipt
    except Exception as exc:
        receipt["status"] = "FAIL"
        receipt["ended_utc"] = utc_now()
        receipt["wall_seconds"] = time.perf_counter() - started
        receipt["peak_rss_bytes"] = max_rss_bytes()
        receipt["error_type"] = type(exc).__name__
        receipt["error"] = str(exc)
        receipt["resource_guard"]["status"] = "FAIL_CLOSED"
        failure_path = output_root / "failure.json"
        if not failure_path.exists() and not failure_path.is_symlink():
            write_json_create_only(failure_path, {"schema": FAILURE_SCHEMA, **receipt, "failure_path": str(failure_path)})
        raise


def parse_schedule_args(values: Sequence[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise P05DiagnosticError("schedule arguments must be SEED=PATH")
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in result:
            raise P05DiagnosticError(f"duplicate schedule seed: {seed}")
        result[seed] = Path(path_text)
    if set(result) != {1737, 2711}:
        raise P05DiagnosticError("P05 requires schedules for seeds 1737 and 2711")
    return result


__all__ = [
    "CONTROL_COUNT",
    "DEFAULT_PROJECTION_CHUNK",
    "DEFAULT_RECORD_BATCH",
    "GRADIENT_CHECKPOINTS",
    "GRADIENT_METHODS",
    "GRADIENT_STATE_RULE",
    "SCHEDULE_STEPS",
    "SELECTION_SEED",
    "StateSpec",
    "P05DiagnosticError",
    "build_sample_index",
    "collect_state_specs",
    "gradient_cell",
    "load_sample_index",
    "parse_schedule_args",
    "prepare_sample_from_paths",
    "run_diagnostics",
    "runtime_setup",
]
