"""Bounded public-prefix teacher qualification for TRR-P04.

The teacher is a training-only, privileged public-prefix diagnostic. It uses
public labels to construct a cache through the requested row, then simulates
only the fixed K=32 A1 proposal candidates at that row. Its numeric scores are
frozen as relative supervision; no teacher state or candidate data is needed by
P04 deployed inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import resource
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from .p04_student import (
    BOS_TOKEN_ID,
    METHOD_AFFINE,
    P04StudentError,
    StudentArchitectureConfig,
    build_student,
    validate_embedding_table,
)
from .p04_training import (
    CANDIDATE_PREPARATION_SCHEMA,
    DEFAULT_CANDIDATE_K,
    _deterministic_topk,
    TEACHER_EVIDENCE_SCHEMA,
    P04TrainingError,
    PublicPool,
    canonical_hash,
    file_sha256,
    tensor_sha256,
)


TEACHER_SCHEMA = TEACHER_EVIDENCE_SCHEMA
CANDIDATE_PROPOSER_ID = "p04_public_affine"
CANDIDATE_PROPOSER_RESOURCE = "pr7_public_affine_state"
CANDIDATE_TIE_POLICY = "descending_score_then_ascending_token_id"
PUBLIC_MODEL_SPEC = {
    "id": "meta-llama/Llama-3.2-1B-Instruct",
    "revision": "9213176726f574b556790deb65791e0c5aa438b6",
    "prefix_layers": [0, 1, 2, 3],
    "dtype": "bfloat16",
    "attention_implementation": "sdpa",
    "local_files_only": True,
}


def prepare_candidate_ids(
    pool: PublicPool,
    embedding_table: torch.Tensor,
    *,
    affine_state: Mapping[str, torch.Tensor],
    affine_path: Path,
    embedding_path: Path,
    output_path: Path,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    proposal_k: int = 512,
    device: torch.device = torch.device("cpu"),
    record_batch_size: int = 8,
    projection_chunk: int = 256,
) -> dict[str, Any]:
    """Create the sole proposer artifact consumed by teacher, H, and D.

    The frozen PR7 affine decoder is evaluated once over the public training
    pool. Its deterministic top-512 rows and K=32 prefix are retained in one
    immutable artifact. Teacher qualification consumes these exact rows; it
    does not call a second proposer.
    """

    if candidate_k != DEFAULT_CANDIDATE_K or proposal_k != 512:
        raise P04TeacherError("P04 candidate budgets are fixed at K=32 and proposal K=512")
    if record_batch_size <= 0 or projection_chunk <= 0:
        raise P04TeacherError("candidate preparation batching must be positive")
    output_path = output_path.expanduser().resolve()
    if output_path.exists() or output_path.is_symlink():
        raise P04TeacherError(f"candidate preparation is create-only: {output_path}")
    if embedding_table.ndim != 2 or embedding_table.shape[1] != 2048 or embedding_table.shape[0] != 128256:
        raise P04TeacherError("candidate preparation embedding geometry changed")
    validate_embedding_table(embedding_table, hidden_size=2048, vocab_size=128256, require_unit_norm=True)
    if set(affine_state) != {"W", "b", "s"}:
        raise P04TeacherError("candidate proposer affine state must contain W, b, and s")
    config = StudentArchitectureConfig(hidden_size=2048, vocab_size=128256, gru_width=256)
    try:
        proposer = build_student(METHOD_AFFINE, config=config, affine_state=affine_state).to(device).eval()
    except Exception as exc:
        raise P04TeacherError("candidate proposer affine state failed geometry checks") from exc
    table = embedding_table.to(device=device, dtype=torch.float32)
    candidate_tensor = torch.empty((pool.rows, pool.positions, candidate_k), dtype=torch.int32)
    proposal_tensor = torch.empty((pool.rows, pool.positions, proposal_k), dtype=torch.int32)
    confidence_tensor = torch.empty((pool.rows, pool.positions), dtype=torch.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, pool.rows, record_batch_size):
            stop = min(start + record_batch_size, pool.rows)
            activation = pool.observations[start:stop].to(device=device, dtype=torch.float32)
            hidden = proposer.projected_hidden(activation).reshape(-1, config.hidden_size)
            flat_start = start * pool.positions
            for chunk_start in range(0, int(hidden.shape[0]), projection_chunk):
                chunk_stop = min(chunk_start + projection_chunk, int(hidden.shape[0]))
                logits = hidden[chunk_start:chunk_stop] @ table.transpose(0, 1)
                logits = logits * proposer.logit_scale.float().exp()
                ids = _deterministic_topk(logits, proposal_k)
                top_values = logits.gather(1, ids[:, :2].to(device=device, dtype=torch.long))
                confidence = torch.sigmoid(top_values[:, 0] - top_values[:, 1])
                destination = slice(flat_start + chunk_start, flat_start + chunk_stop)
                proposal_tensor.reshape(-1, proposal_k)[destination] = ids.detach().cpu().to(torch.int32)
                candidate_tensor.reshape(-1, candidate_k)[destination] = ids[:, :candidate_k].detach().cpu().to(torch.int32)
                confidence_tensor.reshape(-1)[destination] = confidence.detach().cpu().float()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": CANDIDATE_PREPARATION_SCHEMA,
        "task_id": "TRR-P04",
        "proposer_id": CANDIDATE_PROPOSER_ID,
        "proposer_resource": CANDIDATE_PROPOSER_RESOURCE,
        "tie_policy": CANDIDATE_TIE_POLICY,
        "candidate_k": str(candidate_k),
        "proposal_k": str(proposal_k),
        "a1_ranked_k": str(proposal_k),
        "pool_record_ids_json": json.dumps(list(pool.record_ids), separators=(",", ":")),
        "pool_rows": str(pool.rows),
        "pool_positions": str(pool.positions),
        "pool_observation_path": pool.source_path,
        "pool_record_order_sha256": canonical_hash(list(pool.record_ids)),
        "pool_observation_sha256": pool.source_sha256,
        "pool_records_sha256": pool.records_sha256,
        "embedding_path": str(embedding_path.expanduser().resolve()),
        "embedding_file_sha256": file_sha256(embedding_path),
        "embedding_tensor_sha256": tensor_sha256(embedding_table),
        "affine_path": str(affine_path.expanduser().resolve()),
        "affine_file_sha256": file_sha256(affine_path),
        "affine_state_tensor_sha256": canonical_hash({key: tensor_sha256(value) for key, value in sorted(affine_state.items())}),
        "device": str(device),
        "record_batch_size": str(record_batch_size),
        "projection_chunk": str(projection_chunk),
    }
    save_file(
        {"candidate_ids": candidate_tensor, "proposal_ids": proposal_tensor, "a1_confidence": confidence_tensor},
        str(output_path),
        metadata=metadata,
    )
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": file_sha256(output_path),
        "candidate_tensor_sha256": tensor_sha256(candidate_tensor),
        "proposal_tensor_sha256": tensor_sha256(proposal_tensor),
        "confidence_tensor_sha256": tensor_sha256(confidence_tensor),
        "proposer_id": CANDIDATE_PROPOSER_ID,
        "proposer_resource": CANDIDATE_PROPOSER_RESOURCE,
        "tie_policy": CANDIDATE_TIE_POLICY,
        "rows": pool.rows,
        "positions": pool.positions,
        "candidate_k": candidate_k,
        "proposal_k": proposal_k,
        "wall_seconds": time.perf_counter() - started,
    }


def _load_candidate_preparation(
    path: Path,
    *,
    rows: int | None = None,
    positions: int | None = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, str]]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P04TeacherError(f"candidate preparation must be a regular file: {path}")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            required = {"candidate_ids", "proposal_ids", "a1_confidence"}
            if not required.issubset(keys):
                raise P04TeacherError("candidate preparation lacks canonical proposer tensors")
            candidates = handle.get_tensor("candidate_ids").contiguous()
            proposals = handle.get_tensor("proposal_ids").contiguous()
            confidence = handle.get_tensor("a1_confidence").contiguous()
            metadata = dict(handle.metadata() or {})
    except P04TeacherError:
        raise
    except Exception as exc:
        raise P04TeacherError(f"cannot load candidate preparation: {path}") from exc
    if metadata.get("schema") != CANDIDATE_PREPARATION_SCHEMA:
        raise P04TeacherError("candidate preparation schema changed")
    if metadata.get("proposer_id") != CANDIDATE_PROPOSER_ID:
        raise P04TeacherError("candidate preparation proposer identity is not the frozen P04 affine proposer")
    if metadata.get("proposer_resource") != CANDIDATE_PROPOSER_RESOURCE:
        raise P04TeacherError("candidate preparation proposer resource is not the frozen PR7 affine state")
    if metadata.get("tie_policy") != CANDIDATE_TIE_POLICY:
        raise P04TeacherError("candidate preparation tie policy changed")
    if metadata.get("candidate_k") != str(candidate_k) or metadata.get("proposal_k") != "512":
        raise P04TeacherError("candidate preparation budgets changed")
    if candidates.ndim != 3 or proposals.ndim != 3 or confidence.ndim != 2:
        raise P04TeacherError("candidate preparation tensors must be rank 3, rank 3, and rank 2")
    if candidates.shape[2] != candidate_k or proposals.shape[2] != 512 or confidence.shape != proposals.shape[:2] or candidates.shape[:2] != proposals.shape[:2]:
        raise P04TeacherError("candidate preparation geometry changed")
    if rows is not None and candidates.shape[0] != rows:
        raise P04TeacherError("candidate preparation row count does not match requested pool")
    if positions is not None and candidates.shape[1] != positions:
        raise P04TeacherError("candidate preparation position count does not match requested pool")
    if not torch.isfinite(confidence).all().item() or confidence.lt(0).any().item() or confidence.gt(1).any().item():
        raise P04TeacherError("candidate confidence is invalid")
    if candidates.dtype not in (torch.int32, torch.int64) or proposals.dtype not in (torch.int32, torch.int64):
        raise P04TeacherError("candidate preparation IDs must be integer")
    if candidates.lt(0).any().item() or candidates.ge(128256).any().item() or proposals.lt(0).any().item() or proposals.ge(128256).any().item():
        raise P04TeacherError("candidate preparation IDs are outside the public vocabulary")
    if not torch.equal(candidates, proposals[:, :, :candidate_k].to(dtype=candidates.dtype)):
        raise P04TeacherError("candidate preparation K=32 rows are not the canonical proposal prefix")
    sorted_candidates = candidates.to(dtype=torch.int64).sort(dim=-1).values
    if sorted_candidates.shape[-1] > 1 and sorted_candidates[..., 1:].eq(sorted_candidates[..., :-1]).any().item():
        raise P04TeacherError("candidate preparation contains duplicate candidate IDs")
    return candidates.to(torch.int32), proposals.to(torch.int32), confidence.float(), metadata


class P04TeacherError(RuntimeError):
    """Raised when the bounded public teacher cannot be qualified."""


def _load_reference(path: Path) -> Any:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P04TeacherError(f"teacher reference must be a regular file: {path}")
    spec = importlib.util.spec_from_file_location("trr_p04_frozen_teacher", path)
    if spec is None or spec.loader is None:
        raise P04TeacherError(f"cannot import teacher reference: {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclass decoration in the frozen helper resolves its module through
    # sys.modules; register the isolated name before executing the file.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _selection_rows(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P04TeacherError(f"cannot parse teacher qualification selection: {path}") from exc
    rows = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or not rows:
        raise P04TeacherError("teacher qualification selection must contain rows")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str):
            raise P04TeacherError(f"teacher selection row {index} is malformed")
        key = (str(row["record_id"]), int(row.get("position", -1)))
        if key in seen:
            raise P04TeacherError("teacher selection contains duplicate record/position")
        seen.add(key)
        result.append({"record_id": key[0], "position": key[1], "kind": str(row.get("kind", "qualified"))})
    return result


def _default_selection(
    pool: PublicPool,
    *,
    prepared_proposals: torch.Tensor,
    prepared_index: Mapping[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    """Select fixed public rows from the sole frozen P04 proposer artifact.

    The proposal tensor already contains the PR7-affine-state top-512 rows.
    Reading its top-1 column avoids materializing a full correction-pool by
    vocabulary score matrix (which would be tens of GiB) and makes difficult
    row selection use exactly the proposer that feeds H/D.
    """

    if prepared_proposals.ndim != 3 or prepared_proposals.shape[2] != 512:
        raise P04TeacherError("candidate preparation proposal geometry changed")
    try:
        prepared_rows = [int(prepared_index[record_id]) for record_id in pool.record_ids]
    except KeyError as exc:
        raise P04TeacherError("candidate preparation is missing a correction record") from exc
    row_indices = torch.tensor(prepared_rows, dtype=torch.long)
    top1 = prepared_proposals[row_indices, :, 0].to(dtype=torch.int64, device="cpu")
    labels = pool.labels.to(device="cpu", dtype=torch.int64)
    valid = pool.valid_mask.to(device="cpu", dtype=torch.bool).clone()
    valid[:, 0] = False
    wrong = top1.ne(labels) & valid
    wrong_flat = torch.nonzero(wrong, as_tuple=False).tolist()
    if len(wrong_flat) < 256:
        raise P04TeacherError(f"correction pool has only {len(wrong_flat)} frozen P04 proposer errors; need 256")
    difficult = [(int(row), int(position)) for row, position in wrong_flat[:256]]
    difficult_keys = set(difficult)
    coords = torch.nonzero(valid, as_tuple=False).tolist()
    remaining = [
        (int(row), int(position))
        for row, position in coords
        if (int(row), int(position)) not in difficult_keys
    ]
    if len(remaining) < 128:
        raise P04TeacherError("correction pool has fewer than 128 non-difficult audit rows")
    random.Random(int(seed)).shuffle(remaining)
    selected = [(row, position, "difficult_a1_error") for row, position in difficult]
    selected.extend((row, position, "uniform_audit") for row, position in remaining[:128])
    return [
        {"record_id": pool.record_ids[row], "position": position, "kind": kind}
        for row, position, kind in selected
    ]


def _validate_selection(pool: PublicPool, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {record_id: index for index, record_id in enumerate(pool.record_ids)}
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        record_id = str(row["record_id"])
        position = int(row["position"])
        if record_id not in by_id:
            raise P04TeacherError(f"teacher selection record is outside correction pool: {record_id}")
        index = by_id[record_id]
        valid_length = int(pool.valid_mask[index].sum().item())
        if position <= 0 or position >= valid_length:
            raise P04TeacherError(f"teacher selection position is invalid: {record_id}:{position}")
        key = (record_id, position)
        if key in seen:
            raise P04TeacherError("teacher selection contains duplicate rows")
        seen.add(key)
        result.append({"record_id": record_id, "position": position, "kind": str(row.get("kind", "qualified"))})
    if len(result) != 384:
        raise P04TeacherError(f"teacher qualification requires exactly 384 rows, got {len(result)}")
    difficult = sum(row["kind"] == "difficult_a1_error" for row in result)
    audit = sum(row["kind"] == "uniform_audit" for row in result)
    if difficult != 256 or audit != 128:
        raise P04TeacherError(f"teacher selection kinds must be 256 difficult/128 audit, got {difficult}/{audit}")
    return result


def _build_known_public_cache(precut: Any, labels: torch.Tensor, *, position: int, device: torch.device) -> Any:
    cache = precut.new_cache()
    precut.run_cached(torch.tensor([[BOS_TOKEN_ID]], dtype=torch.long, device=device), cache, 0)
    for logical in range(1, int(position)):
        token = int(labels[logical].item())
        precut.run_cached(torch.tensor([[token]], dtype=torch.long, device=device), cache, logical)
    return cache


def qualify_teacher(
    pool: PublicPool,
    *,
    model_identity: Mapping[str, Any],
    lens_path: Path,
    reference_path: Path,
    candidate_preparation_path: Path,
    embedding_path: Path,
    selection_path: Path | None,
    output_root: Path,
    selection_seed: int = 20260906,
) -> dict[str, Any]:
    """Run the fixed privileged public-prefix teacher and freeze its evidence."""

    if not torch.cuda.is_available():
        raise P04TeacherError("P04 teacher qualification requires CUDA")
    device = torch.device("cuda")
    if dict(model_identity.get("model_spec", PUBLIC_MODEL_SPEC)) != PUBLIC_MODEL_SPEC:
        raise P04TeacherError("public teacher model specification changed")
    reference = _load_reference(reference_path)
    try:
        precut, lens, normalized_embeddings, loaded_device, observed_identity = reference.load_public_teacher(
            PUBLIC_MODEL_SPEC,
            model_identity,
            lens_path=lens_path.expanduser().resolve(),
        )
    except Exception as exc:
        raise P04TeacherError("frozen public teacher failed identity/load checks") from exc
    if loaded_device != device:
        raise P04TeacherError("teacher device changed during load")
    prepared_candidates, prepared_proposals, prepared_confidence, preparation_metadata = _load_candidate_preparation(
        candidate_preparation_path,
        positions=pool.positions,
        candidate_k=DEFAULT_CANDIDATE_K,
    )
    if preparation_metadata.get("embedding_file_sha256") != file_sha256(embedding_path):
        raise P04TeacherError("candidate preparation embedding asset does not match teacher input")
    try:
        prepared_record_ids = json.loads(preparation_metadata["pool_record_ids_json"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise P04TeacherError("candidate preparation lacks its frozen pool record order") from exc
    if not isinstance(prepared_record_ids, list) or not all(isinstance(value, str) and value for value in prepared_record_ids):
        raise P04TeacherError("candidate preparation pool record order is malformed")
    if len(prepared_record_ids) != prepared_candidates.shape[0]:
        raise P04TeacherError("candidate preparation pool record order length is inconsistent")
    if len(set(prepared_record_ids)) != len(prepared_record_ids):
        raise P04TeacherError("candidate preparation pool record order is duplicated")
    if preparation_metadata.get("pool_record_order_sha256") != canonical_hash(prepared_record_ids):
        raise P04TeacherError("candidate preparation pool record order hash is inconsistent")
    prepared_index = {record_id: index for index, record_id in enumerate(prepared_record_ids)}
    missing = [record_id for record_id in pool.record_ids if record_id not in prepared_index]
    if missing:
        raise P04TeacherError(f"candidate preparation is missing correction record {missing[0]}")
    if int(preparation_metadata.get("pool_rows", prepared_candidates.shape[0])) != prepared_candidates.shape[0]:
        raise P04TeacherError("candidate preparation pool row metadata is inconsistent")
    selection = _selection_rows(selection_path)
    if selection is None:
        selection = _default_selection(
            pool,
            prepared_proposals=prepared_proposals,
            prepared_index=prepared_index,
            seed=selection_seed,
        )
    selection = _validate_selection(pool, selection)
    index_by_id = {record_id: row for row, record_id in enumerate(pool.record_ids)}
    candidate_rows: list[torch.Tensor] = []
    score_rows: list[torch.Tensor] = []
    proposal_rows: list[torch.Tensor] = []
    confidence_rows: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    qualification_gaps: list[float] = []
    started = time.perf_counter()
    for selection_row in selection:
        record_id = str(selection_row["record_id"])
        position = int(selection_row["position"])
        pool_row = index_by_id[record_id]
        activation = pool.observations[pool_row]
        labels = pool.labels[pool_row]
        prepared_row = prepared_index[record_id]
        # Candidate identities and their 512-row proposal context were frozen
        # once by the canonical proposer. The teacher only simulates this exact
        # row; it never calls an independent proposer.
        proposals = prepared_proposals[prepared_row, position].contiguous()
        confidence_value = float(prepared_confidence[prepared_row, position].item())
        candidates = prepared_candidates[prepared_row, position].contiguous()
        cache = _build_known_public_cache(precut, labels, position=position, device=device)
        try:
            simulated = reference._simulate_candidates(
                precut,
                cache=cache,
                candidate_ids=candidates,
                position=position,
                device=device,
            )
            scores = reference._centered_cosine_scores(simulated, activation[position].to(device=device))
        except Exception as exc:
            raise P04TeacherError(f"candidate simulation failed at {record_id}:{position}") from exc
        scores = scores.detach().float().cpu().contiguous()
        if not torch.isfinite(scores).all().item():
            raise P04TeacherError("teacher score row is non-finite")
        target = int(labels[position].item())
        teacher_index = int(scores.argmax().item())
        teacher_token = int(candidates[teacher_index].item())
        a1_token = int(candidates[0].item())
        non_gold_scores = scores[candidates.ne(target)]
        ordered_non_gold = torch.sort(non_gold_scores, descending=True).values
        row_gaps = (ordered_non_gold[:-1] - ordered_non_gold[1:]).tolist()
        qualification_gaps.extend(float(gap) for gap in row_gaps if float(gap) > 1.0e-6)
        diagnostics.append({
            "record_id": record_id,
            "position": position,
            "kind": str(selection_row["kind"]),
            "proposal_miss": target not in set(int(value) for value in candidates.tolist()),
            "proposal_miss_k32": target not in set(int(value) for value in candidates.tolist()),
            "proposal_miss_k512": target not in set(int(value) for value in proposals.tolist()),
            "candidate_recall_k32": target in set(int(value) for value in candidates.tolist()),
            "candidate_recall_k512": target in set(int(value) for value in proposals.tolist()),
            "a1_token": a1_token,
            "teacher_token": teacher_token,
            "target": target,
            "a1_correct": a1_token == target,
            "teacher_correct": teacher_token == target,
            "teacher_fix": a1_token != target and teacher_token == target,
            "teacher_introduced_error": a1_token == target and teacher_token != target,
            "score_min": float(scores.min().item()),
            "score_max": float(scores.max().item()),
            "score_span": float((scores.max() - scores.min()).item()),
            "score_tie_count": int(scores.eq(scores.max()).sum().item()),
            "non_gold_adjacent_gap_count": len(row_gaps),
            "finite": True,
        })
        candidate_rows.append(candidates)
        score_rows.append(scores)
        proposal_rows.append(proposals)
        confidence_rows.append(confidence_value)
    candidate_tensor = torch.stack(candidate_rows).to(dtype=torch.int32)
    score_tensor = torch.stack(score_rows).to(dtype=torch.float32)
    proposal_tensor = torch.stack(proposal_rows).to(dtype=torch.int32)
    confidence_tensor = torch.tensor(confidence_rows, dtype=torch.float32)
    if not qualification_gaps:
        raise P04TeacherError("teacher evidence has no nonzero non-gold score gaps")
    sigma_q = float(torch.tensor(qualification_gaps, dtype=torch.float64).median().item())
    tie_tolerance = max(1.0e-6, 0.01 * sigma_q)
    for row_index, (row, scores) in enumerate(zip(diagnostics, score_tensor)):
        target = int(row["target"])
        candidates = candidate_tensor[row_index]
        non_gold_scores = scores[candidates.ne(target)]
        ordered_non_gold = torch.sort(non_gold_scores, descending=True).values
        gaps = [float(value) for value in (ordered_non_gold[:-1] - ordered_non_gold[1:]).tolist()]
        retained = [gap for gap in gaps if gap > tie_tolerance]
        row["non_gold_pairs_above_tie"] = len(retained)
        row["omitted_non_gold_tie_pairs"] = len(gaps) - len(retained)
        row["rank_weight_min"] = min((min(gap / sigma_q, 1.0) for gap in retained), default=0.0)
        row["rank_weight_max"] = max((min(gap / sigma_q, 1.0) for gap in retained), default=0.0)
    rows_json = [
        {"record_id": row["record_id"], "position": row["position"], "kind": row["kind"]}
        for row in selection
    ]
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_path = output_root / "teacher_evidence.safetensors"
    if evidence_path.exists() or evidence_path.is_symlink():
        raise P04TeacherError(f"teacher evidence is create-only: {evidence_path}")
    metadata = {
        "schema": TEACHER_SCHEMA,
        "task_id": "TRR-P04",
        "mode": "privileged_public_prefix",
        "candidate_k": str(DEFAULT_CANDIDATE_K),
        "candidate_prefix_k": str(DEFAULT_CANDIDATE_K),
        "proposal_k": "512",
        "a1_ranked_k": "512",
        "candidate_proposer_id": preparation_metadata.get("proposer_id", ""),
        "candidate_proposer_resource": preparation_metadata.get("proposer_resource", ""),
        "candidate_tie_policy": preparation_metadata.get("tie_policy", ""),
        "rows_json": json.dumps(rows_json, separators=(",", ":"), sort_keys=True),
        "sigma_q": repr(sigma_q),
        "tie_tolerance": repr(tie_tolerance),
        "selection_seed": str(selection_seed),
        "selection_basis": "p04_public_affine_proposal_top1_from_frozen_candidate_artifact",
        "selection_sha256": canonical_hash(rows_json),
        "reference_path": str(reference_path.expanduser().resolve()),
        "reference_sha256": file_sha256(reference_path),
        "candidate_preparation_path": str(candidate_preparation_path.expanduser().resolve()),
        "candidate_preparation_sha256": file_sha256(candidate_preparation_path),
        "candidate_preparation_pool_order_sha256": preparation_metadata.get("pool_record_order_sha256", ""),
        "candidate_preparation_affine_file_sha256": preparation_metadata.get("affine_file_sha256", ""),
        "candidate_preparation_tensor_sha256": tensor_sha256(prepared_candidates),
        "candidate_preparation_proposal_tensor_sha256": tensor_sha256(prepared_proposals),
        "embedding_path": str(embedding_path.expanduser().resolve()),
        "embedding_file_sha256": file_sha256(embedding_path),
        "lens_path": str(lens_path.expanduser().resolve()),
        "lens_sha256": file_sha256(lens_path),
        "model_identity_json": json.dumps(dict(observed_identity), sort_keys=True, default=str),
        "correction_record_order_sha256": canonical_hash(list(pool.record_ids)),
        "correction_observation_sha256": pool.source_sha256,
    }
    save_file(
        {
            "candidate_ids": candidate_tensor,
            "teacher_scores": score_tensor,
            "proposal_ids": proposal_tensor,
            "a1_confidence": confidence_tensor,
        },
        str(evidence_path),
        metadata=metadata,
    )
    metrics = {
        "rows": len(diagnostics),
        "finite_rows": sum(bool(row["finite"]) for row in diagnostics),
        "proposal_miss_rate": sum(bool(row["proposal_miss_k32"]) for row in diagnostics) / len(diagnostics),
        "proposal_miss_rate_k32": sum(bool(row["proposal_miss_k32"]) for row in diagnostics) / len(diagnostics),
        "proposal_miss_rate_k512": sum(bool(row["proposal_miss_k512"]) for row in diagnostics) / len(diagnostics),
        "candidate_recall_k32": sum(bool(row["candidate_recall_k32"]) for row in diagnostics) / len(diagnostics),
        "candidate_recall_k512": sum(bool(row["candidate_recall_k512"]) for row in diagnostics) / len(diagnostics),
        "a1_accuracy": sum(bool(row["a1_correct"]) for row in diagnostics) / len(diagnostics),
        "teacher_accuracy": sum(bool(row["teacher_correct"]) for row in diagnostics) / len(diagnostics),
        "teacher_fixes": sum(bool(row["teacher_fix"]) for row in diagnostics),
        "teacher_introduced_errors": sum(bool(row["teacher_introduced_error"]) for row in diagnostics),
        "rows_with_two_non_gold_pairs_above_tie": sum(
            int(row.get("non_gold_pairs_above_tie", 0)) >= 2 for row in diagnostics
        ),
        "omitted_non_gold_tie_pairs": sum(int(row.get("omitted_non_gold_tie_pairs", 0)) for row in diagnostics),
        "retained_non_gold_pairs": sum(int(row.get("non_gold_pairs_above_tie", 0)) for row in diagnostics),
        "rank_weight_min": min((float(row.get("rank_weight_min", 0.0)) for row in diagnostics), default=0.0),
        "rank_weight_max": max((float(row.get("rank_weight_max", 0.0)) for row in diagnostics), default=0.0),
        "sigma_q": sigma_q,
        "tie_tolerance": tie_tolerance,
        "score_min": float(score_tensor.min().item()),
        "score_max": float(score_tensor.max().item()),
        "score_mean": float(score_tensor.mean().item()),
        "score_tensor_sha256": tensor_sha256(score_tensor),
        "candidate_tensor_sha256": tensor_sha256(candidate_tensor),
        "proposal_tensor_sha256": tensor_sha256(proposal_tensor),
        "candidate_simulations": int(len(diagnostics) * DEFAULT_CANDIDATE_K),
        "wall_seconds": time.perf_counter() - started,
        "rows_detail": diagnostics,
    }
    metrics["informative_gate"] = {
        "all_finite": len(diagnostics) == 384,
        "proposal_miss_at_most_half": metrics["proposal_miss_rate"] <= 0.5,
        "two_non_gold_pairs_at_least_half": metrics["rows_with_two_non_gold_pairs_above_tie"] >= 192,
        "status": "REVIEW_REQUIRED",
    }
    gate = metrics["informative_gate"]
    gate["status"] = "PASS" if all(bool(gate[key]) for key in ("all_finite", "proposal_miss_at_most_half", "two_non_gold_pairs_at_least_half")) else "FAIL"
    metrics_path = output_root / "teacher_qualification.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return {
        "evidence": {"path": str(evidence_path), "bytes": evidence_path.stat().st_size, "sha256": file_sha256(evidence_path)},
        "metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path), **{key: value for key, value in metrics.items() if key != "rows_detail"}},
        "identity": observed_identity,
    }


__all__ = [
    "P04TeacherError",
    "PUBLIC_MODEL_SPEC",
    "CANDIDATE_PROPOSER_ID",
    "CANDIDATE_PROPOSER_RESOURCE",
    "CANDIDATE_TIE_POLICY",
    "TEACHER_SCHEMA",
    "prepare_candidate_ids",
    "qualify_teacher",
]
