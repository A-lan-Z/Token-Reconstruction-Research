"""Bounded synthetic tests for the strict TRR-P03 Stage-1 joint gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from token_reconstruction.trr_p03.io import (  # noqa: E402
    BOS_TOKEN_ID,
    CUT_DEPTH,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    OBSERVATION_INDEX_SCHEMA,
    PREDICTION_SCHEMA,
    file_record,
    freeze_prediction_bundle,
    save_observation_bundle,
    sha256_file,
    write_freeze_receipt,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from trr_p03.validate_stage1 import (  # noqa: E402
    A2_METHOD,
    ANCHOR_IDS,
    BASE_METHODS,
    EXPECTED_CONDITIONS,
    EXPECTED_NUMERICS,
    GENERATION_EVIDENCE_SCHEMA,
    STAGE1_IDS,
    STAGE1_METHODS,
    STAGE1_SEQUENCE_LENGTHS,
    Stage1ValidationError,
    validate_stage1,
)


_LENGTHS = (16, 39, 64, 128)
_IMPLEMENTATION_COMMIT = "synthetic-implementation"


def _tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    descriptor = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(descriptor + b"\0" + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _write_plan(path: Path) -> None:
    write_json_exclusive(
        path,
        {
            "schema": "token-reconstruction.trr-p03-stage1-stage2-plan.v2",
            "task_id": "TRR-P03",
            "truth_opened": False,
            "source_truth_included": False,
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "cut_depth": CUT_DEPTH,
                "hidden_size": HIDDEN_SIZE,
                "vocab_size": 128256,
                "bos_token_id": BOS_TOKEN_ID,
            },
            "conditions": {
                expected["condition_id"]: {
                    "condition_id": expected["condition_id"],
                    "required_for_stage1": True,
                    "target_model_id": expected["model_id"],
                    "target_model_revision": expected["revision"],
                }
                for expected in EXPECTED_CONDITIONS.values()
            },
            "panel": {
                "stage1": {
                    "records": 24,
                    "records_per_length": 6,
                    "post_bos_lengths": list(_LENGTHS),
                    "scored_tokens_total_per_target": 1482,
                },
                "a1_a2_anchor": {
                    "record_ids": list(ANCHOR_IDS),
                    "record_order_indices_zero_based": [0, 2, 4, 5],
                    "post_bos_length": 39,
                },
            },
        },
    )


def _write_observations(root: Path, bundle_id: str, *, geometry_delta: bool = False) -> tuple[Path, dict[str, str]]:
    public = root / f"public-{bundle_id}"
    observations_root = public / "observations" / bundle_id
    observations_root.mkdir(parents=True)
    descriptors: list[dict[str, object]] = []
    observation_hashes: dict[str, str] = {}
    start = 0
    for length in _LENGTHS:
        actual_length = length + 1
        if geometry_delta and bundle_id == "bundle-b" and length == 64:
            actual_length += 1
        ids = list(STAGE1_IDS[start : start + 6])
        start += 6
        count = len(ids)
        activations = torch.zeros((count, actual_length, HIDDEN_SIZE), dtype=torch.bfloat16)
        masks = torch.ones((count, actual_length), dtype=torch.int64)
        positions = torch.arange(actual_length, dtype=torch.int64).view(1, -1).expand(count, -1)
        relative = Path(f"observations/{bundle_id}/stage1_len{length}.safetensors")
        artifact = public / relative
        digest = save_observation_bundle(
            activations=activations,
            attention_mask=masks,
            position_ids=positions,
            path=artifact,
            bundle_id=bundle_id,
            stage="stage1",
            record_ids=ids,
        )
        for record_id in ids:
            observation_hashes[record_id] = digest
        descriptors.append(
            {
                "bundle_id": bundle_id,
                "stage": "stage1",
                "scored_tokens": actual_length - 1,
                "sequence_length": actual_length,
                "record_ids": ids,
                "relative_path": relative.as_posix(),
                "keys": {
                    "activations": "activations",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                },
                "expected_shapes": {
                    "activations": [count, actual_length, HIDDEN_SIZE],
                    "attention_mask": [count, actual_length],
                    "position_ids": [count, actual_length],
                },
                "bytes": artifact.stat().st_size,
                "sha256": digest,
                "mask_digest": _tensor_digest(masks),
                "position_digest": _tensor_digest(positions),
            }
        )
    index_path = public / "observation_index.json"
    write_json_exclusive(
        index_path,
        {
            "schema": OBSERVATION_INDEX_SCHEMA,
            "task_id": "TRR-P03",
            "status": "OBSERVATIONS_READY_BEFORE_RECONSTRUCTION",
            "truth_opened": False,
            "source_truth_included": False,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "cut_depth": CUT_DEPTH,
            "bos_token_id": BOS_TOKEN_ID,
            "bundle_id": bundle_id,
            "record_order": list(STAGE1_IDS),
            "bundles": descriptors,
        },
    )
    return index_path, observation_hashes


def _write_resource_receipt(
    path: Path,
    *,
    bundle_id: str,
    snapshot: Path,
    model_id: str,
    revision: str,
    panel_path: Path,
    observation_index: Path,
) -> None:
    write_json_exclusive(
        path,
        {
            "schema": GENERATION_EVIDENCE_SCHEMA,
            "task_id": "TRR-P03",
            "truth_opened": False,
            "source_truth_included": False,
            "status": "OPAQUE_OBSERVATIONS_WRITTEN_AS_DISTINCT_ARTIFACTS",
            "implementation_commit": _IMPLEMENTATION_COMMIT,
            "environment": {
                "device": "cpu",
                "cuda_visible_devices": "",
                "torch_threads": 8,
                "torch_interop_threads": 1,
                "deterministic_algorithms": True,
            },
            "model": {"id": model_id, "revision": revision, "path": str(snapshot)},
            "bundle": {"bundle_id": bundle_id, "stages": ["stage1"]},
            "panel": {
                "path": str(panel_path),
                "bytes": panel_path.stat().st_size,
                "sha256": sha256_file(panel_path),
                "records": 24,
            },
            "geometry": {
                "records": 24,
                "scored_tokens": 1482,
                "lengths": list(_LENGTHS),
            },
            "resource_guard": {"status": "PASS"},
            "observation_index": file_record(observation_index),
        },
    )


def _write_prediction_root(
    root: Path,
    *,
    plan_path: Path,
    observation_index: Path,
    observation_hashes: dict[str, str],
    assets: Path,
    drop: str | None = None,
    wrong_model_path: bool = False,
) -> None:
    root.mkdir(parents=True)
    source = assets / "source.py"
    prototype = assets / "prototype.bin"
    lens = assets / "lens.bin"
    projected = assets / "projected.bin"
    methods = list(STAGE1_METHODS)
    selected: dict[str, list[str]] = {
        method: list(STAGE1_IDS) for method in BASE_METHODS
    }
    selected[A2_METHOD] = list(ANCHOR_IDS)
    if drop == "method":
        selected["projected_boundary.cosine"] = []
    elif drop == "anchor":
        selected[A2_METHOD] = list(ANCHOR_IDS[:-1])
    elif drop == "record":
        selected["raw_boundary.cosine"] = list(STAGE1_IDS[:-1])

    tensors: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    masks: dict[str, list[int]] = {}
    expected_lengths = dict(zip(STAGE1_IDS, STAGE1_SEQUENCE_LENGTHS, strict=True))
    for method in methods:
        tensor = torch.full((24, 129), -1, dtype=torch.int32)
        ids = selected[method]
        masks[method] = [int(record_id in ids) for record_id in STAGE1_IDS]
        for index, record_id in enumerate(STAGE1_IDS):
            if record_id not in ids:
                continue
            sequence_length = expected_lengths[record_id]
            tensor[index, :sequence_length] = torch.tensor(
                [BOS_TOKEN_ID] + [0] * (sequence_length - 1), dtype=torch.int32
            )
            rows.append(
                {
                    "record_id": record_id,
                    "method": method,
                    "sequence_length": sequence_length,
                    "prediction_tokens": [BOS_TOKEN_ID] + [0] * (sequence_length - 1),
                    "top1_tie_count": [1] * (sequence_length - 1),
                    "top1_scores": [0.5] * (sequence_length - 1),
                    "top1_runner_margins": [0.25] * (sequence_length - 1),
                    "score_units": "synthetic",
                    "observation_sha256": observation_hashes[record_id],
                    "truth_opened": False,
                }
            )
        tensors[method.replace(".", "_")] = tensor

    prediction_path = root / "predictions.safetensors"
    save_file(
        tensors,
        prediction_path,
        metadata={
            "schema": PREDICTION_SCHEMA,
            "task_id": "TRR-P03",
            "truth_opened": "false",
            "methods_json": json.dumps(methods, separators=(",", ":")),
            "field_map_json": json.dumps(
                {method.replace(".", "_"): method for method in methods},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "record_order_json": json.dumps(list(STAGE1_IDS), separators=(",", ":")),
            "sequence_lengths_json": json.dumps(list(STAGE1_SEQUENCE_LENGTHS), separators=(",", ":")),
            "method_masks_json": json.dumps(masks, sort_keys=True, separators=(",", ":")),
        },
    )
    diagnostics_path = root / "lookup_diagnostics.safetensors"
    save_file({"placeholder": torch.zeros(1)}, diagnostics_path)
    candidate_path = root / "candidate_sets.safetensors"
    save_file({"placeholder": torch.zeros(1, dtype=torch.int32)}, candidate_path)
    rows_path = root / "predictions.jsonl"
    write_jsonl_exclusive(rows_path, rows)
    progress_path = root / "phase_progress.jsonl"
    write_jsonl_exclusive(progress_path, [{"truth_opened": False, "event": "ready"}])
    preflight_path = root / "preflight.json"
    write_json_exclusive(
        preflight_path,
        {
            "truth_opened": False,
            "source_truth_included": False,
            "methods": methods,
            "input": file_record(observation_index),
            "resource_guard": {"status": "PASS"},
            "numerics": dict(EXPECTED_NUMERICS) | {"cuda_visible_devices": ""},
        },
    )
    evidence_path = root / "reconstructor_evidence.json"
    write_json_exclusive(
        evidence_path,
        {
            "truth_opened": False,
            "source_truth_included": False,
            "implementation_commit": _IMPLEMENTATION_COMMIT,
            "command": {
                "argv": [
                    "python3",
                    "scripts/trr_p03/reconstruct.py",
                    "--model-path",
                    str(plan_path.parent / "snapshots" / ("shifted" if wrong_model_path else "base")),
                    "--implementation-commit",
                    _IMPLEMENTATION_COMMIT,
                ],
                "cwd": str(plan_path.parent),
            },
            "observation_index": file_record(observation_index),
            "prototype": file_record(prototype),
            "historical_lens": file_record(lens),
            "projected_prototype": file_record(projected),
            "methods": methods,
            "method_metadata": {A2_METHOD: {"anchor_record_ids": list(ANCHOR_IDS)}},
            "records": 24,
            "scored_tokens": 1482,
            "code_files": [file_record(source)],
        },
    )
    freeze = freeze_prediction_bundle(
        root=root,
        plan_hash=sha256_file(plan_path),
        implementation_commit=_IMPLEMENTATION_COMMIT,
        artifacts=[
            prediction_path,
            diagnostics_path,
            candidate_path,
            rows_path,
            progress_path,
            preflight_path,
            evidence_path,
        ],
        metadata={
            "methods": methods,
            "records": 24,
            "record_ids": list(STAGE1_IDS),
            "anchor_record_ids": list(ANCHOR_IDS),
        },
    )
    write_freeze_receipt(root, freeze)


def _make_case(
    tmp_path: Path,
    *,
    drop: str | None = None,
    geometry_delta: bool = False,
    wrong_model_path: bool = False,
) -> dict[str, Path]:
    case = tmp_path / "case"
    case.mkdir()
    plan_path = case / "plan.json"
    _write_plan(plan_path)
    assets = case / "assets"
    assets.mkdir()
    for name, content in {
        "source.py": b"synthetic source\n",
        "prototype.bin": b"prototype\n",
        "lens.bin": b"lens\n",
        "projected.bin": b"projected\n",
    }.items():
        (assets / name).write_bytes(content)
    snapshots = case / "snapshots"
    snapshots.mkdir()
    base_snapshot = snapshots / "base"
    shifted_snapshot = snapshots / "shifted"
    base_snapshot.mkdir()
    shifted_snapshot.mkdir()
    (base_snapshot / "config.json").write_text("{}\n")
    (shifted_snapshot / "config.json").write_text("{}\n")
    panel_path = case / "panel.json"
    write_json_exclusive(panel_path, {"truth_opened": False, "records": 24})
    resource_mapping = case / "resource-map.json"
    write_json_exclusive(
        resource_mapping,
        {
            "schema": "token-reconstruction.trr-p03-evaluator-resource-map.v1",
            "truth_opened": False,
            "targets": {
                "bundle-a": {
                    "condition_id": "matched_public",
                    "model_id": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "snapshot": str(base_snapshot),
                },
                "bundle-b": {
                    "condition_id": "shifted_full_sft",
                    "model_id": EXPECTED_CONDITIONS["bundle-b"]["model_id"],
                    "revision": EXPECTED_CONDITIONS["bundle-b"]["revision"],
                    "snapshot": str(shifted_snapshot),
                },
            },
        },
    )
    observation_indexes: dict[str, Path] = {}
    observation_hashes: dict[str, dict[str, str]] = {}
    for bundle_id in ("bundle-a", "bundle-b"):
        index_path, hashes = _write_observations(
            case, bundle_id, geometry_delta=geometry_delta
        )
        observation_indexes[bundle_id] = index_path
        observation_hashes[bundle_id] = hashes
    resource_receipts: dict[str, Path] = {}
    for bundle_id, snapshot, expected in (
        ("bundle-a", base_snapshot, EXPECTED_CONDITIONS["bundle-a"]),
        ("bundle-b", shifted_snapshot, EXPECTED_CONDITIONS["bundle-b"]),
    ):
        receipt = case / f"{bundle_id}-generation-evidence.json"
        _write_resource_receipt(
            receipt,
            bundle_id=bundle_id,
            snapshot=snapshot,
            model_id=expected["model_id"],
            revision=expected["revision"],
            panel_path=panel_path,
            observation_index=observation_indexes[bundle_id],
        )
        resource_receipts[bundle_id] = receipt
    prediction_roots: dict[str, Path] = {}
    for bundle_id in ("bundle-a", "bundle-b"):
        prediction_root = case / f"predictions-{bundle_id}"
        _write_prediction_root(
            prediction_root,
            plan_path=plan_path,
            observation_index=observation_indexes[bundle_id],
            observation_hashes=observation_hashes[bundle_id],
            assets=assets,
            drop=drop if bundle_id == "bundle-b" else None,
            wrong_model_path=wrong_model_path and bundle_id == "bundle-b",
        )
        prediction_roots[bundle_id] = prediction_root
    return {
        "plan": plan_path,
        "mapping": resource_mapping,
        "observation_a": observation_indexes["bundle-a"],
        "observation_b": observation_indexes["bundle-b"],
        "receipt_a": resource_receipts["bundle-a"],
        "receipt_b": resource_receipts["bundle-b"],
        "prediction_a": prediction_roots["bundle-a"],
        "prediction_b": prediction_roots["bundle-b"],
    }


def _run(case: dict[str, Path], tmp_path: Path, *, prediction_b: Path | None = None) -> dict[str, object]:
    return validate_stage1(
        plan_path=case["plan"],
        observation_index_a=case["observation_a"],
        observation_index_b=case["observation_b"],
        prediction_root_a=case["prediction_a"],
        prediction_root_b=prediction_b or case["prediction_b"],
        resource_mapping=case["mapping"],
        resource_receipt_a=case["receipt_a"],
        resource_receipt_b=case["receipt_b"],
        output_root=tmp_path / "joint-receipt",
        implementation_commit=_IMPLEMENTATION_COMMIT,
    )


def test_joint_validation_writes_create_only_truth_free_receipt(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    result = _run(case, tmp_path)
    receipt_path = Path(str(result["receipt"]["path"]))
    payload = json.loads(receipt_path.read_text())
    assert result["status"] == "VALIDATED"
    assert payload["truth_opened"] is False
    assert payload["validation"] == "STAGE1_JOINT_VALIDATION_PASS"
    assert payload["stage1_record_order"] == list(STAGE1_IDS)
    assert payload["anchor_record_ids"] == list(ANCHOR_IDS)
    assert payload["score_prerequisite"]["paired_prediction_root_required"] is True
    assert receipt_path.stat().st_mode & 0o222 == 0


@pytest.mark.parametrize("drop", ["method", "anchor", "record"])
def test_joint_validation_rejects_incomplete_prediction_matrix(tmp_path: Path, drop: str) -> None:
    case = _make_case(tmp_path, drop=drop)
    with pytest.raises(Stage1ValidationError):
        _run(case, tmp_path)


def test_joint_validation_rejects_missing_prediction_arm(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    with pytest.raises(Stage1ValidationError):
        _run(case, tmp_path, prediction_b=tmp_path / "missing-arm")


def test_joint_validation_rejects_mismatched_observation_geometry(tmp_path: Path) -> None:
    case = _make_case(tmp_path, geometry_delta=True)
    with pytest.raises(Stage1ValidationError, match="geometry|length|sequence"):
        _run(case, tmp_path)


def test_joint_validation_rejects_shifted_checkpoint_in_reconstruction_command(tmp_path: Path) -> None:
    case = _make_case(tmp_path, wrong_model_path=True)
    with pytest.raises(Stage1ValidationError, match="public base model"):
        _run(case, tmp_path)
