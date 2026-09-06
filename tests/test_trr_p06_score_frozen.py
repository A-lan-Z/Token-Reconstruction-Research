from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.trr_p06 import score_frozen
from token_reconstruction.trr_p06_metrics import METHOD_ORDER, TRAINING_REPLICATE_SEEDS


ROOT_COMMIT = "a" * 40
DOMAINS = ("pile", "finance")
TARGETS = ("public_base", "public_lora_2601")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path
    (root / "assets").mkdir()
    truth = np.zeros((256, 128), dtype=np.int64)
    truth[:, 0] = score_frozen.BOS_TOKEN_ID
    truth[:, 1:] = 1000 + np.arange(127)
    truth_path = root / "assets" / "truth.npy"
    np.save(truth_path, truth)

    plan_path = root / "plan.json"
    selection_path = root / "selection.json"
    observation_path = root / "observations.json"
    for path in (plan_path, selection_path, observation_path):
        _write_json(path, {"task_id": "TRR-P06", "truth_opened": False})
    bindings = {
        "plan": _record(plan_path, root),
        "source_selection": _record(selection_path, root),
        "observation_manifest": _record(observation_path, root),
    }

    state_bindings: dict[str, dict[str, object]] = {}
    state_sha: dict[tuple[int, str], str] = {}
    for seed in TRAINING_REPLICATE_SEEDS:
        for method in METHOD_ORDER:
            path = root / "assets" / f"state-{seed}-{method}.bin"
            path.write_bytes(f"state {seed} {method}".encode())
            row = _record(path, root)
            state_bindings[f"{seed}::{method}"] = row
            state_sha[(seed, method)] = str(row["sha256"])
    anchor_state_path = root / "assets" / "anchor-state.bin"
    anchor_state_path.write_bytes(b"anchor state")
    anchor_state_record = _record(anchor_state_path, root)
    anchor_state_sha = str(anchor_state_record["sha256"])

    record_ids_sha = {domain: _digest([f"{domain}-{i}" for i in range(256)]) for domain in DOMAINS}
    mask_sha = {domain: hashlib.sha256(f"mask-{domain}".encode()).hexdigest() for domain in DOMAINS}
    position_sha = {domain: hashlib.sha256(f"position-{domain}".encode()).hexdigest() for domain in DOMAINS}
    subset_sha = {domain: hashlib.sha256(f"anchor-{domain}".encode()).hexdigest() for domain in DOMAINS}

    student_cells: dict[str, object] = {}
    for domain in DOMAINS:
        for target in TARGETS:
            cell_id = f"{domain}__{target}"
            observation_sha = hashlib.sha256(f"observation-{cell_id}".encode()).hexdigest()
            replicates: dict[str, object] = {}
            for seed in TRAINING_REPLICATE_SEEDS:
                methods: dict[str, object] = {}
                for method in METHOD_ORDER:
                    prediction_path = root / "assets" / f"prediction-{domain}-{target}-{seed}-{method}.npy"
                    np.save(prediction_path, truth)
                    methods[method] = {
                        "task_id": "TRR-P06",
                        "domain": domain,
                        "target": target,
                        "method_id": method,
                        "seed": seed,
                        "records": 256,
                        "shape": [256, 128],
                        "sequence_tokens": 128,
                        "scored_post_bos_tokens": 127,
                        "record_ids_sha256": record_ids_sha[domain],
                        "attention_mask_sha256": mask_sha[domain],
                        "position_ids_sha256": position_sha[domain],
                        "observation_sha256": observation_sha,
                        "state_sha256": state_sha[(seed, method)],
                        "prediction": _record(prediction_path, root),
                        "truth_opened": False,
                    }
                replicates[str(seed)] = methods
            student_cells[cell_id] = {"domain": domain, "target": target, "replicates": replicates}

    anchor_cells: dict[str, object] = {}
    for domain in DOMAINS:
        prediction_path = root / "assets" / f"anchor-{domain}.npy"
        np.save(prediction_path, truth[:64])
        anchor_cells[domain] = {
            "task_id": "TRR-P06",
            "domain": domain,
            "target": "public_base",
            "method_id": score_frozen.ANCHOR_METHOD_ID,
            "subset": "first64_public_base",
            "records": 64,
            "shape": [64, 128],
            "scored_post_bos_tokens": 127,
            "record_ids_sha256": subset_sha[domain],
            "anchor_subset_record_ids_sha256": subset_sha[domain],
            "attention_mask_sha256": mask_sha[domain],
            "position_ids_sha256": position_sha[domain],
            "state_sha256": anchor_state_sha,
            "prediction": _record(prediction_path, root),
            "truth_opened": False,
        }

    manifest_path = root / "prediction_manifest.json"
    manifest = {
        "schema": score_frozen.PREDICTION_SCHEMA,
        "task_id": "TRR-P06",
        "status": "FROZEN_P06_PREDICTIONS_NO_TRUTH",
        "truth_opened": False,
        "code_commit": ROOT_COMMIT,
        **bindings,
        "domains": list(DOMAINS),
        "target_conditions": list(TARGETS),
        "method_order": list(METHOD_ORDER),
        "replicate_seeds": list(TRAINING_REPLICATE_SEEDS),
        "state_bindings": state_bindings,
        "student_cells": student_cells,
        "anchor_state_sha256": anchor_state_sha,
        "anchor_subset_record_ids_sha256": subset_sha,
        "anchor_cells": anchor_cells,
    }
    _write_json(manifest_path, manifest)

    freeze_path = root / "freeze_receipt.json"
    freeze = {
        "schema": score_frozen.FREEZE_SCHEMA,
        "task_id": "TRR-P06",
        "status": "FROZEN_P06_MATRIX_NO_TRUTH",
        "truth_opened": False,
        "code_commit": ROOT_COMMIT,
        **bindings,
        "prediction_manifest_sha256": _sha(manifest_path),
        "scientific_preconditions": {
            "plan_frozen": True,
            "resource_qualified": True,
            "capacity_qualified": True,
            "all_fits_finite": True,
            "source_pairing_validated": True,
        },
        "anchor_state_sha256": anchor_state_sha,
        "anchor_subset_record_ids_sha256": subset_sha,
    }
    _write_json(freeze_path, freeze)

    truth_manifest_path = root / "truth_manifest.json"
    truth_manifest = {
        "schema": score_frozen.TRUTH_SCHEMA,
        "task_id": "TRR-P06",
        "status": "TRUTH_READY_AFTER_JOINT_FREEZE",
        "domains": {domain: _record(truth_path, root) for domain in DOMAINS},
    }
    _write_json(truth_manifest_path, truth_manifest)
    return {
        "root": root,
        "freeze": freeze_path,
        "manifest": manifest_path,
        "truth_manifest": truth_manifest_path,
        "truth": truth_path,
    }


def test_joint_freeze_validates_full_student_matrix_and_separate_anchor(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    verified = score_frozen.validate_joint_freeze(
        repository_root=fixture["root"],
        freeze_receipt_path=fixture["freeze"],
        prediction_manifest_path=fixture["manifest"],
    )
    assert verified["status"] == "JOINT_FREEZE_VALIDATED_NO_TRUTH"
    assert set(verified["student_cells"]) == {
        f"{domain}__{target}" for domain in DOMAINS for target in TARGETS
    }
    assert set(verified["anchor_cells"]) == set(DOMAINS)
    assert verified["truth_opened"] is False
    assert verified["replicate_seeds"] == list(TRAINING_REPLICATE_SEEDS)


def test_runner_scores_synthetic_truth_after_gate_and_keeps_anchor_separate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = fixture["root"] / "score.json"
    result = score_frozen.run(
        repository_root=fixture["root"],
        freeze_receipt_path=fixture["freeze"],
        prediction_manifest_path=fixture["manifest"],
        truth_manifest_path=fixture["truth_manifest"],
        output_path=output,
    )
    assert result["status"] == "TRR-P06_SCORED_AFTER_JOINT_FREEZE"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["gate"]["decision"] == "QUALIFIED_NEGATIVE_RETAIN_PAST_ONLY"
    assert payload["anchor"]["separate_denominator"] is True
    assert set(payload["anchor"]["domains"]) == set(DOMAINS)
    assert "anchor" not in payload["student"]["bootstrap"]["domains"]
    assert payload["truth_opened"] is True
    assert payload["truth_payload_persisted"] is False


def test_joint_freeze_rejects_missing_method_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    del manifest["student_cells"]["pile__public_base"]["replicates"]["6107"][METHOD_ORDER[-1]]
    bad_manifest = fixture["root"] / "bad_manifest.json"
    _write_json(bad_manifest, manifest)
    freeze = json.loads(fixture["freeze"].read_text(encoding="utf-8"))
    freeze["prediction_manifest_sha256"] = _sha(bad_manifest)
    bad_freeze = fixture["root"] / "bad_freeze.json"
    _write_json(bad_freeze, freeze)
    with pytest.raises(score_frozen.P06ScoreError, match="method matrix"):
        score_frozen.validate_joint_freeze(
            repository_root=fixture["root"],
            freeze_receipt_path=bad_freeze,
            prediction_manifest_path=bad_manifest,
        )


def test_joint_freeze_rehashes_prediction_bytes_before_truth(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    descriptor = manifest["student_cells"]["finance__public_lora_2601"]["replicates"]["6106"][METHOD_ORDER[0]]
    prediction_path = fixture["root"] / descriptor["prediction"]["path"]
    with prediction_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(score_frozen.P06ScoreError, match="hash or byte binding changed"):
        score_frozen.validate_joint_freeze(
            repository_root=fixture["root"],
            freeze_receipt_path=fixture["freeze"],
            prediction_manifest_path=fixture["manifest"],
        )
