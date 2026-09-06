from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.trr_p06 import freeze_matrix, score_frozen


DOMAINS = score_frozen.DOMAINS
TARGETS = score_frozen.TARGETS
METHODS = score_frozen.METHOD_ORDER
SEEDS = score_frozen.REPLICATE_SEEDS
ROOT_COMMIT = "a" * 40


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    root = tmp_path
    assets = root / "assets"
    assets.mkdir()
    monkeypatch.setattr(freeze_matrix, "_git_head", lambda _root: ROOT_COMMIT)

    plan_path = root / "plan.json"
    _write_json(plan_path, {"task_id": "TRR-P06", "truth_opened": False, "status": "ROOT_APPROVED_FROZEN"})

    record_ids = {domain: [f"{domain}-record-{i:03d}" for i in range(256)] for domain in DOMAINS}
    sequence_hashes = {
        domain: [hashlib.sha256(f"sequence-{domain}-{i}".encode()).hexdigest() for i in range(256)]
        for domain in DOMAINS
    }
    record_ids_sha = {domain: _digest(record_ids[domain]) for domain in DOMAINS}
    selection_path = root / "selection.json"
    _write_json(
        selection_path,
        {
            "schema": "token-reconstruction.trr-p06-source-selection.v1",
            "task_id": "TRR-P06",
            "truth_opened": False,
            "paired_conditions": True,
            "records_per_domain": 256,
            "selection_rule": {
                "records": {
                    domain: [
                        {"record_id": record_ids[domain][i], "final_sequence_sha256": sequence_hashes[domain][i]}
                        for i in range(256)
                    ]
                    for domain in DOMAINS
                }
            },
        },
    )
    selection_record = _record(selection_path, root)

    # Observation files are opaque bytes to this test and to the production
    # assembler; only their file records and declared geometry are checked.
    observation_cells: list[dict[str, object]] = []
    observation_sha: dict[str, str] = {}
    for domain in DOMAINS:
        for target in TARGETS:
            cell_id = f"{domain}__{target}"
            path = assets / f"observation-{cell_id}.bin"
            path.write_bytes(f"observation {cell_id}".encode())
            rec = _record(path, root)
            observation_sha[cell_id] = str(rec["sha256"])
            observation_cells.append(
                {
                    "cell_id": cell_id,
                    "style": domain,
                    "condition": target,
                    "record_ids_sha256": record_ids_sha[domain],
                    "observation": {
                        **rec,
                        "shape": [256, 128, 2048],
                        "record_ids_sha256": record_ids_sha[domain],
                        "selection_plan_sha256": selection_record["sha256"],
                    },
                }
            )
    observation_path = root / "observations.json"
    _write_json(
        observation_path,
        {
            "schema": "token-reconstruction.trr-p06-public-observation-manifest.v1",
            "task_id": "TRR-P06",
            "status": "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH",
            "truth_opened": False,
            "source_text_written": False,
            "token_ids_written": False,
            "target_labels_loaded": False,
            "records_per_domain": 256,
            "sequence_tokens_including_bos": 128,
            "scored_post_bos_tokens": 127,
            "hidden_size": 2048,
            "cell_order": [f"{domain}__{target}" for domain in DOMAINS for target in TARGETS],
            "selection_plan": selection_record,
            "cells": observation_cells,
            "source_pairing": {"same_record_ids_across_targets": True, "record_ids_sha256": record_ids_sha},
        },
    )
    observation_record = _record(observation_path, root)

    capture_path = root / "capture.json"
    _write_json(
        capture_path,
        {
            "schema": "token-reconstruction.trr-p06-public-capture.v1",
            "task_id": "TRR-P06",
            "status": "PUBLIC_OBSERVATIONS_CAPTURE_COMPLETE_NO_TRUTH",
            "truth_opened": False,
            "source_pairing": {"same_record_ids_across_targets": True},
            "observations": observation_record,
            "selection_plan": selection_record,
        },
    )

    state_bindings: dict[str, dict[str, object]] = {}
    for seed in SEEDS:
        for method in METHODS:
            path = assets / f"state-{seed}-{method}.bin"
            path.write_bytes(f"state {seed} {method}".encode())
            state_bindings[f"{seed}::{method}"] = _record(path, root)

    student_cells: dict[str, object] = {}
    for domain in DOMAINS:
        for target in TARGETS:
            cell_id = f"{domain}__{target}"
            replicates: dict[str, object] = {}
            for seed in SEEDS:
                methods: dict[str, object] = {}
                for method in METHODS:
                    path = assets / f"prediction-{domain}-{target}-{seed}-{method}.bin"
                    path.write_bytes(f"prediction {domain} {target} {seed} {method}".encode())
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
                        "attention_mask_sha256": hashlib.sha256(f"mask-{domain}".encode()).hexdigest(),
                        "position_ids_sha256": hashlib.sha256(f"position-{domain}".encode()).hexdigest(),
                        "observation_sha256": observation_sha[cell_id],
                        "state_sha256": state_bindings[f"{seed}::{method}"]["sha256"],
                        "prediction": _record(path, root),
                        "truth_opened": False,
                        "candidate_arrays_persisted": False,
                    }
                replicates[str(seed)] = methods
            student_cells[cell_id] = {"domain": domain, "target": target, "replicates": replicates}

    student_path = root / "student.json"
    _write_json(
        student_path,
        {
            "schema": "token-reconstruction.trr-p06-student-prediction-manifest.v1",
            "task_id": "TRR-P06",
            "status": "STUDENT_PREDICTIONS_COMPLETE_NO_TRUTH",
            "truth_opened": False,
            "code_commit": ROOT_COMMIT,
            "domains": list(DOMAINS),
            "target_conditions": list(TARGETS),
            "method_order": list(METHODS),
            "replicate_seeds": list(SEEDS),
            "state_bindings": state_bindings,
            "student_cells": student_cells,
        },
    )

    anchor_state = assets / "anchor-state.bin"
    anchor_state.write_bytes(b"anchor state")
    anchor_state_record = _record(anchor_state, root)
    anchor_subset = {domain: _digest(record_ids[domain][:64]) for domain in DOMAINS}
    anchor_cells: dict[str, object] = {}
    anchor_mask = hashlib.sha256(b"anchor-mask").hexdigest()
    anchor_position = hashlib.sha256(b"anchor-position").hexdigest()
    for domain in DOMAINS:
        path = assets / f"anchor-{domain}.bin"
        path.write_bytes(f"anchor {domain}".encode())
        anchor_cells[domain] = {
            "task_id": "TRR-P06",
            "domain": domain,
            "target": "public_base",
            "method_id": score_frozen.ANCHOR_METHOD_ID,
            "subset": "first64_public_base",
            "records": 64,
            "shape": [64, 128],
            "scored_post_bos_tokens": 127,
            "record_ids_sha256": anchor_subset[domain],
            "anchor_subset_record_ids_sha256": anchor_subset[domain],
            "attention_mask_sha256": anchor_mask,
            "position_ids_sha256": anchor_position,
            "observation_sha256": observation_sha[f"{domain}__public_base"],
            "state_sha256": anchor_state_record["sha256"],
            "prediction": _record(path, root),
            "truth_opened": False,
        }
    anchor_path = root / "anchor.json"
    _write_json(
        anchor_path,
        {
            "schema": "token-reconstruction.trr-p06-anchor-manifest.v1",
            "task_id": "TRR-P06",
            "truth_opened": False,
            "code_commit": ROOT_COMMIT,
            "anchor_state_sha256": anchor_state_record["sha256"],
            "state_binding": anchor_state_record,
            "anchor_cells": anchor_cells,
        },
    )

    preflight_path = root / "preflight.json"
    _write_json(preflight_path, {"task_id": "TRR-P06", "schema": "preflight", "status": "SOURCE_ONLY_PREFLIGHT_PASS", "truth_opened": False})
    qualification_path = root / "qualification.json"
    _write_json(qualification_path, {"task_id": "TRR-P06", "schema": "qualification", "status": "PASS", "truth_opened": False})
    capacity_path = root / "capacity.json"
    _write_json(
        capacity_path,
        {
            "task_id": "TRR-P06",
            "schema": "capacity",
            "status": "PASS",
            "truth_opened": False,
            "methods": [
                {
                    "method_id": method,
                    "status": "PASS",
                    "direct_affine_frozen": True,
                    "initial_metrics": {"total": 256, "correct": 0},
                    "final_metrics": {"total": 256, "correct": 256},
                    "pass_threshold_correct": 52,
                }
                for method in METHODS
            ],
        },
    )
    main_fit_path = root / "main-fit.json"
    _write_json(
        main_fit_path,
        {
            "task_id": "TRR-P06",
            "schema": "main-fit",
            "status": "PASS",
            "truth_opened": False,
            "runtime_components": {"target_truth_access": False, "source_token_access": False, "a2_student": False},
            "methods": [
                {
                    "seed": seed,
                    "method_id": method,
                    "status": "PASS",
                    "state": {"sha256": state_bindings[f"{seed}::{method}"]["sha256"]},
                }
                for seed in SEEDS
                for method in METHODS
            ],
        },
    )

    return {
        "root": root,
        "student": student_path,
        "anchor": anchor_path,
        "plan": plan_path,
        "selection": selection_path,
        "observation": observation_path,
        "capture": capture_path,
        "preflight": preflight_path,
        "qualification": qualification_path,
        "capacity": capacity_path,
        "main_fit": main_fit_path,
    }


def _assemble(
    fixture: dict[str, Path],
    *,
    suffix: str = "",
    anchor_paths: list[Path] | None = None,
) -> dict[str, object]:
    root = fixture["root"]
    manifest = root / f"joint-manifest{suffix}.json"
    freeze = root / f"joint-freeze{suffix}.json"
    result = freeze_matrix.assemble_joint_freeze(
        repository_root=root,
        student_manifest_path=fixture["student"],
        anchor_manifest_paths=anchor_paths or [fixture["anchor"]],
        plan_path=fixture["plan"],
        source_selection_path=fixture["selection"],
        observation_manifest_path=fixture["observation"],
        capture_receipt_path=fixture["capture"],
        preflight_receipt_path=fixture["preflight"],
        qualification_receipt_path=fixture["qualification"],
        capacity_receipt_path=fixture["capacity"],
        main_fit_receipt_path=fixture["main_fit"],
        output_manifest_path=manifest,
        output_freeze_path=freeze,
    )
    result["manifest"] = manifest
    result["freeze"] = freeze
    return result


def test_assembler_creates_scorer_compatible_matrix_without_truth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    result = _assemble(fixture)
    assert result["status"] == "FROZEN_P06_MATRIX_NO_TRUTH"
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    freeze = json.loads(Path(result["freeze"]).read_text(encoding="utf-8"))
    assert manifest["truth_opened"] is False
    assert freeze["truth_opened"] is False
    verified = score_frozen.validate_joint_freeze(
        repository_root=fixture["root"],
        freeze_receipt_path=Path(result["freeze"]),
        prediction_manifest_path=Path(result["manifest"]),
    )
    assert verified["status"] == "JOINT_FREEZE_VALIDATED_NO_TRUTH"
    assert set(verified["student_cells"]) == {f"{domain}__{target}" for domain in DOMAINS for target in TARGETS}
    assert set(verified["anchor_cells"]) == set(DOMAINS)




def test_assembler_merges_separate_per_domain_anchor_descriptors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    payload = json.loads(fixture["anchor"].read_text(encoding="utf-8"))
    split_paths: list[Path] = []
    for domain in DOMAINS:
        path = fixture["root"] / f"anchor-{domain}.json"
        _write_json(
            path,
            {
                "schema": payload["schema"],
                "task_id": payload["task_id"],
                "truth_opened": False,
                "code_commit": payload["code_commit"],
                "anchor_state_sha256": payload["anchor_state_sha256"],
                "state_binding": payload["state_binding"],
                "anchor_cells": {domain: payload["anchor_cells"][domain]},
            },
        )
        split_paths.append(path)
    result = _assemble(fixture, suffix="-split", anchor_paths=split_paths)
    verified = score_frozen.validate_joint_freeze(
        repository_root=fixture["root"],
        freeze_receipt_path=Path(result["freeze"]),
        prediction_manifest_path=Path(result["manifest"]),
    )
    assert set(verified["anchor_cells"]) == set(DOMAINS)


def test_assembler_rejects_prediction_hash_change_before_writing_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    student = json.loads(fixture["student"].read_text(encoding="utf-8"))
    descriptor = student["student_cells"]["pile__public_base"]["replicates"]["6106"][METHODS[0]]
    prediction = fixture["root"] / descriptor["prediction"]["path"]
    prediction.write_bytes(prediction.read_bytes() + b"tamper")
    with pytest.raises(freeze_matrix.FreezeMatrixError, match="hash or byte binding changed"):
        _assemble(fixture)
    assert not (fixture["root"] / "joint-manifest.json").exists()
    assert not (fixture["root"] / "joint-freeze.json").exists()


def test_assembler_outputs_are_create_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _assemble(fixture)
    with pytest.raises(freeze_matrix.FreezeMatrixError, match="create-only output already exists"):
        _assemble(fixture)
