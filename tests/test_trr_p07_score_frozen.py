from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from scripts.trr_p07 import score_frozen


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_observation(path: Path) -> None:
    mask = np.ones((256, 128), dtype=np.uint8)
    positions = np.broadcast_to(np.arange(128, dtype=np.int64), (256, 128)).copy()
    activations = np.zeros((256, 128, 1), dtype=np.float32)
    save_file({"activations": activations, "attention_mask": mask, "position_ids": positions}, str(path))


def test_validate_freeze_rejects_missing_fixture_gate_before_prediction_reads(tmp_path: Path) -> None:
    replay = {
        "schema": score_frozen.REPLAY_SCHEMA,
        "task_id": "TRR-P07",
        "status": "FROZEN_P07_PREDICTIONS_NO_TRUTH",
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
        "code_commit": "a" * 40,
        "prediction_count": 48,
        "source_bindings": {"canonical_plan": {"sha256": score_frozen.APPROVED_PLAN_SHA256}},
        "geometry": {"sequence_tokens": 128, "scored_post_bos_tokens": 127},
        "panels": {
            "p06_panel": {"records_per_domain": 256},
            "trr0006_subset": {
                "records_per_domain": 256,
                "subset": {"rule": "rows 6*k, k=0..255", "row_indices": list(range(0, 1536, 6))},
            },
        },
    }
    replay_path = tmp_path / "replay_manifest.json"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    with pytest.raises(score_frozen.P07ScoreError, match="fixture/provenance"):
        score_frozen.validate_prediction_freeze(repository_root=tmp_path, replay_manifest_path=replay_path)


def test_score_arrays_binds_method_families_and_returns_all_eight_cells(tmp_path: Path) -> None:
    record_ids = tuple(f"source-{index}" for index in range(256))
    source_digest = score_frozen._newline_digest(record_ids)
    truth = np.zeros((256, 128), dtype=np.int64)
    truth[:, 0] = 128000
    truth[:, 1:] = 1000 + np.arange(127)

    observation_paths: dict[tuple[str, str], Path] = {}
    for panel in score_frozen.PANELS:
        for domain in score_frozen.DOMAINS:
            path = tmp_path / f"{panel}_{domain}.safetensors"
            _write_observation(path)
            observation_paths[(panel, domain)] = path

    descriptors: dict[tuple[str, str, str], dict[str, object]] = {}
    for panel in score_frozen.PANELS:
        for domain in score_frozen.DOMAINS:
            for target in score_frozen.TARGETS:
                cell_id = f"{domain}__{target}"
                for method in score_frozen.METHODS:
                    for seed in (score_frozen.REPLICATE_SEEDS if method in score_frozen.P06_METHODS else (None,)):
                        seed_label = "retained" if seed is None else str(seed)
                        path = tmp_path / f"{panel}_{domain}_{target}_{method}_{seed_label}.safetensors"
                        save_file({"predictions": truth.astype(np.int64)}, str(path))
                        descriptor = {
                            "schema": "token-reconstruction.trr-p07-predictions.v1",
                            "task_id": "TRR-P07",
                            "records": 256,
                            "shape": [256, 128],
                            "scored_post_bos_tokens": 127,
                            "seed": seed,
                            "record_ids_sha256": source_digest,
                            "observation": _record(observation_paths[(panel, domain)]),
                            "timing": {"selected_row_indices": list(range(256))},
                            "prediction": _record(path),
                        }
                        descriptors[(panel, cell_id, method)] = descriptor

    validated = {
        "status": "JOINT_FREEZE_VALIDATED_NO_TRUTH",
        "truth_opened": False,
        "replay_manifest": {"path": "replay_manifest.json", "sha256": "b" * 64},
        "descriptors": descriptors,
    }
    truth_by_cell = {(panel, domain): (truth, record_ids) for panel in score_frozen.PANELS for domain in score_frozen.DOMAINS}
    result = score_frozen.score_arrays(
        validated,
        repository_root=tmp_path,
        truth_by_cell=truth_by_cell,
        bootstrap_draws=8,
        bootstrap_seed=7007,
    )
    assert result["status"] == "TRR-P07_SCORED_AFTER_PREDICTION_FREEZE"
    assert set(result["bootstrap"]["cells"]) == {
        f"{panel}/{domain}/{target}"
        for panel in score_frozen.PANELS
        for domain in score_frozen.DOMAINS
        for target in score_frozen.TARGETS
    }
    for cell in result["bootstrap"]["cells"].values():
        contrast = cell["contrasts"]["past_minus_reference"]
        assert set(contrast["per_seed"]) == {"6106", "6107"}
        assert contrast["replicate_averaged"]["records"] == 256
    assert result["gate"]["disposition"] == "PANEL_DEPENDENT_OR_UNCERTAIN"

