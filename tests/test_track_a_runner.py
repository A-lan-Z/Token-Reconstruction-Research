from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from token_reconstruction.footing import PanelCell


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trr0003_track_a.py"
SPEC = importlib.util.spec_from_file_location("trr0003_track_a_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TRACK_A = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACK_A
SPEC.loader.exec_module(TRACK_A)


def test_registered_algorithm_includes_identity_and_doubled_branch_calls() -> None:
    algorithm = TRACK_A._algorithm_config(
        iterations=TRACK_A.DEFAULT_ITERATIONS,
        damping=0.5,
        top_k=16,
        vocab_chunk_size=8192,
    )
    assert algorithm["iterations"] == [0, 1, 2, 4, 8, 16, 32]
    assert algorithm["zero_fit_identity_baseline"]["iterations"] == 0
    assert algorithm["branch_forward_calls_per_step"] == 2
    assert algorithm["candidate_prefix_simulations"] == 0
    raw = json.loads(
        (Path(__file__).resolve().parents[1] / "experiments/TRR-0003/track_a/preregistration.json").read_text()
    )
    assert raw["algorithm"] == algorithm


def test_observation_metadata_requires_explicit_private_state_assertion() -> None:
    TRACK_A._validate_observation_metadata(
        {
            "schema": "test",
            "truth_included": "false",
            "source_text_included": "false",
        }
    )
    with pytest.raises(TRACK_A.TrackAError):
        TRACK_A._validate_observation_metadata(
            {"schema": "test", "truth_included": "true"}
        )
    with pytest.raises(TRACK_A.TrackAError):
        TRACK_A._validate_observation_metadata({"schema": "test"})


def test_resource_manifest_binds_actual_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"public-weight-fixture")
    rows = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": TRACK_A.sha256_file(path),
        }
        for path in sorted(snapshot.iterdir())
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": TRACK_A.RESOURCE_SCHEMA,
                "model": {"id": TRACK_A.MODEL_ID, "revision": TRACK_A.MODEL_REVISION},
                "snapshot_path": str(snapshot.resolve()),
                "files": rows,
            }
        ),
        encoding="utf-8",
    )
    resource = TRACK_A._load_resource_manifest(
        manifest,
        model_path=snapshot,
        model_id=TRACK_A.MODEL_ID,
        revision=TRACK_A.MODEL_REVISION,
    )
    assert resource.total_bytes == sum(row["bytes"] for row in rows)
    weights.write_bytes(b"changed")
    with pytest.raises(TRACK_A.TrackAError):
        TRACK_A._load_resource_manifest(
            manifest,
            model_path=snapshot,
            model_id=TRACK_A.MODEL_ID,
            revision=TRACK_A.MODEL_REVISION,
        )


def test_prediction_validity_rejects_partial_padded_candidate_rows() -> None:
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    predictions = torch.tensor([[TRACK_A.BOS_TOKEN_ID, 5, -1, -1]], dtype=torch.int32)
    candidates = torch.tensor(
        [[[TRACK_A.BOS_TOKEN_ID, 2], [5, 7], [-1, -1], [-1, -1]]],
        dtype=torch.int32,
    )
    scores = torch.tensor(
        [[[0.0, -1.0], [-0.5, -2.0], [float("-inf"), float("-inf")], [float("-inf"), float("-inf")]]], dtype=torch.float32
    )
    TRACK_A._prediction_validity(
        predictions,
        candidates,
        scores,
        attention_mask=mask,
        selected=(0,),
    )
    candidates[0, 2, 1] = 9
    with pytest.raises(TRACK_A.TrackAError):
        TRACK_A._prediction_validity(
            predictions,
            candidates,
            scores,
            attention_mask=mask,
            selected=(0,),
        )


def test_valid_record_requires_arange_position_ids() -> None:
    cell = SimpleNamespace(
        attention_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
        position_ids=torch.tensor([[0, 2, 2, 2]], dtype=torch.long),
    )
    with pytest.raises(TRACK_A.TrackAError):
        TRACK_A._valid_record(cell=cell, record_index=0)


def test_module_state_digest_changes_with_parameter() -> None:
    module = torch.nn.Linear(3, 2, bias=True)
    first, bytes_first, tensors_first = TRACK_A._hash_module_state(module)
    with torch.no_grad():
        module.weight[0, 0] += 1
    second, bytes_second, tensors_second = TRACK_A._hash_module_state(module)
    assert first != second
    assert bytes_first == bytes_second
    assert tensors_first == tensors_second
