from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from scripts import trr0006_prediction_contract as contract
from scripts import trr0006_run_predictions as runner


def _registration(records: int = 1024) -> dict:
    methods = OrderedDict()
    for method_id in contract.METHOD_IDS:
        methods[method_id] = {
            "base_method_id": contract.BASE_METHOD_IDS[method_id],
            "decision_rule": contract.METHOD_RULES[method_id],
            "state": {
                "path": contract.PUBLISHED_STATE_BINDINGS[method_id]["path"],
                "bytes": contract.PUBLISHED_STATE_BINDINGS[method_id]["bytes"],
                "sha256": contract.PUBLISHED_STATE_BINDINGS[method_id]["sha256"],
                "source_commit": contract.SCIENTIFIC_SOURCE_COMMIT,
            },
        }
    return {
        "schema": contract.REGISTRATION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_PREDICTION_REGISTRATION",
        "code_commit": "c" * 40,
        "records_per_domain": records,
        "cell_order": list(contract.CELL_ORDER),
        "method_ids": list(contract.METHOD_IDS),
        "geometry": {
            "capture_batch_records": 8,
            "capture_sequence_tokens": 192,
            "stored_sequence_tokens": 128,
            "scored_sequence_tokens": 128,
            "scored_post_bos_tokens": 127,
            "hidden_size": 2048,
            "chunk_records": 8,
        },
        "runtime_assets": {
            "normalized_public_E": {
                "path": "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors",
                "bytes": contract.NORMALIZED_PUBLIC_E_BYTES,
                "sha256": contract.NORMALIZED_PUBLIC_E_SHA256,
                "shape": [contract.VOCAB_SIZE, contract.HIDDEN_SIZE],
                "dtype": "torch.float32",
            }
        },
        "methods": methods,
        "observation_manifest": {
            "path": "observations.json",
            "bytes": 1,
            "sha256": "e" * 64,
        },
        "output_root": "experiments/TRR-0006/predictions",
        "timing_contract": {
            "warmup_runs_per_record": 1,
            "measured_runs_per_record": 1,
            "repeat_integrity": "Require warmup and measured predicted IDs to match exactly",
        },
        "resource_guard": {
            "minimum_free_gpu_bytes": contract.MIN_FREE_GPU_BYTES,
            "maximum_reserved_gpu_bytes": contract.MAX_RESERVED_GPU_BYTES,
            "maximum_rss_bytes": contract.MAX_RSS_BYTES,
            "minimum_host_available_bytes": contract.MIN_HOST_AVAILABLE_BYTES,
            "maximum_seconds": 1800,
        },
        "numerical_settings": dict(contract.NUMERICAL_SETTINGS),
        "code_bindings": [
            {"role": role, "path": path, "bytes": 1, "sha256": "f" * 64}
            for role, path in contract.CODE_BINDING_SPECS
        ],
        "truth_opened": False,
        "candidate_arrays_persisted": False,
    }


def test_registration_accepts_one_declared_count_without_defaulting_to_1024():
    first = contract.validate_registration(_registration(1024))
    second = contract.validate_registration(_registration(1536))
    assert first["records_per_domain"] == 1024
    assert second["records_per_domain"] == 1536


def test_registration_rejects_count_not_aligned_to_capture_chunk():
    with pytest.raises(contract.ContractError, match="divisible"):
        contract.validate_registration(_registration(1025))


def test_registration_rejects_missing_required_executed_source_binding():
    value = _registration()
    value["code_bindings"] = value["code_bindings"][:-1]
    with pytest.raises(contract.ContractError, match="code bindings"):
        contract.validate_registration(value)
    value = _registration()
    value["code_bindings"][2]["path"] = "src/token_reconstruction/not_decoder.py"
    with pytest.raises(contract.ContractError, match="required code binding"):
        contract.validate_registration(value)


def test_registration_rejects_alternate_state_hash_or_path():
    value = _registration()
    value["methods"][contract.METHOD_IDS[0]]["state"]["sha256"] = "a" * 64
    with pytest.raises(contract.ContractError, match="published state binding"):
        contract.validate_registration(value)


def test_registration_rejects_reordered_method_or_cell_completeness():
    value = _registration()
    value["method_ids"] = list(reversed(contract.METHOD_IDS))
    with pytest.raises(contract.ContractError, match="method order"):
        contract.validate_registration(value)
    value = _registration()
    value["cell_order"] = list(contract.CELL_ORDER[:-1])
    with pytest.raises(contract.ContractError, match="cell order"):
        contract.validate_registration(value)


def test_normalize_prediction_sets_bos_and_padding_without_truth():
    raw = torch.tensor([91, 4, 5, 6], dtype=torch.long)
    mask = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    actual = contract.normalize_prediction(raw, mask)
    assert actual.tolist() == [contract.BOS_TOKEN_ID, 4, contract.INVALID_TOKEN_ID, contract.INVALID_TOKEN_ID]


def test_validate_prediction_rejects_non_suffix_padding():
    prediction = torch.tensor(
        [[contract.BOS_TOKEN_ID, 4, contract.INVALID_TOKEN_ID, 7]], dtype=torch.long
    )
    with pytest.raises(contract.ContractError, match="suffix"):
        contract.validate_prediction_tensor(prediction, records=1, sequence_tokens=4)


def test_observation_chunks_are_fixed_and_do_not_accept_token_ids(tmp_path: Path):
    path = tmp_path / "observation.safetensors"
    activations = torch.zeros((16, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE), dtype=torch.bfloat16)
    mask = torch.ones((16, contract.STORED_SEQUENCE_TOKENS), dtype=torch.uint8)
    positions = torch.arange(contract.STORED_SEQUENCE_TOKENS, dtype=torch.long).repeat(16, 1)
    save_file(
        {"activations": activations, "attention_mask": mask, "position_ids": positions},
        str(path),
    )
    cell = {
        "cell_id": "pile__public_base",
        "observation": {"path": str(path)},
    }
    chunks = list(runner._iter_observation_chunks(cell, records=16, chunk_records=8))
    assert [(chunk.start, chunk.stop) for chunk in chunks] == [(0, 8), (8, 16)]
    assert all(tuple(chunk.activations.shape) == (8, 128, 2048) for chunk in chunks)
    assert all(tuple(chunk.mask.shape) == (8, 128) for chunk in chunks)

    token_path = tmp_path / "token_ids.safetensors"
    save_file(
        {
            "activations": activations,
            "attention_mask": mask,
            "position_ids": positions,
            "token_ids": torch.zeros((16, 128), dtype=torch.int32),
        },
        str(token_path),
    )
    token_cell = {"cell_id": "pile__public_base", "observation": {"path": str(token_path)}}
    with pytest.raises(runner.RunnerError, match="tensor keys changed"):
        list(runner._iter_observation_chunks(token_cell, records=16, chunk_records=8))


def test_observation_chunks_reject_partial_clips_nonbinary_masks_and_positions(tmp_path: Path):
    path = tmp_path / "bad_observation.safetensors"
    activations = torch.zeros((8, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE), dtype=torch.bfloat16)
    mask = torch.ones((8, contract.STORED_SEQUENCE_TOKENS), dtype=torch.uint8)
    positions = torch.arange(contract.STORED_SEQUENCE_TOKENS, dtype=torch.long).repeat(8, 1)
    mask[0, -1] = 0
    save_file({"activations": activations, "attention_mask": mask, "position_ids": positions}, str(path))
    cell = {"cell_id": "pile__public_base", "observation": {"path": str(path)}}
    with pytest.raises(runner.RunnerError, match="fully valid"):
        list(runner._iter_observation_chunks(cell, records=8, chunk_records=8))
    mask.fill_(1)
    mask[0, 0] = 2
    save_file({"activations": activations, "attention_mask": mask, "position_ids": positions}, str(path))
    with pytest.raises(runner.RunnerError, match="not binary"):
        list(runner._iter_observation_chunks(cell, records=8, chunk_records=8))
    mask.fill_(1)
    positions[0, 4] = 99
    save_file({"activations": activations, "attention_mask": mask, "position_ids": positions}, str(path))
    with pytest.raises(runner.RunnerError, match="not 0..127"):
        list(runner._iter_observation_chunks(cell, records=8, chunk_records=8))


def test_resource_guard_fails_closed_when_host_memory_is_unavailable(monkeypatch):
    class _UnavailableMeminfo:
        def read_text(self, **_kwargs):
            raise OSError("meminfo unavailable")

    monkeypatch.setattr(runner, "Path", lambda _value: _UnavailableMeminfo())
    guard = {
        "maximum_seconds": 900,
        "maximum_rss_bytes": contract.MAX_RSS_BYTES,
        "minimum_host_available_bytes": contract.MIN_HOST_AVAILABLE_BYTES,
        "minimum_free_gpu_bytes": contract.MIN_FREE_GPU_BYTES,
        "maximum_reserved_gpu_bytes": contract.MAX_RESERVED_GPU_BYTES,
    }
    with pytest.raises(runner.RunnerError, match="host available-memory guard unavailable"):
        runner._guard(device=torch.device("cpu"), guard=guard, started=runner.time.perf_counter(), stage="test")


def test_prediction_path_is_stable_and_cell_qualified(tmp_path: Path):
    path = runner._prediction_path(tmp_path, "finance__public_lora_2601", contract.METHOD_IDS[0])
    assert path == tmp_path / "finance" / "public_lora_2601" / f"{contract.METHOD_IDS[0]}.safetensors"
