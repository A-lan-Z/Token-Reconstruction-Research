"""Synthetic integration tests for the truth-blind TRR-0010 prediction runner."""
from __future__ import annotations

import json
import shutil
import time
import types
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts import trr0010_eval_gate as gate
from scripts import trr0010_eval_runner as runner
import trr0004_predict_confirmation as a2_source
from tests.test_trr0010_eval_gate import _fixture


class _CallableIds:
    def __init__(self, offset: int) -> None:
        self.offset = offset
        self.calls = 0

    def __call__(self, activation: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del activation, mask, positions
        self.calls += 1
        values = torch.empty(gate.STORED_SEQUENCE_TOKENS, dtype=torch.long)
        values[0] = gate.BOS_TOKEN_ID
        values[1:] = 1 + (self.offset + torch.arange(gate.SCORED_POST_BOS_TOKENS)) % (gate.VOCABULARY_SIZE - 1)
        return values


def _bind_runtime_embedding(fixture: dict[str, Any], *, hidden_size: int = 2) -> Path:
    root = fixture["root"]
    path = root / "assets" / "runtime_embedding.safetensors"
    save_file(
        {"embeddings": torch.zeros((gate.VOCABULARY_SIZE, hidden_size), dtype=torch.float32)},
        str(path),
    )
    registration_path = fixture["registration"]
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    registration["runtime_embedding"] = gate.file_record(path, root=root)
    registration["resource_guard"] = {"inference": dict(runner.REGISTERED_INFERENCE_CAPS)}
    registration_path.write_text(json.dumps(registration, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_execute_emits_complete_matrix_and_passes_public_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "OBSERVATION_HIDDEN_SIZE", 2)
    fixture = _fixture(tmp_path)
    _bind_runtime_embedding(fixture)
    # The shared gate fixture contains a completed dummy matrix; the runner is
    # create-only, so remove only those synthetic outputs before this run.
    for name in ("predictions", "timings"):
        shutil.rmtree(fixture["output_root"] / name)
    fixture["run"].unlink()

    adapters: dict[str, _CallableIds] = {}

    def factory(
        method_id: str,
        row: dict[str, Any],
        embedding: torch.Tensor,
        device: torch.device,
    ) -> _CallableIds:
        del row, embedding, device
        adapter = _CallableIds(len(adapters) * 17)
        adapters[method_id] = adapter
        return adapter

    result = runner.execute(
        registration_path=fixture["registration"],
        repository_root=fixture["root"],
        device_name="cpu",
        require_current_head=False,
        method_factory=factory,
    )

    assert result["status"] == gate.RUN_STATUS
    assert result["freeze"]["status"] == gate.FREEZE_STATUS
    assert set(adapters) == set(gate.METHOD_ORDER)
    assert all(adapter.calls == 2 * len(gate.CELL_ORDER) * gate.RECORDS_PER_CELL for adapter in adapters.values())

    run = json.loads(Path(result["run_manifest"]["path"]).read_text(encoding="utf-8"))
    expected = {f"{method}::{cell}" for method in gate.METHOD_ORDER for cell in gate.CELL_ORDER}
    assert set(run["predictions"]) == expected
    assert set(run["timings"]) == expected
    assert run["truth_opened"] is False
    assert run["source_text_loaded"] is False
    assert run["candidate_arrays_persisted"] is False
    common = run["common_preparation"]
    assert common["registration_hash_verification_seconds"] >= 0.0
    assert common["seconds"] >= common["registration_hash_verification_seconds"]
    assert common["included_in_elapsed_seconds"] is True
    assert all(item["payload"]["truth_opened"] is False for item in result["freeze"]["timings"].values())
    assert all(
        "host process_max_rss_bytes is ru_maxrss high-water" in item["payload"]["peak_memory_scope"]
        and "cuda_peak_* fields are reset immediately before this cell" in item["payload"]["cuda_peak_memory_scope"]
        and item["payload"]["model_preparation_accounting"]["shared_across_cells"] is True
        and item["payload"]["model_preparation_accounting"]["cells_covered"] == len(gate.CELL_ORDER)
        and item["payload"]["model_preparation_accounting"]["charge_count"] == 1
        and "adapter_evidence" in item["payload"]
        for item in result["freeze"]["timings"].values()
    )


def test_resource_guard_rejects_fixed_cuda_peak_without_cuda_context() -> None:
    caps = runner._method_resource_caps(
        runner.REGISTERED_INFERENCE_CAPS, gate.CURRENT_FIXED_METHOD_ID
    )
    snapshot = {
        "host_available_bytes": caps["host_available_floor_bytes_runtime"],
        "host_rss_bytes": caps["host_rss_limit_bytes"],
        "disk_free_bytes": caps["disk_free_floor_bytes"],
        "gpu": {
            "available": True,
            "free_bytes": caps["gpu_free_floor_bytes_runtime"],
            "max_reserved_bytes": caps["cuda_reserved_limit_bytes"] + 1,
        },
    }
    with pytest.raises(runner.RunnerError, match="reserved-GPU cap failed"):
        runner._enforce_resource_guard(
            snapshot,
            caps,
            started=time.perf_counter(),
            stage="synthetic_fixed_threshold",
            require_gpu=True,
        )


def test_resource_guard_ignores_prior_peak_after_current_reservation_check() -> None:
    caps = runner._method_resource_caps(
        runner.REGISTERED_INFERENCE_CAPS, gate.CURRENT_FIXED_METHOD_ID
    )
    snapshot = {
        "host_available_bytes": caps["host_available_floor_bytes_runtime"],
        "host_rss_bytes": caps["host_rss_limit_bytes"],
        "disk_free_bytes": caps["disk_free_floor_bytes"],
        "gpu": {
            "available": True,
            "free_bytes": caps["gpu_free_floor_bytes_runtime"],
            "reserved_bytes": caps["cuda_reserved_limit_bytes"],
            "max_reserved_bytes": caps["directional_cuda_reserved_limit_bytes"] + 1,
        },
    }
    runner._enforce_resource_guard(
        snapshot,
        caps,
        started=time.perf_counter(),
        stage="synthetic_current_reservation",
        require_gpu=True,
        gpu_peak_field="reserved_bytes",
    )


def test_guard_callbacks_are_outside_timed_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    activation = torch.zeros((gate.STORED_SEQUENCE_TOKENS, 2), dtype=torch.bfloat16)
    mask = torch.ones(gate.STORED_SEQUENCE_TOKENS, dtype=torch.bool)
    positions = torch.arange(gate.STORED_SEQUENCE_TOKENS, dtype=torch.long)

    def rows(cell: Mapping[str, Any], *, records: int, hidden_size: int):
        del cell, hidden_size
        assert records == 1
        yield 0, activation, mask, positions

    monkeypatch.setattr(runner, "_iter_rows", rows)
    calls: list[str] = []

    def guard(stage: str) -> None:
        calls.append(stage)
        time.sleep(0.01)

    started = time.perf_counter()
    _values, timing = runner._run_cell(
        adapter=_CallableIds(0),
        cell={},
        records=1,
        hidden_size=2,
        device=torch.device("cpu"),
        method_id="synthetic",
        guard_callback=guard,
    )
    elapsed = time.perf_counter() - started
    assert calls == [
        "before_cell",
        "after_cell_begin",
        "before_record_0",
        "after_record_0",
        "after_cell",
    ]
    assert elapsed - timing["measured_seconds_sum"] > 0.02



class _BaseDecoder(torch.nn.Module):
    def projected_hidden(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        del mask
        return hidden

    def logits_from_rows(
        self,
        projected: torch.Tensor,
        rows: torch.Tensor,
        positions: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        del rows, positions, embedding
        logits = torch.zeros(
            (gate.SCORED_POST_BOS_TOKENS, gate.VOCABULARY_SIZE),
            dtype=torch.float32,
            device=projected.device,
        )
        logits[:, 17] = 1.0
        return logits


class _DirectionalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = _BaseDecoder()
        self.delta = torch.nn.Parameter(torch.ones(1))
        self.materializations = 0

    def materialize_effective_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        self.materializations += 1
        return embedding.clone()


def test_directional_adapter_materializes_one_fixed_readout_and_drops_delta() -> None:
    embedding = torch.zeros((gate.VOCABULARY_SIZE, 2), dtype=torch.float32)
    model = _DirectionalModel()
    adapter = runner._MergedCurrentHAdapter(
        model,
        embedding,
        device=torch.device("cpu"),
        method_id=gate.CURRENT_DIRECTIONAL_METHOD_ID,
        allow_materialization=True,
    )

    activation = torch.zeros((gate.STORED_SEQUENCE_TOKENS, 2), dtype=torch.bfloat16)
    mask = torch.ones(gate.STORED_SEQUENCE_TOKENS, dtype=torch.bool)
    positions = torch.arange(gate.STORED_SEQUENCE_TOKENS, dtype=torch.long)
    first = adapter(activation, mask, positions)
    second = adapter(activation, mask, positions)

    assert model.materializations == 1
    assert adapter.materialized_once is True
    assert adapter.model is model.base
    assert not hasattr(adapter.model, "delta")
    assert adapter.embedding.data_ptr() != embedding.data_ptr()
    assert torch.equal(first, second)
    assert first[0].item() == gate.BOS_TOKEN_ID
    assert torch.all(first[1:] == 17)


def test_prediction_normalization_rejects_float_token_ids() -> None:
    mask = torch.ones(gate.STORED_SEQUENCE_TOKENS, dtype=torch.bool)
    values = torch.zeros(gate.STORED_SEQUENCE_TOKENS, dtype=torch.float32)
    with pytest.raises(runner.RunnerError, match="token IDs must be signed integers"):
        runner._normalize_prediction(values, mask, method_id="synthetic")


def test_a1_nested_snapshot_and_reference_bindings_are_rehashed(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = snapshot / "config.json"
    config.write_text("{}", encoding="utf-8")
    reference = tmp_path / "published_reference.py"
    reference.write_text("# public reference", encoding="utf-8")
    p0 = tmp_path / "p0.json"
    descriptor = {
        "model_snapshot": {
            "path": str(snapshot),
            "files": {"config.json": gate.file_record(config, root=tmp_path)},
        },
        "reference_binding": gate.file_record(reference, root=tmp_path),
    }
    p0.write_text(json.dumps(descriptor, sort_keys=True), encoding="utf-8")
    snapshot_path, reference_path, evidence = runner._verify_a1_nested_assets(
        gate.file_record(p0, root=tmp_path),
        path_args={},
        root=tmp_path,
    )
    assert snapshot_path == snapshot.resolve()
    assert reference_path == reference.resolve()
    assert evidence["snapshot_files"]["config.json"]["sha256"] == gate.sha256_file(config)
    config.write_text("{changed}", encoding="utf-8")
    with pytest.raises(gate.GateError, match="snapshot file config.json hash or size changed"):
        runner._verify_a1_nested_assets(
            gate.file_record(p0, root=tmp_path),
            path_args={},
            root=tmp_path,
        )


def test_swapped_loader_source_binding_fails_closed(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong_loader.py"
    wrong.write_text("# wrong source", encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="not one of the frozen code bindings"):
        runner._verify_source_path_bound(
            Path(a2_source.__file__).resolve(),
            code_bindings={"wrong": gate.file_record(wrong, root=tmp_path)},
            root=tmp_path,
            description="synthetic loader",
        )


def test_dynamic_loader_rejects_unbound_source_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state.bin"
    state.write_bytes(b"state")
    wrong = tmp_path / "wrong_loader.py"
    wrong.write_text("# wrong source", encoding="utf-8")
    calls: list[bool] = []

    def load_model(*args: Any, **kwargs: Any) -> torch.nn.Module:
        del args, kwargs
        calls.append(True)
        return _BaseDecoder()

    module = types.SimpleNamespace(
        __file__=str(tmp_path / "unbound_loader.py"),
        load_model=load_model,
    )
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: module)
    row = {
        "state": gate.file_record(state, root=tmp_path),
        "loader": {"module": "synthetic_unbound_loader", "function": "load_model", "kwargs": {}},
    }
    with pytest.raises(runner.RunnerError, match="not one of the frozen code bindings"):
        runner._dynamic_loader(
            row,
            method_id=gate.CURRENT_DIRECTIONAL_METHOD_ID,
            root=tmp_path,
            device=torch.device("cpu"),
            embedding=torch.zeros((gate.VOCABULARY_SIZE, 2)),
            code_bindings={"wrong": gate.file_record(wrong, root=tmp_path)},
        )
    assert calls == []


def test_published_a1_a2_candidate_geometry_is_not_top256_proposal() -> None:
    assert a2_source.DEFAULT_A2_PROPOSAL_K == 512
    assert a2_source.DEFAULT_A1_CHUNK == 256
    assert a2_source.DEFAULT_A2_K == 256
