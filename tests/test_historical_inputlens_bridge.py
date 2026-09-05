from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from token_reconstruction.historical_inputlens_bridge import (
    HISTORICAL_HIDDEN_SIZE,
    HistoricalInputLensBridge,
    HistoricalInputLensError,
    equivalence_metrics,
    load_historical_lens_checkpoint,
    prepare_normalized_embeddings,
    validate_normalized_embeddings,
)


def _state(*, identity: bool = False) -> dict[str, torch.Tensor]:
    W = torch.eye(HISTORICAL_HIDDEN_SIZE, dtype=torch.float32)
    if not identity:
        W[0, 1] = 2.5
        W[1, 0] = -1.25
    b = torch.zeros(HISTORICAL_HIDDEN_SIZE, dtype=torch.float32)
    b[0] = 0.25
    b[3] = -0.5
    return {"W": W, "b": b, "s": torch.tensor(1.5, dtype=torch.float32)}


def _reference_logits(
    state: dict[str, torch.Tensor],
    activation: torch.Tensor,
    normalized_embeddings: torch.Tensor,
) -> torch.Tensor:
    projected = activation.float() @ state["W"].float().T + state["b"].float()
    projected = F.normalize(projected, dim=-1)
    logits = projected.to(normalized_embeddings.dtype) @ normalized_embeddings.T
    return logits.float() * state["s"].float().exp()


def test_bridge_matches_independent_historical_formula_and_has_no_parameters() -> None:
    torch.manual_seed(17)
    state = _state()
    bridge = HistoricalInputLensBridge.from_state_dict(state)
    activation = torch.randn(5, HISTORICAL_HIDDEN_SIZE, dtype=torch.bfloat16)
    embedding_table = F.normalize(torch.randn(11, HISTORICAL_HIDDEN_SIZE), dim=-1).to(torch.bfloat16)

    actual = bridge(activation, embedding_table)
    expected = _reference_logits(state, activation, embedding_table)

    assert torch.equal(actual, expected)
    assert list(bridge.parameters()) == []
    assert bridge.W.dtype == torch.float32
    assert bridge.s.dtype == torch.float32


def test_projection_orientation_bias_scale_and_normalization_are_explicit() -> None:
    state = _state()
    bridge = HistoricalInputLensBridge.from_state_dict(state)
    activation = torch.zeros(1, HISTORICAL_HIDDEN_SIZE, dtype=torch.bfloat16)
    activation[0, 0] = 1.0
    activation[0, 1] = 2.0
    expected_projection = activation.float() @ state["W"].T + state["b"]

    assert torch.equal(bridge.projected(activation), expected_projection)
    assert bridge.logit_scale_value == pytest.approx(float(state["s"].exp()))
    assert bridge.spec.projection == "activation.float32 @ W.float32.T + b.float32"
    assert bridge.spec.embedding_runtime_cast == "projected_normalized.to(normalized_embeddings.dtype)"


def test_prepare_normalized_embeddings_matches_reference_and_handles_zero_rows() -> None:
    raw = torch.tensor(
        [[3.0] + [0.0] * (HISTORICAL_HIDDEN_SIZE - 1), [0.0] * HISTORICAL_HIDDEN_SIZE],
        dtype=torch.bfloat16,
    )
    actual = prepare_normalized_embeddings(raw)
    expected = torch.nan_to_num(F.normalize(raw.detach().float(), dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
    assert torch.equal(actual, expected)
    validate_normalized_embeddings(actual, vocabulary_size=2, check_unit_norm=True)


def test_direct_prediction_is_complete_and_batched_without_fallback() -> None:
    bridge = HistoricalInputLensBridge.from_state_dict(_state(identity=True))
    table = F.normalize(torch.randn(7, HISTORICAL_HIDDEN_SIZE), dim=-1)
    activation = table[[4, 1, 6]].reshape(3, 1, HISTORICAL_HIDDEN_SIZE)
    prediction = bridge.predict(activation, table, batch_size=2)
    topk = bridge.topk(activation, table, k=3, batch_size=2)

    assert prediction.shape == (3, 1)
    assert prediction.dtype == torch.int32
    assert prediction.tolist() == [[4], [1], [6]]
    assert topk.shape == (3, 1, 3)
    assert topk[:, :, 0].tolist() == [[4], [1], [6]]


def test_equivalence_metrics_reports_logit_and_rank_agreement() -> None:
    logits = torch.tensor([[4.0, 1.0, -2.0], [0.5, 5.0, 1.0]])
    ranks = torch.argsort(logits, dim=-1, descending=True)
    result = equivalence_metrics(logits, logits.clone(), actual_topk=ranks, reference_topk=ranks.clone())
    assert result["exact_equal"] is True
    assert result["relative_l2"] == 0.0
    assert result["top1_mismatches"] == 0
    assert result["topk_position_mismatches"] == 0


def test_checkpoint_loader_is_strict_and_binds_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "lens.pt"
    torch.save({"sd": _state(), "hidden": 0, "corpus": "alpaca"}, checkpoint)
    bridge = load_historical_lens_checkpoint(checkpoint)

    assert bridge.checkpoint_path == str(checkpoint.resolve())
    assert len(bridge.lens_state_sha256) == 64
    assert bridge.lens_state_sha256 == bridge.lens_state_sha256

    bad = tmp_path / "bad.pt"
    torch.save({"sd": {**_state(), "extra": torch.zeros(())}, "hidden": 0, "corpus": "alpaca"}, bad)
    with pytest.raises(HistoricalInputLensError, match="exactly W, b, and s"):
        load_historical_lens_checkpoint(bad)


def test_runtime_validation_fails_closed() -> None:
    bridge = HistoricalInputLensBridge.from_state_dict(_state())
    with pytest.raises(HistoricalInputLensError, match="non-finite"):
        bridge(torch.full((1, HISTORICAL_HIDDEN_SIZE), float("nan")), torch.eye(3, HISTORICAL_HIDDEN_SIZE))
    with pytest.raises(HistoricalInputLensError, match="unit norm"):
        validate_normalized_embeddings(torch.ones(3, HISTORICAL_HIDDEN_SIZE), check_unit_norm=True)



def test_reference_loader_imports_dataclass_based_legacy_module(tmp_path: Path) -> None:
    import trr0004_historical_inputlens_bridge as diagnostic

    checkpoint = tmp_path / "lens.pt"
    torch.save({"sd": _state(), "hidden": 0, "corpus": "alpaca"}, checkpoint)
    reference_path = Path("reference/strict_bos/round001_teacher.py").resolve()

    module, lens = diagnostic._load_reference(reference_path, checkpoint, torch.device("cpu"))

    assert module.FrozenAffineLens is not None
    assert lens.W.shape == (HISTORICAL_HIDDEN_SIZE, HISTORICAL_HIDDEN_SIZE)



def test_bridge_diagnostic_resource_policy_and_cpu_preflight() -> None:
    import trr0004_historical_inputlens_bridge as diagnostic

    args = diagnostic._parser().parse_args([])
    policy = diagnostic._resource_policy(args)
    assert policy["minimum_free_gpu_bytes"] == 8 * 1024**3
    assert policy["maximum_gpu_reserved_bytes"] == 4 * 1024**3
    assert policy["maximum_host_rss_bytes"] == 16 * 1024**3
    assert policy["maximum_wall_seconds"] == 120.0
    assert diagnostic._resource_preflight(torch.device("cpu"), policy)["status"] == "host_only_no_gpu_check"


def test_bridge_diagnostic_source_snapshot_binds_all_executed_sources() -> None:
    import trr0004_historical_inputlens_bridge as diagnostic

    root = Path(__file__).resolve().parents[1]
    snapshot = diagnostic._execution_snapshot(root)
    paths = {Path(item["path"]).resolve() for item in snapshot["code"]}
    assert (root / "scripts/trr0004_historical_inputlens_bridge.py").resolve() in paths
    assert (root / "src/token_reconstruction/historical_inputlens_bridge.py").resolve() in paths
    assert (root / "reference/strict_bos/round001_teacher.py").resolve() in paths
    assert len(snapshot["git_commit"]) == 40
    assert "CUDA_VISIBLE_DEVICES" in snapshot["environment"]


def test_bridge_diagnostic_output_check_precedes_execution(tmp_path: Path) -> None:
    import trr0004_historical_inputlens_bridge as diagnostic

    output = tmp_path / "existing.json"
    output.write_text("{}\n")
    with pytest.raises(diagnostic.BridgeDiagnosticError, match="overwrite"):
        diagnostic._ensure_create_only(output)
