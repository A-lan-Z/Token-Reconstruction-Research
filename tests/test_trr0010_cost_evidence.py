"""Synthetic public timing tests for the TRR-0010 cost adapter."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import trr0010_cost_evidence as cost
from scripts import trr0010_eval_gate as gate
from scripts import trr0010_score as scorer
from tests.test_trr0010_eval_gate import _fixture
from tests.test_trr0010_score import _decision_inputs


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "experiments" / "TRR-0010" / "planning" / "shared_contract.proposal.json"


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _bind_contract(fixture: dict[str, Any]) -> Path:
    path = fixture["root"] / "assets" / "shared_contract.proposal.json"
    path.write_bytes(CONTRACT.read_bytes())
    return path


def _rewrite_timing_matrix(
    fixture: dict[str, Any],
    *,
    runtimes: dict[str, float] | None = None,
    peaks: dict[str, int] | None = None,
    include_gpu: bool = True,
    delete_runtime_for: str | None = None,
) -> None:
    runtimes = runtimes or {}
    peaks = peaks or {}
    run = json.loads(fixture["run"].read_text(encoding="utf-8"))
    for method_id in gate.METHOD_ORDER:
        for cell_id in gate.CELL_ORDER:
            key = f"{method_id}::{cell_id}"
            path = Path(run["timings"][key]["path"])
            payload = json.loads(path.read_text(encoding="utf-8"))
            if key == delete_runtime_for:
                payload.pop("measured_seconds_sum", None)
            else:
                payload["measured_seconds_sum"] = float(runtimes.get(method_id, 1.0))
            payload["model_preparation_seconds"] = 99.0
            if include_gpu:
                payload["peak_memory"]["cuda_peak_reserved_bytes"] = int(peaks.get(method_id, 100))
                payload["peak_memory"]["cuda_peak_allocated_bytes"] = int(peaks.get(method_id, 100) - 10)
            _write(path, payload)
            run["timings"][key] = gate.file_record(path, root=fixture["root"])
    _write(fixture["run"], run)


def test_cost_gate_uses_warm_runtime_and_isolated_gpu_peak(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    contract_path = _bind_contract(fixture)
    _rewrite_timing_matrix(
        fixture,
        runtimes={
            gate.CURRENT_FIXED_METHOD_ID: 1.0,
            gate.CURRENT_DIRECTIONAL_METHOD_ID: 1.1,
            gate.EXPANDED_FIXED_METHOD_ID: 1.0,
            gate.EXPANDED_DIRECTIONAL_METHOD_ID: 1.2,
            gate.A1_A2_METHOD_ID: 10.0,
        },
        peaks={
            gate.CURRENT_FIXED_METHOD_ID: 100,
            gate.CURRENT_DIRECTIONAL_METHOD_ID: 110,
            gate.EXPANDED_FIXED_METHOD_ID: 100,
            gate.EXPANDED_DIRECTIONAL_METHOD_ID: 120,
            gate.A1_A2_METHOD_ID: 90,
        },
    )

    result = cost.build_cost_evidence(
        run_manifest_path=fixture["run"],
        repository_root=fixture["root"],
        contract_path=contract_path,
    )

    assert result["cost_gate"]["status"] == cost.PASS
    assert result["comparisons"]["candidate_vs_fixed"]["status"] == cost.PASS
    assert result["comparisons"]["candidate_vs_a1_a2"]["status"] == cost.PASS
    primary = result["comparisons"]["candidate_vs_fixed"]["by_cell"][gate.CELL_ORDER[0]]
    assert primary["runtime"]["ratio"] == pytest.approx(1.2)
    assert primary["memory"]["ratio"] == pytest.approx(1.2)
    assert primary["runtime"]["status"] == cost.PASS
    assert primary["memory"]["status"] == cost.PASS
    measurement = result["measurements"][gate.EXPANDED_DIRECTIONAL_METHOD_ID][gate.CELL_ORDER[0]]
    assert measurement["warmed_runtime_seconds"] == pytest.approx(1.2)
    assert measurement["model_preparation_seconds"] == pytest.approx(99.0)
    assert measurement["preparation_shared_across_cells"] is True


def test_missing_cuda_peak_remains_unknown_and_host_rss_is_not_memory_gate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    contract_path = _bind_contract(fixture)
    _rewrite_timing_matrix(
        fixture,
        runtimes={
            gate.CURRENT_FIXED_METHOD_ID: 1.0,
            gate.CURRENT_DIRECTIONAL_METHOD_ID: 1.0,
            gate.EXPANDED_FIXED_METHOD_ID: 1.0,
            gate.EXPANDED_DIRECTIONAL_METHOD_ID: 1.0,
            gate.A1_A2_METHOD_ID: 10.0,
        },
        include_gpu=False,
    )

    result = cost.build_cost_evidence(
        run_manifest_path=fixture["run"],
        repository_root=fixture["root"],
        contract_path=contract_path,
    )

    assert result["comparisons"]["candidate_vs_fixed"]["status"] == cost.UNKNOWN
    assert result["comparisons"]["candidate_vs_a1_a2"]["status"] == cost.PASS
    assert result["cost_gate"]["status"] == cost.UNKNOWN
    row = result["cost_gate"]["by_cell"][gate.CELL_ORDER[0]]["candidate_vs_fixed"]
    assert row["memory"]["status"] == cost.UNKNOWN
    assert row["memory"]["reason"] == "required_measurement_unavailable"
    assert result["criteria"]["host_rss_is_gate"] is False


def test_runtime_or_memory_failure_is_reported_without_pooling(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    contract_path = _bind_contract(fixture)
    _rewrite_timing_matrix(
        fixture,
        runtimes={
            gate.CURRENT_FIXED_METHOD_ID: 1.0,
            gate.CURRENT_DIRECTIONAL_METHOD_ID: 1.0,
            gate.EXPANDED_FIXED_METHOD_ID: 1.0,
            gate.EXPANDED_DIRECTIONAL_METHOD_ID: 2.0,
            gate.A1_A2_METHOD_ID: 10.0,
        },
        peaks={
            gate.EXPANDED_FIXED_METHOD_ID: 100,
            gate.EXPANDED_DIRECTIONAL_METHOD_ID: 200,
            gate.A1_A2_METHOD_ID: 90,
        },
    )
    result = cost.build_cost_evidence(
        run_manifest_path=fixture["run"],
        repository_root=fixture["root"],
        contract_path=contract_path,
    )
    assert result["cost_gate"]["status"] == cost.FAIL
    primary = result["comparisons"]["candidate_vs_fixed"]["by_cell"][gate.CELL_ORDER[0]]
    assert primary["runtime"]["status"] == cost.FAIL
    assert primary["memory"]["status"] == cost.FAIL
    assert primary["runtime"]["ratio"] == pytest.approx(2.0)
    # Every cell remains explicit; no domains or target conditions are pooled.
    assert set(result["cost_gate"]["by_cell"]) == set(gate.CELL_ORDER)


def test_unmeasured_runtime_is_unknown_when_public_gate_is_not_available(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    contract_path = _bind_contract(fixture)
    _rewrite_timing_matrix(
        fixture,
        runtimes={gate.EXPANDED_DIRECTIONAL_METHOD_ID: 1.0, gate.A1_A2_METHOD_ID: 10.0},
        include_gpu=True,
        delete_runtime_for=f"{gate.EXPANDED_DIRECTIONAL_METHOD_ID}::{gate.CELL_ORDER[0]}",
    )

    result = cost.build_cost_evidence(
        run_manifest_path=fixture["run"],
        repository_root=fixture["root"],
        contract_path=contract_path,
        validate_public_gate=False,
    )
    primary = result["comparisons"]["candidate_vs_fixed"]["by_cell"][gate.CELL_ORDER[0]]
    assert primary["status"] == cost.UNKNOWN
    assert primary["runtime"]["status"] == cost.UNKNOWN
    assert primary["runtime"]["reason"] == "required_measurement_unavailable"
    assert result["cost_gate"]["status"] == cost.UNKNOWN


def test_v4_threshold_change_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    contract_path = _bind_contract(fixture)
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["safeguards_and_cost"]["warmed_inference_runtime_ratio_max"] = 9.0
    _write(contract_path, payload)
    with pytest.raises(cost.CostEvidenceError, match="v4 cost threshold changed"):
        cost.build_cost_evidence(
            run_manifest_path=fixture["run"],
            repository_root=fixture["root"],
            contract_path=contract_path,
        )


def test_report_table_contains_all_methods_cells_without_quality_invention(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = cost.build_cost_evidence(
        run_manifest_path=fixture["run"],
        repository_root=fixture["root"],
        contract_path=_bind_contract(fixture),
    )
    table = result["report_comparison_table"]
    assert table["schema"] == cost.REPORT_TABLE_SCHEMA
    assert len(table["rows"]) == len(gate.METHOD_ORDER) * len(gate.CELL_ORDER)
    assert {row["method_id"] for row in table["rows"]} == set(gate.METHOD_ORDER)
    assert {row["cell_id"] for row in table["rows"]} == set(gate.CELL_ORDER)
    assert all("quality_metrics" in row for row in table["rows"])
    assert all(row["quality_metrics"].startswith("supplied by the score artifact") for row in table["rows"])

def test_published_runner_receipt_uses_same_runtime_and_peak_fields() -> None:
    # This is the existing public TRR-0009 runner output, read only to pin the
    # producer field names used by the TRR-0010 runner adapter.  Its method and
    # record counts differ, so it is an interface fixture rather than a TRR-0010
    # cost result.
    path = ROOT / "experiments" / "TRR-0009" / "evaluation" / "predictions_v2" / "finance" / "public_base" / "continued_adaptable_readout.run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    # TRR-0009 predates the explicit target-label false flag; the producer
    # timing fields themselves are unchanged.  Normalize only that boundary
    # metadata in memory for the interface assertion.
    payload = dict(payload)
    for flag in ("source_text_loaded", "source_text_written", "target_labels_loaded", "token_ids_written", "candidate_arrays_persisted"):
        payload.setdefault(flag, False)
    row = cost._measurement(  # noqa: SLF001 - producer-interface regression
        method_id=str(payload["method_id"]),
        cell_id=str(payload["cell_id"]),
        timing_binding={"payload": payload},
    )
    assert row["warmed_runtime_seconds"] == pytest.approx(payload["measured_seconds_sum"])
    assert row["model_preparation_seconds"] == pytest.approx(payload["model_preparation_seconds"])
    assert row["isolated_gpu_peak_reserved_bytes"] == payload["peak_memory"]["cuda_peak_reserved_bytes"]
    assert row["host_process_max_rss_bytes"] == payload["peak_memory"]["process_max_rss_bytes"]

def test_scorer_keeps_directional_and_data_cost_routes_separate() -> None:
    contrasts, remaining, gap = _decision_inputs()

    def comparison(status: str) -> dict[str, Any]:
        return {
            "status": status,
            "by_cell": {cell: {"status": status} for cell in gate.CELL_ORDER},
        }

    cost_payload = {
        "comparisons": {
            "candidate_vs_fixed": comparison("PASS"),
            "candidate_vs_current_fixed_robustness": comparison("PASS"),
            "candidate_vs_a1_a2": comparison("PASS"),
            "data_expanded_fixed_vs_current_fixed": comparison("FAIL"),
            "data_expanded_fixed_vs_a1_a2": comparison("PASS"),
        }
    }
    result = scorer._decision_readout(
        contrasts, remaining, gap, cost_evidence=cost_payload
    )
    cell = gate.CELL_ORDER[0]
    assert result["cost_gate"]["routes"]["directional"]["status"] == "PASS"
    assert result["cost_gate"]["routes"]["data"]["status"] == "FAIL"
    assert result["directional"]["by_cell"][cell]["cost_gate"]["status"] == "PASS"
    assert result["data"]["by_cell"][cell]["cost_gate"]["status"] == "FAIL"
    assert result["directional"]["by_cell"][cell]["metric_routes"]["token"]["cost_gate"]["status"] == "PASS"
    assert result["data"]["by_cell"][cell]["metric_routes"]["token"]["status"] == "FAIL"


@pytest.mark.parametrize(
    ("cost_status", "expected_status", "expected_reason"),
    [
        ("PASS", "PASS", None),
        ("FAIL", "FAIL", "cost_gate_failed"),
        ("UNKNOWN", "UNKNOWN", "cost_gate_unknown"),
    ],
)
def test_scientific_pass_preserves_cost_fail_vs_unknown(
    cost_status: str, expected_status: str, expected_reason: str | None
) -> None:
    result = scorer._effective_metric_route(
        [{"status": "PASS"}], cost_gate={"status": cost_status}
    )
    assert result["scientific_status"] == "PASS"
    assert result["status"] == expected_status
    if expected_reason is None:
        assert "reason" not in result
    else:
        assert result["reason"] == expected_reason

def test_cost_artifact_binds_to_the_frozen_public_run_manifest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    contract_path = _bind_contract(fixture)
    gate.write_freeze(
        registration_path=fixture["registration"],
        run_manifest_path=fixture["run"],
        output_path=fixture["freeze"],
        repository_root=fixture["root"],
    )
    evidence = cost.build_cost_evidence(
        run_manifest_path=fixture["run"],
        repository_root=fixture["root"],
        contract_path=contract_path,
    )
    artifact_path = fixture["task_root"] / "evaluation" / "cost_evidence.json"
    artifact = cost._write_create_only(  # noqa: SLF001 - create-only test artifact
        artifact_path, evidence
    )
    freeze = gate.validate_before_truth(
        freeze_path=fixture["freeze"], repository_root=fixture["root"]
    )
    loaded, bound = scorer._load_cost_evidence(  # noqa: SLF001 - boundary regression
        artifact_path, freeze=freeze, repository_root=fixture["root"]
    )
    assert loaded["run_manifest"] == freeze["run_manifest"]
    assert bound == artifact

    tampered = json.loads(artifact_path.read_text(encoding="utf-8"))
    tampered["run_manifest"]["sha256"] = "0" * 64
    tampered_path = fixture["task_root"] / "evaluation" / "cost_evidence_tampered.json"
    _write(tampered_path, tampered)
    with pytest.raises(scorer.ScoreError, match="run manifest sha256 changed"):
        scorer._load_cost_evidence(  # noqa: SLF001 - boundary regression
            tampered_path, freeze=freeze, repository_root=fixture["root"]
        )

    tampered = json.loads(artifact_path.read_text(encoding="utf-8"))
    tampered["comparisons"]["candidate_vs_fixed"]["fixed_method_id"] = gate.CURRENT_FIXED_METHOD_ID
    identity_path = fixture["task_root"] / "evaluation" / "cost_evidence_identity_tampered.json"
    _write(identity_path, tampered)
    with pytest.raises(scorer.ScoreError, match="cost comparison denominator changed"):
        scorer._load_cost_evidence(identity_path, freeze=freeze, repository_root=fixture["root"])
