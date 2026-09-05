from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import trr0004_fresh_confirmation as fc
import trr0004_score_confirmation as scorer
from token_reconstruction.freeze import FreezeError

from test_trr0004_fresh_confirmation import _evaluation_fixture


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _scorer_bundle(tmp_path: Path) -> dict[str, object]:
    bundle = _evaluation_fixture(tmp_path)
    binding_path = tmp_path / "truth_binding.json"
    _write_json(binding_path, bundle["truth_binding"])
    receipt_path = tmp_path / "receipts" / "confirmation.freeze.json"
    scorer.freeze_confirmation(
        root=tmp_path,
        panel_path=bundle["panel_path"],
        selection_plan_path=bundle["selection_plan"],
        registration_path=bundle["registration"],
        truth_binding_path=binding_path,
        output_root=bundle["output"],
        receipt_path=receipt_path,
    )
    bundle["truth_binding_path"] = binding_path
    bundle["scorer_receipt"] = receipt_path
    return bundle


def _score_kwargs(bundle: dict[str, object], result_path: Path) -> dict[str, object]:
    return {
        "root": bundle["root"],
        "panel_path": bundle["panel_path"],
        "selection_plan_path": bundle["selection_plan"],
        "registration_path": bundle["registration"],
        "truth_binding_path": bundle["truth_binding_path"],
        "truth_path": bundle["truth_path"],
        "output_root": bundle["output"],
        "receipt_path": bundle["scorer_receipt"],
        "result_path": result_path,
        "fit_data_path": None,
        "fit_token_key": "token_ids",
        "fit_mask_key": "attention_mask",
        "evidence_paths": [],
        "bootstrap_draws": 25,
        "bootstrap_seed": 4004,
    }


def test_score_revalidates_public_matrix_before_truth_and_writes_rows(tmp_path: Path) -> None:
    bundle = _scorer_bundle(tmp_path)
    result_path = tmp_path / "score.json"
    result = scorer._score(**_score_kwargs(bundle, result_path))
    assert result["truth_gate"]["verified_before_truth"] is True
    assert result["truth_gate"]["truth_opened_after_gate"] is True
    assert len(result["cells"]) == 4 * len(fc.METHOD_IDS)
    assert result["bootstrap"]["draws"] == 25
    assert result_path.is_file()


def test_missing_prediction_fails_before_truth_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _scorer_bundle(tmp_path)
    cell = fc.load_fresh_cells(bundle["panel"], repository_root=tmp_path)[0]
    missing = fc.expected_prediction_path(bundle["output"], cell=cell, method_id=fc.METHOD_IDS[0])
    missing.unlink()
    sidecar_calls: list[Path] = []
    original = fc.validate_confirmation_truth_sidecar

    def wrapped(path: Path, **kwargs):
        sidecar_calls.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(fc, "validate_confirmation_truth_sidecar", wrapped)
    with pytest.raises((scorer.ConfirmationScoreError, fc.ConfirmationError, FreezeError), match="frozen|incomplete|unavailable"):
        scorer._score(**_score_kwargs(bundle, tmp_path / "score.json"))
    assert sidecar_calls == []


def test_corrupt_bound_state_fails_before_truth_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _scorer_bundle(tmp_path)
    state_descriptor = bundle["bindings"][fc.METHOD_IDS[0]]["method_state"][0]
    state_path = tmp_path / state_descriptor["path"]
    state_path.write_bytes(state_path.read_bytes() + b"corruption")
    sidecar_calls: list[Path] = []
    monkeypatch.setattr(
        fc,
        "validate_confirmation_truth_sidecar",
        lambda path, **kwargs: sidecar_calls.append(path),
    )
    with pytest.raises((scorer.ConfirmationScoreError, fc.ConfirmationError), match="binding|hash|changed"):
        scorer._score(**_score_kwargs(bundle, tmp_path / "score.json"))
    assert sidecar_calls == []


def test_tampered_truth_is_checked_only_after_public_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _scorer_bundle(tmp_path)
    bundle["truth_path"].write_bytes(bundle["truth_path"].read_bytes() + b"tamper")
    public_gate_calls: list[bool] = []
    original_gate = fc.validate_complete_confirmation_predictions

    def wrapped_gate(*args, **kwargs):
        public_gate_calls.append(True)
        return original_gate(*args, **kwargs)

    monkeypatch.setattr(fc, "validate_complete_confirmation_predictions", wrapped_gate)
    with pytest.raises(fc.ConfirmationError, match="sidecar hash|size|changed"):
        scorer._score(**_score_kwargs(bundle, tmp_path / "score.json"))
    assert public_gate_calls == [True]


def test_public_fit_frequency_bins_and_paired_bootstrap_are_deterministic(tmp_path: Path) -> None:
    token_ids = torch.tensor(
        [[fc.BOS_TOKEN_ID, 1, 2, 2, 3, 0], [fc.BOS_TOKEN_ID, 4, 5, 0, 0, 0]],
        dtype=torch.int64,
    )
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]],
        dtype=torch.int64,
    )
    fit_path = tmp_path / "fit.safetensors"
    save_file({"token_ids": token_ids, "attention_mask": mask}, str(fit_path))
    counts, source = scorer._load_fit_frequency(
        fit_path,
        vocab_size=fc.VOCAB_SIZE,
        token_key="token_ids",
        mask_key="attention_mask",
        root=tmp_path,
    )
    assert source["post_bos_fit_examples"] == 6
    assert counts[2].item() == 2
    assert counts[1].item() == 1
    first = scorer._paired_bootstrap([0.5, 1.0, 0.0], [0.0, 0.5, 0.0], draws=100, seed=4004)
    second = scorer._paired_bootstrap([0.5, 1.0, 0.0], [0.0, 0.5, 0.0], draws=100, seed=4004)
    assert first == second
    assert first["delta_estimate"] == pytest.approx(1.0 / 3.0)




def test_timing_sources_use_predictor_top_level_and_cell_records(tmp_path: Path) -> None:
    output = tmp_path / "merged"
    output.mkdir()
    evidence_path = output / "method_run_evidence.json"
    _write_json(
        evidence_path,
        {
            "started_utc": "2026-09-05T00:00:00Z",
            "ended_utc": "2026-09-05T00:00:02Z",
            "wall_seconds": 2.0,
            "startup": {"seconds": 0.75, "boundary": "before first timed call"},
            "cold_peak_memory": {"cuda_peak_reserved_bytes": 18},
            "per_cell_peak_memory": {
                "finance__public_base": {"cuda_peak_reserved_bytes": 20}
            },
            "model": {
                "model_load_seconds": 0.25,
                "public_embedding_load_seconds": 0.05,
                "method_state_load_seconds": 0.01,
            },
            "method_timings": {
                "historical_alpaca_a1": [
                    {
                        "cell_id": "finance__public_base",
                        "records": 16,
                        "measured_runs_per_record": 3,
                        "warmup_runs_per_record": 1,
                        "timed_interval_total_seconds": 1.5,
                        "warmup_seconds_sum": 0.2,
                        "measured_seconds_sum": 1.2,
                        "per_record_measured_seconds": [0.075] * 16,
                        "peak_memory": {
                            "cuda_peak_allocated_bytes": 10,
                            "cuda_peak_reserved_bytes": 20,
                            "process_max_rss_bytes": 30,
                        },
                    }
                ]
            },
        },
    )
    timing, costs = scorer._timing_sources(
        output_root=output,
        evidence_paths=[evidence_path],
        root=tmp_path,
    )
    assert costs["cold_runs"][0]["started_utc"] == "2026-09-05T00:00:00Z"
    assert costs["cold_runs"][0]["ended_utc"] == "2026-09-05T00:00:02Z"
    assert costs["cold_runs"][0]["cold_components"]["model_load_seconds"] == 0.25
    assert costs["cold_runs"][0]["startup"]["seconds"] == 0.75
    assert costs["cold_runs"][0]["cold_peak_memory"]["cuda_peak_reserved_bytes"] == 18
    assert costs["cold_runs"][0]["per_cell_peak_memory"]["finance__public_base"]["cuda_peak_reserved_bytes"] == 20
    assert timing[("finance__public_base", "historical_alpaca_a1")]["record"]["measured_seconds_sum"] == 1.2


def test_steady_costs_reports_one_run_latency_and_method_peak(tmp_path: Path) -> None:
    bundle = _evaluation_fixture(tmp_path)
    cells = fc.load_fresh_cells(bundle["panel"], repository_root=tmp_path)
    timing: dict[tuple[str, str], dict[str, object]] = {}
    for cell in cells:
        for method_id in fc.METHOD_IDS:
            timing[(cell.cell_id, method_id)] = {
                "source": {"path": "timing.json", "bytes": 1, "sha256": "0" * 64},
                "record": {
                    "cell_id": cell.cell_id,
                    "method_id": method_id,
                    "records": cell.records,
                    "measured_runs_per_record": 3,
                    "warmup_runs_per_record": 1,
                    "timed_interval_total_seconds": 1.2,
                    "warmup_seconds_sum": 0.2,
                    "measured_seconds_sum": 0.9,
                    "per_record_measured_seconds": [0.05625] * cell.records,
                    "peak_memory": {
                        "cuda_peak_allocated_bytes": 10,
                        "cuda_peak_reserved_bytes": 20,
                        "process_max_rss_bytes": 30,
                    },
                },
            }
    result = scorer._steady_costs(timing, method_ids=fc.METHOD_IDS, cells=cells)
    row = result["per_method"][fc.METHOD_IDS[0]]
    assert row["deployed_measured_seconds_sum_per_one_run"] == pytest.approx(1.2)
    assert row["deployed_latency_mean_seconds_per_record"] == pytest.approx(0.01875)
    assert row["deployed_latency_median_seconds_per_record"] == pytest.approx(0.01875)
    assert row["peak_memory_max_across_cells"]["process_max_rss_bytes"] == 30
