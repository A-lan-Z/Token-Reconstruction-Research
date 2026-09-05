from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts import trr0005_fit_joint_decoders as fit_runner
from scripts import trr0005_run_predictions as runner
from token_reconstruction import trr0005_contract as contract


class _ShapeOnly:
    shape = (contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS, 2048)


class _FakeAdapter:
    def __init__(self, *, method_id: str, requires_device: bool = False) -> None:
        self.method_id = method_id
        self.device = torch.device("cpu")
        self.input_device_required = requires_device
        self.calls = 0
        self.devices: list[tuple[torch.device, torch.device, torch.device]] = []

    def __call__(self, activation, mask, positions):
        self.calls += 1
        self.devices.append((activation.device, mask.device, positions.device))
        output = torch.full((contract.SEQUENCE_TOKENS,), 7, dtype=torch.long, device=activation.device)
        output[0] = contract.BOS_TOKEN_ID
        return output

    def evidence(self):
        return {
            "calls": self.calls,
            "public_prefix_calls": 0,
            "candidate_simulations": 0,
            "candidate_output": "forbidden",
        }


def _record() -> runner.FreshRecord:
    return runner.FreshRecord(
        record_id="qualification-record",
        activation=torch.zeros((contract.SEQUENCE_TOKENS, 2048), dtype=torch.bfloat16),
        attention_mask=torch.ones(contract.SEQUENCE_TOKENS, dtype=torch.bool),
        position_ids=torch.arange(contract.SEQUENCE_TOKENS, dtype=torch.long),
    )


def test_timed_predictor_stages_legacy_a1_inputs_on_adapter_device():
    adapter = _FakeAdapter(method_id="historical_alpaca_a1", requires_device=True)
    output = runner._timed_predictor(adapter)(_record())
    assert output.shape == (contract.SEQUENCE_TOKENS,)
    assert output[0].item() == contract.BOS_TOKEN_ID
    assert adapter.calls == 1
    assert adapter.devices == [(torch.device("cpu"),) * 3]


def test_one_record_qualification_attests_real_warm_and_measured_calls():
    adapter = _FakeAdapter(method_id="historical_alpaca_a1")
    output, timing = runner._one_record_warm_measured(
        adapter,
        _record(),
        device=torch.device("cpu"),
    )
    assert output.shape == (contract.SEQUENCE_TOKENS,)
    assert adapter.calls == 2
    assert timing["records"] == 1
    assert timing["warmup_runs_per_record"] == 1
    assert timing["measured_runs_per_record"] == 1
    assert timing["warmup_output_exact_match_measured"] is True
    assert timing["measured_output_selected"] is True
    assert timing["warmup_seconds"] >= 0.0
    assert timing["measured_seconds"] >= 0.0


def test_merged_driver_receipt_keeps_prediction_shape_contract():
    method_id = "original__joint_full_affine"
    mask = torch.ones(
        (contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS), dtype=torch.bool
    )
    prediction = torch.full(
        (contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS),
        contract.INVALID_TOKEN_ID,
        dtype=torch.long,
    )
    prediction[:, 0] = contract.BOS_TOKEN_ID
    cell = runner.FreshCell(
        cell_id="pile__public_base",
        style="pile",
        condition="public_base",
        record_ids=tuple(f"record-{i}" for i in range(contract.RECORDS_PER_DOMAIN)),
        activations=_ShapeOnly(),  # type: ignore[arg-type]
        attention_mask=mask,
        position_ids=mask.to(torch.long).cumsum(1).sub(1),
        observation_path=Path("observation.safetensors"),
        observation_sha256="d" * 64,
    )
    timing = {
        "warmup_runs_per_record": 1,
        "measured_runs_per_record": 1,
        "warmup_output_exact_match_measured": True,
        "measured_output_selected": True,
        "predictions_for_digest": prediction,
    }
    descriptor = runner.prediction_descriptor(
        cell_id=cell.cell_id,
        method_id=method_id,
        predictions=prediction,
        timing=timing,
        panel_sha256="a" * 64,
        selection_plan_sha256="b" * 64,
        observation_sha256=cell.observation_sha256,
    )
    receipt = runner._method_timing_receipt(
        timing=timing,
        cell=cell,
        method_id=method_id,
        adapter=_FakeAdapter(method_id=method_id),
        artifact={"path": "prediction.safetensors"},
        load_evidence={"runtime_load_seconds": 0.1},
        peak={"process_max_rss_bytes": 1},
        root=Path.cwd(),
    )
    descriptor.update(receipt)
    contract.validate_prediction_descriptor(
        descriptor,
        cell_id=cell.cell_id,
        method_id=method_id,
    )
    assert descriptor["shape"] == [contract.RECORDS_PER_DOMAIN, contract.SEQUENCE_TOKENS]
    assert descriptor["observation_shape"] == [
        contract.RECORDS_PER_DOMAIN,
        contract.SEQUENCE_TOKENS,
        2048,
    ]


def test_archived_prediction_row_reads_predictions_only(tmp_path):
    path = tmp_path / "archived.safetensors"
    row = torch.full((1, contract.SEQUENCE_TOKENS), 4, dtype=torch.long)
    row[:, 0] = contract.BOS_TOKEN_ID
    save_file(
        {"predictions": row},
        str(path),
        metadata={
            "task_id": "TRR-0004",
            "method_id": "historical_alpaca_a1",
            "cell_id": "finance__public_base",
        },
    )
    loaded, evidence = runner._load_archived_prediction_row(
        path,
        record_index=0,
        method_id="historical_alpaca_a1",
    )
    assert torch.equal(loaded, row[0])
    assert evidence["tensor_key"] == "predictions"
    assert evidence["shape"] == [1, contract.SEQUENCE_TOKENS]


def test_cosine_repair_cli_is_causal_only_and_keeps_fixed_recipe():
    parser = fit_runner._parser()
    required = [
        "--original-manifest", "original.json",
        "--enriched-manifest", "enriched.json",
        "--output-root", "output",
    ]
    old_mode = parser.parse_args(required + ["--attention-score-mode", "cosine_scale4"])
    with pytest.raises(fit_runner.JointFitRunnerError, match="causal arms"):
        fit_runner._validate_args(old_mode)

    repair = parser.parse_args(
        required
        + [
            "--qualification-only",
            "--causal-only",
            "--attention-score-mode",
            "cosine_scale4",
        ]
    )
    fit_runner._validate_args(repair)
    assert repair.attention_score_mode == "cosine_scale4"
    assert repair.causal_only is True
    assert repair.steps == 3000
    assert repair.record_batch_size == 8
    assert repair.position_budget == 512


def test_driver_defaults_to_repaired_causal_root():
    args = runner._parser().parse_args(["--output-root", "output"] )
    assert args.fit_root == Path("experiments/TRR-0005/joint_fit_v1")
    assert args.causal_fit_root == Path("experiments/TRR-0005/joint_fit_qknorm_v1")
    fitted, causal = runner._resolve_fit_roots(args)
    assert fitted.name == "joint_fit_v1"
    assert causal.name == "joint_fit_qknorm_v1"


def test_registered_joint_state_paths_use_causal_root_only_for_causal_arm(tmp_path):
    fitted = tmp_path / "joint_fit_v1"
    causal = tmp_path / "joint_fit_qknorm_v1"
    methods = {}
    for distribution in ("original", "enriched"):
        for base_method in runner.JOINT_STATE_METHODS:
            state_root = causal if base_method == "affine_causal_h_attention128" else fitted
            method_id = f"{distribution}__{base_method}"
            methods[method_id] = runner.RegisteredMethod(
                method_id=method_id,
                binding={},
                state_path=state_root / distribution / base_method / "selected.safetensors",
                config_paths=(),
                code_paths=(),
                runtime_paths={},
            )
    runner._validate_joint_state_roots(
        methods, fit_root=fitted, causal_fit_root=causal
    )
    causal_id = "original__affine_causal_h_attention128"
    methods[causal_id] = runner.RegisteredMethod(
        method_id=causal_id,
        binding={},
        state_path=fitted / "original" / "affine_causal_h_attention128" / "selected.safetensors",
        config_paths=(),
        code_paths=(),
        runtime_paths={},
    )
    with pytest.raises(runner.PredictionRunnerError, match="canonical root"):
        runner._validate_joint_state_roots(
            methods, fit_root=fitted, causal_fit_root=causal
        )


def test_fit_summary_uses_repaired_causal_source(tmp_path):
    fitted = tmp_path / "joint_fit_v1"
    causal = tmp_path / "joint_fit_qknorm_v1"
    methods = tuple(runner.JOINT_STATE_METHODS)

    def write_evidence(root, method_names, mode):
        distributions = {}
        for distribution in ("original", "enriched"):
            rows = {}
            for method_name in method_names:
                curve = root / distribution / method_name / "learning_curve.json"
                curve.parent.mkdir(parents=True, exist_ok=True)
                curve.write_text("{}")
                rows[method_name] = {
                    "canonical_method_id": f"{distribution}__{method_name}",
                    "selected_step": 100,
                    "best_validation_style_balanced_token_accuracy": 0.5,
                    "checkpoint_steps": [0, 100],
                    "curve": {
                        "path": str(curve),
                        "bytes": curve.stat().st_size,
                        "sha256": hashlib.sha256(curve.read_bytes()).hexdigest(),
                        "points": [],
                    },
                    "optimization_update_seconds": 1.0,
                    "selection_validation_seconds": 2.0,
                    "final_fit_diagnostic_seconds": 3.0,
                    "state_io_seconds": 0.1,
                    "arm_wall_seconds": 6.0,
                    "timing_accounting": {},
                }
            distributions[distribution] = {
                "contract_distribution_id": distribution,
                "fit_geometry": [1200, 192, 2048],
                "fit_record_count": 1200,
                "fit_post_bos_positions": 124371,
                "validation_geometry": [48, 192, 2048],
                "preparation_timing": {},
                "methods": rows,
            }
        root.mkdir(parents=True, exist_ok=True)
        (root / "run_evidence.json").write_text(
            json.dumps({
                "task_id": "TRR-0005",
                "status": "JOINT_FIT_COMPLETE_NO_FINAL_EVALUATION",
                "final_holdout_loaded": False,
                "git_commit": "a" * 40,
                "elapsed_seconds": 1.0,
                "fixed_settings": {
                    "methods": list(method_names),
                    "attention_score_mode": mode,
                },
                "sampler_cross_distribution": {"status": "IDENTICAL_MASK_AND_SCHEDULE"},
                "distributions": distributions,
            })
        )

    write_evidence(fitted, methods, "dot_product")
    write_evidence(causal, ("affine_causal_h_attention128",), "cosine_scale4")
    summary = runner._fit_evidence_summary(
        fitted, causal_fit_root=causal, repository_root=tmp_path
    )
    original = summary["distributions"]["original"]["methods"]
    assert original["joint_full_affine"]["source_fit_root"] == str(fitted.resolve())
    assert original["affine_trained_diagonal_attention128"]["source_fit_root"] == str(fitted.resolve())
    assert original["affine_causal_h_attention128"]["source_fit_root"] == str(causal.resolve())
    assert original["affine_causal_h_attention128"]["source_attention_score_mode"] == "cosine_scale4"
    assert summary["fit_roots"]["causal"] == str(causal.resolve())
