from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import trr0004_fit_contextual_extensions as runner


def _write_tensor(path: Path, key: str, value: torch.Tensor) -> None:
    save_file({key: value.contiguous()}, str(path))


def _fixture_inputs(tmp_path: Path) -> list[str]:
    hidden = 4
    vocab = 10
    fit_n = 8
    val_n = 4
    sequence = 5
    torch.manual_seed(401)
    fit_x = torch.randn(fit_n, sequence, hidden, dtype=torch.float32)
    val_x = torch.randn(val_n, sequence, hidden, dtype=torch.float32)
    fit_y = torch.randint(0, vocab, (fit_n, sequence), dtype=torch.int32)
    val_y = torch.randint(0, vocab, (val_n, sequence), dtype=torch.int32)
    fit_y[:, 0] = runner.BOS_TOKEN_ID
    val_y[:, 0] = runner.BOS_TOKEN_ID
    table = torch.nn.functional.normalize(torch.randn(vocab, hidden), dim=-1)
    base = {
        "W": torch.eye(hidden, dtype=torch.float32),
        "b": torch.zeros(hidden, dtype=torch.float32),
        "s": torch.tensor(2.0, dtype=torch.float32),
    }
    fit_ids = [f"alpaca-fit-{i}" for i in range(fit_n)]
    val_ids = [f"alpaca-val-{i}" for i in range(val_n)]
    registration = {
        "schema": "token-reconstruction.trr0004-contextual-extension-registration.v1",
        "contains_source_text": False,
        "contains_token_ids": False,
        "fit": {
            "records": [
                {"record_id": record_id, "full_token_count": sequence}
                for record_id in fit_ids
            ],
            "large_nested": {"post_bos_positions": fit_n * (sequence - 1)},
        },
        "validation": {
            "records": [{"record_id": record_id} for record_id in val_ids]
        },
    }
    paths = {
        "base": tmp_path / "base.safetensors",
        "fit_x": tmp_path / "fit_observations.safetensors",
        "fit_y": tmp_path / "fit_truth.safetensors",
        "fit_records": tmp_path / "fit_records.json",
        "val_x": tmp_path / "validation_observations.safetensors",
        "val_y": tmp_path / "validation_truth.safetensors",
        "val_records": tmp_path / "validation_records.json",
        "embedding": tmp_path / "embedding.safetensors",
        "registration": tmp_path / "registration.json",
        "groups": tmp_path / "groups.json",
        "output": tmp_path / "run",
    }
    save_file(base, str(paths["base"]))
    _write_tensor(paths["fit_x"], "activations", fit_x)
    _write_tensor(paths["fit_y"], "token_ids", fit_y)
    _write_tensor(paths["val_x"], "activations", val_x)
    _write_tensor(paths["val_y"], "token_ids", val_y)
    _write_tensor(paths["embedding"], "embeddings", table)
    paths["fit_records"].write_text(json.dumps({"records": [{"record_id": x, "full_token_count": sequence} for x in fit_ids]}))
    paths["val_records"].write_text(json.dumps({"records": [{"record_id": x} for x in val_ids]}))
    paths["registration"].write_text(json.dumps(registration))
    paths["groups"].write_text(json.dumps({"groups": ["style_a", "style_b", "style_a", "style_b"]}))
    return [
        "--base-state", str(paths["base"]),
        "--registration", str(paths["registration"]),
        "--fit-observations", str(paths["fit_x"]),
        "--fit-truth", str(paths["fit_y"]),
        "--fit-records", str(paths["fit_records"]),
        "--validation-observations", str(paths["val_x"]),
        "--validation-truth", str(paths["val_y"]),
        "--validation-records", str(paths["val_records"]),
        "--validation-groups", str(paths["groups"]),
        "--embedding-table", str(paths["embedding"]),
        "--output-root", str(paths["output"]),
        "--steps", "32",
        "--subset-steps", "4",
        "--subset-records", "8",
        "--validation-every", "4",
        "--position-budget", "3",
        "--device", "cpu",
    ]


def test_common_checkpoint_schedule_has_early_and_regular_points() -> None:
    assert runner.checkpoint_steps(3000) == tuple(
        [0, 25, 50, 75, 100, 150, 200] + list(range(300, 3001, 100))
    )


def test_parser_defaults_bound_contextual_runs() -> None:
    args = runner._parser().parse_args(
        ["--base-state", "base.safetensors", "--registration", "registration.json", "--output-root", "out"]
    )
    assert args.minimum_free_gib == 8.0
    assert args.maximum_gpu_reserved_gib == 6.0
    assert args.maximum_host_rss_gib == 16.0
    assert args.max_seconds == 1200.0


def test_resource_guard_fails_closed_on_deadline_and_host_rss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        minimum_free_gib=8.0,
        maximum_gpu_reserved_gib=6.0,
        maximum_host_rss_gib=16.0,
    )
    with pytest.raises(runner.ContextualFitError, match="wall-time guard"):
        runner._resource_guard(args, torch.device("cpu"), deadline=0.0, stage="expired")

    monkeypatch.setattr(
        runner.resource,
        "getrusage",
        lambda _kind: SimpleNamespace(ru_maxrss=17 * 2**30 // 1024),
    )
    with pytest.raises(runner.ContextualFitError, match="host RSS guard"):
        runner._resource_guard(args, torch.device("cpu"), stage="rss")


def test_position_schedule_is_seed_bound_and_excludes_bos() -> None:
    valid = torch.ones(8, 6, dtype=torch.bool)
    first = runner.build_position_schedule(valid, steps=5, position_budget=3, seed=17)
    second = runner.build_position_schedule(valid, steps=5, position_budget=3, seed=17)
    assert torch.equal(first.record_indices, second.record_indices)
    assert torch.equal(first.selected_mask, second.selected_mask)
    assert not first.selected_mask[:, :, 0].any().item()
    assert int(first.selected_mask.sum(dim=(1, 2)).max()) <= 3
    assert runner.schedule_digest(first) == runner.schedule_digest(second)


def test_contextual_fit_saves_both_methods_curves_states_and_shared_schedule(tmp_path: Path) -> None:
    argv = _fixture_inputs(tmp_path)
    args = runner._parser().parse_args(argv)
    evidence = runner.run_fit(args)
    assert evidence["method_ids"] == list(runner.EXTENSION_METHODS)
    assert evidence["current_evaluator_truth_accessed"] is False
    assert evidence["candidate_simulations"] == 0
    assert evidence["sampler"]["same_main_schedule_for_methods"] is True
    assert "earliest nonzero checkpoint" in evidence["fixed_settings"]["selection_rule"]
    methods = evidence["methods"]
    for method_id in runner.EXTENSION_METHODS:
        main = methods[method_id]["main"]
        subset = methods[method_id]["subset"]
        assert main["selected_step"] > 0
        assert main["schedule_sha256"] == evidence["sampler"]["schedule"]["main"]["schedule_sha256"]
        assert main["validation_max_projection_rows"] <= 3
        assert main["final_fit_evaluation_selection_independent"] is True
        assert main["final_fit_evaluation"]["record_count"] == 8
        assert main["final_fit_evaluation"]["token_rows"] == 32
        assert main["final_fit_evaluation_seconds"] >= 0.0
        assert subset["final_fit_evaluation"] is None
        assert subset["final_fit_subset_token_accuracy"] is not None
        assert Path(main["state"]["path"]).is_file()
        assert Path(subset["state"]["path"]).is_file()
        curve = json.loads(Path(main["curve"]["path"]).read_text())
        assert [point["step"] for point in curve["curve"]] == [0, 25, 32]
        subset_curve = json.loads(Path(subset["curve"]["path"]).read_text())
        assert "fit_subset" in subset_curve["curve"][0]
    schedule = evidence["sampler"]["schedule"]
    assert Path(schedule["path"]).is_file()
    assert (args.output_root / "run_evidence.json").is_file()
    assert evidence["resource_guard"]["checks"] >= 2 * (32 + 4)
    assert evidence["resource_guard"]["limits"]["minimum_free_gpu_bytes"] == 8 * 2**30



def test_contextual_fit_accepts_pinned_padded_fit_manifest(tmp_path: Path) -> None:
    argv = _fixture_inputs(tmp_path)
    args = runner._parser().parse_args(argv)
    paths = {
        "fit_observations": args.fit_observations,
        "fit_truth": args.fit_truth,
        "fit_records": args.fit_records,
        "validation_observations": args.validation_observations[0],
        "validation_truth": args.validation_truth[0],
        "validation_records": args.validation_records[0],
        "embedding_table": args.embedding_table,
    }
    resources = {}
    for name, path in paths.items():
        record = runner._file_record(path, label=name)
        record["path"] = path.name
        resources[name] = record
    manifest = {
        "schema": runner.FIT_DATA_SCHEMA,
        "layout": "padded_records",
        "alignment": {
            "mode": "current_token",
            "observation_index": "i",
            "label_index": "i",
            "bos_position": 0,
            "scored_positions": "post_bos",
        },
        "resources": resources,
    }
    manifest_path = tmp_path / "fit_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    args.fit_manifest = manifest_path
    args.output_root = tmp_path / "manifest_run"
    evidence = runner.run_fit(args)
    assert evidence["data"]["fit_manifest_sha256"] == runner.file_sha256(manifest_path)
    assert evidence["data"]["resources"]["fit_truth"]["sha256"] == runner.file_sha256(args.fit_truth)


def test_registration_mismatch_is_rejected_before_truth_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    argv = _fixture_inputs(tmp_path)
    args = runner._parser().parse_args(argv)
    original = runner._load_tensor
    seen_keys: list[str] = []

    def fail_if_truth(path, *, key, label):
        seen_keys.append(key)
        if key == "token_ids":
            raise AssertionError("truth was opened")
        return original(path, key=key, label=label)

    monkeypatch.setattr(runner, "_load_tensor", fail_if_truth)
    registration = json.loads(args.registration.read_text())
    registration["fit"]["records"][0]["record_id"] = "overlap-without-data"
    args.registration.write_text(json.dumps(registration))
    with pytest.raises(runner.ContextualFitError, match="exactly match"):
        runner._load_context_data(args)
    assert "token_ids" not in seen_keys


def _write_combined_activation(path: Path, observations: torch.Tensor, truth: torch.Tensor) -> None:
    rows, positions, _ = observations.shape
    mask = torch.ones((rows, positions), dtype=torch.uint8)
    position_ids = torch.arange(positions, dtype=torch.int64).expand(rows, -1).contiguous()
    selector = mask.to(dtype=torch.bool)
    save_file(
        {
            "activations": observations.contiguous(),
            "token_ids": truth.to(dtype=torch.int32).contiguous(),
            "attention_mask": mask.contiguous(),
            "position_ids": position_ids,
            "post_bos_selector_small": selector.clone(),
            "post_bos_selector_large": selector.clone(),
        },
        str(path),
    )


def test_runtime_uint8_mask_is_binary_at_load_boundary() -> None:
    with pytest.raises(runner.ContextualFitError, match="binary"):
        runner._validate_mask(
            torch.tensor([[1, 2, 0]], dtype=torch.uint8),
            rows=1,
            positions=3,
            label="runtime mask",
        )


def test_combined_validation_adapter_preserves_native_active_outputs(tmp_path: Path) -> None:
    """Exercise footing's six-key artifacts and two native validation geometries."""

    hidden = 4
    vocab = 10
    fit_n = 8
    torch.manual_seed(991)
    fit_x = torch.randn(fit_n, 5, hidden)
    fit_y = torch.randint(0, vocab, (fit_n, 5), dtype=torch.int32)
    fit_y[:, 0] = runner.BOS_TOKEN_ID
    alpaca_x = torch.randn(24, 5, hidden)
    alpaca_y = torch.randint(0, vocab, (24, 5), dtype=torch.int32)
    alpaca_y[:, 0] = runner.BOS_TOKEN_ID
    pile_x = torch.randn(24, 3, hidden)
    pile_y = torch.randint(0, vocab, (24, 3), dtype=torch.int32)
    pile_y[:, 0] = runner.BOS_TOKEN_ID
    table = torch.nn.functional.normalize(torch.randn(vocab, hidden), dim=-1)
    paths = {
        "base": tmp_path / "base.safetensors",
        "fit": tmp_path / "fit_combined.safetensors",
        "alpaca": tmp_path / "validation_alpaca_combined.safetensors",
        "pile": tmp_path / "validation_pile_combined.safetensors",
        "fit_records": tmp_path / "fit_records.json",
        "alpaca_records": tmp_path / "alpaca_records.json",
        "pile_records": tmp_path / "pile_records.json",
        "registration": tmp_path / "registration.json",
        "groups": tmp_path / "validation_groups.json",
        "embedding": tmp_path / "embedding.safetensors",
        "output": tmp_path / "run",
    }
    save_file(
        {
            "W": torch.eye(hidden),
            "b": torch.zeros(hidden),
            "s": torch.tensor(2.0),
        },
        str(paths["base"]),
    )
    _write_combined_activation(paths["fit"], fit_x, fit_y)
    _write_combined_activation(paths["alpaca"], alpaca_x, alpaca_y)
    _write_combined_activation(paths["pile"], pile_x, pile_y)
    save_file({"embeddings": table}, str(paths["embedding"]))
    fit_ids = [f"fit-{index}" for index in range(fit_n)]
    alpaca_ids = [f"alpaca-val-{index}" for index in range(24)]
    pile_ids = [f"pile-val-{index}" for index in range(24)]
    paths["fit_records"].write_text(
        json.dumps({"records": [{"record_id": value, "full_token_count": 5} for value in fit_ids]})
    )
    paths["alpaca_records"].write_text(
        json.dumps({"records": [{"record_id": value} for value in alpaca_ids]})
    )
    paths["pile_records"].write_text(
        json.dumps({"records": [{"record_id": value} for value in pile_ids]})
    )
    registration = {
        "schema": "token-reconstruction.trr0004-contextual-extension-registration.v1",
        "contains_source_text": False,
        "contains_token_ids": False,
        "fit": {
            "records": [{"record_id": value} for value in fit_ids],
            "large_nested": {"post_bos_positions": fit_n * 4},
        },
        "validation": {
            "records": [{"record_id": value} for value in alpaca_ids + pile_ids]
        },
    }
    paths["registration"].write_text(json.dumps(registration))
    paths["groups"].write_text(json.dumps({"groups": ["alpaca"] * 24 + ["pile"] * 24}))
    argv = [
        "--base-state", str(paths["base"]),
        "--registration", str(paths["registration"]),
        "--fit-artifact", str(paths["fit"]),
        "--fit-records", str(paths["fit_records"]),
        "--validation-artifact", str(paths["alpaca"]),
        "--validation-artifact", str(paths["pile"]),
        "--validation-records", str(paths["alpaca_records"]),
        "--validation-records", str(paths["pile_records"]),
        "--validation-groups", str(paths["groups"]),
        "--embedding-table", str(paths["embedding"]),
        "--output-root", str(paths["output"]),
        "--steps", "1",
        "--subset-steps", "1",
        "--subset-records", "8",
        "--validation-every", "1",
        "--position-budget", "3",
        "--device", "cpu",
    ]
    args = runner._parser().parse_args(argv)
    data = runner._load_context_data(args)
    assert len(data.validation_record_ids) == 48
    assert data.validation_groups.count("alpaca") == 24
    assert data.validation_groups.count("pile") == 24
    assert data.validation_native_geometries == ((24, 5, hidden), (24, 3, hidden))
    assert data.validation_padding["mode"] == "right_pad_to_max_sequence_for_masked_causal_pass"

    base_state = runner.load_file(str(paths["base"]), device="cpu")
    model = runner.build_causal_extension(
        runner.FrozenAffineBase(base_state), runner.CAUSAL_ATTENTION_METHOD
    )
    torch.manual_seed(992)
    with torch.no_grad():
        for parameter in model.added_path.parameters():
            parameter.normal_(mean=0.0, std=0.05)
    model.eval()
    with torch.no_grad():
        native_logits = model(alpaca_x, torch.ones((24, 5), dtype=torch.bool), table)
        padded_logits = model(
            data.validation_observations[:24], data.validation_valid_mask[:24], table
        )
    torch.testing.assert_close(native_logits, padded_logits[:, :5], rtol=0, atol=2e-6)

    evidence = runner.run_fit(args)
    assert evidence["data"]["validation_record_count"] == 48
    assert evidence["data"]["validation_group_record_counts"] == {"alpaca": 24, "pile": 24}
    assert evidence["data"]["validation_native_geometries"] == [[24, 5, hidden], [24, 3, hidden]]
    assert evidence["data"]["validation_padding"]["masked_padding"] is True
