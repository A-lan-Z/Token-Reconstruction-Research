from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from trr0004_build_affine_fit_manifest import (
    AffineManifestAdapterError,
    _check_right_padded_mask,
)
from token_reconstruction.historical_affine_ce import (
    FIT_DATA_SCHEMA,
    HistoricalAffineCEConfig,
    HistoricalAffineCEDecoder,
    HistoricalAffineCEError,
    direct_prediction_tensor,
    evaluation_schedule,
    flatten_current_token_records,
    load_historical_affine_ce,
    load_public_fit_bundle,
    normalized_embedding_table,
    save_historical_affine_ce,
    tensor_sha256,
    train_historical_affine_ce,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_resource(
    root: Path,
    name: str,
    key: str,
    value: torch.Tensor,
) -> dict[str, object]:
    path = root / name
    save_file({key: value.contiguous()}, str(path))
    return {
        "path": name,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _records_resource(root: Path, name: str, records: list[dict[str, object]]) -> dict[str, object]:
    path = root / name
    path.write_text(json.dumps({"records": records}, sort_keys=True) + "\n")
    return {
        "path": name,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _manifest(
    root: Path,
    *,
    layout: str = "padded_records",
    fit_records: list[dict[str, object]] | None = None,
    validation_records: list[dict[str, object]] | None = None,
    fit_observations: torch.Tensor | None = None,
    validation_observations: torch.Tensor | None = None,
    fit_truth: torch.Tensor | None = None,
    validation_truth: torch.Tensor | None = None,
    table: torch.Tensor | None = None,
    fit_valid_mask: torch.Tensor | None = None,
    validation_valid_mask: torch.Tensor | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    fit_records = fit_records or [
        {"record_id": "fit-a", "full_token_count": 4, "post_bos_token_count": 3},
        {"record_id": "fit-b", "full_token_count": 4, "post_bos_token_count": 3},
    ]
    validation_records = validation_records or [
        {"record_id": "validation-a", "full_token_count": 4, "post_bos_token_count": 3}
    ]
    table = table if table is not None else torch.eye(6)
    fit_observations = fit_observations if fit_observations is not None else table[[5, 0, 1, 2, 5, 2, 3, 4]].reshape(2, 4, 6)
    validation_observations = validation_observations if validation_observations is not None else table[[5, 1, 0, 3]].reshape(1, 4, 6)
    fit_truth = fit_truth if fit_truth is not None else torch.tensor([[5, 0, 1, 2], [5, 2, 3, 4]], dtype=torch.int32)
    validation_truth = validation_truth if validation_truth is not None else torch.tensor([[5, 1, 0, 3]], dtype=torch.int32)
    resources = {
        "fit_records": _records_resource(root, "fit_records.json", fit_records),
        "validation_records": _records_resource(root, "validation_records.json", validation_records),
        "fit_observations": _tensor_resource(root, "fit_observations.safetensors", "activations", fit_observations),
        "validation_observations": _tensor_resource(root, "validation_observations.safetensors", "activations", validation_observations),
        "fit_truth": _tensor_resource(root, "fit_truth.safetensors", "token_ids", fit_truth),
        "validation_truth": _tensor_resource(root, "validation_truth.safetensors", "token_ids", validation_truth),
        "embedding_table": _tensor_resource(root, "embeddings.safetensors", "embeddings", table),
    }
    if fit_valid_mask is not None:
        resources["fit_valid_mask"] = _tensor_resource(
            root, "fit_valid_mask.safetensors", "attention_mask", fit_valid_mask
        )
    if validation_valid_mask is not None:
        resources["validation_valid_mask"] = _tensor_resource(
            root, "validation_valid_mask.safetensors", "attention_mask", validation_valid_mask
        )
    manifest = {
        "schema": FIT_DATA_SCHEMA,
        "task_id": "TRR-0004",
        "layout": layout,
        "bos_token_id": 5,
        "embedding_table_normalized": True,
        "alignment": {
            "mode": "current_token",
            "observation_index": "i",
            "label_index": "i",
            "bos_position": 0,
            "scored_positions": "post_bos",
        },
        "resources": resources,
    }
    path = root / "fit_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def test_initialization_matches_historical_recipe_and_bias_is_the_only_arm_difference() -> None:
    no_bias = HistoricalAffineCEDecoder(4, 7, bias_mode="none")
    with_bias = HistoricalAffineCEDecoder(4, 7, bias_mode="vocab")
    assert torch.equal(no_bias.W.detach(), torch.eye(4))
    assert torch.equal(no_bias.b.detach(), torch.zeros(4))
    assert no_bias.s.item() == pytest.approx(3.0)
    assert no_bias.vocab_bias is None
    assert with_bias.vocab_bias is not None
    assert with_bias.vocab_bias.shape == (7,)
    assert with_bias.parameter_count() == no_bias.parameter_count() + 7
    assert with_bias.resolved_method_id == "historical_affine_ce_vocab_bias"


def test_forward_orientation_scale_and_optional_vocab_bias() -> None:
    table = torch.eye(4)
    activation = table[[0, 2]]
    no_bias = HistoricalAffineCEDecoder(4, 4, bias_mode="none")
    logits = no_bias(activation, table)
    assert logits.argmax(dim=-1).tolist() == [0, 2]
    assert logits[0, 0].item() == pytest.approx(torch.exp(torch.tensor(3.0)).item())
    with_bias = HistoricalAffineCEDecoder(4, 4, bias_mode="vocab")
    with torch.no_grad():
        with_bias.vocab_bias[1] = 100.0
    assert with_bias(activation[:1], table).argmax(dim=-1).item() == 1


def test_current_token_flattening_preserves_same_position_and_shift_is_detectable() -> None:
    table = torch.eye(6)
    labels = torch.tensor([[5, 0, 1, 2]], dtype=torch.int32)
    observations = table[labels].to(torch.float32)
    x, y = flatten_current_token_records(observations, labels, bos_token_id=5)
    model = HistoricalAffineCEDecoder(6, 6, bias_mode="none")
    predictions = model(x, table).argmax(dim=-1)
    assert y.tolist() == [0, 1, 2]
    assert predictions.tolist() == y.tolist()
    shifted = torch.tensor([1, 2, 0])
    assert predictions.eq(shifted).float().mean().item() < 1.0
    x_with_bos, y_with_bos = flatten_current_token_records(
        observations, labels, bos_token_id=5, include_bos=True
    )
    assert x_with_bos.shape[0] == 4
    assert y_with_bos.tolist() == [5, 0, 1, 2]


def test_evaluation_schedule_has_dense_early_points_and_fixed_late_grid() -> None:
    assert evaluation_schedule(3000) == (
        0,
        25,
        50,
        75,
        100,
        150,
        200,
        *range(300, 3001, 100),
    )
    assert evaluation_schedule(2) == (0, 2)


def test_public_bundle_checks_split_before_opening_truth_and_loads_padded_data(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "good")
    bundle = load_public_fit_bundle(manifest)
    assert bundle.layout == "padded_records"
    assert bundle.fit_record_count == 2
    assert bundle.validation_record_count == 1
    x, y = bundle.fit_tensors(include_bos=True, bos_token_id=5)
    assert x.shape == (8, 6)
    assert y.tolist() == [5, 0, 1, 2, 5, 2, 3, 4]

    overlap_root = tmp_path / "overlap"
    overlap_manifest = _manifest(
        overlap_root,
        fit_records=[{"record_id": "same", "full_token_count": 4, "post_bos_token_count": 3}],
        validation_records=[{"record_id": "same", "full_token_count": 4, "post_bos_token_count": 3}],
        fit_observations=torch.eye(6)[[5, 0, 1]].reshape(1, 3, 6),
        fit_truth=torch.tensor([[5, 0, 1]], dtype=torch.int32),
    )
    # The overlap is detected from metadata before the missing/invalid truth
    # resource would be touched, so this malformed public-label path cannot
    # turn a split error into a truth access.
    data = json.loads(overlap_manifest.read_text())
    data["resources"]["fit_truth"]["path"] = "missing_fit_truth.safetensors"
    overlap_manifest.write_text(json.dumps(data))
    with pytest.raises(HistoricalAffineCEError, match="overlap before public truth access"):
        load_public_fit_bundle(overlap_manifest)


def test_public_bundle_supports_packed_records_and_nested_post_bos_limit(tmp_path: Path) -> None:
    table = torch.eye(6)
    labels = torch.tensor([5, 0, 1, 5, 2, 3], dtype=torch.int32)
    observations = table[labels]
    validation_labels = torch.tensor([5, 1, 0, 3], dtype=torch.int32)
    validation_observations = table[validation_labels]
    manifest = _manifest(
        tmp_path,
        layout="packed_records",
        fit_records=[
            {"record_id": "fit-a", "full_token_count": 3, "post_bos_token_count": 2},
            {"record_id": "fit-b", "full_token_count": 3, "post_bos_token_count": 2},
        ],
        validation_records=[
            {"record_id": "validation-a", "full_token_count": 4, "post_bos_token_count": 3}
        ],
        fit_observations=observations,
        fit_truth=labels,
        validation_observations=validation_observations,
        validation_truth=validation_labels,
    )
    bundle = load_public_fit_bundle(manifest)
    assert bundle.fit_record_counts == (3, 3)
    x, y = bundle.fit_tensors(bos_token_id=5, include_bos=True, position_limit=2)
    assert x.shape[0] == 3  # BOS plus two post-BOS rows from the first record.
    assert y.tolist() == [5, 0, 1]
    x_post, y_post = bundle.fit_tensors(bos_token_id=5, include_bos=False, position_limit=2)
    assert x_post.shape[0] == 2
    assert y_post.tolist() == [0, 1]


def test_public_bundle_uses_right_padding_masks_and_preserves_record_groups(tmp_path: Path) -> None:
    table = torch.eye(6)
    fit_labels = torch.tensor([[5, 0, 1, 2], [5, 3, 4, 999]], dtype=torch.int32)
    validation_labels = torch.tensor([[5, 1, 0, 2], [5, 3, 4, 999]], dtype=torch.int32)
    fit_observations = table[fit_labels.clamp_max(5)]
    validation_observations = table[validation_labels.clamp_max(5)]
    fit_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.uint8)
    validation_mask = fit_mask.clone()
    manifest = _manifest(
        tmp_path,
        fit_records=[
            {"record_id": "fit-a", "full_token_count": 4, "post_bos_token_count": 3},
            {"record_id": "fit-b", "full_token_count": 3, "post_bos_token_count": 2},
        ],
        validation_records=[
            {
                "record_id": "validation-a",
                "full_token_count": 4,
                "post_bos_token_count": 3,
                "style": "alpaca",
            },
            {
                "record_id": "validation-b",
                "full_token_count": 3,
                "post_bos_token_count": 2,
                "style": "pile",
            },
        ],
        fit_observations=fit_observations,
        fit_truth=fit_labels,
        validation_observations=validation_observations,
        validation_truth=validation_labels,
        fit_valid_mask=fit_mask,
        validation_valid_mask=validation_mask,
        table=table,
    )

    # The real preparation artifact combines activations, token IDs, and the
    # right-padding mask.  Point all three manifest resources at one toy file
    # to exercise key-selective loading and active-row flattening.
    fit_combined = tmp_path / "fit_combined.safetensors"
    validation_combined = tmp_path / "validation_combined.safetensors"
    save_file(
        {"activations": fit_observations, "token_ids": fit_labels, "attention_mask": fit_mask},
        str(fit_combined),
    )
    save_file(
        {
            "activations": validation_observations,
            "token_ids": validation_labels,
            "attention_mask": validation_mask,
        },
        str(validation_combined),
    )
    data = json.loads(manifest.read_text())
    def combined_entry(path: Path, key: str, value: torch.Tensor) -> dict[str, object]:
        return {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "tensor_key": key,
        }
    data["resources"]["fit_observations"] = combined_entry(
        fit_combined, "activations", fit_observations
    )
    data["resources"]["fit_truth"] = combined_entry(fit_combined, "token_ids", fit_labels)
    data["resources"]["fit_valid_mask"] = combined_entry(
        fit_combined, "attention_mask", fit_mask
    )
    data["resources"]["validation_observations"] = combined_entry(
        validation_combined, "activations", validation_observations
    )
    data["resources"]["validation_truth"] = combined_entry(
        validation_combined, "token_ids", validation_labels
    )
    data["resources"]["validation_valid_mask"] = combined_entry(
        validation_combined, "attention_mask", validation_mask
    )
    manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    bundle = load_public_fit_bundle(manifest)
    x, y = bundle.fit_tensors(bos_token_id=5, include_bos=True)
    assert x.shape[0] == 7
    assert y.tolist() == [5, 0, 1, 2, 5, 3, 4]
    vx, vy = bundle.validation_tensors(bos_token_id=5, include_bos=False)
    assert vx.shape[0] == 5
    assert vy.tolist() == [1, 0, 2, 3, 4]
    assert bundle.validation_flat_groups() == ("alpaca", "alpaca", "alpaca", "pile", "pile")


def test_style_balanced_validation_metric_is_separate_from_row_weighted_accuracy() -> None:
    table = torch.eye(3)
    fit_labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    fit_x = table.index_select(0, fit_labels)
    validation_x = table[[0, 0, 0, 1]]
    validation_y = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    model, evidence = train_historical_affine_ce(
        HistoricalAffineCEDecoder(3, 3, bias_mode="none"),
        fit_x,
        fit_labels,
        table,
        config=HistoricalAffineCEConfig(steps=1, batch_size=2, learning_rate=1e-3),
        device=torch.device("cpu"),
        validation=(validation_x, validation_y),
        validation_groups=("alpaca", "alpaca", "alpaca", "pile"),
    )
    del model
    initial = evidence["learning_curve"][0]
    assert initial["validation_token_accuracy"] == pytest.approx(0.25)
    assert initial["validation_style_balanced_token_accuracy"] == pytest.approx(0.5)
    assert evidence["selection_metric"] == "validation_style_balanced_token_accuracy"


def test_fixed_training_probe_is_reproducible_and_fit_only() -> None:
    from token_reconstruction.historical_affine_ce import fixed_training_probe

    activations = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    labels = torch.arange(10, dtype=torch.long)
    first = fixed_training_probe(activations, labels, size=4, seed=17)
    second = fixed_training_probe(activations, labels, size=4, seed=17)
    assert torch.equal(first[2], second[2])
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[2].numel() == 4


def test_alignment_contract_rejects_next_token_supervision(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    data = json.loads(manifest.read_text())
    data["alignment"]["mode"] = "next_token"
    manifest.write_text(json.dumps(data))
    with pytest.raises(HistoricalAffineCEError, match="current-token"):
        load_public_fit_bundle(manifest)


def test_controlled_fit_records_validation_curve_and_round_trips_selected_state(tmp_path: Path) -> None:
    torch.manual_seed(9)
    table = normalized_embedding_table(torch.randn(7, 4))
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2], dtype=torch.long)
    activations = table.index_select(0, labels) + 0.02 * torch.randn(10, 4)
    model, evidence = train_historical_affine_ce(
        HistoricalAffineCEDecoder(4, 7, bias_mode="none"),
        activations[:7],
        labels[:7],
        table,
        config=HistoricalAffineCEConfig(
            steps=2,
            batch_size=3,
            learning_rate=1e-2,
            log_every=25,
            seed=13,
        ),
        device=torch.device("cpu"),
        validation=(activations[7:], labels[7:]),
    )
    assert evidence["selected_step"] in (0, 2)
    assert evidence["evaluation_schedule"] == [0, 2]
    assert [point["step"] for point in evidence["learning_curve"]] == [0, 2]
    selected = evidence.pop("selected_state_dict")
    assert set(selected) == set(model.state_dict())
    assert all(torch.isfinite(value).all().item() for value in model.parameters())
    selected_path = tmp_path / "selected.safetensors"
    save_historical_affine_ce(model, selected_path, state=selected)
    loaded = load_historical_affine_ce(
        selected_path,
        hidden_size=4,
        vocab_size=7,
        bias_mode="none",
        device=torch.device("cpu"),
    )
    assert tensor_sha256(loaded.W) == tensor_sha256(selected["W"])
    prediction = direct_prediction_tensor(loaded, activations[:2], table, device=torch.device("cpu"))
    assert prediction.shape == (2,)


def test_training_configuration_rejects_nonhistorical_scheduler() -> None:
    with pytest.raises(HistoricalAffineCEError, match="CosineAnnealingLR"):
        HistoricalAffineCEConfig(scheduler="linear").validate()


def test_adapter_rejects_non_binary_integer_attention_mask() -> None:
    labels = torch.tensor([[128000, 7, 128001]], dtype=torch.int32)
    mask = torch.tensor([[1, 2, 0]], dtype=torch.int32)
    with pytest.raises(AffineManifestAdapterError, match="exactly 0 or 1"):
        _check_right_padded_mask(mask, labels, label="toy mask")


def test_zero_gradient_clip_only_measures_finite_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    table = normalized_embedding_table(torch.randn(7, 4))
    labels = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    activations = table.index_select(0, labels)
    calls: list[tuple[float, bool]] = []
    original = torch.nn.utils.clip_grad_norm_

    def record_call(parameters, max_norm, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((float(max_norm), bool(kwargs.get("error_if_nonfinite", False))))
        return original(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_call)
    _, evidence = train_historical_affine_ce(
        HistoricalAffineCEDecoder(4, 7, bias_mode="none"),
        activations,
        labels,
        table,
        config=HistoricalAffineCEConfig(steps=1, batch_size=3),
        device=torch.device("cpu"),
        validation=(activations, labels),
    )
    assert calls == [(float("inf"), True)]
    assert evidence["gradient_norm_max"] > 0.0

