from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from token_reconstruction.trr0005_joint_decoder import BOS_TOKEN_ID, PublicJointData, build_position_schedule
from token_reconstruction.trr0007_positionwise import (
    CURRENT_METHOD_ID,
    RESIDUAL_MLP_METHOD_ID,
    build_current_positionwise,
    build_residual_mlp512,
    load_positionwise_model_state,
    save_positionwise_state,
    step_zero_equivalence,
)
import trr0007_train_positionwise as trainer


def _inputs(
    *,
    records: int = 2,
    positions: int = 6,
    hidden: int = 8,
    vocab: int = 17,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7007)
    activation = torch.randn(records, positions, hidden)
    truth = torch.randint(0, vocab, (records, positions), dtype=torch.long)
    truth[:, 0] = BOS_TOKEN_ID
    valid_mask = torch.ones(records, positions, dtype=torch.bool)
    embedding = F.normalize(torch.randn(vocab, hidden), dim=-1)
    return activation, truth, valid_mask, embedding


def test_neutral_current_and_residual_logits_are_exactly_equal() -> None:
    activation, _, valid_mask, embedding = _inputs()
    current = build_current_positionwise(
        hidden_size=8, vocabulary_size=17, context_width=4, seed=4005
    )
    extension = build_residual_mlp512(
        hidden_size=8, vocabulary_size=17, context_width=4, bottleneck_size=3, seed=4005
    )
    receipt = step_zero_equivalence(
        current, extension, activation, valid_mask, embedding, max_rows=16
    )
    assert receipt["projected_hidden_exact"] is True
    assert receipt["logits_exact"] is True
    assert receipt["max_logits_abs_delta"] == 0.0


@pytest.mark.parametrize("method", ["current", "extension"])
def test_positionwise_models_read_only_current_hidden(
    method: str,
) -> None:
    activation, _, valid_mask, embedding = _inputs(records=1, positions=7)
    if method == "current":
        model = build_current_positionwise(
            hidden_size=8, vocabulary_size=17, context_width=4, seed=4005
        )
        with torch.no_grad():
            model.output.weight.normal_(0.0, 0.05)
            model.output.bias.normal_(0.0, 0.05)
    else:
        model = build_residual_mlp512(
            hidden_size=8, vocabulary_size=17, context_width=4, bottleneck_size=3, seed=4005
        )
        with torch.no_grad():
            model.base.output.weight.normal_(0.0, 0.05)
            model.base.output.bias.normal_(0.0, 0.05)
            model.up.weight.normal_(0.0, 0.05)
            model.up.bias.normal_(0.0, 0.05)
    changed = activation.clone()
    changed[:, 1, :] += 13.0
    with torch.inference_mode():
        before = model(activation, valid_mask, embedding)
        after = model(changed, valid_mask, embedding)
    assert torch.equal(before[:, 2:], after[:, 2:])
    assert not torch.equal(before[:, 1], after[:, 1])


def test_residual_extension_changes_only_after_nonzero_output_path() -> None:
    activation, _, valid_mask, embedding = _inputs()
    extension = build_residual_mlp512(
        hidden_size=8, vocabulary_size=17, context_width=4, bottleneck_size=3, seed=4005
    )
    neutral = extension(activation, valid_mask, embedding)
    with torch.no_grad():
        extension.up.weight.normal_(0.0, 0.05)
        extension.up.bias.normal_(0.0, 0.05)
    changed = extension(activation, valid_mask, embedding)
    assert not torch.equal(neutral[:, 1:], changed[:, 1:])


@pytest.mark.parametrize(
    ("method_id", "builder"),
    [
        (
            CURRENT_METHOD_ID,
            lambda: build_current_positionwise(
                hidden_size=8, vocabulary_size=17, context_width=4, seed=4005
            ),
        ),
        (
            RESIDUAL_MLP_METHOD_ID,
            lambda: build_residual_mlp512(
                hidden_size=8, vocabulary_size=17, context_width=4, bottleneck_size=3, seed=4005
            ),
        ),
    ],
)
def test_positionwise_state_roundtrip(tmp_path: Path, method_id: str, builder) -> None:
    activation, _, valid_mask, embedding = _inputs()
    model = builder()
    path = tmp_path / f"{method_id}.safetensors"
    saved = save_positionwise_state(
        path,
        model,
        method_id=method_id,
        selected_step=0,
        initialization="neutral test initialization",
        distribution="test",
        bottleneck_size=3 if method_id == RESIDUAL_MLP_METHOD_ID else None,
    )
    loaded = load_positionwise_model_state(
        path,
        method_id=method_id,
        hidden_size=8,
        vocabulary_size=17,
        context_width=4,
        bottleneck_size=3,
    )
    assert saved["sha256"]
    assert list(loaded.state_dict()) == list(model.state_dict())
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name
    with torch.inference_mode():
        assert torch.equal(
            model(activation, valid_mask, embedding),
            loaded(activation, valid_mask, embedding),
        )


def test_positionwise_loader_rejects_wrong_method(tmp_path: Path) -> None:
    model = build_current_positionwise(
        hidden_size=8, vocabulary_size=17, context_width=4, seed=4005
    )
    path = tmp_path / "current.safetensors"
    save_positionwise_state(
        path,
        model,
        method_id=CURRENT_METHOD_ID,
        selected_step=0,
        initialization="neutral",
        distribution="test",
    )
    with pytest.raises(Exception, match="method"):
        load_positionwise_model_state(
            path,
            method_id=RESIDUAL_MLP_METHOD_ID,
            hidden_size=8,
            vocabulary_size=17,
            context_width=4,
            bottleneck_size=3,
        )


def test_resource_preflight_records_actual_batch_geometry(tmp_path: Path) -> None:
    parser = trainer.build_arg_parser()
    manifest = Path("experiments/TRR-0005/public_activation_v1/enriched_manifest.json")
    args = parser.parse_args(
        [
            "--improved-fit-manifest",
            str(manifest),
            "--improved-validation-manifest",
            str(manifest),
            "--output",
            str(tmp_path / "run"),
            "--preflight-only",
        ]
    )
    receipt = trainer._resource_preflight(
        args, output_path=tmp_path / "run" / "resource_preflight.json"
    )
    assert receipt["manifest_geometry"]["fit_observations"] == [1200, 192, 2048]
    assert receipt["materialized_batch_geometry"]["activation"] == [8, 192, 2048]
    assert receipt["materialized_batch_geometry"]["draws"] == [512]
    assert receipt["parameter_counts"]["residual_mlp_added"] == 2_099_712
    assert receipt["bytes"]["selected_logits_fp32"] == 512 * 128256 * 4
    assert receipt["bytes"]["conservative_peak_fp32"] >= trainer.MEASURED_TRR0005_FLOOR_BYTES
    assert receipt["safety"]["qualified_largest_representative_batch"] is False


def test_challenge_selection_is_seeded_and_never_includes_bos() -> None:
    wrong = torch.ones(3, 7, dtype=torch.bool)
    selected_a, receipt_a = trainer._select_challenge(wrong, cap=4, seed=7007)
    selected_b, receipt_b = trainer._select_challenge(wrong, cap=4, seed=7007)
    assert torch.equal(selected_a, selected_b)
    assert receipt_a["mask_sha256"] == receipt_b["mask_sha256"]
    assert not selected_a[:, 0].any().item()
    assert int(selected_a.sum()) == 4


def test_generic_train_step_updates_positionwise_model() -> None:
    activation, truth, valid_mask, embedding = _inputs(records=2, positions=6)
    data = PublicJointData(
        fit_observations=activation,
        fit_truth=truth,
        fit_valid_mask=valid_mask,
        fit_record_ids=("fit-0", "fit-1"),
        validation_observations=activation.clone(),
        validation_truth=truth.clone(),
        validation_valid_mask=valid_mask.clone(),
        validation_record_ids=("val-0", "val-1"),
        validation_groups=("a", "b"),
        embedding_table=embedding,
        metadata={},
    )
    schedule = build_position_schedule(
        valid_mask, steps=1, record_batch_size=2, position_budget=4, seed=4005
    )
    model = build_residual_mlp512(
        hidden_size=8, vocabulary_size=17, context_width=4, bottleneck_size=3, seed=4005
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    result = trainer._train_step(
        model,
        data,
        schedule,
        0,
        device=torch.device("cpu"),
        runtime_embedding=embedding,
        optimizer=optimizer,
        gradient_clip_norm=1.0,
    )
    assert result["activation_shape"] == [2, 6, 8]
    assert result["draw_shape"] == [4]
    assert result["token_rows"] == 4
    assert all(torch.isfinite(value).all() for value in model.parameters())
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())
