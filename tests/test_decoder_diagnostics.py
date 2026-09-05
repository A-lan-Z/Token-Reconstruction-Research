from __future__ import annotations

import json

import pytest
import torch

from token_reconstruction.decoder_diagnostics import (
    DEFAULT_VARIANTS,
    DiagnosticVariant,
    diagnose_model,
    expected_scale_invariance,
    flatten_public_labels,
    flatten_public_records,
    token_frequency,
)
from token_reconstruction.inverse import ResidualAffineInverse
from token_reconstruction.standalone_decoder import (
    TiedAffineTokenDecoder,
    normalized_embedding_table,
)


BOS = 128000


def _public_rows() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    table = normalized_embedding_table(torch.eye(7))
    validation_observations = table[[0, 1, 2, 3, 4, 5, 6, 0]].reshape(2, 4, 7)
    validation_truth = torch.tensor(
        [[BOS, 1, 2, 3], [BOS, 4, 5, 6]], dtype=torch.int32
    )
    fit_truth = torch.tensor(
        [[BOS, 1, 1, 2], [BOS, 2, 3, 3], [BOS, 4, 4, 4]], dtype=torch.int32
    )
    return validation_observations, validation_truth, fit_truth


def test_public_flattening_excludes_bos_and_preserves_record_shape() -> None:
    observations, validation_truth, fit_truth = _public_rows()
    flat_x, flat_y, shape = flatten_public_records(observations, validation_truth)
    assert tuple(flat_x.shape) == (6, 7)
    assert flat_y.tolist() == [1, 2, 3, 4, 5, 6]
    assert shape == (2, 3)
    assert flatten_public_labels(fit_truth).tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 4]


def test_frequency_counts_are_public_fit_only() -> None:
    counts = token_frequency(torch.tensor([1, 1, 2, 4, 4, 4]), vocab_size=7)
    assert counts.tolist() == [0, 2, 1, 0, 3, 0, 0]


def test_posthoc_bias_scale_and_normalization_diagnostics_have_public_metrics() -> None:
    observations, validation_truth, fit_truth = _public_rows()
    validation_x, validation_y, record_shape = flatten_public_records(observations, validation_truth)
    fit_y = flatten_public_labels(fit_truth)
    table = normalized_embedding_table(torch.eye(7))
    model = TiedAffineTokenDecoder(7, 7, logit_scale=16.0)
    with torch.no_grad():
        model.classifier_bias[6] = 0.25
    result = diagnose_model(
        model,
        method_id=model.method_id,
        validation_observations=validation_x,
        validation_labels=validation_y,
        fit_labels=fit_y,
        embedding_table=table,
        record_shape=record_shape,
        batch_size=4,
    )
    assert result["method_id"] == "tied_affine_token_ce"
    assert result["vocabulary_coverage"]["fit_unique_tokens"] == 4
    assert result["bias_diagnostics"]["present"] is True
    assert result["bias_diagnostics"]["by_frequency_bin"]["rare_2_4"]["token_count"] == 4
    assert set(result["variants"]) == {variant.variant_id for variant in DEFAULT_VARIANTS}
    pairwise = result["pairwise_variant_comparisons"]
    scale_pair = pairwise["no_bias_scale_1__vs__vocab_bias_disabled"]
    assert scale_pair["prediction_changed_examples"] == 0
    scale_control = expected_scale_invariance(
        result["variants"]["vocab_bias_disabled"],
        result["variants"]["no_bias_scale_1"],
    )
    assert scale_control["expected_argmax_invariant"] is True
    assert result["variants"]["original"]["metrics"]["examples"] == 6
    assert result["variants"]["original"]["metrics"]["records"] == 2
    assert "frequency_bin_metrics" in result["variants"]["vocab_bias_disabled"]["metrics"]


def test_angular_control_reports_absent_vocab_bias() -> None:
    observations, validation_truth, fit_truth = _public_rows()
    validation_x, validation_y, record_shape = flatten_public_records(observations, validation_truth)
    fit_y = flatten_public_labels(fit_truth)
    table = normalized_embedding_table(torch.eye(7))
    model = ResidualAffineInverse(7)
    result = diagnose_model(
        model,
        method_id="angular_inverse_control",
        validation_observations=validation_x,
        validation_labels=validation_y,
        fit_labels=fit_y,
        embedding_table=table,
        record_shape=record_shape,
        batch_size=2,
        variants=(DiagnosticVariant("original"), DiagnosticVariant("no_bias", bias_mode="zero")),
    )
    assert result["bias_diagnostics"] == {
        "present": False,
        "reason": "angular control has no vocabulary bias",
    }
    assert result["pairwise_variant_comparisons"]["no_bias__vs__original"]["prediction_changed_examples"] == 0


def test_bos_and_geometry_validation_fail_closed() -> None:
    observations, validation_truth, _fit_truth = _public_rows()
    bad_truth = validation_truth.clone()
    bad_truth[0, 0] = 0
    with pytest.raises(ValueError, match="BOS"):
        flatten_public_records(observations, bad_truth)
    with pytest.raises(ValueError, match="non-empty"):
        token_frequency(torch.empty(0, dtype=torch.long), vocab_size=7)


def test_runner_requires_prechecked_disjoint_public_record_manifests(tmp_path) -> None:
    import sys

    sys.path.insert(0, "scripts")
    import trr0004_track_b_bias_diagnostic as runner

    fit_path = tmp_path / "fit_records.json"
    validation_path = tmp_path / "validation_records.json"
    fit_path.write_text(json.dumps({"records": [{"record_id": "fit-0"}]}))
    validation_path.write_text(
        json.dumps(
            {
                "disjointness_checked_before_label_access": True,
                "overlap_counts": {"panel": 0},
                "records": [{"record_id": "validation-0"}],
            }
        )
    )
    evidence = runner._check_public_split(fit_path, validation_path)
    assert evidence["fit_validation_overlap_count"] == 0
    validation_path.write_text(
        json.dumps(
            {
                "disjointness_checked_before_label_access": False,
                "records": [{"record_id": "validation-0"}],
            }
        )
    )
    with pytest.raises(runner.DiagnosticRunnerError, match="before label access"):
        runner._check_public_split(fit_path, validation_path)
