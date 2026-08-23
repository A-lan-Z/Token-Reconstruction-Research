from __future__ import annotations

import torch

import trr0002_r4_historical_target_bridge as bridge


def test_r4_matrix_keeps_historical_and_plain_a1_separate() -> None:
    policies = [
        {"policy_id": policy_id}
        for policy_id in (
            "a1a2_589f6e179eb4626877c2",
            "a1a2_43ea0bb737bc075531ca",
            "a1a2_13f73c306bf8946e9a28",
            "another",
        )
    ]
    plan = {"policies": policies}
    historical = bridge.proposer_entries(
        plan, "historical_alpaca_affine_a1"
    )
    checkpoint = bridge.proposer_entries(
        plan, "checkpoint_identity_a1"
    )
    assert historical == policies
    assert [item["policy_id"] for item in checkpoint] == list(
        bridge.CHECKPOINT_POLICY_IDS
    )


def test_predict_cli_has_no_truth_dataset_or_target_model_argument() -> None:
    args = bridge.build_parser().parse_args(
        [
            "predict",
            "--plan",
            "plan.json",
            "--input-root",
            "sanitized",
            "--model-path",
            "public-model",
            "--proposer",
            "checkpoint_identity_a1",
            "--output-directory",
            "predictions",
        ]
    )
    names = set(vars(args))
    assert args.command_name == "predict"
    assert not any("truth" in name for name in names)
    assert not any("dataset" in name for name in names)
    assert not any("target" in name for name in names)


def test_tensor_identity_hash_binds_dtype_shape_and_values() -> None:
    value = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    assert bridge.tensor_sha256(value) == bridge.tensor_sha256(value.clone())
    assert bridge.tensor_sha256(value) != bridge.tensor_sha256(value.to(torch.int32))
    changed = value.clone()
    changed[1, 1] = 5
    assert bridge.tensor_sha256(value) != bridge.tensor_sha256(changed)


def test_repeated_proposal_equality_allows_only_matching_nan() -> None:
    reference = torch.tensor([1.0, float("nan"), -2.0])
    assert bridge.tensors_equal_with_matching_nan(
        reference, reference.clone()
    )

    finite_change = reference.clone()
    finite_change[0] = 2.0
    assert not bridge.tensors_equal_with_matching_nan(
        reference, finite_change
    )

    moved_nan = torch.tensor([float("nan"), 1.0, -2.0])
    assert not bridge.tensors_equal_with_matching_nan(
        reference, moved_nan
    )
    assert not bridge.tensors_equal_with_matching_nan(
        reference, reference.to(torch.float64)
    )


def test_candidate_recall_excludes_bos_and_padding() -> None:
    truth = torch.tensor([[128000, 5, 6, 0], [128000, 7, 8, 9]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    candidates = torch.full((2, 4, 4), -1, dtype=torch.long)
    candidates[0, 1] = torch.tensor([5, 10, 11, 12])
    candidates[0, 2] = torch.tensor([10, 6, 11, 12])
    candidates[1, 1] = torch.tensor([10, 11, 7, 12])
    candidates[1, 2] = torch.tensor([10, 11, 12, 8])
    candidates[1, 3] = torch.tensor([10, 11, 12, 13])
    assert bridge.candidate_recall(candidates, truth, mask, 1) == {
        "k": 1,
        "hits": 1,
        "scored_tokens": 5,
        "recall": 0.2,
    }
    assert bridge.candidate_recall(candidates, truth, mask, 4) == {
        "k": 4,
        "hits": 4,
        "scored_tokens": 5,
        "recall": 0.8,
    }
