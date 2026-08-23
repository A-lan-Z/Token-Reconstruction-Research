from __future__ import annotations

import json
from pathlib import Path

import torch

import trr0002_r2_finance_target_shortlist as target


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "experiments/TRR-0002/configuration-search/causal-selection/table.json"


def test_shortlist_is_unique_and_bound_to_frozen_public_table() -> None:
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    indexed = target.table_index(table)
    ids = [entry["policy_id"] for entry in target.SHORTLIST]
    labels = [entry["label"] for entry in target.SHORTLIST]
    assert len(ids) == 12
    assert len(set(ids)) == len(ids)
    assert len(set(labels)) == len(labels)
    assert all(policy_id in indexed for policy_id in ids)
    assert [indexed[policy_id][0] for policy_id in ids[3:8]] == [1, 2, 3, 4, 5]
    assert target.r2.sha256_file(TABLE) == target.TABLE_SHA256


def test_predict_cli_has_no_truth_or_dataset_argument() -> None:
    args = target.build_parser().parse_args(
        [
            "predict",
            "--historical-root",
            "/historical",
            "--plan",
            "plan.json",
            "--prediction-artifact",
            "predictions.safetensors",
            "--evidence",
            "evidence.json",
            "--freeze-receipt",
            "receipt.json",
        ]
    )
    argument_names = set(vars(args))
    assert args.command_name == "predict"
    assert not any("truth" in name for name in argument_names)
    assert not any("dataset" in name for name in argument_names)


def test_expected_tensor_registry_is_exact() -> None:
    entries = [
        {"policy_id": "first"},
        {"policy_id": "second"},
    ]
    keys = target.expected_tensor_keys(entries)
    assert len(keys) == 12
    assert "common.candidates_top512" in keys
    assert "first.predictions" in keys
    assert "first.selected_signal" in keys
    assert "second.routes" in keys


def test_candidate_recall_excludes_bos_and_padding() -> None:
    truth = torch.tensor([[128000, 5, 6, 0], [128000, 7, 8, 9]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    candidates = torch.full((2, 4, 4), -1, dtype=torch.long)
    candidates[0, 1] = torch.tensor([5, 10, 11, 12])
    candidates[0, 2] = torch.tensor([10, 6, 11, 12])
    candidates[1, 1] = torch.tensor([10, 11, 7, 12])
    candidates[1, 2] = torch.tensor([10, 11, 12, 8])
    candidates[1, 3] = torch.tensor([10, 11, 12, 13])
    at_one = target.candidate_recall(candidates, truth, mask, 1)
    at_four = target.candidate_recall(candidates, truth, mask, 4)
    assert at_one == {"k": 1, "hits": 1, "scored_tokens": 5, "recall": 0.2}
    assert at_four == {"k": 4, "hits": 4, "scored_tokens": 5, "recall": 0.8}


def test_owner_r2_request_preserves_target_surrogate_direction() -> None:
    request = (
        ROOT / "coordination/requests/TRR-0002-OWNER-REVISION-R2.md"
    ).read_text(encoding="utf-8")
    assert "Finance-Instruct-fine-tuned" in request
    assert "untouched public" in request
    assert "retrospective stress evidence" in request
    assert "Does this make sense?" in request
