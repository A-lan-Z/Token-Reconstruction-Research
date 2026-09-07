from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from trr0009_train import (
    CHALLENGE_CAP,
    CHALLENGE_SEED,
    EXPECTED_POSITION_BUDGET,
    EXPECTED_SCHEDULE_DIGEST,
    TrainingConfig,
    _challenge_from_wrong_mask,
    _resource_paths,
    build_parser,
    main,
    _frequency_counts,
    _support_receipt,
    _validation_steps,
    _consider_checkpoint,
    resource_estimate,
)


def test_challenge_selection_is_deterministic_and_post_bos_only() -> None:
    wrong = torch.zeros(4, 8, dtype=torch.bool)
    wrong[0, 0] = True
    wrong[0, 1] = True
    wrong[1, 2] = True
    wrong[2, 3] = True
    wrong[3, 7] = True
    selected, receipt = _challenge_from_wrong_mask(wrong, cap=3, seed=CHALLENGE_SEED)
    selected_again, receipt_again = _challenge_from_wrong_mask(wrong, cap=3, seed=CHALLENGE_SEED)
    assert torch.equal(selected, selected_again)
    assert receipt == receipt_again
    assert not selected[:, 0].any()
    assert receipt["all_initially_wrong_rows"] == 4
    assert receipt["selected_rows"] == 3
    assert bool((selected & ~wrong).sum()) is False


def test_empty_selected_state_challenge_is_explicit() -> None:
    selected, receipt = _challenge_from_wrong_mask(torch.zeros(3, 5, dtype=torch.bool), cap=CHALLENGE_CAP, seed=CHALLENGE_SEED)
    assert int(selected.sum()) == 0
    assert receipt["empty_challenge"] is True
    assert receipt["initial_selected_accuracy"] is None


def test_frequency_support_uses_five_fresh_bins() -> None:
    counts = torch.zeros(128256, dtype=torch.long)
    counts[1] = 1
    counts[2] = 4
    counts[3] = 5
    counts[4] = 9
    counts[5] = 10
    counts[6] = 49
    counts[7] = 50
    ids, values, receipt = _support_receipt(counts)
    assert ids.tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert values.tolist() == [1, 4, 5, 9, 10, 49, 50]
    assert set(receipt["frequency_bins"]) == {"unseen_0", "seen_1_4", "seen_5_9", "seen_10_49", "seen_50_plus"}
    assert receipt["frequency_bins"]["seen_5_9"]["rows"] == 14
    assert receipt["frequency_bins"]["seen_10_49"]["rows"] == 59
    assert receipt["frequency_bins"]["seen_50_plus"]["rows"] == 50


def test_validation_steps_are_exactly_registered_interval() -> None:
    config = TrainingConfig(steps=3000, validation_every=100)
    steps = _validation_steps(config)
    assert steps[0] == 0
    assert steps[-1] == 3000
    assert steps == tuple(range(0, 3001, 100))


def test_resource_estimate_declares_b1_full_output_separately() -> None:
    estimate = resource_estimate()
    assert estimate["largest_activation_shape"] == [8, 192, 2048]
    assert estimate["largest_draw_shape"] == [EXPECTED_POSITION_BUDGET]
    assert estimate["b1_two_full_logits_peak_bytes_if_compared_simultaneously"] > 0
    assert estimate["qualification_full_output_fixture"].startswith("B=1")


def test_best_checkpoint_state_is_restored_when_final_is_worse() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    best_score, best_step, best_state = _consider_checkpoint(
        model, {"step": 0, "validation": {"style_balanced_token_accuracy": 0.2}}, float("-inf"), 0, None
    )
    with torch.no_grad():
        model.weight.fill_(2.0)
    best_score, best_step, best_state = _consider_checkpoint(
        model, {"step": 100, "validation": {"style_balanced_token_accuracy": 0.9}}, best_score, best_step, best_state
    )
    with torch.no_grad():
        model.weight.fill_(3.0)
    best_score, best_step, best_state = _consider_checkpoint(
        model, {"step": 3000, "validation": {"style_balanced_token_accuracy": 0.8}}, best_score, best_step, best_state
    )
    model.load_state_dict(best_state, strict=True)
    assert best_step == 100
    assert torch.equal(model.weight, torch.full_like(model.weight, 2.0))


def test_resource_paths_and_preflight_failure_receipt_are_bounded(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    output_root = repository_root / "experiments" / "TRR-0009" / "training" / "preflight_failure"
    parser = build_parser()
    args = parser.parse_args([
        "--repository-root", str(repository_root),
        "--output-root", str(output_root),
        "--device", "cpu",
        "--preflight-only",
    ])
    paths = _resource_paths(args)
    assert paths.repository_root == repository_root.resolve()
    assert paths.output_root == output_root.resolve()
    assert all(value.is_absolute() for value in (
        paths.fit_manifest, paths.validation_manifest, paths.embedding_path,
        paths.starting_state, paths.schedule_path, paths.output_root,
    ))

    status = main([
        "--repository-root", str(repository_root),
        "--output-root", str(output_root),
        "--device", "cpu",
        "--preflight-only",
    ])
    assert status == 2
    failure_path = output_root / "failure.json"
    assert failure_path.is_file()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["schema"] == "token-reconstruction.trr0009-failure.v1"
    assert failure["task_id"] == "TRR-0009"
    assert failure["error_type"] in {"TRR0009TrainError", "RuntimeError"}
