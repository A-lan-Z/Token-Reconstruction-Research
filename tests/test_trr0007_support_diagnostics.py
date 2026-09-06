from __future__ import annotations

from collections import Counter

import torch

from scripts import trr0007_support_diagnostics as support


def _labels(lengths: list[int], values: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = torch.full((len(lengths), support.EXPECTED_SEQUENCE_LENGTH), support.PAD_TOKEN_ID, dtype=torch.int32)
    mask = torch.zeros((len(lengths), support.EXPECTED_SEQUENCE_LENGTH), dtype=torch.uint8)
    token_ids[:, 0] = support.BOS_TOKEN_ID
    mask[:, 0] = 1
    for index, (length, row_values) in enumerate(zip(lengths, values)):
        assert length == len(row_values)
        token_ids[index, 1 : length + 1] = torch.tensor(row_values, dtype=torch.int32)
        mask[index, : length + 1] = 1
    return token_ids, mask


def test_joint_support_materializes_zero_cells_and_keeps_exact_counts() -> None:
    token_ids, mask = _labels([2, 127], [[5, 5], [6] * 127])
    rows = [
        {"record_id": "natural-0", "domain": "alpaca_instruction", "slot": 0, "post_bos_token_count": 2},
        {"record_id": "controlled-1", "domain": "controlled_pile_context", "slot": 1, "synthetic": True, "post_bos_token_count": 127},
    ]
    result = support._support_for_rows(
        token_ids,
        mask,
        rows,
        frequency_counts=Counter({5: 2, 6: 127}),
        label="synthetic",
        expected_records=2,
        expected_width=support.EXPECTED_SEQUENCE_LENGTH,
    )

    assert result["geometry"]["post_bos_positions"] == 129
    assert result["coverage"]["evaluation_range_positions"] == 129
    cells = result["joint_style_position_frequency"]
    assert set(cells) == {"natural_alpaca", "controlled_pile"}
    for style in cells:
        assert set(cells[style]) == {name for name, _lower, _upper in support.POSITION_BINS}
        for position in cells[style].values():
            assert set(position) == {name for name, _lower, _upper in support.FREQUENCY_BINS}
    assert cells["natural_alpaca"]["1-15"]["seen_2_4"]["examples"] == 2
    assert cells["controlled_pile"]["80-127"]["seen_65_plus"]["examples"] == 48
    assert cells["natural_alpaca"]["80-127"]["seen_65_plus"]["examples"] == 0
    zero = cells["natural_alpaca"]["80-127"]["seen_65_plus"]
    assert zero["correct"] is None and zero["errors"] is None
    assert zero["token_accuracy"] is None and zero["correctness_status"] == "not_computed"
    observed = cells["controlled_pile"]["80-127"]["seen_65_plus"]
    assert observed["correct"] is None and observed["errors"] is None
    assert observed["correctness_status"] == "not_computed"


def test_broader_identity_pool_preserves_existing_ids_and_adds_public_candidates() -> None:
    refs = {
        "path": "/tmp/frequency-references.json",
        "bytes": 10,
        "sha256": "reference-hash",
        "schema": "token-reconstruction.trr0005-frequency-references.v1",
        "enriched": {10: 8, 11: 7},
        "original": {},
    }
    candidate_pool = {
        "path": "/tmp/candidate-frequency.json",
        "bytes": 11,
        "sha256": "candidate-hash",
        "schema": support.CANDIDATE_FREQUENCY_SCHEMA,
        "frequencies": {10: 8, 11: 7, 12: 6, 13: 5},
        "special_token_ids": [support.BOS_TOKEN_ID, support.PAD_TOKEN_ID],
    }
    rows = [
        {"synthetic": True, "replacement_token_ids": [10]},
        {"synthetic": True, "replacement_token_ids": [11]},
    ]
    result = support._broader_identity_pool(
        rows, refs, candidate_pool, target_count=4, baseline_count=2
    )

    assert result["status"] == "computed"
    assert result["candidate_count"] == 2
    assert result["baseline_identity_count"] == 2
    assert result["additional_identity_count"] == 2
    assert result["selected_token_ids"] == [10, 11, 12, 13]
    assert result["baseline_ids_preserved"] is True
    assert result["private_truth_accessed"] is False
    assert result["additional_ids_currently_unseen"] is True


def test_replacement_support_reports_raw_and_one_based_coordinates() -> None:
    rows = [
        {
            "record_id": "controlled-a",
            "synthetic": True,
            "domain": "controlled_pile_context",
            "target_post_bos_token_count": 150,
            "replacement_positions": [0, 126, 127],
            "replacement_token_ids": [10, 11, 12],
        },
        {
            "record_id": "controlled-b",
            "synthetic": True,
            "domain": "controlled_finance_context",
            "target_post_bos_token_count": 150,
            "replacement_positions": [14, 79, 149],
            "replacement_token_ids": [13, 14, 15],
        },
    ]
    result = support._replacement_support(rows)

    assert result["replacement_occurrences"] == 6
    assert result["replacement_offsets_zero_based"]["min"] == 0
    assert result["replacement_offsets_zero_based"]["max"] == 149
    assert result["replacement_offsets_zero_based"]["evaluation_range_occurrences_offsets_0_126"] == 4
    assert result["replacement_positions_after_bos_one_based"]["min"] == 1
    assert result["replacement_positions_after_bos_one_based"]["max"] == 150
    assert result["replacement_positions_after_bos_one_based"]["evaluation_range_occurrences_1_127"] == 4
    assert result["replacement_positions_after_bos_one_based"]["counts_by_position_bin"]["1-15"] == 2


def test_recipe_preserves_fixed_record_and_length_stratum_budget() -> None:
    lengths = [40] * 237 + [64] * 372 + [96] * 248 + [128] * 343
    rows = [
        {"record_id": f"row-{index}", "slot": index, "post_bos_token_count": length}
        for index, length in enumerate(lengths)
    ]
    recipe = support._recipe(rows, seed=7007)

    assert recipe["baseline_binding"]["fit_record_count"] == 1200
    assert recipe["baseline_binding"]["post_bos_positions"] == 124371
    assert recipe["baseline_binding"]["draw_schedule"]["draws"] == 1_536_000
    assert len(recipe["selected_controlled_slot_indices"]) == 120
    assert len(set(recipe["selected_controlled_slot_indices"])) == 120
    assert [item["selected_controlled_slots"] for item in recipe["controlled_component"]["target_length_strata"]] == [0, 0, 0, 120]
    assert recipe["slot_selection"]["selected_target_length_min"] == 128
    quotas = [item["per_record_quota"] for item in recipe["controlled_component"]["replacement_offset_policy"]["position_bins"]]
    assert quotas == [3, 6, 9, 12]
    repeat = support._recipe(rows, seed=7007)
    assert recipe["selected_controlled_slot_indices"] == repeat["selected_controlled_slot_indices"]
    assert recipe["selected_controlled_slot_indices_sha256"] == repeat["selected_controlled_slot_indices_sha256"]


def test_recorded_replacement_verification_binds_offset_to_captured_label() -> None:
    token_ids, mask = _labels([2], [[17, 18]])
    rows = [{
        "record_id": "controlled-0",
        "slot": 0,
        "synthetic": True,
        "replacement_positions": [0, 1],
        "replacement_token_ids": [17, 999],
    }]
    result = support._verify_recorded_replacements(token_ids, mask, rows)

    assert result["status"] == "FAIL"  # incomplete plan and one captured mismatch
    assert result["checked_controlled_records"] == 1
    assert result["checked_replacement_occurrences"] == 2
    assert result["mismatch_count"] == 1
    mismatch = result["mismatches"][0]
    assert mismatch["offset_zero_based_after_bos"] == 1
    assert mismatch["observed_token_id"] == 18


def test_improved_position_plan_covers_each_evaluation_bin() -> None:
    source = (support.BOS_TOKEN_ID,) + tuple(range(1, 151))
    offsets = support._planned_replacement_positions(
        source,
        target_post_bos_token_count=150,
        record_key="controlled-record",
        seed=7007,
    )
    positions = [offset + 1 for offset in offsets]
    counts = {
        name: sum(lower <= position <= upper for position in positions)
        for name, lower, upper, _quota in support.RECIPE_POSITION_PLAN
    }

    assert len(offsets) == 30
    assert len(set(offsets)) == 30
    assert min(positions) >= 1 and max(positions) <= 127
    assert counts == {"1-15": 3, "16-39": 6, "40-79": 9, "80-127": 12}
    assert offsets == support._planned_replacement_positions(
        source,
        target_post_bos_token_count=150,
        record_key="controlled-record",
        seed=7007,
    )


def test_exclusion_manifest_binds_record_and_sequence_identity() -> None:
    token_ids, mask = _labels([2], [[17, 18]])
    rows = [{
        "record_id": "fit-0",
        "source_record_id": "source-0",
        "rendered_sha256": "rendered-hash",
        "slot": 0,
    }]
    manifest = support._exclusion_manifest(token_ids, mask, rows, None, None, None)
    entry = manifest["fit_bank"]["records"][0]

    assert entry["record_id"] == "fit-0"
    assert entry["source_record_id"] == "source-0"
    assert entry["full_token_count"] == 3
    assert entry["post_bos_token_count"] == 2
    assert entry["sequence_sha256"] == support._sequence_digest([support.BOS_TOKEN_ID, 17, 18])
    assert manifest["truth_separation"]["private_truth_accessed"] is False
