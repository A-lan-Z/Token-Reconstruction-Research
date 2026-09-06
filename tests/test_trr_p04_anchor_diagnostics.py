from __future__ import annotations

from scripts.trr_p04 import score_predictions as score


def test_native_anchor_subset_reports_paired_wins_and_d_recovery() -> None:
    anchor_ids = [f"anchor-{index:02d}" for index in range(12)]
    panel = {
        record_id: {"record_id": record_id, "anchor": True, "length_stratum": 32}
        for record_id in anchor_ids
    }
    truths_by_condition = {
        condition: {record_id: [0] * 32 for record_id in anchor_ids}
        for condition in score.DEFAULT_CONDITIONS
    }
    # One native-fixed affine error (anchor-00), and one affine-fixed native
    # error (anchor-03), make both directions and D recovery observable.
    for condition in score.DEFAULT_CONDITIONS:
        truths_by_condition[condition]["anchor-03"] = [1] * 32

    native = {record_id: [0] * 32 for record_id in anchor_ids}
    affine = {record_id: [0] * 32 for record_id in anchor_ids}
    affine["anchor-00"] = [1] * 32
    affine["anchor-03"] = [1] * 32
    student_d = {record_id: [0] * 32 for record_id in anchor_ids}
    student_d["anchor-03"] = [1] * 32

    groups = {}
    expected = []
    for condition in score.DEFAULT_CONDITIONS:
        native_group = ("native_a1_a2", None, condition, True)
        groups[native_group] = native
        expected.append(native_group)
        for seed in score.DEFAULT_SEEDS:
            for method, predictions in (("affine_same_data", affine), ("student_d", student_d)):
                group = (method, seed, condition, False)
                groups[group] = predictions
                expected.append(group)

    first = score._native_anchor_subset_diagnostics(groups, truths_by_condition, panel, expected)
    second = score._native_anchor_subset_diagnostics(groups, truths_by_condition, panel, expected)

    assert first == second
    assert len(first) == len(score.DEFAULT_CONDITIONS) * len(score.DEFAULT_SEEDS)
    for row in first:
        assert row["anchor_records"] == 12
        assert row["scored_tokens"] == 384
        assert row["native_fixed_affine_error_tokens"] == 32
        assert row["student_d_recovers_native_fixed_affine_error_tokens"] == 32
        assert row["student_d_recovery_rate_on_native_fixed_affine_tokens"] == 1.0
        assert row["native_fixed_affine_error_records"] == 1
        assert row["student_d_recovers_native_fixed_affine_error_records"] == 1
        assert row["student_d_recovery_rate_on_native_fixed_affine_records"] == 1.0

        native_vs_affine = row["native_vs_affine"]
        assert native_vs_affine["left_correct_tokens"] == 352
        assert native_vs_affine["right_correct_tokens"] == 352
        assert native_vs_affine["token_gains"] == 32
        assert native_vs_affine["token_losses"] == 32
        assert native_vs_affine["token_ties"] == 320
        assert native_vs_affine["exact_record_gain"] == 1
        assert native_vs_affine["exact_record_loss"] == 1
        assert native_vs_affine["exact_record_ties"] == 10
        assert native_vs_affine["token_accuracy_delta"] == 0.0

        native_vs_d = row["native_vs_d"]
        assert native_vs_d["left_correct_tokens"] == 352
        assert native_vs_d["right_correct_tokens"] == 384
        assert native_vs_d["token_gains"] == 0
        assert native_vs_d["token_losses"] == 32
        assert native_vs_d["token_ties"] == 352
        assert native_vs_d["exact_record_gain"] == 0
        assert native_vs_d["exact_record_loss"] == 1
        assert native_vs_d["exact_record_ties"] == 11
        assert native_vs_d["token_accuracy_delta"] == -32 / 384
