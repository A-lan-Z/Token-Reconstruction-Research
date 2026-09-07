"""Synthetic tests for the P09 CPU input compiler.

These tests use only synthetic rows and plan metadata.  They never load a
public dataset, model, activation bank, or evaluator truth.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from scripts.trr_p09 import prepare_stage1_inputs as prep
from scripts.trr_p09.prepare_stage1_inputs import (
    BIN_RANGES,
    DIAGNOSTIC_QUOTAS,
    STRATA,
    TARGET_ROWS,
    InputRow,
    PreparationErrorLocal,
    exact_addition_quotas,
    diagnostic_indices,
    per_bank_diagnostic_indices,
    h128_digest,
    make_controlled_rows,
    select_candidates,
    validate_plan_object,
    replacement_cycle_audit,
    _public_record_sha256,
    _scan_declared_hashes,
    build_b0_template_buckets,
    controlled_template_assignment_audit,
)


def _plan() -> dict:
    import json
    return json.loads(Path("experiments/TRR-P09/planning/stage1-public-bank-plan.json").read_text())


def test_signed_recipe_has_exact_five_strata_and_bin_totals() -> None:
    validate_plan_object(_plan())
    assert len(STRATA) == 5
    assert sum(sum(item[4]) for item in STRATA) == 10800
    assert sum(DIAGNOSTIC_QUOTAS.values()) == 64


def test_exact_addition_quotas_are_nine_times_b0() -> None:
    rows = []
    for name, domain, controlled, targets, _additions in STRATA:
        per_bin = [int(value // 10) for value in targets]
        for index, count in enumerate(per_bin):
            rows.extend({
                "dataset_key": domain,
                "synthetic": controlled,
                "target_post_bos_token_count": BIN_RANGES[index][0],
            } for _ in range(count))
    values = exact_addition_quotas(rows)
    assert len(rows) == 1200
    assert sum(sum(item.values()) for item in values.values()) == 10800
    for name, *_ in STRATA:
        assert sum(values[name].values()) == sum(next(item[4] for item in STRATA if item[0] == name))
        assert all(value == 9 * values[name][length] // 9 for length, value in values[name].items())


def test_h128_digest_is_stable_for_int32_prefix_and_none_when_short() -> None:
    values = tuple(range(128))
    assert h128_digest(values) == h128_digest(list(values))
    assert h128_digest(values[:-1]) is None


def test_controlled_replacements_are_30_unique_nonstructural_offsets() -> None:
    candidate = type("Candidate", (), {
        "record_id": "synthetic-pile-row",
        "dataset_key": "pile",
        "dataset_id": "pile",
        "split": "train",
        "revision": "rev",
        "row_index": 5,
        "rendered_sha256": "a" * 64,
        "token_ids": tuple([128000] + list(range(1, 200))),
        "full_token_count": 200,
    })()
    rows, cursor = make_controlled_rows([(candidate, 191)], "pile_controlled", list(range(1000, 4600)), 0)
    row = rows[0]
    assert len(row.replacement_positions) == 30
    assert len(set(row.replacement_positions)) == 30
    assert all(0 <= value < 191 for value in row.replacement_positions)
    assert row.token_ids[0] == 128000
    assert cursor == 30


def test_diagnostic_selection_is_deterministic_and_stratified() -> None:
    rows = []
    for name, quota in DIAGNOSTIC_QUOTAS.items():
        for index in range(100):
            rows.append(InputRow(
                record_id=f"{name}-{index}", source_record_id=f"{name}-{index}",
                dataset_key=name.split("_")[0], stratum=name, source_row_index=index,
                rendered_sha256=f"{index:064x}", source_full_token_count=65,
                target_post_bos_token_count=64, token_ids=tuple([128000] + list(range(1, 65))),
            ))
    first = diagnostic_indices(rows)
    second = diagnostic_indices(rows)
    assert first == second
    assert len(first["indices"]) == 64
    assert first["stratum_quotas"] == DIAGNOSTIC_QUOTAS


def test_input_row_sidecar_contains_capture_global_offset() -> None:
    row = InputRow(
        record_id="r", source_record_id="s", dataset_key="pile", stratum="pile_natural",
        source_row_index=4, rendered_sha256="a" * 64, source_full_token_count=65,
        target_post_bos_token_count=64, token_ids=tuple([128000] + list(range(1, 65))),
    )
    value = row.sidecar(1200)
    assert value["global_row"] == 1200
    assert value["target_full_token_count"] == 65


def test_plan_rejects_changed_exact_public_boundary() -> None:
    value = _plan()
    value["access_boundary"]["truth_opened"] = True
    with pytest.raises(PreparationErrorLocal, match="access boundary"):
        validate_plan_object(value)


def test_selection_allocates_descending_target_lengths_before_source_order() -> None:
    def candidate(index: int, length: int):
        return type("Candidate", (), {
            "record_id": f"candidate-{index}", "dataset_key": "pile", "dataset_id": "pile",
            "split": "train", "revision": "rev", "row_index": index,
            "rendered_sha256": f"{index + 1:064x}",
            "token_ids": tuple([128000] + list(range(1, length + 1))),
            "full_token_count": length + 1,
        })()
    candidates = [candidate(0, 191), candidate(1, 100), candidate(2, 80)]
    exclusions = {"record_ids": set(), "rendered_sha256": set(), "public_record_sha256": set(), "h128_sha256": set(), "source_indices": {}}
    selected = select_candidates(candidates, stratum="pile_natural", exact_quota={100: 1, 190: 1}, exclusions=exclusions, used_ids=set(), used_rendered=set(), used_h128=set())
    assert [target for _, target in selected] == [190, 100]


def test_hash_namespaces_and_source_indices_remain_separate() -> None:
    ids, rendered, public, h128 = set(), set(), set(), set()
    source_indices = {}
    _scan_declared_hashes({"pile": {"source_index": 7, "public_record_sha256": {"values": ["a" * 64]}, "final_sequence_sha256": {"values": ["b" * 64]}}, "rendered_sha256": "c" * 64}, ids=ids, rendered=rendered, public_record=public, h128=h128, source_indices=source_indices, allowed={"public_record_sha256", "final_sequence_sha256", "rendered_sha256", "source_indices"})
    assert rendered == {"c" * 64}
    assert public == {"a" * 64}
    assert h128 == {"b" * 64}
    assert source_indices == {"pile": {7}}


def test_controlled_cursor_and_sidecar_bind_replacement_lists() -> None:
    candidate = type("Candidate", (), {
        "record_id": "synthetic-row", "dataset_key": "pile", "dataset_id": "pile",
        "split": "train", "revision": "rev", "row_index": 5,
        "rendered_sha256": "a" * 64, "token_ids": tuple([128000] + list(range(1, 210))),
        "full_token_count": 210,
    })()
    candidate2 = type("Candidate", (), {
        "record_id": "synthetic-row-2", "dataset_key": "pile", "dataset_id": "pile",
        "split": "train", "revision": "rev", "row_index": 6,
        "rendered_sha256": "b" * 64, "token_ids": tuple([128000] + list(range(1, 210))),
        "full_token_count": 210,
    })()
    rows, cursor = make_controlled_rows([(candidate, 191), (candidate2, 190)], "pile_controlled", list(range(1000, 4600)), 0)
    assert cursor == 60
    sidecar = rows[0].sidecar(1200)
    assert sidecar["replacement_count"] == 30
    assert len(sidecar["replacement_token_ids"]) == 30

def test_published_b0_cycle_is_bound_separately_from_signed_b1_recipe() -> None:
    observed_b0 = list(range(3600))
    signed_recipe = list(range(10000, 13600))
    audit = replacement_cycle_audit(observed_b0, signed_recipe)
    assert audit["b0_identity_cycle_matches_signed_recipe"] is False
    assert audit["b0_replacement_occurrences"] == 3600
    assert audit["signed_b1_cycle_count"] == 9
    assert audit["full_replacement_occurrence_count"] == 36000
    assert audit["full_ten_cycle_composition"] == "published_b0_actual_3600_plus_signed_recipe_3600_x9"
    assert audit["b0_replacement_token_digest"] != audit["signed_b1_replacement_token_digest"]


def test_synthetic_sidecar_preserves_positions_and_token_ids() -> None:
    row = InputRow(
        record_id="synthetic-b0", source_record_id="parent", dataset_key="pile",
        stratum="pile_controlled", source_row_index=5, rendered_sha256="a" * 64,
        source_full_token_count=65, target_post_bos_token_count=64,
        token_ids=tuple([128000] + list(range(1, 65))), synthetic=True,
        parent_h128_sha256="b" * 64, sequence_h128_sha256="c" * 64,
        replacement_positions=(3, 8), replacement_token_ids=(123, 456),
    )
    sidecar = row.sidecar(1200)
    assert sidecar["capture_local_row"] == 0
    assert sidecar["replacement_positions_after_bos"] == [3, 8]
    assert sidecar["replacement_positions_one_based"] == [4, 9]
    assert sidecar["replacement_token_ids"] == [123, 456]

def test_public_record_digest_uses_source_canonicalization_not_rendering() -> None:
    pile = _public_record_sha256("pile", {"text": "  raw text  "}, "f" * 64)
    assert pile != "f" * 64
    finance = _public_record_sha256(
        "finance",
        {"system": " sys ", "user": " user ", "assistant": " answer "},
        "e" * 64,
    )
    assert finance != "e" * 64
    assert finance == _public_record_sha256(
        "finance",
        {"system": "sys", "user": "user", "assistant": "answer"},
        "different-rendered-digest",
    )


def test_selection_rejects_public_record_hash_without_rejecting_rendered_namespace() -> None:
    def candidate(index: int, rendered: str, public: str):
        return type("Candidate", (), {
            "record_id": f"candidate-{index}", "dataset_key": "pile", "dataset_id": "pile",
            "split": "train", "revision": "rev", "row_index": index, "rendered_sha256": rendered,
            "public_record_sha256": public, "token_ids": tuple([128000] + list(range(1, 101))),
            "full_token_count": 101,
        })()
    blocked = candidate(0, "a" * 64, "b" * 64)
    eligible = candidate(1, "b" * 64, "c" * 64)
    exclusions = {
        "record_ids": set(), "rendered_sha256": set(), "public_record_sha256": {"b" * 64},
        "h128_sha256": set(), "source_indices": {},
    }
    selected = select_candidates(
        [blocked, eligible], stratum="pile_natural", exact_quota={100: 1},
        exclusions=exclusions, used_ids=set(), used_rendered=set(), used_h128=set(), used_public=set(),
    )
    assert [item.record_id for item, _ in selected] == [eligible.record_id]

def test_per_bank_diagnostics_bind_current_and_expanded_subsets_separately() -> None:
    rows = []
    strata = [item[0] for item in STRATA]
    for index in range(1200):
        stratum = strata[index % len(strata)]
        rows.append(InputRow(
            record_id=f"{stratum}-{index}", source_record_id=f"source-{index}",
            dataset_key=stratum.split("_")[0], stratum=stratum, source_row_index=index,
            rendered_sha256=f"{index + 1:064x}", source_full_token_count=65,
            target_post_bos_token_count=64, token_ids=tuple([128000] + list(range(1, 65))),
        ))
    value = per_bank_diagnostic_indices(rows)
    assert value["per_bank_record_count"] == 64
    assert len(value["current_bank"]["indices"]) == 64
    assert len(value["expanded_bank"]["indices"]) == 64
    assert value["current_bank"]["seed"] == value["expanded_bank"]["seed"] == 4010


def test_r2_controlled_assignment_reuses_each_b0_template_nine_times() -> None:
    b0 = []
    for template_index in range(2):
        b0.append(InputRow(
            record_id=f"b0-template-{template_index}", source_record_id=f"b0-parent-{template_index}",
            dataset_key="pile", stratum="pile_controlled", source_row_index=template_index,
            rendered_sha256=f"{template_index + 1:064x}", source_full_token_count=65,
            target_post_bos_token_count=64, token_ids=tuple([128000] + list(range(1, 65))),
            synthetic=True, replacement_positions=tuple(range(1 + template_index, 31 + template_index)),
            replacement_token_ids=tuple(1000 * (template_index + 1) + value for value in range(30)),
        ))
    buckets = build_b0_template_buckets(b0, expected_total=None)
    candidates = []
    for index in range(18):
        candidates.append((type("Candidate", (), {
            "record_id": f"candidate-{index}", "dataset_key": "pile", "dataset_id": "pile",
            "split": "train", "revision": "rev", "row_index": index,
            "rendered_sha256": f"{index + 10:064x}",
            "token_ids": tuple([128000] + list(range(1, 65))), "full_token_count": 65,
        })(), 64))
    additions, cursor = make_controlled_rows(
        candidates, "pile_controlled", list(range(4000, 7600)), 0,
        template_buckets=buckets,
    )
    assert cursor == 18 * 30
    audit = controlled_template_assignment_audit(
        b0, additions, expected_b0_templates=2, expected_controlled_addition_rows=18,
    )
    assert audit["exact_nine_use_check"] is True
    assert audit["controlled_addition_rows"] == 18
    assert audit["b0_template_bucket_counts"] == {"pile_controlled|64": 2}
    assert audit["controlled_addition_bucket_counts"] == {"pile_controlled|64": 18}


def test_controlled_selection_skips_parent_with_structural_fixed_template_slot() -> None:
    template = InputRow(
        record_id="b0-template", source_record_id="b0-parent", dataset_key="pile",
        stratum="pile_controlled", source_row_index=0, rendered_sha256="a" * 64,
        source_full_token_count=129, target_post_bos_token_count=128,
        token_ids=tuple([128000] + list(range(1, 129))), synthetic=True,
        replacement_positions=tuple(range(1, 31)),
        replacement_token_ids=tuple(range(1000, 1030)),
    )
    def candidate(index: int, second_token: int):
        return type("Candidate", (), {
            "record_id": f"candidate-{index}", "dataset_key": "pile", "dataset_id": "pile",
            "split": "train", "revision": "rev", "row_index": index,
            "rendered_sha256": f"{index + 100:064x}",
            "token_ids": tuple([128000, second_token] + list(range(2, 128))),
            "full_token_count": 129,
        })()
    exclusions = {
        "record_ids": set(), "rendered_sha256": set(), "public_record_sha256": set(),
        "h128_sha256": set(), "source_indices": {},
    }
    selected = select_candidates(
        [candidate(0, 128000), candidate(1, 7)], stratum="pile_controlled",
        exact_quota={128: 1}, exclusions=exclusions, used_ids=set(),
        used_rendered=set(), used_h128=set(), template_buckets={
            ("pile_controlled", 128): ((0, template),),
        },
    )
    assert [item.record_id for item, _target in selected] == ["candidate-1"]


def test_controlled_selection_revisits_structural_skip_for_later_template_slot() -> None:
    def template(index: int, offsets: tuple[int, ...]) -> InputRow:
        return InputRow(
            record_id=f"b0-template-{index}", source_record_id=f"b0-parent-{index}", dataset_key="pile",
            stratum="pile_controlled", source_row_index=index, rendered_sha256=f"{index + 1:064x}",
            source_full_token_count=129, target_post_bos_token_count=128,
            token_ids=tuple([128000] + list(range(1, 129))), synthetic=True,
            replacement_positions=offsets,
            replacement_token_ids=tuple(range(1000 + index * 30, 1030 + index * 30)),
        )

    templates = {
        ("pile_controlled", 128): (
            (0, template(0, tuple(range(0, 30)))),
            (1, template(1, tuple(range(1, 31)))),
        ),
    }

    def candidate(index: int, first_source_token: int, row_index: int):
        return type("Candidate", (), {
            "record_id": f"candidate-{index}", "dataset_key": "pile", "dataset_id": "pile",
            "split": "train", "revision": "rev", "row_index": row_index,
            "rendered_sha256": f"{index + 100:064x}",
            "token_ids": tuple([128000, first_source_token] + list(range(2, 129))),
            "full_token_count": 129,
        })()

    exclusions = {
        "record_ids": set(), "rendered_sha256": set(), "public_record_sha256": set(),
        "h128_sha256": set(), "source_indices": {},
    }
    assignments: dict[tuple[str, int, str], tuple[int, InputRow]] = {}
    skip_audit: dict[tuple[str, int], dict[str, object]] = {}
    selected = select_candidates(
        [candidate(0, 999, 2), candidate(1, 7, 10000)], stratum="pile_controlled",
        exact_quota={128: 2}, exclusions=exclusions, used_ids=set(),
        used_rendered=set(), used_h128=set(), template_buckets=templates,
        template_assignments=assignments, template_skip_audit=skip_audit,
        structural_token_ids={128000, 128001, 999},
    )
    assert [item.record_id for item, _target in selected] == ["candidate-1", "candidate-0"]
    assert assignments[("pile_controlled", 128, "candidate-1")][0] == 0
    assert assignments[("pile_controlled", 128, "candidate-0")][0] == 1
    audit = skip_audit[("pile_controlled", 128)]
    assert audit["required_slot_count"] == 2
    assert audit["selected_count"] == 2
    assert audit["structural_skip_count"] == 1
    assert audit["residual_ranked_eligible_count"] == 0
    assert len(str(audit["ordered_structural_skip_digest"])) == 64


def test_parent_exclusion_manifest_binds_dataset_scoped_rows(monkeypatch, tmp_path: Path) -> None:
    import hashlib
    import json
    manifest = {
        "selector_exclusion_sets": {
            "source_row_keys": [
                {"dataset_key": "alpaca", "row_index": 4},
                {"dataset_key": "alpaca", "row_index": 9},
                {"dataset_key": "pile", "row_index": 3},
                {"dataset_key": "finance", "row_index": 8},
            ],
            "record_ids": ["parent-record"],
            "opaque_sequence_or_reservation_digests": ["a" * 64],
        },
        "current_fit_records_excluded": [
            {"dataset_key": "alpaca", "record_id": "fit-a0", "source_record_id": "fit-source-a0", "rendered_sha256": "b" * 64},
            {"dataset_key": "alpaca", "record_id": "fit-a1", "source_record_id": "fit-source-a1", "rendered_sha256": "c" * 64},
            {"dataset_key": "pile", "record_id": "fit-p0", "source_record_id": "fit-source-p0", "rendered_sha256": "d" * 64},
            {"dataset_key": "finance", "record_id": "fit-f0", "source_record_id": "fit-source-f0", "rendered_sha256": "e" * 64},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(prep, "PARENT_EXCLUSION_MANIFEST_REL", Path("manifest.json"))
    monkeypatch.setattr(prep, "PARENT_EXCLUSION_MANIFEST_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(prep, "PARENT_EXCLUSION_SOURCE_ROW_COUNTS", {"alpaca": 2, "pile": 1, "finance": 1})
    monkeypatch.setattr(prep, "PARENT_EXCLUSION_CURRENT_FIT_COUNTS", {"alpaca": 2, "pile": 1, "finance": 1})
    result = prep.load_parent_exclusion_manifest(tmp_path)
    assert result["source_indices"] == {"alpaca": {4, 9}, "pile": {3}, "finance": {8}}
    assert {"fit-a0", "fit-source-a0", "parent-record"}.issubset(result["record_ids"])
    assert result["metadata"]["source_row_key_count"] == 4
