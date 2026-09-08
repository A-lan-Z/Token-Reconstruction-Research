from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.trr_p10.build_exclusion_audit import (
    IdentityBundle,
    Namespace,
    candidate_sequence_fingerprints,
    check_candidate,
)
from scripts.trr_p11.exclusions import (
    ExclusionAuditError,
    build_audit,
    load_p04,
    rehash_public_token_rows,
    validate_panel_selection,
    write_identity_union_export,
    load_identity_union_export,
    _canonical_anchor_matches,
    _load_trr0003_h40_identity_export,
    _row_identity_fields,
)


ROOT = Path(__file__).resolve().parents[1]
PR20 = ROOT.parent / "TRR-0010"


def test_h128_h129_prefixes_are_exact_and_separate() -> None:
    values = list(range(192))
    full = candidate_sequence_fingerprints(values)
    first40 = candidate_sequence_fingerprints(values[:40])
    first128 = candidate_sequence_fingerprints(values[:128])
    first129 = candidate_sequence_fingerprints(values[:129])
    assert full["trr0002_h40_token_ids_sha256"] == first40["trr0002_h40_token_ids_sha256"]
    assert full["h128_sequence_sha256"] == first128["h128_sequence_sha256"]
    assert full["h129_sequence_sha256"] == first129["h129_sequence_sha256"]
    assert full["h128_sequence_sha256"] != full["h129_sequence_sha256"]


def test_h40_historical_prefix_rejects_longer_candidate_with_same_prefix() -> None:
    historical = list(range(40))
    longer_candidate = historical + list(range(1000, 1128))
    historical_fp = candidate_sequence_fingerprints(historical)
    candidate_fp = candidate_sequence_fingerprints(longer_candidate)
    bundle = IdentityBundle("trr2-pile", "opened_development", Path("trr2-pile.json"), "", 0)
    ns = Namespace("pile", "NeelNanda/pile-10k", "train", "rev")
    bundle.add("trr0002_h40_token_ids_sha256", historical_fp["trr0002_h40_token_ids_sha256"], ns)
    reasons = check_candidate(
        {**candidate_fp, "style": "pile", "dataset_id": "NeelNanda/pile-10k", "split": "train", "revision": "rev"},
        bundle,
    )
    assert any(item["field"] == "trr0002_h40_token_ids_sha256" for item in reasons)


def test_h129_helper_key_is_consumed_without_cross_namespace_match() -> None:
    values = list(range(192))
    fp = candidate_sequence_fingerprints(values)
    bundle = IdentityBundle("h129", "opened_evaluation", Path("h129.json"), "", 0)
    ns = Namespace("pile", "dataset", "train", "rev")
    bundle.add("h129_sequence_sha256", fp["h129_sequence_sha256"], ns)
    candidate = {**fp, "style": "pile", "dataset_id": "dataset", "split": "train", "revision": "rev"}
    assert {item["field"] for item in check_candidate(candidate, bundle)} == {"h129_sequence_sha256"}
    assert check_candidate(
        {
            "h128_sequence_sha256": fp["h129_sequence_sha256"],
            "style": "pile",
            "dataset_id": "dataset",
            "split": "train",
            "revision": "rev",
        },
        bundle,
    ) == []


def test_trr0003_h40_overlay_binds_producer_and_excludes_h128_h129() -> None:
    bundle, proof = _load_trr0003_h40_identity_export(ROOT)
    assert proof["record_count"] == 128
    assert proof["h40_rows"] == 128
    assert proof["h128_rows"] == 0
    assert proof["h129_rows"] == 0
    assert bundle.counts()["trr0002_h40_token_ids_sha256"] == 128
    assert "h128_sequence_sha256" not in bundle.counts()
    assert "h129_sequence_sha256" not in bundle.counts()
    assert proof["producer"]["sequence_convention"].startswith("first 40 BOS-inclusive")
    assert proof["failed_attempt_sha256"] == "fbfa33ca082dbaf00a5c0a725bcdc8d62570164096b6750f37b3ab8a168ce086"


def test_trr0004_truncated_hash_uses_verified_geometry_namespace() -> None:
    common = {
        "record_id": "pile10k-00000-fixture",
        "public_record_sha256": "a" * 64,
        "truncated_sequence_sha256": "b" * 64,
        "dataset_id": "NeelNanda/pile-10k",
        "split": "train",
        "revision": "rev",
    }
    _, pile_fields = _row_identity_fields({**common, "valid_tokens": 40}, source_label="trr0004_selection_plan")
    assert pile_fields["trr0002_h40_token_ids_sha256"] == {"b" * 64}
    assert "h128_sequence_sha256" not in pile_fields
    assert "h129_sequence_sha256" not in pile_fields

    _, finance_fields = _row_identity_fields(
        {**common, "record_id": "finance-public-000001-fixture", "valid_tokens": 128},
        source_label="trr0004_selection_plan",
    )
    assert finance_fields["h128_sequence_sha256"] == {"b" * 64}
    assert "trr0002_h40_token_ids_sha256" not in finance_fields
    assert "h129_sequence_sha256" not in finance_fields

    _, legacy_fields = _row_identity_fields(common, source_label="trr0008_selection_exclusions")
    assert legacy_fields["h129_sequence_sha256"] == {"b" * 64}


def test_row_anchor_requires_shared_strong_commitment() -> None:
    namespace = Namespace("pile", "fixture", "train", "rev")
    ref = ("canonical", ("h128",))
    index = {
        "global": {
            "record_id": {"same-id": [ref]},
            "rendered_sha256": {"a" * 64: [ref]},
        },
        "source_index": {},
    }
    assert _canonical_anchor_matches(namespace, {"record_id": {"same-id"}}, index) == set()
    assert _canonical_anchor_matches(namespace, {"h129_sequence_sha256": {"b" * 64}}, index) == set()
    assert _canonical_anchor_matches(namespace, {"rendered_sha256": {"a" * 64}}, index) == {ref}


def test_p04_mapping_is_strict_and_targetfit_is_explicitly_unavailable() -> None:
    result = load_p04(ROOT)
    counts = result.bundle.counts()
    assert counts["rendered_sha256"] == 1720
    assert counts["h129_sequence_sha256"] == 520
    assert counts["h128_sequence_sha256"] == 520
    assert result.proof["status"] == "PASS_PRODUCER_CONVENTION_VERIFIED_H128_TARGETFIT_PARTIAL"
    assert result.proof["producer_source_bytes_available"] is True
    assert result.proof["top_level_exchange_digest_recomputed"] is True
    assert result.proof["individual_record_ids_available"] is True
    assert result.proof["declared_convention_checks"]["signed_int32_binary"] is True
    assert result.proof["declared_convention_checks"]["bos_plus_128_h129"] is True
    assert result.proof["targetfit_individual_hashes_available"] is False
    assert result.proof["h128_individual_hashes_available"] is True
    assert result.proof["h128_identity_export"]["record_count"] == 520
    assert result.proof["h128_identity_export"]["recipe_migration"]["status"] == "PASS_RECIPE_PATH_MIGRATION_NO_RERUN"
    assert result.proof["targetfit_plan_proof"]["status"] == "PASS_TARGET_RULE_COUNTS_ONLY"
    assert result.proof["targetfit_plan_proof"]["selected_rows"] == 256
    assert result.proof["targetfit_plan_proof"]["row_ids_serialized_in_public_metadata"] is False
    assert all(result.proof["recovered_ledger_proof"][pool]["h128_values_present"] for pool in ("correction", "validation", "fresh_evaluation"))
    assert not any(gap["field"].endswith(".h128_sequence_sha256") for gap in result.gaps)
    assert any(gap["field"] == "targetfit.truncated_sequence_sha256" for gap in result.gaps)


def _real_panel_and_selection() -> tuple[dict, dict]:
    panel = json.loads(
        (ROOT / "experiments/TRR-0009/evaluation/public_observations_v2/panel.json").read_text()
    )
    selection = json.loads(
        (ROOT / "experiments/TRR-0009/selection_v2/source_selection.json").read_text()
    )
    return panel, selection


def test_aggregate_binding_rejects_digest_count_and_descriptor_mutations() -> None:
    panel, selection = _real_panel_and_selection()
    passed = validate_panel_selection(panel, selection, label="fixture")
    assert passed["status"] == "PASS"

    bad_digest = copy.deepcopy(panel)
    bad_digest["record_ids_sha256"]["pile"] = "0" * 64
    with pytest.raises(ExclusionAuditError, match="digest mismatch"):
        validate_panel_selection(bad_digest, selection, label="bad_digest")

    bad_count = copy.deepcopy(panel)
    bad_count["records_by_domain"]["pile"] -= 1
    with pytest.raises(ExclusionAuditError, match="row count mismatch"):
        validate_panel_selection(bad_count, selection, label="bad_count")

    bad_row = copy.deepcopy(selection)
    bad_row["selection_rule"]["records"]["pile"][0]["record_id"] = ""
    with pytest.raises(ExclusionAuditError, match="record_id"):
        validate_panel_selection(panel, bad_row, label="bad_row")

    bad_flags = copy.deepcopy(panel)
    bad_flags["truth_opened"] = True
    with pytest.raises(ExclusionAuditError, match="access flags"):
        validate_panel_selection(bad_flags, selection, label="bad_flags")


def test_actual_trr9_selection_record_is_rejected() -> None:
    selection = json.loads(
        (ROOT / "experiments/TRR-0009/selection_v2/source_selection.json").read_text()
    )
    rows = selection["selection_rule"]["records"]["pile"]
    row = rows[0]
    bundle = IdentityBundle("trr9", "opened_evaluation", Path("selection.json"), "", 0)
    ns = Namespace("pile", row["dataset_id"], row["split"], row["revision"])
    bundle.add("record_id", row["record_id"], ns)
    bundle.add("rendered_sha256", row["public_record_sha256"], ns)
    bundle.add("h128_sequence_sha256", row["final_sequence_sha256"], ns)
    reasons = check_candidate(
        {
            "record_id": row["record_id"],
            "public_record_sha256": row["public_record_sha256"],
            "final_sequence_sha256": row["final_sequence_sha256"],
            "style": "pile",
            "dataset_id": row["dataset_id"],
            "split": row["split"],
            "revision": row["revision"],
        },
        bundle,
    )
    assert {item["field"] for item in reasons} >= {"record_id", "rendered_sha256", "h128_sequence_sha256"}


def test_bounded_public_token_rehash_marks_short_rows_without_emitting_tokens(tmp_path: Path) -> None:
    metadata = [
        {
            "record_id": "pile/row-0",
            "dataset_key": "pile",
            "dataset_id": "pile",
            "split": "train",
            "revision": "rev",
            "rendered_sha256": "a" * 64,
        },
        {
            "record_id": "pile/row-1",
            "dataset_key": "pile",
            "dataset_id": "pile",
            "split": "train",
            "revision": "rev",
            "rendered_sha256": "b" * 64,
        },
    ]
    long_ids = [128000] + list(range(1, 128))
    short_ids = [128000] + list(range(1, 20))
    payload_path = tmp_path / "public.safetensors"
    payload_path.write_bytes(b"fixture")
    bundle, receipt = rehash_public_token_rows(
        metadata,
        [long_ids, short_ids],
        [[1] * len(long_ids), [1] * len(short_ids)],
        payload_path=payload_path,
        payload_sha256="c" * 64,
    )
    assert receipt["h128_rows"] == 1
    assert receipt["short_rows_h128_inapplicable"] == 1
    assert bundle.counts()["h128_sequence_sha256"] == 1
    assert receipt["token_values_emitted"] is False
    assert json.dumps(long_ids) not in json.dumps(receipt)


def test_full_metadata_audit_is_partial_but_binds_all_aggregate_panels() -> None:
    if not PR20.exists():
        pytest.skip("TRR-0010 worktree unavailable")
    audit = build_audit(root=ROOT, pr20_root=PR20)
    assert audit["status"] == "PARTIAL_CANONICAL_SEQUENCE_EXCLUSION_AUDIT"
    assert audit["coverage_complete"] is False
    assert audit["source_count"] == 48
    assert audit["aggregate_panel_binding"]["status"] == "PASS_ALL_SIX"
    assert audit["trr0009_selection_manifest_required_and_loaded"] is True
    assert audit["p04_convention_proof"]["status"] == "PASS_PRODUCER_CONVENTION_VERIFIED_H128_TARGETFIT_PARTIAL"
    assert audit["p04_convention_proof"]["h128_individual_hashes_available"] is True
    assert audit["canonical_sequence_audit"]["public_payload_rehash"]["status"] == "PENDING_ROOT_LEASE"
    sequence_by_label = {item["label"]: item for item in audit["canonical_sequence_audit"]["per_source"]}
    finance = sequence_by_label["trr0002_public_finance_records"]
    assert finance["direct_h128_count"] == 21
    assert finance["verified_short_rows_h128_inapplicable"] == 11
    assert finance["eligible_rows_without_h128"] == 0
    assert finance["status"] == "PARTIAL_H128_DERIVED_FROM_TRR2_ACTIVE_INPUT_PLUS_VERIFIED_SHORT_ROWS"
    pile = sequence_by_label["trr0002_public_pile_records"]
    assert pile["status"] == "H40_ONLY_H128_INAPPLICABLE_TO_OPENED_40_TOKEN_OBSERVATION"
    assert pile["eligible_rows_without_h128"] == 0
    assert any("P03" in gap for gap in audit["coverage_gaps"])
    assert any("targetfit" in gap for gap in audit["coverage_gaps"])
    assert not any("producer source bytes are unavailable" in gap for gap in audit["coverage_gaps"])
    assert audit["selection_release"] is False
    aliases = audit["canonical_sequence_audit"]["legacy_alias_reconciliation"]
    assert aliases["source_count_including_replication_metadata"] == 50
    assert aliases["unique_uncovered_identity_keys_across_sources"] > 0
    assert aliases["rows_without_verified_canonical_anchor"] > 0
    by_label = {item["label"]: item for item in aliases["per_source"]}
    assert by_label["trr0002_public_finance_records"]["rows_without_verified_canonical_anchor"] == 0
    assert by_label["trr0005_enriched_fit"]["rows_without_verified_canonical_anchor"] > 0
    assert by_label["trr0005_original_fit"]["eligible_rows_without_verified_canonical_anchor"] == 350
    assert all(item["available"] and item["sha256_match"] for item in aliases["replication_metadata"])


def test_sanitized_identity_union_round_trip_is_payload_free_and_create_only(tmp_path: Path) -> None:
    bundle = IdentityBundle("fixture", "union", tmp_path / "fixture.json", "", 0)
    namespace = Namespace("pile", "fixture", "train", "rev")
    bundle.add("record_id", "fixture/row-1", namespace)
    bundle.add("source_index", 17, namespace)
    bundle.add("rendered_sha256", "a" * 64, namespace)
    bundle.add("h128_sequence_sha256", "b" * 64, namespace)
    bundle.add("h129_sequence_sha256", "c" * 64, namespace)
    bundle.add("trr0002_h40_token_ids_sha256", "d" * 64, namespace)
    output = tmp_path / "identity_union.json"
    descriptor = write_identity_union_export(output, bundle, root=tmp_path)
    assert descriptor["schema"] == "token-reconstruction.trr-p11-identity-union.v1"
    assert descriptor["status"] == "PARTIAL_IDENTITY_UNION_NO_SELECTION_RELEASE"
    loaded = load_identity_union_export(output, expected_counts=bundle.counts())
    assert loaded.counts() == bundle.counts()
    assert loaded.namespace_counts() == bundle.namespace_counts()
    reasons = check_candidate(
        {
            "record_id": "fixture/row-1",
            "public_record_sha256": "a" * 64,
            "final_sequence_sha256": "b" * 64,
            "truncated_sequence_sha256": "c" * 64,
            "style": "pile",
            "dataset_id": "fixture",
            "split": "train",
            "revision": "rev",
        },
        loaded,
    )
    assert {item["field"] for item in reasons} >= {"record_id", "rendered_sha256", "h128_sequence_sha256", "h129_sequence_sha256"}
    with pytest.raises(ExclusionAuditError, match="overwrite"):
        write_identity_union_export(output, bundle, root=tmp_path)
    exported = json.loads(output.read_text())
    assert "source_text" not in exported
    assert "token_ids" not in exported
    assert "source_text" not in exported["fields"]
    assert "token_ids" not in exported["fields"]
    assert exported["counts"] == bundle.counts()


def test_recovered_canonical_closure_fails_closed_on_row_residuals(tmp_path: Path) -> None:
    if not PR20.exists():
        pytest.skip("TRR-0010 worktree unavailable")
    union_path = tmp_path / "identity_union_partial.json"
    closure_path = tmp_path / "closure_checkpoint.json"
    audit = build_audit(
        root=ROOT,
        pr20_root=PR20,
        include_recovered_identity_exports=True,
        identity_union_output=union_path,
        closure_output=closure_path,
    )
    assert audit["status"] == "PARTIAL_CANONICAL_SEQUENCE_EXCLUSION_AUDIT"
    assert audit["coverage_complete"] is False
    assert audit["selection_release"] is False
    assert audit["descriptor_pointer_proof"]["status"] == "PASS_DESCRIPTOR_POINTER_BINDINGS"
    failed = audit["completion_assessment"]["tests"]
    assert failed["all_eligible_rows_have_verified_canonical_anchor"] is False
    assert failed["no_unresolved_identity_rows"] is False
    aliases = audit["completion_assessment"]["legacy_alias_summary"]
    assert aliases["rows_without_verified_canonical_anchor"] > 0
    assert aliases["eligible_rows_without_verified_canonical_anchor"] > 0
    by_label = {item["label"]: item for item in audit["canonical_sequence_audit"]["legacy_alias_reconciliation"]["per_source"]}
    assert by_label["trr0005_original_fit"]["eligible_rows_without_verified_canonical_anchor"] == 350
    assert by_label["trr0007_original_fit"]["eligible_rows_without_verified_canonical_anchor"] == 350
    assert by_label["trr0003_fit_records"]["rows_without_verified_canonical_anchor"] == 0
    assert by_label["trr0003_fit_records_h40_public_identity"]["verified_h40_rows"] == 128
    assert by_label["trr0002_public_pile_records"]["verified_h40_rows"] == 96
    assert by_label["trr0006_p04_targetfit_public_identity"]["rows_without_verified_canonical_anchor"] == 0
    assert audit["identity_union_export"]["status"] == "PARTIAL_IDENTITY_UNION_NO_SELECTION_RELEASE"
    assert audit["closure_checkpoint"]["status"] == "PARTIAL_EXACT_PREFIX_COVERAGE"
    exported = json.loads(union_path.read_text())
    assert exported["coverage_complete"] is False
    assert exported["selection_release"] is False
    loaded = load_identity_union_export(union_path, expected_counts=audit["union_identity_counts"])
    assert loaded.counts() == audit["union_identity_counts"]
    closure = json.loads(closure_path.read_text())
    assert closure["rows_without_verified_canonical_anchor"] > 0
    assert closure["unresolved_rows_without_verified_canonical_anchor"] > 0
    assert closure["access_boundary"]["p03_holdout_accessed"] is False
