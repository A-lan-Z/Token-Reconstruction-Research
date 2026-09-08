from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.trr_p10.build_exclusion_audit import (
    AuditError,
    IdentityBundle,
    Namespace,
    SourceSpec,
    build_audit,
    candidate_sequence_fingerprints,
    _compare_ordered_selection_records,
    check_candidate,
    load_bundle,
    source_specs,
)


def _bundle() -> IdentityBundle:
    bundle = IdentityBundle("validation", "opened_evaluation", Path("validation.json"), "", 0)
    namespace = Namespace("pile", "dataset", "train", "rev")
    bundle.add("record_id", "reused-record", namespace)
    bundle.add("rendered_sha256", "a" * 64, namespace)
    bundle.add("h128_sequence_sha256", "b" * 64, namespace)
    bundle.add("source_index", 17, namespace)
    return bundle


def test_reused_validation_record_is_rejected_by_all_identity_conventions() -> None:
    reasons = check_candidate(
        {
            "record_id": "reused-record",
            "public_record_sha256": "a" * 64,
            "final_sequence_sha256": "b" * 64,
            "source_index": 17,
            "style": "pile",
            "dataset_id": "dataset",
            "split": "train",
            "revision": "rev",
        },
        _bundle(),
    )
    assert {reason["field"] for reason in reasons} == {"record_id", "rendered_sha256", "h128_sequence_sha256", "source_index"}


def test_source_index_is_namespace_scoped_but_h128_is_global() -> None:
    exclusions = _bundle()
    candidate = {"source_index": 17, "style": "finance", "dataset_id": "other", "split": "train", "revision": "rev", "final_sequence_sha256": "b" * 64}
    reasons = check_candidate(candidate, exclusions)
    assert [reason["field"] for reason in reasons] == ["h128_sequence_sha256"]


def test_h129_does_not_match_h128() -> None:
    assert check_candidate({"truncated_sequence_sha256": "b" * 64, "style": "pile", "dataset_id": "dataset"}, _bundle()) == []


def test_payload_bearing_generic_source_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text(json.dumps({"records": [{"record_id": "x", "input_ids": [1, 2, 3]}]}), encoding="utf-8")
    with pytest.raises(AuditError, match="payload-bearing"):
        load_bundle(SourceSpec("payload", "fitting_bank", str(path)), root=tmp_path)


def test_panel_mode_skips_payload_and_keeps_identity(tmp_path: Path) -> None:
    path = tmp_path / "panel.json"
    path.write_text(json.dumps({"cells": [{"records": [{"record_id": "r", "public_record_sha256": "a" * 64, "input_ids": [1, 2], "attention_mask": [1, 1]}]}]}), encoding="utf-8")
    bundle = load_bundle(SourceSpec("panel", "opened_evaluation", str(path), metadata_mode="panel_identity"), root=tmp_path)
    assert bundle is not None
    assert bundle.counts()["record_id"] == 1
    assert bundle.counts()["rendered_sha256"] == 1


def test_p03_is_absent_from_explicit_inventory() -> None:
    assert all("P03" not in spec.path.upper() for spec in source_specs())


def test_actual_trr0009_selection_row_is_rejected() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = next(spec for spec in source_specs() if spec.label == "trr0009_selection_v2")
    bundle = load_bundle(spec, root=root)
    assert bundle is not None
    payload = json.loads((root / spec.path).read_text(encoding="utf-8"))
    row = payload["selection_rule"]["records"]["pile"][0]
    reasons = check_candidate(
        {
            "record_id": row["record_id"],
            "public_record_sha256": row["public_record_sha256"],
            "final_sequence_sha256": row["final_sequence_sha256"],
            "source_index": row["source_index"],
            "dataset_id": row["dataset_id"],
            "split": row["split"],
            "revision": row["revision"],
            "style": "pile",
        },
        bundle,
    )
    assert {reason["field"] for reason in reasons} >= {"record_id", "rendered_sha256", "h128_sequence_sha256"}


def test_candidate_fingerprint_conventions_are_separate() -> None:
    h40 = candidate_sequence_fingerprints([1] * 40)
    h128 = candidate_sequence_fingerprints([1] * 128)
    assert "trr0002_h40_token_ids_sha256" in h40
    assert "h128_sequence_sha256" not in h40
    assert "h128_sequence_sha256" in h128


def test_build_audit_is_partial_and_binds_trr0009(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    pr20 = root.parent / "TRR-0010"
    if not pr20.exists():
        pytest.skip("Agent1 worktree is unavailable in this isolated checkout")
    audit = build_audit(root=root, pr20_root=pr20)
    assert audit["status"] == "PARTIAL_METADATA_ONLY_EXCLUSION_AUDIT"
    assert audit["coverage"]["trr0009_selection_manifest_included"] is True
    assert audit["coverage"]["accessible_explicit_sources_loaded"] is True
    assert audit["focused_intersections"]["trr0009_selection_v2_vs_pr20"]["status"] == "PASS_128_PER_DOMAIN"
    assert any("P03" in gap for gap in audit["coverage_gaps"])



def test_long_candidate_emits_prefix_hashes_from_exact_slices() -> None:
    values = list(range(192))
    full = candidate_sequence_fingerprints(values)
    first40 = candidate_sequence_fingerprints(values[:40])
    first128 = candidate_sequence_fingerprints(values[:128])
    first129 = candidate_sequence_fingerprints(values[:129])
    assert full["trr0002_h40_token_ids_sha256"] == first40["trr0002_h40_token_ids_sha256"]
    assert full["h128_sequence_sha256"] == first128["h128_sequence_sha256"]
    assert full["h129_sequence_sha256"] == first129["h129_sequence_sha256"]
    assert full["h128_sequence_sha256"] != full["h129_sequence_sha256"]
    assert full["trr0002_active_token_ids_sha256"] != first128["trr0002_active_token_ids_sha256"]


def test_long_candidate_helper_output_rejects_h129_without_cross_namespace_match() -> None:
    values = list(range(192))
    fingerprints = candidate_sequence_fingerprints(values)
    exclusions = IdentityBundle("h129", "opened_evaluation", Path("h129.json"), "", 0)
    namespace = Namespace("pile", "dataset", "train", "rev")
    exclusions.add("h129_sequence_sha256", fingerprints["h129_sequence_sha256"], namespace)

    candidate = {
        **fingerprints,
        "style": "pile",
        "dataset_id": "dataset",
        "split": "train",
        "revision": "rev",
    }
    reasons = check_candidate(candidate, exclusions)
    assert {reason["field"] for reason in reasons} == {"h129_sequence_sha256"}

    # An H128 field carrying the same bytes must remain in the H128 namespace;
    # it cannot match an exclusion recorded as H129.
    h128_labelled_as_h129 = {
        "h128_sequence_sha256": fingerprints["h129_sequence_sha256"],
        "style": "pile",
        "dataset_id": "dataset",
        "split": "train",
        "revision": "rev",
    }
    assert check_candidate(h128_labelled_as_h129, exclusions) == []


def test_ordered_binding_negative_fixture_rejects_mismatch_and_malformed_rows() -> None:
    def row(index: int) -> dict[str, str]:
        return {
            "record_id": f"record-{index}",
            "public_record_sha256": f"{index + 1:064x}",
            "final_sequence_sha256": f"{index + 2:064x}",
        }

    left = {"finance": [row(i) for i in range(256)], "pile": [row(1000 + i) for i in range(128)]}
    right = {"finance": [dict(item) for item in left["finance"][:128]], "pile": [dict(item) for item in left["pile"]]}
    right["finance"][0]["record_id"] = "different-record"
    result = _compare_ordered_selection_records(left, right)
    assert result["status"] == "FAIL_ORDERED_IDENTITY_MISMATCH"

    malformed = {domain: [dict(item) for item in rows] for domain, rows in right.items()}
    malformed["pile"][0]["record_id"] = ""
    result = _compare_ordered_selection_records(left, malformed)
    assert result["status"] == "FAIL_MALFORMED_IDENTITY"



def test_unrecognized_identity_hash_key_is_reported_without_admission(tmp_path: Path) -> None:
    path = tmp_path / "unknown.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "known-record",
                        "mystery_sequence_sha256": "a" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    bundle = load_bundle(SourceSpec("unknown", "fitting_bank", str(path)), root=tmp_path)
    assert bundle is not None
    assert bundle.counts() == {"record_id": 1}
    assert dict(bundle.unknown_identity_keys) == {"mystery_sequence_sha256": 1}
