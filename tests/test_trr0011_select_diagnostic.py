from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trr0011_select_diagnostic import (
    LEDGER_SCHEMA,
    LEDGER_STATUS,
    P10_AUDIT_SCHEMA,
    P10_AUDIT_STATUS,
    P10_OPAQUE_SCHEMA,
    P10_OPAQUE_STATUS,
    P10_RELEASE_SCHEMA,
    P10_RELEASE_STATUS,
    REQUIRED_EXCLUSION_SCOPES,
    RESERVATION_SCHEMA,
    RESERVATION_STATUS,
    SCHEMA,
    SELECTION_STATUS,
    SelectionError,
    build_selection_binding,
    run_released_selector,
    validate_panel_spec,
    validate_p10_audit,
    validate_p10_opaque_reservation,
    validate_p10_release,
    validate_p10_release_bundle,
    validate_p10_reservation,
    validate_zero_overlap,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _panel() -> dict:
    return {
        "schema": "token-reconstruction.trr0011-transfer-panel.v2",
        "task_id": "TRR-0011",
        "status": "TEMPLATE_PENDING_OPAQUE_RESERVATION",
        "records_per_domain": 32,
        "same_record_order_across_targets": True,
        "target_variants": [
            "clean_public_base",
            "early_prefix_eps1e3",
            "early_prefix_eps1e2",
            "near_cut_prefix_eps1e3",
            "near_cut_prefix_eps1e2",
            "after_cut_suffix_eps1e2_null",
        ],
        "proposed_source_ranges_half_open": {
            "finance": "[28000,30000)",
            "pile": "[9000,10000)",
            "selection_seed": 5011,
            "max_records_per_domain": 32,
        },
        "opaque_reservation": {"status": "PENDING", "receipt_sha256": "PENDING"},
        "selection_performed": False,
        "truth_opened": False,
        "source_text_loaded": False,
    }


def _rows(offset: int) -> dict[str, list[dict[str, str]]]:
    return {
        style: [
            {
                "record_id": f"{style}:record:{offset + i}",
                "final_sequence_sha256": _sha(f"{style}:sequence:{offset + i}"),
            }
            for i in range(32)
        ]
        for style in ("finance", "pile")
    }


def _reservation(rows: dict[str, list[dict[str, str]]]) -> dict:
    return {
        "schema": RESERVATION_SCHEMA,
        "task_id": "TRR-P10",
        "status": RESERVATION_STATUS,
        "reservation_receipt_sha256": "a" * 64,
        "selection_seed": 5011,
        "source_ranges_half_open": {"finance": [28000, 30000], "pile": [9000, 10000]},
        "records_by_domain": rows,
        "order_digests": {
            style: {
                "record_ids_sha256": hashlib.sha256(
                    json.dumps([row["record_id"] for row in rows[style]], sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "final_sequence_sha256": hashlib.sha256(
                    json.dumps([row["final_sequence_sha256"] for row in rows[style]], sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            for style in ("finance", "pile")
        },
        "selection_performed": False,
        "truth_opened": False,
    }


def _ledger(scope: str, rows: dict[str, list[dict[str, str]]], path: Path) -> dict:
    payload = {
        "schema": LEDGER_SCHEMA,
        "task_id": "TRR-P10",
        "status": LEDGER_STATUS,
        "scope": scope,
        "namespace": f"trr-p10-test/{scope}",
        "records_by_domain": rows,
        "selection_performed": False,
        "truth_opened": False,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    raw = path.read_bytes()
    payload["file"] = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    raw = path.read_bytes()
    payload["file"] = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    # The descriptor is intentionally self-binding only when the second write
    # reaches a stable bytes/hash pair.  The production handoff uses an
    # external descriptor; the test exercises the normal no-self-binding path.
    payload.pop("file")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _write_inputs(tmp_path: Path):
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(_panel(), indent=2) + "\n")
    reservation_rows = _rows(1000)
    reservation_path = tmp_path / "reservation.json"
    reservation_path.write_text(json.dumps(_reservation(reservation_rows), indent=2) + "\n")
    ledger_paths: dict[str, Path] = {}
    for index, scope in enumerate(REQUIRED_EXCLUSION_SCOPES):
        path = tmp_path / f"{scope}.json"
        _ledger(scope, _rows(5000 + index * 100), path)
        ledger_paths[scope] = path
    return panel_path, reservation_path, ledger_paths, reservation_rows


def _write_p10_release_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create the smallest valid P10 release/audit/opaque handoff fixture."""

    audit_path = tmp_path / "p10_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema": P10_AUDIT_SCHEMA,
                "task_id": "TRR-P10",
                "status": P10_AUDIT_STATUS,
                "coverage_complete": True,
                "coverage_gaps": [],
                "access_boundary": {
                    "source_text_read": False,
                    "token_ids_read": False,
                    "activation_payload_read": False,
                    "truth_or_scores_read": False,
                    "new_panel_selected": False,
                    "p03_holdout_accessed": False,
                },
                "source_inventory": [],
                "union_identity_counts": {"record_ids": 0, "sequence_hashes_h128": 0},
            },
            indent=2,
        )
        + "\n"
    )
    audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()

    opaque_path = tmp_path / "p10_opaque.json"
    opaque_path.write_text(
        json.dumps(
            {
                "schema": P10_OPAQUE_SCHEMA,
                "task_id": "TRR-P10",
                "status": P10_OPAQUE_STATUS,
                "hash_conventions": {
                    "source_hash": "public_record_sha256",
                    "sequence_hash": "final_sequence_sha256 (canonical H128 including BOS)",
                },
                "source_hashes": [_sha(f"source:{i}") for i in range(64)],
                "sequence_hashes_h128": [_sha(f"sequence:{i}") for i in range(64)],
                "record_count": 64,
                "domain_counts_emitted": False,
                "source_ids_emitted": False,
                "source_indices_emitted": False,
                "source_payload_emitted": False,
                "token_ids_emitted": False,
                "truth_emitted": False,
            },
            indent=2,
        )
        + "\n"
    )

    release_path = tmp_path / "p10_release.json"
    release_path.write_text(
        json.dumps(
            {
                "schema": P10_RELEASE_SCHEMA,
                "task_id": "TRR-P10",
                "status": P10_RELEASE_STATUS,
                "exclusion_coverage_complete": True,
                "source_contract": {
                    "selection_seed": 5011,
                    "diagnostic_ranges": {"finance": [28000, 30000], "pile": [9000, 10000]},
                    "diagnostic_max_records_per_domain": 32,
                    "stored_sequence_tokens": 128,
                    "scored_post_bos_tokens": 127,
                    "target_conditions": ["public_base", "public_lora_2601"],
                },
                "bindings": {
                    "exclusion_audit_sha256": audit_sha,
                    "selector_code_sha256": "b" * 64,
                    "exclusion_worker_code_sha256": "c" * 64,
                },
                "truth_opened": False,
                "source_text_written": False,
                "token_ids_written": False,
                "model_loaded": False,
            },
            indent=2,
        )
        + "\n"
    )
    return release_path, audit_path, opaque_path


def test_panel_proposal_has_exact_ranges_variants_and_is_not_selection() -> None:
    result = validate_panel_spec(_panel())
    assert result["source_ranges_half_open"] == {"finance": [28000, 30000], "pile": [9000, 10000]}
    assert result["selection_performed"] is False


def test_pending_or_changed_reservation_fails_closed(tmp_path: Path) -> None:
    _, reservation_path, _, rows = _write_inputs(tmp_path)
    payload = _reservation(rows)
    payload["status"] = "PENDING"
    reservation_path.write_text(json.dumps(payload))
    with pytest.raises(SelectionError, match="approved"):
        validate_p10_reservation(reservation_path, root=tmp_path)


def test_zero_overlap_requires_all_namespaces_and_passes_disjoint_ledgers(tmp_path: Path) -> None:
    _, reservation_path, ledger_paths, _ = _write_inputs(tmp_path)
    reservation = validate_p10_reservation(reservation_path, root=tmp_path)
    from trr0011_select_diagnostic import validate_exclusion_ledger

    ledgers = [
        validate_exclusion_ledger(ledger_paths[scope], root=tmp_path, expected_scope=scope)
        for scope in REQUIRED_EXCLUSION_SCOPES
    ]
    result = validate_zero_overlap(reservation, ledgers)
    assert result["status"] == "PASS_ZERO_OVERLAP"
    assert all(counts == {"record_ids": 0, "final_sequence_sha256": 0} for counts in result["overlap_counts"].values())


def test_zero_overlap_rejects_cross_domain_duplicate_sequence(tmp_path: Path) -> None:
    _, reservation_path, ledger_paths, reservation_rows = _write_inputs(tmp_path)
    changed = _rows(5000)
    changed["pile"][0]["final_sequence_sha256"] = reservation_rows["finance"][0]["final_sequence_sha256"]
    _ledger("cross_range", changed, ledger_paths["cross_range"])
    reservation = validate_p10_reservation(reservation_path, root=tmp_path)
    from trr0011_select_diagnostic import validate_exclusion_ledger

    ledgers = [
        validate_exclusion_ledger(ledger_paths[scope], root=tmp_path, expected_scope=scope)
        for scope in REQUIRED_EXCLUSION_SCOPES
    ]
    with pytest.raises(SelectionError, match="overlaps"):
        validate_zero_overlap(reservation, ledgers)


def test_zero_overlap_rejects_missing_namespace(tmp_path: Path) -> None:
    _, reservation_path, ledger_paths, _ = _write_inputs(tmp_path)
    reservation = validate_p10_reservation(reservation_path, root=tmp_path)
    from trr0011_select_diagnostic import validate_exclusion_ledger

    ledgers = [
        validate_exclusion_ledger(ledger_paths[scope], root=tmp_path, expected_scope=scope)
        for scope in REQUIRED_EXCLUSION_SCOPES[:-1]
    ]
    with pytest.raises(SelectionError, match="four required"):
        validate_zero_overlap(reservation, ledgers)


def test_binding_is_identity_only_and_create_only(tmp_path: Path) -> None:
    panel_path, reservation_path, ledger_paths, _ = _write_inputs(tmp_path)
    output = tmp_path / "experiments" / "TRR-0011" / "evaluation" / "selection_binding.json"
    result = build_selection_binding(
        panel_path=panel_path,
        reservation_path=reservation_path,
        exclusion_paths=ledger_paths,
        root=tmp_path,
        output_path=output,
    )
    payload = json.loads(output.read_text())
    assert result["binding"]["sha256"]
    assert payload["selection_authorized"] is False
    assert payload["selection_performed"] is False
    assert "finance:record:1000" not in output.read_text()
    assert "pile:record:1000" not in output.read_text()
    with pytest.raises(SelectionError, match="create-only"):
        build_selection_binding(
            panel_path=panel_path,
            reservation_path=reservation_path,
            exclusion_paths=ledger_paths,
            root=tmp_path,
            output_path=output,
        )


def test_p10_release_bundle_gates_injected_selector(tmp_path: Path) -> None:
    release_path, audit_path, opaque_path = _write_p10_release_bundle(tmp_path)
    audit = validate_p10_audit(audit_path, root=tmp_path)
    opaque = validate_p10_opaque_reservation(opaque_path, root=tmp_path)
    release = validate_p10_release(release_path, root=tmp_path, audit_path=audit_path)
    assert audit["coverage_complete"] is True
    assert opaque["record_count"] == 64
    assert release["status"] == P10_RELEASE_STATUS
    bundle = validate_p10_release_bundle(
        release_path=release_path,
        audit_path=audit_path,
        opaque_path=opaque_path,
        root=tmp_path,
    )
    calls: list[int] = []

    def selector() -> dict:
        calls.append(1)
        return {"status": SELECTION_STATUS, "selection_performed": True, "truth_created_or_opened": False}

    with pytest.raises(SelectionError, match="different"):
        validate_p10_release(release_path, root=tmp_path, audit_path=opaque_path)
    assert calls == []
    assert run_released_selector(release_bundle=bundle, selector=selector)["selection_performed"] is True
    assert calls == [1]

    def trusted_t10_selector() -> dict:
        return {
            "status": "FROZEN_TRR0010_SOURCE_SELECTION_NO_TRUTH",
            "truth_created_or_opened": False,
            "execution": {"selection_performed": True, "truth_created_or_opened": False},
        }

    assert run_released_selector(release_bundle=bundle, selector=trusted_t10_selector)["status"].startswith("FROZEN_TRR0010")


def test_p10_release_rejects_incomplete_exclusion_audit(tmp_path: Path) -> None:
    release_path, audit_path, opaque_path = _write_p10_release_bundle(tmp_path)
    payload = json.loads(audit_path.read_text())
    payload["coverage_complete"] = False
    audit_path.write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(SelectionError, match="not complete"):
        validate_p10_release_bundle(
            release_path=release_path,
            audit_path=audit_path,
            opaque_path=opaque_path,
            root=tmp_path,
        )
