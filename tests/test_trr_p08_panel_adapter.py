from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.trr_p06.source_binding import ExclusionIndex
from scripts.trr_p06 import capture_public as p06_capture
from scripts.trr_p08 import capture_public
from scripts.trr_p08 import prepare_public_panel as panel
from scripts.trr_p08 import run_predictions


def _empty_exclusions() -> ExclusionIndex:
    return ExclusionIndex(
        ids=frozenset(),
        hashes=frozenset(),
        sequence_hashes=frozenset(),
        text_hashes=frozenset(),
        indices=frozenset(),
        descriptors=(),
        coverage_complete=False,
        missing_labels=("approved-ledger-pending",),
        catalog_sha256="synthetic-catalog",
    )


def test_p08_plan_binding_requires_frozen_design_and_no_payload(tmp_path: Path) -> None:
    plan = {
        "schema": panel.PLAN_SCHEMA,
        "task_id": panel.TASK_ID,
        "status": "FROZEN_DESIGN_PRE_FIT",
        "scope": {"fit_arms": 8},
        "fresh_evaluation": {
            "records_per_domain": panel.RECORDS_PER_DOMAIN,
            "target_conditions": list(panel.CONDITION_ORDER),
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    binding = panel._plan_binding(path)
    assert binding["fit_arms"] == 8
    assert binding["records_per_domain"] == 256

    plan["fresh_evaluation"]["source_text"] = "forbidden"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(panel.PanelPreparationError, match="source/truth payload"):
        panel._plan_binding(path)


def test_universe_is_metadata_only_until_explicit_freeze(tmp_path: Path) -> None:
    ranges = {"pile": [0, 7000], "finance": [20000, 26000]}
    value = panel._universe_metadata(
        root=tmp_path,
        plan_binding={"path": "plan.json", "sha256": "plan-sha"},
        seed=8088,
        ranges=ranges,
        exclusions=_empty_exclusions(),
    )
    assert value["status"] == "PROPOSED_BEFORE_ENUMERATION"
    assert value["access_boundary"]["source_rows_read"] is False
    assert value["access_boundary"]["panel_selected"] is False
    assert value["panel_contract"]["unique_source_records"] == 512
    path = tmp_path / "universe.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    loaded = panel.load_universe(path)
    assert loaded["status"] == "PROPOSED_BEFORE_ENUMERATION"
    with pytest.raises(panel.PanelPreparationError, match="requires FROZEN_SOURCE_UNIVERSE"):
        panel.load_universe(path, require_frozen=True)


def test_superseded_frozen_universe_without_required_ledger_contract_fails_closed(tmp_path: Path) -> None:
    value = panel._universe_metadata(
        root=tmp_path,
        plan_binding={"path": "plan.json", "sha256": "plan-sha"},
        seed=8088,
        ranges={"pile": [0, 7000], "finance": [20000, 26000]},
        exclusions=_empty_exclusions(),
    )
    value["status"] = "FROZEN_SOURCE_UNIVERSE"
    path = tmp_path / "superseded-frozen.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(panel.PanelPreparationError, match="required opaque ledger contract"):
        panel.load_universe(path, require_frozen=True)


def _valid_required_opaque_contract() -> dict:
    bindings = {
        spec["key"]: [{
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": spec["sha256"],
            "schema": spec["schema"],
            "counts": dict(spec["counts"]),
        }]
        for path, spec in panel.APPROVED_OPAQUE_SPECS.items()
    }
    return panel._required_opaque_ledger_contract(bindings)


@pytest.mark.parametrize("field", ["sha256", "path"])
def test_frozen_contract_rejects_changed_approved_ledger_entry(field: str) -> None:
    contract = _valid_required_opaque_contract()
    entry = contract["ledgers"]["approved_trr0009_replacement_opaque"]
    entry[field] = "0" * 64 if field == "sha256" else "/tmp/substituted-reservation.json"
    with pytest.raises(panel.PanelPreparationError, match="does not match approved binding"):
        panel._validate_required_opaque_ledger_contract(
            {"exclusion_binding": {"required_opaque_ledger_contract": contract}}
        )


def test_frozen_contract_rejects_missing_approved_ledger_entry() -> None:
    contract = _valid_required_opaque_contract()
    del contract["ledgers"]["approved_trr0009_replacement_opaque"]
    with pytest.raises(panel.PanelPreparationError, match="entries do not match approved set"):
        panel._validate_required_opaque_ledger_contract(
            {"exclusion_binding": {"required_opaque_ledger_contract": contract}}
        )


def test_universe_ranges_are_bound_to_both_domains() -> None:
    with pytest.raises(panel.PanelPreparationError, match="missing finance"):
        panel._configure_p06(seed=8088, ranges={"pile": [0, 7000]})


def test_capture_adapter_rebinds_only_p08_metadata_contract() -> None:
    universe = {
        "provenance": {"selection_seed": 8088},
        "candidate_source_universe": {
            "pile": {"candidate_range_half_open": [0, 7000]},
            "finance": {"candidate_range_half_open": [20000, 26000]},
        },
    }
    capture_public._configure_capture(universe)
    assert p06_capture.TASK_ID == "TRR-P08"
    assert p06_capture.SELECTION_SCHEMA == panel.SELECTION_SCHEMA
    assert p06_capture.SEQUENCE_TOKENS == 128
    assert p06_capture.CAPTURE_SEQUENCE_TOKENS == 192
    assert p06_capture.CELL_ORDER == (
        "pile__public_base",
        "pile__public_lora_2601",
        "finance__public_base",
        "finance__public_lora_2601",
    )


def test_capture_manifest_matches_prediction_validator_without_payloads(tmp_path: Path) -> None:
    universe = {
        "provenance": {"selection_seed": 8088},
        "candidate_source_universe": {
            "pile": {"candidate_range_half_open": [0, 7000]},
            "finance": {"candidate_range_half_open": [20000, 26000]},
        },
    }
    capture_public._configure_capture(universe)
    selection_path = tmp_path / "selection.json"
    universe_path = tmp_path / "universe.json"
    selection_path.write_text("{}\n", encoding="utf-8")
    universe_path.write_text("{}\n", encoding="utf-8")
    observations = {}
    for cell_id in capture_public.CELL_ORDER:
        path = tmp_path / f"{cell_id}.safetensors"
        path.write_bytes(b"metadata-only-fixture")
        digest = capture_public._sha256_file(path)
        style, condition = cell_id.split("__", 1)
        observations[cell_id] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "shape": [256, 128, 2048],
            "stored_sequence_tokens": 128,
            "scored_post_bos_tokens": 127,
        }
    manifest = capture_public.p06_capture._observation_manifest(
        selection_path=selection_path,
        selection_sha256=capture_public._sha256_file(selection_path),
        universe_path=universe_path,
        universe_sha256=capture_public._sha256_file(universe_path),
        observations=observations,
        record_ids_sha256={"pile": "a" * 64, "finance": "b" * 64},
    )
    manifest_path = tmp_path / "observations.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cells, evidence = run_predictions._validate_observation_manifest(manifest_path, root=tmp_path)
    assert list(cells) == list(capture_public.CELL_ORDER)
    assert evidence["record_ids_sha256"] == {"pile": "a" * 64, "finance": "b" * 64}


def test_explicit_prior_bindings_include_all_required_opaque_ledgers():
    opaque_paths = tuple(panel.APPROVED_OPAQUE_SPECS)
    bindings, paths = panel._explicit_prior_exclusion_bindings(
        Path("/tmp/trr-p08"),
        opaque_paths,
    )
    assert paths[0] == (Path("/tmp/trr-p08") / panel.PUBLISHED_P06_SELECTION_RELATIVE).resolve()
    assert bindings["published_p06_selection"]["sha256"] == panel.PUBLISHED_P06_SELECTION_SHA256
    assert bindings["approved_trr0007_opaque"][0]["counts"] == {
        "public_record_sha256": 256,
        "final_sequence_sha256": 256,
    }
    assert bindings["approved_trr0008_opaque"][0]["counts"] == {
        "public_record_sha256": 1408,
        "final_sequence_sha256": 1408,
    }
    assert bindings["approved_trr0009_opaque"][0]["counts"] == {
        "public_record_sha256": 384,
        "final_sequence_sha256": 384,
    }
    assert bindings["approved_trr0009_replacement_opaque"][0]["counts"] == {
        "public_record_sha256": 384,
        "final_sequence_sha256": 384,
    }


def test_explicit_prior_bindings_fail_closed_when_any_required_ledger_is_missing():
    opaque_paths = tuple(panel.APPROVED_OPAQUE_SPECS)
    with pytest.raises(panel.PanelPreparationError, match="ledger set is incomplete"):
        panel._explicit_prior_exclusion_bindings(
            Path("/tmp/trr-p08"),
            opaque_paths[:-1],
        )
