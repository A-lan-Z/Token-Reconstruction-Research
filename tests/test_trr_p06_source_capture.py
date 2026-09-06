from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from scripts.trr_p06 import capture_public as capture
from scripts.trr_p06 import prepare_public_panel as panel
from scripts.trr_p06 import source_binding as binding


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_nested_opaque_hash_ledgers_bind_without_token_array_identity(tmp_path: Path) -> None:
    known_id = "published-record-17"
    known_context = "published-context-17"
    known_rendered = _digest("rendered")
    known_sequence = _digest("sequence")
    nested_rendered = _digest("nested-rendered")
    nested_sequence = _digest("nested-sequence")
    metadata = tmp_path / "opaque.json"
    metadata.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": known_id,
                        "context_id": known_context,
                        "public_record_sha256": known_rendered,
                        "final_sequence_sha256": known_sequence,
                        "token_ids": [known_id, 1, 2],
                        "context_token_ids": [3, 4, 5],
                    }
                ],
                "hashes": {
                    "public_record_sha256": [nested_rendered],
                    "final_sequence_sha256": [nested_sequence],
                },
            }
        ),
        encoding="utf-8",
    )

    index = binding.collect_exclusions(
        tmp_path,
        metadata_paths=[metadata],
        include_default_catalog=False,
    )

    assert known_id in index.ids
    assert known_context in index.ids
    assert known_rendered in index.hashes
    assert known_sequence in index.sequence_hashes
    assert nested_rendered in index.hashes
    assert nested_sequence in index.sequence_hashes
    assert known_id not in index.hashes
    assert index.block_reason(
        record_id=known_id,
        public_record_sha256=_digest("new-rendered"),
        final_sequence_sha256=_digest("new-sequence"),
    ) == "prior_record_id"
    assert index.block_reason(
        record_id="new-record",
        public_record_sha256=nested_rendered,
        final_sequence_sha256=_digest("new-sequence"),
    ) == "prior_rendered_or_text_hash"
    assert index.block_reason(
        record_id="new-record",
        public_record_sha256=_digest("new-rendered"),
        final_sequence_sha256=nested_sequence,
    ) == "prior_sequence_hash"


def test_missing_approved_opaque_ledger_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(binding.SourceBindingError, match="required exclusion metadata is missing"):
        binding.collect_exclusions(
            tmp_path,
            include_default_catalog=False,
            approved_opaque_paths=[tmp_path / "approved-trr0007.json"],
        )


def test_approved_trr0007_export_digest_is_bound_before_json_scan(tmp_path: Path) -> None:
    path = tmp_path / "p06_opaque_source_sequence_reservation.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(binding.SourceBindingError, match="approved metadata hash changed"):
        binding.collect_exclusions(
            tmp_path,
            include_default_catalog=False,
            approved_opaque_paths=[path],
        )


def test_default_catalog_does_not_guess_trr0007_path(tmp_path: Path) -> None:
    specs = binding.catalog_specs(tmp_path)
    assert not any("trr0007" in spec.path.casefold() for spec in specs)


def test_source_universe_is_proposed_until_explicit_freeze(tmp_path: Path) -> None:
    empty = binding.ExclusionIndex(
        ids=frozenset(),
        hashes=frozenset(),
        sequence_hashes=frozenset(),
        text_hashes=frozenset(),
        indices=frozenset(),
        descriptors=(),
        coverage_complete=True,
        missing_labels=(),
        catalog_sha256="synthetic-catalog",
    )
    proposed = panel._universe_metadata(
        root=tmp_path,
        plan_binding={"path": "synthetic-plan.json", "sha256": "synthetic-plan"},
        exclusions=empty,
    )
    path = tmp_path / "universe.json"
    path.write_text(json.dumps(proposed), encoding="utf-8")
    loaded = panel.load_universe(path)
    assert loaded["status"] == "PROPOSED_BEFORE_ENUMERATION"
    with pytest.raises(panel.PanelPreparationError, match="FROZEN_SOURCE_UNIVERSE"):
        panel.load_universe(path, require_frozen=True)

    frozen = copy.deepcopy(proposed)
    frozen["status"] = "FROZEN_SOURCE_UNIVERSE"
    path.write_text(json.dumps(frozen), encoding="utf-8")
    assert panel.load_universe(path, require_frozen=True)["status"] == "FROZEN_SOURCE_UNIVERSE"


def test_payload_fields_are_rejected_before_selection_or_capture() -> None:
    with pytest.raises(panel.PanelPreparationError, match="source/truth payload"):
        panel._reject_payload({"record": {"token_ids": [1, 2, 3]}})
    with pytest.raises(panel.PanelPreparationError, match="source/truth payload"):
        panel._reject_payload({"record": {"source_text": "never persisted"}})


def test_candidate_reader_enforces_declared_ranges_without_dataset_load() -> None:
    class SyntheticDataset:
        def __getitem__(self, index: int) -> dict[str, int]:
            return {"index": index}

    dataset = SyntheticDataset()
    assert panel._read_candidate_row(dataset, style="pile", row_index=0)["index"] == 0
    assert panel._read_candidate_row(dataset, style="finance", row_index=25999)["index"] == 25999
    with pytest.raises(panel.PanelPreparationError, match="escaped declared range"):
        panel._read_candidate_row(dataset, style="pile", row_index=7000)
    with pytest.raises(panel.PanelPreparationError, match="escaped declared range"):
        panel._read_candidate_row(dataset, style="finance", row_index=19999)


def test_capture_adapter_matches_published_helper_keyword_contract() -> None:
    parameters = inspect.signature(capture.trusted._capture_prefix).parameters
    assert "lora_update" in parameters
    assert "lora_update_path" not in parameters
    assert capture.CAPTURE_SEQUENCE_TOKENS == 192
    assert capture.SEQUENCE_TOKENS == 128
