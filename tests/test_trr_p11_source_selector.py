"""Pure Python metadata tests for the bounded TRR-P11 selector."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.trr_p10.build_exclusion_audit import IdentityBundle, Namespace
from scripts.trr_p11 import source_selector as selector


ROOT = Path(__file__).resolve().parents[1]


class _Candidate:
    def __init__(self, token_ids: list[int], metadata: dict[str, object]) -> None:
        self.token_ids = token_ids
        self._metadata = metadata

    def selection_metadata(self) -> dict[str, object]:
        return dict(self._metadata)


def _candidate(domain: str, index: int, *, offset: int = 0) -> dict[str, object]:
    token_ids = [128000] + [offset + index * 1000 + value for value in range(1, 128)]
    raw = _Candidate(
        token_ids,
        {
            "record_id": f"{domain}/row-{index}",
            "public_record_sha256": hashlib.sha256(f"rendered:{domain}:{index}".encode()).hexdigest(),
            "dataset_key": domain,
            "dataset_id": selector._DATASET_META[domain]["dataset_id"],
            "split": "train",
            "revision": selector._DATASET_META[domain]["revision"],
            "row_index": index,
            "source_index": index,
            "full_token_count": 128,
            "post_bos_token_count": 127,
            "valid_tokens": selector.STORED_SEQUENCE_TOKENS,
        },
    )
    return selector._candidate_identity(raw)


def _candidate_for_tokens(domain: str, index: int, token_ids: list[int], *, label: str) -> dict[str, object]:
    raw = _Candidate(
        token_ids,
        {
            "record_id": f"{domain}/row-{index}",
            "public_record_sha256": hashlib.sha256(label.encode()).hexdigest(),
            "dataset_key": domain,
            "dataset_id": selector._DATASET_META[domain]["dataset_id"],
            "split": "train",
            "revision": selector._DATASET_META[domain]["revision"],
            "row_index": index,
            "source_index": index,
            "full_token_count": len(token_ids),
            "post_bos_token_count": len(token_ids) - 1,
            "valid_tokens": selector.STORED_SEQUENCE_TOKENS,
        },
    )
    return selector._candidate_identity(raw)


def test_exact_128_candidate_has_h128_and_optional_h129() -> None:
    row = _candidate("pile", 7)
    assert isinstance(row["h128_sequence_sha256"], str)
    assert row["final_sequence_sha256"] == row["h128_sequence_sha256"]
    assert row["h129_sequence_sha256"] is None
    assert isinstance(row["trr0002_active_token_ids_sha256"], str)
    assert isinstance(row["trr0002_h40_token_ids_sha256"], str)
    assert isinstance(row["h40_sequence_sha256"], str)
    assert row["h40_sequence_sha256"] != row["trr0002_h40_token_ids_sha256"]


def test_raw_h40_shared_prefix_rejects_different_rendered_and_h128_rows() -> None:
    first = [128000] + list(range(1, 128))
    second = first[:40] + [10000 + value for value in range(40, 128)]
    left = _candidate_for_tokens("pile", 21, first, label="left-rendered")
    right = _candidate_for_tokens("pile", 22, second, label="right-rendered")
    assert left["public_record_sha256"] != right["public_record_sha256"]
    assert left["h128_sequence_sha256"] != right["h128_sequence_sha256"]
    assert left["h40_sequence_sha256"] == right["h40_sequence_sha256"]
    assert left["h40_sequence_sha256"] != left["trr0002_h40_token_ids_sha256"]

    union = IdentityBundle("raw-h40", "opened_development", Path("raw-h40.json"), "", 0)
    namespace = Namespace(
        "pile",
        selector._DATASET_META["pile"]["dataset_id"],
        "train",
        selector._DATASET_META["pile"]["revision"],
    )
    union.add("h40_sequence_sha256", str(left["h40_sequence_sha256"]), namespace)
    reasons = selector._candidate_exclusion_reasons(right, union)
    assert any(reason["field"] == "h40_sequence_sha256" for reason in reasons)


def test_h40_historical_prefix_rejects_longer_candidate() -> None:
    row = _candidate("pile", 11)
    union = IdentityBundle("historical", "opened_development", Path("historical.json"), "", 0)
    namespace = Namespace(
        "pile",
        selector._DATASET_META["pile"]["dataset_id"],
        "train",
        selector._DATASET_META["pile"]["revision"],
    )
    union.add("trr0002_h40_token_ids_sha256", str(row["trr0002_h40_token_ids_sha256"]), namespace)
    reasons = selector._candidate_exclusion_reasons(row, union)
    assert any(reason["field"] == "trr0002_h40_token_ids_sha256" for reason in reasons)


def test_choose_identity_rows_excludes_union_and_keeps_exact_128() -> None:
    candidates = {
        "pile": [_candidate("pile", 0), _candidate("pile", 1), _candidate("pile", 2)],
        "finance": [_candidate("finance", 0, offset=10000), _candidate("finance", 1, offset=10000)],
    }
    union = IdentityBundle("union", "union", Path("union.json"), "", 0)
    namespace = Namespace(
        "pile",
        selector._DATASET_META["pile"]["dataset_id"],
        "train",
        selector._DATASET_META["pile"]["revision"],
    )
    union.add("trr0002_h40_token_ids_sha256", str(candidates["pile"][0]["trr0002_h40_token_ids_sha256"]), namespace)
    chosen, diagnostics = selector.choose_identity_rows(candidates, union=union, records_per_domain=2)
    assert [row["record_id"] for row in chosen["pile"]] == ["pile/row-1", "pile/row-2"]
    assert diagnostics["pile"]["excluded_identity"] == 1
    assert all(row["h129_sequence_sha256"] is None for rows in chosen.values() for row in rows)


def test_incomplete_audit_blocks_before_union_or_source_access(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema": selector.EXCLUSION_SCHEMA,
                "task_id": selector.TASK_ID,
                "coverage_complete": False,
                "selection_release": False,
                "access_boundary": {"p03_holdout_accessed": False},
                "source_inventory": [],
            }
        )
    )
    with pytest.raises(selector.SelectionError, match="complete exclusion coverage"):
        selector.load_complete_exclusions(audit_path, root=tmp_path)


def test_select_sources_rejects_unreleased_audit_before_source_input_load(tmp_path: Path) -> None:
    manifest_path, _states = _write_bound_manifest(tmp_path)
    audit_path = tmp_path / "incomplete-audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema": selector.EXCLUSION_SCHEMA,
                "task_id": selector.TASK_ID,
                "coverage_complete": False,
                "selection_release": False,
                "access_boundary": {"p03_holdout_accessed": False},
                "source_inventory": [],
            },
            sort_keys=True,
        )
    )
    # This path deliberately does not exist. The release gate must fail before
    # source-input normalization or any trusted public loader is reached.
    with pytest.raises(selector.SelectionError, match="complete exclusion coverage"):
        selector.select_sources(
            manifest_path=manifest_path,
            audit_path=audit_path,
            source_inputs=tmp_path / "unopened-source-inputs.json",
            output_path=tmp_path / "selection.json",
            repository_root=ROOT,
        )


def test_scorer_settings_are_explicit_and_nondefault() -> None:
    assert selector.scorer_contract() == {
        "bootstrap_seed": 9009,
        "bootstrap_resamples": 10000,
        "one_sided_alpha": 0.025,
        "exact_route_alpha": 0.025,
    }


def _replication_input_fixture() -> dict[str, object]:
    return {
        "status": "BOUND_INPUT_BANKS_AND_DEVELOPMENT_SELECTION",
        "banks": {
            "B0": {
                "bank": "B0",
                "bank_manifest": {
                    "path": "agent1/banks/b0_manifest.json",
                    "sha256": "1" * 64,
                },
                "ordered_identity": {
                    "path": "agent1/banks/b0_ordered_identity.json",
                    "sha256": "2" * 64,
                },
            },
            "B1": {
                "bank": "B1",
                "bank_manifest": {
                    "path": "agent1/banks/b1_manifest.json",
                    "sha256": "3" * 64,
                },
                "ordered_identity": {
                    "path": "agent1/banks/b1_ordered_identity.json",
                    "sha256": "4" * 64,
                },
            },
        },
        "development_selection": {
            "path": "agent1/development_selection.json",
            "sha256": "5" * 64,
        },
    }


def _state_binding_fixture() -> dict[str, dict[str, object]]:
    receipt = "c" * 64
    return {
        "current_b0": {
            "bank": "B0",
            "model_id": "new-b0",
            "selected_step": 8000,
            "state_id": "new-b0-state",
            "state_sha256": "a" * 64,
            "bank_manifest_sha256": "b" * 64,
            "selection_receipt_sha256": receipt,
        },
        "expanded_b1": {
            "bank": "B1",
            "model_id": "new-b1",
            "selected_step": 13000,
            "state_id": "new-b1-state",
            "state_sha256": "d" * 64,
            "bank_manifest_sha256": "e" * 64,
            "selection_receipt_sha256": receipt,
        },
    }


def _write_bound_manifest(tmp_path: Path) -> tuple[Path, dict[str, dict[str, object]]]:
    payload = json.loads((ROOT / "experiments/TRR-P11/manifest.json").read_text())
    states = _state_binding_fixture()
    payload["replication_inputs"] = _replication_input_fixture()
    payload["new_model_states"] = {
        "status": "PENDING_AGENT1_FIT",
        "current_b0": None,
        "expanded_b1": None,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    return path, states


def _write_complete_audit(
    tmp_path: Path,
    states: dict[str, dict[str, object]] | None = None,
    *,
    fields: dict[str, dict[str, list[object]]] | None = None,
    replication_inputs: dict[str, object] | None = None,
) -> Path:
    union_path = tmp_path / "identity_union.json"
    union_fields = fields or {}
    counts = {field: sum(len(values) for values in namespaces.values()) for field, namespaces in union_fields.items()}
    union = {
        "schema": "token-reconstruction.trr-p11-identity-union.v1",
        "task_id": selector.TASK_ID,
        "status": "IDENTITY_UNION_COMPLETE_NO_PAYLOAD",
        "fields": union_fields,
        "identity_counts": counts,
        "source_text_or_token_ids_written": False,
        "truth_opened": False,
        "p03_holdout_accessed": False,
    }
    union_path.write_text(json.dumps(union, sort_keys=True))
    union_bytes = union_path.read_bytes()
    binding = {
        "path": str(union_path),
        "bytes": len(union_bytes),
        "sha256": hashlib.sha256(union_bytes).hexdigest(),
    }
    audit = {
        "schema": selector.EXCLUSION_SCHEMA,
        "task_id": selector.TASK_ID,
        "coverage_complete": True,
        "selection_release": True,
        "access_boundary": {
            "p03_holdout_accessed": False,
            "source_or_token_payload_emitted": False,
            "truth_or_scores_read": False,
            "model_loaded": False,
            "new_panel_selected": False,
        },
        "source_inventory": [],
        "identity_union_export": binding,
        "union_identity_counts": counts,
    }
    if replication_inputs is not None:
        audit["replication_inputs"] = replication_inputs
    if states is not None:
        audit["agent1_replication_assets"] = {
            "status": "BOUND_AGENT1_FIT_COMPLETE",
            **states,
        }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit, sort_keys=True))
    return path


def test_complete_identity_union_roundtrip_and_hash_tamper_rejection(tmp_path: Path) -> None:
    namespace = "pile|NeelNanda/pile-10k|train|127bfedcd5047750df5ccf3a12979a47bfa0bafa"
    fields = {
        "h40_sequence_sha256": {namespace: ["e" * 64]},
        "trr0002_h40_token_ids_sha256": {namespace: ["f" * 64]},
    }
    audit_path = _write_complete_audit(tmp_path, fields=fields)
    context = selector.load_complete_exclusions(audit_path, root=tmp_path)
    assert context.union.counts() == {"h40_sequence_sha256": 1, "trr0002_h40_token_ids_sha256": 1}
    union_path = tmp_path / "identity_union.json"
    union_path.write_text(union_path.read_text() + "\n")
    with pytest.raises(selector.SelectionError, match="binding changed"):
        selector.load_complete_exclusions(audit_path, root=tmp_path)


def test_state_binding_requires_exact_b0_b1_bank_and_selection_identities(tmp_path: Path) -> None:
    manifest_path, _states = _write_bound_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    bound_states = _state_binding_fixture()
    assert selector.validate_p11_manifest(manifest, require_replication_inputs=True)["task_id"] == selector.TASK_ID
    manifest["decision"]["new_state_identities"] = "BOUND_AGENT1_FIT_COMPLETE"
    manifest["new_model_states"] = {
        "status": "BOUND_AGENT1_FIT_COMPLETE",
        **bound_states,
    }
    assert selector.validate_p11_manifest(manifest, require_state_bindings=True)["task_id"] == selector.TASK_ID
    bad = json.loads(json.dumps(manifest))
    bad["new_model_states"]["expanded_b1"]["bank"] = "B0"
    with pytest.raises(selector.SelectionError, match="bank/model identity"):
        selector.validate_p11_manifest(bad, require_state_bindings=True)


def test_select_sources_uses_real_gate_and_renderer_path_with_injected_trusted_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    manifest_path, states = _write_bound_manifest(tmp_path)
    audit_path = _write_complete_audit(
        tmp_path,
        replication_inputs=_replication_input_fixture(),
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_file = tokenizer_dir / "tokenizer.json"
    tokenizer_file.write_bytes(b"fake-tokenizer")
    pile_arrow = tmp_path / "pile.arrow"
    finance_arrow = tmp_path / "finance.arrow"
    pile_arrow.write_bytes(b"fake-pile")
    finance_arrow.write_bytes(b"fake-finance")

    class _Dataset:
        def __init__(self, size: int) -> None:
            self.size = size

        def __len__(self) -> int:
            return self.size

        def __getitem__(self, index: int) -> dict[str, int]:
            return {"index": index}

    trusted = types.ModuleType("scripts.trr0005_produce_confirmation")
    trusted.ProducerError = type("ProducerError", (Exception,), {})
    trusted._load_tokenizer = lambda path: object()
    trusted._load_arrow_dataset = lambda paths: _Dataset(2000 if Path(paths[0]).name == "pile.arrow" else 28000)

    def render(domain: str, row: dict[str, int], index: int, tokenizer: object) -> _Candidate:
        base = 1000 if domain == "pile" else 1000000
        token_ids = [128000] + [base + index * 256 + value for value in range(1, 128)]
        return _Candidate(
            token_ids,
            {
                "record_id": f"{domain}/row-{index}",
                "public_record_sha256": hashlib.sha256(f"rendered:{domain}:{index}".encode()).hexdigest(),
                "dataset_key": domain,
                "dataset_id": selector._DATASET_META[domain]["dataset_id"],
                "split": "train",
                "revision": selector._DATASET_META[domain]["revision"],
                "row_index": index,
                "source_index": index,
                "full_token_count": 128,
                "post_bos_token_count": 127,
                "valid_tokens": selector.STORED_SEQUENCE_TOKENS,
            },
        )

    trusted._render_row = render
    corpus = types.ModuleType("token_reconstruction.trr0005_public_corpus")
    corpus.deterministic_row_order = lambda values, *, dataset_key, seed: list(values)
    monkeypatch.setitem(sys.modules, "scripts.trr0005_produce_confirmation", trusted)
    monkeypatch.setitem(sys.modules, "token_reconstruction.trr0005_public_corpus", corpus)
    monkeypatch.setattr(selector, "_task_output", lambda path, *, root, phase: tmp_path / "selection.json")

    def record(path: Path) -> dict[str, object]:
        value = path.read_bytes()
        return {"path": str(path), "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}

    source_inputs = {
        "pile": {
            **selector._DATASET_META["pile"],
            "arrow_files": [record(pile_arrow)],
        },
        "finance": {
            **selector._DATASET_META["finance"],
            "arrow_files": [record(finance_arrow)],
        },
        "tokenizer": {
            "path": str(tokenizer_dir),
            "files": {"tokenizer.json": record(tokenizer_file)},
        },
    }
    result = selector.select_sources(
        manifest_path=manifest_path,
        audit_path=audit_path,
        source_inputs=source_inputs,
        output_path=ROOT / "experiments/TRR-P11/selection/synthetic.json",
        repository_root=ROOT,
    )
    assert result["status"] == selector.SELECTION_STATUS
    payload = json.loads((tmp_path / "selection.json").read_text())
    assert {domain: len(rows) for domain, rows in payload["selection_rule"]["records"].items()} == {
        "pile": selector.RECORDS_PER_DOMAIN,
        "finance": selector.RECORDS_PER_DOMAIN,
    }
    assert payload["truth_opened"] is False
    assert payload["selection_release"] is True
