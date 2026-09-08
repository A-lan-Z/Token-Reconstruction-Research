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


def test_exact_128_candidate_has_h128_and_optional_h129() -> None:
    row = _candidate("pile", 7)
    assert isinstance(row["h128_sequence_sha256"], str)
    assert row["final_sequence_sha256"] == row["h128_sequence_sha256"]
    assert row["h129_sequence_sha256"] is None
    assert isinstance(row["trr0002_active_token_ids_sha256"], str)
    assert isinstance(row["trr0002_h40_token_ids_sha256"], str)


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


def test_scorer_settings_are_explicit_and_nondefault() -> None:
    assert selector.scorer_contract() == {
        "bootstrap_seed": 9009,
        "bootstrap_resamples": 10000,
        "one_sided_alpha": 0.025,
        "exact_route_alpha": 0.025,
    }
