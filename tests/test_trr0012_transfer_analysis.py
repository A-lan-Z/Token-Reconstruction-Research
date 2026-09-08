from __future__ import annotations

import pytest

from scripts.trr0012_transfer_analysis import (
    TransferAnalysisError,
    validate_truth_order_declarations,
)


def test_truth_order_binding_rejects_swapped_domain_digests() -> None:
    expected = {
        "panel_sha256": "p" * 64,
        "domain_record_order_sha256": {
            "finance": "f" * 64,
            "pile": "b" * 64,
        },
    }
    manifest = {
        "panel_sha256": expected["panel_sha256"],
        "domain_record_order_sha256": dict(expected["domain_record_order_sha256"]),
    }
    assert validate_truth_order_declarations(
        manifest,
        expected_panel_sha256=expected["panel_sha256"],
        expected_domain_record_order_sha256=expected["domain_record_order_sha256"],
    ) == expected

    swapped = {
        "panel_sha256": expected["panel_sha256"],
        "domain_record_order_sha256": {
            "finance": "b" * 64,
            "pile": "f" * 64,
        },
    }
    with pytest.raises(TransferAnalysisError, match="ordered-ID digests"):
        validate_truth_order_declarations(
            swapped,
            expected_panel_sha256=expected["panel_sha256"],
            expected_domain_record_order_sha256=expected["domain_record_order_sha256"],
        )
