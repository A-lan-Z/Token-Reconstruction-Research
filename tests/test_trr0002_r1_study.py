from __future__ import annotations

import numpy as np

from token_reconstruction.a1a2_configuration_search import PolicySpec, resolve_policy
from trr0002_r1_public_finance import (
    BOS_TOKEN_ID,
    PUBLIC_CURSOR_START,
    RECORDS,
    select_records,
)
from trr0002_r1_search import DomainSurface, evaluate_domain


class _FinanceRows:
    def __len__(self) -> int:
        return 50000

    def __getitem__(self, index: int) -> dict[str, str]:
        return {
            "system": "",
            "user": f"public question {index}",
            "assistant": f"public answer {index}",
        }


class _Tokenizer:
    pad_token_id = 128001

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "add_generation_prompt": False,
            "tokenize": True,
            "date_string": "06 Aug 2026",
        }
        raw_index = int(messages[-2]["content"].split()[-1])
        return [BOS_TOKEN_ID, 100 + raw_index % 1000, 200 + raw_index % 1000]


def test_public_finance_selection_starts_after_historical_cursor() -> None:
    records, cursor_end = select_records(_Tokenizer(), _FinanceRows())
    assert len(records) == RECORDS
    assert records[0]["raw_index"] == PUBLIC_CURSOR_START
    assert records[-1]["raw_index"] == PUBLIC_CURSOR_START + RECORDS - 1
    assert cursor_end == PUBLIC_CURSOR_START + RECORDS
    assert all(row["valid_tokens"] == 3 for row in records)
    assert len({row["content_sha256"] for row in records}) == RECORDS


def test_surface_suffix_stop_charges_failure_position_only() -> None:
    truth = np.array([10, 11, 12], dtype=np.int64)
    winners = {
        "direct_cosine": {
            2: truth.copy(),
            4: truth.copy(),
        }
    }
    signals = {
        "direct_cosine": {
            "raw_margin": {
                2: np.ones(3, dtype=np.float32),
                4: np.ones(3, dtype=np.float32),
            }
        }
    }
    surface = DomainSurface(
        truth=truth,
        a1_top=truth.copy(),
        a1_confidence=np.zeros(3, dtype=np.float32),
        record_index=np.zeros(3, dtype=np.int32),
        record_slices=((0, 3),),
        winners=winners,
        signals=signals,
    )
    spec = PolicySpec(
        kind="adaptive",
        score_rule="direct_cosine",
        schedule=(2, 4),
        fast_path_id="off",
        fast_path_threshold=None,
        routing_signal="raw_margin",
        gate_mode="never_accept",
        terminal_action="abstain_and_stop_suffix",
    )
    metrics = evaluate_domain(spec, resolve_policy(spec, {}), surface)
    assert metrics["correct_tokens"] == 0
    assert metrics["covered_tokens"] == 0
    assert metrics["candidate_simulations"] == 4
    assert metrics["exact_records"] == 0
