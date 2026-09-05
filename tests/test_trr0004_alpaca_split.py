from __future__ import annotations

import pytest

from token_reconstruction.alpaca_split import (
    AlpacaSplitError,
    DEFAULT_BOS_TOKEN_ID,
    HISTORICAL_MAX_USER_CHARS,
    assert_disjoint_record_sets,
    build_split_registration,
    historical_permutation,
    historical_rendered_text,
    public_record_id,
    validate_confirmation_ids,
)


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.input_ids = ids


class _FakeTokenizer:
    bos_token_id = DEFAULT_BOS_TOKEN_ID

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], bool, bool]] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.calls.append((messages, tokenize, add_generation_prompt))
        return "<BOS>" + messages[0]["content"] + "<GEN>"

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        # Length is intentionally deterministic and includes the expected BOS.
        return _Encoding([DEFAULT_BOS_TOKEN_ID] + list(range(max(0, len(text) - 1))))


def _dataset(n: int = 20) -> list[dict[str, str]]:
    return [
        {
            "instruction": f"instruction-{i} " + ("x" * 45),
            "input": "input-" + str(i),
            "output": "answer-" + str(i),
        }
        for i in range(n)
    ]


def test_historical_rendering_caps_user_and_output_and_calls_chat_template() -> None:
    tokenizer = _FakeTokenizer()
    record = {"instruction": "a" * 1300, "input": "b" * 200, "output": "c" * 1500}
    rendered = historical_rendered_text(record, tokenizer)
    messages, tokenize, add_generation_prompt = tokenizer.calls[0]

    assert tokenize is False
    assert add_generation_prompt is True
    assert messages[0]["content"] == ("a" * 1300 + "\n\n" + "b" * 200)[:HISTORICAL_MAX_USER_CHARS]
    assert rendered.startswith("<BOS>")
    assert rendered.endswith("c" * 1200)


def test_historical_permutation_matches_torch_and_is_deterministic() -> None:
    first = historical_permutation(100, seed=7)
    second = historical_permutation(100, seed=7)
    assert first == second
    assert sorted(first) == list(range(100))
    assert first[:5] == [15, 59, 71, 57, 2]


def test_registration_is_nested_and_disjoint_without_token_ids() -> None:
    tokenizer = _FakeTokenizer()
    registration = build_split_registration(
        _dataset(200),
        tokenizer,
        fit_candidate_rows=32,
        expected_fit_records=32,
        validation_records=8,
        small_post_bos_positions=1000,
    )
    assert registration["fit"]["small_nested"]["post_bos_positions"] == 1000
    assert registration["fit"]["large_nested"]["post_bos_positions"] > 1000
    assert registration["fit"]["nested_order_is_identical"] is True
    assert registration["validation"]["record_count"] == 8
    assert registration["contains_source_text"] is False
    assert registration["contains_token_ids"] is False
    assert not set(record["record_id"] for record in registration["fit"]["records"]) & set(
        record["record_id"] for record in registration["validation"]["records"]
    )


def test_disjoint_checker_rejects_duplicates_and_overlap() -> None:
    with pytest.raises(AlpacaSplitError, match="duplicate"):
        assert_disjoint_record_sets({"fit": ["r1", "r1"]})
    with pytest.raises(AlpacaSplitError, match="overlaps"):
        assert_disjoint_record_sets({"fit": ["r1"], "validation": ["r1"]})


def test_confirmation_checker_requires_all_public_exclusion_sources() -> None:
    candidate = [public_record_id(9000)]
    with pytest.raises(AlpacaSplitError, match="historical_fitting"):
        validate_confirmation_ids(candidate, exclusion_sources={})
    exclusions = {
        "historical_fitting": (),
        "historical_evaluation": (),
        "current_fitting": (),
        "current_evaluation": (),
    }
    validate_confirmation_ids(candidate, exclusion_sources=exclusions)
    exclusions["current_evaluation"] = candidate
    with pytest.raises(AlpacaSplitError, match="overlaps"):
        validate_confirmation_ids(candidate, exclusion_sources=exclusions)


def test_registration_fails_closed_when_bos_is_not_first() -> None:
    class BadTokenizer(_FakeTokenizer):
        def __call__(self, text, *, add_special_tokens):
            return _Encoding([123] + list(range(max(0, len(text) - 1))))

    with pytest.raises(AlpacaSplitError, match="expected BOS"):
        build_split_registration(
            _dataset(40),
            BadTokenizer(),
            fit_candidate_rows=8,
            expected_fit_records=8,
            validation_records=2,
        )

