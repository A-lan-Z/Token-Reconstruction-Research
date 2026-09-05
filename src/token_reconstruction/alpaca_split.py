"""Public, deterministic Alpaca fit/validation split registration for TRR-0004.

This module deliberately records only public row identities, rendering
fingerprints, lengths, and order digests.  It never returns source text or token
IDs from the registration API.  The historical recipe is reproduced closely
enough to make a controlled nested fit comparison possible while keeping the
retained TRR-0002 A1 state's missing provenance explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


ALPACA_DATASET_ID = "tatsu-lab/alpaca"
ALPACA_SPLIT = "train"
ALPACA_CACHE_REVISION = "dce01c9b08f87459cf36a430d809084718273017"
HISTORICAL_DATASET_FINGERPRINT = "a5048bbbf302198f"
HISTORICAL_SELECTION_SEED = 7
HISTORICAL_FIT_CANDIDATE_ROWS = 1200
HISTORICAL_MIN_FULL_TOKENS = 32
HISTORICAL_MAX_TOKENS = 192
HISTORICAL_MAX_USER_CHARS = 1200
HISTORICAL_MAX_OUTPUT_CHARS = 1200
HISTORICAL_SMALL_POST_BOS_POSITIONS = 5000
DEFAULT_BOS_TOKEN_ID = 128000

# These names are deliberately required by the future-confirmation checker.
# A caller must provide public metadata for every one before a confirmation
# pool can be accepted.  An unavailable source is not equivalent to an empty
# source.
REQUIRED_CONFIRMATION_EXCLUSION_SOURCES = (
    "historical_fitting",
    "historical_evaluation",
    "current_fitting",
    "current_evaluation",
)


class AlpacaSplitError(ValueError):
    """Raised when a public split cannot be established fail-closed."""


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA256 digest used by all task-local split metadata."""

    return hashlib.sha256(payload).hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    """Hash an ordered sequence using one UTF-8 newline-delimited record per line."""

    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def public_record_id(row_index: int, *, dataset_revision: str = ALPACA_CACHE_REVISION) -> str:
    """Return a stable public identity for one cached dataset row."""

    if not isinstance(row_index, int) or row_index < 0:
        raise AlpacaSplitError("row index must be a non-negative integer")
    if not dataset_revision:
        raise AlpacaSplitError("dataset revision is required for a public identity")
    return f"{ALPACA_DATASET_ID}/{ALPACA_SPLIT}@{dataset_revision}:row-{row_index:05d}"


def historical_permutation(dataset_size: int, *, seed: int = HISTORICAL_SELECTION_SEED) -> list[int]:
    """Reproduce the historical ``torch.randperm`` row order on CPU."""

    if not isinstance(dataset_size, int) or dataset_size <= 0:
        raise AlpacaSplitError("dataset size must be positive")
    if not isinstance(seed, int) or seed < 0:
        raise AlpacaSplitError("selection seed must be a non-negative integer")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime environment supplies torch
        raise AlpacaSplitError("torch is required to reproduce the historical permutation") from exc

    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(dataset_size, generator=generator).tolist()


def _text_field(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise AlpacaSplitError(f"Alpaca field {key!r} must be text or null")
    return value


def historical_user_text(record: Mapping[str, Any]) -> str:
    """Build the historical Alpaca user string before its 1,200-char cap."""

    instruction = _text_field(record, "instruction")
    input_text = _text_field(record, "input")
    return instruction + (("\n\n" + input_text) if input_text else "")


def historical_rendered_text(
    record: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_user_chars: int = HISTORICAL_MAX_USER_CHARS,
    max_output_chars: int = HISTORICAL_MAX_OUTPUT_CHARS,
) -> str:
    """Render one public Alpaca row using the historical Llama chat recipe."""

    if max_user_chars <= 0 or max_output_chars <= 0:
        raise AlpacaSplitError("rendering character limits must be positive")
    user = historical_user_text(record)[:max_user_chars]
    output = _text_field(record, "output")[:max_output_chars]
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:  # pragma: no cover - tokenizer-specific error detail
        raise AlpacaSplitError("tokenizer chat-template rendering failed") from exc
    if not isinstance(prompt, str):
        raise AlpacaSplitError("chat template must return text when tokenize=False")
    return prompt + output


def _input_ids(tokenizer: Any, rendered_text: str) -> list[int]:
    try:
        encoded = tokenizer(rendered_text, add_special_tokens=False)
        ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    except Exception as exc:  # pragma: no cover - tokenizer-specific error detail
        raise AlpacaSplitError("tokenizer ID conversion failed") from exc
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
        raise AlpacaSplitError("tokenizer did not return a one-dimensional ID sequence")
    result: list[int] = []
    for token_id in ids:
        if not isinstance(token_id, int):
            raise AlpacaSplitError("tokenizer IDs must be integers")
        result.append(token_id)
    return result


def post_bos_length(
    token_ids: Sequence[int], *, expected_bos_token_id: int = DEFAULT_BOS_TOKEN_ID
) -> int:
    """Validate the historical BOS convention and return post-BOS length."""

    if not token_ids:
        raise AlpacaSplitError("tokenizer returned an empty sequence")
    if token_ids[0] != expected_bos_token_id:
        raise AlpacaSplitError(
            f"expected BOS token {expected_bos_token_id} at position zero, got {token_ids[0]}"
        )
    return len(token_ids) - 1


@dataclass(frozen=True)
class PublicRecordMetadata:
    """Metadata for one public record, without source text or token IDs."""

    row_index: int
    record_id: str
    rendered_sha256: str
    rendered_char_count: int
    full_token_count: int
    post_bos_token_count: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "row_index": self.row_index,
            "record_id": self.record_id,
            "rendered_sha256": self.rendered_sha256,
            "rendered_char_count": self.rendered_char_count,
            "full_token_count": self.full_token_count,
            "post_bos_token_count": self.post_bos_token_count,
        }


def metadata_for_record(
    row_index: int,
    record: Mapping[str, Any],
    tokenizer: Any,
    *,
    dataset_revision: str = ALPACA_CACHE_REVISION,
    max_tokens: int = HISTORICAL_MAX_TOKENS,
    expected_bos_token_id: int = DEFAULT_BOS_TOKEN_ID,
) -> PublicRecordMetadata:
    """Tokenize one public row and retain only non-sensitive registration metadata."""

    if max_tokens <= 0:
        raise AlpacaSplitError("maximum token length must be positive")
    rendered = historical_rendered_text(record, tokenizer)
    raw_ids = _input_ids(tokenizer, rendered)
    ids = raw_ids[:max_tokens]
    post_len = post_bos_length(ids, expected_bos_token_id=expected_bos_token_id)
    return PublicRecordMetadata(
        row_index=row_index,
        record_id=public_record_id(row_index, dataset_revision=dataset_revision),
        rendered_sha256=sha256_bytes(rendered.encode("utf-8")),
        rendered_char_count=len(rendered),
        full_token_count=len(ids),
        post_bos_token_count=post_len,
    )


def _with_position_ranges(
    records: Sequence[PublicRecordMetadata], *, small_limit: int
) -> list[dict[str, int | str]]:
    if small_limit <= 0:
        raise AlpacaSplitError("small nested position limit must be positive")
    result: list[dict[str, int | str]] = []
    cursor = 0
    for record in records:
        start = cursor
        end = start + record.post_bos_token_count
        item = record.as_dict()
        item.update(
            {
                "post_bos_start": start,
                "post_bos_end_exclusive": end,
                "small_post_bos_start": start,
                "small_post_bos_end_exclusive": min(end, small_limit),
                "small_post_bos_count": max(0, min(end, small_limit) - start),
            }
        )
        result.append(item)
        cursor = end
    return result


def _position_lines(records: Sequence[Mapping[str, Any]], *, limit: int | None = None) -> Iterable[str]:
    emitted = 0
    for record in records:
        start = int(record["post_bos_start"])
        end = int(record["post_bos_end_exclusive"])
        record_id = str(record["record_id"])
        for offset in range(end - start):
            if limit is not None and emitted >= limit:
                return
            yield f"{record_id}\t{offset}"
            emitted += 1


def assert_disjoint_record_sets(named_sets: Mapping[str, Iterable[str]]) -> None:
    """Reject duplicates and overlap between named public record sets."""

    seen: dict[str, str] = {}
    for name, records in named_sets.items():
        if not name:
            raise AlpacaSplitError("record-set names must be non-empty")
        local: set[str] = set()
        for record_id in records:
            record_id = str(record_id)
            if record_id in local:
                raise AlpacaSplitError(f"duplicate record {record_id!r} within {name}")
            local.add(record_id)
            prior = seen.get(record_id)
            if prior is not None:
                raise AlpacaSplitError(f"record {record_id!r} overlaps {prior} and {name}")
            seen[record_id] = name


def validate_confirmation_ids(
    candidate_ids: Iterable[str],
    *,
    exclusion_sources: Mapping[str, Iterable[str] | None],
) -> None:
    """Fail closed unless a future confirmation pool is disjoint from all public exclusions.

    ``None`` means the source is unavailable and is unsafe.  An empty iterable
    is accepted only when the caller has explicit public metadata establishing
    that that source contains no records.
    """

    missing = [
        name
        for name in REQUIRED_CONFIRMATION_EXCLUSION_SOURCES
        if name not in exclusion_sources or exclusion_sources[name] is None
    ]
    if missing:
        raise AlpacaSplitError(
            "confirmation exclusions unavailable for: " + ", ".join(missing)
        )
    candidate = [str(record_id) for record_id in candidate_ids]
    assert_disjoint_record_sets({"confirmation": candidate})
    named = {name: exclusion_sources[name] or () for name in REQUIRED_CONFIRMATION_EXCLUSION_SOURCES}
    assert_disjoint_record_sets({"confirmation": candidate, **named})


def build_split_registration(
    dataset: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    dataset_revision: str = ALPACA_CACHE_REVISION,
    dataset_fingerprint: str = HISTORICAL_DATASET_FINGERPRINT,
    selection_seed: int = HISTORICAL_SELECTION_SEED,
    fit_candidate_rows: int = HISTORICAL_FIT_CANDIDATE_ROWS,
    expected_fit_records: int = HISTORICAL_FIT_CANDIDATE_ROWS,
    validation_records: int = 24,
    minimum_full_tokens: int = HISTORICAL_MIN_FULL_TOKENS,
    maximum_tokens: int = HISTORICAL_MAX_TOKENS,
    expected_bos_token_id: int = DEFAULT_BOS_TOKEN_ID,
    small_post_bos_positions: int = HISTORICAL_SMALL_POST_BOS_POSITIONS,
) -> dict[str, Any]:
    """Construct a nested fit stream and disjoint continuation validation metadata."""

    if fit_candidate_rows <= 0 or expected_fit_records <= 0 or validation_records <= 0:
        raise AlpacaSplitError("fit and validation sizes must be positive")
    if len(dataset) < fit_candidate_rows:
        raise AlpacaSplitError("dataset is shorter than the historical fit candidate prefix")
    permutation = historical_permutation(len(dataset), seed=selection_seed)
    fit_candidates = permutation[:fit_candidate_rows]
    fit_records: list[PublicRecordMetadata] = []
    fit_rejected: list[dict[str, int | str]] = []
    for row_index in fit_candidates:
        metadata = metadata_for_record(
            row_index,
            dataset[row_index],
            tokenizer,
            dataset_revision=dataset_revision,
            max_tokens=maximum_tokens,
            expected_bos_token_id=expected_bos_token_id,
        )
        if metadata.full_token_count < minimum_full_tokens:
            fit_rejected.append(
                {
                    "row_index": row_index,
                    "record_id": metadata.record_id,
                    "full_token_count": metadata.full_token_count,
                }
            )
        else:
            fit_records.append(metadata)
    if len(fit_records) != expected_fit_records:
        raise AlpacaSplitError(
            f"historical fit prefix yielded {len(fit_records)} accepted records; "
            f"expected {expected_fit_records}"
        )

    validation: list[PublicRecordMetadata] = []
    validation_rejected: list[dict[str, int | str]] = []
    for row_index in permutation[fit_candidate_rows:]:
        metadata = metadata_for_record(
            row_index,
            dataset[row_index],
            tokenizer,
            dataset_revision=dataset_revision,
            max_tokens=maximum_tokens,
            expected_bos_token_id=expected_bos_token_id,
        )
        if metadata.full_token_count < minimum_full_tokens:
            validation_rejected.append(
                {
                    "row_index": row_index,
                    "record_id": metadata.record_id,
                    "full_token_count": metadata.full_token_count,
                }
            )
            continue
        validation.append(metadata)
        if len(validation) == validation_records:
            break
    if len(validation) != validation_records:
        raise AlpacaSplitError(
            f"continuation yielded {len(validation)} accepted validation records; "
            f"expected {validation_records}"
        )

    fit_ids = [record.record_id for record in fit_records]
    validation_ids = [record.record_id for record in validation]
    assert_disjoint_record_sets({"fit": fit_ids, "validation": validation_ids})
    positioned_fit = _with_position_ranges(fit_records, small_limit=small_post_bos_positions)
    total_post_bos = sum(int(record["post_bos_token_count"]) for record in positioned_fit)
    if total_post_bos < small_post_bos_positions:
        raise AlpacaSplitError(
            f"fit stream has only {total_post_bos} post-BOS positions; "
            f"cannot form {small_post_bos_positions}-position nested prefix"
        )

    return {
        "dataset": {
            "id": ALPACA_DATASET_ID,
            "split": ALPACA_SPLIT,
            "revision": dataset_revision,
            "expected_historical_fingerprint": dataset_fingerprint,
            "row_count": len(dataset),
        },
        "rendering": {
            "user_expression": "instruction + ('\\n\\n' + input if input else '')",
            "user_character_cap": HISTORICAL_MAX_USER_CHARS,
            "output_character_cap": HISTORICAL_MAX_OUTPUT_CHARS,
            "chat_template": "apply_chat_template([{'role':'user','content':user[:1200]}], tokenize=False, add_generation_prompt=True)",
            "rendered_sequence": "chat_template_text + output[:1200]",
            "tokenizer_add_special_tokens": False,
            "maximum_tokens": maximum_tokens,
            "minimum_full_tokens": minimum_full_tokens,
            "expected_bos_token_id": expected_bos_token_id,
            "post_bos_positions_are": "truncated token sequence with the leading BOS removed",
        },
        "selection": {
            "algorithm": "torch.Generator().manual_seed(seed); torch.randperm(dataset_rows, generator=g)",
            "seed": selection_seed,
            "fit_candidate_rows": fit_candidate_rows,
            "fit_candidate_order_sha256": sha256_lines(
                public_record_id(row, dataset_revision=dataset_revision) for row in fit_candidates
            ),
            "fit_accepted_records": len(fit_records),
            "fit_rejected_records": len(fit_rejected),
            "validation_starts_after_candidate_row": fit_candidate_rows,
            "validation_rejected_before_target": len(validation_rejected),
        },
        "fit": {
            "record_count": len(positioned_fit),
            "record_order_sha256": sha256_lines(record["record_id"] for record in positioned_fit),
            "records": positioned_fit,
            "rejected_records": fit_rejected,
            "small_nested": {
                "post_bos_positions": small_post_bos_positions,
                "position_range": [0, small_post_bos_positions],
                "record_count_touched": sum(
                    int(record["small_post_bos_count"]) > 0 for record in positioned_fit
                ),
                "terminal_record_id": next(
                    record["record_id"]
                    for record in positioned_fit
                    if int(record["small_post_bos_end_exclusive"]) >= small_post_bos_positions
                ),
                "position_stream_sha256": sha256_lines(
                    _position_lines(positioned_fit, limit=small_post_bos_positions)
                ),
            },
            "large_nested": {
                "post_bos_positions": total_post_bos,
                "position_range": [0, total_post_bos],
                "record_count_touched": len(positioned_fit),
                "position_stream_sha256": sha256_lines(_position_lines(positioned_fit)),
            },
            "nested_order_is_identical": True,
        },
        "validation": {
            "record_count": len(validation),
            "record_order_sha256": sha256_lines(validation_ids),
            "records": [record.as_dict() for record in validation],
            "rejected_records_before_target": validation_rejected,
            "source": "continuation of the same seed-7 permutation after the 1,200 fit candidates",
        },
        "future_confirmation": {
            "generated": False,
            "status": "NOT_GENERATED",
            "truth_accessed": False,
            "policy": "fail closed until public metadata IDs for every required historical/current fitting/evaluation source are supplied; private evaluator contents are prohibited",
            "required_public_exclusion_sources": list(REQUIRED_CONFIRMATION_EXCLUSION_SOURCES),
            "fit_ids_are_excluded": True,
            "validation_ids_are_excluded": True,
        },
        "contains_source_text": False,
        "contains_token_ids": False,
    }

