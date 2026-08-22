"""Deterministic, content-blind assignment of public text records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class SelectedRecord:
    """Public record identity without plaintext or token truth."""

    index: int
    record_id: str
    text_sha256: str
    selection_key: str

    def as_json(self) -> dict[str, int | str]:
        return asdict(self)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record_ids_sha256(records: Iterable[SelectedRecord]) -> str:
    payload = "\n".join(record.record_id for record in records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_record_splits(
    texts: Iterable[str],
    *,
    token_length: Callable[[str], int],
    dataset_revision: str,
    seed: int,
    minimum_tokens: int,
    split_sizes: Mapping[str, int],
) -> dict[str, list[SelectedRecord]]:
    """Assign eligible rows by an index-only hash permutation.

    Text is inspected only for the preregistered minimum token length and its
    public content fingerprint. The ordering key depends on the dataset
    revision, row index, and seed, never on text content or evaluation quality.
    """

    if not dataset_revision or minimum_tokens <= 0 or seed < 0:
        raise ValueError("dataset revision, positive minimum, and seed are required")
    if not split_sizes or any(size <= 0 for size in split_sizes.values()):
        raise ValueError("all split sizes must be positive")

    eligible: list[tuple[str, SelectedRecord]] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError("dataset text rows must be strings")
        if token_length(text) < minimum_tokens:
            continue
        selection_key = _sha256_text(
            f"TRR-0001|{dataset_revision}|row:{index}|seed:{seed}"
        )
        fingerprint = _sha256_text(text)
        record = SelectedRecord(
            index=index,
            record_id=f"pile10k-{index:05d}-{fingerprint[:16]}",
            text_sha256=fingerprint,
            selection_key=selection_key,
        )
        eligible.append((selection_key, record))

    eligible.sort(key=lambda item: (item[0], item[1].index))
    required = sum(split_sizes.values())
    if len(eligible) < required:
        raise ValueError(f"only {len(eligible)} eligible records for {required} required")

    result: dict[str, list[SelectedRecord]] = {}
    offset = 0
    for name, size in split_sizes.items():
        result[name] = [record for _, record in eligible[offset : offset + size]]
        offset += size
    return result
