from __future__ import annotations

from token_reconstruction.records import record_ids_sha256, select_record_splits


def test_selection_is_deterministic_disjoint_and_length_gated() -> None:
    texts = ["x" * size for size in range(1, 12)]
    kwargs = {
        "token_length": len,
        "dataset_revision": "a" * 40,
        "seed": 17,
        "minimum_tokens": 4,
        "split_sizes": {"aux": 3, "dev": 2, "blind": 2},
    }
    first = select_record_splits(texts, **kwargs)
    second = select_record_splits(texts, **kwargs)

    assert first == second
    flat = [record for records in first.values() for record in records]
    assert len({record.index for record in flat}) == len(flat)
    assert all(len(texts[record.index]) >= 4 for record in flat)
    assert len(record_ids_sha256(first["blind"])) == 64


def test_selection_changes_with_seed() -> None:
    texts = [f"record-{index}" for index in range(20)]
    common = {
        "token_length": len,
        "dataset_revision": "b" * 40,
        "minimum_tokens": 3,
        "split_sizes": {"blind": 5},
    }
    assert select_record_splits(texts, seed=1, **common) != select_record_splits(
        texts, seed=2, **common
    )
