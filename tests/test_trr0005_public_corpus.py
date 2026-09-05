from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import pytest
import torch
from safetensors.torch import save_file

from scripts import trr0005_prepare_public_corpus as prepare
from token_reconstruction.trr0005_public_corpus import (
    BOS_TOKEN_ID,
    CONTROLLED_REPLACEMENTS_PER_RECORD,
    MAX_SEQUENCE_LENGTH,
    PAD_TOKEN_ID,
    POST_BOS_POSITION_COUNT,
    SOURCE_PARTITIONS,
    TRR0005CorpusError,
    apply_replacements,
    expected_sampler_exposure,
    length_multiset,
    load_trr4_length_slots,
    replacement_positions,
    select_public_token_ids,
    source_record_id,
    token_frequency_summary,
    validate_partition_index,
)


TRR4_LENGTHS = Path("experiments/TRR-0004/alpaca_split_plan.json")


def test_frozen_length_vector_and_post_bos_exposure_are_exact() -> None:
    slots = load_trr4_length_slots(TRR4_LENGTHS)
    assert len(slots) == 1200
    assert sum(slot.post_bos_token_count for slot in slots) == POST_BOS_POSITION_COUNT
    assert sum(slot.post_bos_token_count >= 128 for slot in slots) >= 120
    exposure = expected_sampler_exposure()
    assert exposure["position_scope"] == "post_bos_only"
    assert exposure["post_bos_positions"] == POST_BOS_POSITION_COUNT
    assert exposure["bos_positions_drawn"] == 0
    assert exposure["draws"] == 512 * 3000
    assert exposure["seed"] == 4005
    assert exposure["expected_draws_per_post_bos_position"] == pytest.approx(
        1536000 / POST_BOS_POSITION_COUNT
    )


def test_controlled_slots_prefer_128_without_changing_global_vector() -> None:
    slots = load_trr4_length_slots(TRR4_LENGTHS)
    allocation = prepare._slot_allocation(slots, seed=5005)
    controlled = allocation["controlled_pile_context"] + allocation["controlled_finance_context"]
    assert len(controlled) == 120
    assert len(set(controlled)) == 120
    assert all(slots[index].post_bos_token_count >= 128 for index in controlled)
    assert sorted(index for values in allocation.values() for index in values) == list(range(1200))
    assigned_lengths = [slots[index].post_bos_token_count for values in allocation.values() for index in values]
    assert length_multiset(slots) == Counter(assigned_lengths)


def test_partition_guard_rejects_reserved_rows_before_dataset_access() -> None:
    touched: list[int] = []

    class Dataset:
        def __getitem__(self, index: int):
            touched.append(index)
            raise AssertionError("reserved row was accessed")

    with pytest.raises(TRR0005CorpusError, match="outside the declared fit partition"):
        prepare._candidate(
            "pile",
            Dataset(),
            7000,
            object(),
            deadline=prepare._Deadline(time.monotonic(), 30),
            excluded_ids=set(),
            excluded_row_keys=set(),
            excluded_hashes=set(),
        )
    assert touched == []
    validate_partition_index("pile", 2000, role="fit")
    with pytest.raises(TRR0005CorpusError, match="outside the declared holdout partition"):
        validate_partition_index("finance", 2000, role="holdout")
    validate_partition_index("finance", 12000, role="holdout")


def test_public_token_selection_is_deterministic_and_excludes_special_ids() -> None:
    legacy = {token_id: 1 for token_id in range(1, 101)}
    public = {token_id: 1 for token_id in range(1, 2202)}
    public[BOS_TOKEN_ID] = 100
    public[PAD_TOKEN_ID] = 100
    selected = select_public_token_ids(
        legacy,
        public,
        special_token_ids=(BOS_TOKEN_ID, PAD_TOKEN_ID),
        target_count=2000,
        min_legacy_absent=1800,
    )
    assert len(selected) == 2000
    assert len(set(selected)) == 2000
    assert BOS_TOKEN_ID not in selected and PAD_TOKEN_ID not in selected
    assert sum(token_id not in legacy for token_id in selected) >= 1800
    assert selected == select_public_token_ids(
        legacy,
        public,
        special_token_ids=(BOS_TOKEN_ID, PAD_TOKEN_ID),
        target_count=2000,
        min_legacy_absent=1800,
    )


def test_post_bos_frequency_summary_can_preserve_special_id_values() -> None:
    summary = token_frequency_summary(
        [BOS_TOKEN_ID, PAD_TOKEN_ID, 17],
        exclude_special_values=False,
    )
    assert summary["token_rows"] == 3
    assert summary["distinct_token_ids"] == 3
    assert summary["token_frequency_by_id"][str(BOS_TOKEN_ID)] == 1
    default_summary = token_frequency_summary([BOS_TOKEN_ID, PAD_TOKEN_ID, 17])
    assert default_summary["token_rows"] == 1


def test_replacement_geometry_keeps_bos_and_valid_post_bos_ids() -> None:
    source = (BOS_TOKEN_ID,) + tuple(range(1, 65))
    positions = replacement_positions(
        source,
        target_post_bos_token_count=64,
        count=CONTROLLED_REPLACEMENTS_PER_RECORD,
        structural_token_ids=(BOS_TOKEN_ID, PAD_TOKEN_ID),
    )
    constructed = apply_replacements(
        source,
        positions,
        tuple(1000 + index for index in range(CONTROLLED_REPLACEMENTS_PER_RECORD)),
        target_post_bos_token_count=64,
        structural_token_ids=(BOS_TOKEN_ID, PAD_TOKEN_ID),
    )
    assert len(positions) == CONTROLLED_REPLACEMENTS_PER_RECORD
    assert len(constructed) == 65
    assert constructed[0] == BOS_TOKEN_ID
    assert all(0 <= token_id < 128256 for token_id in constructed)


def test_masked_legacy_frequency_counts_exact_post_bos_positions(tmp_path: Path) -> None:
    slots = load_trr4_length_slots(TRR4_LENGTHS)
    token_ids = torch.full((1200, MAX_SEQUENCE_LENGTH), PAD_TOKEN_ID, dtype=torch.int32)
    mask = torch.zeros((1200, MAX_SEQUENCE_LENGTH), dtype=torch.uint8)
    token_ids[:, 0] = BOS_TOKEN_ID
    mask[:, 0] = 1
    for row, slot in enumerate(slots):
        mask[row, 1 : slot.post_bos_token_count + 1] = 1
        token_ids[row, 1 : slot.post_bos_token_count + 1] = 17
    # An in-record special value remains part of the valid post-BOS stream.
    token_ids[0, 1] = BOS_TOKEN_ID
    path = tmp_path / "legacy_labels.safetensors"
    save_file({"token_ids": token_ids, "attention_mask": mask}, str(path))
    report: dict[str, object] = {}
    frequencies = prepare._legacy_frequency(
        path,
        deadline=prepare._Deadline(time.monotonic(), 30),
        report=report,
    )
    assert sum(frequencies.values()) == POST_BOS_POSITION_COUNT
    assert frequencies[BOS_TOKEN_ID] == 1
    assert frequencies[17] == POST_BOS_POSITION_COUNT - 1
    assert report["valid_post_bos_positions"] == POST_BOS_POSITION_COUNT
    assert report["distinct_post_bos_labels"] == 2
    assert report["distinct_labels_including_bos"] == 2
    assert report["bos_excluded_by_position_mask"] is True
    assert report["values_filtered_by_token_id"] is False


def test_design_only_mode_does_not_require_public_cache_paths(capsys: pytest.CaptureFixture[str]) -> None:
    assert prepare.main(["--design", "experiments/TRR-0005/corpus_design.json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "DESIGN_ONLY_NO_CACHE_READ"
    assert payload["post_bos_positions"] == POST_BOS_POSITION_COUNT
