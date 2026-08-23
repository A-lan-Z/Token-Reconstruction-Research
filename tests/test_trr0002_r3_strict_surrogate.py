from __future__ import annotations

import torch

from token_reconstruction.strict_base_surrogate import (
    canonical_mapping_bytes,
    exact_input_summary,
    isolated_record_batch_size,
    length_stratified_summary,
    propose_checkpoint_identity,
    right_padded_position_ids,
)


class TinyTokenizer:
    def decode(self, ids, **kwargs):
        del kwargs
        return ",".join(str(value) for value in ids)


def test_identity_proposer_uses_observation_embedding_cosine() -> None:
    embeddings = torch.eye(6, dtype=torch.float32)
    observations = torch.zeros((1, 3, 6), dtype=torch.float32)
    observations[0, 1, 3] = 2.0
    observations[0, 2, 5] = 3.0
    mask = torch.ones((1, 3), dtype=torch.long)
    try:
        propose_checkpoint_identity(
            observations=observations,
            attention_mask=mask,
            normalized_embeddings=embeddings,
            max_k=2,
            chunk=2,
        )
    except Exception as exc:
        assert "constants changed" in str(exc)
    expanded = torch.cat((embeddings, torch.zeros((506, 6))), dim=0)
    result = propose_checkpoint_identity(
        observations=observations,
        attention_mask=mask,
        normalized_embeddings=expanded,
    )
    assert result.candidates.shape == (1, 3, 512)
    assert int(result.candidates[0, 1, 0]) == 3
    assert int(result.candidates[0, 2, 0]) == 5


def test_exact_input_summary_reports_error_concentration_and_text() -> None:
    truth = torch.tensor(
        [[128000, 1, 2, 3, 0], [128000, 4, 5, 0, 0]], dtype=torch.long
    )
    predictions = torch.tensor(
        [[128000, 1, 9, 3, -1], [128000, 4, 5, -1, -1]], dtype=torch.long
    )
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.long)
    summary, rows = exact_input_summary(
        predictions=predictions,
        truth=truth,
        attention_mask=mask,
        tokenizer=TinyTokenizer(),
        source_texts=["1,2,3", "4,5"],
    )
    assert summary.exact_token_records == 1
    assert summary.exact_decoded_text_records == 1
    assert summary.exact_source_text_records == 1
    assert summary.failed_records == 1
    assert summary.errors_in_failed_records == 1
    assert rows[0]["first_error_position"] == 2
    bins = length_stratified_summary(rows, bins=((3, 3), (4, 4)))
    assert bins["3-3"]["exact_token_records"] == 1
    assert bins["4-4"]["exact_token_records"] == 0


def test_all_final_processes_use_the_same_memory_safe_batch() -> None:
    assert isolated_record_batch_size(None) == 4
    assert isolated_record_batch_size("grandmaster_vikhr_heavy_cut4") == 4


def test_right_padded_positions_hold_the_final_valid_index() -> None:
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.long
    )
    positions = right_padded_position_ids(mask)
    assert torch.equal(
        positions,
        torch.tensor([[0, 1, 2, 2, 2], [0, 1, 2, 3, 3]], dtype=torch.long),
    )


def test_canonical_mapping_encoding_is_ordered_and_stable() -> None:
    rows = [
        {
            "opaque_id": "r-1",
            "dataset_index": 7,
            "source_sha256": "a" * 64,
            "token_length": 40,
            "length_bin": "33-64",
        },
        {
            "opaque_id": "r-2",
            "dataset_index": 3,
            "source_sha256": "b" * 64,
            "token_length": 20,
            "length_bin": "16-32",
        },
    ]
    encoded = canonical_mapping_bytes(rows)
    assert encoded.startswith(b"r-1|7|")
    assert encoded.endswith(b"|20|16-32\n")
