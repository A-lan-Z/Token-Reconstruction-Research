"""CPU-only integrity tests for the TRR-P01 prototype mechanism.

These fixtures exercise deterministic table construction and the optional
reference correction without loading a real checkpoint or evaluator data.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from token_reconstruction.trr_p01.boundary_prototype import (
    BOUNDARY_TABLE_SCHEMA,
    BOS_TOKEN_ID,
    PrototypeError,
    PrototypeTable,
    nearest_embedding,
    apply_reference_correction,
)


def _small_table() -> PrototypeTable:
    prototypes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return PrototypeTable(
        prototypes,
        model_id="fake-model",
        model_revision="fake-revision",
        cut_depth=4,
    )


def test_nearest_is_invariant_to_query_and_prototype_chunking() -> None:
    table = _small_table()
    queries = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    for metric in ("cosine", "l2"):
        whole = table.nearest(
            queries, metric=metric, query_chunk_size=8, prototype_chunk_size=8
        )
        tiled = table.nearest(
            queries, metric=metric, query_chunk_size=1, prototype_chunk_size=1
        )
        assert torch.equal(whole.predictions, tiled.predictions)
        assert torch.equal(whole.runner_up_scores, tiled.runner_up_scores)
        assert torch.equal(whole.scores, tiled.scores)
        assert torch.equal(whole.margins, tiled.margins)

    # IDs 0 and 1 tie exactly; the declared order must win in every chunking.
    tied = table.nearest(
        torch.tensor([[1.0, 0.0, 0.0]]),
        metric="cosine",
        query_chunk_size=1,
        prototype_chunk_size=1,
    )
    assert tied.predictions.tolist() == [0]
    assert tied.runner_up_scores.tolist() == [1.0]


class _FakeEmbedding(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.num_embeddings = vocab_size


class _RecordingPrefix(nn.Module):
    def __init__(self, vocab_size: int = 5) -> None:
        super().__init__()
        self.embed_tokens = _FakeEmbedding(vocab_size)
        self.anchor = nn.Parameter(torch.zeros(1))
        self.calls: list[torch.Tensor] = []

    @torch.inference_mode()
    def forward_full(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.calls.append(input_ids.detach().cpu().clone())
        ids = input_ids[:, 1].to(torch.float32)
        values = torch.stack((ids, ids + 1.0, ids + 2.0), dim=1)
        output = torch.zeros(
            input_ids.shape[0], 2, 3, dtype=torch.bfloat16, device=input_ids.device
        )
        output[:, 1] = values.to(output.dtype)
        return output


def test_build_uses_bos_full_vocab_and_counts_partial_batch_tokens() -> None:
    prefix = _RecordingPrefix(vocab_size=5)
    table, stats = PrototypeTable.build(
        prefix, batch_size=2, device=torch.device("cpu"), return_stats=True
    )

    assert table.prototypes.dtype == torch.bfloat16
    assert table.prototypes.shape == (5, 3)
    assert stats.vocab_size == 5
    assert stats.hidden_size == 3
    assert stats.batch_size == 2
    assert stats.forward_calls == 3
    # Each of five [BOS, v] inputs contributes two committed tokens.
    assert stats.committed_tokens == 2 * 5
    assert stats.output_dtype == "bfloat16"
    observed = torch.cat(prefix.calls, dim=0)
    expected = torch.stack(
        (
            torch.full((5,), BOS_TOKEN_ID, dtype=torch.long),
            torch.arange(5, dtype=torch.long),
        ),
        dim=1,
    )
    assert torch.equal(observed, expected)
    assert torch.equal(
        table.prototypes[3], torch.tensor([3.0, 4.0, 5.0], dtype=torch.bfloat16)
    )


def test_table_artifact_round_trip_and_truth_state_rejection(tmp_path: Path) -> None:
    table = _small_table()
    destination = tmp_path / "table.safetensors"
    digest = table.save(destination)
    assert len(digest) == 64
    loaded = PrototypeTable.load(
        destination,
        expected_model_id="fake-model",
        expected_model_revision="fake-revision",
        expected_cut_depth=4,
        expected_vocab_size=5,
        expected_hidden_size=3,
    )
    assert torch.equal(loaded.prototypes, table.prototypes)
    with pytest.raises(PrototypeError, match="already exists"):
        table.save(destination)

    tampered = tmp_path / "truth-opened.safetensors"
    save_file(
        {"prototypes": table.prototypes},
        tampered,
        metadata={
            "schema": BOUNDARY_TABLE_SCHEMA,
            "model_id": "fake-model",
            "model_revision": "fake-revision",
            "cut_depth": "4",
            "bos_token_id": str(BOS_TOKEN_ID),
            "vocab_size": "5",
            "hidden_size": "3",
            "dtype": "float32",
            "construction": "public_prefix([BOS,v])[1]",
            "truth_opened": "true",
        },
    )
    with pytest.raises(PrototypeError, match="truth state"):
        PrototypeTable.load(tampered)


def test_table_and_queries_reject_nonfinite_values() -> None:
    with pytest.raises(PrototypeError, match="non-finite"):
        PrototypeTable(torch.tensor([[float("nan"), 0.0]]))
    table = PrototypeTable(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(PrototypeError, match="non-finite"):
        table.nearest(torch.tensor([[float("inf"), 0.0]]))


class _FakeCache:
    def __init__(
        self, *, kind: str = "persistent", length: int = 0, tokens: list[int] | None = None
    ) -> None:
        self.kind = kind
        self.length = length
        self.tokens = list(tokens or [])

    def __deepcopy__(self, memo: dict[int, object]) -> "_FakeCache":
        del memo
        return _FakeCache(kind="probe", length=self.length, tokens=self.tokens)


class _RecordingCorrectionPrefix(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.calls: list[dict[str, object]] = []

    def new_cache(self) -> _FakeCache:
        return _FakeCache()

    @torch.inference_mode()
    def run_cached(
        self, input_ids: torch.Tensor, cache: _FakeCache, start_pos: int
    ) -> torch.Tensor:
        assert input_ids.shape == (1, 1)
        assert cache.length == start_pos
        token = int(input_ids[0, 0])
        cache.tokens.append(token)
        cache.length += 1
        self.calls.append(
            {
                "kind": cache.kind,
                "token": token,
                "history": list(cache.tokens),
            }
        )
        # The public reference output equals its prototype, so this fixture's
        # correction offset is exactly zero at every position.
        return torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)


def test_reference_correction_counts_probes_and_uses_only_predicted_prefix() -> None:
    prototypes = PrototypeTable(
        torch.tensor(
            [
                [0.0, 1.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )
    prefix = _RecordingCorrectionPrefix()
    observations = torch.tensor(
        [[[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32
    )

    result = apply_reference_correction(
        observations=observations,
        public_prefix=prefix,
        prototypes=prototypes,
        reference_token=2,
        device=torch.device("cpu"),
    )

    assert result.predictions.tolist() == [[BOS_TOKEN_ID, 3, 3]]
    assert result.reference_evaluations == 2
    assert result.persistent_cache_commits == 3
    assert result.probe_cache_commits == 2
    assert torch.equal(result.offsets, torch.zeros_like(result.offsets))

    persistent = [call for call in prefix.calls if call["kind"] == "persistent"]
    probes = [call for call in prefix.calls if call["kind"] == "probe"]
    assert [call["token"] for call in persistent] == [BOS_TOKEN_ID, 3, 3]
    assert [call["history"] for call in probes] == [
        [BOS_TOKEN_ID, 2],
        [BOS_TOKEN_ID, 3, 2],
    ]
    # The reference token is sent only to copied probe caches, never
    # committed to the persistent reconstruction prefix.
    assert all(2 not in call["history"] for call in persistent)


def test_raw_embedding_lookup_shares_stable_scoring_contract() -> None:
    table = _small_table()
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.8, 0.0]], dtype=torch.float32)
    for metric in ("cosine", "l2"):
        table_result = table.nearest(
            queries,
            metric=metric,
            query_chunk_size=1,
            prototype_chunk_size=2,
        )
        embedding_result = nearest_embedding(
            queries,
            table.prototypes,
            metric=metric,
            query_chunk_size=1,
            prototype_chunk_size=2,
        )
        assert torch.equal(table_result.predictions, embedding_result.predictions)
        assert torch.equal(table_result.scores, embedding_result.scores)
        assert torch.equal(table_result.runner_up_scores, embedding_result.runner_up_scores)
        assert torch.equal(table_result.margins, embedding_result.margins)


def test_build_output_is_equivalent_across_batch_sizes() -> None:
    one_at_a_time = _RecordingPrefix(vocab_size=5)
    batched = _RecordingPrefix(vocab_size=5)
    first = PrototypeTable.build(
        one_at_a_time, batch_size=1, device=torch.device("cpu")
    )
    second = PrototypeTable.build(
        batched, batch_size=3, device=torch.device("cpu")
    )
    assert torch.equal(first.prototypes, second.prototypes)


def test_table_load_rejects_metadata_shape_mismatch(tmp_path: Path) -> None:
    table = _small_table()
    destination = tmp_path / "shape-mismatch.safetensors"
    save_file(
        {"prototypes": table.prototypes},
        destination,
        metadata={
            "schema": BOUNDARY_TABLE_SCHEMA,
            "model_id": "fake-model",
            "model_revision": "fake-revision",
            "cut_depth": "4",
            "bos_token_id": str(BOS_TOKEN_ID),
            "vocab_size": "4",
            "hidden_size": "3",
            "dtype": "float32",
            "construction": "public_prefix([BOS,v])[1]",
            "truth_opened": "false",
        },
    )
    with pytest.raises(PrototypeError, match="shape"):
        PrototypeTable.load(destination)
