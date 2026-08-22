from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from token_reconstruction.dual_benchmark import (
    BOS_TOKEN_ID,
    METHOD_IDS,
    SETUP_IDS,
    causal_k16,
    propose_k16,
    score_predictions,
    stable_candidate_order,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class NormalizeIdentity(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value.float(), dim=-1)


class FakeCache:
    def __init__(self) -> None:
        self.length = 0
        self.batch = 1

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.batch *= repeats


class FakePrecut(torch.nn.Module):
    def new_cache(self) -> FakeCache:
        return FakeCache()

    def run_cached(
        self,
        input_ids: torch.Tensor,
        cache: FakeCache,
        start_pos: int,
    ) -> torch.Tensor:
        assert cache.length == start_pos
        assert input_ids.shape[0] == cache.batch
        hidden = torch.zeros(
            (*input_ids.shape, 2048),
            dtype=torch.float32,
            device=input_ids.device,
        )
        for row in range(input_ids.shape[0]):
            for column in range(input_ids.shape[1]):
                hidden[row, column, int(input_ids[row, column]) % 2048] = 1.0
        cache.length += input_ids.shape[1]
        return hidden


def test_registry_is_exact_cartesian_product() -> None:
    registry = json.loads(
        (REPOSITORY_ROOT / "research" / "dual_benchmark_registry.json").read_text(
            encoding="utf-8"
        )
    )
    setup_ids = tuple(item["id"] for item in registry["setups"])
    method_ids = tuple(item["id"] for item in registry["methods"])
    cells = {
        (item["setup_id"], item["method_id"])
        for item in registry["required_cells"]
    }
    assert setup_ids == SETUP_IDS
    assert method_ids == METHOD_IDS
    assert len(cells) == len(registry["required_cells"])
    assert cells == {
        (setup_id, method_id)
        for setup_id in SETUP_IDS
        for method_id in METHOD_IDS
    }
    assert registry["missing_cell_disposition"] == "comparison-incomplete"
    assert registry["cross_setup_pooling"] is False
    protocol = (
        REPOSITORY_ROOT / registry["protocol_path"]
    ).read_text(encoding="utf-8")
    for identifier in (*SETUP_IDS, *METHOD_IDS):
        assert identifier in protocol


def test_stable_candidate_order_breaks_score_ties_by_token_id() -> None:
    ids = torch.tensor([[9, 3, 7], [5, 4, 6]], dtype=torch.long)
    scores = torch.tensor([[0.2, 0.2, 0.1], [0.4, 0.5, 0.4]])
    ordered_ids, ordered_scores = stable_candidate_order(ids, scores)
    assert ordered_ids.tolist() == [[3, 9, 7], [4, 5, 6]]
    assert ordered_scores.tolist() == [
        [scores[0, 1].item(), scores[0, 0].item(), scores[0, 2].item()],
        [scores[1, 1].item(), scores[1, 0].item(), scores[1, 2].item()],
    ]


def test_variable_geometry_k16_proposal_and_causal_selection() -> None:
    observations = torch.zeros((2, 4, 2048), dtype=torch.float32)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long)
    positions = mask.cumsum(1).sub(1).clamp_min(0)
    desired = torch.tensor([[3, 5, 7], [2, 4, 0]], dtype=torch.long)
    for row in range(2):
        for position in range(1, 4):
            if mask[row, position]:
                observations[row, position, desired[row, position - 1]] = 1.0
    embedding_table = F.normalize(torch.eye(20, 2048), dim=-1)
    direct, candidates, _, _ = propose_k16(
        observations=observations,
        attention_mask=mask,
        inverse=NormalizeIdentity(),
        embedding_table=embedding_table,
    )
    assert direct[0, 1:].tolist() == [3, 5, 7]
    assert direct[1].tolist() == [BOS_TOKEN_ID, 2, 4, -1]

    causal, scores, _, simulations = causal_k16(
        observations=observations,
        attention_mask=mask,
        position_ids=positions,
        candidates=candidates,
        precut=FakePrecut(),
        device=torch.device("cpu"),
    )
    assert torch.equal(causal, direct)
    assert simulations == 5 * 16
    assert torch.isfinite(scores[mask.to(torch.bool) & positions.gt(0)]).all()


def test_score_counts_abstention_as_end_to_end_error() -> None:
    truth = torch.tensor(
        [
            [BOS_TOKEN_ID, 1, 2, 3],
            [BOS_TOKEN_ID, 4, 5, 0],
        ],
        dtype=torch.long,
    )
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long)
    predictions = torch.tensor(
        [
            [BOS_TOKEN_ID, 1, -1, -1],
            [BOS_TOKEN_ID, 4, 9, -1],
        ],
        dtype=torch.long,
    )
    metrics, rows = score_predictions(
        predictions=predictions,
        truth=truth,
        attention_mask=mask,
        record_ids=["a", "b"],
    )
    assert metrics["scored_tokens"] == 5
    assert metrics["covered_tokens"] == 3
    assert metrics["correct_tokens"] == 2
    assert metrics["token_accuracy"] == 0.4
    assert metrics["coverage"] == 0.6
    assert metrics["selective_accuracy"] == 2 / 3
    assert metrics["exact_records"] == 0
    assert [row["record_id"] for row in rows] == ["a", "b"]

