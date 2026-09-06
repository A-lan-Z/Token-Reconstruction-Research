from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.trr_p06 import run_anchor as anchor


def _selection(path: Path) -> None:
    rows = {
        domain: [
            {
                "record_id": f"{domain}-record-{index}",
                "final_sequence_sha256": f"{index + (1 if domain == 'finance' else 1000):064x}",
            }
            for index in range(256)
        ]
        for domain in anchor.DOMAINS
    }
    path.write_text(
        json.dumps(
            {
                "schema": "token-reconstruction.trr-p06-source-selection.v1",
                "task_id": anchor.TASK_ID,
                "selection_rule": {"records": rows},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_anchor_selection_hash_is_the_ordered_first64_subset(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    _selection(path)
    selected = anchor._load_selection(path, root=tmp_path)
    for domain in anchor.DOMAINS:
        assert selected["subset_ids"][domain] == selected["record_ids"][domain][:64]
        assert selected["subset_record_ids_sha256"][domain] == anchor._json_digest(selected["subset_ids"][domain])
        assert len(selected["subset_sequence_hashes"][domain]) == 64


def test_anchor_reads_record_hash_from_manifest_cell(tmp_path: Path) -> None:
    manifest = {
        "cells": [
            {
                "cell_id": "pile__public_base",
                "record_ids_sha256": "a" * 64,
                "observation": {"path": "observations/pile.safetensors"},
            }
        ]
    }
    cell = anchor._observation_cell(manifest, cell_id="pile__public_base")
    assert cell["record_ids_sha256"] == "a" * 64
    assert anchor._observation_descriptor(manifest, cell_id="pile__public_base")["path"] == "observations/pile.safetensors"


def test_anchor_normalization_keeps_bos_and_padding_without_rewriting_active_ids() -> None:
    mask = torch.tensor([True, True, True] + [False] * 125)
    prediction = torch.tensor([999, 17, 23] + [-1] * 125)
    normalized = anchor._normalize_prediction(prediction, mask)
    assert normalized[:4].tolist() == [anchor.BOS_TOKEN_ID, 17, 23, anchor.INVALID_TOKEN_ID]
    assert normalized.shape == (128,)


def test_anchor_descriptor_matches_score_frozen_contract(tmp_path: Path) -> None:
    prediction = tmp_path / "anchor.safetensors"
    prediction.write_bytes(b"immutable prediction placeholder")
    selection = {
        "subset_record_ids_sha256": {domain: "a" * 64 for domain in anchor.DOMAINS},
        "subset_sequence_hashes": {domain: ["b" * 64] * 64 for domain in anchor.DOMAINS},
    }
    observation = {
        "attention_mask_sha256": "c" * 64,
        "position_ids_sha256": "d" * 64,
        "file": {"sha256": "e" * 64},
    }
    descriptor = anchor._anchor_descriptor(
        domain="pile",
        prediction_path=prediction,
        root=tmp_path,
        selection=selection,
        observation=observation,
        timing={"warmup_runs_per_record": 1, "measured_runs_per_record": 1},
        state_record={"sha256": anchor.ANCHOR_STATE_SHA256},
    )
    assert descriptor["method_id"] == "frozen_a1_a2_k256"
    assert descriptor["subset"] == "first64_public_base"
    assert descriptor["shape"] == [64, 128]
    assert descriptor["scored_post_bos_tokens"] == 127
    assert descriptor["anchor_subset_record_ids_sha256"] == "a" * 64
    assert descriptor["state_sha256"] == anchor.ANCHOR_STATE_SHA256
    assert descriptor["truth_opened"] is False
    assert descriptor["candidate_arrays_persisted"] is False


def test_anchor_contract_has_no_student_or_fallback_path() -> None:
    assert anchor.ANCHOR_METHOD_ID == "frozen_a1_a2_k256"
    assert anchor.PROPOSAL_K == 512
    assert anchor.SELECTOR_K == 256
    assert anchor.ANCHOR_SUBSET == "first64_public_base"
