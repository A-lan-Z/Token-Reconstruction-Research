#!/usr/bin/env python3
"""Prepare an eight-record, no-truth qualification view from retained observations.

Only the public activation, mask, and position tensors are requested from the
already-opened TRR-0005 observations.  The script never reads token IDs,
source text, labels, or any private truth sidecar.  It writes a bounded H=128
view and a source-paired observation manifest for the task-local prediction
runner; this is qualification evidence, not the TRR-0006 study panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts import trr0006_prediction_contract as contract

QUALIFICATION_RECORDS = 8
QUALIFICATION_SCHEMA = "token-reconstruction.trr0006-qualification-observation-preparation.v1"
SOURCE_OBSERVATION_RELATIVE = Path(
    "experiments/TRR-0005/fresh_confirmation_v1/panel_capture_v2/observations"
)


def _sha256_file(path: Path) -> str:
    return contract.sha256_file(path)


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _record_ids_digest(style: str) -> str:
    # The qualification view uses row-ordinal commitments only.  These are
    # deliberately not asserted to be TRR-0006 study source IDs.
    ids = [f"retained_trr5_fixture:{style}:{index:04d}" for index in range(QUALIFICATION_RECORDS)]
    encoded = json.dumps(ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_path(source_root: Path, cell_id: str) -> Path:
    return source_root / SOURCE_OBSERVATION_RELATIVE / f"{cell_id}.safetensors"


def prepare(*, source_root: Path, repository_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    repository_root = repository_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not source_root.is_dir() or source_root.is_symlink():
        raise RuntimeError(f"source root is unavailable: {source_root}")
    if output_root.is_symlink():
        raise RuntimeError(f"qualification output is a symlink: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, Any]] = []
    source_records: dict[str, Any] = {}
    for cell_id in contract.CELL_ORDER:
        source_path = _source_path(source_root, cell_id)
        if source_path.is_symlink() or not source_path.is_file():
            raise RuntimeError(f"retained source observation is unavailable: {source_path}")
        source_record = _file_record(source_path)
        source_records[cell_id] = source_record
        destination = output_root / "observations" / f"{cell_id}.safetensors"
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"qualification observation is create-only: {destination}")
        with safe_open(str(source_path), framework="pt", device="cpu") as handle:
            required = {"activations", "attention_mask", "position_ids"}
            if not required.issubset(set(handle.keys())):
                raise RuntimeError(f"retained observation lacks required tensors: {source_path}")
            # Deliberately request only these three public tensors; token_ids is
            # present in the old artifact but is never loaded.
            activations = handle.get_slice("activations")[:QUALIFICATION_RECORDS].contiguous()
            mask = handle.get_slice("attention_mask")[:QUALIFICATION_RECORDS].contiguous()
            positions = handle.get_slice("position_ids")[:QUALIFICATION_RECORDS].contiguous()
        if tuple(activations.shape) != (QUALIFICATION_RECORDS, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE):
            raise RuntimeError(f"retained activation geometry changed: {source_path}")
        if tuple(mask.shape) != (QUALIFICATION_RECORDS, contract.STORED_SEQUENCE_TOKENS):
            raise RuntimeError(f"retained mask geometry changed: {source_path}")
        if tuple(positions.shape) != tuple(mask.shape):
            raise RuntimeError(f"retained position geometry changed: {source_path}")
        if activations.dtype != torch.bfloat16:
            raise RuntimeError(f"retained activation dtype changed: {source_path}")
        if mask.dtype not in (torch.bool, torch.uint8):
            raise RuntimeError(f"retained mask dtype changed: {source_path}")
        if mask.dtype == torch.uint8 and ((mask != 0) & (mask != 1)).any().item():
            raise RuntimeError(f"retained mask is not binary: {source_path}")
        if not mask.to(torch.bool).all().item():
            raise RuntimeError(f"retained qualification clip is not full: {source_path}")
        expected_positions = torch.arange(contract.STORED_SEQUENCE_TOKENS, dtype=torch.long).unsqueeze(0).expand_as(positions)
        if not torch.equal(positions.to(torch.long), expected_positions):
            raise RuntimeError(f"retained positions are not 0..127: {source_path}")
        metadata = {
            "schema": "token-reconstruction.trr0006-qualification-observation.v1",
            "task_id": contract.TASK_ID,
            "cell_id": cell_id,
            "records": str(QUALIFICATION_RECORDS),
            "shape": str([QUALIFICATION_RECORDS, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE]),
            "hidden_size": str(contract.HIDDEN_SIZE),
            "capture_batch_records": str(contract.CAPTURE_BATCH_RECORDS),
            "capture_sequence_tokens": str(contract.CAPTURE_SEQUENCE_TOKENS),
            "stored_sequence_tokens": str(contract.STORED_SEQUENCE_TOKENS),
            "scored_post_bos_tokens": str(contract.SCORED_POST_BOS_TOKENS),
            "public_full_forward": "true",
            "producer_only_lora": str(cell_id.endswith("public_lora_2601")).lower(),
            "truth_opened": "false",
            "source_text_loaded": "false",
            "target_labels_loaded": "false",
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {"activations": activations, "attention_mask": mask, "position_ids": positions},
            str(destination),
            metadata=metadata,
        )
        style, condition = cell_id.split("__", 1)
        cells.append(
            {
                "cell_id": cell_id,
                "style": style,
                "condition": condition,
                "records": QUALIFICATION_RECORDS,
                "record_id_namespace": "retained_trr5_fixture_row_ordinal",
                "record_ids_sha256": _record_ids_digest(style),
                "source_observation": source_record,
                "observation": {
                    **_file_record(destination),
                    "shape": [QUALIFICATION_RECORDS, contract.STORED_SEQUENCE_TOKENS, contract.HIDDEN_SIZE],
                    "stored_sequence_tokens": contract.STORED_SEQUENCE_TOKENS,
                    "scored_post_bos_tokens": contract.SCORED_POST_BOS_TOKENS,
                    "capture_batch_records": contract.CAPTURE_BATCH_RECORDS,
                    "capture_sequence_tokens": contract.CAPTURE_SEQUENCE_TOKENS,
                    "activations_key": "activations",
                    "attention_mask_key": "attention_mask",
                    "position_ids_key": "position_ids",
                    "public_full_forward": True,
                    "producer_only_lora": condition == "public_lora_2601",
                },
            }
        )
    manifest = {
        "schema": contract.OBSERVATION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "FROZEN_PUBLIC_OBSERVATIONS_NO_TRUTH",
        "qualification_only": True,
        "records_per_domain": QUALIFICATION_RECORDS,
        "cell_order": list(contract.CELL_ORDER),
        "record_id_namespace": "retained_trr5_fixture_row_ordinal",
        "source_panel": "TRR-0005 opened public observations; no new source selection",
        "cells": cells,
        "source_observation_records": source_records,
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }
    manifest_path = output_root / "observation_manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        raise RuntimeError(f"qualification manifest is create-only: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema": QUALIFICATION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": "QUALIFICATION_OBSERVATIONS_PREPARED_NO_TRUTH",
        "records_per_domain": QUALIFICATION_RECORDS,
        "manifest": _file_record(manifest_path),
        "cells": source_records,
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "candidate_arrays_persisted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        source_root=args.source_root,
        repository_root=args.repository_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
