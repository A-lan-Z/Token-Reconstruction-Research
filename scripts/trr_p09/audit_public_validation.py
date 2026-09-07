#!/usr/bin/env python3
"""Audit the immutable TRR-P09 public validation preparation.

The audit reads the generated label payloads, re-materializes the frozen
public records through the trusted renderer, and opens only the existing
public_base attention-mask and position-ID tensors.  It hashes the existing
H files for their declared file binding but never loads their activation
tensor.  It writes no files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if str(Path(__file__).resolve().parents[2] / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from safetensors import safe_open

from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0006_capture_public as trr6_capture
from scripts import trr0009_eval_capture as trr9_capture
from scripts.trr_p09.fixed_control_caller import join_public_validation_labels


DOMAINS = ("Finance", "Pile")
STYLE_BY_DOMAIN = {"Finance": "finance", "Pile": "pile"}
EXPECTED_RECORDS = {"Finance": 256, "Pile": 128}
SEQUENCE_TOKENS = 128
HIDDEN_SIZE = 2048


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON object required: {path}")
    return dict(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("experiments/TRR-0009/selection_v2/source_selection.json"))
    parser.add_argument("--observations", type=Path, default=Path("experiments/TRR-0009/evaluation/public_observations_v2/observations.json"))
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.expanduser().resolve()
    runtime = args.runtime_root.expanduser().resolve()
    selection_path = (args.selection if args.selection.is_absolute() else root / args.selection).resolve()
    observations_path = (args.observations if args.observations.is_absolute() else root / args.observations).resolve()
    manifest = _load_json(runtime / "validation_manifest.json")
    rows_doc = _load_json(runtime / "validation_rows.json")
    receipt = _load_json(runtime / "preparation_receipt.json")
    observations = _load_json(observations_path)
    if manifest.get("status") != "PUBLIC_VALIDATION_PREPARED_NO_TRUTH":
        raise RuntimeError("preparation manifest is not a successful public-only artifact")
    if receipt.get("status") != "PASS_PUBLIC_VALIDATION_PREPARED_NO_TRUTH":
        raise RuntimeError("preparation receipt is not a successful public-only run")
    if receipt.get("truth_opened") is not False or receipt.get("model_loaded") is not False or receipt.get("lora_loaded") is not False:
        raise RuntimeError("preparation access boundary changed")
    if len(rows_doc.get("rows", [])) != sum(EXPECTED_RECORDS.values()):
        raise RuntimeError("validation row count changed")
    join = join_public_validation_labels(rows_doc["rows"], required_domains=DOMAINS)
    if join.semantic_sha256 != manifest["label_join"]["semantic_sha256"]:
        raise RuntimeError("label-join digest changed")

    selection, selection_record, selected_rows, _counts = trr9_capture.load_selection(
        selection_path,
        repository_root=root,
        expected_counts={"finance": 256, "pile": 128},
    )
    if selection_record["sha256"] != manifest["input_bindings"]["source_selection"]["sha256"]:
        raise RuntimeError("selection file binding changed")
    pile_paths = tuple(trr9_capture._source_paths(selection, style="pile", root=root))
    finance_paths = tuple(trr9_capture._source_paths(selection, style="finance", root=root))
    tokenizer_path = trr9_capture._tokenizer_path(selection, root=root)
    trr6_capture._validate_source_descriptors(selection, pile_paths=pile_paths, finance_paths=finance_paths, tokenizer_path=tokenizer_path)
    tokenizer = trusted._load_tokenizer(tokenizer_path)
    datasets = {"pile": trusted._load_arrow_dataset(pile_paths), "finance": trusted._load_arrow_dataset(finance_paths)}
    materialized = trr6_capture._materialize_selected(selected_rows, datasets=datasets, tokenizer=tokenizer)
    batches = trr6_capture._batches(materialized)

    payload_results: dict[str, Any] = {}
    h_results: dict[str, Any] = {}
    observation_cells = {
        str(cell["style"]): cell
        for cell in observations["cells"]
        if isinstance(cell, Mapping) and cell.get("condition") == "public_base"
    }
    for domain in DOMAINS:
        style = STYLE_BY_DOMAIN[domain]
        payload_path = runtime / f"{style}_validation_h128.safetensors"
        with safe_open(str(payload_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"token_ids", "attention_mask", "position_ids"}:
                raise RuntimeError(f"payload tensor keys changed for {domain}")
            payload = {key: handle.get_tensor(key).contiguous() for key in handle.keys()}
        expected_batch = batches[style]
        expected = {
            "token_ids": expected_batch.token_ids[:, :SEQUENCE_TOKENS].contiguous().to(torch.int32),
            "attention_mask": expected_batch.attention_mask[:, :SEQUENCE_TOKENS].contiguous().to(torch.uint8),
            "position_ids": expected_batch.position_ids[:, :SEQUENCE_TOKENS].contiguous().to(torch.int64),
        }
        for key in expected:
            if not torch.equal(payload[key], expected[key]):
                raise RuntimeError(f"trusted public rematerialization differs for {domain}/{key}")
        if not payload["attention_mask"].eq(1).all().item():
            raise RuntimeError(f"generated mask is not fully active for {domain}")
        if not torch.equal(payload["position_ids"], torch.arange(SEQUENCE_TOKENS, dtype=torch.int64).expand(len(selected_rows[style]), -1)):
            raise RuntimeError(f"generated positions changed for {domain}")
        if not payload["token_ids"][:, 0].eq(int(trusted.BOS_TOKEN_ID)).all().item():
            raise RuntimeError(f"generated BOS changed for {domain}")
        payload_results[domain] = {
            "records": int(payload["token_ids"].shape[0]),
            "shape": list(payload["token_ids"].shape),
            "keys": sorted(payload),
            "file_sha256": _sha256_file(payload_path),
            "trusted_token_ids_exact": True,
            "attention_mask_exact": True,
            "position_ids_exact": True,
        }

        cell = observation_cells[style]
        observation = cell["observation"]
        h_path = Path(str(observation["path"])).expanduser().resolve()
        if not h_path.is_file() or h_path.is_symlink() or h_path.stat().st_size != int(observation["bytes"]):
            raise RuntimeError(f"public_base H binding is unavailable for {domain}")
        h_file_sha = _sha256_file(h_path)
        if h_file_sha != str(observation["sha256"]):
            raise RuntimeError(f"public_base H file hash changed for {domain}")
        # Only sidecar metadata tensors are opened.  The activation tensor is
        # deliberately absent from this get_tensor set.
        with safe_open(str(h_path), framework="pt", device="cpu") as handle:
            if "activations" not in handle.keys() or "attention_mask" not in handle.keys() or "position_ids" not in handle.keys():
                raise RuntimeError(f"public_base H tensor keys changed for {domain}")
            h_mask = handle.get_tensor("attention_mask").contiguous()
            h_positions = handle.get_tensor("position_ids").contiguous()
        if not torch.equal(h_mask.to(torch.uint8), payload["attention_mask"]):
            raise RuntimeError(f"public_base mask differs from generated labels for {domain}")
        if not torch.equal(h_positions.to(torch.int64), payload["position_ids"]):
            raise RuntimeError(f"public_base positions differ from generated labels for {domain}")
        h_results[domain] = {
            "path": str(h_path),
            "bytes": int(h_path.stat().st_size),
            "declared_sha256": str(observation["sha256"]),
            "file_sha256": h_file_sha,
            "mask_shape": list(h_mask.shape),
            "position_shape": list(h_positions.shape),
            "attention_mask_opened": True,
            "position_ids_opened": True,
            "activations_loaded": False,
            "row_join_exact": True,
        }
    return {
        "status": "PASS_PUBLIC_VALIDATION_AUDIT",
        "records": len(rows_doc["rows"]),
        "records_by_domain": {domain: len(join.rows_by_domain[domain]) for domain in DOMAINS},
        "label_join_sha256": join.semantic_sha256,
        "selection_sha256": selection_record["sha256"],
        "observation_manifest_sha256": _sha256_file(observations_path),
        "payloads": payload_results,
        "public_base_h": h_results,
        "h_payloads_opened_for_audit": True,
        "h_activations_loaded": False,
        "model_loaded": False,
        "public_forward_count": 0,
        "lora_loaded": False,
        "source_text_written": False,
        "truth_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    result = run(_parser().parse_args(list(argv) if argv is not None else None))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
