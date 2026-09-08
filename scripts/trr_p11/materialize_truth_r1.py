"""Materialize the frozen TRR-P11 token truth sidecar after prediction freeze.

This small producer deliberately reuses the released P11 selection context,
the existing public-capture materializer, and the trusted public renderer.
It never loads a model or performs inference.  Token values stay in the
private safetensors sidecar; stdout contains only non-sensitive bindings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

_SCRIPT_ROOT = Path(__file__).resolve().parents[5]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

# Keep this bounded producer CPU-only and single-threaded.  These defaults are
# set before importing torch, datasets, or transformers.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from scripts import trr0005_produce_confirmation as trusted
from scripts.trr_p11 import evaluation
from scripts.trr_p11 import public_capture
from scripts.trr_p11 import source_selector as selector


EXPECTED_FREEZE_SHA256 = "1b27af8ca3c395187a68ce548cd60e3b812a5b3353c7c5d38795c36b318d481b"
EXPECTED_SELECTION_SHA256 = "fe7129e9d7230100d1900facea98b15b8000f382a0bd7035b2039b4d9429bf68"
TRUTH_STATUS = "TRUTH_PREPARED_AFTER_PREDICTION_FREEZE"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _json_digest(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"file is unavailable: {resolved}")
    if root is not None:
        try:
            resolved.relative_to(root)
        except ValueError:
            # Public Arrow/tokenizer bindings are intentionally outside the
            # repository; their absolute paths remain hash-bound here.
            pass
    return {"path": str(resolved), "bytes": int(resolved.stat().st_size), "sha256": _sha256_file(resolved)}


def _git_head(root: Path) -> str | None:
    try:
        value = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def _peak_rss_bytes() -> int:
    # Linux ru_maxrss is KiB; retain the conversion explicitly in the receipt.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"output is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return _file_record(path)


def _raw_i32_digest(values: Sequence[int]) -> tuple[str, bytes]:
    if len(values) != selector.STORED_SEQUENCE_TOKENS:
        raise RuntimeError("truth row has the wrong stored-token count")
    normalized = tuple(int(value) for value in values)
    if any(value < -(2**31) or value >= 2**31 for value in normalized):
        raise RuntimeError("truth row contains an out-of-range signed int32")
    raw = struct.pack("<" + "i" * selector.STORED_SEQUENCE_TOKENS, *normalized)
    return _sha256_bytes(raw), raw


def _materialize(
    *,
    root: Path,
    selection_path: Path,
    freeze_path: Path,
    output_dir: Path,
    write: bool,
) -> dict[str, Any]:
    started_utc = _utc_now()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"repository root is unavailable: {root}")
    expected_output = (root / "outputs" / selector.TASK_ID / "private-evaluation" / "evaluation" / "truth-r1").resolve()
    if output_dir.resolve() != expected_output:
        raise RuntimeError(f"truth output must be exactly below evaluation/truth-r1: {output_dir}")

    selection_path = selection_path.expanduser().resolve()
    freeze_path = freeze_path.expanduser().resolve()
    selection_record = _file_record(selection_path)
    freeze_record = _file_record(freeze_path)
    if selection_record["sha256"] != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("frozen source-selection hash differs from the authorized binding")
    if freeze_record["sha256"] != EXPECTED_FREEZE_SHA256:
        raise RuntimeError("frozen prediction hash differs from the authorized binding")

    selection = selector.load_selection(selection_path, root=root)
    inputs = selector._normalize_source_inputs(selection.payload["public_sources_frozen"], root=root)
    public_capture._validate_p11_source_descriptors(inputs, root=root)

    # The trusted loader reads only the frozen public tokenizer and Arrow
    # caches.  No model snapshot, activation, or prediction is loaded.
    tokenizer = trusted._load_tokenizer(Path(inputs["tokenizer"]["path"]))
    datasets = {
        domain: trusted._load_arrow_dataset(tuple(Path(item["path"]) for item in inputs[domain]["arrow_files"]))
        for domain in selector.DOMAIN_ORDER
    }
    context = selection
    records = public_capture._materialize_selected(context, trusted=trusted, datasets=datasets, tokenizer=tokenizer)

    tensors_by_domain: dict[str, torch.Tensor] = {}
    order_bindings: dict[str, Any] = {}
    raw_concat_digests: dict[str, str] = {}
    for domain in selector.DOMAIN_ORDER:
        declared_rows = context.rows[domain]
        rendered_rows = records.get(domain)
        if not isinstance(rendered_rows, list) or len(rendered_rows) != selector.RECORDS_PER_DOMAIN:
            raise RuntimeError(f"materialized row count changed: {domain}")
        values: list[list[int]] = []
        raw_rows: list[bytes] = []
        for declared, candidate in zip(declared_rows, rendered_rows):
            sequence = tuple(int(value) for value in candidate.token_ids[: selector.STORED_SEQUENCE_TOKENS])
            if len(sequence) != selector.STORED_SEQUENCE_TOKENS:
                raise RuntimeError(f"materialized token geometry changed: {domain}")
            if sequence[0] != selector.BOS_TOKEN_ID:
                raise RuntimeError(f"materialized BOS binding changed: {domain}")
            if any(value < 0 or value >= selector.VOCAB_SIZE for value in sequence):
                raise RuntimeError(f"materialized token range changed: {domain}")
            h128, raw = _raw_i32_digest(sequence)
            if h128 != declared["h128_sequence_sha256"]:
                raise RuntimeError(f"raw int32 H128 binding changed: {domain}")
            h40 = _sha256_bytes(raw[: 40 * 4])
            if h40 != declared["h40_sequence_sha256"]:
                raise RuntimeError(f"raw int32 H40 binding changed: {domain}")
            values.append(list(sequence))
            raw_rows.append(raw)
        tensors_by_domain[domain] = torch.tensor(values, dtype=torch.int32).contiguous()
        declared_h128 = [str(row["h128_sequence_sha256"]) for row in declared_rows]
        declared_ids = [str(row["record_id"]) for row in declared_rows]
        order_bindings[domain] = {
            "records": len(values),
            "record_ids_sha256": _json_digest(declared_ids),
            "h128_sequence_sha256_list_sha256": _json_digest(declared_h128),
            "h128_sequence_sha256_row_order": declared_h128,
        }
        raw_concat_digests[domain] = _sha256_bytes(b"".join(raw_rows))

    # Avoid shared-storage aliasing in safetensors while preserving identical
    # paired truth arrays for the two public target conditions.
    tensors = {
        f"{domain}__{target}__token_ids": tensors_by_domain[domain].clone()
        for domain in selector.DOMAIN_ORDER
        for target in selector.TARGET_ORDER
    }
    code_paths = {
        "scripts/trr_p11/public_capture.py": root / "scripts/trr_p11/public_capture.py",
        "scripts/trr_p11/source_selector.py": root / "scripts/trr_p11/source_selector.py",
        "scripts/trr_p11/evaluation.py": root / "scripts/trr_p11/evaluation.py",
        "scripts/trr0005_produce_confirmation.py": root / "scripts/trr0005_produce_confirmation.py",
    }
    code_records = {name: _file_record(path, root=root) for name, path in code_paths.items()}
    source_records = {
        domain: {
            "dataset_key": inputs[domain]["dataset_key"],
            "dataset_id": inputs[domain]["dataset_id"],
            "split": inputs[domain]["split"],
            "revision": inputs[domain]["revision"],
            "arrow_files": [dict(item) for item in inputs[domain]["arrow_files"]],
        }
        for domain in selector.DOMAIN_ORDER
    }
    source_records["tokenizer"] = {
        "path": inputs["tokenizer"]["path"],
        "files": {name: dict(item) for name, item in inputs["tokenizer"]["files"].items()},
    }

    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = output_dir / "truth_tokens.safetensors"
        descriptor_path = output_dir / "truth_descriptor.json"
        receipt_path = output_dir / "producer_receipt.json"
        producer_path = Path(__file__).resolve()
        if any(path.exists() or path.is_symlink() for path in (sidecar_path, descriptor_path, receipt_path)):
            raise RuntimeError("truth-r1 output is create-only and already contains a final artifact")
        metadata = {
            "freeze_sha256": str(freeze_record["sha256"]),
            "truth_opened": "true",
            "truth_schema": evaluation.TRUTH_SCHEMA,
            "task_id": evaluation.TASK_ID,
            "dtype": "int32",
            "records_per_domain": str(selector.RECORDS_PER_DOMAIN),
            "stored_sequence_tokens": str(selector.STORED_SEQUENCE_TOKENS),
            "bos_token_id": str(selector.BOS_TOKEN_ID),
            "source_selection_sha256": str(selection_record["sha256"]),
        }
        from safetensors.torch import save_file

        save_file(tensors, str(sidecar_path), metadata=metadata)
        sidecar_record = _file_record(sidecar_path, root=root)
        elapsed_seconds = time.perf_counter() - _MATERIALIZE_START
        receipt_payload = {
            "schema": "token-reconstruction.trr-p11-truth-producer-receipt.v1",
            "task_id": evaluation.TASK_ID,
            "status": TRUTH_STATUS,
            "started_utc": started_utc,
            "finished_utc": _utc_now(),
            "command": list(sys.argv),
            "command_cwd": str(Path.cwd()),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "git_head": _git_head(root),
            "cpu_threads": 1,
            "device": "cpu",
            "gpu_used": False,
            "model_loaded": False,
            "inference_performed": False,
            "network_used": False,
            "truth_opened_after_prediction_freeze": True,
            "p03_holdout_accessed": False,
            "source_text_read_transiently": True,
            "source_text_written": False,
            "token_ids_written": True,
            "raw_token_values_printed": False,
            "wall_seconds": elapsed_seconds,
            "peak_rss_bytes": _peak_rss_bytes(),
            "freeze": freeze_record,
            "selection": selection_record,
            "sidecar": sidecar_record,
            "producer_script": _file_record(producer_path, root=root),
            "code_files": code_records,
            "source_inputs": source_records,
            "order_bindings": order_bindings,
            "raw_int32_concat_sha256": raw_concat_digests,
            "tensor_names": sorted(tensors),
            "tensor_shapes": {name: list(value.shape) for name, value in tensors.items()},
            "tensor_dtype": {name: str(value.dtype).replace("torch.", "") for name, value in tensors.items()},
            "safetensors_metadata": metadata,
        }
        receipt_record = _write_json_create_only(receipt_path, receipt_payload)
        descriptor_payload = {
            "schema": evaluation.TRUTH_SCHEMA,
            "task_id": evaluation.TASK_ID,
            "status": TRUTH_STATUS,
            "created_utc": _utc_now(),
            "predictions_frozen_before_truth": True,
            "p03_holdout_accessed": False,
            "truth_opened": True,
            "source_text_written": False,
            "token_ids_written": True,
            "target_labels_loaded": False,
            "candidate_arrays_persisted": False,
            "freeze": freeze_record,
            "selection": selection_record,
            "sidecar": sidecar_record,
            "producer_script": _file_record(producer_path, root=root),
            "producer_receipt": receipt_record,
            "source_inputs": source_records,
            "geometry": {
                "domains": list(selector.DOMAIN_ORDER),
                "targets": list(selector.TARGET_ORDER),
                "cells": list(selector.CELL_ORDER),
                "records_per_domain": selector.RECORDS_PER_DOMAIN,
                "stored_sequence_tokens": selector.STORED_SEQUENCE_TOKENS,
                "bos_token_id": selector.BOS_TOKEN_ID,
                "dtype": "int32",
            },
            "order_bindings": order_bindings,
            "raw_int32_concat_sha256": raw_concat_digests,
            "code_files": code_records,
            "access_boundary": {
                "predictions_frozen_before_truth": True,
                "truth_opened_after_freeze": True,
                "source_text_read_transiently": True,
                "source_text_written": False,
                "token_ids_written": True,
                "target_labels_loaded": False,
                "model_loaded": False,
                "gpu_used": False,
                "p03_holdout_accessed": False,
            },
        }
        descriptor_record = _write_json_create_only(descriptor_path, descriptor_payload)
        # Validate through the existing post-freeze gate immediately.  This
        # opens only the authorized truth sidecar and emits no token values.
        values, validation = evaluation.load_truth_after_freeze(
            freeze_path=freeze_path,
            truth_descriptor_path=descriptor_path,
            repository_root=root,
        )
        if set(values) != set(selector.CELL_ORDER):
            raise RuntimeError("post-freeze validator returned unexpected cells")
        return {
            "descriptor": descriptor_record,
            "sidecar": sidecar_record,
            "receipt": receipt_record,
            "validation": {
                "truth_opened": validation["truth_opened"],
                "matrix_status": validation["matrix_status"],
                "cells": sorted(values),
            },
            "order_bindings": order_bindings,
            "raw_int32_concat_sha256": raw_concat_digests,
        }

    return {
        "status": "PREFLIGHT_PASS_NO_OUTPUT_WRITTEN",
        "freeze_sha256": freeze_record["sha256"],
        "selection_sha256": selection_record["sha256"],
        "order_bindings": order_bindings,
        "raw_int32_concat_sha256": raw_concat_digests,
        "records_by_domain": {domain: int(tensor.shape[0]) for domain, tensor in tensors_by_domain.items()},
        "tensor_shape": [selector.RECORDS_PER_DOMAIN, selector.STORED_SEQUENCE_TOKENS],
        "dtype": "int32",
        "model_loaded": False,
        "gpu_used": False,
        "raw_token_values_printed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write", action="store_true", help="create the private truth sidecar, receipt, and descriptor")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    global _MATERIALIZE_START
    _MATERIALIZE_START = time.perf_counter()
    result = _materialize(
        root=args.root.expanduser().resolve(),
        selection_path=args.selection,
        freeze_path=args.freeze,
        output_dir=args.output_dir,
        write=bool(args.write),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
