#!/usr/bin/env python3
"""Generate frozen truth-free P04 student predictions from observations.

Only activation and mask tensors are opened. The output JSONL follows the
setup-owned scorer schema and contains one post-BOS prediction vector per
record; tie counts are retained in a separate diagnostic file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from safetensors import safe_open
import torch

from token_reconstruction.p04_student import (
    PREDICTION_SCHEMA,
    P04StudentError,
    StudentArchitectureConfig,
    load_student_state,
    prediction_tensor,
    validate_embedding_table,
)
from token_reconstruction.p04_training import file_sha256


TASK_ID = "TRR-P04"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--embedding-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tie-output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--record-batch-size", type=int, default=8)
    parser.add_argument("--projection-chunk", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=1)
    return parser.parse_args()


def _records(path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P04StudentError(f"cannot parse prediction record metadata: {path}") from exc
    values = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise P04StudentError("prediction record metadata must contain records")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict) or not isinstance(value.get("record_id"), str):
            raise P04StudentError(f"prediction record {index} has no record_id")
        if any(key in value for key in ("token_ids", "source_text", "truth", "labels")):
            raise P04StudentError("prediction metadata contains source/truth fields")
        record_id = str(value["record_id"])
        if record_id in seen:
            raise P04StudentError("prediction record IDs are duplicated")
        seen.add(record_id)
        result.append(value)
    return result


def _component(path: Path, key: str) -> torch.Tensor:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P04StudentError(f"prediction asset must be a regular file: {path}")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if key not in keys:
                raise P04StudentError(f"prediction observations lack {key}")
            return handle.get_tensor(key).contiguous()
    except P04StudentError:
        raise
    except Exception as exc:
        raise P04StudentError(f"cannot read prediction observations: {path}") from exc


def _load_observations(path: Path, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise P04StudentError(f"prediction observations must be a regular file: {path}")
    try:
        handle_context = safe_open(str(path), framework="pt", device="cpu")
    except Exception as exc:
        raise P04StudentError(f"cannot inspect prediction observations: {path}") from exc
    with handle_context as handle:
        keys = set(handle.keys())
        forbidden = keys.intersection({"token_ids", "truth", "labels", "source_text"})
        if forbidden:
            raise P04StudentError(f"prediction observation artifact contains forbidden fields: {sorted(forbidden)}")
        if "activations" not in keys:
            raise P04StudentError("prediction observations lack activations")
        activations = handle.get_tensor("activations").contiguous()
        if "attention_mask" in keys:
            mask = handle.get_tensor("attention_mask").contiguous()
        elif "valid_mask" in keys:
            mask = handle.get_tensor("valid_mask").contiguous()
        else:
            raise P04StudentError("prediction observations lack attention/valid mask")
    if activations.ndim != 3 or activations.shape[0] != rows or activations.shape[1] <= 1 or activations.shape[2] <= 0:
        raise P04StudentError("prediction observations have invalid geometry")
    if not activations.dtype.is_floating_point or not torch.isfinite(activations).all().item():
        raise P04StudentError("prediction observations must be finite floating point")
    if mask.shape != activations.shape[:2] or mask.dtype not in (torch.bool, torch.uint8):
        raise P04StudentError("prediction observation mask geometry or dtype changed")
    mask = mask.to(dtype=torch.bool)
    if not mask[:, 0].all().item() or not mask[:, 1:].any(dim=1).all().item() or not torch.equal(mask, mask.cumprod(dim=1).to(torch.bool)):
        raise P04StudentError("prediction observation mask must include BOS and be right-padded")
    return activations, mask


def main() -> int:
    args = _args()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.interop_threads)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True, warn_only=False)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise P04StudentError("CUDA requested for prediction but unavailable")
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    records = _records(args.records)
    observations, valid_mask = _load_observations(args.observations, len(records))
    embedding_path = args.embedding_table.expanduser().resolve()
    if embedding_path.is_symlink() or not embedding_path.is_file():
        raise P04StudentError(f"embedding table must be a regular file: {embedding_path}")
    with safe_open(str(embedding_path), framework="pt", device="cpu") as handle:
        if "embeddings" not in set(handle.keys()):
            raise P04StudentError("public embedding asset lacks embeddings")
        embedding_table = handle.get_tensor("embeddings").contiguous().float()
    hidden_size = int(observations.shape[-1])
    vocab_size = int(embedding_table.shape[0])
    validate_embedding_table(embedding_table, hidden_size=hidden_size, vocab_size=vocab_size, require_unit_norm=True)
    config = StudentArchitectureConfig(hidden_size=hidden_size, vocab_size=vocab_size, gru_width=256)
    device = torch.device(args.device)
    started = time.perf_counter()
    model = load_student_state(args.state.expanduser().resolve(), method_id=args.method_id, device=device, config=config)
    predictions, ties = prediction_tensor(model, observations, embedding_table, device=device, valid_mask=valid_mask, record_batch_size=args.record_batch_size, projection_chunk=args.projection_chunk)
    output = args.output.expanduser().resolve()
    tie_output = args.tie_output.expanduser().resolve()
    if output.exists() or output.is_symlink() or tie_output.exists() or tie_output.is_symlink():
        raise P04StudentError("prediction outputs are create-only")
    output.parent.mkdir(parents=True, exist_ok=True)
    tie_output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    tie_rows: list[dict[str, object]] = []
    for row, metadata in enumerate(records):
        length = int(valid_mask[row].sum().item())
        values = predictions[row, 1:length].tolist()
        if any(int(value) < 0 or int(value) >= vocab_size for value in values):
            raise P04StudentError(f"prediction contains an invalid token at {metadata['record_id']}")
        lines.append(json.dumps({"schema": PREDICTION_SCHEMA, "method_id": args.method_id, "seed": args.seed, "condition": args.condition, "record_id": str(metadata["record_id"]), "predicted_token_ids": [int(value) for value in values]}, sort_keys=True))
        tie_rows.append({"record_id": str(metadata["record_id"]), "tie_counts": [int(value) for value in ties[row, 1:length].tolist()]})
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tie_output.write_text(json.dumps({"schema": "token-reconstruction.trr-p04-tie-diagnostics.v1", "method_id": args.method_id, "seed": args.seed, "condition": args.condition, "rows": tie_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {"schema": "token-reconstruction.trr-p04-prediction-receipt.v1", "task_id": TASK_ID, "status": "PASS", "argv": sys.argv, "method_id": args.method_id, "seed": args.seed, "condition": args.condition, "observations": {"path": str(args.observations.resolve()), "sha256": file_sha256(args.observations)}, "records": {"path": str(args.records.resolve()), "sha256": file_sha256(args.records)}, "state": {"path": str(args.state.resolve()), "sha256": file_sha256(args.state)}, "embedding_table": {"path": str(args.embedding_table.resolve()), "sha256": file_sha256(args.embedding_table)}, "prediction": {"path": str(output), "sha256": file_sha256(output)}, "tie_diagnostics": {"path": str(tie_output), "sha256": file_sha256(tie_output)}, "rows": len(records), "post_bos_positions": int(valid_mask[:, 1:].sum().item()), "uses_source_tokens": False, "uses_teacher_or_candidates": False, "uses_prefix_calls": False, "full_vocabulary": True, "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, "wall_seconds": time.perf_counter() - started, "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    receipt_path = output.parent / "prediction_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise P04StudentError("prediction receipt is create-only")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "prediction": str(output), "rows": len(records), "post_bos_positions": receipt["post_bos_positions"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (P04StudentError, RuntimeError) as exc:
        print(f"P04 prediction failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
