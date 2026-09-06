#!/usr/bin/env python3
"""Qualify the largest P04 GRU/full-vocabulary training cell.

This is a public correction-pool capacity and resource probe. It runs a short
CE-only update on one exact batch of eight 192-position records, then reports
step-0 affine errors and the post-probe result. It never reads evaluator truth.
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

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from token_reconstruction.p04_student import METHOD_S, StudentArchitectureConfig, initialize_student
from token_reconstruction.p04_training import (
    P04TrainingError,
    _projection_logits,
    evaluate_public,
    file_sha256,
    load_embedding_table,
    load_public_pool,
    tensor_sha256,
    canonical_hash,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction-observations", type=Path, required=True)
    parser.add_argument("--correction-records", type=Path, required=True)
    parser.add_argument("--correction-truth", type=Path)
    parser.add_argument("--correction-mask", type=Path)
    parser.add_argument("--embedding-table", type=Path, required=True)
    parser.add_argument("--affine-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _load_state(path: Path, hidden: int) -> dict[str, torch.Tensor]:
    path = path.expanduser().resolve()
    if path.suffix == ".pt":
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        state = checkpoint.get("sd") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            raise P04TrainingError("pickled affine state lacks sd")
    else:
        state = load_file(str(path), device="cpu")
    if set(state) != {"W", "b", "s"} or state["W"].shape != (hidden, hidden) or state["b"].shape != (hidden,) or state["s"].ndim != 0:
        raise P04TrainingError("qualifier affine state geometry changed")
    return {key: value.float().contiguous() for key, value in state.items()}


def _set_runtime(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.interop_threads)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True, warn_only=False)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise P04TrainingError("CUDA requested for largest-cell qualifier but unavailable")
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _probe_mask(pool, records: list[int], budget: int = 512) -> torch.Tensor:
    selected = torch.zeros((len(records), pool.positions), dtype=torch.bool)
    per_record = max(1, budget // len(records))
    for local, row in enumerate(records):
        valid_length = int(pool.valid_mask[row].sum().item())
        positions = list(range(1, valid_length))
        if len(positions) > per_record:
            indices = torch.linspace(0, len(positions) - 1, steps=per_record).round().to(torch.long).tolist()
            positions = [positions[int(index)] for index in indices]
        selected[local, positions] = True
    if int(selected.sum().item()) > budget or selected[:, 0].any().item():
        raise P04TrainingError("qualifier probe mask exceeded 512 or selected BOS")
    return selected


def main() -> int:
    args = _args()
    _set_runtime(args)
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise P04TrainingError("qualifier output root must be empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    correction = load_public_pool(args.correction_observations, args.correction_records, truth_path=args.correction_truth, mask_path=args.correction_mask, embedding_vocab_size=128256)
    if correction.positions != 192 or correction.hidden_size != 2048:
        raise P04TrainingError(f"largest qualifier requires [records,192,2048], got {tuple(correction.observations.shape)}")
    table = load_embedding_table(args.embedding_table, hidden_size=2048, vocab_size=128256)
    affine_state = _load_state(args.affine_state, 2048)
    config = StudentArchitectureConfig(hidden_size=2048, vocab_size=128256, gru_width=256)
    device = torch.device(args.device)
    # The trainable affine step-0 control is evaluated before any probe update.
    initial = initialize_student(METHOD_S, seed=1737, config=config, affine_state=affine_state).to(device).eval()
    initial_metrics = evaluate_public(initial, correction, table, device=device, record_batch_size=8, projection_chunk=512)
    initial_predictions = initial_metrics.pop("predictions")
    initial_ties = initial_metrics.pop("tie_counts")
    scored = correction.valid_mask.clone()
    scored[:, 0] = False
    wrong = initial_predictions.ne(correction.labels) & scored
    wrong_per_record = wrong.sum(dim=1)
    # The probe is fixed to the first eight correction records with at least
    # one initial affine error. Measure this exact batch before any update;
    # do not select extra rows and then truncate a larger error accumulator.
    chosen = [row for row, count in enumerate(wrong_per_record.tolist()) if count][:8]
    if len(chosen) < 8:
        raise P04TrainingError("capacity qualifier needs eight correction records with initial affine errors")
    running_wrong = int(wrong_per_record[torch.tensor(chosen)].sum().item())
    selected = _probe_mask(correction, chosen)
    probe = type(correction)(
        observations=correction.observations[chosen], labels=correction.labels[chosen], valid_mask=correction.valid_mask[chosen], record_ids=tuple(correction.record_ids[row] for row in chosen), styles=tuple(correction.styles[row] for row in chosen), source_path=correction.source_path, source_sha256=correction.source_sha256, records_path=correction.records_path, records_sha256=correction.records_sha256,
    )
    probe_initial = evaluate_public(initial, probe, table, device=device, record_batch_size=8, projection_chunk=512)
    probe_initial_predictions = probe_initial.pop("predictions")
    probe_initial.pop("tie_counts")
    # evaluate_public leaves the model in eval mode; cuDNN requires training
    # mode for the GRU backward used by the actual capacity probe.
    initial.train()
    optimizer = torch.optim.AdamW(initial.parameters(), lr=1.0e-3, weight_decay=0.0)
    batch_x = probe.observations.to(device=device, dtype=torch.float32)
    batch_y = probe.labels.to(device=device, dtype=torch.long)
    selected_device = selected.to(device=device)
    losses: list[float] = []
    for step in range(args.steps):
        logits = _projection_logits(initial, batch_x, table.to(device=device, dtype=torch.float32), selected_device)
        target = batch_y[selected_device]
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss).item():
            raise P04TrainingError("qualifier loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(initial.parameters(), 1.0, error_if_nonfinite=True)
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all().item() for parameter in initial.parameters()):
            raise P04TrainingError("qualifier gradient is non-finite")
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    post_metrics = evaluate_public(initial, probe, table, device=device, record_batch_size=8, projection_chunk=512)
    post_predictions = post_metrics.pop("predictions")
    post_ties = post_metrics.pop("tie_counts")
    probe_scored = probe.valid_mask.clone()
    probe_scored[:, 0] = False
    probe_initial_wrong = int((probe_initial_predictions.ne(probe.labels) & probe_scored).sum().item())
    probe_scored_total = int(probe_scored.sum().item())
    probe_initial_accuracy = 1.0 - (probe_initial_wrong / probe_scored_total) if probe_scored_total else 0.0
    if probe_initial_wrong < 256:
        raise P04TrainingError(
            f"capacity probe has only {probe_initial_wrong} initial affine errors; need at least 256 on the fixed eight-row probe"
        )
    if probe_initial_accuracy >= 0.99:
        raise P04TrainingError(
            f"capacity probe initial accuracy {probe_initial_accuracy:.6f} is not below 0.99"
        )
    receipt = {
        "schema": "token-reconstruction.trr-p04-largest-cell-qualifier.v1",
        "task_id": "TRR-P04",
        "status": "PASS",
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "geometry": {"batch_records": 8, "sequence_positions": 192, "hidden_size": 2048, "gru_width": 256, "position_budget": int(selected.sum().item()), "vocabulary_size": 128256},
        "source": {"observations": {"path": correction.source_path, "sha256": correction.source_sha256}, "records": {"path": correction.records_path, "sha256": correction.records_sha256}, "embedding": {"path": str(args.embedding_table.resolve()), "sha256": file_sha256(args.embedding_table)}, "affine_state": {"path": str(args.affine_state.resolve()), "sha256": file_sha256(args.affine_state)}},
        "probe": {
            "record_ids": list(probe.record_ids),
            "record_order_sha256": canonical_hash(list(probe.record_ids)),
            "selected_mask_sha256": tensor_sha256(selected),
            "initial_wrong_positions_all_correction": int(wrong.sum().item()),
            "initial_wrong_positions_probe": probe_initial_wrong,
            "scored_positions_probe": probe_scored_total,
            "initial_accuracy_probe": probe_initial_accuracy,
            "gate": {"wrong_at_least_256": True, "accuracy_below_0_99": True},
        },
        "initial_metrics": initial_metrics,
        "probe_initial_metrics": probe_initial,
        "post_probe_metrics": post_metrics,
        "losses": losses,
        "tie_counts": {"initial_probe_total": int(initial_ties.sum().item()), "post_probe_total": int(post_ties.sum().item())},
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "wall_seconds": time.perf_counter() - started,
    }
    (output / "probe_selection.json").write_text(json.dumps({"record_ids": list(probe.record_ids), "indices": chosen, "selected_mask_sha256": tensor_sha256(selected)}, indent=2, sort_keys=True) + "\n")
    (output / "qualifier_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"status": "PASS", "initial_wrong_positions": running_wrong, "probe_post_accuracy": post_metrics["token_accuracy"], "peak_rss_bytes": receipt["peak_rss_bytes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (P04TrainingError, RuntimeError) as exc:
        print(f"P04 qualifier failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
