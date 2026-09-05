#!/usr/bin/env python3
"""Replay public-only Track B checkpoints selected from a frozen curve.

This small runner deliberately follows ``trr0003_track_b.py``'s main-method
initialization order.  The selected step counts are supplied by the committed
public-validation amendment; no evaluator-private inputs are loaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import torch
from safetensors.torch import load_file

from token_reconstruction.experiment_runtime import (
    PhaseTimer,
    MODEL_ID,
    MODEL_REVISION,
    seed_everything,
    sha256_file,
    utc_now,
    write_json_exclusive,
)
from token_reconstruction.inverse import InverseTrainingConfig, ResidualAffineInverse, save_inverse
from token_reconstruction.standalone_decoder import (
    DecoderTrainingConfig,
    decoder_from_method,
    decoder_source_hash,
    save_token_decoder,
    train_angular_control_with_curve,
    train_token_decoder,
    validate_embedding_table,
)


ROOT = Path(__file__).resolve().parents[3]
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
BOS_TOKEN_ID = 128000
METHOD_ORDER = (
    "angular_inverse_control",
    "tied_affine_token_ce",
    "residual_mlp256_token_ce",
)
SELECTED_STEPS = {
    "angular_inverse_control": 150,
    "tied_affine_token_ce": 175,
    "residual_mlp256_token_ce": 375,
}


def _sha(path: Path) -> str:
    return sha256_file(path)


def _file(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "scripts/trr0003_track_b.py",
        ROOT / "src/token_reconstruction/standalone_decoder.py",
        Path(__file__),
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_tensor(path: Path, key: str) -> torch.Tensor:
    state = load_file(path.resolve(), device="cpu")
    if set(state) != {key}:
        raise RuntimeError(f"{path} must contain exactly {key!r}")
    return state[key].contiguous()


def _flatten_observations(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3 or value.shape[2] != HIDDEN_SIZE or value.shape[1] <= 1:
        raise RuntimeError("invalid observation geometry")
    return value[:, 1:, :].reshape(-1, HIDDEN_SIZE).contiguous()


def _flatten_labels(value: torch.Tensor, records: int, positions: int) -> torch.Tensor:
    if value.ndim != 2 or tuple(value.shape) != (records, positions):
        raise RuntimeError("invalid label geometry")
    if value[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise RuntimeError("public labels do not begin with the declared BOS")
    return value[:, 1:].reshape(-1).long().contiguous()


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _rng_digest() -> str:
    """Digest CPU RNG state; model initialization happens on CPU before .to()."""

    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def _discard_tiny_initialization(method: str) -> None:
    """Consume the original fit loop's separate tiny-model initialization.

    The tiny diagnostic is deliberately not retrained in this replay, but its
    constructor consumes global RNG state in the original runner.  Preserving
    that draw is required for the subsequent main-arm initialization, while
    each optimizer's minibatch generator remains independently seeded.
    """

    if method == "angular_inverse_control":
        _ = ResidualAffineInverse(HIDDEN_SIZE)
    else:
        _ = decoder_from_method(
            method, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE,
            logit_scale=16.0, bottleneck_size=256,
        )


def _reference_initialization_trace() -> dict[str, str]:
    """Return the CPU RNG checkpoints for the original main/tiny order."""

    seed_everything(1737)
    trace: dict[str, str] = {}
    for method in METHOD_ORDER:
        trace[f"{method}.before_main"] = _rng_digest()
        if method == "angular_inverse_control":
            _ = ResidualAffineInverse(HIDDEN_SIZE)
        else:
            _ = decoder_from_method(
                method, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE,
                logit_scale=16.0, bottleneck_size=256,
            )
        trace[f"{method}.after_main_initialization"] = _rng_digest()
        _discard_tiny_initialization(method)
        trace[f"{method}.after_tiny_initialization"] = _rng_digest()
    return trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--validation-observations", type=Path, required=True)
    parser.add_argument("--validation-truth", type=Path, required=True)
    parser.add_argument("--curve-evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    if args.output_root.exists() or args.output_root.is_symlink():
        raise RuntimeError(f"output must be create-only: {args.output_root}")
    args.output_root.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    input_root = args.input_root.resolve()
    fit_observations_path = input_root / "fit_observations.safetensors"
    fit_truth_path = input_root / "fit_truth.safetensors"
    embedding_path = input_root / "public_normalized_embeddings.safetensors"
    fit_observations = _load_tensor(fit_observations_path, "activations")
    fit_truth = _load_tensor(fit_truth_path, "token_ids").long()
    embeddings = _load_tensor(embedding_path, "embeddings").float().contiguous()
    validate_embedding_table(embeddings, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE)
    if tuple(fit_observations.shape[:2]) != tuple(fit_truth.shape):
        raise RuntimeError("fit observations and labels do not match")
    fit_x = _flatten_observations(fit_observations)
    fit_y = _flatten_labels(fit_truth, int(fit_observations.shape[0]), int(fit_observations.shape[1]))

    validation_observations = _load_tensor(args.validation_observations, "activations")
    validation_truth = _load_tensor(args.validation_truth, "token_ids").long()
    if tuple(validation_observations.shape[:2]) != tuple(validation_truth.shape):
        raise RuntimeError("validation observations and labels do not match")
    validation_x = _flatten_observations(validation_observations)
    validation_y = _flatten_labels(
        validation_truth,
        int(validation_observations.shape[0]),
        int(validation_observations.shape[1]),
    )
    eval_sets = {"public_validation": (validation_x, validation_y)}

    curve = json.loads(args.curve_evidence.read_text(encoding="utf-8"))
    expected_rng_trace = _reference_initialization_trace()
    seed_everything(1737)
    timer = PhaseTimer()
    artifacts: dict[str, dict[str, Any]] = {}
    training: dict[str, dict[str, Any]] = {}
    actual_rng_trace: dict[str, str] = {}
    for method in METHOD_ORDER:
        steps = SELECTED_STEPS[method]
        actual_rng_trace[f"{method}.before_main"] = _rng_digest()
        if method == "angular_inverse_control":
            with timer.measure(f"replay_{method}"):
                model, evidence = train_angular_control_with_curve(
                    fit_x,
                    fit_y,
                    embeddings,
                    config=InverseTrainingConfig(
                        steps=steps,
                        batch_size=512,
                        learning_rate=1e-3,
                        weight_decay=0.0,
                        gradient_clip_norm=1.0,
                        seed=1737,
                    ),
                    device=device,
                    eval_sets=eval_sets,
                    log_every=25,
                )
            actual_rng_trace[f"{method}.after_main_initialization"] = _rng_digest()
            path = args.output_root / f"{method}.safetensors"
            save_inverse(model, path, cut_depth=4)
        else:
            with timer.measure(f"replay_{method}"):
                model = decoder_from_method(
                    method,
                    hidden_size=HIDDEN_SIZE,
                    vocab_size=VOCAB_SIZE,
                    logit_scale=16.0,
                    bottleneck_size=256,
                )
                actual_rng_trace[f"{method}.after_main_initialization"] = _rng_digest()
                model, evidence = train_token_decoder(
                    model,
                    fit_x,
                    fit_y,
                    embeddings,
                    config=DecoderTrainingConfig(
                        steps=steps,
                        batch_size=512,
                        learning_rate=1e-3,
                        weight_decay=0.0,
                        gradient_clip_norm=1.0,
                        log_every=25,
                        logit_scale=16.0,
                        seed=1737,
                    ),
                    device=device,
                    eval_sets=eval_sets,
                )
            path = args.output_root / f"{method}.safetensors"
            save_token_decoder(
                model,
                path,
                method_id=method,
                metadata={
                    "task_id": "TRR-0003",
                    "cut_depth": "4",
                    "checkpoint_selection": "public_validation_max_earliest_tie",
                    "embedding_sha256": _sha(embedding_path),
                    "fit_observations_sha256": _sha(fit_observations_path),
                    "fit_truth_sha256": _sha(fit_truth_path),
                },
            )
        _discard_tiny_initialization(method)
        actual_rng_trace[f"{method}.after_tiny_initialization"] = _rng_digest()
        row = next(
            point for point in evidence["learning_curve"] if point["step"] == steps
        )
        expected_rows = [
            point
            for point in curve["methods"][method]["training"]["learning_curve"]
            if point["step"] == steps
        ]
        if len(expected_rows) != 1:
            raise RuntimeError(f"curve evidence has no unique selected row for {method}")
        expected = expected_rows[0]
        if row != expected:
            raise RuntimeError(
                f"selected replay curve differs for {method}: {row!r} != {expected!r}"
            )
        artifacts[method] = _file(path)
        training[method] = {
            "steps": steps,
            "seed": 1737,
            "batch_size": 512,
            "learning_rate": 1e-3,
            "curve_row": row,
            "curve_row_verified_against": _file(args.curve_evidence),
            "state_sha256": _sha(path),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if actual_rng_trace != expected_rng_trace:
        raise RuntimeError(
            "replay CPU RNG initialization trace differs from original method order: "
            + json.dumps({"expected": expected_rng_trace, "actual": actual_rng_trace}, sort_keys=True)
        )
    evidence = {
        "schema": "token-reconstruction.trr0003-track-b-selected-replay.v1",
        "task_id": "TRR-0003",
        "track": "track_b",
        "selection_rule": {
            "metric": "public_validation_token_accuracy",
            "tie_break": "earliest logged checkpoint",
            "curve_source": _file(args.curve_evidence),
            "panel_truth_opened": False,
            "target_weights_or_prefix_calls": False,
        },
        "selected_steps": SELECTED_STEPS,
        "initialization": {
            "seed": 1737,
            "method_order": list(METHOD_ORDER),
            "same_order_as": "scripts/trr0003_track_b.py main fit loop",
            "replay_uses_public_fit_labels_only": True,
            "cpu_rng_trace_verified": True,
            "cpu_rng_trace": actual_rng_trace,
        },
        "git_commit": _commit(),
        "code_sha256": _code_hash(),
        "decoder_source_sha256": decoder_source_hash(),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "cut_depth": 4},
        "fit_inputs": {
            "observations": _file(fit_observations_path),
            "truth": _file(fit_truth_path),
            "embedding_table": _file(embedding_path),
        },
        "public_validation_inputs": {
            "observations": _file(args.validation_observations),
            "truth": _file(args.validation_truth),
            "truth_role": "public auxiliary validation only",
        },
        "training": training,
        "artifacts": artifacts,
        "phases": timer.records,
        "peak_memory": {
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
            if device.type == "cuda"
            else None,
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved())
            if device.type == "cuda"
            else None,
        },
        "notes": [
            "This is a public-only checkpoint replay selected before shared-panel prediction.",
            "The 1800-step final states and tiny-subset overfit states remain retained as diagnostics.",
            "The replay emits compact decoder/inverse state only; it uses no A2 fallback.",
        ],
    }
    write_json_exclusive(args.output_root / "selected_replay_evidence.json", evidence)
    print(json.dumps({"status": "selected_replay_complete", "output_root": str(args.output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
