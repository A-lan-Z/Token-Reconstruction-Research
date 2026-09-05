#!/usr/bin/env python3
"""Prepare, fit, predict, and score the TRR-0003 Track B pilot.

The command is intentionally split into three phases.  ``prepare`` and
``fit`` may use public token labels, while ``predict`` accepts only boundary
activations, a fixed public embedding table, and frozen decoder state.  The
shared TRR-0003 footing owns prediction freeze and truth-gated scoring.  This
is an exploratory retrospective pilot; it does not alter the active
dual-benchmark registry.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file, save_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from token_reconstruction.experiment_runtime import (
    BOS_TOKEN_ID,
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    PhaseTimer,
    command_record,
    load_json,
    peak_memory,
    records_for_split,
    seed_everything,
    sha256_file,
    utc_now,
    write_json_exclusive,
)
from token_reconstruction.inverse import (
    InverseTrainingConfig,
    save_inverse,
    load_inverse,
)
from token_reconstruction.standalone_decoder import (
    StandaloneDecoderError,
    angular_prediction_tensor,
    complete_prediction_check,
    DecoderTrainingConfig,
    decoder_from_method,
    decoder_method_ids,
    decoder_source_hash,
    load_token_decoder,
    train_angular_control_with_curve,
    normalized_embedding_table,
    prediction_tensor,
    save_predictions,
    save_token_decoder,
    tensor_sha256,
    train_token_decoder,
    validate_embedding_table,
)


TASK_ID = "TRR-0003"
TRACK_ID = "track_b"
HIDDEN_SIZE = 2048
VOCAB_SIZE = 128256
SEQUENCE_TOKENS = 40
SCORED_TOKENS = 39
ANGULAR_METHOD = "angular_inverse_control"
METHODS = (ANGULAR_METHOD, *decoder_method_ids())
DEFAULT_EMBEDDING_NAME = "public_normalized_embeddings.safetensors"
DEFAULT_FIT_OBSERVATIONS = "fit_observations.safetensors"
DEFAULT_FIT_TRUTH = "fit_truth.safetensors"
DEFAULT_RECORDS = "fit_records.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="capture public inverse-train observations")
    prepare.add_argument("--source-plan", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--record-batch-size", type=int, default=8)

    fit = sub.add_parser("fit", help="fit standalone token decoders")
    fit.add_argument("--input-root", type=Path, required=True)
    fit.add_argument("--output-root", type=Path, required=True)
    fit.add_argument("--public-validation-observations", type=Path)
    fit.add_argument("--public-validation-truth", type=Path)
    fit.add_argument("--methods", default=",".join(METHODS))
    fit.add_argument("--steps", type=int, default=600)
    fit.add_argument("--angular-steps", type=int, default=600)
    fit.add_argument("--overfit-records", type=int, default=8)
    fit.add_argument("--overfit-steps", type=int, default=1500)
    fit.add_argument("--batch-size", type=int, default=512)
    fit.add_argument("--learning-rate", type=float, default=1e-3)
    fit.add_argument("--log-every", type=int, default=25)
    fit.add_argument("--logit-scale", type=float, default=16.0)
    fit.add_argument("--bottleneck-size", type=int, default=256)
    fit.add_argument("--seed", type=int, default=1737)
    fit.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))

    predict = sub.add_parser("predict", help="emit direct tokens without truth or A2")
    predict.add_argument("--observations", type=Path, required=True)
    predict.add_argument("--embedding-table", type=Path, required=True)
    predict.add_argument("--fit-root", type=Path, required=True)
    predict.add_argument("--output-root", type=Path, required=True)
    predict.add_argument("--methods", default=",".join(METHODS))
    predict.add_argument("--batch-size", type=int, default=512)
    predict.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))



    return parser


def _methods(raw: str) -> tuple[str, ...]:
    result = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not result or len(set(result)) != len(result) or any(value not in METHODS for value in result):
        raise StandaloneDecoderError(f"methods must be a unique subset of {METHODS}")
    return result


def _ensure_new_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise StandaloneDecoderError(f"output must be create-only: {path}")
    path.mkdir(parents=True)


def _device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raise StandaloneDecoderError("CUDA was requested but is unavailable")
    return torch.device(raw)


def _load_tensor(path: Path, key: str) -> torch.Tensor:
    if path.is_symlink() or not path.is_file():
        raise StandaloneDecoderError(f"tensor input must be a regular file: {path}")
    state = load_file(path, device="cpu")
    if set(state) != {key}:
        raise StandaloneDecoderError(f"{path} must contain exactly the {key!r} tensor")
    return state[key].contiguous()


def _load_observations(path: Path) -> torch.Tensor:
    value = _load_tensor(path, "activations")
    if value.ndim != 3 or value.shape[0] <= 0 or value.shape[1] <= 1 or value.shape[2] != HIDDEN_SIZE:
        raise StandaloneDecoderError("observations must have shape [records, positions, 2048]")
    if not value.dtype.is_floating_point or not torch.isfinite(value).all().item():
        raise StandaloneDecoderError("observations are not finite floating point")
    return value


def _load_truth(path: Path) -> torch.Tensor:
    value = _load_tensor(path, "token_ids")
    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 1:
        raise StandaloneDecoderError("truth must have shape [records, positions]")
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise StandaloneDecoderError("truth token IDs must be integer")
    if value[:, 0].ne(BOS_TOKEN_ID).any().item():
        raise StandaloneDecoderError("truth rows must begin with the declared BOS token")
    if value[:, 1:].lt(0).any().item() or value[:, 1:].ge(VOCAB_SIZE).any().item():
        raise StandaloneDecoderError("truth contains an out-of-range token")
    return value.to(torch.long).contiguous()


def _flatten_post_bos(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3 or value.shape[1] <= 1:
        raise StandaloneDecoderError("activation sequence geometry is invalid")
    return value[:, 1:, :].reshape(-1, value.shape[-1]).contiguous()


def _flatten_labels(value: torch.Tensor, *, records: int, positions: int) -> torch.Tensor:
    if value.ndim != 2 or tuple(value.shape) != (records, positions):
        raise StandaloneDecoderError("label geometry does not match observations")
    return value[:, 1:].reshape(-1).contiguous()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _code_hash() -> str:
    paths = [Path(__file__), Path(__file__).resolve().parents[1] / "src/token_reconstruction/standalone_decoder.py"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.resolve().read_bytes())
    return digest.hexdigest()


def _file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StandaloneDecoderError(f"artifact must be a regular file: {path}")
    label = path.resolve().relative_to(root.resolve()).as_posix() if root else str(path.resolve())
    return {"path": label, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _model() -> tuple[Any, Any, torch.nn.Module]:
    if not torch.cuda.is_available():
        raise StandaloneDecoderError("prepare requires the pinned CUDA model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
    if tokenizer.bos_token_id != BOS_TOKEN_ID:
        raise StandaloneDecoderError("pinned tokenizer BOS changed")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(torch.device("cuda")).eval()
    if model.config.hidden_size != HIDDEN_SIZE or model.config.vocab_size != VOCAB_SIZE:
        raise StandaloneDecoderError("pinned model geometry changed")
    model.requires_grad_(False)
    return tokenizer, dataset, model


def _token_tensor(records: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    value = torch.tensor([row["token_ids"] for row in records], dtype=torch.long)
    if tuple(value.shape[1:]) != (SEQUENCE_TOKENS,):
        raise StandaloneDecoderError("fit record sequence geometry changed")
    return value.to(device)


@torch.inference_mode()
def _capture_cut4(model: torch.nn.Module, records: list[dict[str, Any]], batch_size: int) -> torch.Tensor:
    if batch_size <= 0:
        raise StandaloneDecoderError("record batch size must be positive")
    device = next(model.parameters()).device
    tokens = _token_tensor(records, device)
    collected: list[torch.Tensor] = []
    for start in range(0, len(records), batch_size):
        output = model(
            input_ids=tokens[start : start + batch_size], output_hidden_states=True, use_cache=False
        )
        collected.append(output.hidden_states[4].detach().cpu().to(torch.bfloat16))
        del output
    result = torch.cat(collected, dim=0).contiguous()
    if tuple(result.shape) != (len(records), SEQUENCE_TOKENS, HIDDEN_SIZE):
        raise StandaloneDecoderError("captured fit observation geometry changed")
    return result


def _prepare(args: argparse.Namespace) -> int:
    _ensure_new_directory(args.output_root)
    seed_everything(1736)
    timer = PhaseTimer()
    with timer.measure("load_public_model_tokenizer_dataset"):
        tokenizer, dataset, model = _model()
        plan = load_json(args.source_plan)
    with timer.measure("materialize_public_inverse_train_records"):
        records = records_for_split(plan, "inverse_train", tokenizer=tokenizer, dataset=dataset)
    if len(records) != 128:
        raise StandaloneDecoderError("the exact public inverse_train split must contain 128 records")
    with timer.measure("capture_public_cut4_fit_observations"):
        observations = _capture_cut4(model, records, args.record_batch_size)
    truth = torch.tensor([row["token_ids"] for row in records], dtype=torch.int32).contiguous()
    observation_path = args.output_root / DEFAULT_FIT_OBSERVATIONS
    truth_path = args.output_root / DEFAULT_FIT_TRUTH
    records_path = args.output_root / DEFAULT_RECORDS
    embedding_path = args.output_root / DEFAULT_EMBEDDING_NAME
    save_file(
        {"activations": observations},
        observation_path,
        metadata={
            "schema": "token-reconstruction.trr0003-public-fit-observations.v1",
            "task_id": TASK_ID,
            "condition": "public_base",
            "cut_depth": "4",
            "source_truth_included": "false",
        },
    )
    save_file(
        {"token_ids": truth},
        truth_path,
        metadata={
            "schema": "token-reconstruction.trr0003-public-fit-labels.v1",
            "task_id": TASK_ID,
            "access": "public-auxiliary",
        },
    )
    with timer.measure("extract_public_embedding_table"):
        embeddings = normalized_embedding_table(model.get_input_embeddings().weight.detach().cpu())
        validate_embedding_table(embeddings, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE)
        save_file(
            {"embeddings": embeddings},
            embedding_path,
            metadata={
                "schema": "token-reconstruction.trr0003-public-normalized-embeddings.v1",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "cut_depth": "4",
                "fixed_runtime_resource": "true",
            },
        )
    write_json_exclusive(
        records_path,
        {
            "schema": "token-reconstruction.trr0003-public-fit-records.v1",
            "task_id": TASK_ID,
            "source_plan": _file_record(args.source_plan),
            "split": "inverse_train",
            "records": [
                {
                    "record_id": row["record_id"],
                    "dataset_index": row["dataset_index"],
                    "text_sha256": row["text_sha256"],
                }
                for row in records
            ],
            "observation_geometry": list(observations.shape),
            "truth_geometry": list(truth.shape),
            "public_labels_are_permitted": True,
        },
    )
    evidence = {
        "schema": "token-reconstruction.trr0003-track-b-prepare.v1",
        "task_id": TASK_ID,
        "track": TRACK_ID,
        "command": command_record(),
        "git_commit": _git_commit(),
        "code_sha256": _code_hash(),
        "started_utc": timer.records[0]["started_utc"] if timer.records else utc_now(),
        "ended_utc": utc_now(),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16", "cut_depth": 4},
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "split": "train"},
        "records": {"count": len(records), "split": "inverse_train"},
        "phases": timer.records,
        "artifacts": {
            "observations": _file_record(observation_path, root=args.output_root),
            "truth": _file_record(truth_path, root=args.output_root),
            "records": _file_record(records_path, root=args.output_root),
            "embedding_table": _file_record(embedding_path, root=args.output_root),
        },
        "peak_memory": peak_memory(),
    }
    write_json_exclusive(args.output_root / "prepare_evidence.json", evidence)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({"status": "prepared", "output_root": str(args.output_root), "records": len(records)}))
    return 0


def _public_validation(args: argparse.Namespace) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    if (args.public_validation_observations is None) != (args.public_validation_truth is None):
        raise StandaloneDecoderError(
            "public validation observations and truth must be supplied together"
        )
    if args.public_validation_observations is None:
        return {}, {}
    observations = _load_observations(args.public_validation_observations)
    truth = _load_truth(args.public_validation_truth)
    labels = _flatten_labels(
        truth, records=observations.shape[0], positions=observations.shape[1]
    )
    return {"public_validation": (_flatten_post_bos(observations), labels)}, {
        "observations": _file_record(args.public_validation_observations),
        "truth": _file_record(args.public_validation_truth),
        "records": int(observations.shape[0]),
        "positions": int(observations.shape[1]),
        "truth_role": "public auxiliary validation only",
    }


def _fit(args: argparse.Namespace) -> int:
    methods = _methods(args.methods)
    _ensure_new_directory(args.output_root)
    device = _device(args.device)
    input_root = args.input_root.resolve()
    observations_path = input_root / DEFAULT_FIT_OBSERVATIONS
    truth_path = input_root / DEFAULT_FIT_TRUTH
    embedding_path = input_root / DEFAULT_EMBEDDING_NAME
    observations = _load_observations(observations_path)
    truth = _load_truth(truth_path)
    embeddings = _load_tensor(embedding_path, "embeddings").float().contiguous()
    validate_embedding_table(embeddings, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE)
    if tuple(observations.shape[:2]) != tuple(truth.shape):
        raise StandaloneDecoderError("fit observations and labels have different geometry")
    x = _flatten_post_bos(observations)
    y = _flatten_labels(truth, records=observations.shape[0], positions=observations.shape[1])
    eval_sets, eval_evidence = _public_validation(args)
    if args.overfit_records <= 0 or args.overfit_records > observations.shape[0]:
        raise StandaloneDecoderError("overfit record count is outside fit geometry")
    overfit_x = _flatten_post_bos(observations[: args.overfit_records])
    overfit_y = _flatten_labels(
        truth[: args.overfit_records], records=args.overfit_records, positions=observations.shape[1]
    )
    seed_everything(args.seed)
    timer = PhaseTimer()
    method_evidence: dict[str, Any] = {}
    state_paths: dict[str, Path] = {}
    for method in methods:
        if method == ANGULAR_METHOD:
            config = InverseTrainingConfig(
                steps=args.angular_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=0.0,
                gradient_clip_norm=1.0,
                seed=args.seed,
            )
            target_embeddings = embeddings.index_select(0, y)
            with timer.measure("fit_angular_inverse"):
                model, evidence = train_angular_control_with_curve(
                    x,
                    y,
                    embeddings,
                    config=config,
                    device=device,
                    eval_sets=eval_sets,
                    log_every=args.log_every,
                )
            path = args.output_root / f"{ANGULAR_METHOD}.safetensors"
            save_inverse(model, path, cut_depth=4)
            state_paths[method] = path
            # The canonical API reports its objective curve at step endpoints
            # only; CE arms below provide the token-classification curves.
            method_evidence[method] = {
                "architecture": "ResidualAffineInverse",
                "objective": "one-minus-cosine to public input embedding",
                "training": evidence,
                "artifact": _file_record(path, root=args.output_root),
            }
            del model
        else:
            model = decoder_from_method(
                method,
                hidden_size=HIDDEN_SIZE,
                vocab_size=VOCAB_SIZE,
                logit_scale=args.logit_scale,
                bottleneck_size=args.bottleneck_size,
            )
            config = DecoderTrainingConfig(
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=0.0,
                gradient_clip_norm=1.0,
                log_every=args.log_every,
                logit_scale=args.logit_scale,
                seed=args.seed,
            )
            with timer.measure(f"fit_{method}"):
                model, evidence = train_token_decoder(
                    model,
                    x,
                    y,
                    embeddings,
                    config=config,
                    device=device,
                    eval_sets=eval_sets,
                )
            path = args.output_root / f"{method}.safetensors"
            save_token_decoder(
                model,
                path,
                method_id=method,
                metadata={
                    "task_id": TASK_ID,
                    "cut_depth": "4",
                    "embedding_sha256": tensor_sha256(embeddings),
                    "fit_observations_sha256": tensor_sha256(observations),
                    "fit_truth_sha256": tensor_sha256(truth),
                },
            )
            state_paths[method] = path
            method_evidence[method] = {
                "architecture": model.__class__.__name__,
                "objective": "full-vocabulary cross-entropy with normalized tied projection",
                "training": evidence,
                "artifact": _file_record(path, root=args.output_root),
            }
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # A separately initialized tiny-subset fit tests optimization/capacity
        # without changing the deployed state above.
        if method == ANGULAR_METHOD:
            target_embeddings = embeddings.index_select(0, overfit_y)
            with timer.measure("overfit_angular_inverse"):
                tiny_model, tiny_evidence = train_angular_control_with_curve(
                    overfit_x,
                    overfit_y,
                    embeddings,
                    config=InverseTrainingConfig(
                        steps=args.overfit_steps,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        gradient_clip_norm=1.0,
                        seed=args.seed + 101,
                    ),
                    device=device,
                    log_every=args.log_every,
                )
        else:
            tiny_model = decoder_from_method(
                method,
                hidden_size=HIDDEN_SIZE,
                vocab_size=VOCAB_SIZE,
                logit_scale=args.logit_scale,
                bottleneck_size=args.bottleneck_size,
            )
            tiny_config = DecoderTrainingConfig(
                steps=args.overfit_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=0.0,
                gradient_clip_norm=1.0,
                log_every=args.log_every,
                logit_scale=args.logit_scale,
                seed=args.seed + 101,
            )
            with timer.measure(f"overfit_{method}"):
                tiny_model, tiny_evidence = train_token_decoder(
                    tiny_model,
                    overfit_x,
                    overfit_y,
                    embeddings,
                    config=tiny_config,
                    device=device,
                )
        tiny_path = args.output_root / f"overfit_{method}.safetensors"
        if method == ANGULAR_METHOD:
            save_inverse(tiny_model, tiny_path, cut_depth=4)
        else:
            save_token_decoder(
                tiny_model,
                tiny_path,
                method_id=method,
                metadata={
                    "task_id": TASK_ID,
                    "diagnostic": "tiny_public_subset_overfit",
                    "embedding_sha256": tensor_sha256(embeddings),
                },
            )
        method_evidence[method]["tiny_subset_overfit"] = {
            "records": args.overfit_records,
            "examples": int(overfit_x.shape[0]),
            "steps": args.overfit_steps,
            "training": tiny_evidence,
            "artifact": _file_record(tiny_path, root=args.output_root),
        }
        del tiny_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "schema": "token-reconstruction.trr0003-track-b-fit.v1",
        "task_id": TASK_ID,
        "track": TRACK_ID,
        "command": command_record(),
        "git_commit": _git_commit(),
        "code_sha256": _code_hash(),
        "decoder_source_sha256": decoder_source_hash(),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "cut_depth": 4},
        "fit_inputs": {
            "observations": _file_record(observations_path),
            "truth": _file_record(truth_path),
            "embedding_table": _file_record(embedding_path),
            "records": int(observations.shape[0]),
            "scored_examples": int(x.shape[0]),
            "public_labels": True,
        },
        "public_validation": eval_evidence,
        "methods": method_evidence,
        "phases": timer.records,
        "peak_memory": peak_memory(),
        "notes": [
            "All methods train on public inverse_train labels and public-base cut4 activations.",
            "Learning curves use only disjoint public auxiliary validation records; evaluator-private panel truth stays closed until shared scoring.",
            "The tiny-subset arms are diagnostics and are never used for deployed prediction.",
            "The tied CE projection retains the public embedding table as a fixed runtime resource.",
            "No target or canonical truth is used for fitting, routing, or stopping.",
        ],
    }
    write_json_exclusive(args.output_root / "fit_evidence.json", result)
    print(json.dumps({"status": "fit_complete", "output_root": str(args.output_root), "methods": methods}))
    return 0



def _state_hashes(fit_root: Path, methods: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for method in methods:
        path = fit_root / f"{method}.safetensors"
        result[method] = sha256_file(path)
    return result


def _predict(args: argparse.Namespace) -> int:
    methods = _methods(args.methods)
    _ensure_new_directory(args.output_root)
    device = _device(args.device)
    observations = _load_observations(args.observations)
    embeddings = _load_tensor(args.embedding_table, "embeddings").float().contiguous()
    validate_embedding_table(embeddings, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE)
    embedding_hash = tensor_sha256(embeddings)
    state_hashes = _state_hashes(args.fit_root, methods)
    seed_everything(1738)
    predictions: dict[str, torch.Tensor] = {}
    timer = PhaseTimer()
    flat_observations = _flatten_post_bos(observations)
    for method in methods:
        if method == ANGULAR_METHOD:
            model = load_inverse(
                args.fit_root / f"{method}.safetensors",
                hidden_size=HIDDEN_SIZE,
                device=device,
            )
            with timer.measure(f"predict_{method}"):
                flat = angular_prediction_tensor(
                    model,
                    flat_observations,
                    embeddings,
                    device=device,
                    batch_size=args.batch_size,
                )
        else:
            model = load_token_decoder(
                args.fit_root / f"{method}.safetensors",
                method_id=method,
                hidden_size=HIDDEN_SIZE,
                vocab_size=VOCAB_SIZE,
                device=device,
            )
            with timer.measure(f"predict_{method}"):
                flat = prediction_tensor(
                    model,
                    flat_observations,
                    embeddings,
                    device=device,
                    batch_size=args.batch_size,
                )
        full = torch.full(
            (observations.shape[0], observations.shape[1]),
            BOS_TOKEN_ID,
            dtype=torch.int32,
        )
        full[:, 1:] = flat.reshape(observations.shape[0], observations.shape[1] - 1)
        predictions[method] = full
        del model, flat
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    complete_prediction_check(
        predictions,
        expected_methods=methods,
        expected_shape=(int(observations.shape[0]), int(observations.shape[1])),
        vocab_size=VOCAB_SIZE,
    )
    path = args.output_root / "predictions.safetensors"
    code_hash = _code_hash()
    save_predictions(
        predictions,
        path,
        input_sha256=tensor_sha256(observations),
        embedding_sha256=embedding_hash,
        method_state_hashes=state_hashes,
        code_sha256=code_hash,
        metadata={
            "task_id": TASK_ID,
            "cut_depth": "4",
            "records": str(observations.shape[0]),
            "positions": str(observations.shape[1]),
            "known_bos_emitted": "true",
            "public_prefix_calls": "0",
            "candidate_simulations": "0",
        },
    )
    evidence = {
        "schema": "token-reconstruction.trr0003-track-b-predict.v1",
        "task_id": TASK_ID,
        "track": TRACK_ID,
        "command": command_record(),
        "git_commit": _git_commit(),
        "code_sha256": code_hash,
        "decoder_source_sha256": decoder_source_hash(),
        "input": _file_record(args.observations),
        "input_tensor_sha256": tensor_sha256(observations),
        "embedding_table": _file_record(args.embedding_table),
        "embedding_tensor_sha256": embedding_hash,
        "fit_root": str(args.fit_root.resolve()),
        "method_state_hashes": state_hashes,
        "methods": list(methods),
        "geometry": {
            "records": int(observations.shape[0]),
            "positions": int(observations.shape[1]),
        },
        "phases": timer.records,
        "candidate_simulations": 0,
        "public_prefix_calls": 0,
        "output": _file_record(path, root=args.output_root),
        "peak_memory": peak_memory(),
        "truth_opened": False,
        "decision_rule": "direct argmax token per position; BOS is fixed to 128000; no A2 fallback",
    }
    write_json_exclusive(args.output_root / "prediction_evidence.json", evidence)
    print(json.dumps({"status": "predictions_frozen", "output": str(path), "methods": methods}))
    return 0

def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "prepare":
            return _prepare(args)
        if args.command == "fit":
            return _fit(args)
        if args.command == "predict":
            return _predict(args)
        raise StandaloneDecoderError(f"unknown command: {args.command}")
    except (StandaloneDecoderError, RuntimeError, OSError, ValueError) as exc:
        print(f"TRR-0003 Track B failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

