#!/usr/bin/env python3
"""Emit the Track B standalone-decoder prediction cells.

The process reads only the sanitized public panel and public decoder state.
Each cell is written in the shared footing layout before any private truth
sidecar is opened.  Predictions are direct argmax outputs; no candidate list,
public-prefix call, target model, or source token labels are available here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from token_reconstruction.experiment_runtime import seed_everything, synchronize, utc_now
from token_reconstruction.footing import (
    BOS_TOKEN_ID,
    CUT_DEPTH,
    HIDDEN_SIZE,
    INVALID_TOKEN_ID,
    MODEL_ID,
    MODEL_REVISION,
    PREDICTION_SCHEMA,
    FootingError,
    expected_prediction_path,
    file_record,
    load_all_cells,
    load_panel,
    make_binding,
    sha256_file,
    tensor_sha256,
    validate_complete_prediction_set,
    validate_prediction_artifact,
)
from token_reconstruction.inverse import load_inverse
from token_reconstruction.standalone_decoder import (
    angular_prediction_tensor,
    decoder_source_hash,
    load_token_decoder,
    prediction_tensor,
    validate_embedding_table,
)


TASK_ID = "TRR-0003"
TRACK_ID = "track_b"
METHOD_ORDER = (
    "angular_inverse_control",
    "tied_affine_token_ce",
    "residual_mlp256_token_ce",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root", type=Path, default=Path("."), help="repository root"
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("experiments/TRR-0003/footing/panel.json"),
    )
    parser.add_argument(
        "--embedding-table",
        type=Path,
        default=Path("outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors"),
    )
    parser.add_argument(
        "--fit-root",
        type=Path,
        default=Path("outputs/TRR-0003/track_b/selected_checkpoints_v1"),
    )
    parser.add_argument(
        "--selection-amendment",
        type=Path,
        default=Path("experiments/TRR-0003/track_b/checkpoint_selection_amendment.json"),
    )
    parser.add_argument(
        "--curve-evidence",
        type=Path,
        default=Path("outputs/TRR-0003/track_b/extended_fit_1800_v1/fit_evidence.json"),
    )
    parser.add_argument(
        "--fit-evidence",
        type=Path,
        default=Path("outputs/TRR-0003/track_b/extended_fit_1800_v1/fit_evidence.json"),
    )
    parser.add_argument(
        "--selected-replay-evidence",
        type=Path,
        default=Path("outputs/TRR-0003/track_b/selected_checkpoints_v1/selected_replay_evidence.json"),
    )
    parser.add_argument(
        "--fit-prepare-evidence",
        type=Path,
        default=Path("outputs/TRR-0003/track_b/public_fit_v2/prepare_evidence.json"),
    )
    parser.add_argument(
        "--fit-records",
        type=Path,
        default=Path("outputs/TRR-0003/track_b/public_fit_v2/fit_records.json"),
    )
    parser.add_argument(
        "--fit-observations",
        type=Path,
        default=Path("outputs/TRR-0003/track_b/public_fit_v2/fit_observations.safetensors"),
    )
    parser.add_argument(
        "--fit-truth",
        type=Path,
        default=Path("outputs/TRR-0003/track_b/public_fit_v2/fit_truth.safetensors"),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("experiments/TRR-0003/track_b/plan.json"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def _commit(root: Path) -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()
    if len(value) != 40:
        raise FootingError("executable commit is not a full hash")
    return value


def _load_embedding(path: Path) -> torch.Tensor:
    if path.is_symlink() or not path.is_file():
        raise FootingError(f"embedding table is unavailable: {path}")
    state = load_file(path.resolve(), device="cpu")
    if set(state) != {"embeddings"}:
        raise FootingError("embedding table tensor fields changed")
    value = state["embeddings"].contiguous()
    validate_embedding_table(value, hidden_size=HIDDEN_SIZE, vocab_size=128256)
    return value.float().contiguous()


def _write_prediction(
    *,
    path: Path,
    cell: Any,
    method_id: str,
    predictions: torch.Tensor,
    binding: Mapping[str, Any],
    panel_sha256: str,
    input_sha256: str,
) -> None:
    if path.exists() or path.is_symlink():
        raise FootingError(f"prediction output is not create-only: {path}")
    if tuple(predictions.shape) != tuple(cell.attention_mask.shape):
        raise FootingError(f"prediction geometry changed for {cell.cell_id}")
    if predictions.dtype not in (torch.int32, torch.int64):
        raise FootingError("predictions must be integer")
    path.parent.mkdir(parents=True, exist_ok=True)
    geometry = {
        "records": cell.records,
        "sequence_tokens": cell.sequence_tokens,
        "hidden_size": HIDDEN_SIZE,
        "cut_depth": CUT_DEPTH,
    }
    metadata = {
        "schema": PREDICTION_SCHEMA,
        "task_id": TASK_ID,
        "panel_sha256": panel_sha256,
        "cell_id": cell.cell_id,
        "style": cell.style,
        "condition": cell.condition,
        "method_id": method_id,
        "geometry_json": json.dumps(geometry, sort_keys=True),
        "binding_json": json.dumps(dict(binding), sort_keys=True),
        "input_tensor_sha256": input_sha256,
        "attention_mask_sha256": tensor_sha256(cell.attention_mask),
        "position_ids_sha256": tensor_sha256(cell.position_ids),
        "candidate_simulations": "0",
        "public_prefix_calls": "0",
        "truth_opened": "false",
        "decision_rule": "direct argmax per post-BOS activation; BOS fixed at 128000; padding marked -1",
    }
    save_file(
        {"predictions": predictions.detach().cpu().to(torch.int64).contiguous()},
        str(path),
        metadata=metadata,
    )


def _full_predictions(cell: Any, flat: torch.Tensor) -> torch.Tensor:
    expected = cell.records * (cell.sequence_tokens - 1)
    if flat.ndim != 1 or flat.shape[0] != expected:
        raise FootingError(f"prediction coverage changed for {cell.cell_id}")
    value = torch.full(
        (cell.records, cell.sequence_tokens),
        BOS_TOKEN_ID,
        dtype=torch.int64,
    )
    value[:, 1:] = flat.reshape(cell.records, cell.sequence_tokens - 1).to(torch.int64)
    active = cell.attention_mask.to(torch.bool)
    if not active[:, 0].all().item():
        raise FootingError(f"panel masks do not include BOS for {cell.cell_id}")
    value[~active] = INVALID_TOKEN_ID
    value[:, 0] = BOS_TOKEN_ID
    return value


def _state_paths(args: argparse.Namespace, method: str) -> list[Path]:
    return [
        args.fit_root / f"{method}.safetensors",
        args.embedding_table,
        args.fit_observations,
        args.fit_truth,
        args.fit_records,
        args.selection_amendment,
        args.curve_evidence,
        args.fit_evidence,
        args.selected_replay_evidence,
        args.fit_prepare_evidence,
        args.plan,
    ]


def _validate_selected_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Bind state bytes to the public-only replay manifest before loading models."""

    try:
        replay = json.loads(args.selected_replay_evidence.read_text(encoding="utf-8"))
        amendment = json.loads(args.selection_amendment.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FootingError("selected checkpoint manifests are unreadable") from exc
    if replay.get("schema") != "token-reconstruction.trr0003-track-b-selected-replay.v1":
        raise FootingError("selected replay manifest identity changed")
    if amendment.get("schema") != "token-reconstruction.trr0003-track-b-checkpoint-selection-amendment.v1":
        raise FootingError("checkpoint selection amendment identity changed")
    selected = amendment.get("selected")
    replay_steps = replay.get("selected_steps")
    if not isinstance(selected, Mapping) or not isinstance(replay_steps, Mapping):
        raise FootingError("selected checkpoint steps are absent")
    for method in METHOD_ORDER:
        if int(selected[method]["step"]) != int(replay_steps[method]):
            raise FootingError(f"selected step changed for {method}")
        state_path = args.fit_root / f"{method}.safetensors"
        actual_sha = sha256_file(state_path)
        declared = replay.get("artifacts", {}).get(method, {})
        declared_sha = declared.get("sha256")
        if declared_sha != actual_sha:
            raise FootingError(f"selected decoder state hash differs from replay manifest: {method}")
        training_declared = replay.get("training", {}).get(method, {}).get("state_sha256")
        if training_declared != actual_sha:
            raise FootingError(f"selected decoder state training hash differs: {method}")
    return {"replay": replay, "amendment": amendment}


def _code_paths(root: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        root / "scripts/trr0003_track_b.py",
        root / "src/token_reconstruction/standalone_decoder.py",
        root / "src/token_reconstruction/inverse.py",
        root / "src/token_reconstruction/experiment_runtime.py",
        root / "src/token_reconstruction/footing.py",
    ]


def main() -> int:
    args = _parser().parse_args()
    root = args.repository_root.resolve()
    def rooted(path: Path) -> Path:
        return (root / path).resolve() if not path.is_absolute() else path.resolve()

    panel_path = rooted(args.panel)
    output = rooted(args.output_root)
    for name in (
        "embedding_table",
        "fit_root",
        "selection_amendment",
        "curve_evidence",
        "fit_evidence",
        "selected_replay_evidence",
        "fit_prepare_evidence",
        "fit_records",
        "fit_observations",
        "fit_truth",
        "plan",
    ):
        setattr(args, name, rooted(getattr(args, name)))
    if output.exists() or output.is_symlink():
        raise FootingError(f"prediction output root must be create-only: {output}")
    if args.batch_size <= 0:
        raise FootingError("batch size must be positive")
    output.mkdir(parents=True)

    selected_manifest = _validate_selected_manifest(args)
    started_utc = utc_now()
    panel = load_panel(panel_path, repository_root=root)
    cells = load_all_cells(panel, repository_root=root)
    panel_record = file_record(panel_path, repository_root=root)
    panel_sha = panel_record["sha256"]
    embedding_path = args.embedding_table.resolve()
    embedding_load_started = time.perf_counter()
    embedding_cpu = _load_embedding(embedding_path)
    embedding_load_validation_seconds = time.perf_counter() - embedding_load_started
    embedding_hash_started = time.perf_counter()
    embedding_sha = tensor_sha256(embedding_cpu)
    embedding_hash_seconds = time.perf_counter() - embedding_hash_started
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise FootingError("CUDA requested but unavailable")
    seed_everything(1738)
    transfer_started = time.perf_counter()
    embedding_runtime = embedding_cpu.to(device=device, dtype=torch.float32)
    synchronize()
    embedding_transfer_seconds = time.perf_counter() - transfer_started

    commit = _commit(root)
    code_paths = _code_paths(root)
    bindings = {
        method: make_binding(
            panel_path=panel_path,
            repository_root=root,
            method_state_paths=_state_paths(args, method),
            code_paths=code_paths,
            code_commit=commit,
        )
        for method in METHOD_ORDER
    }
    (output / "bindings.json").write_text(
        json.dumps(bindings, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    timings: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_ORDER}
    method_phases: list[dict[str, Any]] = []
    loaded_models: dict[str, torch.nn.Module] = {}
    for method in METHOD_ORDER:
        state_path = args.fit_root / f"{method}.safetensors"
        started = time.perf_counter()
        if method == "angular_inverse_control":
            model = load_inverse(state_path, hidden_size=HIDDEN_SIZE, device=device)
        else:
            model = load_token_decoder(
                state_path,
                method_id=method,
                hidden_size=HIDDEN_SIZE,
                vocab_size=128256,
                device=device,
                logit_scale=16.0,
                bottleneck_size=256,
            )
        synchronize()
        method_phases.append(
            {
                "phase": f"load_{method}",
                "elapsed_seconds": time.perf_counter() - started,
                "state": file_record(state_path, repository_root=root),
            }
        )
        loaded_models[method] = model

    for cell in cells:
        flat_observations = cell.activations[:, 1:, :].reshape(-1, HIDDEN_SIZE).contiguous()
        input_sha = tensor_sha256(cell.activations)
        for method in METHOD_ORDER:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize()
            started = time.perf_counter()
            model = loaded_models[method]
            if method == "angular_inverse_control":
                flat = angular_prediction_tensor(
                    model,
                    flat_observations,
                    embedding_runtime,
                    device=device,
                    batch_size=args.batch_size,
                )
            else:
                flat = prediction_tensor(
                    model,
                    flat_observations,
                    embedding_runtime,
                    device=device,
                    batch_size=args.batch_size,
                )
            synchronize()
            first_inference_seconds = time.perf_counter() - started
            warm_started = time.perf_counter()
            if method == "angular_inverse_control":
                warm_flat = angular_prediction_tensor(
                    model,
                    flat_observations,
                    embedding_runtime,
                    device=device,
                    batch_size=args.batch_size,
                )
            else:
                warm_flat = prediction_tensor(
                    model,
                    flat_observations,
                    embedding_runtime,
                    device=device,
                    batch_size=args.batch_size,
                )
            synchronize()
            warm_inference_seconds = time.perf_counter() - warm_started
            warm_repeat_exact = bool(torch.equal(flat, warm_flat))
            if not warm_repeat_exact:
                raise FootingError(f"same-cell warm prediction changed for {cell.cell_id}/{method}")
            predictions = _full_predictions(cell, flat)
            path = expected_prediction_path(output, cell=cell, method_id=method)
            io_started = time.perf_counter()
            _write_prediction(
                path=path,
                cell=cell,
                method_id=method,
                predictions=predictions,
                binding=bindings[method],
                panel_sha256=panel_sha,
                input_sha256=input_sha,
            )
            io_seconds = time.perf_counter() - io_started
            validate_prediction_artifact(
                path,
                cell=cell,
                panel_sha256=panel_sha,
                expected_method_id=method,
                expected_binding=bindings[method],
                repository_root=root,
            )
            timings[method].append(
                {
                    "cell_id": cell.cell_id,
                    "records": cell.records,
                    "sequence_tokens": cell.sequence_tokens,
                    "active_scored_tokens": int(cell.attention_mask[:, 1:].sum().item()),
                    "first_inference_seconds": first_inference_seconds,
                    "warm_inference_seconds": warm_inference_seconds,
                    "warm_repeat_exact": warm_repeat_exact,
                    "serialization_seconds": io_seconds,
                    "peak_memory": {
                        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else None,
                        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
                        if device.type == "cuda"
                        else None,
                    },
                    "prediction": file_record(path, repository_root=root),
                    "input_tensor_sha256": input_sha,
                    "embedding_transfer_once": True,
                }
            )
            del flat, warm_flat, predictions

    for model in loaded_models.values():
        del model
    validate_complete_prediction_set(
        output,
        panel=panel,
        panel_path=panel_path,
        repository_root=root,
        method_ids=METHOD_ORDER,
        expected_bindings=bindings,
        candidate_policies={method: "forbidden" for method in METHOD_ORDER},
    )
    timing_summary: dict[str, Any] = {}
    for method, rows in timings.items():
        timing_summary[method] = {
            "cells": len(rows),
            "warm_repeat_exact_all_cells": all(row["warm_repeat_exact"] for row in rows),
            "total_first_inference_seconds": sum(row["first_inference_seconds"] for row in rows),
            "total_warm_inference_seconds": sum(row["warm_inference_seconds"] for row in rows),
            "total_serialization_seconds": sum(row["serialization_seconds"] for row in rows),
            "max_cuda_peak_allocated_bytes": max(
                (row["peak_memory"]["cuda_peak_allocated_bytes"] or 0 for row in rows),
                default=0,
            ),
            "max_cuda_peak_reserved_bytes": max(
                (row["peak_memory"]["cuda_peak_reserved_bytes"] or 0 for row in rows),
                default=0,
            ),
        }
    evidence = {
        "schema": "token-reconstruction.trr0003-track-b-predict-cells.v1",
        "task_id": TASK_ID,
        "track": TRACK_ID,
        "status": "PUBLIC_PREDICTIONS_COMPLETE_BEFORE_FREEZE",
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "git_commit": commit,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "cut_depth": CUT_DEPTH},
        "panel": panel_record,
        "methods": list(METHOD_ORDER),
        "bindings": bindings,
        "fit": {
            "root": str(args.fit_root.resolve()),
            "selected_manifest_verified": True,
            "selected_steps": selected_manifest["replay"]["selected_steps"],
            "selection_amendment": file_record(args.selection_amendment, repository_root=root),
            "curve_evidence": file_record(args.curve_evidence, repository_root=root),
            "fit_evidence": file_record(args.fit_evidence, repository_root=root),
            "selected_replay_evidence": file_record(args.selected_replay_evidence, repository_root=root),
            "fit_prepare_evidence": file_record(args.fit_prepare_evidence, repository_root=root),
            "fit_records": file_record(args.fit_records, repository_root=root),
            "fit_observations": file_record(args.fit_observations, repository_root=root),
            "fit_truth": file_record(args.fit_truth, repository_root=root),
            "embedding_table": file_record(embedding_path, repository_root=root),
            "embedding_tensor_sha256": embedding_sha,
        },
        "runtime": {
            "device": str(device),
            "batch_size": args.batch_size,
            "embedding_load_validation_seconds": embedding_load_validation_seconds,
            "embedding_hash_seconds": embedding_hash_seconds,
            "embedding_transfer_seconds": embedding_transfer_seconds,
            "embedding_runtime_bytes": int(embedding_runtime.numel() * embedding_runtime.element_size()),
            "candidate_simulations": 0,
            "public_prefix_calls": 0,
            "truth_opened": False,
            "target_weights_loaded": False,
            "source_tokens_loaded": False,
        },
        "method_load_phases": method_phases,
        "cells": timings,
        "timing_summary": timing_summary,
        "code_paths": [file_record(path, repository_root=root) for path in code_paths],
        "decoder_source_sha256": decoder_source_hash(),
        "notes": [
            "The adapter reads only the sanitized public panel and validates every artifact before completion.",
            "The fixed public embedding table is transferred to the runtime device once before prediction.",
            "Padding positions are emitted as -1; BOS is fixed to 128000; direct argmax is used at every active position.",
            "No A2 candidate simulation or target/public-prefix call is performed.",
        ],
    }
    evidence_path = output / "prediction_evidence.json"
    if evidence_path.exists() or evidence_path.is_symlink():
        raise FootingError("prediction evidence path is not create-only")
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"status": evidence["status"], "output": str(output), "cells": len(cells)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FootingError, OSError, RuntimeError, ValueError) as exc:
        print(f"TRR-0003 Track B prediction failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
