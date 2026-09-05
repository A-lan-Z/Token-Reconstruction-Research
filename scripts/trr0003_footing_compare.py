#!/usr/bin/env python3
"""Run and score the small TRR-0003 shared-panel comparator.

``reconstruct`` reads only the sanitized panel, its hashed public activation
assets, and public method/model state.  It never loads the private sidecar.
``freeze`` makes the complete method matrix immutable.  ``score`` verifies that
frozen state first and only then opens a caller-provided private sidecar.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoModelForCausalLM

from token_reconstruction.a1a2_configuration_search import (
    PolicySpec,
    ResolvedPolicy,
    decode_policy,
)
from token_reconstruction.component_crossover import (
    prediction_from_rank_one,
    propose_public_a1,
    propose_residual_affine,
)
from token_reconstruction.dual_benchmark import score_predictions
from token_reconstruction.experiment_runtime import peak_memory, seed_everything, synchronize
from token_reconstruction.footing import (
    BOS_TOKEN_ID,
    CONDITION_ORDER,
    CUT_DEPTH,
    HIDDEN_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    PREDICTION_SCHEMA,
    STYLE_ORDER,
    FootingError,
    expected_prediction_path,
    file_record,
    load_all_cells,
    load_panel,
    make_binding,
    sha256_file,
    validate_complete_prediction_set,
    validate_before_truth,
)
from token_reconstruction.inverse import load_inverse


METHOD_IDS = ("historical_alpaca_a1", "frozen_a1_a2_k256", "direct_inverse")
DEFAULT_REFERENCE = Path(
    "outputs/TRR-0002/configuration-search/fresh-blind-code/reference/strict_bos/round001_teacher.py"
)
DEFAULT_LENS = Path("outputs/TRR-0002/blind/reconstructor_input/public_a1_lens.pt")
DEFAULT_INVERSE = Path("outputs/TRR-0001/reconstructor_public/inverses/cut4.safetensors")
DEFAULT_MODEL = Path(
    "/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
    + MODEL_REVISION
)


class ComparatorError(RuntimeError):
    """Raised when a comparator run violates the shared panel contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ComparatorError(f"JSON input unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparatorError(f"invalid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise ComparatorError(f"JSON root is not an object: {path}")
    return value


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ComparatorError(f"refusing to overwrite artifact: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _import_reference(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ComparatorError(f"public prefix reference unavailable: {path}")
    spec = importlib.util.spec_from_file_location("trr0003_public_reference", path)
    if spec is None or spec.loader is None:
        raise ComparatorError("unable to import public prefix reference")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _current_commit() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ComparatorError("unable to resolve executable commit") from exc
    if len(value) != 40:
        raise ComparatorError("executable commit is not full")
    return value


def _public_model_binding(snapshot: Path) -> dict[str, Any]:
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ComparatorError(f"public model snapshot unavailable: {snapshot}")
    return {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot": str(snapshot),
        "local_files_only": True,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
    }


def _fixed_k256_policy() -> ResolvedPolicy:
    spec = PolicySpec(
        kind="fixed",
        score_rule="direct_cosine",
        schedule=(256,),
        fast_path_id="off",
        fast_path_threshold=None,
        routing_signal=None,
        gate_mode=None,
        terminal_action="commit_last_winner",
    )
    result = ResolvedPolicy(spec=spec, thresholds=())
    result.validate()
    return result


def _method_binding(
    *,
    root: Path,
    panel_path: Path,
    method_id: str,
    lens_path: Path,
    inverse_path: Path,
    reference_path: Path,
    snapshot: Path,
) -> dict[str, Any]:
    state_paths = [lens_path] if method_id != "direct_inverse" else [inverse_path]
    code_paths = [
        root / "src/token_reconstruction/footing.py",
        root / "src/token_reconstruction/component_crossover.py",
        root / "src/token_reconstruction/a1a2_configuration_search.py",
        root / "scripts/trr0003_footing_compare.py",
    ]
    if method_id == "frozen_a1_a2_k256":
        code_paths.append(reference_path)
    binding = make_binding(
        panel_path=panel_path,
        repository_root=root,
        method_state_paths=state_paths,
        code_paths=code_paths,
        code_commit=_current_commit(),
    )
    binding["public_model"] = _public_model_binding(snapshot)
    binding["method_rule"] = {
        "historical_alpaca_a1": "rank-one historical public Alpaca A1 proposal",
        "frozen_a1_a2_k256": "public Alpaca A1 top-256 plus fixed public-prefix direct-cosine K256",
        "direct_inverse": "existing residual-affine inverse rank-one proposal",
    }[method_id]
    return binding


def _write_prediction(
    *,
    path: Path,
    cell: Any,
    method_id: str,
    predictions: torch.Tensor,
    candidates: torch.Tensor,
    candidate_scores: torch.Tensor,
    binding: Mapping[str, Any],
    panel_sha: str,
) -> None:
    if path.exists() or path.is_symlink():
        raise ComparatorError(f"prediction artifact already exists: {path}")
    if predictions.shape != cell.attention_mask.shape:
        raise ComparatorError(f"prediction geometry changed for {cell.cell_id}")
    if candidates.shape[:2] != predictions.shape or candidate_scores.shape != candidates.shape:
        raise ComparatorError(f"candidate geometry changed for {cell.cell_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": PREDICTION_SCHEMA,
        "task_id": "TRR-0003",
        "panel_sha256": panel_sha,
        "cell_id": cell.cell_id,
        "style": cell.style,
        "condition": cell.condition,
        "method_id": method_id,
        "geometry_json": json.dumps(
            {
                "records": cell.records,
                "sequence_tokens": cell.sequence_tokens,
                "hidden_size": HIDDEN_SIZE,
                "cut_depth": CUT_DEPTH,
            },
            sort_keys=True,
        ),
        "binding_json": json.dumps(dict(binding), sort_keys=True),
    }
    save_file(
        {
            "predictions": predictions.detach().cpu().to(torch.int64).contiguous(),
            "candidates": candidates.detach().cpu().to(torch.int64).contiguous(),
            "candidate_scores": candidate_scores.detach().cpu().to(torch.float32).contiguous(),
        },
        str(path),
        metadata=metadata,
    )


def _load_public_model(
    *, snapshot: Path, reference_path: Path, lens_path: Path, inverse_path: Path, device: torch.device
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    if not torch.cuda.is_available() or device.type != "cuda":
        raise ComparatorError("shared comparator requires CUDA")
    reference = _import_reference(reference_path)
    started = time.perf_counter()
    full = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    full.requires_grad_(False)
    if full.config.hidden_size != HIDDEN_SIZE or full.config.vocab_size != 128256:
        raise ComparatorError("public model geometry changed")
    precut = reference.PublicP0Precut(full, (0, 1, 2, 3)).to(device).eval()
    embeddings = reference.normalize_public_embeddings(precut.embed_tokens.weight).to(device)
    lens = reference.load_frozen_lens(lens_path, device=device)
    inverse = load_inverse(inverse_path, hidden_size=HIDDEN_SIZE, device=device)
    del full
    synchronize()
    return precut, embeddings, lens, inverse, {
        "model_preparation_seconds": time.perf_counter() - started,
        "model": _public_model_binding(snapshot),
    }


def reconstruct(args: argparse.Namespace) -> int:
    root = args.repository_root.resolve()
    panel_path = args.panel.resolve()
    panel = load_panel(panel_path, repository_root=root)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
            raise ComparatorError("comparator output must be a new empty directory")
    else:
        output.mkdir(parents=True)
    snapshot = args.model.resolve()
    reference_path = args.reference.resolve()
    lens_path = args.lens.resolve()
    inverse_path = args.inverse.resolve()
    bindings = {
        method_id: _method_binding(
            root=root,
            panel_path=panel_path,
            method_id=method_id,
            lens_path=lens_path,
            inverse_path=inverse_path,
            reference_path=reference_path,
            snapshot=snapshot,
        )
        for method_id in METHOD_IDS
    }
    _write_exclusive(output / "bindings.json", bindings)
    panel_sha = sha256_file(panel_path)
    device = torch.device("cuda")
    seed_everything(args.seed)
    model_started = time.perf_counter()
    precut, embeddings, lens, inverse, model_evidence = _load_public_model(
        snapshot=snapshot,
        reference_path=reference_path,
        lens_path=lens_path,
        inverse_path=inverse_path,
        device=device,
    )
    model_evidence["model_load_wall_seconds"] = time.perf_counter() - model_started
    policy = _fixed_k256_policy()
    cells = load_all_cells(panel, repository_root=root)
    method_timings: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_IDS}
    for cell_index, cell in enumerate(cells):
        observations = cell.activations
        mask = cell.attention_mask
        positions = cell.position_ids
        for method_id in METHOD_IDS:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            synchronize()
            started = time.perf_counter()
            if method_id == "historical_alpaca_a1":
                proposal = propose_public_a1(
                    observations=observations,
                    attention_mask=mask,
                    lens=lens,
                    normalized_embeddings=embeddings,
                    max_k=512,
                    chunk=256,
                )
                predictions = prediction_from_rank_one(proposal.candidates, mask)
                candidates = proposal.candidates[:, :, :16].contiguous()
                scores = proposal.scores[:, :, :16].contiguous()
                simulations = 0
                prefix_tokens = 0
                rule = "rank_one"
            elif method_id == "direct_inverse":
                proposal = propose_residual_affine(
                    observations=observations,
                    attention_mask=mask,
                    inverse=inverse,
                    embedding_table=embeddings,
                    max_k=512,
                )
                predictions = prediction_from_rank_one(proposal.candidates, mask)
                candidates = proposal.candidates[:, :, :16].contiguous()
                scores = proposal.scores[:, :, :16].contiguous()
                simulations = 0
                prefix_tokens = 0
                rule = "rank_one"
            else:
                proposal = propose_public_a1(
                    observations=observations,
                    attention_mask=mask,
                    lens=lens,
                    normalized_embeddings=embeddings,
                    max_k=512,
                    chunk=256,
                )
                decoded = decode_policy(
                    observations=observations,
                    attention_mask=mask,
                    position_ids=positions,
                    candidates=proposal.candidates[:, :, :256].contiguous(),
                    a1_confidence=proposal.top1_confidence,
                    precut=precut,
                    device=device,
                    policy=policy,
                    record_batch_size=4,
                )
                predictions = decoded.predictions
                candidates = proposal.candidates[:, :, :256].contiguous()
                scores = proposal.scores[:, :, :256].contiguous()
                simulations = decoded.executed_candidate_simulations
                prefix_tokens = decoded.prefix_commit_tokens
                rule = "fixed_k256_public_prefix"
            synchronize()
            elapsed = time.perf_counter() - started
            path = expected_prediction_path(output, cell=cell, method_id=method_id)
            _write_prediction(
                path=path,
                cell=cell,
                method_id=method_id,
                predictions=predictions,
                candidates=candidates,
                candidate_scores=scores,
                binding=bindings[method_id],
                panel_sha=panel_sha,
            )
            timing = {
                "method_id": method_id,
                "cell_id": cell.cell_id,
                "records": cell.records,
                "active_tokens": int(mask.to(torch.bool).sum().item()),
                "scored_tokens": int(mask.to(torch.bool).sum().item()) - cell.records,
                "cold_start": cell_index == 0,
                "elapsed_seconds": elapsed,
                "per_record_seconds": elapsed / cell.records,
                "per_scored_token_seconds": elapsed / max(1, int(mask.to(torch.bool).sum().item()) - cell.records),
                "proposal_seconds": float(getattr(proposal, "elapsed_seconds", elapsed)),
                "candidate_budget": int(candidates.shape[2]),
                "candidate_simulations": int(simulations),
                "prefix_commit_tokens": int(prefix_tokens),
                "peak_memory": peak_memory(),
                "rule": rule,
                "artifact": str(path.relative_to(root).as_posix()),
            }
            method_timings[method_id].append(timing)
            evidence = {
                "schema": "token-reconstruction.trr0003-footing-cell-evidence.v1",
                "task_id": "TRR-0003",
                "created_utc": utc_now(),
                "cell": {"id": cell.cell_id, "style": cell.style, "condition": cell.condition, "shape": list(cell.shape)},
                "method": timing,
                "model": model_evidence["model"],
                "record_batch_size": 4,
                "public_prefix_calls": 0,
                "binding_sha256": sha256_file(path),
            }
            _write_exclusive(output / cell.style / cell.condition / f"{method_id}.evidence.json", evidence)
    validate_complete_prediction_set(
        output,
        panel=panel,
        panel_path=panel_path,
        repository_root=root,
        method_ids=METHOD_IDS,
        expected_bindings=bindings,
    )
    evidence = {
        "schema": "token-reconstruction.trr0003-footing-run-evidence.v1",
        "task_id": "TRR-0003",
        "created_utc": utc_now(),
        "panel": file_record(panel_path, repository_root=root),
        "model": model_evidence,
        "method_ids": list(METHOD_IDS),
        "timing": method_timings,
        "preparation_and_steady_state": {
            "cold_start": "model preparation plus first cell per method",
            "steady_state": "subsequent cells of each method; no cross-cell state is reused",
        },
        "status": "PUBLIC_PREDICTIONS_COMPLETE_BEFORE_FREEZE",
    }
    _write_exclusive(output / "run_evidence.json", evidence)
    print(json.dumps({"output": str(output), "cells": len(cells), "methods": list(METHOD_IDS)}, indent=2))
    return 0


def freeze(args: argparse.Namespace) -> int:
    from token_reconstruction.freeze import create_freeze_receipt, verify_freeze_receipt

    root = args.repository_root.resolve()
    output = args.output.resolve()
    panel_path = args.panel.resolve()
    plan_path = args.plan.resolve()
    receipt = args.receipt.resolve()
    panel = load_panel(panel_path, repository_root=root)
    bindings = _load_json(output / "bindings.json")
    expected = {method: bindings[method] for method in METHOD_IDS}
    validate_complete_prediction_set(
        output,
        panel=panel,
        panel_path=panel_path,
        repository_root=root,
        method_ids=METHOD_IDS,
        expected_bindings=expected,
    )
    commit = args.preregistration_commit or _current_commit()
    payload = create_freeze_receipt(
        repository_root=root,
        frozen_root=output,
        plan_path=plan_path,
        receipt_path=receipt,
        preregistration_commit=commit,
        created_utc=utc_now(),
        metadata={
            "task_id": "TRR-0003",
            "panel_sha256": sha256_file(panel_path),
            "method_ids": list(METHOD_IDS),
            "output_contract": "style__condition__method Cartesian complete",
            "truth_opened": False,
        },
    )
    verify_freeze_receipt(receipt, repository_root=root)
    print(json.dumps({"receipt": str(receipt), "entries": len(payload["entries"])}, indent=2))
    return 0


def _truth_for_cells(path: Path, cells: tuple[Any, ...]) -> dict[str, torch.Tensor]:
    """Load the private sidecar; callers invoke this only after the gate."""

    if path.is_symlink() or not path.is_file():
        raise ComparatorError(f"private sidecar unavailable: {path}")
    state = load_file(path, device="cpu")
    expected = {f"{cell.cell_id}__token_ids" for cell in cells}
    if set(state) != expected:
        raise ComparatorError("private sidecar cell set changed")
    result: dict[str, torch.Tensor] = {}
    for cell in cells:
        tensor = state[f"{cell.cell_id}__token_ids"]
        if tuple(tensor.shape) != cell.attention_mask.shape or tensor.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64
        ):
            raise ComparatorError(f"private sidecar geometry changed for {cell.cell_id}")
        if tensor.lt(0).any().item() or tensor.ge(128256).any().item():
            raise ComparatorError(f"private sidecar token range changed for {cell.cell_id}")
        if tensor[:, 0].ne(BOS_TOKEN_ID).any().item():
            raise ComparatorError(f"private sidecar BOS changed for {cell.cell_id}")
        result[cell.cell_id] = tensor.to(torch.long)
    return result


def score(args: argparse.Namespace) -> int:
    root = args.repository_root.resolve()
    output = args.output.resolve()
    panel_path = args.panel.resolve()
    panel = load_panel(panel_path, repository_root=root)
    bindings = _load_json(output / "bindings.json")
    expected = {method: bindings[method] for method in METHOD_IDS}
    receipt_payload = validate_before_truth(
        receipt_path=args.receipt.resolve(),
        repository_root=root,
        truth_path=args.truth.resolve(),
        output_root=output,
        panel=panel,
        panel_path=panel_path,
        method_ids=METHOD_IDS,
        expected_bindings=expected,
    )
    # This is the first private read in this process.  Keep it after the gate.
    cells = load_all_cells(panel, repository_root=root)
    truth = _truth_for_cells(args.truth.resolve(), cells)
    rows: dict[str, Any] = {}
    for cell in cells:
        for method_id in METHOD_IDS:
            path = expected_prediction_path(output, cell=cell, method_id=method_id)
            with safe_open(path, framework="pt", device="cpu") as handle:
                predictions = handle.get_tensor("predictions").to(torch.long)
                candidates = handle.get_tensor("candidates").to(torch.long)
            metrics, per_record = score_predictions(
                predictions=predictions,
                truth=truth[cell.cell_id],
                attention_mask=cell.attention_mask,
                candidates=candidates,
                record_ids=cell.record_ids,
            )
            rows[f"{cell.cell_id}__{method_id}"] = {
                "cell_id": cell.cell_id,
                "style": cell.style,
                "condition": cell.condition,
                "method_id": method_id,
                "metrics": metrics,
                "per_record": per_record,
                "timing_path": str((output / cell.style / cell.condition / f"{method_id}.evidence.json").relative_to(root).as_posix()),
            }
    result = {
        "schema": "token-reconstruction.trr0003-footing-result.v1",
        "task_id": "TRR-0003",
        "status": "RETROSPECTIVE_DEVELOPMENT_PANEL_SCORED_AFTER_FREEZE",
        "scored_utc": utc_now(),
        "freeze": {
            "receipt": file_record(args.receipt.resolve(), repository_root=root),
            "verified_before_private_read": True,
            "receipt_metadata": receipt_payload.get("metadata"),
        },
        "panel": file_record(panel_path, repository_root=root),
        "comparison_status": {
            "canonical_new_methods": "NOT_RUN",
            "dual_benchmark": "INCOMPLETE",
            "claim_scope": "pilot diagnostic only",
        },
        "cells": rows,
        "truth_sidecar": {"path": str(args.truth.resolve()), "opened_after_gate": True},
    }
    _write_exclusive(args.result.resolve(), result)
    print(json.dumps({"result": str(args.result.resolve()), "cells": len(rows)}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    recon = sub.add_parser("reconstruct")
    recon.add_argument("--repository-root", type=Path, default=Path("."))
    recon.add_argument("--panel", type=Path, default=Path("experiments/TRR-0003/footing/panel.json"))
    recon.add_argument("--output", type=Path, required=True)
    recon.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    recon.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    recon.add_argument("--lens", type=Path, default=DEFAULT_LENS)
    recon.add_argument("--inverse", type=Path, default=DEFAULT_INVERSE)
    recon.add_argument("--seed", type=int, default=3003)
    frz = sub.add_parser("freeze")
    frz.add_argument("--repository-root", type=Path, default=Path("."))
    frz.add_argument("--panel", type=Path, default=Path("experiments/TRR-0003/footing/panel.json"))
    frz.add_argument("--plan", type=Path, default=Path("experiments/TRR-0003/footing/plan.json"))
    frz.add_argument("--output", type=Path, required=True)
    frz.add_argument("--receipt", type=Path, required=True)
    frz.add_argument("--preregistration-commit", default=None)
    scr = sub.add_parser("score")
    scr.add_argument("--repository-root", type=Path, default=Path("."))
    scr.add_argument("--panel", type=Path, default=Path("experiments/TRR-0003/footing/panel.json"))
    scr.add_argument("--output", type=Path, required=True)
    scr.add_argument("--receipt", type=Path, required=True)
    scr.add_argument("--truth", type=Path, required=True)
    scr.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "reconstruct":
            return reconstruct(args)
        if args.command == "freeze":
            return freeze(args)
        return score(args)
    except (ComparatorError, FootingError) as exc:
        raise SystemExit(f"TRR-0003 comparator error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
