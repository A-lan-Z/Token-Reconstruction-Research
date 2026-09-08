"""Truth-free opened-fixture qualification for the native A1+A2 trace hook.

This is a small qualification adapter, not the P10 scientific runner.  It
reuses the hash-bound TRR-0010 registration, observation iterator, and native
A1+A2 loader.  The only added behavior is a temporary wrapper around
``propose_public_a1`` that records proposal IDs/scores at every actual adapter
call.  It compares trace-off and trace-on final IDs against the frozen PR20
A1+A2 predictions and writes a create-only trace/receipt under ``/tmp``.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import torch
from safetensors.torch import save_file

try:
    from .execution import (
        CELL_ORDER,
        ExecutionError,
        STORED_SEQUENCE_TOKENS,
        VOCABULARY_SIZE,
        validate_frozen_package,
        sha256_file,
        tensor_digest,
    )
except ImportError:  # direct ``python scripts/trr_p10/qualification.py`` invocation
    from scripts.trr_p10.execution import (
        CELL_ORDER,
        ExecutionError,
        STORED_SEQUENCE_TOKENS,
        VOCABULARY_SIZE,
        validate_frozen_package,
        sha256_file,
        tensor_digest,
    )


METHOD_ID = "frozen_a1_a2_k256"
PROPOSAL_BUDGET = 512
CANDIDATE_BUDGET = 256
QUALIFICATION_SCHEMA = "token-reconstruction.trr-p10-a1-trace-qualification.v1"
TRACE_SCHEMA = "token-reconstruction.trr-p10-a1-proposal-trace.v1"


class QualificationError(RuntimeError):
    """Raised when opened-fixture trace qualification fails closed."""


def _record(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"asset is unavailable or symlinked: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_old_modules(root: Path) -> tuple[Any, Any, Any, Any]:
    """Import the frozen PR20 modules only after the caller chooses that root."""
    for candidate in (root, root / "src", root / "scripts"):
        value = str(candidate.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)
    importlib.invalidate_caches()
    try:
        gate = importlib.import_module("scripts.trr0010_eval_gate")
        runner = importlib.import_module("scripts.trr0010_eval_runner")
        legacy = importlib.import_module("scripts.trr0004_predict_confirmation")
        footing = importlib.import_module("trr0003_footing_compare")
    except Exception as exc:
        raise QualificationError("frozen PR20 modules could not be imported") from exc
    return gate, runner, legacy, footing


def _guard(*, legacy: Any, args: argparse.Namespace, device: torch.device, started: float, output_root: Path, stage: str) -> dict[str, Any]:
    if time.perf_counter() - started > args.max_seconds:
        raise QualificationError(f"wall-time guard failed at {stage}")
    rss = int(__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss) * 1024
    if rss > int(args.maximum_rss_gib * 2**30):
        raise QualificationError(f"host RSS guard failed at {stage}: {rss}")
    try:
        available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines() if line.startswith("MemAvailable:"))
    except (OSError, StopIteration, ValueError) as exc:
        raise QualificationError(f"host availability guard unavailable at {stage}") from exc
    if available < int(args.minimum_host_available_gib * 2**30):
        raise QualificationError(f"host availability guard failed at {stage}: {available}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_root.parent).free < int(args.minimum_disk_gib * 2**30):
        raise QualificationError(f"disk guard failed at {stage}")
    try:
        gpu = legacy._resource_preflight(  # noqa: SLF001 - retained reviewed guard
            argparse.Namespace(
                minimum_free_gib=args.minimum_free_gib,
                maximum_reserved_gib=args.maximum_reserved_gib,
                maximum_rss_gib=args.maximum_rss_gib,
                max_seconds=args.max_seconds,
            ),
            device,
            stage=stage,
            started=started,
        )
    except Exception as exc:
        raise QualificationError(f"GPU guard failed at {stage}: {exc}") from exc
    return {"host_rss_bytes": rss, "host_available_bytes": available, "gpu": gpu}


def _capture_proposals(footing: Any, captured: list[dict[str, torch.Tensor]]) -> tuple[Any, Any]:
    original = footing.propose_public_a1

    def wrapped(*call_args: Any, **call_kwargs: Any) -> Any:
        proposal = original(*call_args, **call_kwargs)
        candidates = torch.as_tensor(proposal.candidates).detach().cpu().to(torch.long).contiguous()
        scores = torch.as_tensor(proposal.scores).detach().cpu().to(torch.float32).contiguous()
        if tuple(candidates.shape) != (1, STORED_SEQUENCE_TOKENS, PROPOSAL_BUDGET):
            raise QualificationError(f"proposal geometry changed: {tuple(candidates.shape)}")
        if tuple(scores.shape) != tuple(candidates.shape):
            raise QualificationError(f"proposal score geometry changed: {tuple(scores.shape)}")
        if candidates.lt(0).any().item() or candidates.ge(VOCABULARY_SIZE).any().item():
            raise QualificationError("proposal IDs are outside the vocabulary")
        captured.append({"ids": candidates, "scores": scores})
        return proposal

    footing.propose_public_a1 = wrapped
    return footing, original


def _restore_proposals(footing: Any, original: Any) -> None:
    footing.propose_public_a1 = original


def _run_mode(
    *,
    runner: Any,
    row: Mapping[str, Any],
    checked: Mapping[str, Any],
    root: Path,
    observations: Mapping[str, Mapping[str, Any]],
    cells: tuple[str, ...],
    records: int,
    device: torch.device,
    mode: str,
    footing: Any,
    started: float,
    args: argparse.Namespace,
    output_root: Path,
    legacy: Any,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    loaded = runner._load_native_a1_a2(  # noqa: SLF001 - existing frozen loader
        row,
        root=root,
        device=device,
        embedding=None,
        code_bindings=checked["code_bindings"],
    )
    adapter = loaded.adapter
    captured: list[dict[str, torch.Tensor]] = []
    original = None
    loader_label = "scripts.trr0010_eval_runner._load_native_a1_a2"
    if mode == "trace_on":
        footing, original = _capture_proposals(footing, captured)
    outputs: dict[str, torch.Tensor] = {}
    rows_receipt: list[dict[str, Any]] = []
    try:
        for cell_id in cells:
            _guard(legacy=legacy, args=args, device=device, started=started, output_root=output_root, stage=f"{mode}/{cell_id}/before_cell")
            begin_cell = getattr(adapter, "begin_cell", None)
            if callable(begin_cell):
                begin_cell()
            cell = observations[cell_id]
            values: list[torch.Tensor] = []
            full_records = int(cell.get("records", 0))
            if full_records < records:
                raise QualificationError(f"observation cell has only {full_records} rows: {cell_id}")
            for index, activation, mask, positions in runner._iter_rows(  # noqa: SLF001
                cell,
                records=full_records,
                hidden_size=2048,
            ):
                if index >= records:
                    break
                _guard(legacy=legacy, args=args, device=device, started=started, output_root=output_root, stage=f"{mode}/{cell_id}/record{index}/before")
                predicted = runner._normalize_prediction(  # noqa: SLF001
                    adapter(activation, mask, positions),
                    mask,
                    method_id=METHOD_ID,
                )
                values.append(predicted)
                rows_receipt.append({"mode": mode, "cell_id": cell_id, "row": int(index), "prediction_sha256": tensor_digest(predicted)})
                _guard(legacy=legacy, args=args, device=device, started=started, output_root=output_root, stage=f"{mode}/{cell_id}/record{index}/after")
            if len(values) != records:
                raise QualificationError(f"observation rows cover {len(values)}, expected {records}: {cell_id}")
            outputs[cell_id] = torch.stack(values).to(torch.long).contiguous()
    finally:
        if original is not None:
            _restore_proposals(footing, original)
        del adapter, loaded
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return outputs, captured, {"loader": loader_label}


def _trace_artifact(path: Path, *, ids: torch.Tensor, scores: torch.Tensor, records: int, cells: tuple[str, ...]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise QualificationError(f"trace output is not create-only: {path}")
    if ids.ndim != 3 or scores.shape != ids.shape:
        raise QualificationError("trace tensor geometry changed")
    metadata = {
        "schema": TRACE_SCHEMA,
        "task_id": "TRR-P10",
        "method_id": METHOD_ID,
        "cells": json.dumps(list(cells), sort_keys=True),
        "records_per_cell": str(records),
        "proposal_budget": str(PROPOSAL_BUDGET),
        "candidate_budget": str(CANDIDATE_BUDGET),
        "trace_role": "a1_proposal_ids_and_scores_at_decision_point",
        "truth_opened": "false",
        "target_labels_loaded": "false",
        "source_text_loaded": "false",
        "candidate_arrays_persisted": "true",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"proposal_ids": ids, "proposal_scores": scores}, str(path), metadata=metadata)
    return _record(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.expanduser().resolve()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise QualificationError("A1 trace qualification requires CUDA")
    output_root = args.output_root.expanduser().resolve()
    if not str(output_root).startswith("/tmp/"):
        raise QualificationError("qualification output must be under /tmp")
    if output_root.exists() or output_root.is_symlink():
        raise QualificationError(f"qualification output is not create-only: {output_root}")
    started = time.perf_counter()
    _gate, runner, legacy, footing = _load_old_modules(root)
    registration_path = args.registration.expanduser().resolve()
    freeze_path = args.freeze.expanduser().resolve()
    registration, checked, registration_record = runner._load_registration(  # noqa: SLF001
        registration_path,
        root=root,
        require_current_head=False,
        require_runtime_sources=True,
    )
    frozen = validate_frozen_package(freeze_path, repository_root=root, registration_path=registration_path)
    row = next((item for item in registration["methods"] if isinstance(item, Mapping) and item.get("id") == METHOD_ID), None)
    if not isinstance(row, Mapping):
        raise QualificationError("A1+A2 registration row is absent")
    cells = tuple(args.cells)
    if not cells or any(cell not in CELL_ORDER for cell in cells):
        raise QualificationError("qualification cells must be a nonempty subset of the four frozen cells")
    observations = checked["observation_bindings"]
    records = int(args.records)
    if records < 1 or records > 2:
        raise QualificationError("qualification records must be 1 or 2")
    device = torch.device("cuda")
    guard_start = _guard(legacy=legacy, args=args, device=device, started=started, output_root=output_root, stage="before_load")
    off, _, load_evidence = _run_mode(
        runner=runner, row=row, checked=checked, root=root, observations=observations,
        cells=cells, records=records, device=device, mode="trace_off", footing=footing,
        started=started, args=args, output_root=output_root, legacy=legacy,
    )
    on, captures, _ = _run_mode(
        runner=runner, row=row, checked=checked, root=root, observations=observations,
        cells=cells, records=records, device=device, mode="trace_on", footing=footing,
        started=started, args=args, output_root=output_root, legacy=legacy,
    )
    trace_count = records * len(cells)
    if len(captures) != trace_count:
        raise QualificationError(f"trace calls cover {len(captures)} rows, expected {trace_count}")
    trace_ids = torch.cat([item["ids"] for item in captures], dim=0).contiguous()
    trace_scores = torch.cat([item["scores"] for item in captures], dim=0).contiguous()
    row_results: list[dict[str, Any]] = []
    for cell in cells:
        expected = frozen.predictions[f"{METHOD_ID}::{cell}"][:records]
        if not torch.equal(off[cell], on[cell]):
            raise QualificationError(f"trace-on/off final IDs differ: {cell}")
        if not torch.equal(on[cell], expected):
            raise QualificationError(f"trace output differs from frozen A1+A2 prediction: {cell}")
        for index in range(records):
            row_results.append({
                "cell_id": cell,
                "row": index,
                "trace_off_prediction_sha256": tensor_digest(off[cell][index]),
                "trace_on_prediction_sha256": tensor_digest(on[cell][index]),
                "frozen_prediction_sha256": tensor_digest(expected[index]),
                "exact_trace_on_off": True,
                "exact_frozen_match": True,
            })
    trace_path = output_root / "a1_proposal_trace.safetensors"
    trace_record = _trace_artifact(trace_path, ids=trace_ids, scores=trace_scores, records=records, cells=cells)
    return {
        "schema": QUALIFICATION_SCHEMA,
        "task_id": "TRR-P10",
        "status": "PASS_TRUTH_FREE_A1_TRACE_OUTPUT_EQUIVALENCE",
        "method_id": METHOD_ID,
        "cells": list(cells),
        "records_per_cell": records,
        "registration": registration_record,
        "freeze": frozen.freeze_record,
        "trace": trace_record,
        "trace_tensor_shapes": {"proposal_ids": list(trace_ids.shape), "proposal_scores": list(trace_scores.shape)},
        "trace_tensor_digests": {"proposal_ids": tensor_digest(trace_ids), "proposal_scores": tensor_digest(trace_scores)},
        "trace_row_order": [f"{cell}::row{row}" for cell in cells for row in range(records)],
        "proposal_budget": PROPOSAL_BUDGET,
        "candidate_budget": CANDIDATE_BUDGET,
        "decoder_rank_trace": "UNAVAILABLE; no fixed-decoder top-256 tensor was retained",
        "rows": row_results,
        "source_code_bindings": {
            **checked["code_bindings"],
            "qualification_wrapper": _record(Path(__file__)),
        },
        "loader": load_evidence,
        "guard_before_load": guard_start,
        "guard_policy": {
            "minimum_free_gib": args.minimum_free_gib,
            "maximum_reserved_gib": args.maximum_reserved_gib,
            "maximum_rss_gib": args.maximum_rss_gib,
            "minimum_host_available_gib": args.minimum_host_available_gib,
            "minimum_disk_gib": args.minimum_disk_gib,
            "maximum_seconds": args.max_seconds,
        },
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", default=["pile__public_base", "finance__public_base"])
    parser.add_argument("--records", type=int, default=1)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--minimum-free-gib", type=float, default=8.0)
    parser.add_argument("--maximum-reserved-gib", type=float, default=6.0)
    parser.add_argument("--maximum-rss-gib", type=float, default=16.0)
    parser.add_argument("--minimum-host-available-gib", type=float, default=10.0)
    parser.add_argument("--minimum-disk-gib", type=float, default=20.0)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args)
        output = args.output_root.expanduser().resolve() / "qualification.json"
        if output.exists() or output.is_symlink():
            raise QualificationError(f"qualification receipt is not create-only: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "output": str(output), "trace": result["trace"]}, sort_keys=True))
        return 0
    except (QualificationError, ExecutionError) as exc:
        print(f"TRR-P10 qualification failed closed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"TRR-P10 qualification failed closed with an unexpected error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
