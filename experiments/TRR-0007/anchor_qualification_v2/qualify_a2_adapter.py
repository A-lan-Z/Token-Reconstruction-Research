#!/usr/bin/env python3
"""Bounded one-row TRR-0007 A1+A2 adapter qualification (v2).

This harness deliberately calls the new runner's private loader and adapter
with a minimal, locally validated descriptor.  It uses the published TRR-0004
Finance public-base H row and prediction fixture, and never opens source text,
target labels, or a truth sidecar.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from typing import Any

import torch
from safetensors import safe_open

from scripts import trr0007_eval_contract as contract
from scripts import trr0007_eval_runner as runner
from safetensors.torch import load_file


TASK_ID = "TRR-0007"
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CPU_RECEIPT = Path("/tmp/trr0007_a1_cpu_fixture_equivalence_v1.json")

H_PATH = ROOT.parent / "TRR-0005/experiments/TRR-0004/fresh_confirmation_v1/panel_capture/observations/finance/public_base.safetensors"
A1_PATH = ROOT.parent / "TRR-0005/experiments/TRR-0004/fresh_confirmation_v1/predictions_v2/finance/public_base/historical_alpaca_a1.safetensors"
A2_PATH = ROOT.parent / "TRR-0005/experiments/TRR-0004/fresh_confirmation_v1/predictions_v2/finance/public_base/frozen_a1_a2_k256.safetensors"
E_PATH = ROOT.parent.parent / "outputs/TRR-0003/track_b/public_fit_v2/public_normalized_embeddings.safetensors"
LENS_PATH = ROOT / "experiments/TRR-0004/evidence/comparators/public_a1_lens.pt"
REFERENCE_PATH = ROOT / "experiments/TRR-0004/evidence/comparators/round001_teacher.py"
SNAPSHOT = Path("/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6")

MIN_FREE_BYTES = 8 * 1024**3
MAX_RESERVED_BYTES = 6 * 1024**3
MIN_HOST_AVAILABLE_BYTES = 10 * 1024**3
MAX_PEAK_ALLOCATED_BYTES = int(3.55 * 1024**3)


class QualifierError(RuntimeError):
    """Raised when the bounded qualifier fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise QualifierError(f"fixture/source unavailable: {path}")
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def host_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            return int(fields[1]) * 1024
    raise QualifierError("host available-memory guard unavailable")


def compute_apps() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise QualifierError("nvidia-smi is unavailable for GPU exclusivity guard") from exc
    if result.returncode != 0:
        raise QualifierError(f"nvidia-smi guard failed: {result.stderr.strip()}")
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 3 and fields[0] and fields[0] != str(os.getpid()):
            rows.append({"pid": fields[0], "process_name": fields[1], "used_memory": fields[2]})
    return rows


def resource_state(stage: str, *, started: float, enforce: bool = True) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise QualifierError(f"CUDA unavailable at {stage}")
    free, total = torch.cuda.mem_get_info()
    reserved = int(torch.cuda.memory_reserved())
    state = {
        "stage": stage,
        "elapsed_seconds": float(time.perf_counter() - started),
        "cuda_free_bytes": int(free),
        "cuda_total_bytes": int(total),
        "cuda_reserved_bytes": reserved,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "host_available_bytes": host_available_bytes(),
        "compute_apps": compute_apps(),
    }
    if enforce:
        violations: list[str] = []
        if state["compute_apps"]:
            violations.append(f"non-exclusive compute apps: {state['compute_apps']!r}")
        if state["cuda_free_bytes"] < MIN_FREE_BYTES:
            violations.append(f"free CUDA below 8 GiB: {state['cuda_free_bytes']}")
        if state["cuda_reserved_bytes"] > MAX_RESERVED_BYTES:
            violations.append(f"reserved CUDA above 6 GiB: {state['cuda_reserved_bytes']}")
        if state["host_available_bytes"] < MIN_HOST_AVAILABLE_BYTES:
            violations.append(f"host available below 10 GiB: {state['host_available_bytes']}")
        if state["cuda_peak_allocated_bytes"] > MAX_PEAK_ALLOCATED_BYTES:
            violations.append(
                f"peak allocated above 3.55 GiB: {state['cuda_peak_allocated_bytes']}"
            )
        if violations:
            raise QualifierError(f"resource guard failed at {stage}: {'; '.join(violations)}")
    return state


def load_fixture_row() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    for path in (H_PATH, A1_PATH, A2_PATH, E_PATH, LENS_PATH, REFERENCE_PATH):
        if not path.is_file() or path.is_symlink():
            raise QualifierError(f"fixture unavailable or symlink: {path}")
    with safe_open(str(H_PATH), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"activations"}:
            raise QualifierError(f"H fixture keys changed: {sorted(handle.keys())}")
        activations = handle.get_tensor("activations")
        if tuple(activations.shape) != (16, 128, 2048) or activations.dtype != torch.bfloat16:
            raise QualifierError("H fixture geometry or dtype changed")
        activation = activations[0].contiguous()
    archived: list[torch.Tensor] = []
    for path, method_id in ((A1_PATH, "historical_alpaca_a1"), (A2_PATH, "frozen_a1_a2_k256")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if "predictions" not in keys:
                raise QualifierError(f"prediction fixture lacks public predictions key: {path}")
            metadata = dict(handle.metadata() or {})
            if metadata.get("task_id") != "TRR-0004" or metadata.get("method_id") != method_id:
                raise QualifierError(f"prediction fixture binding changed: {path}")
            value = handle.get_tensor("predictions")
            if tuple(value.shape) != (16, 128):
                raise QualifierError(f"prediction fixture geometry changed: {path}")
            row = value[0].to(torch.long).contiguous()
            if row[0].item() != contract.BOS_TOKEN_ID or row.lt(0).any().item() or row.ge(contract.VOCAB_SIZE).any().item():
                raise QualifierError(f"prediction fixture IDs invalid: {path}")
            archived.append(row)
    mask = torch.ones(128, dtype=torch.bool)
    positions = torch.arange(128, dtype=torch.long)
    return activation, mask, positions, archived[0], archived[1], {
        "H": file_record(H_PATH),
        "A1": file_record(A1_PATH),
        "A2": file_record(A2_PATH),
        "E": file_record(E_PATH),
        "lens": file_record(LENS_PATH),
        "reference": file_record(REFERENCE_PATH),
    }


def tensor_digest(value: torch.Tensor) -> str:
    return contract.tensor_digest(value.detach().cpu().to(torch.long).contiguous())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / "qualification_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise QualifierError(f"create-only receipt already exists: {receipt_path}")
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    resource_history: list[dict[str, Any]] = []
    adapter = None
    embedding = None
    cpu_copy: dict[str, Any] | None = None
    comparison: dict[str, Any] = {}
    fixtures: dict[str, Any] = {}
    source_bindings: dict[str, Any] = {}
    snapshot_binding: dict[str, Any] = {}
    embedding_evidence: dict[str, Any] | None = None
    adapter_evidence: dict[str, Any] | None = None
    configured_numerics: dict[str, Any] | None = None
    try:
        if CPU_RECEIPT.is_file() and not CPU_RECEIPT.is_symlink():
            copied = OUT / "cpu_a1_fixture_receipt.json"
            shutil.copyfile(CPU_RECEIPT, copied)
            cpu_copy = file_record(copied)
            cpu_copy["source"] = str(CPU_RECEIPT)
        else:
            raise QualifierError(f"existing CPU receipt unavailable: {CPU_RECEIPT}")
        activation, mask, positions, expected_a1, expected_a2, fixtures = load_fixture_row()
        code_paths = [
            Path(__file__),
            ROOT / "scripts/trr0007_eval_runner.py",
            ROOT / "scripts/trr0007_eval_contract.py",
            ROOT / "scripts/trr0003_footing_compare.py",
            ROOT / "scripts/trr0004_predict_confirmation.py",
            ROOT / "src/token_reconstruction/component_crossover.py",
            ROOT / "src/token_reconstruction/a1a2_configuration_search.py",
            REFERENCE_PATH,
        ]
        source_bindings = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): file_record(path) for path in code_paths}
        snapshot_binding = {
            "path": str(SNAPSHOT),
            "revision": "9213176726f574b556790deb65791e0c5aa438b6",
            "config": {"path": str(SNAPSHOT / "config.json"), "bytes": 877, "sha256": "2febf68cea25bf4611be02b7536f2488a5ba523bb1134986e3610152abe74fdb"},
            "weights": {"path": str(SNAPSHOT / "model.safetensors"), "bytes": 2471645608, "sha256": "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"},
        }
        if not (SNAPSHOT / "config.json").is_file() or not (SNAPSHOT / "model.safetensors").is_file():
            raise QualifierError(f"public model snapshot is incomplete: {SNAPSHOT}")
        registration = {
            "runtime_assets": {
                "normalized_public_E": fixtures["E"] | {"shape": [contract.VOCAB_SIZE, contract.HIDDEN_SIZE], "dtype": "torch.float32"},
                "a1_a2": {
                    "public_model_snapshot": {"path": str(SNAPSHOT)},
                    "lens": fixtures["lens"],
                    "reference": fixtures["reference"],
                },
            }
        }
        configured_numerics = runner._configure_numerics(contract.NUMERICAL_SETTINGS)
        device = torch.device("cuda")
        resource_history.append(resource_state("preflight_before_embedding", started=started))
        torch.cuda.reset_peak_memory_stats(device)
        embedding, embedding_evidence = runner._load_embedding(registration, root=ROOT, device=device)
        resource_history.append(resource_state("after_embedding_load", started=started))
        adapter, adapter_evidence = runner._load_a2_adapter(registration, root=ROOT, embedding=embedding, device=device)
        resource_history.append(resource_state("after_a2_adapter_load", started=started))
        adapter.begin_cell()
        t0 = time.perf_counter()
        warm_a2 = adapter.predict(activation, mask, positions)
        torch.cuda.synchronize(device)
        warm_seconds = time.perf_counter() - t0
        warm_a1 = adapter.last_a1_prediction.detach().cpu().clone() if adapter.last_a1_prediction is not None else None
        resource_history.append(resource_state("after_warmup", started=started))
        t1 = time.perf_counter()
        measured_a2 = adapter.predict(activation, mask, positions)
        torch.cuda.synchronize(device)
        measured_seconds = time.perf_counter() - t1
        measured_a1 = adapter.last_a1_prediction.detach().cpu().clone() if adapter.last_a1_prediction is not None else None
        resource_history.append(resource_state("after_measured", started=started))
        if warm_a1 is None or measured_a1 is None:
            raise QualifierError("A1 diagnostic missing from A2 adapter")
        warm_a2_cpu = warm_a2.detach().cpu().to(torch.long).contiguous()
        measured_a2_cpu = measured_a2.detach().cpu().to(torch.long).contiguous()
        comparison = {
            "warmup_measured_a1_exact": bool(torch.equal(warm_a1, measured_a1)),
            "warmup_measured_a2_exact": bool(torch.equal(warm_a2_cpu, measured_a2_cpu)),
            "measured_a1_archived_exact": bool(torch.equal(measured_a1, expected_a1)),
            "measured_a2_archived_exact": bool(torch.equal(measured_a2_cpu, expected_a2)),
            "warmup_a1_digest": tensor_digest(warm_a1),
            "measured_a1_digest": tensor_digest(measured_a1),
            "archived_a1_digest": tensor_digest(expected_a1),
            "warmup_a2_digest": tensor_digest(warm_a2_cpu),
            "measured_a2_digest": tensor_digest(measured_a2_cpu),
            "archived_a2_digest": tensor_digest(expected_a2),
            "warmup_seconds": warm_seconds,
            "measured_seconds": measured_seconds,
            "a1_mismatch_positions": ((measured_a1 != expected_a1).nonzero(as_tuple=False).flatten().tolist()),
            "a2_mismatch_positions": ((measured_a2_cpu != expected_a2).nonzero(as_tuple=False).flatten().tolist()),
            "bos_id": int(contract.BOS_TOKEN_ID),
            "active_positions": int(mask.sum().item()),
        }
        if not all(comparison[key] for key in ("warmup_measured_a1_exact", "warmup_measured_a2_exact", "measured_a1_archived_exact", "measured_a2_archived_exact")):
            raise QualifierError(f"adapter fixture comparison failed: {comparison}")
        resource_history.append(resource_state("complete", started=started))
        status = "PASS_A2_ADAPTER_WARMUP_MEASURED_EXACT_ON_PUBLISHED_FIXTURE"
    except Exception as exc:
        status = "FAILED_PRESERVED_NO_TRUTH"
        error = {"type": type(exc).__name__, "message": str(exc)}
    else:
        error = None
    finally:
        if adapter is not None:
            del adapter
        if embedding is not None:
            del embedding
        gc = __import__("gc")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    try:
        peak_alloc = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        peak_reserved = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
    except RuntimeError:
        peak_alloc = None
        peak_reserved = None
    receipt = {
        "schema": "token-reconstruction.trr0007-a2-adapter-qualification.v2",
        "task_id": TASK_ID,
        "status": status,
        "started_utc": started_utc,
        "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": float(time.perf_counter() - started),
        "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "git_status_relevant": subprocess.run(["git", "-C", str(ROOT), "status", "--short", "--", "scripts/trr0007_eval_runner.py", "scripts/trr0007_eval_contract.py", str(Path(__file__).relative_to(ROOT))], check=False, capture_output=True, text=True).stdout.splitlines(),
        "command": {"argv": list(sys.argv), "cwd": str(Path.cwd()), "python": sys.executable, "environment": {key: os.environ.get(key) for key in ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
        "device": "cuda",
        "numerical_settings": dict(contract.NUMERICAL_SETTINGS),
        "configured_numerics": configured_numerics,
        "resource_limits": {"minimum_free_bytes": MIN_FREE_BYTES, "maximum_reserved_bytes": MAX_RESERVED_BYTES, "minimum_host_available_bytes": MIN_HOST_AVAILABLE_BYTES, "maximum_peak_allocated_bytes": MAX_PEAK_ALLOCATED_BYTES},
        "resource_history": resource_history,
        "peak_memory": {"cuda_peak_allocated_bytes": peak_alloc, "cuda_peak_reserved_bytes": peak_reserved, "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024},
        "fixtures": fixtures,
        "model_snapshot": snapshot_binding,
        "source_bindings": source_bindings,
        "cpu_a1_receipt_copy": cpu_copy,
        "adapter_loader_evidence": locals().get("adapter_evidence"),
        "embedding_loader_evidence": locals().get("embedding_evidence"),
        "comparison": comparison,
        "error": error,
        "truth_opened": False,
        "target_labels_loaded": False,
        "source_text_loaded": False,
        "private_truth_accessed": False,
        "candidate_arrays_persisted": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if status != "PASS_A2_ADAPTER_WARMUP_MEASURED_EXACT_ON_PUBLISHED_FIXTURE":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
