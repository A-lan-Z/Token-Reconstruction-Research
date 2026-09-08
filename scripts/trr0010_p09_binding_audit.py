"""CPU-only validator for the finalized TRR-0010 qualifier binding."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

from scripts.trr0010_p09_qualifier import validate_qualification_bindings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binding_path = args.binding.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    start = datetime.now(timezone.utc)
    started = time.monotonic()
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    validated = validate_qualification_bindings(binding)
    end = datetime.now(timezone.utc)
    receipt = {
        "schema": "token-reconstruction.trr0010-qualification-binding-cpu-audit.v1",
        "task_id": "TRR-0010",
        "status": "PASS_QUALIFICATION_BINDING_CPU",
        "truth_boundary": {
            "gpu_started": False,
            "fit_started": False,
            "capture_started": False,
            "truth_opened": False,
            "public_payload_loaded": False,
        },
        "command": "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=.:src:scripts python3 scripts/trr0010_p09_binding_audit.py --binding experiments/TRR-0010/setup/p09_largest_cell_qualification_bindings_v2.json --output experiments/TRR-0010/setup/p09_qualification_binding_cpu_v1.json",
        "execution": {
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "device": "cpu",
            "threads": {"omp": "1", "mkl": "1", "openblas": "1"},
        },
        "binding": {
            "path": str(binding_path),
            "bytes": binding_path.stat().st_size,
            "sha256": sha256_file(binding_path),
            "schema": binding.get("schema"),
            "status": binding.get("status"),
            "finalized": binding.get("finalized"),
        },
        "validated": {
            "artifact_roles": sorted(validated["artifacts"]),
            "a2_source_roles": sorted(validated["a2_sources"]),
            "schedule_steps": int(validated["schedule"]["steps"]),
            "probe_steps": int(validated["settings"]["probe_steps"]),
            "validation_geometry": validated["validation_geometry"],
            "b0_binding": validated["b0_binding"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "output": str(output_path), "receipt_sha256": sha256_file(output_path), "schedule_steps": receipt["validated"]["schedule_steps"], "probe_steps": receipt["validated"]["probe_steps"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
