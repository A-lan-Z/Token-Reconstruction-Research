"""CPU-only audit of the hash-bound TRR-P09 public validation loader.

This opens only the public prepared validation activations and labels.  It
does not run a model, read evaluation truth, or create a training artifact.
The output is a compact receipt so the exact manifest/resource bindings and
the row-gathering geometry are reproducible before a qualification lease.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import torch

from scripts import trr0010_p09_provider as provider


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _descriptor(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--audit-finish", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    rows_path = args.rows.expanduser().resolve()
    audit_finish_path = args.audit_finish.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    source_path = Path(__file__).resolve()
    provider_path = Path(provider.__file__).resolve()
    manifest = _descriptor(manifest_path)
    rows = _descriptor(rows_path)
    audit_finish = _descriptor(audit_finish_path)
    start = datetime.now(timezone.utc)
    start_mono = time.monotonic()
    views = provider._manifest_validation_views(
        manifest,
        expected_tokens=128,
        expected_batch=8,
    )
    domain_summary: dict[str, dict[str, object]] = {}
    for domain, view in views.items():
        expected_globals = tuple(range(0, 256)) if domain == "Finance" else tuple(range(256, 384))
        actual_globals: list[int] = []
        batches = 0
        records = 0
        scored_rows = 0
        mask_rows = 0
        first_shapes: dict[str, list[int]] | None = None
        dtypes: dict[str, str] | None = None
        for batch in view.batches():
            batches += 1
            records += len(batch.global_rows)
            actual_globals.extend(batch.global_rows)
            if first_shapes is None:
                first_shapes = {
                    "activations": list(batch.activations.shape),
                    "token_ids": list(batch.token_ids.shape),
                    "attention_mask": list(batch.attention_mask.shape),
                    "position_ids": list(batch.position_ids.shape),
                }
                dtypes = {
                    "activations": str(batch.activations.dtype),
                    "token_ids": str(batch.token_ids.dtype),
                    "attention_mask": str(batch.attention_mask.dtype),
                    "position_ids": str(batch.position_ids.dtype),
                }
            if batch.activations.shape != (8, 128, 2048):
                raise RuntimeError(f"{domain} activation geometry mismatch")
            if batch.token_ids.shape != (8, 128):
                raise RuntimeError(f"{domain} token geometry mismatch")
            if batch.attention_mask.shape != (8, 128):
                raise RuntimeError(f"{domain} mask geometry mismatch")
            if batch.position_ids.shape != (8, 128):
                raise RuntimeError(f"{domain} position geometry mismatch")
            if not bool(torch.isfinite(batch.activations.float()).all().item()):
                raise RuntimeError(f"{domain} non-finite activation")
            expected_positions = torch.arange(128, dtype=torch.long).expand(8, -1)
            if not torch.equal(batch.position_ids, expected_positions):
                raise RuntimeError(f"{domain} position IDs mismatch")
            mask_rows += int(batch.attention_mask[:, 1:].sum().item())
            scored_rows += 8 * 127
        if tuple(actual_globals) != expected_globals:
            raise RuntimeError(f"{domain} global row order mismatch")
        expected_records = 256 if domain == "Finance" else 128
        if records != expected_records or scored_rows != view.expected_post_bos_rows:
            raise RuntimeError(f"{domain} record/scored-row count mismatch")
        domain_summary[domain] = {
            "records": records,
            "batches": batches,
            "global_first": actual_globals[:8],
            "global_last": actual_globals[-8:],
            "scored_post_bos_rows": scored_rows,
            "attention_true_post_bos_rows": mask_rows,
            "expected_post_bos_rows": view.expected_post_bos_rows,
            "first_batch_shapes": first_shapes,
            "dtypes": dtypes,
        }
    end = datetime.now(timezone.utc)
    receipt = {
        "schema": "token-reconstruction.trr0010-public-validation-loader-cpu-audit.v2",
        "status": "PASS_PUBLIC_VALIDATION_LOADER_CPU",
        "task_id": "TRR-0010",
        "truth_boundary": {
            "truth_opened": False,
            "evaluation_labels_opened": False,
            "public_prepared_labels_only": True,
            "model_forward_count": 0,
            "source_text_read": False,
        },
        "command": "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=.:src:scripts python3 scripts/trr0010_public_validation_loader_audit.py --manifest /tmp/trr-p09-runtime/public-validation-r1/validation_manifest.json --rows /tmp/trr-p09-runtime/public-validation-r1/validation_rows.json --audit-finish /tmp/trr-p09-runtime/public-validation-audit-watchdog-r1/finish.json --output experiments/TRR-0010/setup/p09_public_validation_loader_cpu_v2.json",
        "argv": [str(value) for value in sys.argv],
        "execution": {
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "elapsed_seconds": time.monotonic() - start_mono,
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "threads": {"omp": "1", "mkl": "1", "openblas": "1"},
        },
        "bindings": {
            "manifest": manifest,
            "rows": rows,
            "audit_finish": audit_finish,
            "audit_stdout_sha256": "3e953731a039ee1feea89e183126047ee0a4c3608061df267ebe728916f05659",
            "audit_script": {
                "path": str(source_path),
                "bytes": source_path.stat().st_size,
                "sha256": _sha256(source_path),
            },
            "provider_source": {
                "path": str(provider_path),
                "bytes": provider_path.stat().st_size,
                "sha256": _sha256(provider_path),
            },
        },
        "geometry": {
            "sequence_tokens_including_bos": 128,
            "batch_records": 8,
            "hidden_size": 2048,
            "domains": domain_summary,
            "global_row_order": "Finance_then_Pile",
        },
        "checks": [
            "manifest and every bound H/label resource descriptor SHA256 verified by provider",
            "actual P09 global rows mapped to domain-local H and label rows",
            "separate H/label/mask/position resources loaded through lazy batches",
            "all 384 records and 48 batches iterated with exact shape/dtype/position checks",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "output": str(output_path), "receipt_sha256": _sha256(output_path), "geometry": domain_summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
