"""CPU-only validation for retained directional recovery inputs.

This check deliberately loads one retained directional checkpoint on CPU and
exercises the production provider's metadata-only configuration path.  It
does not load the public embedding, bank, activations, labels, or truth, and
it does not claim a CUDA lease.  The checkpoint load is strict over the full
base decoder and supported-row delta state so a later recovery launch cannot
fail because the hook owns only one half of the state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


TASK_ID = "TRR-0010"
LEASE_SCHEMA = "token-reconstruction.trr0010-qualifier-lease.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"output is create-only: {path}")
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def _lease_fixture(max_seconds: int) -> dict[str, Any]:
    # This is an in-memory validation fixture only.  It is never passed to a
    # CUDA runner and therefore cannot claim the exclusive lease itself.
    return {
        "schema": LEASE_SCHEMA,
        "task_id": TASK_ID,
        "status": "GRANTED",
        "exclusive": True,
        "device": "cuda:0",
        "owner": "TRR-0010-recovery-CPU-metadata-validation-only",
        "max_seconds": int(max_seconds),
        "gpu_reserved_limit_bytes": 10 * 1024**3,
        "gpu_free_floor_bytes": 2 * 1024**3,
        "host_rss_limit_bytes": 12 * 1024**3,
        "host_available_floor_bytes": 8 * 1024**3,
        "disk_free_floor_bytes": 20 * 1024**3,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    from trr0010_model import load_directional_state
    from trr0010_p09_qualifier import validate_exclusive_lease
    from trr0010_directional_fit_cli import _load_diagnostic_binding
    from trr0010_production_provider import configuration_dry_run

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    raw_runner = Path(args.raw_runner).expanduser().resolve()
    binding_path = Path(args.binding).expanduser().resolve()
    diagnostic_path = Path(args.diagnostic_binding).expanduser().resolve()
    for path, label in (
        (checkpoint, "checkpoint"),
        (raw_runner, "raw runner receipt"),
        (binding_path, "production binding"),
        (diagnostic_path, "diagnostic binding"),
    ):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"{label} is not a regular file: {path}")

    raw = _load_object(raw_runner, label="raw runner receipt")
    binding = _load_object(binding_path, label="production binding")
    if raw.get("status") != "COMPLETED":
        raise RuntimeError("raw runner receipt is not COMPLETED")
    if int(raw.get("selected_step", -1)) != 0:
        raise RuntimeError("the retained current directional selection is not step 0")

    lease_fixture = _lease_fixture(int(args.max_seconds))
    lease_caps = validate_exclusive_lease(lease_fixture)
    if int(lease_caps["max_seconds"]) != int(args.max_seconds):
        raise RuntimeError("recovery lease max_seconds was not accepted exactly")

    # The model loader reads only the retained checkpoint.  No public E/H or
    # validation payload is touched by this strict state-restoration check.
    torch.set_num_threads(1)
    model = load_directional_state(
        checkpoint,
        hidden_size=2048,
        vocabulary_size=128256,
        context_width=128,
        bottleneck_size=512,
    )
    model.eval()
    raw_state = load_file(str(checkpoint), device="cpu")
    model.load_state_dict(raw_state, strict=True)
    if model.training or getattr(model, "base", model).training:
        raise RuntimeError("model.eval() did not remain set after strict state load")
    model_state = model.state_dict()
    if set(model_state) != set(raw_state):
        raise RuntimeError("strict state load changed the directional state key set")
    mismatches = [key for key in sorted(raw_state) if not torch.equal(raw_state[key], model_state[key])]
    if mismatches:
        raise RuntimeError(f"strict state restoration differs for {mismatches[:5]}")
    base_keys = sorted(key for key in raw_state if key.startswith("base."))
    delta_keys = [key for key in ("delta_rows", "support_ids", "support_counts", "anchor_weights") if key in raw_state]
    if not base_keys or "delta_rows" not in delta_keys:
        raise RuntimeError("checkpoint did not restore both base decoder and delta state")

    diagnostic_binding = _load_diagnostic_binding(diagnostic_path)
    # This path hashes/validates the production metadata, schedules, source
    # bindings, and qualification evidence, but explicitly allocates no
    # model/embedding/bank tensors.  It receives the same 900-second recovery
    # cap that the eventual lease will use; the signed production cap remains
    # untouched in the production binding.
    config = configuration_dry_run(
        binding_receipts={"current_directional": binding},
        diagnostic_binding=diagnostic_binding,
        lease_caps=lease_caps,
        device=torch.device("cpu"),
        output_root=None,
        arm_name="current_directional",
    )
    if config.get("status") != "PASS_CONFIGURATION_DRY_RUN":
        raise RuntimeError("provider configuration dry run did not pass")
    arm = config.get("arms", {}).get("current_directional")
    if not isinstance(arm, dict) or arm.get("model_allocated") is not False or arm.get("updates") is not False:
        raise RuntimeError("provider dry run unexpectedly allocated or updated a model")

    return {
        "schema": "token-reconstruction.trr0010-recovery-cpu-validation.v1",
        "task_id": TASK_ID,
        "status": "PASS_CPU_STATE_AND_CONFIGURATION",
        "command": list(sys.argv),
        "truth_opened": False,
        "gpu_launched": False,
        "updates": False,
        "source_runner_receipt": {
            "path": str(raw_runner),
            "bytes": int(raw_runner.stat().st_size),
            "sha256": _sha256_file(raw_runner),
            "status": raw.get("status"),
            "selected_step": int(raw.get("selected_step")),
            "selected_state_sha256": raw.get("selected_state_sha256"),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": int(checkpoint.stat().st_size),
            "sha256": _sha256_file(checkpoint),
            "selected_step": 0,
            "strict_state_load": True,
            "base_decoder_tensor_count": len(base_keys),
            "delta_state_keys": delta_keys,
            "all_state_tensors_exact_after_load": True,
            "eval_after_load": True,
        },
        "recovery_lease_validation": {
            "accepted_max_seconds": int(lease_caps["max_seconds"]),
            "gpu_reserved_limit_bytes": int(lease_caps["gpu_reserved_limit_bytes"]),
            "gpu_free_floor_bytes": int(lease_caps["gpu_free_floor_bytes"]),
            "host_rss_limit_bytes": int(lease_caps["host_rss_limit_bytes"]),
            "host_available_floor_bytes": int(lease_caps["host_available_floor_bytes"]),
            "disk_free_floor_bytes": int(lease_caps["disk_free_floor_bytes"]),
            "production_signed_wall_limit_untouched": binding.get("resource_qualification", {}).get("wall_limit_seconds"),
            "lease_claimed": False,
        },
        "provider_configuration_dry_run": {
            "status": config.get("status"),
            "arm": arm,
            "model_allocated": config.get("model_allocated"),
            "updates": config.get("updates"),
            "truth_opened": config.get("truth_opened"),
        },
        "bindings": {
            "production_binding": {"path": str(binding_path), "bytes": int(binding_path.stat().st_size), "sha256": _sha256_file(binding_path)},
            "diagnostic_binding": {"path": str(diagnostic_path), "bytes": int(diagnostic_path.stat().st_size), "sha256": _sha256_file(diagnostic_path)},
            "model_loader_source": {"path": str(Path("scripts/trr0010_model.py").resolve()), "sha256": _sha256_file(Path("scripts/trr0010_model.py").resolve())},
            "provider_source": {"path": str(Path("scripts/trr0010_production_provider.py").resolve()), "sha256": _sha256_file(Path("scripts/trr0010_production_provider.py").resolve())},
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-runner", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--diagnostic-binding", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=int, default=900)
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    try:
        result = run(args)
    except Exception as exc:
        failure = {
            "schema": "token-reconstruction.trr0010-recovery-cpu-validation.v1",
            "task_id": TASK_ID,
            "status": "FAIL_CPU_VALIDATION",
            "command": list(sys.argv),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "truth_opened": False,
            "gpu_launched": False,
            "updates": False,
        }
        _write_create_only(output, failure)
        raise
    _write_create_only(output, result)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
