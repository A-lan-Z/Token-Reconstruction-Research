from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import torch

from token_reconstruction.trr0007_positionwise import build_residual_mlp512
from trr0010_p09_qualifier import (
    QualificationError,
    build_qualification_runtime,
    enforce_resource_guard,
    validate_exclusive_lease,
    validate_qualification_bindings,
    _validate_validation_contract,
    verify_zero_delta_equivalence,
)


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _settings() -> dict[str, object]:
    return {
        "hidden_size": 8,
        "vocabulary_size": 17,
        "sequence_tokens": 4,
        "batch_records": 2,
        "position_budget": 3,
        "base_learning_rate": 2.0e-4,
        "directional_learning_rate": 1.0e-4,
        "weight_decay": 0.0,
        "anchor_strength": 1.0e-4,
        "probe_steps": 2,
        "compute_base_logits": False,
    }


def _binding_manifest(tmp_path: Path) -> dict[str, object]:
    artifacts = {
        role: {**_write(tmp_path / "artifacts" / f"{role}.bin", role.encode()), "status": status}
        for role, status in {
            "contract": "FROZEN",
            "bank_manifest": "VERIFIED",
            "schedule": "FROZEN",
            "base_state": "SELECTED",
            "public_embedding": "PUBLIC_NORMALIZED",
            "support_ids": "VERIFIED",
            "support_counts": "VERIFIED",
        }.items()
    }
    sources = {}
    for index, relative in enumerate(
        (
            "src/token_reconstruction/trr_p09_fixed_control_adapter.py",
            "scripts/trr_p09/fixed_control_runner.py",
            "scripts/trr_p09/prepare_streamed_bank.py",
        )
    ):
        payload = f"source-{index}".encode()
        sources[relative] = {**_write(tmp_path / relative, payload), "commit": "a" * 40}
    return {
        "schema": "token-reconstruction.trr0010-qualifier-bindings.v1",
        "task_id": "TRR-0010",
        "finalized": True,
        "status": "READY_FOR_QUALIFICATION",
        "artifacts": artifacts,
        "a2_sources": sources,
        "settings": _settings(),
        "schedule": {"seed": 4005, "steps": 2, "semantic_sha256": "b" * 64, "exposure": {"draws": 6}},
        "validation_geometry": {
            "sequence_tokens": 4,
            "batch_records": 2,
            "activation_dtype": "torch.float32",
        },
    }


def test_binding_manifest_requires_final_schedule_and_validation_geometry(tmp_path: Path) -> None:
    receipt = validate_qualification_bindings(_binding_manifest(tmp_path))
    assert receipt["settings"]["probe_steps"] == 2
    assert receipt["schedule"]["semantic_sha256"] == "b" * 64
    assert receipt["validation_geometry"]["batch_records"] == 2


def test_lease_requires_exclusive_explicit_caps() -> None:
    lease = {
        "schema": "token-reconstruction.trr0010-qualifier-lease.v1",
        "task_id": "TRR-0010",
        "status": "GRANTED",
        "exclusive": True,
        "device": "cuda:0",
        "owner": "test",
        "max_seconds": 600,
        "gpu_reserved_limit_bytes": 6 * 2**30,
        "gpu_free_floor_bytes": 8 * 2**30,
        "host_rss_limit_bytes": 16 * 2**30,
        "host_available_floor_bytes": 10 * 2**30,
        "disk_free_floor_bytes": 20 * 2**30,
    }
    caps = validate_exclusive_lease(lease)
    assert caps["device"] == "cuda:0"
    lease["exclusive"] = False
    with pytest.raises(QualificationError, match="exclusive"):
        validate_exclusive_lease(lease)


def test_zero_delta_probe_is_exact_on_same_public_rows() -> None:
    torch.manual_seed(10010)
    base = build_residual_mlp512(hidden_size=8, vocabulary_size=17, context_width=4, bottleneck_size=3, seed=4005)
    embedding = torch.nn.functional.normalize(torch.randn(17, 8), dim=-1)
    runtime = build_qualification_runtime(
        base_decoder=base,
        support_ids=torch.tensor([1, 4, 8, 13]),
        support_counts=torch.tensor([1, 4, 25, 100]),
        public_embedding=embedding,
        settings=_settings(),
    )

    def shared_decoder_rows(decoder, hook, activation, valid_mask, records, positions, table, targets, *, compute_base_logits):
        del targets, compute_base_logits
        projected = decoder.projected_hidden(activation, valid_mask)
        rows = projected[records, positions]
        return None, hook.score_rows(rows, decoder.logit_scale, table), hook.loss_terms(
            hook.score_rows(rows, decoder.logit_scale, table), torch.tensor([1, 4, 8])
        )

    batch = SimpleNamespace(
        activations=torch.randn(2, 4, 8),
        attention_mask=torch.ones(2, 4, dtype=torch.bool),
        token_ids=torch.tensor([[0, 1, 4, 8], [0, 4, 8, 13]]),
    )
    step = SimpleNamespace(
        step=0,
        batch_global_rows=(0, 1),
        draw_record_slots=(0, 0, 1),
        draw_position_slots=(1, 2, 1),
    )
    result = verify_zero_delta_equivalence(
        runtime=runtime,
        runner=SimpleNamespace(shared_decoder_rows=shared_decoder_rows),
        batch=batch,
        schedule_step=step,
        embedding=embedding,
        config=SimpleNamespace(hidden_size=8),
    )
    assert result["exact_logits"] is True
    assert result["exact_argmax"] is True


def test_resource_guard_fails_closed_when_gpu_cap_is_missing() -> None:
    snapshot = {
        "host_available_bytes": 20 * 2**30,
        "host_rss_bytes": 1 * 2**30,
        "disk_free_bytes": 100 * 2**30,
        "gpu": {"available": False},
    }
    caps = {
        "max_seconds": 600,
        "gpu_reserved_limit_bytes": 6 * 2**30,
        "gpu_free_floor_bytes": 8 * 2**30,
        "host_rss_limit_bytes": 16 * 2**30,
        "host_available_floor_bytes": 10 * 2**30,
        "disk_free_floor_bytes": 20 * 2**30,
    }
    with pytest.raises(QualificationError, match="GPU telemetry"):
        enforce_resource_guard(snapshot, caps, started=time.perf_counter())


def test_domain_balanced_selection_requires_explicit_a2_callback() -> None:
    config = SimpleNamespace(selection_metric="domain_balanced_token_accuracy")
    with pytest.raises(QualificationError, match="explicit validation_callback"):
        _validate_validation_contract(config, None)
    callback = lambda step, evaluate_view: {"step": step, "domain_balanced_token_accuracy": 1.0}
    assert _validate_validation_contract(config, callback) == "domain_balanced_token_accuracy"


def test_pooled_selection_rejects_domain_callback() -> None:
    config = SimpleNamespace(selection_metric="token_accuracy")
    with pytest.raises(QualificationError, match="only valid"):
        _validate_validation_contract(config, lambda step, evaluate_view: {})
