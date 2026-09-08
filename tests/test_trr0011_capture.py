from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from trr0011_capture import (
    CaptureError,
    EXPECTED_VARIANTS,
    HIDDEN_SIZE,
    RETAINED_SEQUENCE_TOKENS,
    apply_fixed_sign_block_update,
    default_capture_plan,
    freeze_capture_plan,
    opaque_record_ids_digest,
    save_observation_artifact,
    validate_after_cut_null,
    validate_capture_plan,
    validate_observation_artifact,
    validate_opaque_panel_descriptor,
)


def _variant(plan: dict, variant_id: str) -> dict:
    return next(row for row in plan["variants"] if row["variant_id"] == variant_id)


def _panel() -> dict:
    records = [
        {"domain": domain, "order": order, "opaque_id": f"trr11-{domain}-{order:02d}"}
        for domain in ("finance", "pile")
        for order in range(32)
    ]
    return {
        "schema": "token-reconstruction.trr0011-opaque-capture-panel.v1",
        "task_id": "TRR-0011",
        "status": "RESERVED_OPAQUE_PANEL_BEFORE_CAPTURE",
        "records_per_domain": 32,
        "domains": ["finance", "pile"],
        "same_record_order_across_variants": True,
        "selection_performed": False,
        "opaque_reservation": {"receipt_sha256": "a" * 64},
        "records": records,
        "truth_opened": False,
        "source_text_loaded": False,
        "target_labels_loaded": False,
        "opaque_record_ids_sha256": opaque_record_ids_digest(records),
    }


class _TinyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Exactly one 4,096-element block keeps the synthetic recipe faithful.
        self.q_proj = torch.nn.Linear(64, 64, bias=False)


class _TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _TinyAttention()


class _TinyModel(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_TinyLayer() for _ in range(5)])
        self.to(dtype=dtype)


def test_default_recipe_freezes_and_validates() -> None:
    plan = freeze_capture_plan(default_capture_plan(), frozen_utc="2026-09-08T00:00:00Z")
    checked = validate_capture_plan(plan, require_frozen=True)
    assert checked["status"] == "FROZEN_CAPTURE_RECIPE_BEFORE_OBSERVATIONS"
    assert tuple(checked["variant_ids"]) == EXPECTED_VARIANTS
    historical = _variant(plan, "historical_public_lora_2601")
    assert historical["role"] == "historical_trained_benchmark_adaptation"
    assert historical["independent_adaptation_claim"] is False
    assert plan["truth_boundary"]["truth_opened"] is False

    changed = copy.deepcopy(plan)
    changed["geometry"]["retained_sequence_tokens"] = 127
    with pytest.raises(CaptureError, match="geometry changed"):
        validate_capture_plan(changed, require_frozen=True)


def test_fixed_sign_update_resolves_nested_parameter_and_reports_effective_distance() -> None:
    plan = freeze_capture_plan(default_capture_plan(), frozen_utc="2026-09-08T00:00:00Z")
    model = _TinyModel()
    before = model.model.layers[1].self_attn.q_proj.weight.detach().clone()
    receipt = apply_fixed_sign_block_update(model, _variant(plan, "layer1_block_rel1e-3"))
    after = model.model.layers[1].self_attn.q_proj.weight.detach()
    assert receipt["status"] == "UPDATED"
    assert receipt["parameter_path"] == "model.layers.1.self_attn.q_proj.weight"
    assert receipt["effective_update_nonzero"] is True
    assert receipt["requested_relative_parameter_l2"] == pytest.approx(1e-3)
    assert receipt["effective_relative_parameter_l2"] == pytest.approx(1e-3, rel=2e-4)
    assert not torch.equal(before, after)

    bad = copy.deepcopy(plan)
    _variant(bad, "layer1_block_rel1e-3")["update_location"]["parameter_path"] = "model.layers.2.self_attn.q_proj.weight"
    with pytest.raises(CaptureError, match="parameter path"):
        validate_capture_plan(bad, require_frozen=True)


def test_opaque_panel_is_paired_and_rejects_source_fields() -> None:
    panel = _panel()
    checked = validate_opaque_panel_descriptor(panel)
    assert checked["record_count"] == 64
    assert checked["opaque_record_ids_sha256"] == panel["opaque_record_ids_sha256"]

    bad = copy.deepcopy(panel)
    bad["records"][0]["source_text"] = "must stay evaluator-side"
    with pytest.raises(CaptureError, match="source/token/truth"):
        validate_opaque_panel_descriptor(bad)

    bad = copy.deepcopy(panel)
    bad["records"][1]["opaque_id"] = bad["records"][0]["opaque_id"]
    with pytest.raises(CaptureError, match="duplicate"):
        validate_opaque_panel_descriptor(bad)


def test_observation_artifact_contains_only_permitted_tensors(tmp_path: Path) -> None:
    ids = [f"finance-{index:02d}" for index in range(32)]
    activations = torch.zeros((32, RETAINED_SEQUENCE_TOKENS, HIDDEN_SIZE), dtype=torch.bfloat16)
    mask = torch.ones((32, RETAINED_SEQUENCE_TOKENS), dtype=torch.uint8)
    positions = torch.arange(RETAINED_SEQUENCE_TOKENS, dtype=torch.int64).unsqueeze(0).expand(32, -1).contiguous()
    panel_sha = "b" * 64
    path = tmp_path / "clean_public" / "finance.safetensors"
    written = save_observation_artifact(
        path,
        activations=activations,
        attention_mask=mask,
        position_ids=positions,
        variant_id="clean_public",
        domain="finance",
        opaque_record_ids=ids,
        panel_sha256=panel_sha,
        capture_code_binding={"runner": {"sha256": "c" * 64}},
    )
    checked = validate_observation_artifact(
        path,
        expected_variant_id="clean_public",
        expected_domain="finance",
        expected_panel_sha256=panel_sha,
        expected_opaque_record_ids_sha256=written["opaque_record_ids_sha256"],
    )
    assert checked["tensor_keys"] == ["activations", "attention_mask", "position_ids"]
    assert checked["metadata"]["token_ids_written"] == "false"
    assert checked["metadata"]["truth_opened"] == "false"


def test_after_cut_null_requires_bitwise_prefix_equality() -> None:
    clean = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
    assert validate_after_cut_null(clean, clean.clone())["activation_exact_equal"] is True
    with pytest.raises(CaptureError, match="cut activation"):
        validate_after_cut_null(clean, clean + torch.tensor(1, dtype=torch.bfloat16))

