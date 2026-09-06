"""Lightweight checks for the evaluator-only P04 preparation contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.trr_p04.native_anchor_runner import (
    _decoder_position_ids,
    build_preflight as build_anchor_preflight,
)
from scripts.trr_p04.prepare_evaluator_observations import (
    EvaluatorObservationError,
    build_preflight as build_observation_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "experiments/TRR-P04/setup/public_selection-r2.json"
TARGET_PLAN = ROOT / "experiments/TRR-P04/setup/evaluator_target_plan.json"


def test_evaluator_preflight_binds_frozen_panel_without_model(tmp_path: Path) -> None:
    output = tmp_path / "observation-preflight"
    receipt = build_observation_preflight(
        selection_path=SELECTION,
        target_plan_path=TARGET_PLAN,
        output_root=output,
        argv=["prepare_evaluator_observations.py", "--preflight-only"],
    )
    assert receipt["status"] == "PASS_NO_MODEL_NO_TARGET_NO_TRUTH"
    assert receipt["selection"]["record_count"] == 72
    assert receipt["selection"]["anchor_count"] == 12
    assert receipt["access"]["model_loaded"] is False
    assert receipt["access"]["target_update_loaded"] is False
    saved = json.loads((output / "evaluator_capture_preflight.json").read_text(encoding="utf-8"))
    assert saved["target_plan"]["seed"] == 20260910
    assert "token_ids" in saved["forbidden_serialized_fields"]


def test_native_anchor_preflight_binds_separate_384_position_denominator(tmp_path: Path) -> None:
    receipt = build_anchor_preflight(
        selection_path=SELECTION,
        target_plan_path=TARGET_PLAN,
        output_root=tmp_path / "anchor-preflight",
        argv=["native_anchor_runner.py", "--preflight-only"],
    )
    assert receipt["status"] == "PASS_NO_MODEL_NO_TARGET_NO_TRUTH"
    assert receipt["anchor"]["record_count"] == 12
    assert receipt["anchor"]["scored_positions_per_target"] == 384
    assert receipt["algorithm"]["expected_candidate_simulations"] == 98304
    assert receipt["anchor"]["denominator_separate"] is True


def test_evaluator_plan_rejects_modified_target_seed(tmp_path: Path) -> None:
    plan = json.loads(TARGET_PLAN.read_text(encoding="utf-8"))
    plan["update"]["initialization_seed"] = 2711
    changed = tmp_path / "changed-plan.json"
    changed.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(EvaluatorObservationError, match="configuration changed"):
        build_observation_preflight(
            selection_path=SELECTION,
            target_plan_path=changed,
            output_root=tmp_path / "rejected",
            argv=["prepare_evaluator_observations.py", "--preflight-only"],
        )


def test_native_anchor_uses_same_public_reference_without_private_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Both paired anchor conditions must differ only in observation activations."""
    import inspect
    import sys
    import types

    import torch

    from scripts.trr_p04 import native_anchor_runner as anchor

    model_calls: list[str] = []

    class FakeModel:
        config = types.SimpleNamespace(hidden_size=anchor.HIDDEN_SIZE, vocab_size=128256)

        def to(self, device: torch.device) -> "FakeModel":
            return self

        def eval(self) -> "FakeModel":
            return self

        def requires_grad_(self, value: bool) -> "FakeModel":
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> FakeModel:
            model_calls.append(path)
            return FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    class FakePrecut:
        embed_tokens = types.SimpleNamespace(weight=torch.ones(2, 2))

        def to(self, device: torch.device) -> "FakePrecut":
            return self

        def eval(self) -> "FakePrecut":
            return self

    fake_reference = types.SimpleNamespace(
        PublicP0Precut=lambda model, layers: FakePrecut(),
        normalize_public_embeddings=lambda weight: weight,
        load_frozen_lens=lambda path, device: torch.ones(1),
    )
    monkeypatch.setattr(anchor, "_load_module", lambda path, name: fake_reference)
    snapshot = tmp_path / "public-snapshot"
    snapshot.mkdir()
    descriptors = []
    for condition in anchor.CONDITIONS:
        _, _, _, descriptor = anchor._load_reference_resources(
            model_snapshot=snapshot,
            reference_path=tmp_path / "reference.py",
            lens_path=tmp_path / "lens.pt",
            condition=condition,
            device=torch.device("cpu"),
        )
        descriptors.append(descriptor)
        assert descriptor["public_reference_loaded"] is True
        assert descriptor["evaluator_target_update_loaded"] is False
        assert descriptor["target_update_weights_available_to_reconstructor"] is False

    assert model_calls == [str(snapshot.resolve()), str(snapshot.resolve())]
    assert descriptors[0]["public_reference_identity"] == descriptors[1]["public_reference_identity"]
    source = inspect.getsource(anchor._load_reference_resources)
    assert "target_update_path" not in source
    assert "install_target_lora" not in source
    assert "load_target_lora" not in source


def test_native_anchor_adapts_only_padded_position_suffix() -> None:
    import torch

    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    persisted = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 3, 0]], dtype=torch.long)
    adapted = _decoder_position_ids(mask)
    assert torch.equal(adapted[:, :3], persisted[:, :3])
    assert adapted.tolist() == [[0, 1, 2, 2, 2], [0, 1, 2, 3, 3]]
