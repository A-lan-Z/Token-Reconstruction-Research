from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts import trr0009_eval_capture as capture
from scripts import trr0009_eval_contract as contract


def test_synthetic_selection_to_capture_adapter_is_truth_free(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    selection_path = repo / "experiments" / contract.TASK_ID / "selection.json"
    selection_path.parent.mkdir(parents=True)
    rows = {}
    for style in contract.DOMAIN_ORDER:
        rows[style] = [
            {"record_id": f"{style}-{index}", "public_record_sha256": f"{index + 1:064x}", "dataset_key": style, "valid_tokens": 128, "final_sequence_sha256": f"{index + 100:064x}"}
            for index in range(8)
        ]
    selection = {
        "schema": capture.SELECTION_SCHEMA,
        "task_id": contract.TASK_ID,
        "status": capture.SELECTION_STATUS,
        "records_by_domain": {"pile": 8, "finance": 8},
        "target_conditions": list(contract.TARGET_ORDER),
        "paired_conditions": True,
        "selection_rule": {"records": rows},
        "truth_opened": False,
        "source_text_or_target_labels": False,
        "source_text_written": False,
        "token_ids_written": False,
    }
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    activation = torch.zeros((8, 128, 2048), dtype=torch.bfloat16)
    mask = torch.ones((8, 128), dtype=torch.uint8)
    position_ids = torch.arange(128, dtype=torch.long).repeat(8, 1).contiguous()
    tensors = {cell: {"activations": activation, "attention_mask": mask, "position_ids": position_ids} for cell in contract.CELL_ORDER}
    result = capture.capture_from_observations(selection_path=selection_path, observation_tensors=tensors, output_root=repo / "experiments" / contract.TASK_ID / "evaluation" / "capture", repository_root=repo)
    assert result["status"] == capture.CAPTURE_STATUS
    manifest = json.loads(Path(result["observation_manifest"]["path"]).read_text())
    assert manifest["truth_opened"] is False
    assert manifest["capture_batch_records"] == 8
    assert all(cell["observation"]["shape"] == [8, 128, 2048] for cell in manifest["cells"])
    receipt = json.loads(Path(result["capture"]["path"]).read_text())
    assert receipt["execution"]["producer_semantics"] == "public full forward B8x192; retain first 128 positions"
    assert receipt["execution"]["truth_opened"] is False


def test_capture_cli_entrypoint_requires_explicit_execution(tmp_path: Path, capsys) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text("{}", encoding="utf-8")
    code = capture.main(["capture", "--selection", str(selection), "--model-snapshot", str(tmp_path / "model")])
    assert code == 2
    assert "requires explicit --execute" in capsys.readouterr().err


def test_failure_diagnostics_preserves_exception_chain_and_execution_context(tmp_path: Path) -> None:
    try:
        try:
            raise RuntimeError("underlying CUDA loader detail")
        except RuntimeError as cause:
            raise capture.CaptureError("public-prefix load failed") from cause
    except capture.CaptureError as exc:
        diagnostics = capture._failure_diagnostics(exc, root=tmp_path, stage="public_base")

    assert diagnostics["stage"] == "public_base"
    assert diagnostics["command"]
    assert diagnostics["exception_chain"] == [
        {"type": "CaptureError", "message": "public-prefix load failed"},
        {"type": "RuntimeError", "message": "underlying CUDA loader detail"},
    ]
    assert "underlying CUDA loader detail" in diagnostics["traceback"]
    assert diagnostics["code_commit"] is None


def test_producer_dispatch_passes_actual_lora_update_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    received: dict[str, object] = {}

    def fake_capture_prefix(**kwargs: object) -> tuple[object, dict[str, object]]:
        received.update(kwargs)
        raise RuntimeError("sentinel trusted-loader failure")

    monkeypatch.setattr(capture.trusted, "_capture_prefix", fake_capture_prefix)
    update_path = tmp_path / "public_lora_2601.safetensors"
    config_path = tmp_path / "generation.json"
    with pytest.raises(capture.CaptureError, match="public_lora_2601 public-prefix load failed"):
        capture._capture_condition_with_producer(
            condition="public_lora_2601",
            records={},
            batches={},
            model_snapshot=tmp_path / "model",
            lora_config_path=config_path,
            lora_update_path=update_path,
            output_root=tmp_path / "output",
            counts={"pile": 1, "finance": 1},
            record_ids_sha256={"pile": "p", "finance": "f"},
            selection_sha256="s",
            repository_root=tmp_path,
            device=torch.device("cpu"),
        )

    assert received["lora_update"] == update_path
    assert received["lora_config_path"] == config_path
