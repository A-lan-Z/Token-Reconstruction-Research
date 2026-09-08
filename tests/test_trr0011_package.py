from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts import trr0011_package as package


def test_prediction_digest_and_fake_decoder_keep_bos_and_geometry() -> None:
    class FakeModel:
        def projected_hidden(self, activation, mask):
            return activation

        def logits_from_rows(self, projected, rows, positions, readout):
            del projected, rows, readout
            logits = torch.full((package.SCORED_POST_BOS_TOKENS, package.VOCABULARY_SIZE), -1.0)
            logits[:, 17] = 3.0
            assert positions.shape == (package.SCORED_POST_BOS_TOKENS,)
            return logits

    activation = torch.zeros((package.STORED_SEQUENCE_TOKENS, package.HIDDEN_SIZE), dtype=torch.bfloat16)
    mask = torch.ones(package.STORED_SEQUENCE_TOKENS, dtype=torch.bool)
    readout = torch.empty((package.VOCABULARY_SIZE, package.HIDDEN_SIZE), dtype=torch.float32)
    output = package._predict_row(FakeModel(), readout, activation, mask, device=torch.device("cpu"))
    assert output.shape == (package.STORED_SEQUENCE_TOKENS,)
    assert output.dtype == torch.long
    assert int(output[0]) == package.BOS_TOKEN_ID
    assert torch.equal(output[1:], torch.full((package.SCORED_POST_BOS_TOKENS,), 17, dtype=torch.long))
    assert package.tensor_digest(output) == package.tensor_digest(output.clone())


def test_observation_loader_rejects_wrong_positions(tmp_path: Path) -> None:
    path = tmp_path / "observation.safetensors"
    activations = torch.zeros((128, package.STORED_SEQUENCE_TOKENS, package.HIDDEN_SIZE), dtype=torch.bfloat16)
    mask = torch.ones((128, package.STORED_SEQUENCE_TOKENS), dtype=torch.bool)
    positions = torch.arange(package.STORED_SEQUENCE_TOKENS, dtype=torch.long).repeat(128, 1)
    positions[0, 3] = 99
    save_file({"activations": activations, "attention_mask": mask, "position_ids": positions}, str(path))
    with pytest.raises(package.PackageError, match="mask/positions"):
        package._load_observation_row({"path": str(path)}, 0)


def test_external_binding_requires_explicit_readonly(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"package fixture")
    binding = {"path": str(path), "bytes": path.stat().st_size, "sha256": package.sha256_file(path)}
    with pytest.raises(package.PackageError, match="read-only"):
        package._require_readonly(binding, description="fixture")
    binding["readonly"] = True
    checked, record = package._resolve_binding(binding, root=tmp_path, description="fixture")
    assert checked == path.resolve()
    assert record["sha256"] == binding["sha256"]
