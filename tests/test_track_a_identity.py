from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from token_reconstruction import footing

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trr0003_track_a_identity.py"
SPEC = importlib.util.spec_from_file_location("trr0003_track_a_identity_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
IDENTITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IDENTITY
SPEC.loader.exec_module(IDENTITY)


def test_identity_metrics_detect_exact_and_nonzero_discrepancy() -> None:
    cached = torch.zeros((2, 3), dtype=torch.bfloat16)
    actual = cached.clone()
    exact = IDENTITY._metrics(actual, cached)
    assert exact["max_abs"] == 0.0
    assert exact["relative_l2"] == 0.0
    assert exact["allclose_atol_1e-3_rtol_1e-3"] is True

    actual[0, 0] = 1
    discrepant = IDENTITY._metrics(actual, cached)
    assert discrepant["max_abs"] > 0.0
    assert discrepant["relative_l2"] > 0.0


def test_mocked_public_state_load_reaches_identity_evidence(tmp_path, monkeypatch) -> None:
    assets = IDENTITY.ValidationAssets(
        observation_path=tmp_path / "observations.safetensors",
        truth_path=tmp_path / "labels.safetensors",
        records_path=tmp_path / "records.json",
        evidence_path=tmp_path / "slice_evidence.json",
        observation=torch.zeros((24, 40, IDENTITY.HIDDEN_SIZE), dtype=torch.bfloat16),
        truth=torch.full((24, 40), IDENTITY.BOS_TOKEN_ID, dtype=torch.int64),
        record_ids=tuple(f"record-{index}" for index in range(24)),
        observation_metadata={"schema": IDENTITY.VALIDATION_SCHEMA},
        truth_metadata={"truth_role": "public auxiliary validation only"},
        records_sha256="a" * 64,
        evidence_sha256="b" * 64,
    )
    manifest = Path("experiments/TRR-0003/track_a/public_resource_manifest.json").resolve()
    resource = IDENTITY.track_a.ResourceBundle(
        manifest_path=manifest,
        manifest_sha256=IDENTITY.track_a.sha256_file(manifest),
        snapshot_path=Path("/tmp/pinned-public-snapshot"),
        model={"id": IDENTITY.MODEL_ID, "revision": IDENTITY.MODEL_REVISION},
        files=(),
        total_bytes=0,
    )
    prefix = torch.nn.Linear(2, 2).eval()
    state = IDENTITY.track_a.LoadedPublicState(
        prefix=prefix,
        embedding_weight=torch.zeros((2, 2), dtype=torch.float32),
        prefix_digest="c" * 64,
        embedding_digest="d" * 64,
        parameter_bytes=prefix.weight.numel() * prefix.weight.element_size(),
        preparation_seconds=0.01,
        preparation_peak={"process_max_rss_kib": 1, "cuda_peak_allocated_bytes": 0, "cuda_peak_reserved_bytes": 0},
    )
    monkeypatch.setattr(IDENTITY, "_load_validation_assets", lambda **_: assets)
    monkeypatch.setattr(IDENTITY.track_a, "_configure_deterministic_execution", lambda: None)
    monkeypatch.setattr(IDENTITY.track_a, "_load_resource_manifest", lambda *args, **kwargs: resource)
    monkeypatch.setattr(IDENTITY.track_a, "_resource_preflight", lambda **_: {"probe_passed": True})
    monkeypatch.setattr(IDENTITY.track_a, "_load_public_state", lambda **_: state)
    monkeypatch.setattr(IDENTITY, "_identity_binding", lambda **_: {"mocked": True})
    monkeypatch.setattr(
        IDENTITY,
        "_forward_control",
        lambda **_: (
            [{"record_index": 0, "relative_l2": 0.0}],
            {
                "records": 24,
                "relative_l2_mean": 0.0,
                "allclose_atol_1e-2_rtol_1e-2": True,
                "public_prefix_layer_evaluations": 96,
                "forward_seconds": 0.01,
            },
        ),
    )
    monkeypatch.setattr(
        IDENTITY.track_a,
        "_repo_record",
        lambda path, repository_root: {"path": str(path), "bytes": 0, "sha256": "e" * 64},
    )
    monkeypatch.setattr(IDENTITY.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(IDENTITY.torch.cuda, "max_memory_reserved", lambda: 0)

    args = IDENTITY.build_parser().parse_args(
        [
            "--model-path",
            "/tmp/pinned-public-snapshot",
            "--resource-manifest",
            str(manifest),
            "--output",
            str(tmp_path / "identity.json"),
        ]
    )
    result = IDENTITY.run(args)
    assert result["status"] == "COMPLETED"
    payload = json.loads((tmp_path / "identity.json").read_text())
    assert payload["status"] == "COMPLETED"
    assert payload["public_model"]["loaded_prefix_sha256"] == "c" * 64
    assert payload["execution"]["process_max_rss_kib"] > 0
    assert payload["evaluator_truth_opened"] is False
